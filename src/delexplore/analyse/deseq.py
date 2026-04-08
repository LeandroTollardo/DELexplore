"""PyDESeq2 negative binomial GLM for DEL enrichment analysis.

Wraps PyDESeq2 for DEL count data.  The key challenge is reshaping:
DELT-Hit produces long-format counts (one row per compound per selection),
while PyDESeq2 needs wide format (selections as rows, compounds as columns).

Conceptual mapping (see docs/references/pydeseq2_integration.md):

  DEL concept                 PyDESeq2 concept
  ─────────────────────────── ───────────────────
  Compound / synthon feature  Gene
  Selection instance          Sample
  Count per compound          Read count per gene
  "protein" vs "no_protein"   Condition (treated vs control)
  Selection replicate         Biological replicate
  Bead type                   Batch variable

Performance note
----------------
Run PyDESeq2 at monosynthon and disynthon levels by default.
At trisynthon level for large libraries use z-score/Poisson methods instead.
PyDESeq2 at trisynthon is offered as opt-in for small libraries (< 500K features).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

# Feature count above which we emit a performance warning.
_LARGE_FEATURE_THRESHOLD = 500_000


def prepare_deseq_input(
    counts_df: pl.DataFrame,
    code_cols: list[str],
    selection_col: str = "selection",
    count_col: str = "count",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reshape long-format DEL counts into the wide format required by PyDESeq2.

    Steps:

    1. Build a ``compound_id`` column by joining code column values with ``_``
       (e.g. ``"2_3"`` for a 2-cycle library).
    2. Pivot so that rows = selection instances, columns = compound IDs,
       values = integer counts.
    3. Fill any missing (compound, selection) pairs with 0.
    4. Convert to pandas DataFrames (PyDESeq2 operates on pandas).

    The metadata DataFrame returned contains only the selection names as its
    index; callers should add condition/bead/protocol columns before calling
    :func:`run_deseq2`.

    Args:
        counts_df: Long-format Polars DataFrame with at least the columns
            named by *selection_col*, *code_cols*, and *count_col*.
        code_cols: Code column names that define compound identity
            (e.g. ``["code_1", "code_2"]``).
        selection_col: Name of the column that identifies each selection.
        count_col: Name of the count column.

    Returns:
        ``(counts_wide_pd, metadata_pd)`` where:

        - *counts_wide_pd* — pandas DataFrame, shape
          ``(n_selections, n_compounds)``, integer dtype, index = selection
          names, columns = compound IDs.
        - *metadata_pd* — pandas DataFrame with the same index; empty columns
          (caller adds condition etc.).

    Raises:
        ValueError: If any required column is missing from *counts_df*.
    """
    required = {selection_col, count_col, *code_cols}
    missing = required - set(counts_df.columns)
    if missing:
        raise ValueError(f"counts_df is missing required columns: {sorted(missing)}")

    # Build compound_id: join code column values with "_"
    id_expr = pl.concat_str([pl.col(c).cast(pl.Utf8) for c in code_cols], separator="_")
    df = counts_df.with_columns(id_expr.alias("compound_id"))

    # Pivot to wide format: index=selection, columns=compound_id, values=count
    wide = (
        df.pivot(
            on="compound_id",
            index=selection_col,
            values=count_col,
            aggregate_function="sum",
        )
        .fill_null(0)
    )

    n_features = wide.width - 1  # subtract the selection column
    if n_features > _LARGE_FEATURE_THRESHOLD:
        logger.warning(
            "DESeq2 input has %d features (> %d). "
            "Consider running at a lower synthon level for performance.",
            n_features,
            _LARGE_FEATURE_THRESHOLD,
        )

    counts_pd = wide.to_pandas().set_index(selection_col).astype(int)
    metadata_pd = pd.DataFrame(index=counts_pd.index)
    metadata_pd.index.name = selection_col

    return counts_pd, metadata_pd


def run_deseq2(
    counts_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    code_cols: list[str],
    contrast: tuple[str, str, str] = ("condition", "protein", "no_protein"),
    design: str = "~condition",
) -> pl.DataFrame:
    """Run PyDESeq2 on DEL count data and return per-compound results.

    Args:
        counts_df: Long-format Polars DataFrame (columns: ``selection``,
            code columns, ``count``).
        metadata_df: Polars DataFrame with columns ``selection_name`` and
            at least the condition column referenced in *design* (e.g.
            ``"condition"``).  Must have one row per selection.
        code_cols: Code column names used to build compound IDs
            (e.g. ``["code_1", "code_2"]``).
        contrast: Three-tuple ``(factor, test_level, reference_level)``
            passed to :class:`pydeseq2.ds.DeseqStats`.
        design: R-style design formula string (e.g. ``"~condition"`` or
            ``"~bead_type + condition"``).

    Returns:
        Polars DataFrame with columns:
        ``compound_id``, ``log2FoldChange``, ``pvalue``, ``padj``,
        ``baseMean``, ``lfcSE``, ``stat``.
        Rows with PyDESeq2 convergence failures have ``NaN`` values;
        they are included (not silently dropped) so callers can filter.

    Raises:
        ValueError: If fewer than 2 samples exist for any condition level,
            or if the condition column is missing from *metadata_df*.
        ImportError: If ``pydeseq2`` is not installed.
    """
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError as exc:
        raise ImportError(
            "pydeseq2 is required for this function. "
            "Install it with: pip install 'delexplore[ml]'"
        ) from exc

    # ── 1. Build wide counts + align metadata ─────────────────────────────

    counts_pd, meta_stub = prepare_deseq_input(counts_df, code_cols)

    # Prepare metadata pandas DataFrame aligned to selections in counts_pd
    condition_col = contrast[0]
    meta_cols = {"selection_name", condition_col}
    missing = meta_cols - set(metadata_df.columns)
    if missing:
        raise ValueError(
            f"metadata_df is missing required columns: {sorted(missing)}"
        )

    meta_pd = (
        metadata_df
        .select(["selection_name", condition_col])
        .to_pandas()
        .set_index("selection_name")
    )

    # Restrict to selections present in counts and align order
    common = [s for s in counts_pd.index if s in meta_pd.index]
    counts_pd = counts_pd.loc[common]
    meta_pd = meta_pd.loc[common]

    # ── 2. Validate: ≥ 2 samples per condition ───────────────────────────

    condition_counts: dict[Any, int] = meta_pd[condition_col].value_counts().to_dict()
    under_replicated = {k: v for k, v in condition_counts.items() if v < 2}
    if under_replicated:
        raise ValueError(
            f"PyDESeq2 requires at least 2 samples per condition. "
            f"The following conditions have fewer: {under_replicated}. "
            "Use z-score or Poisson methods for single-replicate conditions."
        )

    # ── 3. Run PyDESeq2 ───────────────────────────────────────────────────

    logger.info(
        "Running DESeq2: %d selections × %d features, design=%r, contrast=%s",
        counts_pd.shape[0],
        counts_pd.shape[1],
        design,
        contrast,
    )

    dds = DeseqDataSet(
        counts=counts_pd,
        metadata=meta_pd,
        design=design,
        quiet=True,
    )

    try:
        dds.deseq2()
    except Exception as exc:
        # Re-raise with context rather than swallowing
        raise RuntimeError(
            f"PyDESeq2 fitting failed: {exc}. "
            "Check that count data has sufficient variation and replicates."
        ) from exc

    stats = DeseqStats(dds, contrast=list(contrast), quiet=True)
    stats.summary()

    results_pd: pd.DataFrame = stats.results_df.reset_index(names="compound_id")

    # Log convergence failures
    n_nan = results_pd["padj"].isna().sum()
    if n_nan > 0:
        frac = n_nan / len(results_pd)
        log_fn = logger.warning if frac > 0.1 else logger.debug
        log_fn(
            "%d / %d features (%.1f%%) have NaN padj (convergence failures or "
            "independent filtering). These are retained in results.",
            n_nan,
            len(results_pd),
            100 * frac,
        )

    # ── 4. Convert to Polars ──────────────────────────────────────────────

    keep_cols = ["compound_id", "log2FoldChange", "pvalue", "padj", "baseMean", "lfcSE", "stat"]
    available = [c for c in keep_cols if c in results_pd.columns]
    return pl.from_pandas(results_pd[available])
