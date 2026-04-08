"""Binder classification based on multi-condition enrichment patterns.

Scientific context (Iqbal et al. 2025, analysis_pipeline_architecture.md Part 3)
---------------------------------------------------------------------------------
A compound's enrichment pattern across selection conditions reveals its binding
mode:

  - **Orthosteric**: enriched vs target, not enriched vs blank, competition
    with an orthosteric inhibitor eliminates enrichment.
  - **Allosteric**: enriched vs target, not enriched vs blank, competition
    with an orthosteric inhibitor does NOT eliminate enrichment (different site).
  - **Cryptic**: enriched only in the presence of an inhibitor (conformational
    change exposes a new site).
  - **Nonspecific**: enriched vs blank controls — bead/matrix artifact.
  - **Not enriched**: no meaningful enrichment in any condition.

For bead-artifact analysis, compounds enriched across multiple bead types are
considered true binders (``"robust"``), while those enriched with only one bead
type are suspected bead-specific artifacts (``"bead_artifact_suspect"``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_BLANK_KEYWORDS: tuple[str, ...] = ("blank", "no_protein", "control")
_INHIBITOR_KEYWORDS: tuple[str, ...] = ("inhibit", "_inh")


def _get_code_cols(df: pl.DataFrame) -> list[str]:
    """Return all columns starting with 'code_' in their original order."""
    return [c for c in df.columns if c.startswith("code_")]


def _is_blank_key(key: str) -> bool:
    lo = key.lower()
    return any(kw in lo for kw in _BLANK_KEYWORDS)


def _is_inhibitor_key(key: str) -> bool:
    lo = key.lower()
    return any(kw in lo for kw in _INHIBITOR_KEYWORDS)


def _union_compounds(dfs: Iterable[pl.DataFrame], code_cols: list[str]) -> pl.DataFrame:
    """Union of all unique compound identities across DataFrames."""
    return pl.concat([df.select(code_cols) for df in dfs]).unique()


def _any_enriched(
    dfs: list[pl.DataFrame],
    code_cols: list[str],
    threshold_col: str,
    threshold_value: float,
    all_compounds: pl.DataFrame,
    alias: str,
) -> pl.DataFrame:
    """Return *all_compounds* with boolean column *alias*.

    *alias* is ``True`` when the compound is enriched (``threshold_col >=
    threshold_value``) in **any** of the supplied DataFrames.  Missing
    compounds default to ``False``.

    Args:
        dfs: Enrichment DataFrames for one condition group.
        code_cols: Compound-identity columns shared across all DataFrames.
        threshold_col: Column name used as the enrichment score.
        threshold_value: Minimum score to be considered enriched.
        all_compounds: Reference set of all compound identities.
        alias: Name for the resulting boolean column.

    Returns:
        *all_compounds* with the extra boolean column *alias*.
    """
    if not dfs:
        return all_compounds.with_columns(pl.lit(False).alias(alias))

    frames = [
        df.select(code_cols + [(pl.col(threshold_col) >= threshold_value).alias("_e")])
        for df in dfs
    ]

    unioned = pl.concat(frames)
    grouped = (
        unioned
        .group_by(code_cols)
        .agg(pl.col("_e").any().alias(alias))
    )

    return (
        all_compounds
        .join(grouped, on=code_cols, how="left")
        .with_columns(pl.col(alias).fill_null(False))
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_binders(
    enrichment_results: dict[str, pl.DataFrame],
    threshold_col: str = "zscore",
    threshold_value: float = 1.0,
) -> pl.DataFrame:
    """Classify compounds by binding mode based on multi-condition enrichment.

    Conditions are detected automatically by substring matching on the key:

    - Keys containing ``"blank"``, ``"no_protein"``, or ``"control"`` →
      blank/control condition.
    - Keys containing ``"inhibit"`` or ``"_inh"`` (and NOT matching blank
      keywords) → target-plus-inhibitor condition.
    - All remaining keys → target condition.

    Classification priority (first matching rule wins):

    1. ``"nonspecific"`` — enriched in any blank condition.
    2. ``"orthosteric"`` — enriched in target AND NOT blank AND NOT inhibitor.
    3. ``"allosteric"``  — enriched in target AND NOT blank AND inhibitor.
    4. ``"cryptic"``     — NOT enriched in target AND enriched in inhibitor.
    5. ``"not_enriched"``— everything else.

    Args:
        enrichment_results: Mapping from condition name to enrichment DataFrame.
            Each DataFrame must contain the code columns (``code_1``,
            ``code_2``, …) and *threshold_col*.  Example keys:
            ``"target"``, ``"target_inhibitor"``, ``"blank"``.
        threshold_col: Column used to determine enrichment status.
        threshold_value: Minimum value of *threshold_col* to be enriched.

    Returns:
        DataFrame with compound identity columns and a ``"binder_type"``
        column (string, one of the five classes above).

    Raises:
        ValueError: If *enrichment_results* is empty, no target condition is
            found, or *threshold_col* is absent from any DataFrame.
    """
    if not enrichment_results:
        raise ValueError("enrichment_results must not be empty")

    first_df = next(iter(enrichment_results.values()))
    code_cols = _get_code_cols(first_df)
    if not code_cols:
        raise ValueError("No code_* columns found in enrichment DataFrames")

    for key, df in enrichment_results.items():
        if threshold_col not in df.columns:
            raise ValueError(
                f"Column '{threshold_col}' not found in condition '{key}'. "
                f"Available columns: {df.columns}"
            )

    blank_keys = [k for k in enrichment_results if _is_blank_key(k)]
    inhibitor_keys = [
        k for k in enrichment_results
        if not _is_blank_key(k) and _is_inhibitor_key(k)
    ]
    target_keys = [
        k for k in enrichment_results
        if not _is_blank_key(k) and not _is_inhibitor_key(k)
    ]

    if not target_keys:
        raise ValueError(
            "No target condition found in enrichment_results. "
            "At least one key must not match blank or inhibitor keywords. "
            f"Keys provided: {list(enrichment_results)}"
        )

    logger.debug(
        "classify_binders — target: %s | blank: %s | inhibitor: %s",
        target_keys, blank_keys, inhibitor_keys,
    )

    all_compounds = _union_compounds(enrichment_results.values(), code_cols)

    target_df = _any_enriched(
        [enrichment_results[k] for k in target_keys],
        code_cols, threshold_col, threshold_value, all_compounds, "enriched_target",
    )
    blank_df = _any_enriched(
        [enrichment_results[k] for k in blank_keys],
        code_cols, threshold_col, threshold_value, all_compounds, "enriched_blank",
    )
    inhibitor_df = _any_enriched(
        [enrichment_results[k] for k in inhibitor_keys],
        code_cols, threshold_col, threshold_value, all_compounds, "enriched_inhibitor",
    )

    combined = (
        target_df
        .join(blank_df, on=code_cols, how="left")
        .join(inhibitor_df, on=code_cols, how="left")
    )

    classified = combined.with_columns(
        pl.when(pl.col("enriched_blank"))
        .then(pl.lit("nonspecific"))
        .when(
            pl.col("enriched_target")
            & ~pl.col("enriched_blank")
            & ~pl.col("enriched_inhibitor")
        )
        .then(pl.lit("orthosteric"))
        .when(
            pl.col("enriched_target")
            & ~pl.col("enriched_blank")
            & pl.col("enriched_inhibitor")
        )
        .then(pl.lit("allosteric"))
        .when(~pl.col("enriched_target") & pl.col("enriched_inhibitor"))
        .then(pl.lit("cryptic"))
        .otherwise(pl.lit("not_enriched"))
        .alias("binder_type")
    )

    return classified.select(code_cols + ["binder_type"])


def classify_bead_artifacts(
    enrichment_by_bead: dict[str, pl.DataFrame],
    threshold_col: str = "zscore",
    threshold_value: float = 1.0,
    agreement_threshold: float = 0.5,
) -> pl.DataFrame:
    """Classify compounds by enrichment consistency across bead types.

    A compound that shows enrichment across many bead types is likely a true
    target binder (``"robust"``).  A compound enriched with only one bead type
    is a suspected bead-specific artifact (``"bead_artifact_suspect"``).

    The *agreement_threshold* sets the minimum fraction of bead types that must
    show enrichment for a compound to be classified as ``"robust"``.  With the
    default value of 0.5, a compound must be enriched in at least half of the
    tested bead types.  Compounds enriched in none are ``"not_enriched"``.

    Args:
        enrichment_by_bead: Mapping from bead-type name to enrichment
            DataFrame.  Each DataFrame must contain the code columns and
            *threshold_col*.
        threshold_col: Column used to determine enrichment status.
        threshold_value: Minimum value of *threshold_col* to be enriched.
        agreement_threshold: Minimum fraction of bead types that must show
            enrichment for ``"robust"`` classification (0–1, inclusive).

    Returns:
        DataFrame with compound identity columns, ``"n_enriched"`` (int count
        of bead types showing enrichment), ``"fraction_enriched"`` (float
        0–1), and ``"bead_classification"`` (string).

    Raises:
        ValueError: If *enrichment_by_bead* is empty or *threshold_col* is
            absent from any DataFrame.
    """
    if not enrichment_by_bead:
        raise ValueError("enrichment_by_bead must not be empty")

    first_df = next(iter(enrichment_by_bead.values()))
    code_cols = _get_code_cols(first_df)
    if not code_cols:
        raise ValueError("No code_* columns found in enrichment DataFrames")

    for bead_type, df in enrichment_by_bead.items():
        if threshold_col not in df.columns:
            raise ValueError(
                f"Column '{threshold_col}' not found for bead type '{bead_type}'"
            )

    n_beads = len(enrichment_by_bead)
    all_compounds = _union_compounds(enrichment_by_bead.values(), code_cols)

    # Build one boolean column per bead type, then sum for n_enriched
    enriched_col_names: list[str] = []
    combined = all_compounds

    for bead_type, df in enrichment_by_bead.items():
        col = f"_enriched_{bead_type}"
        enriched_col_names.append(col)
        mask = _any_enriched(
            [df], code_cols, threshold_col, threshold_value, all_compounds, col
        )
        combined = combined.join(mask, on=code_cols, how="left").with_columns(
            pl.col(col).fill_null(False)
        )

    combined = combined.with_columns(
        pl.sum_horizontal(
            [pl.col(c).cast(pl.Int32) for c in enriched_col_names]
        ).alias("n_enriched")
    ).drop(enriched_col_names)

    combined = combined.with_columns(
        (pl.col("n_enriched") / n_beads).alias("fraction_enriched")
    )

    combined = combined.with_columns(
        pl.when(pl.col("fraction_enriched") >= agreement_threshold)
        .then(pl.lit("robust"))
        .when(pl.col("n_enriched") > 0)
        .then(pl.lit("bead_artifact_suspect"))
        .otherwise(pl.lit("not_enriched"))
        .alias("bead_classification")
    )

    return combined.select(
        code_cols + ["n_enriched", "fraction_enriched", "bead_classification"]
    )
