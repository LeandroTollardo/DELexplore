"""Consensus hit ranking from multi-method, multi-level enrichment scores.

Scientific motivation (analysis_pipeline_architecture.md Part 3)
----------------------------------------------------------------
Three enrichment methods × multiple synthon levels = many independent scores
per compound.  This module combines them into a single actionable ranked list
using three complementary signals:

1. **Method agreement** (rank-product statistic, Breitling 2004)
   Geometric mean of per-method ranks at the full-compound level.
   Low value → consistently enriched across methods (not a fluke of one metric).

2. **Multi-level support score**
   +1 per enriched monosynthon constituent, +2 per enriched disynthon
   constituent.  A compound whose building blocks are independently enriched
   is far more likely to be a true binder than one whose trisynthon score is
   high by chance.

3. **Property penalty** (optional)
   Compounds failing drug-likeness filters get a multiplier > 1 on their
   final score, pushing them down the ranking.

Composite score::

    final_score = agreement_score × (1 / (1 + support_score)) × property_penalty

Lower final_score = better rank (rank 1 = best hit).

Why fold_enrichment, not z-score, for support scoring
------------------------------------------------------
``zscore >= 1.0`` is **wrong** as a binary enrichment threshold for sub-level
support scoring.  The z-score at monosynthon level is inherently small because
each building block's aggregated count is diluted by contributions from many
non-binding compound partners.  In a real 2-cycle library with 612 BBs and
5 M reads, the most enriched BB (found in 7 of the top-10 compounds) only
reaches z_n ≈ 0.31.  Requiring z_n ≥ 1.0 would demand ~30-fold enrichment —
unattainable at monosynthon level.

Use ``fold_enrichment >= 2.0`` (or top-10% percentile as fallback) instead.
The z-score remains the correct metric for *ranking within a single level*;
it is not designed for binary classification across levels.

See ``docs/references/corrected_formulas.md`` §SUPPORT SCORE THRESHOLD WARNING.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

from delexplore.analyse.aggregate import get_level_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants for adaptive threshold resolution
# ---------------------------------------------------------------------------

_AUTO_PRIMARY_COL = "fold_enrichment"
_AUTO_PRIMARY_THRESHOLD = 2.0
_PERCENTILE_FALLBACK = 90  # top 10% of features

# Candidate columns tried in order when falling back to percentile
_FALLBACK_CANDIDATES = ("fold_enrichment", "poisson_ml_enrichment", "zscore")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_enrichment_threshold(
    level_df: pl.DataFrame,
    threshold_col: str,
    threshold_value: float,
) -> tuple[str, float] | None:
    """Return ``(col, cutoff)`` for binary enrichment classification.

    Logic:
    - ``threshold_col == "auto"``:
        Use *fold_enrichment >= 2.0* if that column is present; otherwise
        compute the 90th-percentile of the best available score column.
    - Any other *threshold_col* that is present in *level_df*:
        Use as specified.
    - *threshold_col* absent from *level_df*:
        Fall back to 90th-percentile of the best available candidate column
        (same priority order as ``"auto"``).

    Returns ``None`` if no usable column is found.
    """
    auto_mode = threshold_col == "auto"

    if not auto_mode and threshold_col in level_df.columns:
        return (threshold_col, threshold_value)

    # Either "auto" was requested or the specified column is absent → resolve
    if not auto_mode:
        logger.debug(
            "threshold_col '%s' absent in level, attempting adaptive fallback",
            threshold_col,
        )

    # Prefer fold_enrichment with a fixed threshold
    if _AUTO_PRIMARY_COL in level_df.columns:
        return (_AUTO_PRIMARY_COL, _AUTO_PRIMARY_THRESHOLD)

    # Fall back to top-10% percentile of whatever score column is available
    for candidate in _FALLBACK_CANDIDATES:
        if candidate in level_df.columns:
            scores = level_df[candidate].drop_nulls().to_numpy()
            if len(scores) == 0:
                continue
            cutoff = float(np.percentile(scores, _PERCENTILE_FALLBACK))
            logger.debug(
                "Support score: using top-%d%% percentile of '%s' (cutoff=%.4f)",
                100 - _PERCENTILE_FALLBACK, candidate, cutoff,
            )
            return (candidate, cutoff)

    return None  # nothing usable found


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def compute_method_agreement(
    results_df: pl.DataFrame,
    method_cols: list[str],
) -> np.ndarray:
    """Compute the rank-product agreement score across enrichment methods.

    For each method column the compounds are ranked in descending order of
    enrichment score (rank 1 = highest score = most enriched).  The geometric
    mean of those ranks is returned — lower means consistently high-ranking
    across all methods.

    Args:
        results_df: DataFrame with at least the columns listed in
            *method_cols*.  One row per compound at a given synthon level.
        method_cols: Column names to rank by (higher value = more enriched).
            Must all be present in *results_df*.

    Returns:
        1-D numpy array of length ``len(results_df)``.  Lower values indicate
        stronger multi-method agreement (better hits).

    Raises:
        ValueError: If *method_cols* is empty or any column is absent.
    """
    if not method_cols:
        raise ValueError("method_cols must not be empty")

    missing = [c for c in method_cols if c not in results_df.columns]
    if missing:
        raise ValueError(
            f"Columns not found in results_df: {missing}. "
            f"Available: {results_df.columns}"
        )

    n = len(results_df)
    if n == 0:
        return np.array([], dtype=float)

    log_rank_sum = np.zeros(n, dtype=float)
    for col in method_cols:
        scores = results_df[col].to_numpy().astype(float)
        # Rank descending: rank 1 = highest score.  Ties → average rank.
        ranks = rankdata(-scores, method="average")
        log_rank_sum += np.log(ranks)

    # Geometric mean = exp(mean of log-ranks)
    return np.exp(log_rank_sum / len(method_cols))


def compute_support_score(
    multilevel_results: dict[str, pl.DataFrame],
    compound_df: pl.DataFrame,
    code_cols: list[str],
    threshold_col: str = "fold_enrichment",
    threshold_value: float = 2.0,
) -> np.ndarray:
    """Score each compound by how many of its constituent synthons are enriched.

    For a compound (A_i, B_j) from a 2-cycle library:

    - +1 if monosynthon A_i is enriched at ``mono_code_1``
    - +1 if monosynthon B_j is enriched at ``mono_code_2``

    For a 3-cycle compound (A_i, B_j, C_k):

    - +1 per enriched monosynthon (max 3)
    - +2 per enriched disynthon, e.g. (A_i, B_j) at ``di_code_1_code_2`` (max 6)

    The full-compound level is excluded (it is what we are scoring, not a
    constituent).

    **Threshold column guidance**

    Use ``threshold_col="fold_enrichment"`` (the default) with
    ``threshold_value=2.0`` to classify sub-level features as enriched.  Do
    **not** use ``zscore >= 1.0`` for this purpose — see module docstring and
    ``docs/references/corrected_formulas.md`` §SUPPORT SCORE THRESHOLD WARNING.

    If *threshold_col* is absent from a level DataFrame, or if
    ``threshold_col="auto"`` is passed, the function resolves the threshold
    adaptively:

    1. Use ``fold_enrichment >= 2.0`` if that column is present.
    2. Otherwise use the 90th-percentile of the best available score column
       (priority: ``fold_enrichment`` → ``poisson_ml_enrichment`` → ``zscore``).

    Args:
        multilevel_results: Output of
            :func:`~delexplore.analyse.multilevel.run_multilevel_enrichment`.
            Maps level name → enrichment DataFrame.
        compound_df: One row per full compound, containing all *code_cols*.
            Typically the full-compound level DataFrame from
            *multilevel_results*.
        code_cols: All code column names (e.g. ``["code_1", "code_2"]``).
        threshold_col: Column name used to determine enrichment status.
            Use ``"auto"`` for adaptive resolution.
        threshold_value: Minimum value of *threshold_col* for enrichment.
            Ignored when adaptive fallback kicks in.

    Returns:
        1-D numpy array of support scores, one per row of *compound_df*.
        Higher is better (more independent evidence of binding).

    Raises:
        ValueError: If *code_cols* is empty.
    """
    if not code_cols:
        raise ValueError("code_cols must not be empty")

    n_cycles = len(code_cols)
    n = len(compound_df)
    support = np.zeros(n, dtype=float)

    # Only sub-levels (k < n_cycles) contribute to support
    for k in range(1, n_cycles):
        weight = 1.0 if k == 1 else 2.0  # mono +1, di/higher +2
        for combo in combinations(code_cols, k):
            level_name = get_level_name(combo)
            if level_name not in multilevel_results:
                logger.debug("Support: level %s not found, skipping", level_name)
                continue

            level_df = multilevel_results[level_name]
            resolved = _resolve_enrichment_threshold(
                level_df, threshold_col, threshold_value
            )
            if resolved is None:
                logger.debug(
                    "Support: no usable score column in %s, skipping", level_name
                )
                continue

            col, cutoff = resolved
            level_cols = list(combo)
            enriched = (
                level_df
                .select(level_cols + [col])
                .with_columns(
                    (pl.col(col) >= cutoff).alias("_enriched")
                )
                .select(level_cols + ["_enriched"])
            )

            joined = (
                compound_df
                .select(code_cols)
                .join(enriched, on=level_cols, how="left")
                .with_columns(pl.col("_enriched").fill_null(False))
            )
            support += joined["_enriched"].to_numpy().astype(float) * weight

    return support


def compute_composite_rank(
    multilevel_results: dict[str, pl.DataFrame],
    code_cols: list[str],
    method_cols: tuple[str, ...] = ("zscore", "poisson_ml_enrichment"),
    threshold_col: str = "fold_enrichment",
    threshold_value: float = 2.0,
    properties_df: pl.DataFrame | None = None,
    property_penalty_col: str = "property_penalty",
) -> pl.DataFrame:
    """Produce a composite-ranked hit list from multi-level enrichment results.

    The function:

    1. Identifies the full-compound level (all *code_cols* together).
    2. Computes the method-agreement score at that level (geometric mean of
       per-method ranks; only columns present in the DataFrame are used).
    3. Computes the multi-level support score from sub-levels using
       ``fold_enrichment >= 2.0`` by default (see :func:`compute_support_score`
       for threshold guidance).
    4. Optionally joins *properties_df* and applies a property penalty.
    5. Computes ``final_score = agreement × 1/(1+support) × penalty``.
    6. Returns the DataFrame sorted ascending by *final_score*, with a
       ``rank`` column added.

    Args:
        multilevel_results: Output of
            :func:`~delexplore.analyse.multilevel.run_multilevel_enrichment`.
        code_cols: All code column names (e.g. ``["code_1", "code_2"]``).
        method_cols: Enrichment columns to include in the agreement score.
            Only columns that exist in the full-compound DataFrame are used
            (others are silently skipped).
        threshold_col: Column used for the support-score enrichment check.
            Defaults to ``"fold_enrichment"``.  Pass ``"auto"`` for adaptive
            resolution.  Do **not** use ``"zscore"`` — see module docstring.
        threshold_value: Minimum value of *threshold_col* for enrichment.
            Defaults to ``2.0`` (2-fold enrichment).
        properties_df: Optional DataFrame with *code_cols* plus property
            columns.  If it contains a ``"property_penalty"`` column (or the
            column named by *property_penalty_col*), it is applied directly.
            If it contains ``"lipinski_pass"`` (boolean) but no explicit
            penalty column, a penalty of 2.0 is assigned to failing compounds.
        property_penalty_col: Name of the explicit penalty column to look for
            in *properties_df*.

    Returns:
        DataFrame with all columns from the full-compound level plus
        ``agreement_score``, ``support_score``, ``property_penalty``,
        ``composite_score``, and ``rank`` (1-based integer, 1 = best hit).

    Raises:
        ValueError: If no full-compound level is found in *multilevel_results*,
            or *code_cols* is empty.
    """
    if not code_cols:
        raise ValueError("code_cols must not be empty")

    full_level_name = get_level_name(tuple(code_cols))
    if full_level_name not in multilevel_results:
        raise ValueError(
            f"Full-compound level '{full_level_name}' not found in multilevel_results. "
            f"Available levels: {list(multilevel_results)}"
        )

    compound_df = multilevel_results[full_level_name]

    # Only use method_cols that are present in the full-compound DataFrame
    available_methods = [c for c in method_cols if c in compound_df.columns]
    if not available_methods:
        raise ValueError(
            f"None of the requested method_cols {list(method_cols)} are present "
            f"in the full-compound level DataFrame. Available: {compound_df.columns}"
        )
    if len(available_methods) < len(method_cols):
        skipped = [c for c in method_cols if c not in compound_df.columns]
        logger.warning("Skipping missing method columns: %s", skipped)

    agreement = compute_method_agreement(compound_df, available_methods)
    support = compute_support_score(
        multilevel_results, compound_df, code_cols, threshold_col, threshold_value
    )

    result = compound_df.with_columns([
        pl.Series("agreement_score", agreement),
        pl.Series("support_score", support),
    ])

    # Property penalty
    if properties_df is not None:
        result = result.join(
            properties_df.select(
                code_cols
                + [c for c in properties_df.columns if c not in code_cols]
            ),
            on=code_cols,
            how="left",
        )
        if property_penalty_col in result.columns:
            result = result.with_columns(
                pl.col(property_penalty_col).fill_null(1.0)
            )
        elif "lipinski_pass" in result.columns:
            result = result.with_columns(
                pl.when(pl.col("lipinski_pass").fill_null(True))
                .then(pl.lit(1.0))
                .otherwise(pl.lit(2.0))
                .alias(property_penalty_col)
            )
        else:
            result = result.with_columns(pl.lit(1.0).alias(property_penalty_col))
    else:
        result = result.with_columns(pl.lit(1.0).alias(property_penalty_col))

    # Composite score: lower = better
    result = result.with_columns(
        (
            pl.col("agreement_score")
            * (1.0 / (1.0 + pl.col("support_score")))
            * pl.col(property_penalty_col)
        ).alias("composite_score")
    )

    result = result.sort("composite_score", descending=False).with_columns(
        pl.Series("rank", range(1, len(result) + 1))
    )

    logger.info(
        "composite_rank: %d compounds ranked, method_cols=%s, threshold=%s>=%.2f",
        len(result), available_methods, threshold_col, threshold_value,
    )
    return result


def export_hit_list(
    ranked_df: pl.DataFrame,
    top_n: int = 100,
    output_path: Path | str | None = None,
) -> pl.DataFrame:
    """Export the top-N ranked hits, optionally writing to a CSV file.

    Args:
        ranked_df: Output of :func:`compute_composite_rank` (must have a
            ``"rank"`` column).
        top_n: Number of top compounds to include.  If ``top_n`` is greater
            than the length of *ranked_df*, all rows are returned.
        output_path: If provided, the hit list is written as a CSV to this
            path.  Parent directories are created automatically.

    Returns:
        DataFrame of the top-N hits, sorted by rank ascending.

    Raises:
        ValueError: If *ranked_df* does not contain a ``"rank"`` column.
    """
    if "rank" not in ranked_df.columns:
        raise ValueError(
            "ranked_df must contain a 'rank' column. "
            "Run compute_composite_rank first."
        )

    hits = (
        ranked_df
        .sort("rank", descending=False)
        .head(min(top_n, len(ranked_df)))
    )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        hits.write_csv(out)
        logger.info("Hit list (%d compounds) written to %s", len(hits), out)

    return hits
