"""Multi-level enrichment orchestration.

Runs ALL enrichment methods across ALL synthon levels in a single call,
producing a result dict keyed by level name.

Scientific motivation (Faver 2019, analysis_pipeline_architecture.md Part 1)
-----------------------------------------------------------------------------
A compound from a 3-cycle library is (BB_A, BB_B, BB_C).  True binding can
originate from any sub-combination.  Analyzing every level simultaneously:

  1. Maximises statistical power — monosynthon counts are orders of magnitude
     higher than trisynthon counts at identical sequencing depth.
  2. Reveals SAR — disynthon enrichment that does not propagate to mono often
     indicates a cooperative pharmacophore.
  3. Identifies truncated synthesis products — monosynthon enrichment without
     disynthon propagation flags incomplete reactions.

z_n values are directly comparable across levels (the diversity parameter
in the denominator accounts for the different expected frequencies).

Usage
-----
>>> result = run_multilevel_enrichment(
...     counts_df, n_cycles=2, code_cols=["code_1", "code_2"],
...     post_selections=["target_1", "target_2", "target_3"],
...     control_selections=["blank_1", "blank_2", "blank_3"],
... )
>>> result["mono_code_1"]    # monosynthon enrichment for position 1
>>> result["di_code_1_code_2"]  # disynthon (= full compound for 2-cycle)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

from delexplore.analyse.aggregate import (
    aggregate_to_level,
    get_all_levels,
    get_diversity,
    get_level_name,
)
from delexplore.analyse.poisson import enrichment_with_ci, poisson_ml_enrichment
from delexplore.analyse.zscore import zscore_enrichment

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Supported method names
_VALID_METHODS = frozenset({"zscore", "poisson_ml"})


def merge_replicates(
    counts_df: pl.DataFrame,
    selections: list[str],
    code_cols: list[str],
) -> tuple[pl.DataFrame, int]:
    """Filter to given selections, sum replicate counts per compound.

    Merging replicates by summing counts (not averaging enrichments) is the
    statistically correct approach — the merged data follows a Poisson
    distribution with parameter equal to the sum of individual parameters.

    Args:
        counts_df: Combined counts DataFrame with ``selection``, code columns,
            and ``count``.
        selections: Selection names to include (e.g. all protein replicates).
        code_cols: Code column names that identify each compound at this level.

    Returns:
        ``(merged_df, total_reads)`` where:

        - *merged_df* — DataFrame with *code_cols* and ``count`` (summed
          across replicates).
        - *total_reads* — total summed counts across all replicates and
          all compounds (used as the ``total_reads`` denominator in
          enrichment formulas).

    Raises:
        ValueError: If none of *selections* are present in *counts_df*.
    """
    present = set(counts_df["selection"].unique().to_list())
    found = [s for s in selections if s in present]
    if not found:
        raise ValueError(
            f"None of the requested selections are in counts_df. "
            f"Requested: {selections}. Present: {sorted(present)}"
        )
    if len(found) < len(selections):
        missing = sorted(set(selections) - set(found))
        logger.warning("Some selections not found in counts_df and were skipped: %s", missing)

    filtered = counts_df.filter(pl.col("selection").is_in(found))
    merged = (
        filtered
        .group_by(code_cols)
        .agg(pl.col("count").sum())
    )
    total_reads = int(merged["count"].sum())
    return merged, total_reads


def _apply_zscore(
    merged_post: pl.DataFrame,
    merged_ctrl: pl.DataFrame,
    code_cols: list[str],
    total_post: int,
    total_ctrl: int,
    diversity: int,
    alpha: float,
) -> pl.DataFrame:
    """Inner helper: compute z-score on the merged post vs. total."""
    # z-score uses the post-selection counts and total_reads = post_total
    # (it's a single-condition enrichment metric, not a ratio)
    counts_arr = merged_post["count"].to_numpy().astype(float)
    z, z_lo, z_hi = zscore_enrichment(counts_arr, total_post, diversity, alpha)

    return merged_post.select(code_cols).with_columns([
        pl.Series("zscore", z),
        pl.Series("zscore_ci_lower", z_lo),
        pl.Series("zscore_ci_upper", z_hi),
    ])


def _apply_poisson_ml(
    merged_post: pl.DataFrame,
    merged_ctrl: pl.DataFrame,
    code_cols: list[str],
    total_post: int,
    total_ctrl: int,
    diversity: int,
    alpha: float,
) -> pl.DataFrame:
    """Inner helper: compute Poisson ML enrichment and CI."""
    # Join post and control on code columns, filling 0 for missing compounds
    joined = merged_post.join(
        merged_ctrl.rename({"count": "count_ctrl"}),
        on=code_cols,
        how="left",
    ).fill_null(0)

    post_arr = joined["count"].to_numpy().astype(float)
    ctrl_arr = joined["count_ctrl"].to_numpy().astype(float)

    ml = poisson_ml_enrichment(post_arr, ctrl_arr, total_post, total_ctrl)
    # Poisson CI on the post count, scaled to fold-enrichment
    fe, fe_lo, fe_hi = enrichment_with_ci(post_arr, total_post, diversity, alpha)

    return joined.select(code_cols).with_columns([
        pl.Series("poisson_ml_enrichment", ml),
        pl.Series("fold_enrichment", fe),
        pl.Series("poisson_ci_lower", fe_lo),
        pl.Series("poisson_ci_upper", fe_hi),
    ])


def run_multilevel_enrichment(
    counts_df: pl.DataFrame,
    n_cycles: int,
    code_cols: list[str],
    post_selections: list[str],
    control_selections: list[str],
    methods: tuple[str, ...] = ("zscore", "poisson_ml"),
    alpha: float = 0.05,
) -> dict[str, pl.DataFrame]:
    """Compute enrichment at every synthon level using all requested methods.

    For each level:

    1. Aggregate compound counts to that level.
    2. Merge replicates (sum) for the post-selection and control groups.
    3. Compute diversity at that level.
    4. Run each method, producing per-feature scores.
    5. Join all method results into one DataFrame keyed by level name.

    Output columns (subset present depends on *methods*):

    - *level code columns* — compound or synthon identity
    - ``count_post``, ``count_control`` — merged replicate counts
    - ``total_post``, ``total_control`` — total reads used as denominators
    - ``diversity`` — number of distinct features at this level
    - ``zscore``, ``zscore_ci_lower``, ``zscore_ci_upper`` (if "zscore")
    - ``poisson_ml_enrichment``, ``fold_enrichment``,
      ``poisson_ci_lower``, ``poisson_ci_upper`` (if "poisson_ml")

    Args:
        counts_df: Combined long-format counts (all selections together).
        n_cycles: Library cycle count.
        code_cols: All code column names (e.g. ``["code_1", "code_2"]``).
        post_selections: Selection names for the protein/target condition.
        control_selections: Selection names for the blank/control condition.
        methods: Subset of ``("zscore", "poisson_ml")`` to compute.
        alpha: Significance level for confidence intervals.

    Returns:
        Dict mapping level name → enrichment DataFrame.

    Raises:
        ValueError: If *methods* contains an unrecognised method name, or if
            any selection group is entirely absent from *counts_df*.
    """
    unknown = set(methods) - _VALID_METHODS
    if unknown:
        raise ValueError(
            f"Unknown enrichment method(s): {sorted(unknown)}. "
            f"Valid options: {sorted(_VALID_METHODS)}"
        )

    all_levels = get_all_levels(n_cycles)
    result: dict[str, pl.DataFrame] = {}

    for level_cols in all_levels:
        level_name = get_level_name(level_cols)
        level_list = list(level_cols)

        logger.info("Processing level: %s", level_name)

        # Aggregate full counts to this level (all selections)
        agg = aggregate_to_level(counts_df, level_cols)

        # Compute diversity from the aggregated data
        diversity = get_diversity(counts_df, level_cols)

        # Merge replicates for each group
        post_agg = agg.filter(pl.col("selection").is_in(post_selections))
        ctrl_agg = agg.filter(pl.col("selection").is_in(control_selections))

        merged_post, total_post = merge_replicates(post_agg, post_selections, level_list)
        merged_ctrl, total_ctrl = merge_replicates(ctrl_agg, control_selections, level_list)

        # Start with the merged post counts as the base frame
        base = merged_post.select(level_list + ["count"]).rename({"count": "count_post"})
        base = base.join(
            merged_ctrl.select(level_list + ["count"]).rename({"count": "count_control"}),
            on=level_list,
            how="left",
        ).fill_null(0)

        # Attach scalar metadata as columns
        base = base.with_columns([
            pl.lit(total_post).alias("total_post"),
            pl.lit(total_ctrl).alias("total_control"),
            pl.lit(diversity).alias("diversity"),
        ])

        # Run each method and join results onto base
        if "zscore" in methods:
            zscore_df = _apply_zscore(
                merged_post, merged_ctrl, level_list,
                total_post, total_ctrl, diversity, alpha,
            )
            base = base.join(zscore_df, on=level_list, how="left")

        if "poisson_ml" in methods:
            poisson_df = _apply_poisson_ml(
                merged_post, merged_ctrl, level_list,
                total_post, total_ctrl, diversity, alpha,
            )
            base = base.join(poisson_df, on=level_list, how="left")

        result[level_name] = base
        logger.info(
            "Level %s: %d features, diversity=%d, total_post=%d, total_ctrl=%d",
            level_name, len(base), diversity, total_post, total_ctrl,
        )

    return result


def run_deseq2_enrichment(
    counts_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    n_cycles: int,
    code_cols: list[str],
    levels: list[str] | None = None,
) -> dict[str, pl.DataFrame]:
    """Run PyDESeq2 at specified synthon levels (default: mono + di only).

    Trisynthon is excluded by default due to performance — use z-score and
    Poisson for trisynthon-level analysis.  Pass an explicit *levels* list
    to override.

    Args:
        counts_df: Combined long-format counts.
        metadata_df: Selection metadata with ``selection_name`` and
            ``condition`` columns (values: ``"protein"`` / ``"no_protein"``).
        n_cycles: Library cycle count.
        code_cols: All code column names.
        levels: Level names to run (e.g. ``["mono_code_1", "di_code_1_code_2"]``).
            If ``None``, all levels except the full-compound (N-synthon) level
            are used.

    Returns:
        Dict mapping level name → PyDESeq2 results DataFrame
        (``compound_id``, ``log2FoldChange``, ``pvalue``, ``padj``,
        ``baseMean``).

    Raises:
        ImportError: If ``pydeseq2`` is not installed.
        ValueError: If a requested level name is not valid for *n_cycles*.
    """
    from delexplore.analyse.deseq import run_deseq2  # lazy import for optional dep

    all_levels = get_all_levels(n_cycles)
    all_level_names = [get_level_name(lc) for lc in all_levels]

    # Default: skip the full-compound level (last entry = all code_cols)
    if levels is None:
        levels = [n for n, lc in zip(all_level_names, all_levels) if len(lc) < n_cycles]

    unknown = set(levels) - set(all_level_names)
    if unknown:
        raise ValueError(
            f"Unknown level names: {sorted(unknown)}. "
            f"Valid for n_cycles={n_cycles}: {all_level_names}"
        )

    result: dict[str, pl.DataFrame] = {}
    level_map = dict(zip(all_level_names, all_levels))

    for level_name in levels:
        level_cols = level_map[level_name]
        level_list = list(level_cols)
        logger.info("Running DESeq2 at level: %s", level_name)

        agg = aggregate_to_level(counts_df, level_cols)

        # For DESeq2 we need the "selection" column; aggregate_to_level keeps it.
        result[level_name] = run_deseq2(
            agg,
            metadata_df,
            level_list,
        )

    return result
