"""P(bind) building block productivity analysis.

Scientific motivation (Zhang et al. 2023, JCIM)
------------------------------------------------
For each building block at each library position, ``P(bind)`` is the fraction
of compounds that contain that BB and are classified as binders.  The metric
is position-dependent: a BB at cycle 1 may be a strong pharmacophore carrier
while the same structural motif at cycle 2 is inactive.

Key findings from Zhang et al.:
  - A decision-tree model trained on P(bind) alone achieves 96.1% AUC-PR.
  - P(bind) is a better SAR signal than raw enrichment count, because it
    normalises out differences in library representation.
  - Joint P(bind) between positions reveals cooperative binding: a pair of BBs
    is "cooperative" when the joint P(bind) exceeds the product of their
    individual P(bind) values (positive interaction).

Integration with consensus ranking
-----------------------------------
``compute_pbind_support_score`` produces a continuous support score (sum of
constituent P(bind) values) that replaces or complements the binary support
score in :mod:`delexplore.analyse.rank`.  Compounds whose every BB is a strong
pharmacophore carrier receive a high continuous score, driving them to the top
of the ranked list.

Usage
-----
>>> from delexplore.analyse.bb_productivity import (
...     compute_pbind,
...     compute_joint_pbind,
...     compute_pbind_support_score,
...     identify_productive_bbs,
... )
>>> pbind = compute_pbind(full_compound_df, code_cols=["code_1", "code_2"])
>>> support = compute_pbind_support_score(pbind, compound_df, code_cols)
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_score_col(df: pl.DataFrame, score_col: str) -> None:
    if score_col not in df.columns:
        raise ValueError(
            f"score_col '{score_col}' not found in DataFrame. "
            f"Available columns: {df.columns}"
        )


def _validate_code_cols(df: pl.DataFrame, code_cols: list[str]) -> None:
    if not code_cols:
        raise ValueError("code_cols must not be empty")
    missing = [c for c in code_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"code_cols not found in DataFrame: {missing}. "
            f"Available columns: {df.columns}"
        )


def _mark_binders(
    df: pl.DataFrame,
    score_col: str,
    binder_threshold: float,
) -> pl.DataFrame:
    """Add a boolean ``_is_binder`` column based on score threshold."""
    return df.with_columns(
        (pl.col(score_col) >= binder_threshold).alias("_is_binder")
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_pbind(
    enrichment_df: pl.DataFrame,
    code_cols: list[str],
    score_col: str = "zscore",
    binder_threshold: float = 1.0,
) -> dict[str, pl.DataFrame]:
    """Compute per-BB binding probability for each library position.

    For each code column (position), each unique BB value is assigned a
    ``p_bind`` — the fraction of compounds containing that BB that are
    classified as binders.  This is the core metric from Zhang et al. (2023).

    Args:
        enrichment_df: Full-compound level enrichment DataFrame.  Must contain
            all columns in *code_cols* and *score_col*.  One row per compound.
        code_cols: Column names identifying the library position of each BB
            (e.g. ``["code_1", "code_2"]``).
        score_col: Enrichment score column used to classify binders.  Any
            compound with ``score_col >= binder_threshold`` is a binder.
        binder_threshold: Minimum enrichment score to classify a compound as
            a binder.  For normalized z-score, 1.0 is typical; for
            ``fold_enrichment``, 2.0 is recommended.

    Returns:
        Dict mapping each code column name to a Polars DataFrame with columns:

        - *code_col* — BB index (integer).
        - ``n_total`` — total compounds in the library containing this BB.
        - ``n_binders`` — compounds containing this BB classified as binders.
        - ``p_bind`` — ``n_binders / n_total`` (0.0–1.0).
        - ``n_compatible_partners`` — number of distinct partner BB combinations
          (from all other positions) that appear in binder compounds containing
          this BB.  High values indicate a BB that binds across many chemical
          contexts (robust pharmacophore carrier).

        Rows are sorted by ``p_bind`` descending.

    Raises:
        ValueError: If *code_cols* is empty, any column is absent, or the
            DataFrame is empty.
    """
    _validate_code_cols(enrichment_df, code_cols)
    _validate_score_col(enrichment_df, score_col)

    if len(enrichment_df) == 0:
        raise ValueError("enrichment_df must not be empty")

    tagged = _mark_binders(enrichment_df, score_col, binder_threshold)

    result: dict[str, pl.DataFrame] = {}

    for col in code_cols:
        partner_cols = [c for c in code_cols if c != col]

        # Group by this position's BB to get n_total and n_binders
        agg = (
            tagged
            .group_by(col)
            .agg([
                pl.len().alias("n_total"),
                pl.col("_is_binder").sum().cast(pl.Int64).alias("n_binders"),
            ])
            .with_columns(
                (pl.col("n_binders") / pl.col("n_total")).alias("p_bind")
            )
        )

        # n_compatible_partners: distinct partner combinations in binder compounds
        if partner_cols:
            binders_only = tagged.filter(pl.col("_is_binder"))
            if len(binders_only) > 0:
                partner_counts = (
                    binders_only
                    .group_by(col)
                    .agg(
                        pl.concat_str(partner_cols, separator="_")
                        .n_unique()
                        .alias("n_compatible_partners")
                    )
                )
                agg = agg.join(partner_counts, on=col, how="left").with_columns(
                    pl.col("n_compatible_partners").fill_null(0)
                )
            else:
                agg = agg.with_columns(pl.lit(0).alias("n_compatible_partners"))
        else:
            # Single-position library: no partners
            agg = agg.with_columns(pl.lit(0).alias("n_compatible_partners"))

        result[col] = agg.sort("p_bind", descending=True)

    n_binders_total = int(tagged["_is_binder"].sum())
    logger.info(
        "compute_pbind: %d binders out of %d compounds (threshold %s >= %.2f)",
        n_binders_total,
        len(enrichment_df),
        score_col,
        binder_threshold,
    )

    return result


def compute_joint_pbind(
    enrichment_df: pl.DataFrame,
    code_cols: list[str],
    score_col: str = "zscore",
    binder_threshold: float = 1.0,
    n_bins: int = 4,
) -> dict[str, pl.DataFrame]:
    """Compute joint P(bind) across pairs of library positions.

    For each pair of positions ``(code_i, code_j)``, each compound is
    assigned to a P(bind) bin for each position based on the per-BB
    ``p_bind`` value computed by :func:`compute_pbind`.  The joint P(bind)
    for each ``(bin_i, bin_j)`` combination is then the fraction of compounds
    in that cell that are binders.

    A high joint P(bind) in the ``(high, high)`` bin that exceeds the product
    of the two marginal P(bind) values indicates **cooperative binding** —
    the pair of BBs synergises to create a pharmacophore that neither creates
    alone.

    Args:
        enrichment_df: Full-compound enrichment DataFrame with *code_cols* and
            *score_col*.
        code_cols: Code column names.  All pairwise combinations are computed.
        score_col: Enrichment score column.
        binder_threshold: Minimum score to classify a compound as a binder.
        n_bins: Number of equally-spaced P(bind) bins per position (default 4
            → quartile bins).

    Returns:
        Dict keyed by ``"code_i__code_j"`` for each pair, mapping to a DataFrame
        with columns:

        - ``bin_i``, ``bin_j`` — bin index (0 = lowest P(bind), n_bins−1 = highest).
        - ``bin_i_label``, ``bin_j_label`` — human-readable bin range strings.
        - ``n_compounds`` — compounds in this bin combination.
        - ``n_binders`` — binders in this bin combination.
        - ``joint_p_bind`` — ``n_binders / n_compounds``.
        - ``marginal_product`` — product of the two marginal P(bind) bin means
          (expected under independence).
        - ``interaction`` — ``joint_p_bind − marginal_product`` (positive =
          cooperative, negative = antagonistic).

        Rows with zero compounds are omitted.

    Raises:
        ValueError: If *code_cols* has fewer than 2 elements, or any required
            column is absent.
    """
    _validate_code_cols(enrichment_df, code_cols)
    _validate_score_col(enrichment_df, score_col)

    if len(code_cols) < 2:
        raise ValueError(
            "compute_joint_pbind requires at least 2 code columns, "
            f"got {code_cols}"
        )

    # Compute per-position P(bind) for each compound
    pbind_per_pos = compute_pbind(
        enrichment_df, code_cols, score_col, binder_threshold
    )

    tagged = _mark_binders(enrichment_df, score_col, binder_threshold)

    result: dict[str, pl.DataFrame] = {}

    for col_i, col_j in combinations(code_cols, 2):
        pb_i = pbind_per_pos[col_i].select([col_i, "p_bind"]).rename(
            {"p_bind": f"_pb_{col_i}"}
        )
        pb_j = pbind_per_pos[col_j].select([col_j, "p_bind"]).rename(
            {"p_bind": f"_pb_{col_j}"}
        )

        # Join P(bind) values back onto the compound DataFrame
        df = (
            tagged
            .join(pb_i, on=col_i, how="left")
            .join(pb_j, on=col_j, how="left")
            .fill_null(0.0)
        )

        # Assign bin indices using floor division of quantile breakpoints
        edges = np.linspace(0.0, 1.0, n_bins + 1)

        def _bin_col(pb_col: str, alias: str) -> pl.Expr:
            # np.searchsorted-style binning: bin = argmax(edges > p_bind) - 1
            # Clamp to [0, n_bins-1]
            expr = pl.lit(0)
            for k in range(1, n_bins):
                expr = pl.when(pl.col(pb_col) >= float(edges[k])).then(pl.lit(k)).otherwise(expr)
            return expr.clip(0, n_bins - 1).alias(alias)

        df = df.with_columns([
            _bin_col(f"_pb_{col_i}", "_bin_i"),
            _bin_col(f"_pb_{col_j}", "_bin_j"),
        ])

        # Build bin labels
        bin_labels = [
            f"{edges[k]:.2f}–{edges[k+1]:.2f}" for k in range(n_bins)
        ]

        # Aggregate
        agg = (
            df.group_by(["_bin_i", "_bin_j"])
            .agg([
                pl.len().alias("n_compounds"),
                pl.col("_is_binder").sum().cast(pl.Int64).alias("n_binders"),
                pl.col(f"_pb_{col_i}").mean().alias("_mean_pb_i"),
                pl.col(f"_pb_{col_j}").mean().alias("_mean_pb_j"),
            ])
            .filter(pl.col("n_compounds") > 0)
            .with_columns([
                (pl.col("n_binders") / pl.col("n_compounds")).alias("joint_p_bind"),
                (pl.col("_mean_pb_i") * pl.col("_mean_pb_j")).alias("marginal_product"),
            ])
            .with_columns(
                (pl.col("joint_p_bind") - pl.col("marginal_product")).alias("interaction")
            )
        )

        # Add human-readable labels
        label_map_i = {k: bin_labels[k] for k in range(n_bins)}
        label_map_j = {k: bin_labels[k] for k in range(n_bins)}

        agg = agg.with_columns([
            pl.col("_bin_i").replace_strict(label_map_i, default=None).alias("bin_i_label"),
            pl.col("_bin_j").replace_strict(label_map_j, default=None).alias("bin_j_label"),
        ])

        # Rename and reorder output columns
        agg = (
            agg
            .rename({"_bin_i": "bin_i", "_bin_j": "bin_j"})
            .drop(["_mean_pb_i", "_mean_pb_j"])
            .select([
                "bin_i", "bin_j",
                "bin_i_label", "bin_j_label",
                "n_compounds", "n_binders",
                "joint_p_bind", "marginal_product", "interaction",
            ])
            .sort(["bin_i", "bin_j"])
        )

        key = f"{col_i}__{col_j}"
        result[key] = agg
        logger.debug(
            "joint_pbind %s: %d bin combinations", key, len(agg)
        )

    return result


def compute_pbind_support_score(
    pbind_results: dict[str, pl.DataFrame],
    compound_df: pl.DataFrame,
    code_cols: list[str],
) -> np.ndarray:
    """Compute a continuous support score from per-BB P(bind) values.

    For each compound, the score is the **sum of P(bind) values** for each of
    its constituent BBs across all library positions.  This is a continuous
    analogue of the binary support score in
    :func:`~delexplore.analyse.rank.compute_support_score`:

    - Binary support: +1 if BB is enriched (P(bind) above threshold)
    - P(bind) support: += P(bind) for every BB (no threshold needed)

    The maximum possible value is ``len(code_cols)`` (all BBs have P(bind) = 1).
    A compound whose every BB is a strong pharmacophore carrier will score close
    to the maximum.

    Args:
        pbind_results: Output of :func:`compute_pbind`.  Maps code column name
            to a DataFrame with at least *code_col* and ``p_bind`` columns.
        compound_df: One row per compound to score.  Must contain all columns
            in *code_cols*.
        code_cols: Code column names (must match keys in *pbind_results*).

    Returns:
        1-D numpy array of float64 support scores, one per row of *compound_df*.
        All values are in ``[0.0, len(code_cols)]``.

    Raises:
        ValueError: If *code_cols* is empty or any column is absent from
            *compound_df*.
    """
    if not code_cols:
        raise ValueError("code_cols must not be empty")

    missing = [c for c in code_cols if c not in compound_df.columns]
    if missing:
        raise ValueError(
            f"code_cols not found in compound_df: {missing}. "
            f"Available: {compound_df.columns}"
        )

    n = len(compound_df)
    scores = np.zeros(n, dtype=float)

    for col in code_cols:
        if col not in pbind_results:
            logger.debug("P(bind) not available for %s — skipping", col)
            continue

        pb = pbind_results[col].select([col, "p_bind"])

        joined = (
            compound_df
            .select([col])
            .join(pb, on=col, how="left")
            .with_columns(pl.col("p_bind").fill_null(0.0))
        )
        scores += joined["p_bind"].to_numpy()

    return scores


def identify_productive_bbs(
    pbind_results: dict[str, pl.DataFrame],
    top_fraction: float = 0.01,
) -> dict[str, list[int]]:
    """Identify the top-fraction of BBs by P(bind) per library position.

    The top 1% (by default) of BBs are the **pharmacophore carriers** —
    the building blocks that most consistently appear in active compounds.
    These are strong candidates for SAR investigation and library refinement.

    Args:
        pbind_results: Output of :func:`compute_pbind`.
        top_fraction: Fraction of BBs per position to return (e.g. 0.01 = top
            1%).  At least 1 BB is always returned.

    Returns:
        Dict mapping each code column name to a list of BB indices (integers),
        sorted by P(bind) descending.  Ties are broken by ``n_binders``
        descending.

    Raises:
        ValueError: If *top_fraction* is not in ``(0, 1]``.
    """
    if not (0.0 < top_fraction <= 1.0):
        raise ValueError(
            f"top_fraction must be in (0, 1], got {top_fraction}"
        )

    result: dict[str, list[int]] = {}

    for col, df in pbind_results.items():
        if len(df) == 0:
            result[col] = []
            continue

        n_top = max(1, int(np.ceil(len(df) * top_fraction)))

        top = (
            df
            .sort(["p_bind", "n_binders"], descending=[True, True])
            .head(n_top)
        )
        result[col] = top[col].to_list()

        logger.debug(
            "identify_productive_bbs: %s → %d of %d BBs (top %.1f%%)",
            col, n_top, len(df), top_fraction * 100,
        )

    return result
