"""Naive (unselected) library synthesis quality assessment.

Scientific context (analysis_pipeline_architecture.md Part 4)
-------------------------------------------------------------
The naive library sequencing reveals what the DEL ACTUALLY looks like before
any selection bias.  If all building blocks were incorporated equally, naive
monosynthon counts would be uniform (after Poisson noise).  Deviations reveal:

1. **Synthesis yield bias** — some BBs are under- or over-represented.
2. **Truncated products** — monosynthon count >> expected, but disynthon counts
   for compounds using that BB are systematically low.  This means the first BB
   was incorporated but subsequent reactions failed for many partners.

These signals feed forward into the enrichment analysis:
- ``compute_bb_yield_weights`` produces normalization weights so that
  enrichment scores are not artificially boosted/suppressed by synthesis bias.
- ``detect_truncation`` flags BBs that are likely to produce false-positive
  enrichment hits (truncated products bind non-specifically).

Naive selections
----------------
In DELT-Hit config.yaml, naive / blank control selections are identified by
``target == "No Protein"`` or ``group == "no_protein"``.  The naive library
is not a special file — it is just one or more selections run without target
protein, which represent the ground-state library composition.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WEIGHT_CLIP_LOW  = 0.1   # minimum normalization weight
_WEIGHT_CLIP_HIGH = 10.0  # maximum normalization weight

# A BB is flagged as truncated when its monosynthon fraction exceeds the
# uniform expectation by this factor AND its disynthon fraction is below
# a fraction of that expectation.
_TRUNCATION_MONO_FACTOR = 2.0   # mono count ≥ 2× expected
_TRUNCATION_DI_FACTOR   = 0.5   # AND di count ≤ 0.5× expected


# ---------------------------------------------------------------------------
# 1. Identify naive / blank selections
# ---------------------------------------------------------------------------


def identify_naive_selections(config: dict[str, Any]) -> list[str]:
    """Find selections that represent the naive / unselected library.

    A selection is considered naive when any of the following hold
    (case-insensitive):

    - ``selections[name].target`` equals ``"No Protein"`` (DELT-Hit convention)
      or is null / empty / ``nan``.
    - ``selections[name].group`` equals ``"no_protein"``.

    Args:
        config: Parsed DELT-Hit config dict as returned by
            :func:`~delexplore.io.readers.read_config`.

    Returns:
        List of selection names suitable for naive library analysis.  May be
        empty if no blank/naive selections are configured.
    """
    naive: list[str] = []
    for sel_name, sel_info in config.get("selections", {}).items():
        if not isinstance(sel_info, dict):
            continue
        target = sel_info.get("target", None)
        group  = sel_info.get("group",  "")

        # Normalise target to a comparable string
        target_str = str(target).strip().lower() if target is not None else ""
        is_no_protein = target_str in ("no protein", "no_protein", "nan", "none", "")
        is_no_protein_group = str(group).strip().lower() == "no_protein"

        if is_no_protein or is_no_protein_group:
            naive.append(sel_name)

    logger.info("Identified %d naive selection(s): %s", len(naive), naive)
    return naive


# ---------------------------------------------------------------------------
# 2. Synthesis yield assessment
# ---------------------------------------------------------------------------


def assess_synthesis_yield(
    naive_counts: pl.DataFrame,
    n_cycles: int,
    code_cols: list[str],
) -> dict[str, Any]:
    """Assess per-BB synthesis yield relative to a uniform expectation.

    For each building block position, computes:

    - **observed_fraction**: fraction of total naive reads contributed by
      each BB (summed across all compounds containing that BB and all naive
      selections).
    - **expected_fraction**: ``1 / n_bbs_at_this_position`` (uniform assumption).
    - **yield_ratio**: ``observed / expected`` — ratio > 1 means over-represented,
      < 1 means under-represented.
    - **cv**: coefficient of variation of per-BB counts (std / mean).
    - **gini**: Gini coefficient of per-BB counts (0 = uniform, 1 = monopoly).

    Args:
        naive_counts: Combined long-format counts for naive selections.
            Must contain ``code_*`` columns and a ``count`` column.
            The ``selection`` column is optional; if present, counts are
            summed across all naive selections.
        n_cycles: Number of library cycles.
        code_cols: Code column names (e.g. ``["code_1", "code_2"]``).

    Returns:
        Dict with one key per code column (e.g. ``"code_1"``), each mapping to::

            {
                "bb_ids":             list[int],
                "observed_fraction":  list[float],
                "expected_fraction":  float,
                "yield_ratio":        list[float],
                "cv":                 float,
                "gini":               float,
                "outliers_high":      list[int],   # bb_ids with yield_ratio > 2
                "outliers_zero":      list[int],   # bb_ids with count = 0
            }

    Raises:
        ValueError: If *code_cols* is empty or ``count`` column is absent.
    """
    if not code_cols:
        raise ValueError("code_cols must not be empty")
    if "count" not in naive_counts.columns:
        raise ValueError("naive_counts must contain a 'count' column")

    total_reads = int(naive_counts["count"].sum())
    if total_reads == 0:
        logger.warning("naive_counts has zero total reads — all yields will be 0")

    result: dict[str, Any] = {}

    for col in code_cols:
        if col not in naive_counts.columns:
            logger.warning("Column %s not found in naive_counts, skipping", col)
            continue

        # Sum counts across all naive selections and all compound partners
        bb_counts = (
            naive_counts
            .group_by(col)
            .agg(pl.col("count").sum().alias("total_count"))
            .sort(col)
        )

        bb_ids     = bb_counts[col].to_list()
        counts_arr = bb_counts["total_count"].to_numpy().astype(float)
        n_bbs      = len(bb_ids)

        if n_bbs == 0:
            result[col] = {
                "bb_ids": [], "observed_fraction": [], "expected_fraction": 0.0,
                "yield_ratio": [], "cv": 0.0, "gini": 0.0,
                "outliers_high": [], "outliers_zero": [],
            }
            continue

        total_col        = counts_arr.sum()
        obs_frac         = counts_arr / total_col if total_col > 0 else counts_arr * 0
        expected_frac    = 1.0 / n_bbs
        yield_ratio      = obs_frac / expected_frac  # same as obs_frac * n_bbs

        mean_c = counts_arr.mean()
        cv     = (counts_arr.std() / mean_c) if mean_c > 0 else 0.0

        sorted_c = np.sort(counts_arr)
        n        = len(sorted_c)
        gini     = (
            (2.0 * np.sum(np.arange(1, n + 1) * sorted_c) / (n * sorted_c.sum()) - (n + 1) / n)
            if sorted_c.sum() > 0 else 0.0
        )

        outliers_high = [bb_ids[i] for i, r in enumerate(yield_ratio) if r > 2.0]
        outliers_zero = [bb_ids[i] for i, c in enumerate(counts_arr) if c == 0]

        result[col] = {
            "bb_ids":            bb_ids,
            "observed_fraction": obs_frac.tolist(),
            "expected_fraction": expected_frac,
            "yield_ratio":       yield_ratio.tolist(),
            "cv":                float(cv),
            "gini":              float(gini),
            "outliers_high":     outliers_high,
            "outliers_zero":     outliers_zero,
        }

    return result


# ---------------------------------------------------------------------------
# 3. Truncation detection
# ---------------------------------------------------------------------------


def detect_truncation(
    naive_counts: pl.DataFrame,
    n_cycles: int,
    code_cols: list[str],
) -> list[dict[str, Any]]:
    """Detect building blocks with likely truncated synthesis products.

    A BB at position *k* is flagged as a **truncation candidate** when:

    - Its monosynthon fraction is ≥ ``_TRUNCATION_MONO_FACTOR`` (2×) the
      uniform expectation, AND
    - The average disynthon fraction across all pairs that include this BB
      is ≤ ``_TRUNCATION_DI_FACTOR`` (0.5×) the expected disynthon fraction.

    This pattern means the first reaction incorporated BB *k*, but subsequent
    reactions largely failed — the DNA-encoded sequence shows many copies of
    BB *k* but not the expected partner BBs.  Such truncated products will
    be incorrectly enriched in selections.

    Args:
        naive_counts: Combined long-format counts for naive selections.
        n_cycles: Number of library cycles.
        code_cols: Code column names.

    Returns:
        List of dicts, one per flagged BB::

            {
                "code_col":           str,   # e.g. "code_1"
                "bb_id":              int,
                "mono_yield_ratio":   float,
                "mean_di_yield_ratio":float,
                "evidence":           str,   # human-readable summary
            }

        Empty list when no truncation candidates are found.

    Raises:
        ValueError: If *code_cols* is empty or ``count`` column is absent.
    """
    if not code_cols:
        raise ValueError("code_cols must not be empty")
    if "count" not in naive_counts.columns:
        raise ValueError("naive_counts must contain a 'count' column")

    if n_cycles < 2:
        # Truncation requires at least 2 cycle levels to compare
        return []

    flagged: list[dict[str, Any]] = []
    total_reads = float(naive_counts["count"].sum())
    if total_reads == 0:
        return []

    for col in code_cols:
        if col not in naive_counts.columns:
            continue

        # Monosynthon fraction per BB
        mono_agg = (
            naive_counts
            .group_by(col)
            .agg(pl.col("count").sum().alias("mono_count"))
        )
        n_bbs_mono = mono_agg.height
        if n_bbs_mono == 0:
            continue
        expected_mono_frac = 1.0 / n_bbs_mono

        # Disynthon fractions: for each partner column, compute fraction per (col, partner)
        partner_cols = [c for c in code_cols if c != col]
        if not partner_cols:
            continue

        for row in mono_agg.iter_rows(named=True):
            bb_id      = row[col]
            mono_count = row["mono_count"]
            mono_frac  = mono_count / total_reads if total_reads > 0 else 0.0
            mono_ratio = mono_frac / expected_mono_frac if expected_mono_frac > 0 else 0.0

            if mono_ratio < _TRUNCATION_MONO_FACTOR:
                continue

            # Check average disynthon fraction for this BB across all partner positions
            di_ratios: list[float] = []
            for partner in partner_cols:
                if partner not in naive_counts.columns:
                    continue
                di_agg = (
                    naive_counts
                    .filter(pl.col(col) == bb_id)
                    .group_by(partner)
                    .agg(pl.col("count").sum().alias("di_count"))
                )
                n_partners = di_agg.height
                if n_partners == 0:
                    continue
                expected_di_frac = 1.0 / (n_bbs_mono * n_partners)
                for di_row in di_agg.iter_rows(named=True):
                    di_frac  = di_row["di_count"] / total_reads if total_reads > 0 else 0.0
                    di_ratio = di_frac / expected_di_frac if expected_di_frac > 0 else 0.0
                    di_ratios.append(di_ratio)

            if not di_ratios:
                continue

            mean_di_ratio = float(np.mean(di_ratios))
            if mean_di_ratio <= _TRUNCATION_DI_FACTOR:
                evidence = (
                    f"{col}={bb_id}: mono_yield_ratio={mono_ratio:.2f} "
                    f"(≥{_TRUNCATION_MONO_FACTOR}×), "
                    f"mean_di_yield_ratio={mean_di_ratio:.2f} "
                    f"(≤{_TRUNCATION_DI_FACTOR}×) → suspected truncation"
                )
                flagged.append({
                    "code_col":            col,
                    "bb_id":               bb_id,
                    "mono_yield_ratio":    mono_ratio,
                    "mean_di_yield_ratio": mean_di_ratio,
                    "evidence":            evidence,
                })
                logger.info(evidence)

    flagged.sort(key=lambda r: (-r["mono_yield_ratio"], r["code_col"]))
    return flagged


# ---------------------------------------------------------------------------
# 4. BB yield weights for normalization
# ---------------------------------------------------------------------------


def compute_bb_yield_weights(
    naive_counts: pl.DataFrame,
    n_cycles: int,
    code_cols: list[str],
) -> pl.DataFrame:
    """Compute per-BB normalization weights from naive library counts.

    The weight for BB *i* at position *k* is::

        weight_i = median_yield / yield_i

    clipped to ``[_WEIGHT_CLIP_LOW, _WEIGHT_CLIP_HIGH]`` = [0.1, 10.0].

    A weight > 1 means the BB was under-synthesized — its selection counts
    should be up-weighted to correct for the deficit.  A weight < 1 means
    it was over-synthesized and should be down-weighted.

    The per-compound weight is the product of weights across all positions:
    ``compound_weight = weight_bb1 × weight_bb2 × …``

    This output is written to ``bb_yield_weights.parquet`` and consumed by
    the enrichment analysis pipeline.

    Args:
        naive_counts: Combined long-format counts for naive selections.
        n_cycles: Number of library cycles.
        code_cols: Code column names.

    Returns:
        DataFrame with one row per (code_col, bb_id) combination::

            code_col | bb_id | mono_count | yield_ratio | weight

    Raises:
        ValueError: If *code_cols* is empty or ``count`` column is absent.
    """
    if not code_cols:
        raise ValueError("code_cols must not be empty")
    if "count" not in naive_counts.columns:
        raise ValueError("naive_counts must contain a 'count' column")

    rows: list[dict[str, Any]] = []

    for col in code_cols:
        if col not in naive_counts.columns:
            logger.warning("Column %s not found in naive_counts, skipping", col)
            continue

        bb_counts = (
            naive_counts
            .group_by(col)
            .agg(pl.col("count").sum().alias("mono_count"))
            .sort(col)
        )

        counts_arr = bb_counts["mono_count"].to_numpy().astype(float)
        n_bbs      = len(counts_arr)
        if n_bbs == 0:
            continue

        total_col        = counts_arr.sum()
        obs_frac         = counts_arr / total_col if total_col > 0 else np.ones(n_bbs) / n_bbs
        expected_frac    = 1.0 / n_bbs
        yield_ratio      = obs_frac / expected_frac

        median_yield = float(np.median(yield_ratio))
        if median_yield == 0:
            median_yield = 1.0  # fallback to prevent division by zero

        raw_weight = median_yield / yield_ratio
        clipped    = np.clip(raw_weight, _WEIGHT_CLIP_LOW, _WEIGHT_CLIP_HIGH)

        for i, bb_id in enumerate(bb_counts[col].to_list()):
            rows.append({
                "code_col":    col,
                "bb_id":       int(bb_id),
                "mono_count":  int(counts_arr[i]),
                "yield_ratio": float(yield_ratio[i]),
                "weight":      float(clipped[i]),
            })

    if not rows:
        return pl.DataFrame(
            schema={
                "code_col":    pl.Utf8,
                "bb_id":       pl.Int64,
                "mono_count":  pl.Int64,
                "yield_ratio": pl.Float64,
                "weight":      pl.Float64,
            }
        )

    return pl.DataFrame(rows, schema={
        "code_col":    pl.Utf8,
        "bb_id":       pl.Int64,
        "mono_count":  pl.Int64,
        "yield_ratio": pl.Float64,
        "weight":      pl.Float64,
    })


# ---------------------------------------------------------------------------
# 5. Top-level runner
# ---------------------------------------------------------------------------


def run_naive_qc(
    naive_counts: pl.DataFrame,
    n_cycles: int,
    code_cols: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Run all naive library QC steps and write outputs to *output_dir*.

    Outputs written:
    - ``synthesis_yield.json``   — per-BB yield stats
    - ``truncation_flags.json``  — suspected truncation candidates
    - ``bb_yield_weights.parquet`` — normalization weights

    Args:
        naive_counts: Combined long-format counts for naive selections.
        n_cycles: Number of library cycles.
        code_cols: Code column names.
        output_dir: Directory to write outputs (created if absent).

    Returns:
        Dict with keys ``"synthesis_yield"``, ``"truncation_flags"``,
        ``"bb_yield_weights_path"`` (Path), and ``"n_flagged_bbs"`` (int).
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yield_stats   = assess_synthesis_yield(naive_counts, n_cycles, code_cols)
    truncation    = detect_truncation(naive_counts, n_cycles, code_cols)
    yield_weights = compute_bb_yield_weights(naive_counts, n_cycles, code_cols)

    # Write JSON outputs
    with (output_dir / "synthesis_yield.json").open("w") as fh:
        json.dump(yield_stats, fh, indent=2)

    with (output_dir / "truncation_flags.json").open("w") as fh:
        json.dump(truncation, fh, indent=2)

    weights_path = output_dir / "bb_yield_weights.parquet"
    yield_weights.write_parquet(weights_path)

    n_flagged = len(truncation)
    logger.info(
        "Naive QC complete: %d truncation candidates, weights written to %s",
        n_flagged, weights_path,
    )

    return {
        "synthesis_yield":      yield_stats,
        "truncation_flags":     truncation,
        "bb_yield_weights_path": weights_path,
        "n_flagged_bbs":        n_flagged,
    }
