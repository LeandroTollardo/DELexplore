"""Data quality assessment for DEL screening data.

Produces ``data_quality.json`` (consumed by the analysis module) and an
optional HTML report.  Run this before any enrichment analysis.

Thresholds (from docs/references/analysis_pipeline_architecture.md Part 2)
--------------------------------------------------------------------------

  Metric                 Green      Yellow     Red
  ─────────────────────  ─────────  ─────────  ──────
  Sampling ratio         > 10       1–10       < 1
  Replicate Pearson R    > 0.70     0.50–0.70  < 0.50
  BB coverage (%)        > 95 %     80–95 %    < 80 %
  Max single-BB frac.    < 5 %      5–15 %     > 15 %

Connection to analysis
----------------------
The analysis module reads ``data_quality.json`` and:
  - Flags trisynthon results as "low confidence" when sampling_ratio < 1
  - Excludes a condition when replicate_correlation < 0.5
  - Notes potential synthesis issues when BB coverage < 80 %
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from jinja2 import Environment, FileSystemLoader
from scipy.stats import pearsonr

from delexplore.analyse.aggregate import (
    get_all_levels,
    get_diversity,
    get_level_name,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

_SR_GREEN  = 10.0   # sampling ratio
_SR_YELLOW =  1.0

_RC_GREEN  = 0.70   # replicate correlation
_RC_YELLOW = 0.50

_COV_GREEN  = 0.95  # BB coverage fraction
_COV_YELLOW = 0.80

_MAX_BB_FRAC_GREEN  = 0.05  # max single-BB fraction of total reads
_MAX_BB_FRAC_YELLOW = 0.15

_TEMPLATES_DIR = Path(__file__).parent / "templates"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_high(value: float, green: float, yellow: float) -> str:
    """Classify where higher is better."""
    if value > green:
        return "green"
    if value > yellow:
        return "yellow"
    return "red"


def _classify_low(value: float, green: float, yellow: float) -> str:
    """Classify where lower is better."""
    if value < green:
        return "green"
    if value < yellow:
        return "yellow"
    return "red"


def _worst_status(*statuses: str) -> str:
    """Return the worst status among a collection."""
    priority = {"red": 2, "yellow": 1, "green": 0}
    return max(statuses, key=lambda s: priority.get(s, 0))


def _gini(values: np.ndarray) -> float:
    """Gini coefficient (0 = perfect equality, 1 = complete inequality)."""
    v = np.sort(values.flatten().astype(float))
    n = len(v)
    total = v.sum()
    if n == 0 or total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float(((2 * index - n - 1) * v).sum() / (n * total))


def _cv(values: np.ndarray) -> float:
    """Coefficient of variation (std / mean).  Returns 0 when mean is 0."""
    mean = float(values.mean())
    return 0.0 if mean == 0 else float(values.std() / mean)


def _n_expected_bbs(config: dict[str, Any], position: int) -> int | None:
    """Return expected number of BBs at *position* (0-indexed) from config.

    Returns ``None`` when the config does not enumerate building blocks.
    """
    lib = config.get("library", {})
    key = f"B{position}"
    bbs = lib.get(key)
    if bbs is None:
        return None
    return len(bbs)


# ---------------------------------------------------------------------------
# Public assessment functions
# ---------------------------------------------------------------------------


def assess_sequencing_depth(
    counts_df: pl.DataFrame,
    n_cycles: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute sampling ratio (average reads / diversity) per synthon level.

    Sampling ratio = (mean total reads per selection) / diversity.
    A ratio > 10 means on average each feature is observed 10 times →
    adequate power.  A ratio < 1 means most features are expected to have
    zero counts by chance alone.

    Args:
        counts_df: Combined long-format counts with a ``selection`` column.
        n_cycles: Library cycle count.
        config: Parsed config dict (currently unused; reserved for future use).

    Returns:
        Dict with keys:
        - ``"sampling_ratio"`` — {level_name: float}
        - ``"status"``         — {level_name: "green"|"yellow"|"red"}
        - ``"mean_reads_per_selection"`` — float
    """
    per_sel = (
        counts_df
        .group_by("selection")
        .agg(pl.col("count").sum().alias("total"))
    )
    mean_total = float(per_sel["total"].mean())

    ratios: dict[str, float] = {}
    statuses: dict[str, str] = {}

    for level_cols in get_all_levels(n_cycles):
        name = get_level_name(level_cols)
        diversity = get_diversity(counts_df, level_cols)
        ratio = mean_total / diversity if diversity > 0 else 0.0
        ratios[name] = round(ratio, 2)
        statuses[name] = _classify_high(ratio, _SR_GREEN, _SR_YELLOW)

    return {
        "sampling_ratio": ratios,
        "status": statuses,
        "mean_reads_per_selection": round(mean_total, 1),
    }


def assess_replicate_correlation(
    counts_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    code_cols: list[str],
) -> dict[str, Any]:
    """Compute mean pairwise Pearson correlation between replicates per group.

    For each group in ``metadata_df``, all pairs of selections are aligned
    on compound identity (missing compounds filled with 0) and the Pearson R
    of their count vectors is computed.  The result is the mean over all pairs.

    Args:
        counts_df: Long-format counts.
        metadata_df: Polars DataFrame with ``selection_name`` and ``group``
            columns.
        code_cols: Code column names used to identify compounds.

    Returns:
        Dict with keys:
        - ``"correlation"`` — {group: float}
        - ``"status"``      — {group: "green"|"yellow"|"red"}
        - ``"n_pairs"``     — {group: int}
    """
    # Build compound_id for joining
    id_expr = pl.concat_str([pl.col(c).cast(pl.Utf8) for c in code_cols], separator="_")
    df = counts_df.with_columns(id_expr.alias("_cid"))

    correlations: dict[str, float] = {}
    statuses: dict[str, str] = {}
    n_pairs: dict[str, int] = {}

    groups = (
        metadata_df
        .group_by("group")
        .agg(pl.col("selection_name").alias("selections"))
    )

    for row in groups.iter_rows(named=True):
        group = row["group"]
        sels = row["selections"]

        if len(sels) < 2:
            logger.debug("Group %s has only 1 selection — skipping correlation", group)
            n_pairs[group] = 0
            correlations[group] = float("nan")
            statuses[group] = "yellow"  # can't assess
            continue

        # Build a wide matrix: rows = compound_ids, columns = selections
        sel_frames: list[pl.DataFrame] = []
        for sel in sels:
            sel_df = (
                df.filter(pl.col("selection") == sel)
                .select(["_cid", "count"])
                .rename({"count": sel})
            )
            sel_frames.append(sel_df)

        # Outer join all selections on compound id; fill 0 for absent compounds
        wide = sel_frames[0]
        for sf in sel_frames[1:]:
            wide = wide.join(sf, on="_cid", how="full", coalesce=True).fill_null(0)

        rs: list[float] = []
        pair_count = 0
        for s1, s2 in combinations(sels, 2):
            if s1 not in wide.columns or s2 not in wide.columns:
                continue
            v1 = wide[s1].to_numpy().astype(float)
            v2 = wide[s2].to_numpy().astype(float)
            if v1.std() == 0 or v2.std() == 0:
                continue  # constant vector → correlation undefined
            r, _ = pearsonr(v1, v2)
            rs.append(float(r))
            pair_count += 1

        mean_r = float(np.mean(rs)) if rs else float("nan")
        correlations[group] = round(mean_r, 4)
        n_pairs[group] = pair_count
        statuses[group] = (
            _classify_high(mean_r, _RC_GREEN, _RC_YELLOW)
            if not np.isnan(mean_r)
            else "yellow"
        )

    return {
        "correlation": correlations,
        "status": statuses,
        "n_pairs": n_pairs,
    }


def assess_bb_coverage(
    counts_df: pl.DataFrame,
    n_cycles: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute per-position building block coverage.

    Coverage = (number of BBs with at least one observed read) / (expected BBs).
    Expected count comes from config when available; otherwise the observed
    maximum index + 1 is used as a lower bound.

    Args:
        counts_df: Long-format counts (across all selections combined).
        n_cycles: Library cycle count.
        config: Parsed config dict.

    Returns:
        Dict with keys:
        - ``"coverage"``    — {"code_1": float, ...}
        - ``"n_observed"``  — {"code_1": int, ...}
        - ``"n_expected"``  — {"code_1": int, ...}
        - ``"status"``      — {"code_1": "green"|"yellow"|"red"}
        - ``"max_bb_fraction"`` — {"code_1": float, ...}
        - ``"max_bb_fraction_status"`` — {"code_1": str}
    """
    coverage: dict[str, float] = {}
    n_obs: dict[str, int] = {}
    n_exp: dict[str, int] = {}
    statuses: dict[str, str] = {}
    max_frac: dict[str, float] = {}
    max_frac_status: dict[str, str] = {}

    for i in range(1, n_cycles + 1):
        col = f"code_{i}"
        position = i - 1  # 0-indexed for config lookup

        # Aggregate across all selections to get monosynthon total per BB
        mono_agg = (
            counts_df
            .group_by(col)
            .agg(pl.col("count").sum())
        )

        observed_with_reads = int((mono_agg["count"] > 0).sum())
        total_reads = int(mono_agg["count"].sum())

        # Expected BB count from config, fall back to observed max + 1
        n_from_config = _n_expected_bbs(config, position)
        expected = n_from_config if n_from_config is not None else int(mono_agg[col].max()) + 1

        frac = observed_with_reads / expected if expected > 0 else 1.0
        coverage[col] = round(frac, 4)
        n_obs[col] = observed_with_reads
        n_exp[col] = expected
        statuses[col] = _classify_high(frac, _COV_GREEN, _COV_YELLOW)

        # Max single-BB fraction of total reads (bead-binder / contamination check)
        if total_reads > 0:
            max_bb = int(mono_agg["count"].max())
            mf = max_bb / total_reads
        else:
            mf = 0.0
        max_frac[col] = round(mf, 4)
        max_frac_status[col] = _classify_low(mf, _MAX_BB_FRAC_GREEN, _MAX_BB_FRAC_YELLOW)

    return {
        "coverage": coverage,
        "n_observed": n_obs,
        "n_expected": n_exp,
        "status": statuses,
        "max_bb_fraction": max_frac,
        "max_bb_fraction_status": max_frac_status,
    }


def assess_bb_uniformity(
    counts_df: pl.DataFrame,
    n_cycles: int,
) -> dict[str, Any]:
    """Assess synthesis uniformity per BB position.

    Computes the coefficient of variation (CV) and Gini coefficient of
    monosynthon counts per position.  Low CV and Gini → uniform synthesis.
    Identifies outlier BBs (count > mean + 3 SD or count == 0).

    Args:
        counts_df: Long-format counts.
        n_cycles: Library cycle count.

    Returns:
        Dict with keys per position:
        - ``"cv"``           — {"code_1": float, ...}
        - ``"gini"``         — {"code_1": float, ...}
        - ``"outliers_high"``— {"code_1": [bb_index, ...]}  # over-represented
        - ``"outliers_zero"``— {"code_1": [bb_index, ...]}  # never observed
    """
    cv_result: dict[str, float] = {}
    gini_result: dict[str, float] = {}
    outliers_high: dict[str, list[int]] = {}
    outliers_zero: dict[str, list[int]] = {}

    for i in range(1, n_cycles + 1):
        col = f"code_{i}"

        mono_agg = (
            counts_df
            .group_by(col)
            .agg(pl.col("count").sum())
            .sort(col)
        )

        vals = mono_agg["count"].to_numpy().astype(float)
        codes = mono_agg[col].to_numpy()

        cv_result[col] = round(_cv(vals), 4)
        gini_result[col] = round(_gini(vals), 4)

        mean, std = vals.mean(), vals.std()
        threshold = mean + 3 * std

        outliers_high[col] = [int(c) for c, v in zip(codes, vals) if v > threshold]
        outliers_zero[col] = [int(c) for c, v in zip(codes, vals) if v == 0]

    return {
        "cv": cv_result,
        "gini": gini_result,
        "outliers_high": outliers_high,
        "outliers_zero": outliers_zero,
    }


# ---------------------------------------------------------------------------
# Warnings + recommendations
# ---------------------------------------------------------------------------


def _build_warnings(
    depth: dict[str, Any],
    correlation: dict[str, Any],
    coverage: dict[str, Any],
    uniformity: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Derive human-readable warnings and recommended analysis levels."""
    warnings: list[str] = []
    recommended: list[str] = []

    # Sampling ratio warnings
    for level, status in depth["status"].items():
        ratio = depth["sampling_ratio"][level]
        if status == "red":
            warnings.append(
                f"{level} analysis underpowered (sampling_ratio={ratio:.1f} < 1)"
            )
        elif status == "yellow":
            warnings.append(
                f"{level} analysis may be underpowered (sampling_ratio={ratio:.1f})"
            )
        else:
            recommended.append(level)

    if not recommended:
        # Always recommend at least monosynthon levels
        recommended = [k for k in depth["status"] if k.startswith("mono")]

    # Replicate correlation warnings
    for group, status in correlation["status"].items():
        r = correlation["correlation"].get(group, float("nan"))
        if status == "red":
            warnings.append(
                f"Low replicate correlation in group '{group}' (r={r:.3f} < 0.5). "
                "Consider excluding this condition."
            )
        elif status == "yellow":
            warnings.append(
                f"Moderate replicate correlation in group '{group}' (r={r:.3f})."
            )

    # BB coverage warnings
    for col, status in coverage["status"].items():
        frac = coverage["coverage"][col]
        n_missing = coverage["n_expected"][col] - coverage["n_observed"][col]
        if status == "red":
            warnings.append(
                f"{col}: only {frac:.0%} BB coverage ({n_missing} BBs missing). "
                "Likely synthesis failures."
            )
        elif status == "yellow":
            warnings.append(
                f"{col}: {frac:.0%} BB coverage ({n_missing} BBs missing)."
            )

    # Max single-BB fraction warnings
    for col, status in coverage["max_bb_fraction_status"].items():
        mf = coverage["max_bb_fraction"][col]
        if status == "red":
            warnings.append(
                f"{col}: one BB accounts for {mf:.1%} of reads — "
                "possible bead binder or synthesis artefact."
            )
        elif status == "yellow":
            warnings.append(
                f"{col}: one BB accounts for {mf:.1%} of reads."
            )

    # BB uniformity
    for col in uniformity["outliers_zero"]:
        zeros = uniformity["outliers_zero"][col]
        if zeros:
            warnings.append(
                f"{col}: {len(zeros)} BBs with zero reads "
                f"(indices: {zeros[:5]}{'...' if len(zeros) > 5 else ''})."
            )

    return warnings, recommended


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_quality_report(
    counts_df: pl.DataFrame,
    n_cycles: int,
    config: dict[str, Any],
    output_dir: Path,
    metadata_df: pl.DataFrame | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run all QC assessments, write data_quality.json and HTML report.

    Args:
        counts_df: Combined long-format counts (all selections).
        n_cycles: Library cycle count.
        config: Parsed config dict.
        output_dir: Directory to write outputs.  Created if absent.
        metadata_df: Optional Polars DataFrame with ``selection_name`` and
            ``group`` columns.  When ``None``, replicate correlation is
            skipped.
        alpha: Reserved for future significance tests.

    Returns:
        The full quality report dict (also written to ``data_quality.json``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    code_cols = [f"code_{i}" for i in range(1, n_cycles + 1)]

    # ── Run assessments ───────────────────────────────────────────────────

    depth = assess_sequencing_depth(counts_df, n_cycles, config)

    correlation: dict[str, Any] = {"correlation": {}, "status": {}, "n_pairs": {}}
    if metadata_df is not None and "group" in metadata_df.columns:
        correlation = assess_replicate_correlation(counts_df, metadata_df, code_cols)
    else:
        logger.info("No metadata_df provided — skipping replicate correlation")

    coverage = assess_bb_coverage(counts_df, n_cycles, config)
    uniformity = assess_bb_uniformity(counts_df, n_cycles)

    # ── Overall quality ───────────────────────────────────────────────────

    all_statuses: list[str] = (
        list(depth["status"].values())
        + list(correlation["status"].values())
        + list(coverage["status"].values())
        + list(coverage["max_bb_fraction_status"].values())
    )
    overall = _worst_status(*all_statuses) if all_statuses else "green"

    warnings, recommended = _build_warnings(depth, correlation, coverage, uniformity)

    # ── Build report dict ─────────────────────────────────────────────────

    report: dict[str, Any] = {
        "overall_quality": overall,
        "sampling_ratio": depth["sampling_ratio"],
        "sampling_ratio_status": depth["status"],
        "mean_reads_per_selection": depth["mean_reads_per_selection"],
        "replicate_correlation": correlation["correlation"],
        "replicate_correlation_status": correlation["status"],
        "bb_coverage": coverage["coverage"],
        "bb_coverage_status": coverage["status"],
        "bb_coverage_n_observed": coverage["n_observed"],
        "bb_coverage_n_expected": coverage["n_expected"],
        "max_bb_fraction": coverage["max_bb_fraction"],
        "max_bb_fraction_status": coverage["max_bb_fraction_status"],
        "bb_uniformity_cv": uniformity["cv"],
        "bb_uniformity_gini": uniformity["gini"],
        "bb_outliers_high": uniformity["outliers_high"],
        "bb_outliers_zero": uniformity["outliers_zero"],
        "warnings": warnings,
        "recommended_analysis_levels": recommended,
    }

    # ── Write data_quality.json ───────────────────────────────────────────

    json_path = output_dir / "data_quality.json"
    with json_path.open("w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Wrote %s", json_path)

    # ── Write HTML report ─────────────────────────────────────────────────

    try:
        env = Environment(loader=FileSystemLoader(str(_TEMPLATES_DIR)), autoescape=True)
        template = env.get_template("qc_report.html")
        experiment_name = config.get("experiment", {}).get("name", "DELexplore")
        html = template.render(
            experiment_name=experiment_name,
            report=report,
            n_cycles=n_cycles,
        )
        html_path = output_dir / "qc_report.html"
        html_path.write_text(html)
        logger.info("Wrote %s", html_path)
    except Exception as exc:
        logger.warning("HTML report generation failed: %s", exc)

    return report
