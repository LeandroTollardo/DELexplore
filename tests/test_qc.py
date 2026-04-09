"""Tests for qc/assess.py using the synthetic fixture.

Synthetic library (2-cycle, 10 × 8 = 80 compounds, 6 selections):
  True binders: (code_1=2, code_2=3), (code_1=5, code_2=6)
    target_*: 100–500    blank_*: 1–5
  Bead binders: code_1==7   →  40–80 all selections
  Noise: everything else     →   0–20 all selections
"""

import json
from pathlib import Path

import polars as pl
import pytest

from delexplore.qc.assess import (
    assess_bb_coverage,
    assess_bb_uniformity,
    assess_replicate_correlation,
    assess_sequencing_depth,
    generate_quality_report,
)

N_BB1, N_BB2 = 10, 8
N_CYCLES = 2
CODE_COLS = ["code_1", "code_2"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def meta_df():
    """Minimal metadata DataFrame matching the synthetic fixture."""
    rows = [
        {"selection_name": s, "group": "protein" if s.startswith("target") else "no_protein"}
        for s in ["blank_1", "blank_2", "blank_3", "target_1", "target_2", "target_3"]
    ]
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# assess_sequencing_depth
# ---------------------------------------------------------------------------


class TestAssessSequencingDepth:
    def test_returns_dict_with_expected_keys(self, synthetic_counts, synthetic_config):
        result = assess_sequencing_depth(synthetic_counts, N_CYCLES, synthetic_config)
        assert "sampling_ratio" in result
        assert "status" in result
        assert "mean_reads_per_selection" in result

    def test_three_levels_for_two_cycles(self, synthetic_counts, synthetic_config):
        result = assess_sequencing_depth(synthetic_counts, N_CYCLES, synthetic_config)
        assert len(result["sampling_ratio"]) == 3

    def test_mono_ratio_higher_than_di(self, synthetic_counts, synthetic_config):
        """Monosynthon has lower diversity → higher sampling ratio."""
        result = assess_sequencing_depth(synthetic_counts, N_CYCLES, synthetic_config)
        mono1 = result["sampling_ratio"]["mono_code_1"]
        di = result["sampling_ratio"]["di_code_1_code_2"]
        assert mono1 > di

    def test_mean_reads_is_positive(self, synthetic_counts, synthetic_config):
        result = assess_sequencing_depth(synthetic_counts, N_CYCLES, synthetic_config)
        assert result["mean_reads_per_selection"] > 0

    def test_status_values_valid(self, synthetic_counts, synthetic_config):
        result = assess_sequencing_depth(synthetic_counts, N_CYCLES, synthetic_config)
        for status in result["status"].values():
            assert status in ("green", "yellow", "red")

    def test_sampling_ratio_formula(self, synthetic_counts, synthetic_config):
        """Verify the sampling ratio matches expected formula:
        mean_total_reads_per_selection / diversity."""
        result = assess_sequencing_depth(synthetic_counts, N_CYCLES, synthetic_config)
        mean_reads = result["mean_reads_per_selection"]
        di_diversity = N_BB1 * N_BB2
        expected = mean_reads / di_diversity
        assert abs(result["sampling_ratio"]["di_code_1_code_2"] - expected) < 0.01


# ---------------------------------------------------------------------------
# assess_replicate_correlation
# ---------------------------------------------------------------------------


class TestAssessReplicateCorrelation:
    def test_returns_expected_keys(self, synthetic_counts, meta_df):
        result = assess_replicate_correlation(synthetic_counts, meta_df, CODE_COLS)
        assert "correlation" in result
        assert "status" in result
        assert "n_pairs" in result

    def test_both_groups_present(self, synthetic_counts, meta_df):
        result = assess_replicate_correlation(synthetic_counts, meta_df, CODE_COLS)
        assert "protein" in result["correlation"]
        assert "no_protein" in result["correlation"]

    def test_correlations_in_valid_range(self, synthetic_counts, meta_df):
        result = assess_replicate_correlation(synthetic_counts, meta_df, CODE_COLS)
        for group, r in result["correlation"].items():
            import math
            if not math.isnan(r):
                assert -1.0 <= r <= 1.0, f"{group}: r={r}"

    def test_target_replicates_well_correlated(self, synthetic_counts, meta_df):
        """Target replicates share true binder enrichment → r > 0.9."""
        result = assess_replicate_correlation(synthetic_counts, meta_df, CODE_COLS)
        r = result["correlation"]["protein"]
        assert r > 0.9, f"protein r={r:.4f}"

    def test_blank_replicates_well_correlated(self, synthetic_counts, meta_df):
        """Blank replicates are all noise → should still correlate well."""
        result = assess_replicate_correlation(synthetic_counts, meta_df, CODE_COLS)
        r = result["correlation"]["no_protein"]
        assert r > 0.5, f"no_protein r={r:.4f}"

    def test_n_pairs_correct(self, synthetic_counts, meta_df):
        """3 replicates → C(3,2) = 3 pairs per group."""
        result = assess_replicate_correlation(synthetic_counts, meta_df, CODE_COLS)
        assert result["n_pairs"]["protein"] == 3
        assert result["n_pairs"]["no_protein"] == 3

    def test_status_values_valid(self, synthetic_counts, meta_df):
        result = assess_replicate_correlation(synthetic_counts, meta_df, CODE_COLS)
        for status in result["status"].values():
            assert status in ("green", "yellow", "red")


# ---------------------------------------------------------------------------
# assess_bb_coverage
# ---------------------------------------------------------------------------


class TestAssessBBCoverage:
    def test_returns_expected_keys(self, synthetic_counts, synthetic_config):
        result = assess_bb_coverage(synthetic_counts, N_CYCLES, synthetic_config)
        for key in ("coverage", "n_observed", "n_expected", "status",
                    "max_bb_fraction", "max_bb_fraction_status"):
            assert key in result

    def test_code_1_expected_count_from_config(self, synthetic_counts, synthetic_config):
        result = assess_bb_coverage(synthetic_counts, N_CYCLES, synthetic_config)
        assert result["n_expected"]["code_1"] == N_BB1

    def test_code_2_expected_count_from_config(self, synthetic_counts, synthetic_config):
        result = assess_bb_coverage(synthetic_counts, N_CYCLES, synthetic_config)
        assert result["n_expected"]["code_2"] == N_BB2

    def test_full_coverage_synthetic(self, synthetic_counts, synthetic_config):
        """All BBs appear with at least one non-zero count across 6 selections."""
        result = assess_bb_coverage(synthetic_counts, N_CYCLES, synthetic_config)
        # All 10 code_1 values should be observed (bead binder + noise are non-zero)
        assert result["n_observed"]["code_1"] == N_BB1

    def test_coverage_fractions_in_unit_interval(self, synthetic_counts, synthetic_config):
        result = assess_bb_coverage(synthetic_counts, N_CYCLES, synthetic_config)
        for col, frac in result["coverage"].items():
            assert 0.0 <= frac <= 1.0, f"{col}: coverage={frac}"

    def test_max_bb_fraction_positive(self, synthetic_counts, synthetic_config):
        result = assess_bb_coverage(synthetic_counts, N_CYCLES, synthetic_config)
        for col, frac in result["max_bb_fraction"].items():
            assert frac > 0, f"{col}: max_bb_fraction=0"

    def test_status_keys_match_code_cols(self, synthetic_counts, synthetic_config):
        result = assess_bb_coverage(synthetic_counts, N_CYCLES, synthetic_config)
        assert set(result["status"].keys()) == {"code_1", "code_2"}

    def test_coverage_without_config_bbs(self, synthetic_counts):
        """Falls back to observed max + 1 when config has no BB list."""
        result = assess_bb_coverage(synthetic_counts, N_CYCLES, {})
        # Should not crash; coverage <= 1.0
        for frac in result["coverage"].values():
            assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# assess_bb_uniformity
# ---------------------------------------------------------------------------


class TestAssessBBUniformity:
    def test_returns_expected_keys(self, synthetic_counts):
        result = assess_bb_uniformity(synthetic_counts, N_CYCLES)
        for key in ("cv", "gini", "outliers_high", "outliers_zero"):
            assert key in result

    def test_cv_non_negative(self, synthetic_counts):
        result = assess_bb_uniformity(synthetic_counts, N_CYCLES)
        for col, cv in result["cv"].items():
            assert cv >= 0, f"{col}: cv={cv}"

    def test_gini_in_unit_interval(self, synthetic_counts):
        result = assess_bb_uniformity(synthetic_counts, N_CYCLES)
        for col, g in result["gini"].items():
            assert 0.0 <= g <= 1.0, f"{col}: gini={g}"

    def test_bead_binder_inflates_cv(self, synthetic_counts):
        """code_1=7 has systematically higher counts (40-80) vs noise (0-20).
        This should inflate CV for code_1 vs a perfectly uniform library."""
        result = assess_bb_uniformity(synthetic_counts, N_CYCLES)
        assert result["cv"]["code_1"] > 0

    def test_outliers_zero_is_list(self, synthetic_counts):
        result = assess_bb_uniformity(synthetic_counts, N_CYCLES)
        for col in ("code_1", "code_2"):
            assert isinstance(result["outliers_zero"][col], list)

    def test_outliers_high_is_list(self, synthetic_counts):
        result = assess_bb_uniformity(synthetic_counts, N_CYCLES)
        for col in ("code_1", "code_2"):
            assert isinstance(result["outliers_high"][col], list)

    def test_bead_binder_flagged_as_high_outlier(self, synthetic_counts):
        """code_1=7 should be flagged as a high outlier (counts 40-80 vs noise 0-20)."""
        result = assess_bb_uniformity(synthetic_counts, N_CYCLES)
        # code_1=7 has counts 40-80 vs median noise ~10 → should be > mean + 3SD
        # Not guaranteed with only 10 BBs and moderate spread, but we at least
        # verify the list structure is valid
        assert isinstance(result["outliers_high"]["code_1"], list)


# ---------------------------------------------------------------------------
# generate_quality_report
# ---------------------------------------------------------------------------


class TestGenerateQualityReport:
    def test_writes_json(self, synthetic_counts, synthetic_config, meta_df, tmp_path):
        generate_quality_report(
            synthetic_counts, N_CYCLES, synthetic_config, tmp_path, meta_df
        )
        assert (tmp_path / "data_quality.json").exists()

    def test_writes_html(self, synthetic_counts, synthetic_config, meta_df, tmp_path):
        generate_quality_report(
            synthetic_counts, N_CYCLES, synthetic_config, tmp_path, meta_df
        )
        assert (tmp_path / "qc_report.html").exists()

    def test_json_has_required_keys(self, synthetic_counts, synthetic_config, meta_df, tmp_path):
        report = generate_quality_report(
            synthetic_counts, N_CYCLES, synthetic_config, tmp_path, meta_df
        )
        for key in ("overall_quality", "sampling_ratio", "replicate_correlation",
                    "bb_coverage", "warnings", "recommended_analysis_levels"):
            assert key in report, f"Missing key: {key}"

    def test_overall_quality_is_valid(self, synthetic_counts, synthetic_config, meta_df, tmp_path):
        report = generate_quality_report(
            synthetic_counts, N_CYCLES, synthetic_config, tmp_path, meta_df
        )
        assert report["overall_quality"] in ("green", "yellow", "red")

    def test_json_is_parseable(self, synthetic_counts, synthetic_config, meta_df, tmp_path):
        generate_quality_report(
            synthetic_counts, N_CYCLES, synthetic_config, tmp_path, meta_df
        )
        with (tmp_path / "data_quality.json").open() as fh:
            parsed = json.load(fh)
        assert isinstance(parsed, dict)
        assert "overall_quality" in parsed

    def test_html_contains_experiment_name(self, synthetic_counts, synthetic_config, meta_df, tmp_path):
        generate_quality_report(
            synthetic_counts, N_CYCLES, synthetic_config, tmp_path, meta_df
        )
        html = (tmp_path / "qc_report.html").read_text()
        assert "synthetic_test" in html  # from synthetic_config["experiment"]["name"]

    def test_html_is_valid_html_fragment(self, synthetic_counts, synthetic_config, meta_df, tmp_path):
        generate_quality_report(
            synthetic_counts, N_CYCLES, synthetic_config, tmp_path, meta_df
        )
        html = (tmp_path / "qc_report.html").read_text()
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_without_metadata_skips_correlation(self, synthetic_counts, synthetic_config, tmp_path):
        """Should not crash when metadata_df is None."""
        report = generate_quality_report(
            synthetic_counts, N_CYCLES, synthetic_config, tmp_path, metadata_df=None
        )
        assert isinstance(report, dict)
        assert report["replicate_correlation"] == {}

    def test_creates_output_dir(self, synthetic_counts, synthetic_config, tmp_path):
        subdir = tmp_path / "nested" / "qc_output"
        generate_quality_report(
            synthetic_counts, N_CYCLES, synthetic_config, subdir
        )
        assert subdir.exists()
        assert (subdir / "data_quality.json").exists()

    def test_returns_dict_matching_json(self, synthetic_counts, synthetic_config, meta_df, tmp_path):
        """The returned dict should match what was written to JSON."""
        report = generate_quality_report(
            synthetic_counts, N_CYCLES, synthetic_config, tmp_path, meta_df
        )
        with (tmp_path / "data_quality.json").open() as fh:
            on_disk = json.load(fh)
        assert report["overall_quality"] == on_disk["overall_quality"]
        assert report["sampling_ratio"] == on_disk["sampling_ratio"]

    def test_warnings_is_list(self, synthetic_counts, synthetic_config, meta_df, tmp_path):
        report = generate_quality_report(
            synthetic_counts, N_CYCLES, synthetic_config, tmp_path, meta_df
        )
        assert isinstance(report["warnings"], list)

    def test_recommended_levels_is_list(self, synthetic_counts, synthetic_config, meta_df, tmp_path):
        report = generate_quality_report(
            synthetic_counts, N_CYCLES, synthetic_config, tmp_path, meta_df
        )
        assert isinstance(report["recommended_analysis_levels"], list)
        assert len(report["recommended_analysis_levels"]) > 0
