"""Tests for analyse/rank.py.

Uses the session-scoped synthetic fixture (2-cycle, 10×8 = 80 compounds):
  True binders:  (code_1=2, code_2=3) and (code_1=5, code_2=6) — high zscore
  Bead binders:  code_1 == 7 — moderate, uniform across conditions
  Noise:         everything else

Inline DataFrames are used for unit-level tests to keep them fast and
independent of the multilevel pipeline.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from delexplore.analyse.multilevel import run_multilevel_enrichment
from delexplore.analyse.rank import (
    compute_composite_rank,
    compute_method_agreement,
    compute_support_score,
    export_hit_list,
)

CODE_COLS = ["code_1", "code_2"]
N_CYCLES = 2
POST_SELS = ["target_1", "target_2", "target_3"]
CTRL_SELS = ["blank_1", "blank_2", "blank_3"]
TRUE_BINDERS = [(2, 3), (5, 6)]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ml_result(synthetic_counts):
    return run_multilevel_enrichment(
        synthetic_counts,
        n_cycles=N_CYCLES,
        code_cols=CODE_COLS,
        post_selections=POST_SELS,
        control_selections=CTRL_SELS,
    )


@pytest.fixture(scope="module")
def ranked(ml_result):
    return compute_composite_rank(ml_result, CODE_COLS)


# ---------------------------------------------------------------------------
# Inline helper
# ---------------------------------------------------------------------------


def _score_df(scores: list[float], col: str = "zscore") -> pl.DataFrame:
    return pl.DataFrame(
        {"code_1": list(range(len(scores))), "code_2": [0] * len(scores), col: scores},
        schema={"code_1": pl.Int64, "code_2": pl.Int64, col: pl.Float64},
    )


# ---------------------------------------------------------------------------
# compute_method_agreement
# ---------------------------------------------------------------------------


class TestComputeMethodAgreement:
    def test_returns_numpy_array(self):
        df = _score_df([3.0, 1.0, 2.0])
        result = compute_method_agreement(df, ["zscore"])
        assert isinstance(result, np.ndarray)

    def test_length_matches_dataframe(self):
        df = _score_df([3.0, 1.0, 2.0])
        result = compute_method_agreement(df, ["zscore"])
        assert len(result) == 3

    def test_all_positive(self):
        df = _score_df([3.0, 1.0, 2.0])
        result = compute_method_agreement(df, ["zscore"])
        assert all(r > 0 for r in result)

    def test_best_compound_gets_lowest_score(self):
        """Highest zscore should get rank 1 → lowest agreement score."""
        df = _score_df([5.0, 1.0, 2.0])
        result = compute_method_agreement(df, ["zscore"])
        assert result[0] < result[1]
        assert result[0] < result[2]

    def test_worst_compound_gets_highest_score(self):
        df = _score_df([5.0, 0.1, 2.0])
        result = compute_method_agreement(df, ["zscore"])
        assert result[1] > result[0]
        assert result[1] > result[2]

    def test_single_method_equals_rank(self):
        """With one method, geometric mean of ranks = rank itself."""
        df = _score_df([5.0, 3.0, 1.0])
        result = compute_method_agreement(df, ["zscore"])
        # Ranks should be [1.0, 2.0, 3.0]
        assert abs(result[0] - 1.0) < 1e-9
        assert abs(result[1] - 2.0) < 1e-9
        assert abs(result[2] - 3.0) < 1e-9

    def test_two_methods_geometric_mean(self):
        """Verify geometric mean formula: exp(mean(log(ranks)))."""
        df = pl.DataFrame(
            {
                "code_1": [0, 1, 2],
                "code_2": [0, 0, 0],
                "zscore": [3.0, 1.0, 2.0],       # ranks: [1, 3, 2]
                "poisson_ml_enrichment": [2.0, 3.0, 1.0],  # ranks: [2, 1, 3]
            }
        )
        result = compute_method_agreement(df, ["zscore", "poisson_ml_enrichment"])
        # compound 0: geo_mean(1, 2) = sqrt(2) ≈ 1.414
        assert abs(result[0] - math.sqrt(2)) < 1e-6
        # compound 1: geo_mean(3, 1) = sqrt(3) ≈ 1.732
        assert abs(result[1] - math.sqrt(3)) < 1e-6
        # compound 2: geo_mean(2, 3) = sqrt(6) ≈ 2.449
        assert abs(result[2] - math.sqrt(6)) < 1e-6

    def test_tied_scores_get_average_rank(self):
        df = _score_df([2.0, 2.0, 1.0])
        result = compute_method_agreement(df, ["zscore"])
        # Tied at rank 1.5; lowest compound is rank 3
        assert abs(result[0] - 1.5) < 1e-9
        assert abs(result[1] - 1.5) < 1e-9
        assert abs(result[2] - 3.0) < 1e-9

    def test_empty_method_cols_raises(self):
        df = _score_df([1.0, 2.0])
        with pytest.raises(ValueError, match="must not be empty"):
            compute_method_agreement(df, [])

    def test_missing_column_raises(self):
        df = _score_df([1.0])
        with pytest.raises(ValueError, match="Columns not found"):
            compute_method_agreement(df, ["nonexistent_col"])

    def test_empty_dataframe_returns_empty_array(self):
        df = pl.DataFrame({"code_1": [], "zscore": []}, schema={"code_1": pl.Int64, "zscore": pl.Float64})
        result = compute_method_agreement(df, ["zscore"])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# compute_support_score
# ---------------------------------------------------------------------------


class TestComputeSupportScore:
    def test_returns_numpy_array(self, ml_result):
        compound_df = ml_result["di_code_1_code_2"]
        result = compute_support_score(ml_result, compound_df, CODE_COLS)
        assert isinstance(result, np.ndarray)

    def test_length_matches_compound_df(self, ml_result):
        compound_df = ml_result["di_code_1_code_2"]
        result = compute_support_score(ml_result, compound_df, CODE_COLS)
        assert len(result) == len(compound_df)

    def test_scores_non_negative(self, ml_result):
        compound_df = ml_result["di_code_1_code_2"]
        result = compute_support_score(ml_result, compound_df, CODE_COLS)
        assert np.all(result >= 0)

    def test_scores_within_max(self, ml_result):
        """For 2-cycle: max support = 2 (one mono each, no sub-disynthons)."""
        compound_df = ml_result["di_code_1_code_2"]
        result = compute_support_score(ml_result, compound_df, CODE_COLS)
        assert np.all(result <= 2.0)

    def test_true_binders_have_positive_support_with_fold_enrichment(self, ml_result):
        """True binders' constituent BBs are detected via fold_enrichment >= 2.0.

        The monosynthon level has fold_enrichment from the Poisson method.
        True binder BBs (code_1=2 with code_2=3) are genuinely enriched and
        should have fold_enrichment > 2.0 at the monosynthon level.

        NOTE: z_n >= 1.0 would FAIL here — at monosynthon level each BB
        aggregates many noise partners, so even the strongest enriched BB only
        reaches z_n ≈ 0.6 in this 10-BB synthetic library.  See
        docs/references/corrected_formulas.md §SUPPORT SCORE THRESHOLD WARNING.
        """
        compound_df = ml_result["di_code_1_code_2"]
        # Default: fold_enrichment >= 2.0
        result = compute_support_score(ml_result, compound_df, CODE_COLS)
        idx = compound_df.with_row_index().filter(
            (pl.col("code_1") == 2) & (pl.col("code_2") == 3)
        )["index"][0]
        assert result[idx] > 0, (
            "True binder (2,3) has support=0 with fold_enrichment >= 2.0 default"
        )

    def test_default_threshold_is_fold_enrichment(self, ml_result):
        """Confirm the default threshold_col is fold_enrichment, not zscore."""
        import inspect
        sig = inspect.signature(compute_support_score)
        assert sig.parameters["threshold_col"].default == "fold_enrichment"
        assert sig.parameters["threshold_value"].default == 2.0

    def test_auto_mode_uses_fold_enrichment_when_present(self, ml_result):
        """threshold_col='auto' should resolve to fold_enrichment >= 2.0."""
        compound_df = ml_result["di_code_1_code_2"]
        result_auto = compute_support_score(
            ml_result, compound_df, CODE_COLS, threshold_col="auto"
        )
        result_explicit = compute_support_score(
            ml_result, compound_df, CODE_COLS,
            threshold_col="fold_enrichment", threshold_value=2.0,
        )
        np.testing.assert_array_equal(result_auto, result_explicit)

    def test_auto_mode_percentile_fallback(self):
        """When fold_enrichment is absent, 'auto' falls back to top-10% zscore."""
        # Build a level DF with only zscore (no fold_enrichment)
        mono1_df = pl.DataFrame(
            {"code_1": list(range(10)), "zscore": [float(i) for i in range(10)]}
        )
        # zscore values 0..9; top-10% (90th percentile) ≈ 8.1 → only code_1=9 enriched
        compound_df = pl.DataFrame(
            {"code_1": list(range(10)), "code_2": [0] * 10, "zscore": [0.0] * 10}
        )
        ml = {
            "mono_code_1": mono1_df,
            "di_code_1_code_2": compound_df,
        }
        result = compute_support_score(
            ml, compound_df, ["code_1", "code_2"], threshold_col="auto"
        )
        # Only code_1=9 should be in the top 10% → only rows with code_1=9 get +1
        idx9 = 9
        assert result[idx9] == 1.0
        # code_1=0..8 should have support=0 (all below 90th percentile)
        for i in range(9):
            assert result[i] == 0.0, f"code_1={i} unexpectedly has support={result[i]}"

    def test_missing_threshold_col_falls_back_to_fold_enrichment(self, ml_result):
        """If the specified threshold_col is absent, fall back to fold_enrichment."""
        compound_df = ml_result["di_code_1_code_2"]
        # threshold_col="zscore" but the mono DFs will also have fold_enrichment
        # → adaptive fallback uses fold_enrichment >= 2.0
        result_fallback = compute_support_score(
            ml_result, compound_df, CODE_COLS, threshold_col="nonexistent_col"
        )
        result_explicit = compute_support_score(
            ml_result, compound_df, CODE_COLS,
            threshold_col="fold_enrichment", threshold_value=2.0,
        )
        np.testing.assert_array_equal(result_fallback, result_explicit)

    def test_missing_level_is_skipped_gracefully(self, ml_result):
        """If a level is absent from multilevel_results, no crash."""
        partial = {k: v for k, v in ml_result.items() if k == "di_code_1_code_2"}
        compound_df = ml_result["di_code_1_code_2"]
        result = compute_support_score(partial, compound_df, CODE_COLS)
        # No mono levels → all support = 0
        assert np.all(result == 0.0)

    def test_no_usable_column_skips_level(self):
        """If a level DF has no score columns at all, it's skipped silently."""
        mono1_df = pl.DataFrame({"code_1": [1], "some_other_col": [99.0]})
        compound_df = pl.DataFrame({"code_1": [1], "code_2": [0]})
        ml = {
            "mono_code_1": mono1_df,
            "di_code_1_code_2": compound_df,
        }
        result = compute_support_score(
            ml, compound_df, ["code_1", "code_2"], threshold_col="auto"
        )
        assert result[0] == 0.0

    def test_empty_code_cols_raises(self, ml_result):
        compound_df = ml_result["di_code_1_code_2"]
        with pytest.raises(ValueError, match="must not be empty"):
            compute_support_score(ml_result, compound_df, [])

    def test_3cycle_support_uses_weight_2_for_disynthons(self):
        """For 3-cycle, disynthon enrichment contributes +2."""
        # Build a minimal 3-cycle multilevel result using fold_enrichment
        di_df = pl.DataFrame(
            {"code_1": [1], "code_2": [1], "fold_enrichment": [5.0]},  # enriched
        )
        mono1_df = pl.DataFrame({"code_1": [1], "fold_enrichment": [0.5]})  # not
        mono2_df = pl.DataFrame({"code_2": [1], "fold_enrichment": [0.5]})  # not
        mono3_df = pl.DataFrame({"code_3": [1], "fold_enrichment": [0.5]})  # not
        compound_df = pl.DataFrame({"code_1": [1], "code_2": [1], "code_3": [1]})

        ml = {
            "mono_code_1": mono1_df,
            "mono_code_2": mono2_df,
            "mono_code_3": mono3_df,
            "di_code_1_code_2": di_df,
            # di_code_1_code_3 and di_code_2_code_3 absent → skipped
        }
        result = compute_support_score(
            ml, compound_df, ["code_1", "code_2", "code_3"]
        )
        # Only di_code_1_code_2 contributes: weight 2 × enriched = 2
        assert result[0] == 2.0


# ---------------------------------------------------------------------------
# compute_composite_rank — structure
# ---------------------------------------------------------------------------


class TestComputeCompositeRankStructure:
    def test_returns_dataframe(self, ranked):
        assert isinstance(ranked, pl.DataFrame)

    def test_has_rank_column(self, ranked):
        assert "rank" in ranked.columns

    def test_has_agreement_score_column(self, ranked):
        assert "agreement_score" in ranked.columns

    def test_has_support_score_column(self, ranked):
        assert "support_score" in ranked.columns

    def test_has_composite_score_column(self, ranked):
        assert "composite_score" in ranked.columns

    def test_has_property_penalty_column(self, ranked):
        assert "property_penalty" in ranked.columns

    def test_row_count_equals_full_compound_level(self, ml_result, ranked):
        di_df = ml_result["di_code_1_code_2"]
        assert len(ranked) == len(di_df)

    def test_rank_is_1_based(self, ranked):
        assert ranked["rank"].min() == 1
        assert ranked["rank"].max() == len(ranked)

    def test_rank_is_contiguous(self, ranked):
        ranks = sorted(ranked["rank"].to_list())
        assert ranks == list(range(1, len(ranked) + 1))

    def test_sorted_by_composite_score(self, ranked):
        scores = ranked["composite_score"].to_list()
        assert scores == sorted(scores)

    def test_property_penalty_defaults_to_1(self, ranked):
        assert all(p == 1.0 for p in ranked["property_penalty"].to_list())

    def test_composite_score_positive(self, ranked):
        assert all(s > 0 for s in ranked["composite_score"].to_list())


# ---------------------------------------------------------------------------
# compute_composite_rank — correctness
# ---------------------------------------------------------------------------


class TestComputeCompositeRankCorrectness:
    def test_true_binders_rank_in_top_10(self, ranked):
        """True binders should rank well above noise."""
        for c1, c2 in TRUE_BINDERS:
            row = ranked.filter((pl.col("code_1") == c1) & (pl.col("code_2") == c2))
            assert len(row) == 1
            assert row["rank"][0] <= 10, (
                f"True binder ({c1},{c2}) ranked {row['rank'][0]}, expected ≤ 10"
            )

    def test_composite_formula_is_correct(self, ranked):
        """Verify: composite_score = agreement × 1/(1+support) × penalty."""
        for row in ranked.iter_rows(named=True):
            expected = (
                row["agreement_score"]
                * (1.0 / (1.0 + row["support_score"]))
                * row["property_penalty"]
            )
            assert abs(row["composite_score"] - expected) < 1e-9

    def test_higher_support_gives_lower_or_equal_composite(self):
        """All else equal, higher support → lower composite score → better rank."""
        # Build a controlled 2-compound result using fold_enrichment for support
        compound_df = pl.DataFrame(
            {
                "code_1": [0, 1],
                "code_2": [0, 0],
                "zscore": [3.0, 3.0],  # identical target enrichment
                "poisson_ml_enrichment": [3.0, 3.0],
                "fold_enrichment": [3.0, 3.0],
            }
        )
        # code_1=0: fold_enrichment=5.0 (> 2.0 → enriched); code_1=1: 0.5 (not)
        mono1_df = pl.DataFrame({"code_1": [0, 1], "fold_enrichment": [5.0, 0.5]})
        mono2_df = pl.DataFrame({"code_2": [0], "fold_enrichment": [5.0]})
        ml = {
            "di_code_1_code_2": compound_df,
            "mono_code_1": mono1_df,
            "mono_code_2": mono2_df,
        }
        result = compute_composite_rank(ml, ["code_1", "code_2"])
        r0 = result.filter(pl.col("code_1") == 0)["composite_score"][0]
        r1 = result.filter(pl.col("code_1") == 1)["composite_score"][0]
        # code_1=0: monosynthons for code_1=0 and code_2=0 both enriched → more support
        assert r0 <= r1


# ---------------------------------------------------------------------------
# compute_composite_rank — with properties
# ---------------------------------------------------------------------------


class TestComputeCompositeRankProperties:
    def test_lipinski_fail_increases_composite_score(self, ml_result):
        compound_df = ml_result["di_code_1_code_2"]
        # Mark all compounds as failing Lipinski
        props = compound_df.select(CODE_COLS).with_columns(
            pl.lit(False).alias("lipinski_pass")
        )
        ranked_with = compute_composite_rank(ml_result, CODE_COLS, properties_df=props)
        ranked_base = compute_composite_rank(ml_result, CODE_COLS)

        # With penalty=2.0 (Lipinski fail), all scores should double
        for row_w, row_b in zip(
            ranked_with.sort(CODE_COLS).iter_rows(named=True),
            ranked_base.sort(CODE_COLS).iter_rows(named=True),
        ):
            assert abs(row_w["composite_score"] - row_b["composite_score"] * 2.0) < 1e-9

    def test_explicit_penalty_col_applied(self, ml_result):
        compound_df = ml_result["di_code_1_code_2"]
        props = compound_df.select(CODE_COLS).with_columns(
            pl.lit(3.0).alias("property_penalty")
        )
        ranked_with = compute_composite_rank(
            ml_result, CODE_COLS, properties_df=props, property_penalty_col="property_penalty"
        )
        ranked_base = compute_composite_rank(ml_result, CODE_COLS)

        for row_w, row_b in zip(
            ranked_with.sort(CODE_COLS).iter_rows(named=True),
            ranked_base.sort(CODE_COLS).iter_rows(named=True),
        ):
            assert abs(row_w["composite_score"] - row_b["composite_score"] * 3.0) < 1e-9


# ---------------------------------------------------------------------------
# compute_composite_rank — error handling
# ---------------------------------------------------------------------------


class TestComputeCompositeRankErrors:
    def test_empty_code_cols_raises(self, ml_result):
        with pytest.raises(ValueError, match="must not be empty"):
            compute_composite_rank(ml_result, [])

    def test_missing_full_level_raises(self, ml_result):
        partial = {k: v for k, v in ml_result.items() if "mono" in k}
        with pytest.raises(ValueError, match="not found in multilevel_results"):
            compute_composite_rank(partial, CODE_COLS)

    def test_no_valid_method_cols_raises(self, ml_result):
        with pytest.raises(ValueError, match="None of the requested method_cols"):
            compute_composite_rank(
                ml_result, CODE_COLS, method_cols=("nonexistent_col",)
            )

    def test_partial_method_cols_uses_available(self, ml_result):
        """If some method_cols are missing, available ones are used (no crash)."""
        result = compute_composite_rank(
            ml_result, CODE_COLS,
            method_cols=("zscore", "completely_fake_col"),
        )
        assert isinstance(result, pl.DataFrame)
        assert "rank" in result.columns


# ---------------------------------------------------------------------------
# export_hit_list
# ---------------------------------------------------------------------------


class TestExportHitList:
    def test_returns_dataframe(self, ranked):
        result = export_hit_list(ranked, top_n=10)
        assert isinstance(result, pl.DataFrame)

    def test_top_n_rows_returned(self, ranked):
        result = export_hit_list(ranked, top_n=10)
        assert len(result) == 10

    def test_top_n_exceeds_length_returns_all(self, ranked):
        result = export_hit_list(ranked, top_n=10_000)
        assert len(result) == len(ranked)

    def test_sorted_by_rank_ascending(self, ranked):
        result = export_hit_list(ranked, top_n=20)
        ranks = result["rank"].to_list()
        assert ranks == sorted(ranks)

    def test_first_row_is_rank_1(self, ranked):
        result = export_hit_list(ranked, top_n=5)
        assert result["rank"][0] == 1

    def test_writes_csv(self, ranked, tmp_path):
        out = tmp_path / "hits.csv"
        result = export_hit_list(ranked, top_n=10, output_path=out)
        assert out.exists()
        assert len(result) == 10

    def test_csv_is_parseable(self, ranked, tmp_path):
        out = tmp_path / "hits.csv"
        export_hit_list(ranked, top_n=5, output_path=out)
        loaded = pl.read_csv(out)
        assert "rank" in loaded.columns
        assert len(loaded) == 5

    def test_creates_parent_dirs(self, ranked, tmp_path):
        out = tmp_path / "subdir" / "nested" / "hits.csv"
        export_hit_list(ranked, top_n=3, output_path=out)
        assert out.exists()

    def test_no_output_path_does_not_create_file(self, ranked, tmp_path):
        result = export_hit_list(ranked, top_n=5)
        assert isinstance(result, pl.DataFrame)
        assert not list(tmp_path.glob("*.csv"))

    def test_missing_rank_col_raises(self):
        df = pl.DataFrame({"code_1": [1], "zscore": [2.0]})
        with pytest.raises(ValueError, match="'rank' column"):
            export_hit_list(df, top_n=1)

    def test_output_path_as_string(self, ranked, tmp_path):
        out = str(tmp_path / "hits_str.csv")
        export_hit_list(ranked, top_n=3, output_path=out)
        assert Path(out).exists()
