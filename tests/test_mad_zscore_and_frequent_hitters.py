"""Tests for MAD z-score (zscore.py) and flag_frequent_hitters (classify.py)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from delexplore.analyse.zscore import calculate_mad_zscore
from delexplore.analyse.classify import flag_frequent_hitters
from delexplore.analyse.multilevel import run_multilevel_enrichment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_counts(n: int = 100, base: int = 10) -> np.ndarray:
    return np.full(n, base, dtype=float)


def _enrichment_df(
    code_1_vals: list[int],
    code_2_vals: list[int],
    scores: list[float],
    score_col: str = "zscore",
) -> pl.DataFrame:
    return pl.DataFrame({
        "code_1": code_1_vals,
        "code_2": code_2_vals,
        score_col: scores,
    })


# ---------------------------------------------------------------------------
# 1. calculate_mad_zscore
# ---------------------------------------------------------------------------


class TestCalculateMadZscore:
    # --- basic properties ---

    def test_returns_array_same_shape(self) -> None:
        counts = np.array([10, 20, 30, 5, 5], dtype=float)
        result = calculate_mad_zscore(counts, total_reads=100, diversity=5)
        assert result.shape == counts.shape

    def test_uniform_counts_all_zero(self) -> None:
        """All counts equal → MAD = 0 → returns zeros."""
        counts = _uniform_counts(50, base=10)
        result = calculate_mad_zscore(counts, total_reads=500, diversity=50)
        assert np.all(result == 0.0)

    def test_single_element_returns_zero(self) -> None:
        result = calculate_mad_zscore(np.array([42.0]), total_reads=42, diversity=1)
        assert result[0] == 0.0

    def test_dtype_is_float(self) -> None:
        counts = np.array([1, 2, 3, 4, 5])
        result = calculate_mad_zscore(counts, total_reads=15, diversity=5)
        assert result.dtype == float

    # --- ranking preservation ---

    def test_same_ranking_as_regular_for_uniform_noise(self) -> None:
        """MAD z-score must rank features the same as regular z-score when
        noise is symmetric around the mean (both are monotone in count)."""
        rng = np.random.default_rng(0)
        base = np.full(200, 10, dtype=float)
        # Add one clear enriched compound
        base[0] = 200
        counts = base + rng.integers(0, 3, size=200)

        mad_z = calculate_mad_zscore(counts, total_reads=int(counts.sum()), diversity=200)
        # Enriched compound should rank highest
        assert np.argmax(mad_z) == 0

    def test_enriched_compound_gets_highest_score(self) -> None:
        # Use distinct counts to avoid the MAD=0 pathology (>50% at median)
        counts = np.array([5, 6, 7, 8, 100], dtype=float)
        result = calculate_mad_zscore(counts, total_reads=int(counts.sum()), diversity=5)
        assert np.argmax(result) == 4

    def test_depleted_compound_gets_lowest_score(self) -> None:
        counts = np.array([0, 10, 10, 10, 10], dtype=float)
        result = calculate_mad_zscore(counts, total_reads=40, diversity=5)
        assert np.argmin(result) == 0

    # --- robustness to outliers ---

    def test_robust_to_single_large_outlier(self) -> None:
        """With one massive outlier, remaining rank ordering should be preserved."""
        counts = np.array([5, 10, 15, 20, 25, 1_000_000], dtype=float)
        total = int(counts.sum())
        mad_z = calculate_mad_zscore(counts, total_reads=total, diversity=len(counts))
        # Among the first 5 elements (non-outlier), score increases with count
        non_outlier = mad_z[:5]
        assert np.all(np.diff(non_outlier) > 0), (
            "Scores among non-outlier elements should increase with count"
        )

    def test_mad_and_regular_zscore_use_different_denominators(self) -> None:
        """Regular z-score uses the fixed theoretical sqrt(p_i*(1-p_i)) denominator.
        MAD z-score uses the empirical spread. Confirm they differ numerically."""
        from delexplore.analyse.zscore import calculate_zscore

        # Use distinct counts so MAD > 0; range wide enough to differ from regular
        counts = np.arange(1, 21, dtype=float)  # 1..20, all distinct
        total = int(counts.sum())
        diversity = len(counts)

        regular = calculate_zscore(counts, total, diversity)
        mad = calculate_mad_zscore(counts, total, diversity)

        # Both should rank the highest-count compound highest
        assert np.argmax(regular) == np.argmax(mad) == len(counts) - 1
        # But the actual z-score values differ (different denominators)
        assert not np.allclose(regular, mad)

    # --- MAD=0 edge cases ---

    def test_mad_zero_all_identical_returns_zeros(self) -> None:
        counts = np.full(20, 50, dtype=float)
        result = calculate_mad_zscore(counts, total_reads=1000, diversity=20)
        assert np.all(result == 0.0)

    def test_mad_zero_two_groups_same_count(self) -> None:
        """Two values that are both at the median → MAD=0."""
        counts = np.array([10, 10], dtype=float)
        result = calculate_mad_zscore(counts, total_reads=20, diversity=2)
        assert np.all(result == 0.0)

    # --- input validation ---

    def test_raises_on_zero_total_reads(self) -> None:
        with pytest.raises(ValueError, match="total_reads"):
            calculate_mad_zscore(np.array([1.0, 2.0]), total_reads=0, diversity=2)

    def test_raises_on_negative_total_reads(self) -> None:
        with pytest.raises(ValueError, match="total_reads"):
            calculate_mad_zscore(np.array([1.0, 2.0]), total_reads=-1, diversity=2)

    def test_raises_on_zero_diversity(self) -> None:
        with pytest.raises(ValueError, match="diversity"):
            calculate_mad_zscore(np.array([1.0, 2.0]), total_reads=10, diversity=0)

    # --- numerical correctness ---

    def test_known_value(self) -> None:
        """Manual calculation for a 3-element array.

        counts  = [10, 20, 30], total = 60
        p_i     = [1/6, 1/3, 1/2]
        median  = 1/3
        |p_i - median| = [1/6, 0, 1/6]
        median(|.|) = 1/12   (median of [0, 1/6, 1/6] = 1/6...
                               sorted: [0, 1/6, 1/6], median = 1/6)
        Actually: sorted abs deviations = [0, 1/6, 1/6]; median = 1/6
        MAD = 1.4286 × 1/6 ≈ 0.2381
        z_mad[0] = (1/6 - 1/3) / 0.2381 ≈ -0.7
        z_mad[1] = 0
        z_mad[2] = (1/2 - 1/3) / 0.2381 ≈ 0.7
        """
        counts = np.array([10.0, 20.0, 30.0])
        total = 60
        result = calculate_mad_zscore(counts, total_reads=total, diversity=3)

        p = counts / total  # [1/6, 1/3, 1/2]
        med = np.median(p)
        mad = 1.4286 * np.median(np.abs(p - med))
        expected = (p - med) / mad

        np.testing.assert_allclose(result, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# 2. run_multilevel_enrichment with mad_zscore method
# ---------------------------------------------------------------------------


class TestMadZscoreInMultilevel:
    """Integration: confirm mad_zscore flows through run_multilevel_enrichment."""

    @pytest.fixture()
    def simple_counts_df(self) -> pl.DataFrame:
        rng = np.random.default_rng(1)
        n_compounds = 50
        code_1 = [i % 5 for i in range(n_compounds)]
        code_2 = [i % 10 for i in range(n_compounds)]
        base_count = rng.integers(5, 20, size=n_compounds).tolist()

        rows = []
        for sel in ["post_1", "post_2", "ctrl_1"]:
            for c1, c2, bc in zip(code_1, code_2, base_count):
                rows.append({"selection": sel, "code_1": c1, "code_2": c2, "count": bc})
        return pl.DataFrame(rows)

    def test_mad_zscore_column_present(self, simple_counts_df: pl.DataFrame) -> None:
        result = run_multilevel_enrichment(
            simple_counts_df,
            n_cycles=2,
            code_cols=["code_1", "code_2"],
            post_selections=["post_1", "post_2"],
            control_selections=["ctrl_1"],
            methods=("mad_zscore",),
        )
        for level_df in result.values():
            assert "mad_zscore" in level_df.columns

    def test_mad_zscore_is_numeric(self, simple_counts_df: pl.DataFrame) -> None:
        result = run_multilevel_enrichment(
            simple_counts_df,
            n_cycles=2,
            code_cols=["code_1", "code_2"],
            post_selections=["post_1", "post_2"],
            control_selections=["ctrl_1"],
            methods=("mad_zscore",),
        )
        for level_df in result.values():
            col = level_df["mad_zscore"].to_numpy()
            assert np.isfinite(col).all()

    def test_mad_and_zscore_both_present_when_both_requested(
        self, simple_counts_df: pl.DataFrame
    ) -> None:
        result = run_multilevel_enrichment(
            simple_counts_df,
            n_cycles=2,
            code_cols=["code_1", "code_2"],
            post_selections=["post_1", "post_2"],
            control_selections=["ctrl_1"],
            methods=("zscore", "mad_zscore"),
        )
        for level_df in result.values():
            assert "zscore" in level_df.columns
            assert "mad_zscore" in level_df.columns

    def test_invalid_method_raises(self, simple_counts_df: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown enrichment method"):
            run_multilevel_enrichment(
                simple_counts_df,
                n_cycles=2,
                code_cols=["code_1", "code_2"],
                post_selections=["post_1", "post_2"],
                control_selections=["ctrl_1"],
                methods=("bad_method",),
            )


# ---------------------------------------------------------------------------
# 3. flag_frequent_hitters
# ---------------------------------------------------------------------------


class TestFlagFrequentHitters:
    """Tests for classify.flag_frequent_hitters."""

    @pytest.fixture()
    def three_target_data(self) -> dict[str, pl.DataFrame]:
        """3 targets, 6 compounds.

        Compound (0,0): enriched in all 3 targets  → frequent hitter (n=3)
        Compound (0,1): enriched in 2 targets       → frequent hitter (n=2) with min_targets=2
        Compound (0,2): enriched in 1 target only   → not frequent hitter (n=1)
        Compounds (1,0), (1,1), (1,2): not enriched in any target (n=0)
        """
        def _df(c1: list[int], c2: list[int], z: list[float]) -> pl.DataFrame:
            return pl.DataFrame({"code_1": c1, "code_2": c2, "zscore": z})

        target_a = _df(
            [0, 0, 0, 1, 1, 1],
            [0, 1, 2, 0, 1, 2],
            [3.0, 2.5, 2.0, 0.1, 0.0, -0.1],  # (0,0), (0,1), (0,2) enriched
        )
        target_b = _df(
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [3.0, 2.0, 0.0, 0.0],  # (0,0), (0,1) enriched
        )
        target_c = _df(
            [0, 1],
            [0, 0],
            [2.5, 0.0],  # (0,0) enriched only
        )
        return {"target_a": target_a, "target_b": target_b, "target_c": target_c}

    # --- output shape and columns ---

    def test_returns_dataframe(
        self, three_target_data: dict[str, pl.DataFrame]
    ) -> None:
        result = flag_frequent_hitters(
            three_target_data, code_cols=["code_1", "code_2"]
        )
        assert isinstance(result, pl.DataFrame)

    def test_output_columns(
        self, three_target_data: dict[str, pl.DataFrame]
    ) -> None:
        result = flag_frequent_hitters(
            three_target_data, code_cols=["code_1", "code_2"]
        )
        assert "n_targets_enriched" in result.columns
        assert "is_frequent_hitter" in result.columns
        assert "code_1" in result.columns
        assert "code_2" in result.columns

    def test_all_compounds_included(
        self, three_target_data: dict[str, pl.DataFrame]
    ) -> None:
        result = flag_frequent_hitters(
            three_target_data, code_cols=["code_1", "code_2"]
        )
        # 6 unique compounds across all targets
        assert len(result) == 6

    # --- correct frequent hitter counts ---

    def test_enriched_in_all_three_targets(
        self, three_target_data: dict[str, pl.DataFrame]
    ) -> None:
        result = flag_frequent_hitters(
            three_target_data, code_cols=["code_1", "code_2"],
            threshold_value=1.0, min_targets=2,
        )
        row = result.filter(
            (pl.col("code_1") == 0) & (pl.col("code_2") == 0)
        )
        assert row["n_targets_enriched"][0] == 3
        assert row["is_frequent_hitter"][0] is True

    def test_enriched_in_two_targets(
        self, three_target_data: dict[str, pl.DataFrame]
    ) -> None:
        result = flag_frequent_hitters(
            three_target_data, code_cols=["code_1", "code_2"],
            threshold_value=1.0, min_targets=2,
        )
        row = result.filter(
            (pl.col("code_1") == 0) & (pl.col("code_2") == 1)
        )
        assert row["n_targets_enriched"][0] == 2
        assert row["is_frequent_hitter"][0] is True

    def test_enriched_in_one_target_not_flagged_with_min2(
        self, three_target_data: dict[str, pl.DataFrame]
    ) -> None:
        result = flag_frequent_hitters(
            three_target_data, code_cols=["code_1", "code_2"],
            threshold_value=1.0, min_targets=2,
        )
        row = result.filter(
            (pl.col("code_1") == 0) & (pl.col("code_2") == 2)
        )
        assert row["n_targets_enriched"][0] == 1
        assert row["is_frequent_hitter"][0] is False

    def test_not_enriched_in_any_target(
        self, three_target_data: dict[str, pl.DataFrame]
    ) -> None:
        result = flag_frequent_hitters(
            three_target_data, code_cols=["code_1", "code_2"],
            threshold_value=1.0, min_targets=2,
        )
        not_enriched = result.filter(pl.col("code_1") == 1)
        assert (not_enriched["n_targets_enriched"] == 0).all()
        assert (~not_enriched["is_frequent_hitter"]).all()

    # --- min_targets parameter ---

    def test_min_targets_1_flags_any_enriched(
        self, three_target_data: dict[str, pl.DataFrame]
    ) -> None:
        result = flag_frequent_hitters(
            three_target_data, code_cols=["code_1", "code_2"],
            threshold_value=1.0, min_targets=1,
        )
        # (0,0), (0,1), (0,2) are all enriched in at least 1 target
        flagged = result.filter(pl.col("is_frequent_hitter"))
        assert len(flagged) == 3

    def test_min_targets_3_only_flags_enriched_in_all(
        self, three_target_data: dict[str, pl.DataFrame]
    ) -> None:
        result = flag_frequent_hitters(
            three_target_data, code_cols=["code_1", "code_2"],
            threshold_value=1.0, min_targets=3,
        )
        flagged = result.filter(pl.col("is_frequent_hitter"))
        assert len(flagged) == 1
        assert flagged["code_1"][0] == 0
        assert flagged["code_2"][0] == 0

    def test_min_targets_above_n_targets_flags_none(
        self, three_target_data: dict[str, pl.DataFrame]
    ) -> None:
        result = flag_frequent_hitters(
            three_target_data, code_cols=["code_1", "code_2"],
            threshold_value=1.0, min_targets=10,
        )
        assert result["is_frequent_hitter"].sum() == 0

    # --- threshold_value ---

    def test_high_threshold_reduces_flagged_compounds(
        self, three_target_data: dict[str, pl.DataFrame]
    ) -> None:
        result_low = flag_frequent_hitters(
            three_target_data, code_cols=["code_1", "code_2"],
            threshold_value=1.0, min_targets=2,
        )
        result_high = flag_frequent_hitters(
            three_target_data, code_cols=["code_1", "code_2"],
            threshold_value=3.0, min_targets=2,
        )
        n_low = int(result_low["is_frequent_hitter"].sum())
        n_high = int(result_high["is_frequent_hitter"].sum())
        assert n_high <= n_low

    # --- missing compounds treated as not enriched ---

    def test_missing_compound_counted_as_not_enriched(self) -> None:
        """Compound present in only one target should get count=1."""
        target_a = pl.DataFrame({
            "code_1": [0, 1],
            "code_2": [0, 0],
            "zscore": [5.0, 5.0],
        })
        target_b = pl.DataFrame({
            "code_1": [0],  # compound (1,0) absent from this target
            "code_2": [0],
            "zscore": [5.0],
        })
        result = flag_frequent_hitters(
            {"a": target_a, "b": target_b},
            code_cols=["code_1", "code_2"],
            threshold_value=1.0, min_targets=2,
        )
        row_1_0 = result.filter(
            (pl.col("code_1") == 1) & (pl.col("code_2") == 0)
        )
        assert row_1_0["n_targets_enriched"][0] == 1
        assert row_1_0["is_frequent_hitter"][0] is False

    # --- alternative score column ---

    def test_works_with_fold_enrichment_col(self) -> None:
        data = {
            "target_a": pl.DataFrame({
                "code_1": [0, 1], "code_2": [0, 0],
                "fold_enrichment": [5.0, 1.0],
            }),
            "target_b": pl.DataFrame({
                "code_1": [0, 1], "code_2": [0, 0],
                "fold_enrichment": [4.0, 1.5],
            }),
        }
        result = flag_frequent_hitters(
            data, code_cols=["code_1", "code_2"],
            threshold_col="fold_enrichment", threshold_value=2.0, min_targets=2,
        )
        row = result.filter(
            (pl.col("code_1") == 0) & (pl.col("code_2") == 0)
        )
        assert row["is_frequent_hitter"][0] is True

    # --- error handling ---

    def test_empty_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            flag_frequent_hitters({}, code_cols=["code_1", "code_2"])

    def test_empty_code_cols_raises(self) -> None:
        df = pl.DataFrame({"code_1": [0], "zscore": [1.0]})
        with pytest.raises(ValueError, match="code_cols must not be empty"):
            flag_frequent_hitters({"t": df}, code_cols=[])

    def test_min_targets_zero_raises(self) -> None:
        df = pl.DataFrame({"code_1": [0], "zscore": [1.0]})
        with pytest.raises(ValueError, match="min_targets"):
            flag_frequent_hitters({"t": df}, code_cols=["code_1"], min_targets=0)

    def test_missing_threshold_col_raises(self) -> None:
        df = pl.DataFrame({"code_1": [0], "other_col": [1.0]})
        with pytest.raises(ValueError, match="zscore"):
            flag_frequent_hitters({"t": df}, code_cols=["code_1"])

    def test_missing_code_col_raises(self) -> None:
        df = pl.DataFrame({"code_1": [0], "zscore": [1.0]})
        with pytest.raises(ValueError, match="code_cols"):
            flag_frequent_hitters(
                {"t": df}, code_cols=["code_1", "code_2"]
            )

    # --- single target ---

    def test_single_target_with_min_targets_1(self) -> None:
        df = pl.DataFrame({
            "code_1": [0, 1, 2],
            "zscore": [3.0, 0.0, -1.0],
        })
        result = flag_frequent_hitters(
            {"only_target": df}, code_cols=["code_1"],
            threshold_value=1.0, min_targets=1,
        )
        assert result.filter(pl.col("code_1") == 0)["is_frequent_hitter"][0] is True
        assert result.filter(pl.col("code_1") == 1)["is_frequent_hitter"][0] is False
