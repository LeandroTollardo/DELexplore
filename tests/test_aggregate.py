"""Tests for analyse/aggregate.py using the synthetic_counts fixture.

Synthetic library: 2-cycle, 10 BBs × 8 BBs = 80 compounds, 6 selections.
Columns in synthetic_counts: selection, code_1, code_2, count, id
"""

import polars as pl
import pytest

from delexplore.analyse.aggregate import (
    aggregate_all_levels,
    aggregate_to_level,
    get_all_levels,
    get_diversity,
    get_level_name,
)

N_BB1 = 10
N_BB2 = 8
N_COMPOUNDS = N_BB1 * N_BB2  # 80
N_SELECTIONS = 6


# ---------------------------------------------------------------------------
# get_all_levels
# ---------------------------------------------------------------------------


class TestGetAllLevels:
    def test_two_cycles_returns_three_levels(self):
        levels = get_all_levels(2)
        assert len(levels) == 3

    def test_two_cycles_correct_levels(self):
        levels = get_all_levels(2)
        assert ("code_1",) in levels
        assert ("code_2",) in levels
        assert ("code_1", "code_2") in levels

    def test_three_cycles_returns_seven_levels(self):
        levels = get_all_levels(3)
        assert len(levels) == 7  # C(3,1)=3 + C(3,2)=3 + C(3,3)=1

    def test_three_cycles_mono_levels(self):
        levels = get_all_levels(3)
        assert ("code_1",) in levels
        assert ("code_2",) in levels
        assert ("code_3",) in levels

    def test_three_cycles_di_levels(self):
        levels = get_all_levels(3)
        assert ("code_1", "code_2") in levels
        assert ("code_1", "code_3") in levels
        assert ("code_2", "code_3") in levels

    def test_three_cycles_tri_level(self):
        levels = get_all_levels(3)
        assert ("code_1", "code_2", "code_3") in levels

    def test_one_cycle_returns_one_level(self):
        levels = get_all_levels(1)
        assert levels == [("code_1",)]

    def test_levels_ordered_mono_first(self):
        levels = get_all_levels(3)
        # All 1-column tuples come before 2-column tuples, etc.
        sizes = [len(lvl) for lvl in levels]
        assert sizes == sorted(sizes)

    def test_invalid_n_cycles_raises(self):
        with pytest.raises(ValueError):
            get_all_levels(0)

    def test_four_cycles_returns_fifteen_levels(self):
        # C(4,1)+C(4,2)+C(4,3)+C(4,4) = 4+6+4+1 = 15
        assert len(get_all_levels(4)) == 15


# ---------------------------------------------------------------------------
# get_level_name
# ---------------------------------------------------------------------------


class TestGetLevelName:
    def test_monosynthon_code_1(self):
        assert get_level_name(("code_1",)) == "mono_code_1"

    def test_monosynthon_code_2(self):
        assert get_level_name(("code_2",)) == "mono_code_2"

    def test_disynthon(self):
        assert get_level_name(("code_1", "code_2")) == "di_code_1_code_2"

    def test_trisynthon(self):
        assert get_level_name(("code_1", "code_2", "code_3")) == "tri_code_1_code_2_code_3"

    def test_four_synthon_uses_numeric_prefix(self):
        name = get_level_name(("code_1", "code_2", "code_3", "code_4"))
        assert name.startswith("4syn_")

    def test_accepts_list_as_well_as_tuple(self):
        assert get_level_name(["code_1", "code_2"]) == "di_code_1_code_2"


# ---------------------------------------------------------------------------
# get_diversity
# ---------------------------------------------------------------------------


class TestGetDiversity:
    def test_monosynthon_code_1_diversity(self, synthetic_counts):
        div = get_diversity(synthetic_counts, ("code_1",))
        assert div == N_BB1  # 10 distinct values for code_1

    def test_monosynthon_code_2_diversity(self, synthetic_counts):
        div = get_diversity(synthetic_counts, ("code_2",))
        assert div == N_BB2  # 8 distinct values for code_2

    def test_disynthon_diversity(self, synthetic_counts):
        div = get_diversity(synthetic_counts, ("code_1", "code_2"))
        assert div == N_COMPOUNDS  # 80 unique (code_1, code_2) pairs


# ---------------------------------------------------------------------------
# aggregate_to_level
# ---------------------------------------------------------------------------


class TestAggregateToLevel:
    def test_returns_dataframe(self, synthetic_counts):
        result = aggregate_to_level(synthetic_counts, ("code_1",))
        assert isinstance(result, pl.DataFrame)

    def test_monosynthon_columns(self, synthetic_counts):
        result = aggregate_to_level(synthetic_counts, ("code_1",))
        assert list(result.columns) == ["selection", "code_1", "count"]

    def test_disynthon_columns(self, synthetic_counts):
        result = aggregate_to_level(synthetic_counts, ("code_1", "code_2"))
        assert list(result.columns) == ["selection", "code_1", "code_2", "count"]

    def test_monosynthon_row_count(self, synthetic_counts):
        # 6 selections × 10 code_1 values = 60 rows
        result = aggregate_to_level(synthetic_counts, ("code_1",))
        assert len(result) == N_SELECTIONS * N_BB1

    def test_monosynthon_code2_row_count(self, synthetic_counts):
        # 6 selections × 8 code_2 values = 48 rows
        result = aggregate_to_level(synthetic_counts, ("code_2",))
        assert len(result) == N_SELECTIONS * N_BB2

    def test_disynthon_row_count(self, synthetic_counts):
        # 6 selections × 80 compounds = 480 rows (no further aggregation)
        result = aggregate_to_level(synthetic_counts, ("code_1", "code_2"))
        assert len(result) == N_SELECTIONS * N_COMPOUNDS

    def test_id_column_absent(self, synthetic_counts):
        result = aggregate_to_level(synthetic_counts, ("code_1",))
        assert "id" not in result.columns

    def test_sorted_descending_by_count_within_selection(self, synthetic_counts):
        result = aggregate_to_level(synthetic_counts, ("code_1",))
        for sel in result["selection"].unique().to_list():
            counts = result.filter(pl.col("selection") == sel)["count"].to_list()
            assert counts == sorted(counts, reverse=True)

    def test_monosynthon_count_equals_sum_across_other_position(self, synthetic_counts):
        """Monosynthon count for (sel, code_1=X) must equal sum of all
        (sel, code_1=X, code_2=*) compound counts."""
        mono = aggregate_to_level(synthetic_counts, ("code_1",))
        # Check a specific (selection, code_1) pair
        sel = "target_1"
        c1 = 2
        expected = (
            synthetic_counts
            .filter((pl.col("selection") == sel) & (pl.col("code_1") == c1))["count"]
            .sum()
        )
        actual = (
            mono
            .filter((pl.col("selection") == sel) & (pl.col("code_1") == c1))["count"]
            .item()
        )
        assert actual == expected

    def test_monosynthon_total_count_equals_original_total(self, synthetic_counts):
        """Sum of all monosynthon counts (per position) = sum of all compound counts.
        Each compound contributes once to mono_code_1 and once to mono_code_2."""
        original_total = synthetic_counts["count"].sum()
        mono1 = aggregate_to_level(synthetic_counts, ("code_1",))
        mono2 = aggregate_to_level(synthetic_counts, ("code_2",))
        assert mono1["count"].sum() == original_total
        assert mono2["count"].sum() == original_total

    def test_disynthon_passthrough_matches_original(self, synthetic_counts):
        """For a 2-cycle library, the disynthon level (code_1, code_2) is
        equivalent to the original compound data — counts must be identical."""
        agg = aggregate_to_level(synthetic_counts, ("code_1", "code_2"))

        # Join back on (selection, code_1, code_2) and compare counts
        original = synthetic_counts.select(["selection", "code_1", "code_2", "count"])
        joined = agg.join(
            original.rename({"count": "orig_count"}),
            on=["selection", "code_1", "code_2"],
        )
        assert (joined["count"] == joined["orig_count"]).all()

    def test_no_null_counts(self, synthetic_counts):
        result = aggregate_to_level(synthetic_counts, ("code_1",))
        assert result["count"].null_count() == 0


# ---------------------------------------------------------------------------
# aggregate_all_levels
# ---------------------------------------------------------------------------


class TestAggregateAllLevels:
    def test_returns_dict(self, synthetic_counts):
        result = aggregate_all_levels(synthetic_counts, n_cycles=2)
        assert isinstance(result, dict)

    def test_two_cycles_returns_three_entries(self, synthetic_counts):
        result = aggregate_all_levels(synthetic_counts, n_cycles=2)
        assert len(result) == 3

    def test_keys_are_correct_level_names(self, synthetic_counts):
        result = aggregate_all_levels(synthetic_counts, n_cycles=2)
        assert set(result.keys()) == {
            "mono_code_1",
            "mono_code_2",
            "di_code_1_code_2",
        }

    def test_all_values_are_dataframes(self, synthetic_counts):
        result = aggregate_all_levels(synthetic_counts, n_cycles=2)
        for name, df in result.items():
            assert isinstance(df, pl.DataFrame), f"{name} is not a DataFrame"

    def test_mono_dfs_have_correct_shape(self, synthetic_counts):
        result = aggregate_all_levels(synthetic_counts, n_cycles=2)
        assert len(result["mono_code_1"]) == N_SELECTIONS * N_BB1
        assert len(result["mono_code_2"]) == N_SELECTIONS * N_BB2

    def test_di_df_has_correct_shape(self, synthetic_counts):
        result = aggregate_all_levels(synthetic_counts, n_cycles=2)
        assert len(result["di_code_1_code_2"]) == N_SELECTIONS * N_COMPOUNDS
