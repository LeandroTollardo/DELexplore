"""Tests that verify the synthetic fixtures are internally consistent
and that the written-on-disk files are readable by io/readers.py."""

from pathlib import Path

import polars as pl
import pytest

from delexplore.io.readers import load_experiment, read_counts, read_config

SELECTIONS = ["blank_1", "blank_2", "blank_3", "target_1", "target_2", "target_3"]
N_BB1, N_BB2 = 10, 8
N_COMPOUNDS = N_BB1 * N_BB2  # 80


# ---------------------------------------------------------------------------
# synthetic_counts
# ---------------------------------------------------------------------------


class TestSyntheticCounts:
    def test_is_dataframe(self, synthetic_counts):
        assert isinstance(synthetic_counts, pl.DataFrame)

    def test_columns(self, synthetic_counts):
        assert list(synthetic_counts.columns) == [
            "selection", "code_1", "code_2", "count", "id"
        ]

    def test_total_rows(self, synthetic_counts):
        # 6 selections × 80 compounds each
        assert len(synthetic_counts) == len(SELECTIONS) * N_COMPOUNDS

    def test_all_selections_present(self, synthetic_counts):
        found = set(synthetic_counts["selection"].unique().to_list())
        assert found == set(SELECTIONS)

    def test_each_selection_has_80_rows(self, synthetic_counts):
        for sel in SELECTIONS:
            n = len(synthetic_counts.filter(pl.col("selection") == sel))
            assert n == N_COMPOUNDS, f"{sel} has {n} rows, expected {N_COMPOUNDS}"

    def test_sorted_desc_within_selection(self, synthetic_counts):
        for sel in SELECTIONS:
            counts = (
                synthetic_counts.filter(pl.col("selection") == sel)["count"].to_list()
            )
            assert counts == sorted(counts, reverse=True), (
                f"{sel} is not sorted descending by count"
            )

    def test_id_format(self, synthetic_counts):
        # id must be "{code_1}_{code_2}"
        for row in synthetic_counts.sample(10, seed=42).iter_rows(named=True):
            expected_id = f"{row['code_1']}_{row['code_2']}"
            assert row["id"] == expected_id

    def test_true_binders_enriched_in_target(self, synthetic_counts):
        for c1, c2 in [(2, 3), (5, 6)]:
            target_counts = synthetic_counts.filter(
                (pl.col("selection").str.starts_with("target"))
                & (pl.col("code_1") == c1)
                & (pl.col("code_2") == c2)
            )["count"]
            blank_counts = synthetic_counts.filter(
                (pl.col("selection").str.starts_with("blank"))
                & (pl.col("code_1") == c1)
                & (pl.col("code_2") == c2)
            )["count"]
            assert target_counts.min() >= 100, (
                f"True binder ({c1},{c2}) target min={target_counts.min()}"
            )
            assert blank_counts.max() <= 10, (
                f"True binder ({c1},{c2}) blank max={blank_counts.max()}"
            )

    def test_bead_binders_present_in_blank(self, synthetic_counts):
        bead_blank = synthetic_counts.filter(
            (pl.col("code_1") == 7)
            & (pl.col("selection").str.starts_with("blank"))
        )["count"]
        assert bead_blank.min() >= 40, f"Bead binder blank min={bead_blank.min()}"

    def test_noise_counts_within_range(self, synthetic_counts):
        noise = synthetic_counts.filter(
            ~(
                ((pl.col("code_1") == 2) & (pl.col("code_2") == 3))
                | ((pl.col("code_1") == 5) & (pl.col("code_2") == 6))
                | (pl.col("code_1") == 7)
            )
        )["count"]
        assert noise.max() <= 20, f"Noise max count is {noise.max()}"
        assert noise.min() >= 0


# ---------------------------------------------------------------------------
# synthetic_config
# ---------------------------------------------------------------------------


class TestSyntheticConfig:
    def test_top_level_keys(self, synthetic_config):
        assert "experiment" in synthetic_config
        assert "selections" in synthetic_config
        assert "library" in synthetic_config

    def test_six_selections(self, synthetic_config):
        assert len(synthetic_config["selections"]) == 6

    def test_blank_selections_have_no_protein_group(self, synthetic_config):
        for name in ["blank_1", "blank_2", "blank_3"]:
            assert synthetic_config["selections"][name]["group"] == "no_protein"

    def test_target_selections_have_protein_group(self, synthetic_config):
        for name in ["target_1", "target_2", "target_3"]:
            assert synthetic_config["selections"][name]["group"] == "protein"

    def test_beads_vary_between_groups(self, synthetic_config):
        blank_beads = {
            synthetic_config["selections"][s]["beads"]
            for s in ["blank_1", "blank_2", "blank_3"]
        }
        target_beads = {
            synthetic_config["selections"][s]["beads"]
            for s in ["target_1", "target_2", "target_3"]
        }
        assert blank_beads == {"HisPURE Beads"}
        assert target_beads == {"Dynabeads SA C1"}

    def test_library_has_b0_and_b1(self, synthetic_config):
        lib = synthetic_config["library"]
        assert "B0" in lib
        assert "B1" in lib

    def test_b0_has_10_building_blocks(self, synthetic_config):
        assert len(synthetic_config["library"]["B0"]) == N_BB1

    def test_b1_has_8_building_blocks(self, synthetic_config):
        assert len(synthetic_config["library"]["B1"]) == N_BB2

    def test_building_blocks_have_smiles(self, synthetic_config):
        for bb in synthetic_config["library"]["B0"]:
            assert "smiles" in bb and bb["smiles"]
        for bb in synthetic_config["library"]["B1"]:
            assert "smiles" in bb and bb["smiles"]

    def test_building_block_indices_sequential(self, synthetic_config):
        b0_indices = [bb["index"] for bb in synthetic_config["library"]["B0"]]
        b1_indices = [bb["index"] for bb in synthetic_config["library"]["B1"]]
        assert b0_indices == list(range(N_BB1))
        assert b1_indices == list(range(N_BB2))


# ---------------------------------------------------------------------------
# selection_metadata
# ---------------------------------------------------------------------------


class TestSelectionMetadata:
    def test_is_dataframe(self, selection_metadata):
        assert isinstance(selection_metadata, pl.DataFrame)

    def test_six_rows(self, selection_metadata):
        assert len(selection_metadata) == 6

    def test_required_columns(self, selection_metadata):
        for col in ("selection_name", "target", "group", "date", "operator", "beads"):
            assert col in selection_metadata.columns

    def test_blank_targets_are_null(self, selection_metadata):
        blank_rows = selection_metadata.filter(
            pl.col("selection_name").str.starts_with("blank")
        )
        assert blank_rows["target"].null_count() == 3

    def test_target_rows_have_protein_a(self, selection_metadata):
        target_rows = selection_metadata.filter(
            pl.col("selection_name").str.starts_with("target")
        )
        assert (target_rows["target"] == "ProteinA").all()

    def test_beads_column_populated(self, selection_metadata):
        assert selection_metadata["beads"].null_count() == 0

    def test_protocol_column_populated(self, selection_metadata):
        assert selection_metadata["protocol"].null_count() == 0


# ---------------------------------------------------------------------------
# On-disk files: written by synthetic_data_dir fixture
# ---------------------------------------------------------------------------


class TestSyntheticOnDisk:
    def test_config_yaml_exists(self, synthetic_data_dir):
        assert (synthetic_data_dir / "config.yaml").exists()

    def test_all_selection_dirs_exist(self, synthetic_data_dir):
        for sel in SELECTIONS:
            assert (synthetic_data_dir / sel / "counts.txt").exists(), (
                f"Missing {sel}/counts.txt"
            )

    def test_readable_by_read_counts(self, synthetic_data_dir):
        path = synthetic_data_dir / "blank_1" / "counts.txt"
        df = read_counts(path)
        assert isinstance(df, pl.DataFrame)
        assert df.n_cycles == 2  # type: ignore[attr-defined]
        assert len(df) == N_COMPOUNDS

    def test_read_counts_columns_match_real_format(self, synthetic_data_dir):
        path = synthetic_data_dir / "target_1" / "counts.txt"
        df = read_counts(path)
        assert list(df.columns) == ["code_1", "code_2", "count", "id"]

    def test_readable_by_read_config(self, synthetic_data_dir):
        config = read_config(synthetic_data_dir / "config.yaml")
        assert "selections" in config
        assert len(config["selections"]) == 6

    def test_load_experiment_finds_all_selections(self, synthetic_data_dir):
        result = load_experiment(synthetic_data_dir / "config.yaml")
        assert result["n_cycles"] == 2
        found = set(result["counts"]["selection"].unique().to_list())
        assert found == set(SELECTIONS)

    def test_load_experiment_total_rows(self, synthetic_data_dir):
        result = load_experiment(synthetic_data_dir / "config.yaml")
        assert len(result["counts"]) == len(SELECTIONS) * N_COMPOUNDS
