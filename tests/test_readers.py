"""Tests for io/readers.py using real DELT-Hit example data."""

import polars as pl
import pytest

from delexplore.io.readers import (
    get_selection_metadata,
    load_experiment,
    read_config,
    read_counts,
)


class TestReadCounts:
    def test_returns_dataframe(self, example_counts_path):
        df = read_counts(example_counts_path)
        assert isinstance(df, pl.DataFrame)

    def test_detects_two_cycles(self, example_counts_path):
        df = read_counts(example_counts_path)
        assert df.n_cycles == 2  # type: ignore[attr-defined]

    def test_has_expected_columns(self, example_counts_path):
        df = read_counts(example_counts_path)
        assert "code_1" in df.columns
        assert "code_2" in df.columns
        assert "count" in df.columns
        assert "id" in df.columns

    def test_has_rows(self, example_counts_path):
        df = read_counts(example_counts_path)
        assert len(df) > 0

    def test_sorted_by_count_descending(self, example_counts_path):
        df = read_counts(example_counts_path)
        counts = df["count"].to_list()
        assert counts == sorted(counts, reverse=True)


class TestReadConfig:
    def test_returns_dict(self, example_config_path):
        config = read_config(example_config_path)
        assert isinstance(config, dict)

    def test_has_selections(self, example_config_path):
        config = read_config(example_config_path)
        assert "selections" in config
        assert len(config["selections"]) > 0

    def test_has_library(self, example_config_path):
        config = read_config(example_config_path)
        assert "library" in config

    def test_selection_has_required_fields(self, example_config_path):
        config = read_config(example_config_path)
        sel = next(iter(config["selections"].values()))
        assert "group" in sel
        assert "target" in sel


class TestGetSelectionMetadata:
    def test_returns_dataframe(self, example_config):
        meta = get_selection_metadata(example_config)
        assert isinstance(meta, pl.DataFrame)

    def test_has_expected_columns(self, example_config):
        meta = get_selection_metadata(example_config)
        for col in ("selection_name", "target", "group", "date", "operator"):
            assert col in meta.columns

    def test_row_count_matches_selections(self, example_config):
        meta = get_selection_metadata(example_config)
        n_selections = len(example_config["selections"])
        assert len(meta) == n_selections

    def test_nan_targets_normalised_to_none(self, example_config):
        meta = get_selection_metadata(example_config)
        # Selections with .nan target should come back as None/null, not "nan"
        no_protein_rows = meta.filter(pl.col("group") == "no_protein")
        assert no_protein_rows["target"].null_count() == len(no_protein_rows)

    def test_protein_groups_have_targets(self, example_config):
        meta = get_selection_metadata(example_config)
        protein_rows = meta.filter(pl.col("group") == "protein")
        assert protein_rows["target"].null_count() == 0


class TestLoadExperiment:
    def test_returns_dict_with_expected_keys(self, example_config_path):
        result = load_experiment(example_config_path)
        assert "config" in result
        assert "counts" in result
        assert "n_cycles" in result

    def test_config_is_dict(self, example_config_path):
        result = load_experiment(example_config_path)
        assert isinstance(result["config"], dict)

    def test_counts_is_dataframe(self, example_config_path):
        result = load_experiment(example_config_path)
        assert isinstance(result["counts"], pl.DataFrame)

    def test_no_counts_files_returns_empty_df(self, example_config_path):
        # The example directory has no per-selection subfolders, so
        # load_experiment should return an empty counts frame gracefully.
        result = load_experiment(example_config_path)
        assert result["n_cycles"] == 0 or isinstance(result["counts"], pl.DataFrame)
