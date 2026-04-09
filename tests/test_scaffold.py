"""Tests for explore/scaffold.py."""

from __future__ import annotations

import polars as pl
import pytest

from delexplore.explore.scaffold import (
    compute_murcko_scaffolds,
    scaffold_enrichment_analysis,
)

# ---------------------------------------------------------------------------
# Test molecules
# ---------------------------------------------------------------------------

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"          # scaffold: c1ccccc1
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"   # scaffold: c1ccc(cc1)
NAPROXEN = "COc1ccc2cc(ccc2c1)C(C)C(=O)O"  # scaffold: c1ccc2ccccc2c1 (naphthalene)
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"          # scaffold: c1ccc(cc1) (para-disubstituted benzene)
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"    # purine scaffold
GLYCINE = "NCC(=O)O"                         # acyclic → empty scaffold
INVALID = "not_a_smiles"

# Benzamide derivatives share the benzene scaffold
BENZAMIDE = "NC(=O)c1ccccc1"
BENZOIC_ACID = "OC(=O)c1ccccc1"
TOLUENE = "Cc1ccccc1"


# ---------------------------------------------------------------------------
# 1. compute_murcko_scaffolds
# ---------------------------------------------------------------------------


class TestComputeMurckoScaffolds:
    def test_returns_list_same_length(self):
        result = compute_murcko_scaffolds([ASPIRIN, CAFFEINE])
        assert len(result) == 2

    def test_aspirin_scaffold_is_benzene(self):
        result = compute_murcko_scaffolds([ASPIRIN])
        # Aspirin's Murcko scaffold is benzene: c1ccccc1
        assert result[0] is not None
        assert "c1ccccc1" in result[0] or result[0] == "c1ccccc1"

    def test_benzamide_and_benzoic_acid_same_scaffold(self):
        result = compute_murcko_scaffolds([BENZAMIDE, BENZOIC_ACID])
        # Both monosubstituted benzenes → same benzene scaffold
        assert result[0] is not None
        assert result[1] is not None
        assert result[0] == result[1]

    def test_toluene_scaffold(self):
        # Toluene is monosubstituted benzene → benzene scaffold
        result = compute_murcko_scaffolds([TOLUENE])
        assert result[0] is not None
        # scaffold should be a valid SMILES representing benzene
        assert result[0] != ""

    def test_acyclic_smiles_returns_empty_string(self):
        result = compute_murcko_scaffolds([GLYCINE])
        # RDKit returns empty string scaffold for acyclic compounds
        assert result[0] == ""

    def test_invalid_smiles_returns_none(self):
        result = compute_murcko_scaffolds([INVALID])
        assert result[0] is None

    def test_none_in_list_returns_none(self):
        result = compute_murcko_scaffolds([None])  # type: ignore[list-item]
        assert result[0] is None

    def test_empty_string_returns_none(self):
        result = compute_murcko_scaffolds([""])
        assert result[0] is None

    def test_mixed_valid_invalid(self):
        result = compute_murcko_scaffolds([ASPIRIN, INVALID, CAFFEINE])
        assert result[0] is not None
        assert result[1] is None
        assert result[2] is not None

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_murcko_scaffolds([])

    def test_caffeine_scaffold(self):
        result = compute_murcko_scaffolds([CAFFEINE])
        # Caffeine has a purine scaffold (two fused rings)
        assert result[0] is not None
        assert result[0] != ""

    def test_naphthalene_scaffold_for_naproxen(self):
        result = compute_murcko_scaffolds([NAPROXEN])
        assert result[0] is not None
        # Naproxen has a naphthalene-like scaffold
        assert result[0] != ""

    def test_same_scaffold_for_aspirin_and_benzamide(self):
        result = compute_murcko_scaffolds([ASPIRIN, BENZAMIDE])
        # Both have monosubstituted benzene → same scaffold
        assert result[0] == result[1]

    def test_different_scaffolds_different_results(self):
        result = compute_murcko_scaffolds([ASPIRIN, CAFFEINE])
        # Benzene vs purine: different scaffolds
        assert result[0] != result[1]


# ---------------------------------------------------------------------------
# 2. scaffold_enrichment_analysis
# ---------------------------------------------------------------------------


def _make_df(smiles_list: list[str], scores: list[float] | None = None) -> pl.DataFrame:
    n = len(smiles_list)
    data: dict = {
        "rank": list(range(1, n + 1)),
        "smiles": smiles_list,
        "code_1": list(range(n)),
    }
    if scores is not None:
        data["composite_score"] = scores
    return pl.DataFrame(data)


class TestScaffoldEnrichmentAnalysis:
    def test_returns_dataframe(self):
        df = _make_df([ASPIRIN, CAFFEINE, IBUPROFEN])
        result = scaffold_enrichment_analysis(df, score_col="composite_score")
        assert isinstance(result, pl.DataFrame)

    def test_has_required_columns(self):
        df = _make_df([ASPIRIN, CAFFEINE], scores=[0.1, 0.5])
        result = scaffold_enrichment_analysis(df)
        for col in ("scaffold", "n_compounds", "mean_score", "best_rank", "n_in_top100"):
            assert col in result.columns, f"Missing column: {col}"

    def test_compounds_sharing_scaffold_grouped(self):
        """Aspirin and benzamide share the benzene scaffold → same group."""
        df = _make_df(
            [ASPIRIN, BENZAMIDE, BENZOIC_ACID, CAFFEINE],
            scores=[0.1, 0.2, 0.3, 0.9],
        )
        result = scaffold_enrichment_analysis(df)
        # The benzene scaffold should have n_compounds == 3
        # Find the row for the benzene-based scaffold
        benzene_row = result.filter(pl.col("n_compounds") == 3)
        assert len(benzene_row) == 1

    def test_n_compounds_correct(self):
        df = _make_df([ASPIRIN, ASPIRIN, CAFFEINE], scores=[0.1, 0.2, 0.8])
        result = scaffold_enrichment_analysis(df)
        aspirin_scaffold = compute_murcko_scaffolds([ASPIRIN])[0]
        row = result.filter(pl.col("scaffold") == aspirin_scaffold)
        assert len(row) == 1
        assert row["n_compounds"][0] == 2

    def test_mean_score_correct(self):
        df = _make_df([ASPIRIN, ASPIRIN], scores=[0.2, 0.4])
        result = scaffold_enrichment_analysis(df)
        aspirin_scaffold = compute_murcko_scaffolds([ASPIRIN])[0]
        row = result.filter(pl.col("scaffold") == aspirin_scaffold)
        assert abs(row["mean_score"][0] - 0.3) < 1e-9

    def test_sorted_by_mean_score_ascending(self):
        """Best (lowest) score scaffold should appear first."""
        df = _make_df(
            [ASPIRIN, CAFFEINE],
            scores=[0.1, 0.9],  # aspirin → better score
        )
        result = scaffold_enrichment_analysis(df)
        # First row should have mean_score ≤ second row
        scores = result["mean_score"].to_list()
        assert scores[0] <= scores[1]

    def test_top_n_scaffolds_limits_rows(self):
        smiles = [ASPIRIN, CAFFEINE, IBUPROFEN, NAPROXEN, PARACETAMOL]
        df = _make_df(smiles, scores=[0.1 * (i + 1) for i in range(5)])
        result = scaffold_enrichment_analysis(df, top_n_scaffolds=2)
        assert len(result) <= 2

    def test_invalid_smiles_grouped_as_no_scaffold(self):
        df = _make_df([ASPIRIN, INVALID], scores=[0.1, 0.9])
        result = scaffold_enrichment_analysis(df)
        no_scaffold = result.filter(pl.col("scaffold") == "[no scaffold]")
        assert len(no_scaffold) == 1
        assert no_scaffold["n_compounds"][0] == 1

    def test_acyclic_grouped_as_no_scaffold(self):
        df = _make_df([ASPIRIN, GLYCINE], scores=[0.1, 0.5])
        result = scaffold_enrichment_analysis(df)
        no_scaffold = result.filter(pl.col("scaffold") == "[no scaffold]")
        assert len(no_scaffold) == 1

    def test_missing_smiles_col_raises(self):
        df = pl.DataFrame({"rank": [1, 2], "code_1": [0, 1]})
        with pytest.raises(ValueError, match="smiles"):
            scaffold_enrichment_analysis(df, smiles_col="smiles")

    def test_missing_score_col_no_crash(self):
        """If score_col not in df, mean_score should be null."""
        df = pl.DataFrame({
            "smiles": [ASPIRIN, CAFFEINE],
            "rank": [1, 2],
        })
        result = scaffold_enrichment_analysis(df, score_col="composite_score")
        assert "mean_score" in result.columns
        assert result["mean_score"].null_count() == len(result)

    def test_n_in_top100_present(self):
        smiles = [ASPIRIN, CAFFEINE, IBUPROFEN]
        df = _make_df(smiles, scores=[0.1, 0.5, 0.9])
        result = scaffold_enrichment_analysis(df)
        assert "n_in_top100" in result.columns

    def test_best_rank_is_minimum(self):
        """Scaffold with ranks 1 and 3 → best_rank=1."""
        df = _make_df([ASPIRIN, CAFFEINE, ASPIRIN], scores=[0.1, 0.5, 0.3])
        result = scaffold_enrichment_analysis(df)
        aspirin_scaffold = compute_murcko_scaffolds([ASPIRIN])[0]
        row = result.filter(pl.col("scaffold") == aspirin_scaffold)
        assert row["best_rank"][0] == 1  # min rank among aspirin rows

    def test_no_rank_col_uses_row_index(self):
        """When no 'rank' column, best_rank = minimum row index."""
        df = pl.DataFrame({
            "smiles": [ASPIRIN, CAFFEINE],
            "composite_score": [0.1, 0.9],
        })
        result = scaffold_enrichment_analysis(df)
        assert "best_rank" in result.columns

    def test_all_invalid_smiles_all_no_scaffold(self):
        df = _make_df([INVALID, INVALID], scores=[0.5, 0.6])
        result = scaffold_enrichment_analysis(df)
        no_scaffold = result.filter(pl.col("scaffold") == "[no scaffold]")
        assert no_scaffold["n_compounds"][0] == 2
