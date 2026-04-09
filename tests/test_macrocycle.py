"""Tests for explore/macrocycle.py.

Molecules used:
  Cyclosporin A   — cyclic undecapeptide, one 33-atom macrocyclic ring
  Aspirin         — no macrocycle (6-membered aromatic ring only)
  Erythromycin A  — 14-membered macrolide ring plus two 6-membered sugar rings
  Cyclododecane   — simplest 12-atom macrocycle (boundary case)
  Invalid SMILES  — graceful failure expected
"""

from __future__ import annotations

import polars as pl
import pytest

from delexplore.explore.macrocycle import (
    add_macrocycle_columns,
    assess_macrocycle_druglikeness,
    calculate_macrocycle_descriptors,
    detect_macrocycles,
)

# ---------------------------------------------------------------------------
# SMILES constants
# ---------------------------------------------------------------------------

CYCLOSPORIN_A = (
    "CC[C@@H]1NC(=O)[C@H]([C@@H](CC)C)N(C)C(=O)[C@@H]"
    "(NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)"
    "[C@@H](NC(=O)[C@H](C(C)C)NC(=O)CN(C)C(=O)[C@H](CC(C)C)"
    "N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)[C@H](C)N(C)C1=O)"
    "C(C)C)[C@@H](C)OC"
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

# Erythromycin A — 14-membered macrolide + two 6-membered sugar rings
ERYTHROMYCIN_A = (
    "CCC1OC(=O)[C@@H](C)[C@H](O[C@@H]2C[C@@](C)(OC)[C@H](O)"
    "[C@@H](C)O2)[C@@H](C)[C@H](O[C@@H]3O[C@@H](C)C[C@H]"
    "(N(C)C)[C@@H]3O)[C@](C)(O)C[C@H](C)C(=O)[C@@H](C)"
    "[C@H](O1)C"
)

# Cyclododecane — minimal 12-atom macrocycle (12 CH2 groups)
CYCLODODECANE = "C1CCCCCCCCCCC1"

INVALID_SMILES = "not_a_smiles"


# ---------------------------------------------------------------------------
# 1. detect_macrocycles
# ---------------------------------------------------------------------------


class TestDetectMacrocycles:
    def test_cyclosporin_a_has_33_membered_ring(self):
        rings = detect_macrocycles(CYCLOSPORIN_A)
        assert len(rings) == 1
        assert rings[0] == 33

    def test_aspirin_has_no_macrocycle(self):
        rings = detect_macrocycles(ASPIRIN)
        assert rings == []

    def test_erythromycin_has_14_membered_ring(self):
        rings = detect_macrocycles(ERYTHROMYCIN_A)
        # Erythromycin has one 14-membered macrolide ring
        assert 14 in rings
        # The two 6-membered sugar rings are NOT macrocycles
        assert all(s >= 12 for s in rings)

    def test_cyclododecane_is_exactly_at_boundary(self):
        rings = detect_macrocycles(CYCLODODECANE)
        assert rings == [12]

    def test_invalid_smiles_returns_empty_list(self):
        rings = detect_macrocycles(INVALID_SMILES)
        assert rings == []

    def test_empty_string_returns_empty_list(self):
        assert detect_macrocycles("") == []

    def test_none_like_empty_string(self):
        # detect_macrocycles accepts str; empty string → no mol → []
        assert detect_macrocycles("") == []

    def test_small_ring_not_macrocycle(self):
        # Benzene (6-membered), cyclohexane (6-membered) — not macrocycles
        assert detect_macrocycles("c1ccccc1") == []
        assert detect_macrocycles("C1CCCCC1") == []

    def test_11_membered_ring_not_macrocycle(self):
        # Cycloundecane — 11 atoms, just below threshold
        assert detect_macrocycles("C1CCCCCCCCCC1") == []


# ---------------------------------------------------------------------------
# 2. calculate_macrocycle_descriptors
# ---------------------------------------------------------------------------


class TestCalculateMacrocycleDescriptors:
    def test_cyclosporin_a_is_macrocycle(self):
        desc = calculate_macrocycle_descriptors(CYCLOSPORIN_A)
        assert desc["is_macrocycle"] is True
        assert desc["largest_ring_size"] == 33
        assert desc["num_macrocyclic_rings"] == 1
        assert desc["fraction_sp3"] is not None
        assert 0.0 <= desc["fraction_sp3"] <= 1.0

    def test_cyclosporin_a_3d_descriptors(self):
        desc = calculate_macrocycle_descriptors(CYCLOSPORIN_A)
        # 3D descriptors must be computed for macrocycles
        assert desc["globularity"] is not None
        # Cyclosporin A is moderately globular
        assert 0.0 < desc["globularity"] <= 1.0
        # Should find some low-energy conformers
        assert desc["num_conformers"] is not None
        assert desc["num_conformers"] >= 1

    def test_aspirin_is_not_macrocycle(self):
        desc = calculate_macrocycle_descriptors(ASPIRIN)
        assert desc["is_macrocycle"] is False
        assert desc["largest_ring_size"] == 0
        assert desc["num_macrocyclic_rings"] == 0
        # 3D descriptors must be null for non-macrocycles
        assert desc["globularity"] is None
        assert desc["num_conformers"] is None
        # fraction_sp3 is still computed
        assert desc["fraction_sp3"] is not None

    def test_erythromycin_is_macrocycle(self):
        desc = calculate_macrocycle_descriptors(ERYTHROMYCIN_A)
        assert desc["is_macrocycle"] is True
        assert desc["largest_ring_size"] == 14
        assert desc["globularity"] is not None

    def test_invalid_smiles_returns_safe_defaults(self):
        desc = calculate_macrocycle_descriptors(INVALID_SMILES)
        assert desc["is_macrocycle"] is False
        assert desc["largest_ring_size"] == 0
        assert desc["num_macrocyclic_rings"] == 0
        assert desc["fraction_sp3"] is None
        assert desc["globularity"] is None
        assert desc["num_conformers"] is None

    def test_globularity_range(self):
        """Globularity is in (0, 1] for any valid macrocycle."""
        desc = calculate_macrocycle_descriptors(CYCLODODECANE)
        assert desc["is_macrocycle"] is True
        if desc["globularity"] is not None:  # cyclododecane should embed fine
            assert 0.0 < desc["globularity"] <= 1.0


# ---------------------------------------------------------------------------
# 3. add_macrocycle_columns — DataFrame vectorized
# ---------------------------------------------------------------------------


class TestAddMacrocycleColumns:
    @pytest.fixture(scope="class")
    def mixed_df(self):
        return pl.DataFrame(
            {
                "code_1": [0, 1, 2, 3, 4],
                "smiles": [
                    ASPIRIN,
                    CYCLOSPORIN_A,
                    ERYTHROMYCIN_A,
                    CYCLODODECANE,
                    INVALID_SMILES,
                ],
            }
        )

    def test_columns_added(self, mixed_df):
        result = add_macrocycle_columns(mixed_df)
        assert "is_macrocycle" in result.columns
        assert "largest_ring_size" in result.columns
        assert "globularity" in result.columns

    def test_original_columns_preserved(self, mixed_df):
        result = add_macrocycle_columns(mixed_df)
        assert "code_1" in result.columns
        assert "smiles" in result.columns
        assert len(result) == len(mixed_df)

    def test_aspirin_not_macrocycle(self, mixed_df):
        result = add_macrocycle_columns(mixed_df)
        aspirin_row = result.filter(pl.col("code_1") == 0).row(0, named=True)
        assert aspirin_row["is_macrocycle"] is False
        assert aspirin_row["largest_ring_size"] == 0
        assert aspirin_row["globularity"] is None

    def test_cyclosporin_a_is_macrocycle(self, mixed_df):
        result = add_macrocycle_columns(mixed_df)
        csa_row = result.filter(pl.col("code_1") == 1).row(0, named=True)
        assert csa_row["is_macrocycle"] is True
        assert csa_row["largest_ring_size"] == 33
        assert csa_row["globularity"] is not None
        assert 0.0 < csa_row["globularity"] <= 1.0

    def test_erythromycin_is_macrocycle(self, mixed_df):
        result = add_macrocycle_columns(mixed_df)
        ery_row = result.filter(pl.col("code_1") == 2).row(0, named=True)
        assert ery_row["is_macrocycle"] is True
        assert ery_row["largest_ring_size"] == 14

    def test_invalid_smiles_not_macrocycle(self, mixed_df):
        result = add_macrocycle_columns(mixed_df)
        inv_row = result.filter(pl.col("code_1") == 4).row(0, named=True)
        assert inv_row["is_macrocycle"] is False
        assert inv_row["largest_ring_size"] == 0
        assert inv_row["globularity"] is None

    def test_dtypes(self, mixed_df):
        result = add_macrocycle_columns(mixed_df)
        assert result["is_macrocycle"].dtype == pl.Boolean
        assert result["largest_ring_size"].dtype == pl.Int64
        assert result["globularity"].dtype == pl.Float64

    def test_missing_smiles_col_raises(self):
        df = pl.DataFrame({"code_1": [0], "other": ["CC"]})
        with pytest.raises(ValueError, match="not found"):
            add_macrocycle_columns(df, smiles_col="smiles")

    def test_empty_dataframe(self):
        empty = pl.DataFrame(
            {"code_1": [], "smiles": []},
            schema={"code_1": pl.Int64, "smiles": pl.Utf8},
        )
        result = add_macrocycle_columns(empty)
        assert len(result) == 0
        assert "is_macrocycle" in result.columns
        assert "largest_ring_size" in result.columns
        assert "globularity" in result.columns


# ---------------------------------------------------------------------------
# 4. assess_macrocycle_druglikeness
# ---------------------------------------------------------------------------


def _make_props_df(
    is_macrocycle: bool,
    mw: float,
    tpsa: float,
    hbd: float,
    num_conformers: int | None,
) -> pl.DataFrame:
    """Build a minimal properties DataFrame for assess_macrocycle_druglikeness."""
    return pl.DataFrame(
        {
            "is_macrocycle": [is_macrocycle],
            "mw": [mw],
            "tpsa": [tpsa],
            "hbd": [hbd],
            "num_conformers": [num_conformers],
        },
        schema={
            "is_macrocycle": pl.Boolean,
            "mw": pl.Float64,
            "tpsa": pl.Float64,
            "hbd": pl.Float64,
            "num_conformers": pl.Int64,
        },
    )


class TestAssessMacrocycleDruglikeness:
    def test_non_macrocycle_gets_null_classes(self):
        df = _make_props_df(False, mw=300.0, tpsa=80.0, hbd=2.0, num_conformers=15)
        result = assess_macrocycle_druglikeness(df)
        row = result.row(0, named=True)
        assert row["macro_oral_class"] is None
        assert row["macro_permeability_class"] is None

    def test_oral_likely(self):
        # MW < 1000, TPSA < 250, HBD ≤ 5
        df = _make_props_df(True, mw=800.0, tpsa=200.0, hbd=4.0, num_conformers=12)
        result = assess_macrocycle_druglikeness(df)
        assert result["macro_oral_class"][0] == "likely"

    def test_oral_uncertain(self):
        # MW < 1200, TPSA < 300, but fails "likely" (HBD = 6)
        df = _make_props_df(True, mw=1100.0, tpsa=280.0, hbd=6.0, num_conformers=5)
        result = assess_macrocycle_druglikeness(df)
        assert result["macro_oral_class"][0] == "uncertain"

    def test_oral_unlikely(self):
        # MW ≥ 1200
        df = _make_props_df(True, mw=1300.0, tpsa=350.0, hbd=8.0, num_conformers=3)
        result = assess_macrocycle_druglikeness(df)
        assert result["macro_oral_class"][0] == "unlikely"

    def test_permeability_high(self):
        # num_conformers ≥ 10 AND TPSA < 200
        df = _make_props_df(True, mw=700.0, tpsa=150.0, hbd=2.0, num_conformers=15)
        result = assess_macrocycle_druglikeness(df)
        assert result["macro_permeability_class"][0] == "high"

    def test_permeability_high_requires_low_tpsa(self):
        # num_conformers ≥ 10 but TPSA ≥ 200 → medium (not high)
        df = _make_props_df(True, mw=700.0, tpsa=220.0, hbd=2.0, num_conformers=15)
        result = assess_macrocycle_druglikeness(df)
        assert result["macro_permeability_class"][0] == "medium"

    def test_permeability_medium(self):
        # num_conformers ≥ 5 (but < 10)
        df = _make_props_df(True, mw=700.0, tpsa=180.0, hbd=2.0, num_conformers=7)
        result = assess_macrocycle_druglikeness(df)
        assert result["macro_permeability_class"][0] == "medium"

    def test_permeability_low(self):
        # num_conformers < 5
        df = _make_props_df(True, mw=700.0, tpsa=180.0, hbd=2.0, num_conformers=2)
        result = assess_macrocycle_druglikeness(df)
        assert result["macro_permeability_class"][0] == "low"

    def test_permeability_low_when_null_conformers(self):
        # num_conformers is null (embedding failed) → "low"
        df = _make_props_df(True, mw=700.0, tpsa=150.0, hbd=2.0, num_conformers=None)
        result = assess_macrocycle_druglikeness(df)
        assert result["macro_permeability_class"][0] == "low"

    def test_missing_required_column_raises(self):
        df = pl.DataFrame({"is_macrocycle": [True], "mw": [700.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            assess_macrocycle_druglikeness(df)

    def test_output_columns_added(self):
        df = _make_props_df(True, mw=700.0, tpsa=150.0, hbd=2.0, num_conformers=10)
        result = assess_macrocycle_druglikeness(df)
        assert "macro_oral_class" in result.columns
        assert "macro_permeability_class" in result.columns
        # Original columns preserved
        assert "is_macrocycle" in result.columns
        assert "mw" in result.columns

    def test_mixed_macro_and_small(self):
        """Batch DataFrame with macrocycles and non-macrocycles."""
        df = pl.DataFrame(
            {
                "is_macrocycle": [True, False, True],
                "mw": [800.0, 300.0, 1300.0],
                "tpsa": [150.0, 80.0, 400.0],
                "hbd": [3.0, 1.0, 10.0],
                "num_conformers": [12, None, 1],
            },
            schema={
                "is_macrocycle": pl.Boolean,
                "mw": pl.Float64,
                "tpsa": pl.Float64,
                "hbd": pl.Float64,
                "num_conformers": pl.Int64,
            },
        )
        result = assess_macrocycle_druglikeness(df)
        orals = result["macro_oral_class"].to_list()
        assert orals[0] == "likely"
        assert orals[1] is None   # non-macrocycle
        assert orals[2] == "unlikely"
