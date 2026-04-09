"""Tests for explore/properties.py.

Coverage:
  1. Aspirin (MW≈180) → passes Lipinski, penalty=1.0
  2. Cyclosporin A (MW≈1203) → fails bRo5, penalty=2.0
  3. Molecule with MW in bRo5-only space → bro5_pass=True, penalty=1.0
     (tested by injecting known property values into assess_druglikeness)
  4. Invalid SMILES → all properties null, penalty=1.5
  5. Empty DataFrame → correct schema, no rows
  6. Integration: compute_properties_for_ranking output feeds compute_composite_rank
"""

from __future__ import annotations

import polars as pl
import pytest

from delexplore.explore.properties import (
    assess_druglikeness,
    calculate_properties,
    compute_properties_for_ranking,
)

# ---------------------------------------------------------------------------
# SMILES constants
# ---------------------------------------------------------------------------

# Aspirin: MW=180.16, LogP≈1.2, HBD=1, HBA=4 — comfortably inside Lipinski
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"

# Cyclosporin A: MW=1202.6 — outside bRo5 (MW > 1000)
CYCLOSPORIN_A_SMILES = (
    "CC[C@@H]1NC(=O)[C@H]([C@@H](CC)C)N(C)C(=O)[C@@H]"
    "(NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)"
    "[C@@H](NC(=O)[C@H](C(C)C)NC(=O)CN(C)C(=O)[C@H](CC(C)C)"
    "N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)[C@H](C)N(C)C1=O)"
    "C(C)C)[C@@H](C)OC"
)

INVALID_SMILES = "not_a_smiles"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_smiles_df(smiles: str, code_1: int = 0) -> pl.DataFrame:
    return pl.DataFrame(
        {"code_1": [code_1], "smiles": [smiles]},
        schema={"code_1": pl.Int64, "smiles": pl.Utf8},
    )


def _property_row(
    mw: float,
    logp: float,
    hba: float,
    hbd: float,
    tpsa: float,
    rotatable_bonds: float,
    code_1: int = 0,
) -> pl.DataFrame:
    """Build a DataFrame with known property values for testing assess_druglikeness."""
    return pl.DataFrame(
        {
            "code_1": [code_1],
            "mw": [mw],
            "logp": [logp],
            "hba": [hba],
            "hbd": [hbd],
            "tpsa": [tpsa],
            "rotatable_bonds": [rotatable_bonds],
            "num_rings": [2.0],
            "num_aromatic_rings": [1.0],
            "fraction_sp3": [0.3],
            "heavy_atom_count": [float(int(mw / 8))],
            "qed": [0.5],
        }
    )


# ---------------------------------------------------------------------------
# 1. Aspirin — passes Lipinski → penalty 1.0
# ---------------------------------------------------------------------------


def test_aspirin_lipinski_pass():
    df = _single_smiles_df(ASPIRIN_SMILES)
    result = calculate_properties(df)
    assessed = assess_druglikeness(result)

    row = assessed.row(0, named=True)

    assert row["mw"] == pytest.approx(180.16, abs=0.5), "Aspirin MW"
    assert row["lipinski_pass"] is True, "Aspirin should pass Lipinski Ro5"
    assert row["bro5_pass"] is True, "Aspirin should also pass bRo5"
    assert row["property_penalty"] == pytest.approx(1.0)


def test_aspirin_descriptor_ranges():
    """Spot-check individual descriptors for aspirin."""
    df = _single_smiles_df(ASPIRIN_SMILES)
    result = calculate_properties(df)
    row = result.row(0, named=True)

    assert 170.0 < row["mw"] < 190.0
    assert row["hbd"] == pytest.approx(1.0)   # carboxylic OH only (ester O not donor)
    assert row["hba"] == pytest.approx(3.0)   # acetyl C=O, ester O, carboxyl C=O (OH is donor not acceptor)
    assert row["qed"] is not None
    assert 0.0 < row["qed"] <= 1.0


# ---------------------------------------------------------------------------
# 2. Cyclosporin A — MW > 1000, fails bRo5 → penalty 2.0
# ---------------------------------------------------------------------------


def test_cyclosporin_fails_bro5():
    df = _single_smiles_df(CYCLOSPORIN_A_SMILES)
    result = calculate_properties(df)
    assessed = assess_druglikeness(result)

    row = assessed.row(0, named=True)

    # MW should be ~1202 — definitely above 1000
    assert row["mw"] > 1000.0, f"Expected MW > 1000, got {row['mw']}"
    assert row["lipinski_pass"] is False
    assert row["bro5_pass"] is False
    assert row["property_penalty"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 3. Medium molecule (MW ~700) — fails Lipinski, passes bRo5 → penalty 1.0
#    Tested by injecting known property values into assess_druglikeness directly.
# ---------------------------------------------------------------------------


def test_medium_molecule_bro5_pass():
    """MW=700, all other bRo5 criteria met → lipinski_pass=False, bro5_pass=True."""
    df = _property_row(mw=700.0, logp=4.0, hba=8.0, hbd=3.0, tpsa=120.0, rotatable_bonds=8.0)
    result = assess_druglikeness(df)
    row = result.row(0, named=True)

    assert row["lipinski_pass"] is False, "MW=700 should fail Lipinski (MW > 500)"
    assert row["bro5_pass"] is True, "MW=700 with acceptable other props should pass bRo5"
    assert row["property_penalty"] == pytest.approx(1.0)


def test_medium_molecule_bro5_fail_high_logp():
    """MW=800, LogP=11 → fails bRo5 LogP criterion → penalty 2.0."""
    df = _property_row(mw=800.0, logp=11.0, hba=8.0, hbd=3.0, tpsa=120.0, rotatable_bonds=8.0)
    result = assess_druglikeness(df)
    row = result.row(0, named=True)

    assert row["bro5_pass"] is False
    assert row["property_penalty"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 4. Invalid SMILES → all properties null, penalty 1.5
# ---------------------------------------------------------------------------


def test_invalid_smiles_null_properties():
    df = _single_smiles_df(INVALID_SMILES)
    result = calculate_properties(df)
    assessed = assess_druglikeness(result)

    row = assessed.row(0, named=True)

    property_cols = [
        "mw", "logp", "hba", "hbd", "tpsa", "rotatable_bonds",
        "num_rings", "num_aromatic_rings", "fraction_sp3", "heavy_atom_count", "qed",
    ]
    for col in property_cols:
        assert row[col] is None, f"Expected null for {col} with invalid SMILES"

    # Rule columns should be null (cannot evaluate)
    assert row["lipinski_pass"] is None
    assert row["bro5_pass"] is None

    # Penalty is 1.5 (uncertain, mild penalty)
    assert row["property_penalty"] == pytest.approx(1.5)


def test_invalid_smiles_warning(caplog):
    import logging

    df = _single_smiles_df(INVALID_SMILES, code_1=42)
    with caplog.at_level(logging.WARNING, logger="delexplore.explore.properties"):
        calculate_properties(df)

    assert any("Invalid SMILES" in msg for msg in caplog.messages), (
        "Expected a warning about invalid SMILES"
    )
    assert any("42" in msg or "not_a_smiles" in msg for msg in caplog.messages)


def test_none_smiles_null_properties():
    df = pl.DataFrame({"code_1": [0], "smiles": [None]}, schema={"code_1": pl.Int64, "smiles": pl.Utf8})
    result = calculate_properties(df)
    assert result["mw"][0] is None


def test_all_invalid_smiles_warning(caplog):
    import logging

    df = pl.DataFrame(
        {"code_1": [0, 1], "smiles": ["bad1", "bad2"]},
        schema={"code_1": pl.Int64, "smiles": pl.Utf8},
    )
    with caplog.at_level(logging.WARNING, logger="delexplore.explore.properties"):
        calculate_properties(df)

    assert any("All" in msg and "invalid" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# 5. Empty DataFrame → correct schema, no rows
# ---------------------------------------------------------------------------


def test_empty_dataframe_schema():
    empty = pl.DataFrame(
        {"code_1": [], "smiles": []},
        schema={"code_1": pl.Int64, "smiles": pl.Utf8},
    )
    result = calculate_properties(empty)

    assert len(result) == 0
    expected_cols = [
        "mw", "logp", "hba", "hbd", "tpsa", "rotatable_bonds",
        "num_rings", "num_aromatic_rings", "fraction_sp3", "heavy_atom_count", "qed",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Expected column '{col}' in empty result"
        assert result[col].dtype == pl.Float64, f"Expected Float64 for '{col}'"


def test_empty_dataframe_assess_druglikeness():
    empty = pl.DataFrame(
        {
            "code_1": [],
            "mw": [], "logp": [], "hba": [], "hbd": [],
            "tpsa": [], "rotatable_bonds": [],
            "num_rings": [], "num_aromatic_rings": [],
            "fraction_sp3": [], "heavy_atom_count": [], "qed": [],
        },
        schema={
            "code_1": pl.Int64,
            "mw": pl.Float64, "logp": pl.Float64,
            "hba": pl.Float64, "hbd": pl.Float64,
            "tpsa": pl.Float64, "rotatable_bonds": pl.Float64,
            "num_rings": pl.Float64, "num_aromatic_rings": pl.Float64,
            "fraction_sp3": pl.Float64, "heavy_atom_count": pl.Float64,
            "qed": pl.Float64,
        },
    )
    result = assess_druglikeness(empty)
    assert len(result) == 0
    assert "property_penalty" in result.columns
    assert "lipinski_pass" in result.columns


# ---------------------------------------------------------------------------
# 6. Error handling
# ---------------------------------------------------------------------------


def test_missing_smiles_col_raises():
    df = pl.DataFrame({"code_1": [0], "not_smiles": ["CC"]})
    with pytest.raises(ValueError, match="not found"):
        calculate_properties(df, smiles_col="smiles")


def test_compute_properties_for_ranking_missing_smiles_raises():
    df = pl.DataFrame({"code_1": [0], "other": ["CC"]})
    with pytest.raises(ValueError, match="not found"):
        compute_properties_for_ranking(df, smiles_col="smiles")


def test_compute_properties_for_ranking_no_code_cols_raises():
    df = pl.DataFrame({"x": [0], "smiles": [ASPIRIN_SMILES]})
    with pytest.raises(ValueError, match="No code columns"):
        compute_properties_for_ranking(df, smiles_col="smiles")


# ---------------------------------------------------------------------------
# 6 (cont). Integration: output plugs into compute_composite_rank
# ---------------------------------------------------------------------------


def test_integration_with_compute_composite_rank(synthetic_counts):
    """compute_properties_for_ranking output passes to compute_composite_rank."""
    from delexplore.analyse.multilevel import run_multilevel_enrichment
    from delexplore.analyse.rank import compute_composite_rank

    code_cols = ["code_1", "code_2"]
    post_selections = ["target_1", "target_2", "target_3"]
    control_selections = ["blank_1", "blank_2", "blank_3"]

    ml_result = run_multilevel_enrichment(
        synthetic_counts,
        n_cycles=2,
        code_cols=code_cols,
        post_selections=post_selections,
        control_selections=control_selections,
    )

    # Build a SMILES DataFrame for all compounds in the full-compound level
    from delexplore.analyse.aggregate import get_level_name

    full_level = get_level_name(tuple(code_cols))
    compound_df = ml_result[full_level].select(code_cols)

    # Assign aspirin SMILES to all compounds (arbitrary — just needs to be valid)
    smiles_df = compound_df.with_columns(pl.lit(ASPIRIN_SMILES).alias("smiles"))

    penalty_df = compute_properties_for_ranking(
        smiles_df, smiles_col="smiles", code_cols=code_cols
    )

    # Verify output schema
    assert "property_penalty" in penalty_df.columns
    assert set(code_cols).issubset(set(penalty_df.columns))
    assert len(penalty_df.columns) == len(code_cols) + 1

    # Verify no nulls in penalty column (aspirin is valid)
    assert penalty_df["property_penalty"].null_count() == 0

    # Run composite rank — must not raise
    ranked = compute_composite_rank(
        ml_result,
        code_cols=code_cols,
        properties_df=penalty_df,
        property_penalty_col="property_penalty",
    )

    assert "rank" in ranked.columns
    assert "composite_score" in ranked.columns
    assert len(ranked) == len(compound_df)

    # All penalties are 1.0 (aspirin passes bRo5) so ranks should be contiguous
    assert ranked["rank"].min() == 1
    assert ranked["rank"].max() == len(ranked)


# ---------------------------------------------------------------------------
# Rule edge cases via assess_druglikeness injection
# ---------------------------------------------------------------------------


def test_lipinski_exactly_at_boundary():
    """MW=500, LogP=5, HBD=5, HBA=10 — exactly at Lipinski boundary → pass."""
    df = _property_row(mw=500.0, logp=5.0, hba=10.0, hbd=5.0, tpsa=100.0, rotatable_bonds=5.0)
    result = assess_druglikeness(df)
    assert result["lipinski_pass"][0] is True


def test_lipinski_one_over_boundary():
    """MW=501 — one criterion fails → lipinski_pass=False."""
    df = _property_row(mw=501.0, logp=5.0, hba=10.0, hbd=5.0, tpsa=100.0, rotatable_bonds=5.0)
    result = assess_druglikeness(df)
    assert result["lipinski_pass"][0] is False
    # Still within bRo5
    assert result["bro5_pass"][0] is True
    assert result["property_penalty"][0] == pytest.approx(1.0)


def test_pfizer_3_75_pass():
    """LogP < 3 AND TPSA > 75 → pfizer_3_75_pass=True (low tox risk)."""
    df = _property_row(mw=300.0, logp=2.0, hba=4.0, hbd=2.0, tpsa=80.0, rotatable_bonds=4.0)
    result = assess_druglikeness(df)
    assert result["pfizer_3_75_pass"][0] is True


def test_pfizer_3_75_fail():
    """LogP > 3 OR TPSA ≤ 75 → pfizer_3_75_pass=False (potential tox flag)."""
    df = _property_row(mw=300.0, logp=4.0, hba=2.0, hbd=1.0, tpsa=50.0, rotatable_bonds=4.0)
    result = assess_druglikeness(df)
    assert result["pfizer_3_75_pass"][0] is False


def test_veber_pass():
    df = _property_row(mw=300.0, logp=2.0, hba=4.0, hbd=2.0, tpsa=100.0, rotatable_bonds=10.0)
    result = assess_druglikeness(df)
    assert result["veber_pass"][0] is True


def test_veber_fail_tpsa():
    df = _property_row(mw=300.0, logp=2.0, hba=4.0, hbd=2.0, tpsa=145.0, rotatable_bonds=8.0)
    result = assess_druglikeness(df)
    assert result["veber_pass"][0] is False


# ---------------------------------------------------------------------------
# compute_properties_for_ranking — code_cols auto-detection
# ---------------------------------------------------------------------------


def test_compute_properties_auto_detect_code_cols():
    df = pl.DataFrame(
        {
            "code_1": [0, 1],
            "code_2": [3, 4],
            "smiles": [ASPIRIN_SMILES, ASPIRIN_SMILES],
        }
    )
    result = compute_properties_for_ranking(df)
    assert set(result.columns) == {"code_1", "code_2", "property_penalty"}
    assert len(result) == 2


def test_compute_properties_explicit_code_cols():
    df = pl.DataFrame(
        {
            "code_1": [0],
            "code_2": [1],
            "extra": [99],
            "smiles": [ASPIRIN_SMILES],
        }
    )
    result = compute_properties_for_ranking(df, code_cols=["code_1"])
    assert set(result.columns) == {"code_1", "property_penalty"}


def test_compute_properties_penalty_values():
    """Valid aspirin → 1.0; invalid SMILES → 1.5."""
    df = pl.DataFrame(
        {
            "code_1": [0, 1],
            "smiles": [ASPIRIN_SMILES, INVALID_SMILES],
        }
    )
    result = compute_properties_for_ranking(df)
    penalties = dict(zip(result["code_1"].to_list(), result["property_penalty"].to_list()))
    assert penalties[0] == pytest.approx(1.0)
    assert penalties[1] == pytest.approx(1.5)
