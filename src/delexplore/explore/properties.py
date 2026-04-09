"""Molecular property calculation and drug-likeness assessment.

Computes RDKit-based descriptors from SMILES strings and produces a
``property_penalty`` column compatible with the consensus ranking system in
:mod:`delexplore.analyse.rank`.

Usage
-----
>>> props = calculate_properties(df, smiles_col="smiles")
>>> props = assess_druglikeness(props)
>>> # Or, as a one-liner for ranking:
>>> penalty_df = compute_properties_for_ranking(df, smiles_col="smiles")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RDKit import (gated — optional dependency)
# ---------------------------------------------------------------------------

try:
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors
    from rdkit.rdBase import BlockLogs

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    logger.warning(
        "RDKit is not installed. Property calculation will not be available. "
        "Install with: pip install rdkit"
    )


# ---------------------------------------------------------------------------
# Property column names (single source of truth)
# ---------------------------------------------------------------------------

_PROPERTY_COLS = [
    "mw",
    "logp",
    "hba",
    "hbd",
    "tpsa",
    "rotatable_bonds",
    "num_rings",
    "num_aromatic_rings",
    "fraction_sp3",
    "heavy_atom_count",
    "qed",
]

_RULE_COLS = [
    "lipinski_pass",
    "veber_pass",
    "bro5_pass",
    "pfizer_3_75_pass",
    "property_penalty",
]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_mol(smiles: str | None):
    """Return an RDKit Mol, or None if invalid/missing.

    Args:
        smiles: A SMILES string, or None/empty string.

    Returns:
        Parsed ``Mol`` object, or ``None``.
    """
    if not smiles:
        return None
    with BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    return mol


def _compute_row_properties(smiles: str | None) -> dict[str, float | None]:
    """Compute all descriptors for one SMILES string.

    Args:
        smiles: A SMILES string, or None.

    Returns:
        Dict mapping property name → float value, or None when the molecule
        could not be parsed or a descriptor call fails.
    """
    null_row: dict[str, float | None] = {col: None for col in _PROPERTY_COLS}

    mol = _parse_mol(smiles)
    if mol is None:
        return null_row

    try:
        with BlockLogs():
            return {
                "mw": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "hba": float(Descriptors.NumHAcceptors(mol)),
                "hbd": float(Descriptors.NumHDonors(mol)),
                "tpsa": Descriptors.TPSA(mol),
                "rotatable_bonds": float(Descriptors.NumRotatableBonds(mol)),
                "num_rings": float(Descriptors.RingCount(mol)),
                "num_aromatic_rings": float(Descriptors.NumAromaticRings(mol)),
                "fraction_sp3": Descriptors.FractionCSP3(mol),
                "heavy_atom_count": float(Descriptors.HeavyAtomCount(mol)),
                "qed": QED.qed(mol),
            }
    except Exception:
        logger.debug("Descriptor computation failed for SMILES: %s", smiles)
        return null_row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_properties(
    df: pl.DataFrame,
    smiles_col: str = "smiles",
) -> pl.DataFrame:
    """Compute molecular descriptors from SMILES and append them to *df*.

    For each row, all RDKit descriptors are computed in a single pass.
    Rows with missing or unparseable SMILES receive null for every property
    column; a warning is logged with the row index and SMILES value.

    Args:
        df: Input DataFrame.  Must contain *smiles_col*.  All existing
            columns are preserved unchanged.
        smiles_col: Name of the column holding SMILES strings.

    Returns:
        Copy of *df* with the following Float64 columns appended:
        ``mw``, ``logp``, ``hba``, ``hbd``, ``tpsa``, ``rotatable_bonds``,
        ``num_rings``, ``num_aromatic_rings``, ``fraction_sp3``,
        ``heavy_atom_count``, ``qed``.

        Null values appear for invalid/missing SMILES.

    Raises:
        ImportError: If RDKit is not installed.
        ValueError: If *smiles_col* is absent from *df*.
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for property calculation. "
            "Install with: pip install rdkit"
        )
    if smiles_col not in df.columns:
        raise ValueError(
            f"Column '{smiles_col}' not found in DataFrame. "
            f"Available columns: {df.columns}"
        )

    if len(df) == 0:
        return df.with_columns(
            [pl.lit(None).cast(pl.Float64).alias(col) for col in _PROPERTY_COLS]
        )

    # Compute properties row-by-row; collect per-column lists
    col_data: dict[str, list[float | None]] = {col: [] for col in _PROPERTY_COLS}
    smiles_series = df[smiles_col].to_list()
    invalid_count = 0

    for idx, smiles in enumerate(smiles_series):
        row = _compute_row_properties(smiles)
        is_invalid = row["mw"] is None
        if is_invalid:
            invalid_count += 1
            logger.warning(
                "Invalid SMILES at row %d: %r — all properties set to null",
                idx,
                smiles,
            )
        for col in _PROPERTY_COLS:
            col_data[col].append(row[col])

    if invalid_count == len(df):
        logger.warning(
            "All %d SMILES were invalid; all property columns will be null",
            len(df),
        )

    return df.with_columns(
        [
            pl.Series(col, col_data[col], dtype=pl.Float64)
            for col in _PROPERTY_COLS
        ]
    )


def assess_druglikeness(
    properties_df: pl.DataFrame,
) -> pl.DataFrame:
    """Evaluate drug-likeness rule sets and compute a property penalty.

    Expects the columns produced by :func:`calculate_properties` to be
    present.  Adds four boolean rule-pass columns and one float penalty
    column.

    Rule definitions:

    - **Lipinski Ro5** (``lipinski_pass``): MW ≤ 500, LogP ≤ 5, HBD ≤ 5,
      HBA ≤ 10.  *All four* must pass (stricter than Lipinski's original
      "one violation allowed" to avoid borderline compounds — use
      ``lipinski_violations`` from the design doc if you need the softer
      version).
    - **Veber** (``veber_pass``): rotatable_bonds ≤ 10 AND TPSA ≤ 140.
    - **Beyond-Ro5** (``bro5_pass``): MW ≤ 1000, LogP ≤ 10, HBD ≤ 5,
      HBA ≤ 15.  A safety net for larger drug-like molecules (macrocycles,
      PROTACs).
    - **Pfizer 3/75** (``pfizer_3_75_pass``): LogP < 3 AND TPSA > 75.
      True = *passes* the Pfizer filter (low toxicity risk); False = flag.

    Penalty tiers (bRo5 as primary, since DEL compounds are often outside
    Lipinski space):

    - ``bro5_pass is True`` → 1.0 (acceptable drug-like)
    - ``bro5_pass is False`` → 2.0 (outside extended drug-like space)
    - Any property null (invalid SMILES) → 1.5 (mild uncertainty penalty)

    Args:
        properties_df: DataFrame containing at minimum the columns produced
            by :func:`calculate_properties` (``mw``, ``logp``, ``hba``,
            ``hbd``, ``tpsa``, ``rotatable_bonds``).

    Returns:
        Copy of *properties_df* with boolean columns ``lipinski_pass``,
        ``veber_pass``, ``bro5_pass``, ``pfizer_3_75_pass`` and float column
        ``property_penalty`` appended.
    """
    required = {"mw", "logp", "hba", "hbd", "tpsa", "rotatable_bonds"}
    missing = required - set(properties_df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            "Run calculate_properties first."
        )

    df = properties_df.with_columns(
        [
            # Lipinski Ro5 — all four criteria must pass
            (
                (pl.col("mw") <= 500)
                & (pl.col("logp") <= 5)
                & (pl.col("hbd") <= 5)
                & (pl.col("hba") <= 10)
            ).alias("lipinski_pass"),
            # Veber oral-bioavailability filter
            (
                (pl.col("rotatable_bonds") <= 10) & (pl.col("tpsa") <= 140)
            ).alias("veber_pass"),
            # Beyond-Ro5 (macrocycles / PROTACs)
            (
                (pl.col("mw") <= 1000)
                & (pl.col("logp") <= 10)
                & (pl.col("hbd") <= 5)
                & (pl.col("hba") <= 15)
            ).alias("bro5_pass"),
            # Pfizer 3/75: True = passes (low toxicity risk)
            (
                (pl.col("logp") < 3) & (pl.col("tpsa") > 75)
            ).alias("pfizer_3_75_pass"),
        ]
    )

    # Property penalty based on bRo5 pass/fail; null mw → uncertain penalty
    df = df.with_columns(
        pl.when(pl.col("mw").is_null())
        .then(pl.lit(1.5))
        .when(pl.col("bro5_pass"))
        .then(pl.lit(1.0))
        .otherwise(pl.lit(2.0))
        .alias("property_penalty")
    )

    return df


def compute_properties_for_ranking(
    ranked_df: pl.DataFrame,
    smiles_col: str = "smiles",
    code_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Compute properties and return only code columns + ``property_penalty``.

    Convenience wrapper: runs :func:`calculate_properties` then
    :func:`assess_druglikeness`, then strips all columns except the code
    columns and ``property_penalty``.  The result is ready to pass directly
    as *properties_df* to
    :func:`~delexplore.analyse.rank.compute_composite_rank`.

    Args:
        ranked_df: DataFrame containing *smiles_col* and code columns.
        smiles_col: Name of the SMILES column.
        code_cols: Code columns to keep in the output (e.g.
            ``["code_1", "code_2"]``).  If ``None``, all columns whose name
            starts with ``"code_"`` are used automatically.

    Returns:
        DataFrame with columns ``[*code_cols, "property_penalty"]``.

    Raises:
        ImportError: If RDKit is not installed.
        ValueError: If *smiles_col* is absent or no code columns are found.
    """
    if smiles_col not in ranked_df.columns:
        raise ValueError(
            f"Column '{smiles_col}' not found in DataFrame. "
            f"Available columns: {ranked_df.columns}"
        )

    if code_cols is None:
        code_cols = [c for c in ranked_df.columns if c.startswith("code_")]

    if not code_cols:
        raise ValueError(
            "No code columns found. Either provide code_cols explicitly or "
            "ensure the DataFrame has columns named 'code_*'."
        )

    props = calculate_properties(ranked_df, smiles_col=smiles_col)
    props = assess_druglikeness(props)

    return props.select(code_cols + ["property_penalty"])
