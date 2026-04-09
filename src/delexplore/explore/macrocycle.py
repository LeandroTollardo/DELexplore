"""Macrocycle detection and 3D conformational characterization.

A macrocycle is defined as any ring with ≥ 12 atoms.  Macrocycles present
unique drug-discovery challenges: they often violate Lipinski Ro5 but can
achieve cell permeability via conformational switching, and their 3D shape
matters more than their 2D topology.

Usage
-----
>>> from delexplore.explore.macrocycle import (
...     detect_macrocycles,
...     calculate_macrocycle_descriptors,
...     add_macrocycle_columns,
...     assess_macrocycle_druglikeness,
... )
>>> rings = detect_macrocycles("CCC1OC(=O)...")  # erythromycin → [14]
>>> desc = calculate_macrocycle_descriptors("CCC1OC(=O)...")
>>> df = add_macrocycle_columns(df, smiles_col="smiles")
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

# Minimum ring size to qualify as a macrocycle
_MACROCYCLE_MIN_SIZE = 12

# 3D conformer sampling parameters
_NUM_CONFORMERS = 50
_ENERGY_WINDOW_KCAL = 3.0  # conformers within this of the minimum are counted

# ---------------------------------------------------------------------------
# RDKit import (gated — optional dependency)
# ---------------------------------------------------------------------------

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolTransforms
    from rdkit.rdBase import BlockLogs

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    logger.warning(
        "RDKit is not installed. Macrocycle detection will not be available. "
        "Install with: pip install rdkit"
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_rdkit() -> None:
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for macrocycle analysis. "
            "Install with: pip install rdkit"
        )


def _parse_mol(smiles: str | None):
    """Return an RDKit Mol (with Hs added), or None if invalid/missing."""
    if not smiles:
        return None
    with BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    return mol


def _get_ring_sizes(mol) -> list[int]:
    """Return all ring sizes in *mol* using the SSSR."""
    sssr = Chem.GetSymmSSSR(mol)
    return [len(ring) for ring in sssr]


def _embed_3d(mol):
    """Add explicit Hs and embed one ETKDGv3 conformer.

    Returns the 3D Mol, or None if embedding fails.
    """
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol_h, params)
    if result == -1:
        return None
    return mol_h


def _compute_globularity(mol_3d) -> float | None:
    """Compute globularity from the lowest-energy conformer's PMI.

    Globularity = I1/I3 where I1 ≤ I2 ≤ I3 are the principal moments of
    inertia.  Range: (0, 1].  1.0 = perfectly spherical; ~0 = rod-like.

    Args:
        mol_3d: Mol with at least one conformer and explicit Hs.

    Returns:
        Globularity float, or None if computation fails.
    """
    try:
        conf = mol_3d.GetConformer(0)
        _axes, moments = rdMolTransforms.ComputePrincipalAxesAndMoments(
            conf, ignoreHs=True
        )
        if moments[2] == 0.0:
            return None
        return float(moments[0] / moments[2])
    except Exception as exc:
        logger.debug("PMI computation failed: %s", exc)
        return None


def _count_low_energy_conformers(mol, smiles: str) -> int | None:
    """Generate 50 ETKDGv3 conformers and count those within the energy window.

    Args:
        mol: Mol object (no explicit Hs required; they will be added).
        smiles: Original SMILES string (used only for logging).

    Returns:
        Count of conformers within ``_ENERGY_WINDOW_KCAL`` of the minimum
        MMFF energy, or None if conformer generation or force-field setup
        fails.
    """
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42

    conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=_NUM_CONFORMERS, params=params)
    if len(conf_ids) == 0:
        logger.warning(
            "Failed to embed any conformers for macrocycle: %.60s", smiles
        )
        return None

    mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h)
    if mmff_props is None:
        logger.warning(
            "MMFF properties unavailable for macrocycle: %.60s", smiles
        )
        return None

    energies: list[float] = []
    for cid in conf_ids:
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, mmff_props, confId=cid)
        if ff is None:
            continue
        try:
            ff.Minimize()
            energies.append(ff.CalcEnergy())
        except Exception:
            continue

    if not energies:
        return None

    min_e = min(energies)
    return sum(1 for e in energies if e - min_e <= _ENERGY_WINDOW_KCAL)


# ---------------------------------------------------------------------------
# Public API — per-molecule functions
# ---------------------------------------------------------------------------


def detect_macrocycles(smiles: str) -> list[int]:
    """Return ring sizes for all macrocyclic rings in a molecule.

    A macrocyclic ring is defined as any ring with ≥ 12 atoms (IUPAC
    convention).  Ring membership is determined by the Smallest Set of
    Smallest Rings (SSSR) via ``Chem.GetSymmSSSR``.

    Args:
        smiles: A SMILES string.

    Returns:
        List of ring sizes (integers) for rings with ≥ 12 atoms, in the
        order they appear in the SSSR.  Empty list if the molecule has no
        macrocyclic rings or the SMILES is invalid/empty.

    Raises:
        ImportError: If RDKit is not installed.
    """
    _require_rdkit()
    mol = _parse_mol(smiles)
    if mol is None:
        return []
    return [s for s in _get_ring_sizes(mol) if s >= _MACROCYCLE_MIN_SIZE]


def calculate_macrocycle_descriptors(smiles: str) -> dict[str, Any]:
    """Compute macrocycle topology and 3D shape descriptors for one molecule.

    Always computes topology descriptors (fast).  The expensive 3D descriptors
    (``globularity``, ``num_conformers``) are computed only when
    ``is_macrocycle`` is True.

    Args:
        smiles: A SMILES string.

    Returns:
        Dict with the following keys:

        Topology (always present):
            - ``is_macrocycle`` (bool): True if any ring ≥ 12 atoms.
            - ``largest_ring_size`` (int): Size of the largest macrocyclic
              ring; 0 if no macrocycle.
            - ``num_macrocyclic_rings`` (int): Count of rings ≥ 12 atoms.
            - ``fraction_sp3`` (float | None): Fraction of sp3 carbons;
              None for invalid SMILES.

        3D (None for non-macrocycles or when embedding fails):
            - ``globularity`` (float | None): I1/I3 from PMI on the lowest-
              energy conformer.  Closer to 1.0 = more spherical.
            - ``num_conformers`` (int | None): Number of MMFF-minimized ETKDGv3
              conformers within 3 kcal/mol of the minimum energy.

    Raises:
        ImportError: If RDKit is not installed.
    """
    _require_rdkit()

    result: dict[str, Any] = {
        "is_macrocycle": False,
        "largest_ring_size": 0,
        "num_macrocyclic_rings": 0,
        "fraction_sp3": None,
        "globularity": None,
        "num_conformers": None,
    }

    mol = _parse_mol(smiles)
    if mol is None:
        logger.warning("Invalid SMILES for macrocycle descriptor: %.60s", smiles)
        return result

    with BlockLogs():
        result["fraction_sp3"] = float(Descriptors.FractionCSP3(mol))

    macro_rings = detect_macrocycles(smiles)
    result["num_macrocyclic_rings"] = len(macro_rings)
    result["largest_ring_size"] = max(macro_rings, default=0)
    result["is_macrocycle"] = len(macro_rings) > 0

    if not result["is_macrocycle"]:
        return result

    # --- 3D descriptors (macrocycles only) ---
    mol_3d = _embed_3d(mol)
    if mol_3d is None:
        logger.warning(
            "3D embedding failed for macrocycle (%.60s); "
            "globularity and num_conformers set to null",
            smiles,
        )
        return result

    result["globularity"] = _compute_globularity(mol_3d)
    result["num_conformers"] = _count_low_energy_conformers(mol, smiles)

    return result


# ---------------------------------------------------------------------------
# Public API — DataFrame-level functions
# ---------------------------------------------------------------------------


def add_macrocycle_columns(
    df: pl.DataFrame,
    smiles_col: str = "smiles",
) -> pl.DataFrame:
    """Add macrocycle descriptor columns to a DataFrame.

    Computes topology descriptors for every row; runs the expensive 3D
    calculations only for rows where ``is_macrocycle`` is True.  Non-macrocycle
    rows receive null for ``globularity``.

    Args:
        df: Input DataFrame.  Must contain *smiles_col*.
        smiles_col: Name of the column holding SMILES strings.

    Returns:
        Copy of *df* with the following columns appended:

        - ``is_macrocycle`` (Boolean): True if the compound contains a ring
          with ≥ 12 atoms.
        - ``largest_ring_size`` (Int64): Size of the largest macrocyclic ring;
          0 for non-macrocycles.
        - ``globularity`` (Float64): I1/I3 PMI ratio; null for non-macrocycles
          or when 3D embedding fails.

    Raises:
        ImportError: If RDKit is not installed.
        ValueError: If *smiles_col* is absent from *df*.
    """
    _require_rdkit()
    if smiles_col not in df.columns:
        raise ValueError(
            f"Column '{smiles_col}' not found in DataFrame. "
            f"Available columns: {df.columns}"
        )

    if len(df) == 0:
        return df.with_columns([
            pl.lit(None).cast(pl.Boolean).alias("is_macrocycle"),
            pl.lit(None).cast(pl.Int64).alias("largest_ring_size"),
            pl.lit(None).cast(pl.Float64).alias("globularity"),
        ])

    is_macro: list[bool] = []
    largest: list[int] = []
    globularity: list[float | None] = []

    for smiles in df[smiles_col].to_list():
        macro_rings = detect_macrocycles(smiles) if smiles else []
        is_m = len(macro_rings) > 0
        is_macro.append(is_m)
        largest.append(max(macro_rings, default=0))

        if is_m:
            mol = _parse_mol(smiles)
            mol_3d = _embed_3d(mol) if mol is not None else None
            g = _compute_globularity(mol_3d) if mol_3d is not None else None
            globularity.append(g)
        else:
            globularity.append(None)

    return df.with_columns([
        pl.Series("is_macrocycle", is_macro, dtype=pl.Boolean),
        pl.Series("largest_ring_size", largest, dtype=pl.Int64),
        pl.Series("globularity", globularity, dtype=pl.Float64),
    ])


def assess_macrocycle_druglikeness(
    properties_df: pl.DataFrame,
) -> pl.DataFrame:
    """Classify macrocycle oral exposure and permeability potential.

    Operates on rows where ``is_macrocycle`` is True; non-macrocycle rows
    receive null for both classification columns.

    Required columns (from :func:`add_macrocycle_columns` and
    :func:`~delexplore.explore.properties.calculate_properties`):
        ``is_macrocycle``, ``mw``, ``tpsa``, ``hbd``, ``num_conformers``.

    Classification rules:

    **Oral exposure** (``macro_oral_class``):
        - ``"likely"``     if MW < 1000 AND TPSA < 250 AND HBD ≤ 5
        - ``"uncertain"``  if MW < 1200 AND TPSA < 300
        - ``"unlikely"``   otherwise

    **Permeability** (``macro_permeability_class``):
        - ``"high"``   if num_conformers ≥ 10 AND TPSA < 200
        - ``"medium"`` if num_conformers ≥ 5
        - ``"low"``    otherwise

    Args:
        properties_df: DataFrame with all required columns.

    Returns:
        Copy of *properties_df* with ``macro_oral_class`` (Utf8) and
        ``macro_permeability_class`` (Utf8) columns appended.  Both are null
        for non-macrocycle rows.

    Raises:
        ValueError: If any required column is absent.
    """
    required = {"is_macrocycle", "mw", "tpsa", "hbd", "num_conformers"}
    missing = required - set(properties_df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            "Run add_macrocycle_columns and calculate_properties first."
        )

    # --- Oral exposure class ---
    oral = (
        pl.when(pl.col("is_macrocycle").not_())
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(
            (pl.col("mw") < 1000)
            & (pl.col("tpsa") < 250)
            & (pl.col("hbd") <= 5)
        )
        .then(pl.lit("likely"))
        .when(
            (pl.col("mw") < 1200) & (pl.col("tpsa") < 300)
        )
        .then(pl.lit("uncertain"))
        .otherwise(pl.lit("unlikely"))
        .alias("macro_oral_class")
    )

    # --- Permeability class ---
    permeability = (
        pl.when(pl.col("is_macrocycle").not_())
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(
            (pl.col("num_conformers") >= 10) & (pl.col("tpsa") < 200)
        )
        .then(pl.lit("high"))
        .when(pl.col("num_conformers") >= 5)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("low"))
        .alias("macro_permeability_class")
    )

    return properties_df.with_columns([oral, permeability])
