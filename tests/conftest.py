"""Shared pytest fixtures for DELexplore tests.

Provides two sets of fixtures:
- Real example data: paths to the checked-in DELT-Hit output files.
- Synthetic data: fully deterministic 2-cycle library for unit testing.

Synthetic library layout
------------------------
  code_1: 10 building blocks (indices 0–9)
  code_2:  8 building blocks (indices 0–7)
  compounds: 10 × 8 = 80

Enrichment categories (all counts are deterministic — no random seed):
  True binders  → (code_1=2, code_2=3) and (code_1=5, code_2=6)
                  target counts: 100–500   |  blank counts: 1–5
  Bead binders  → any compound where code_1 == 7
                  all selections: 40–80
  Noise         → all remaining compounds
                  all selections: 0–20
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).parent / "data" / "examples"
SYNTHETIC_DIR = Path(__file__).parent / "data" / "synthetic"

# ---------------------------------------------------------------------------
# Synthetic library parameters
# ---------------------------------------------------------------------------

N_BB1 = 10
N_BB2 = 8
SELECTIONS = ["blank_1", "blank_2", "blank_3", "target_1", "target_2", "target_3"]

_TRUE_BINDERS: set[tuple[int, int]] = {(2, 3), (5, 6)}
_BEAD_BINDER_CODE1: set[int] = {7}

# Per-selection index used only to add a tiny amount of deterministic
# variation to noise counts (prevents perfectly identical selections).
_SEL_IDX: dict[str, int] = {s: i for i, s in enumerate(SELECTIONS)}


def _compound_count(code_1: int, code_2: int, selection: str) -> int:
    """Return a fully deterministic count for one compound in one selection."""
    sel_idx = _SEL_IDX[selection]
    is_target = selection.startswith("target")

    if (code_1, code_2) in _TRUE_BINDERS:
        if is_target:
            # 100–500; varies by position and selection replicate
            return 100 + (code_1 * 37 + code_2 * 17 + sel_idx * 53) % 401
        else:
            # 1–5 in blank
            return 1 + (code_1 + code_2 + sel_idx) % 5

    if code_1 in _BEAD_BINDER_CODE1:
        # 40–80; uniform across target/blank (non-specific)
        return 40 + (code_2 * 5 + sel_idx * 3) % 41

    # Noise: 0–20
    return (code_1 * 3 + code_2 * 7 + sel_idx * 11) % 21


def _build_selection_df(selection: str) -> pl.DataFrame:
    """Build one selection's counts DataFrame (unsorted then sorted desc)."""
    rows = []
    for c1, c2 in itertools.product(range(N_BB1), range(N_BB2)):
        count = _compound_count(c1, c2, selection)
        rows.append(
            {
                "code_1": c1,
                "code_2": c2,
                "count": count,
                "id": f"{c1}_{c2}",
            }
        )

    df = pl.DataFrame(
        rows,
        schema={
            "code_1": pl.Int64,
            "code_2": pl.Int64,
            "count": pl.Int64,
            "id": pl.Utf8,
        },
    ).sort("count", descending=True)

    return df


def _build_combined_df() -> pl.DataFrame:
    """Build the combined DataFrame as load_experiment would produce it."""
    frames = []
    for sel in SELECTIONS:
        sel_df = _build_selection_df(sel).with_columns(
            pl.lit(sel).alias("selection")
        )
        # selection column first — matches load_experiment column order
        sel_df = sel_df.select(
            ["selection", "code_1", "code_2", "count", "id"]
        )
        frames.append(sel_df)
    return pl.concat(frames)


def _build_config() -> dict[str, Any]:
    """Build a synthetic config dict that mirrors the real DELT-Hit YAML format."""
    selections: dict[str, Any] = {}
    barcodes_s0 = [
        "ACACAC", "ACAGCA", "ACATGT",  # blanks
        "ACGACG", "ACGCGA", "ACTAGC",  # targets
    ]
    barcodes_s1 = "TCGATA"
    beads_blank = "HisPURE Beads"
    beads_target = "Dynabeads SA C1"

    for i, sel in enumerate(SELECTIONS):
        is_target = sel.startswith("target")
        selections[sel] = {
            "operator": "Test User",
            "date": "2024-01-15",
            "group": "protein" if is_target else "no_protein",
            "target": "ProteinA" if is_target else float("nan"),
            "beads": beads_target if is_target else beads_blank,
            "protocol": "DECL_5W",
            "S0": barcodes_s0[i],
            "S1": barcodes_s1,
            "ids": [i, 0],
        }

    # Building block SMILES (simplified but chemically valid fragments)
    bb0_smiles = [
        "NC(CC1=CC=CC=C1)C(O)=O",    # 0 Phe-like
        "NC(CC1=CNC2=CC=CC=C12)C(O)=O",  # 1 Trp-like
        "NC(CCCCN)C(O)=O",           # 2 Lys-like  ← TRUE BINDER partner
        "NC(CC1=CC=C(O)C=C1)C(O)=O", # 3 Tyr-like
        "NC(CS)C(O)=O",              # 4 Cys-like
        "NC(CCC(O)=O)C(O)=O",        # 5 Glu-like  ← TRUE BINDER partner
        "NC(CO)C(O)=O",              # 6 Ser-like
        "NC(C)C(O)=O",               # 7 Ala-like  ← BEAD BINDER
        "NCC(O)=O",                  # 8 Gly-like
        "NC(CC1=CC=NC=C1)C(O)=O",    # 9 His-like
    ]
    bb1_smiles = [
        "O=CC1=CC=CC=C1",            # 0 benzaldehyde
        "O=CC1=CC=CO1",              # 1 furfural
        "O=CC1=CC=CN=C1",            # 2 nicotinaldehyde
        "O=CC1=CC=C(F)C=C1",         # 3 4-F-benzaldehyde  ← TRUE BINDER partner
        "O=CC1=CC=C(Cl)C=C1",        # 4 4-Cl-benzaldehyde
        "O=CC1=CC=C(OC)C=C1",        # 5 4-OMe-benzaldehyde
        "O=CC1=CC=C([N+](=O)[O-])C=C1",  # 6 4-NO2-benzaldehyde  ← TRUE BINDER
        "O=CC1CCCCC1",               # 7 cyclohexane carboxaldehyde
    ]

    config: dict[str, Any] = {
        "experiment": {
            "name": "synthetic_test",
            "fastq_path": "synthetic.fastq.gz",
            "save_dir": "experiments/synthetic",
            "num_cores": 4,
        },
        "selections": selections,
        "library": {
            "building_blocks": ["B0", "B1"],
            "B0": [
                {
                    "index": idx,
                    "smiles": smi,
                    "codon": f"CODON{idx:02d}A",
                    "reaction": "ABF",
                    "educt": "scaffold_1",
                    "product": "product_1",
                }
                for idx, smi in enumerate(bb0_smiles)
            ],
            "B1": [
                {
                    "index": idx,
                    "smiles": smi,
                    "codon": f"CODON{idx:02d}B",
                    "reaction": "CuAAC",
                    "educt": "product_1",
                    "product": "product_2",
                }
                for idx, smi in enumerate(bb1_smiles)
            ],
        },
    }
    return config


# ---------------------------------------------------------------------------
# Session-scoped fixture that writes synthetic files to disk once per run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_data_dir() -> Path:
    """Write all synthetic TSV counts files and config.yaml to SYNTHETIC_DIR.

    Creates:
      tests/data/synthetic/config.yaml
      tests/data/synthetic/{selection_name}/counts.txt   (for each selection)

    Returns the path to SYNTHETIC_DIR.
    """
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    # Write per-selection counts.txt files
    for sel in SELECTIONS:
        sel_dir = SYNTHETIC_DIR / sel
        sel_dir.mkdir(exist_ok=True)
        df = _build_selection_df(sel)
        # Write as TSV matching exact real format: tab-separated, with header
        df.write_csv(sel_dir / "counts.txt", separator="\t")

    # Write config.yaml
    config = _build_config()
    config_path = SYNTHETIC_DIR / "config.yaml"
    with config_path.open("w") as fh:
        yaml.dump(config, fh, default_flow_style=False, allow_unicode=True)

    return SYNTHETIC_DIR


# ---------------------------------------------------------------------------
# In-memory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_counts() -> pl.DataFrame:
    """Combined counts DataFrame across all 6 selections (as load_experiment produces).

    Columns: selection, code_1, code_2, count, id
    Sorted descending by count within each selection.

    Enrichment pattern:
      True binders  (2,3) and (5,6): high counts in target_*, low in blank_*
      Bead binders  code_1==7:       moderate counts in all selections
      Noise         everything else: 0–20 in all selections
    """
    return _build_combined_df()


@pytest.fixture(scope="session")
def synthetic_config() -> dict[str, Any]:
    """Config dict mirroring real DELT-Hit YAML structure.

    Selections: blank_1/2/3 (no_protein, HisPURE Beads),
                target_1/2/3 (protein, Dynabeads SA C1, target=ProteinA)
    Library: B0 (10 BBs with SMILES), B1 (8 BBs with SMILES)
    """
    return _build_config()


@pytest.fixture(scope="session")
def selection_metadata(synthetic_config) -> pl.DataFrame:
    """Metadata DataFrame extracted from synthetic_config via get_selection_metadata."""
    from delexplore.io.readers import get_selection_metadata

    return get_selection_metadata(synthetic_config)


# ---------------------------------------------------------------------------
# Real example data fixtures (unchanged)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    """Path to the examples directory containing real DELT-Hit output."""
    return EXAMPLES_DIR


@pytest.fixture(scope="session")
def example_counts_path() -> Path:
    """Path to the example counts.txt file."""
    return EXAMPLES_DIR / "counts.txt"


@pytest.fixture(scope="session")
def example_config_path() -> Path:
    """Path to the example config.yaml file."""
    return EXAMPLES_DIR / "config.yaml"


@pytest.fixture(scope="session")
def example_counts_df(example_counts_path):
    """Pre-loaded counts DataFrame from the example counts.txt."""
    from delexplore.io.readers import read_counts

    return read_counts(example_counts_path)


@pytest.fixture(scope="session")
def example_config(example_config_path):
    """Pre-loaded config dict from the example config.yaml."""
    from delexplore.io.readers import read_config

    return read_config(example_config_path)
