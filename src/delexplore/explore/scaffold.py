"""Murcko scaffold analysis of enriched hit compounds.

Groups enriched compounds by Murcko scaffold to identify scaffold families
that concentrate binding activity for medicinal chemistry follow-up.

Usage
-----
>>> from delexplore.explore.scaffold import (
...     compute_murcko_scaffolds,
...     scaffold_enrichment_analysis,
... )
>>> scaffolds = compute_murcko_scaffolds(df["smiles"].to_list())
>>> summary = scaffold_enrichment_analysis(df)
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.rdBase import BlockLogs

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    logger.warning("RDKit not installed — scaffold analysis unavailable.")


def _require(flag: bool, pkg: str) -> None:
    if not flag:
        raise ImportError(
            f"{pkg} is required for this function. Install with: pip install {pkg}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_murcko_scaffolds(
    smiles_list: list[str],
) -> list[str | None]:
    """Compute Murcko scaffolds for a list of SMILES strings.

    Uses RDKit's ``MurckoScaffold.GetScaffoldForMol`` to extract the Bemis-
    Murcko scaffold from each molecule.  Molecules with no ring systems
    (acyclic compounds) return an empty string ``""`` (the RDKit convention
    for a scaffold of a linear molecule).

    Args:
        smiles_list: List of SMILES strings.  None and empty strings are
            treated as invalid.

    Returns:
        List of scaffold SMILES strings of the same length as *smiles_list*.
        ``None`` for entries where the SMILES could not be parsed.

    Raises:
        ImportError: If RDKit is not installed.
        ValueError: If *smiles_list* is empty.
    """
    _require(_RDKIT_AVAILABLE, "rdkit")
    if not smiles_list:
        raise ValueError("smiles_list must not be empty")

    results: list[str | None] = []
    for i, smiles in enumerate(smiles_list):
        if not smiles:
            results.append(None)
            continue
        with BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(
                "Invalid SMILES at index %d: %r — scaffold set to None", i, smiles
            )
            results.append(None)
            continue
        try:
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            results.append(Chem.MolToSmiles(scaffold_mol))
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Scaffold computation failed for index %d: %s", i, exc
            )
            results.append(None)

    return results


def scaffold_enrichment_analysis(
    df: pl.DataFrame,
    smiles_col: str = "smiles",
    score_col: str = "composite_score",
    top_n_scaffolds: int = 20,
) -> pl.DataFrame:
    """Group compounds by Murcko scaffold and summarise enrichment.

    For each unique Murcko scaffold, computes:

    - ``n_compounds``: total compounds sharing this scaffold.
    - ``mean_score``: mean of *score_col* across scaffold members.
    - ``best_rank``: lowest rank value among scaffold members (if a ``rank``
      column is present in *df*, else minimum row index).
    - ``n_in_top100``: number of scaffold members that appear in the top 100
      rows when *df* is sorted by *score_col* ascending.

    Compounds with no valid scaffold (invalid SMILES, acyclic molecules) are
    grouped under the scaffold label ``"[no scaffold]"``.

    Args:
        df: Input DataFrame.  Must contain *smiles_col* and, if *score_col*
            summaries are requested, *score_col*.
        smiles_col: Column holding SMILES strings.
        score_col: Column used for ranking and mean-score computation.  If
            absent, mean_score and n_in_top100 are set to ``null``.
        top_n_scaffolds: Number of scaffolds to return, sorted by mean_score
            ascending (best first).

    Returns:
        DataFrame with columns ``scaffold``, ``n_compounds``, ``mean_score``,
        ``best_rank``, ``n_in_top100``, truncated to *top_n_scaffolds* rows.

    Raises:
        ImportError: If RDKit is not installed.
        ValueError: If *smiles_col* is absent from *df*.
    """
    _require(_RDKIT_AVAILABLE, "rdkit")
    if smiles_col not in df.columns:
        raise ValueError(
            f"Column '{smiles_col}' not found. Available: {df.columns}"
        )

    smiles_list = df[smiles_col].to_list()
    scaffolds_raw = compute_murcko_scaffolds(smiles_list)

    # Treat acyclic compounds (empty scaffold) and None as "[no scaffold]"
    scaffold_labels = [
        s if (s is not None and s != "") else "[no scaffold]"
        for s in scaffolds_raw
    ]

    df = df.with_columns(
        pl.Series("scaffold", scaffold_labels, dtype=pl.Utf8)
    )

    has_score = score_col in df.columns
    has_rank = "rank" in df.columns

    # Top-100 membership: row indices (0-based) of the top-100 compounds
    top_k = min(100, len(df))
    if has_score:
        top100_scaffolds = (
            df.sort(score_col, nulls_last=True)
            .head(top_k)
            .get_column("scaffold")
        )
        top100_counts = (
            top100_scaffolds.value_counts()
            .rename({"count": "n_in_top100"})
        )
    else:
        top100_counts = None

    # Per-scaffold aggregation
    agg_exprs: list[pl.Expr] = [pl.len().alias("n_compounds")]
    if has_score:
        agg_exprs.append(pl.col(score_col).mean().alias("mean_score"))
    else:
        agg_exprs.append(pl.lit(None).cast(pl.Float64).alias("mean_score"))

    if has_rank:
        agg_exprs.append(pl.col("rank").min().alias("best_rank"))
    else:
        # Use row index: assign a temporary row index column
        df = df.with_row_index("_row_idx")
        agg_exprs.append(pl.col("_row_idx").min().alias("best_rank"))

    summary = df.group_by("scaffold").agg(agg_exprs)

    # Join top-100 counts
    if top100_counts is not None:
        summary = summary.join(top100_counts, on="scaffold", how="left").with_columns(
            pl.col("n_in_top100").fill_null(0)
        )
    else:
        summary = summary.with_columns(pl.lit(0).alias("n_in_top100"))

    # Sort by mean_score ascending (best first), then by n_compounds descending
    sort_cols = ["mean_score", "n_compounds"]
    sort_desc = [False, True]
    summary = summary.sort(sort_cols, descending=sort_desc, nulls_last=True)

    # Drop temporary column if added
    if "_row_idx" in summary.columns:
        summary = summary.drop("_row_idx")

    return summary.head(top_n_scaffolds)
