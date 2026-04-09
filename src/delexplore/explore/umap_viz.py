"""UMAP projections of compound chemical space.

Projects DEL hit compounds into 2D space using Morgan fingerprints (radius=2,
nBits=2048) with Jaccard distance and UMAP dimensionality reduction, then
visualizes the projection colored by enrichment score.

Usage
-----
>>> from delexplore.explore.umap_viz import run_umap_pipeline
>>> result_df = run_umap_pipeline(ranked_df, smiles_col="smiles",
...                                color_col="composite_score",
...                                output_dir=Path("results/umap"))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.rdBase import BlockLogs

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    logger.warning("RDKit not installed — fingerprint computation unavailable.")

try:
    import umap as _umap_module

    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False
    logger.warning("umap-learn not installed — UMAP embedding unavailable.")

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; safe in all environments
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not installed — UMAP plotting unavailable.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require(flag: bool, pkg: str) -> None:
    if not flag:
        raise ImportError(
            f"{pkg} is required for this function. Install with: pip install {pkg}"
        )


def _parse_mol(smiles: str | None):
    if not smiles:
        return None
    with BlockLogs():
        return Chem.MolFromSmiles(smiles)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[np.ndarray, list[bool]]:
    """Convert SMILES strings to Morgan fingerprint bit vectors.

    Uses RDKit's ``rdFingerprintGenerator.GetMorganGenerator`` to compute
    circular (Morgan) fingerprints as binary bit vectors.

    Args:
        smiles_list: List of SMILES strings.  None and empty strings are
            treated as invalid.
        radius: Morgan radius (default 2 = ECFP4 analogue).
        n_bits: Fingerprint bit vector length.

    Returns:
        Tuple ``(fingerprint_matrix, valid_mask)`` where:

        - ``fingerprint_matrix``: ``np.ndarray`` of shape
          ``(len(smiles_list), n_bits)``, dtype ``uint8``.  Rows for invalid
          SMILES are set to all-zeros.
        - ``valid_mask``: ``list[bool]`` of length ``len(smiles_list)``.
          ``valid_mask[i]`` is ``True`` if ``smiles_list[i]`` was parseable.

    Raises:
        ImportError: If RDKit is not installed.
        ValueError: If ``smiles_list`` is empty.
    """
    _require(_RDKIT_AVAILABLE, "rdkit")
    if not smiles_list:
        raise ValueError("smiles_list must not be empty")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    n = len(smiles_list)
    matrix = np.zeros((n, n_bits), dtype=np.uint8)
    valid_mask: list[bool] = []

    for i, smiles in enumerate(smiles_list):
        mol = _parse_mol(smiles)
        if mol is None:
            logger.warning(
                "Invalid SMILES at index %d: %r — fingerprint set to zeros", i, smiles
            )
            valid_mask.append(False)
        else:
            matrix[i] = gen.GetFingerprintAsNumPy(mol)
            valid_mask.append(True)

    n_invalid = valid_mask.count(False)
    if n_invalid > 0:
        logger.info(
            "%d / %d SMILES were invalid and will appear as zero vectors", n_invalid, n
        )

    return matrix, valid_mask


def compute_umap_embedding(
    fingerprints: np.ndarray,
    metric: str = "jaccard",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Run UMAP dimensionality reduction on a fingerprint matrix.

    When the number of samples is smaller than ``n_neighbors``, ``n_neighbors``
    is automatically reduced to ``n_samples - 1`` so UMAP does not fail on
    small inputs.

    Args:
        fingerprints: Float/int array of shape ``(n_compounds, n_bits)``.
            Typically the output of :func:`compute_fingerprints`.
        metric: Distance metric for UMAP.  ``"jaccard"`` is recommended for
            binary fingerprints; it measures chemical dissimilarity correctly.
        n_neighbors: Controls the size of the local neighborhood considered
            by UMAP.  Larger values → more global structure preserved.
        min_dist: Minimum distance between embedded points.  Smaller values
            produce tighter clusters.
        random_state: Random seed for reproducibility.

    Returns:
        ``np.ndarray`` of shape ``(n_compounds, 2)``.

    Raises:
        ImportError: If umap-learn is not installed.
        ValueError: If ``fingerprints`` is empty.
    """
    _require(_UMAP_AVAILABLE, "umap-learn")
    if fingerprints.shape[0] == 0:
        raise ValueError("fingerprints array must not be empty")

    n_samples = fingerprints.shape[0]
    effective_n_neighbors = min(n_neighbors, n_samples - 1)
    if effective_n_neighbors < n_neighbors:
        logger.warning(
            "n_neighbors=%d reduced to %d (only %d samples available)",
            n_neighbors, effective_n_neighbors, n_samples,
        )
    # UMAP requires at least 2 neighbors
    effective_n_neighbors = max(effective_n_neighbors, 2)

    # UMAP's spectral layout requires n_samples > n_components + 1 = 3.
    # For very small inputs fall back to random initialisation to avoid
    # the scipy eigsh constraint k >= N.
    n_components = 2
    init = "spectral" if n_samples > n_components + 1 else "random"

    reducer = _umap_module.UMAP(
        n_components=n_components,
        metric=metric,
        n_neighbors=effective_n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        init=init,
    )
    embedding: np.ndarray = reducer.fit_transform(fingerprints.astype(float))
    return embedding


def plot_umap(
    embedding: np.ndarray,
    color_values: np.ndarray | None = None,
    color_label: str = "enrichment",
    highlight_indices: list[int] | None = None,
    highlight_label: str = "top hits",
    output_path: Path | str | None = None,
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "viridis",
    alpha: float = 0.6,
    point_size: float = 10,
) -> None:
    """Create a UMAP scatter plot, optionally colored by enrichment score.

    Args:
        embedding: Array of shape ``(n_compounds, 2)`` from
            :func:`compute_umap_embedding`.
        color_values: Optional 1-D array of length ``n_compounds``.  When
            provided, points are colored using *cmap* mapped to these values.
            When ``None``, points are rendered in a uniform grey.
        color_label: Label for the colorbar (ignored when *color_values* is
            None).
        highlight_indices: Optional list of row indices to mark with a red
            border (e.g. the top-N hits).
        highlight_label: Legend label for the highlighted points.
        output_path: If provided, the figure is saved to this path.  The
            file format is inferred from the extension (SVG, PNG, PDF…).
            Parent directories are created automatically.
        figsize: ``(width, height)`` in inches.
        cmap: Matplotlib colormap name for *color_values*.
        alpha: Transparency of the scatter points (0–1).
        point_size: Marker size in points².

    Raises:
        ImportError: If matplotlib is not installed.
    """
    _require(_MPL_AVAILABLE, "matplotlib")

    fig, ax = plt.subplots(figsize=figsize)

    x = embedding[:, 0]
    y = embedding[:, 1]

    if color_values is not None:
        sc = ax.scatter(
            x, y,
            c=color_values,
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            linewidths=0,
        )
        plt.colorbar(sc, ax=ax, label=color_label)
    else:
        ax.scatter(x, y, color="steelblue", s=point_size, alpha=alpha, linewidths=0)

    if highlight_indices:
        hx = embedding[highlight_indices, 0]
        hy = embedding[highlight_indices, 1]
        ax.scatter(
            hx, hy,
            facecolors="none",
            edgecolors="red",
            s=point_size * 4,
            linewidths=1.2,
            label=highlight_label,
            zorder=5,
        )
        ax.legend(loc="best", framealpha=0.8)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Chemical Space (UMAP)")

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        logger.info("UMAP plot saved to %s", out)

    plt.close(fig)


def run_umap_pipeline(
    df: pl.DataFrame,
    smiles_col: str = "smiles",
    color_col: str | None = "composite_score",
    top_n_highlight: int = 20,
    output_dir: Path | str | None = None,
    umap_kwargs: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Full pipeline: fingerprints → UMAP → plot → annotated DataFrame.

    For each compound in *df*, computes a Morgan fingerprint, projects all
    compounds into 2D with UMAP, and (optionally) creates a scatter plot
    colored by *color_col*.  Invalid SMILES are kept in the DataFrame but
    receive ``null`` for ``umap_x`` / ``umap_y``.

    Args:
        df: Input DataFrame.  Must contain *smiles_col*.
        smiles_col: Column holding SMILES strings.
        color_col: Column to use for colormap in the plot.  If ``None`` or
            not present in *df*, the plot is rendered without color.
        top_n_highlight: Number of top compounds to highlight (by ascending
            order of *color_col* if *color_col* names a score column where
            lower = better, e.g. ``composite_score``; else by descending
            order).  Set to 0 to disable highlighting.
        output_dir: If provided, writes ``umap_embedding.parquet`` and
            ``umap_plot.svg`` to this directory.
        umap_kwargs: Optional keyword arguments forwarded to
            :func:`compute_umap_embedding` (e.g.
            ``{"n_neighbors": 30, "min_dist": 0.05}``).

    Returns:
        Copy of *df* with two Float64 columns appended:

        - ``umap_x``: First UMAP dimension.  ``null`` for invalid SMILES.
        - ``umap_y``: Second UMAP dimension.  ``null`` for invalid SMILES.

    Raises:
        ImportError: If RDKit or umap-learn is not installed.
        ValueError: If *smiles_col* is absent, *df* is empty, or all SMILES
            are invalid.
    """
    _require(_RDKIT_AVAILABLE, "rdkit")
    _require(_UMAP_AVAILABLE, "umap-learn")

    if smiles_col not in df.columns:
        raise ValueError(
            f"Column '{smiles_col}' not found. Available: {df.columns}"
        )
    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    smiles_list = df[smiles_col].to_list()
    fps, valid_mask = compute_fingerprints(smiles_list)

    n_valid = sum(valid_mask)
    if n_valid == 0:
        raise ValueError("All SMILES were invalid — cannot compute UMAP embedding")
    if n_valid < len(df):
        logger.warning(
            "%d compounds have invalid SMILES and will receive null coordinates",
            len(df) - n_valid,
        )

    # Run UMAP on valid fingerprints only
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    valid_fps = fps[valid_indices]

    kwargs = umap_kwargs or {}
    embedding = compute_umap_embedding(valid_fps, **kwargs)

    # Place coordinates back into full-length arrays (null for invalid rows)
    umap_x: list[float | None] = [None] * len(df)
    umap_y: list[float | None] = [None] * len(df)
    for embed_idx, df_idx in enumerate(valid_indices):
        umap_x[df_idx] = float(embedding[embed_idx, 0])
        umap_y[df_idx] = float(embedding[embed_idx, 1])

    result = df.with_columns([
        pl.Series("umap_x", umap_x, dtype=pl.Float64),
        pl.Series("umap_y", umap_y, dtype=pl.Float64),
    ])

    # --- Plot ---
    color_values: np.ndarray | None = None
    color_label = color_col or "enrichment"
    if color_col is not None and color_col in df.columns:
        raw = df[color_col].to_numpy(allow_copy=True).astype(float)
        # Use only the values for valid compounds (same order as embedding)
        color_values = raw[valid_indices]

    # Determine highlight indices within the valid-compound embedding
    highlight_embed_indices: list[int] | None = None
    if top_n_highlight > 0 and color_col is not None and color_col in df.columns:
        # Map valid df rows to their position in the embedding array
        df_to_embed = {df_idx: embed_idx for embed_idx, df_idx in enumerate(valid_indices)}

        # Get scores for valid compounds sorted by score ascending
        valid_scores = [(df.row(i, named=True).get(color_col, None), i) for i in valid_indices]
        valid_scores_clean = [(s, i) for s, i in valid_scores if s is not None]
        valid_scores_clean.sort(key=lambda t: t[0])
        top_df_indices = [i for _, i in valid_scores_clean[:top_n_highlight]]
        highlight_embed_indices = [df_to_embed[i] for i in top_df_indices if i in df_to_embed]

    embed_for_plot = embedding  # (n_valid, 2)
    plot_umap(
        embed_for_plot,
        color_values=color_values,
        color_label=color_label,
        highlight_indices=highlight_embed_indices,
        output_path=Path(output_dir) / "umap_plot.svg" if output_dir else None,
    )

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result.write_parquet(out_dir / "umap_embedding.parquet")
        logger.info("UMAP embedding written to %s", out_dir / "umap_embedding.parquet")

    return result
