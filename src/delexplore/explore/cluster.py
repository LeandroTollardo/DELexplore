"""HDBSCAN clustering of UMAP-projected compounds.

Clusters compounds in UMAP space and computes per-cluster enrichment
statistics for SAR analysis.

Usage
-----
>>> from delexplore.explore.cluster import cluster_umap, cluster_enrichment_summary
>>> labels = cluster_umap(embedding)
>>> summary = cluster_enrichment_summary(df.with_columns(
...     pl.Series("cluster", labels)
... ))
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------

try:
    import hdbscan as _hdbscan_module

    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False
    logger.warning("hdbscan not installed — HDBSCAN clustering unavailable.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not installed — cluster plotting unavailable.")


def _require(flag: bool, pkg: str) -> None:
    if not flag:
        raise ImportError(
            f"{pkg} is required for this function. Install with: pip install {pkg}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cluster_umap(
    embedding: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5,
) -> np.ndarray:
    """Run HDBSCAN on a 2D UMAP embedding.

    Args:
        embedding: Array of shape ``(n_compounds, 2)`` — typically the output
            of :func:`~delexplore.explore.umap_viz.compute_umap_embedding`.
        min_cluster_size: Minimum number of points to form a cluster.
            Smaller values → more, smaller clusters.
        min_samples: Controls how conservative the clustering is.
            Higher values → fewer points classified as noise.

    Returns:
        Integer array of shape ``(n_compounds,)`` containing cluster labels.
        ``-1`` indicates noise (no cluster assignment).

    Raises:
        ImportError: If hdbscan is not installed.
        ValueError: If ``embedding`` is empty or not 2-D.
    """
    _require(_HDBSCAN_AVAILABLE, "hdbscan")
    if embedding.ndim != 2:
        raise ValueError(
            f"embedding must be 2-D, got shape {embedding.shape}"
        )
    if embedding.shape[0] == 0:
        raise ValueError("embedding array must not be empty")

    # min_cluster_size must be ≥ 2 and ≤ n_samples
    n_samples = embedding.shape[0]
    effective_min_cluster_size = max(2, min(min_cluster_size, n_samples))
    effective_min_samples = max(1, min(min_samples, effective_min_cluster_size))

    if effective_min_cluster_size < min_cluster_size:
        logger.warning(
            "min_cluster_size=%d reduced to %d (%d samples available)",
            min_cluster_size,
            effective_min_cluster_size,
            n_samples,
        )

    clusterer = _hdbscan_module.HDBSCAN(
        min_cluster_size=effective_min_cluster_size,
        min_samples=effective_min_samples,
    )
    labels: np.ndarray = clusterer.fit_predict(embedding)
    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    n_noise = int((labels == -1).sum())
    logger.info(
        "HDBSCAN found %d clusters; %d/%d points classified as noise",
        n_clusters,
        n_noise,
        n_samples,
    )
    return labels


def cluster_enrichment_summary(
    df: pl.DataFrame,
    cluster_col: str = "cluster",
    score_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Compute per-cluster enrichment statistics.

    For each cluster (including noise cluster ``-1``), summarises compound
    count and, for each *score_col*, the mean, median, and maximum value.
    Also computes the fraction of the dataset's top-100 compounds (by first
    score column, ascending) that fall into each cluster.

    Args:
        df: DataFrame containing *cluster_col* and at least the columns in
            *score_cols*.  Must have been annotated with cluster labels
            (e.g. by calling :func:`cluster_umap`).
        cluster_col: Name of the integer cluster-label column.
        score_cols: Columns to summarise per cluster.  If ``None``, defaults
            to ``["composite_score"]`` when that column is present, else no
            score summaries are added.

    Returns:
        One row per unique cluster label, sorted by cluster label ascending.
        Columns: ``cluster``, ``n_compounds``, and for each score column:
        ``{col}_mean``, ``{col}_median``, ``{col}_max``, plus
        ``n_in_top100``, ``frac_top100``.

    Raises:
        ValueError: If *cluster_col* is absent from *df*.
    """
    if cluster_col not in df.columns:
        raise ValueError(
            f"Column '{cluster_col}' not found. Available: {df.columns}"
        )

    if score_cols is None:
        score_cols = ["composite_score"] if "composite_score" in df.columns else []

    # Validate score columns
    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Score columns not found in df: {missing}")

    # --- Per-cluster aggregations ---
    agg_exprs: list[pl.Expr] = [pl.len().alias("n_compounds")]
    for col in score_cols:
        agg_exprs += [
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).median().alias(f"{col}_median"),
            pl.col(col).max().alias(f"{col}_max"),
        ]

    summary = (
        df.group_by(cluster_col)
        .agg(agg_exprs)
        .sort(cluster_col)
    )

    # --- Top-100 fraction ---
    # Determine how many rows are "top-100"
    top_k = min(100, len(df))
    if score_cols:
        primary_col = score_cols[0]
        top100_clusters = (
            df.sort(primary_col, nulls_last=True)
            .head(top_k)
            .get_column(cluster_col)
        )
        top100_counts = (
            top100_clusters.value_counts()
            .rename({"count": "n_in_top100"})
        )
        summary = summary.join(top100_counts, on=cluster_col, how="left")
        summary = summary.with_columns(
            pl.col("n_in_top100").fill_null(0),
        ).with_columns(
            (pl.col("n_in_top100") / top_k).alias("frac_top100"),
        )
    else:
        summary = summary.with_columns(
            pl.lit(0).alias("n_in_top100"),
            pl.lit(0.0).alias("frac_top100"),
        )

    return summary


def plot_clusters(
    embedding: np.ndarray,
    labels: np.ndarray,
    output_path: Path | str | None = None,
    figsize: tuple[int, int] = (10, 8),
    point_size: float = 10,
    alpha: float = 0.6,
    cmap: str = "tab20",
) -> None:
    """Scatter plot of UMAP embedding colored by HDBSCAN cluster label.

    Noise points (label ``-1``) are rendered in grey.  Each cluster gets a
    distinct color from *cmap*.

    Args:
        embedding: Array of shape ``(n_compounds, 2)``.
        labels: Integer cluster labels of shape ``(n_compounds,)``.  ``-1``
            means noise.
        output_path: If provided, saves the figure to this path.  Parent
            directories are created automatically.  Format inferred from
            the file extension.
        figsize: ``(width, height)`` in inches.
        point_size: Marker size in points².
        alpha: Transparency of scatter points (0–1).
        cmap: Matplotlib colormap for cluster colors.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    _require(_MPL_AVAILABLE, "matplotlib")

    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0

    colormap = matplotlib.colormaps.get_cmap(cmap).resampled(max(n_clusters, 1))

    # Plot noise first (underneath clusters)
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            embedding[noise_mask, 0],
            embedding[noise_mask, 1],
            c="lightgrey",
            s=point_size,
            alpha=alpha * 0.6,
            linewidths=0,
            label="noise",
            zorder=1,
        )

    for label in unique_labels:
        if label == -1:
            continue
        mask = labels == label
        color = colormap(label % colormap.N)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[color],
            s=point_size,
            alpha=alpha,
            linewidths=0,
            label=f"cluster {label}",
            zorder=2,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"HDBSCAN Clusters ({n_clusters} clusters)")

    if n_clusters <= 10:
        ax.legend(loc="best", framealpha=0.8, markerscale=2)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", dpi=150)
        logger.info("Cluster plot saved to %s", out)

    plt.close(fig)
