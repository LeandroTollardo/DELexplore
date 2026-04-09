"""Tests for explore/cluster.py."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from delexplore.explore.cluster import (
    cluster_enrichment_summary,
    cluster_umap,
    plot_clusters,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _two_blobs(n: int = 60) -> np.ndarray:
    """Return a (n, 2) embedding with two well-separated blobs."""
    half = n // 2
    blob_a = RNG.normal(loc=[0.0, 0.0], scale=0.3, size=(half, 2))
    blob_b = RNG.normal(loc=[5.0, 5.0], scale=0.3, size=(n - half, 2))
    return np.vstack([blob_a, blob_b])


def _make_df(labels: list[int], scores: list[float] | None = None) -> pl.DataFrame:
    n = len(labels)
    data: dict = {
        "cluster": labels,
        "rank": list(range(1, n + 1)),
        "code_1": list(range(n)),
    }
    if scores is not None:
        data["composite_score"] = scores
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# 1. cluster_umap
# ---------------------------------------------------------------------------


class TestClusterUmap:
    def test_output_shape(self):
        emb = _two_blobs(60)
        labels = cluster_umap(emb, min_cluster_size=5, min_samples=2)
        assert labels.shape == (60,)

    def test_output_dtype(self):
        emb = _two_blobs(40)
        labels = cluster_umap(emb, min_cluster_size=4, min_samples=2)
        assert np.issubdtype(labels.dtype, np.integer)

    def test_finds_two_clusters(self):
        emb = _two_blobs(60)
        labels = cluster_umap(emb, min_cluster_size=5, min_samples=2)
        # Should find 2 clusters (labels 0 and 1) for well-separated blobs
        unique_labels = set(labels.tolist())
        non_noise = unique_labels - {-1}
        assert len(non_noise) == 2

    def test_noise_label_is_minus_one(self):
        emb = _two_blobs(60)
        labels = cluster_umap(emb, min_cluster_size=5, min_samples=2)
        # -1 is the only special value; all others ≥ 0
        assert all(l >= -1 for l in labels.tolist())

    def test_empty_embedding_raises(self):
        with pytest.raises(ValueError, match="empty"):
            cluster_umap(np.zeros((0, 2)))

    def test_1d_embedding_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            cluster_umap(np.zeros((10,)))

    def test_small_dataset_auto_reduces_params(self):
        """min_cluster_size > n_samples → auto-reduced without error."""
        emb = RNG.normal(size=(5, 2))
        labels = cluster_umap(emb, min_cluster_size=100, min_samples=50)
        assert labels.shape == (5,)

    def test_single_dense_blob_returns_labels(self):
        """A sufficiently spread blob with small min_cluster_size → a cluster."""
        emb = RNG.normal(loc=[0, 0], scale=1.0, size=(50, 2))
        labels = cluster_umap(emb, min_cluster_size=3, min_samples=1)
        assert labels.shape == (50,)
        # With loose params, at least some points form a cluster
        assert (labels >= 0).any()


# ---------------------------------------------------------------------------
# 2. cluster_enrichment_summary
# ---------------------------------------------------------------------------


class TestClusterEnrichmentSummary:
    def test_returns_one_row_per_cluster(self):
        labels = [0, 0, 0, 1, 1, -1]
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.9]
        df = _make_df(labels, scores)
        summary = cluster_enrichment_summary(df, score_cols=["composite_score"])
        assert len(summary) == 3  # clusters -1, 0, 1

    def test_n_compounds_correct(self):
        labels = [0, 0, 1, 1, 1]
        df = _make_df(labels, [0.1, 0.2, 0.3, 0.4, 0.5])
        summary = cluster_enrichment_summary(df, score_cols=["composite_score"])
        row_c0 = summary.filter(pl.col("cluster") == 0)
        row_c1 = summary.filter(pl.col("cluster") == 1)
        assert row_c0["n_compounds"][0] == 2
        assert row_c1["n_compounds"][0] == 3

    def test_mean_score_correct(self):
        labels = [0, 0, 1]
        scores = [0.2, 0.4, 0.9]
        df = _make_df(labels, scores)
        summary = cluster_enrichment_summary(df, score_cols=["composite_score"])
        mean_c0 = summary.filter(pl.col("cluster") == 0)["composite_score_mean"][0]
        assert abs(mean_c0 - 0.3) < 1e-9

    def test_sorted_by_cluster_label(self):
        labels = [1, 0, -1, 0]
        df = _make_df(labels, [0.4, 0.1, 0.9, 0.2])
        summary = cluster_enrichment_summary(df, score_cols=["composite_score"])
        assert summary["cluster"].to_list() == [-1, 0, 1]

    def test_n_in_top100_present(self):
        labels = [0] * 5 + [1] * 5
        scores = list(range(10))  # 0-4 → cluster 0, 5-9 → cluster 1
        df = _make_df(labels, scores)
        summary = cluster_enrichment_summary(df, score_cols=["composite_score"])
        assert "n_in_top100" in summary.columns
        assert "frac_top100" in summary.columns

    def test_frac_top100_sums_to_one(self):
        labels = [0] * 50 + [1] * 50
        scores = list(range(100))
        df = _make_df(labels, scores)
        summary = cluster_enrichment_summary(df, score_cols=["composite_score"])
        total_frac = summary["frac_top100"].sum()
        assert abs(total_frac - 1.0) < 1e-9

    def test_no_score_cols(self):
        """When score_cols=[], summary still has n_compounds."""
        labels = [0, 0, 1]
        df = _make_df(labels)
        summary = cluster_enrichment_summary(df, score_cols=[])
        assert "n_compounds" in summary.columns

    def test_missing_cluster_col_raises(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="cluster"):
            cluster_enrichment_summary(df)

    def test_missing_score_col_raises(self):
        labels = [0, 1]
        df = _make_df(labels)
        with pytest.raises(ValueError, match="not found"):
            cluster_enrichment_summary(df, score_cols=["nonexistent"])

    def test_default_score_col_composite_score(self):
        """Default score_cols should auto-detect composite_score."""
        labels = [0, 0, 1]
        df = _make_df(labels, [0.1, 0.2, 0.9])
        summary = cluster_enrichment_summary(df)  # default score_cols=None
        assert "composite_score_mean" in summary.columns

    def test_no_composite_score_falls_back_gracefully(self):
        """When composite_score absent and score_cols=None → no crash."""
        df = pl.DataFrame({"cluster": [0, 1], "rank": [1, 2]})
        summary = cluster_enrichment_summary(df)
        assert "n_compounds" in summary.columns


# ---------------------------------------------------------------------------
# 3. plot_clusters — smoke tests
# ---------------------------------------------------------------------------


class TestPlotClusters:
    @pytest.fixture(scope="class")
    def emb_labels(self):
        emb = _two_blobs(40)
        labels = cluster_umap(emb, min_cluster_size=4, min_samples=2)
        return emb, labels

    def test_no_error_basic(self, emb_labels):
        emb, labels = emb_labels
        plot_clusters(emb, labels)

    def test_writes_svg_file(self, emb_labels, tmp_path):
        emb, labels = emb_labels
        out = tmp_path / "clusters.svg"
        plot_clusters(emb, labels, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_writes_png_file(self, emb_labels, tmp_path):
        emb, labels = emb_labels
        out = tmp_path / "clusters.png"
        plot_clusters(emb, labels, output_path=out)
        assert out.exists()
        assert out.read_bytes()[:4] == b"\x89PNG"

    def test_creates_parent_directories(self, emb_labels, tmp_path):
        emb, labels = emb_labels
        out = tmp_path / "subdir" / "nested" / "clusters.svg"
        plot_clusters(emb, labels, output_path=out)
        assert out.exists()

    def test_all_noise_no_crash(self):
        emb = RNG.normal(size=(10, 2))
        labels = np.full(10, -1)
        plot_clusters(emb, labels)
