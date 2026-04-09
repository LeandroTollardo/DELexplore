"""Tests for explore/umap_viz.py.

All tests use small in-memory SMILES sets; no files are required unless
testing output_dir/output_path behaviour.

UMAP is non-deterministic in shape (n, 2) but the embedding values change
with random_state — we only check shapes and structural properties.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from delexplore.explore.umap_viz import (
    compute_fingerprints,
    compute_umap_embedding,
    plot_umap,
    run_umap_pipeline,
)

# ---------------------------------------------------------------------------
# Molecule constants
# ---------------------------------------------------------------------------

ASPIRIN    = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE   = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
IBUPROFEN  = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
NAPROXEN   = "COc1ccc2cc(ccc2c1)C(C)C(=O)O"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
INVALID    = "not_a_smiles"

FIVE_VALID = [ASPIRIN, CAFFEINE, IBUPROFEN, NAPROXEN, PARACETAMOL]
N_BITS = 2048


# ---------------------------------------------------------------------------
# 1. compute_fingerprints
# ---------------------------------------------------------------------------

class TestComputeFingerprints:
    def test_shape_all_valid(self):
        fps, mask = compute_fingerprints(FIVE_VALID, n_bits=N_BITS)
        assert fps.shape == (5, N_BITS)
        assert mask == [True] * 5

    def test_dtype_is_uint8(self):
        fps, _ = compute_fingerprints(FIVE_VALID)
        assert fps.dtype == np.uint8

    def test_values_are_binary(self):
        fps, _ = compute_fingerprints(FIVE_VALID)
        assert set(np.unique(fps)).issubset({0, 1})

    def test_different_molecules_different_fps(self):
        fps, _ = compute_fingerprints([ASPIRIN, CAFFEINE])
        # Aspirin and caffeine have different fingerprints
        assert not np.array_equal(fps[0], fps[1])

    def test_same_molecule_same_fp(self):
        fps, _ = compute_fingerprints([ASPIRIN, ASPIRIN])
        np.testing.assert_array_equal(fps[0], fps[1])

    # 2. One invalid SMILES → valid_mask correct
    def test_one_invalid_smiles_valid_mask(self):
        fps, mask = compute_fingerprints([ASPIRIN, INVALID, CAFFEINE])
        assert fps.shape == (3, N_BITS)
        assert mask == [True, False, True]

    def test_invalid_row_is_all_zeros(self):
        fps, mask = compute_fingerprints([ASPIRIN, INVALID])
        assert not mask[1]
        np.testing.assert_array_equal(fps[1], np.zeros(N_BITS, dtype=np.uint8))

    def test_all_invalid_smiles(self):
        fps, mask = compute_fingerprints([INVALID, INVALID])
        assert mask == [False, False]
        np.testing.assert_array_equal(fps, np.zeros((2, N_BITS), dtype=np.uint8))

    def test_empty_smiles_string_is_invalid(self):
        _, mask = compute_fingerprints([""])
        assert mask == [False]

    def test_none_in_list_is_invalid(self):
        # None is a valid list entry; should be treated as invalid
        _, mask = compute_fingerprints([None])  # type: ignore[list-item]
        assert mask == [False]

    def test_custom_n_bits(self):
        fps, _ = compute_fingerprints([ASPIRIN], n_bits=512)
        assert fps.shape == (1, 512)

    def test_custom_radius(self):
        fps_r2, _ = compute_fingerprints([ASPIRIN], radius=2)
        fps_r3, _ = compute_fingerprints([ASPIRIN], radius=3)
        # Different radii → different bit patterns
        assert not np.array_equal(fps_r2[0], fps_r3[0])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_fingerprints([])


# ---------------------------------------------------------------------------
# 3. compute_umap_embedding
# ---------------------------------------------------------------------------

class TestComputeUmapEmbedding:
    @pytest.fixture(scope="class")
    def small_fps(self):
        fps, _ = compute_fingerprints(FIVE_VALID)
        return fps

    def test_output_shape(self, small_fps):
        emb = compute_umap_embedding(small_fps, n_neighbors=3)
        assert emb.shape == (5, 2)

    def test_output_dtype_is_float(self, small_fps):
        emb = compute_umap_embedding(small_fps, n_neighbors=3)
        assert emb.dtype in (np.float32, np.float64)

    def test_no_nan_in_embedding(self, small_fps):
        emb = compute_umap_embedding(small_fps, n_neighbors=3)
        assert not np.any(np.isnan(emb))

    def test_reproducible_with_same_seed(self, small_fps):
        emb1 = compute_umap_embedding(small_fps, n_neighbors=3, random_state=42)
        emb2 = compute_umap_embedding(small_fps, n_neighbors=3, random_state=42)
        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_n_neighbors_auto_reduced_for_small_input(self):
        """When n_samples < n_neighbors, n_neighbors is reduced without error."""
        fps, _ = compute_fingerprints([ASPIRIN, CAFFEINE, IBUPROFEN])
        # n_neighbors=15 > 3 samples → must be reduced automatically
        emb = compute_umap_embedding(fps, n_neighbors=15)
        assert emb.shape == (3, 2)

    def test_empty_fingerprints_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_umap_embedding(np.zeros((0, N_BITS)))

    def test_larger_dataset(self):
        """20-compound dataset → embedding shape (20, 2)."""
        smiles = (FIVE_VALID * 4)
        fps, _ = compute_fingerprints(smiles)
        emb = compute_umap_embedding(fps, n_neighbors=5)
        assert emb.shape == (20, 2)


# ---------------------------------------------------------------------------
# 4. plot_umap — smoke tests (just check it runs without error)
# ---------------------------------------------------------------------------

class TestPlotUmap:
    @pytest.fixture(scope="class")
    def embedding(self):
        fps, _ = compute_fingerprints(FIVE_VALID)
        return compute_umap_embedding(fps, n_neighbors=3)

    def test_basic_call_no_error(self, embedding):
        plot_umap(embedding)

    def test_with_color_values(self, embedding):
        colors = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        plot_umap(embedding, color_values=colors, color_label="z-score")

    def test_with_highlight_indices(self, embedding):
        plot_umap(embedding, highlight_indices=[0, 2], highlight_label="top 2")

    def test_writes_svg_file(self, embedding, tmp_path):
        out = tmp_path / "umap.svg"
        plot_umap(embedding, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_writes_png_file(self, embedding, tmp_path):
        out = tmp_path / "umap.png"
        plot_umap(embedding, output_path=out)
        assert out.exists()
        assert out.read_bytes()[:4] == b"\x89PNG"

    def test_creates_parent_directories(self, embedding, tmp_path):
        out = tmp_path / "subdir" / "nested" / "umap.svg"
        plot_umap(embedding, output_path=out)
        assert out.exists()

    def test_no_color_no_highlight_no_crash(self, embedding):
        plot_umap(embedding, color_values=None, highlight_indices=None)

    def test_all_options_combined(self, embedding, tmp_path):
        out = tmp_path / "full.svg"
        colors = np.linspace(0, 1, 5)
        plot_umap(
            embedding,
            color_values=colors,
            color_label="score",
            highlight_indices=[1, 3],
            highlight_label="hits",
            output_path=out,
            figsize=(8, 6),
            cmap="plasma",
            alpha=0.8,
            point_size=20,
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# 5. run_umap_pipeline — end-to-end
# ---------------------------------------------------------------------------

def _make_df(smiles_list: list[str], with_score: bool = True) -> pl.DataFrame:
    n = len(smiles_list)
    data: dict = {
        "code_1": list(range(n)),
        "smiles": smiles_list,
    }
    if with_score:
        data["composite_score"] = [0.1 * (i + 1) for i in range(n)]
    return pl.DataFrame(data)


class TestRunUmapPipeline:
    def test_adds_umap_columns(self):
        df = _make_df(FIVE_VALID)
        result = run_umap_pipeline(df, smiles_col="smiles", color_col="composite_score")
        assert "umap_x" in result.columns
        assert "umap_y" in result.columns

    def test_output_length_unchanged(self):
        df = _make_df(FIVE_VALID)
        result = run_umap_pipeline(df)
        assert len(result) == len(df)

    def test_umap_columns_are_float64(self):
        df = _make_df(FIVE_VALID)
        result = run_umap_pipeline(df)
        assert result["umap_x"].dtype == pl.Float64
        assert result["umap_y"].dtype == pl.Float64

    def test_all_valid_smiles_no_nulls(self):
        df = _make_df(FIVE_VALID)
        result = run_umap_pipeline(df)
        assert result["umap_x"].null_count() == 0
        assert result["umap_y"].null_count() == 0

    def test_invalid_smiles_gets_null_coordinates(self):
        smiles = [ASPIRIN, INVALID, CAFFEINE, IBUPROFEN, NAPROXEN]
        df = _make_df(smiles)
        result = run_umap_pipeline(df)
        # Row 1 (INVALID) must be null
        assert result["umap_x"][1] is None
        assert result["umap_y"][1] is None
        # Other rows must be non-null
        assert result["umap_x"][0] is not None

    def test_no_color_col_no_crash(self):
        df = _make_df(FIVE_VALID, with_score=False)
        result = run_umap_pipeline(df, color_col=None)
        assert "umap_x" in result.columns

    def test_color_col_not_in_df_no_crash(self):
        df = _make_df(FIVE_VALID, with_score=False)
        result = run_umap_pipeline(df, color_col="nonexistent_col")
        assert "umap_x" in result.columns

    def test_output_dir_writes_parquet_and_svg(self, tmp_path):
        df = _make_df(FIVE_VALID)
        run_umap_pipeline(df, output_dir=tmp_path)
        assert (tmp_path / "umap_embedding.parquet").exists()
        assert (tmp_path / "umap_plot.svg").exists()

    def test_parquet_output_has_umap_columns(self, tmp_path):
        df = _make_df(FIVE_VALID)
        run_umap_pipeline(df, output_dir=tmp_path)
        saved = pl.read_parquet(tmp_path / "umap_embedding.parquet")
        assert "umap_x" in saved.columns
        assert "umap_y" in saved.columns

    def test_missing_smiles_col_raises(self):
        df = pl.DataFrame({"code_1": [0], "not_smiles": [ASPIRIN]})
        with pytest.raises(ValueError, match="not found"):
            run_umap_pipeline(df, smiles_col="smiles")

    def test_empty_df_raises(self):
        df = pl.DataFrame(
            {"smiles": []}, schema={"smiles": pl.Utf8}
        )
        with pytest.raises(ValueError, match="empty"):
            run_umap_pipeline(df)

    def test_all_invalid_smiles_raises(self):
        df = _make_df([INVALID, INVALID, INVALID])
        with pytest.raises(ValueError, match="[Aa]ll SMILES"):
            run_umap_pipeline(df)

    def test_umap_kwargs_forwarded(self):
        """Custom n_neighbors / min_dist reach compute_umap_embedding."""
        df = _make_df(FIVE_VALID)
        # n_neighbors=2 is valid for 5 samples; just check it doesn't error
        result = run_umap_pipeline(
            df, umap_kwargs={"n_neighbors": 2, "min_dist": 0.05}
        )
        assert result["umap_x"].null_count() == 0

    def test_top_n_highlight_zero_no_crash(self):
        df = _make_df(FIVE_VALID)
        result = run_umap_pipeline(df, top_n_highlight=0)
        assert "umap_x" in result.columns

    def test_original_columns_preserved(self):
        df = _make_df(FIVE_VALID)
        result = run_umap_pipeline(df)
        assert "code_1" in result.columns
        assert "smiles" in result.columns
        assert "composite_score" in result.columns

    def test_larger_dataset_with_output(self, tmp_path):
        """20 compounds → correct embedding shape in saved parquet."""
        smiles = FIVE_VALID * 4
        df = _make_df(smiles)
        run_umap_pipeline(df, output_dir=tmp_path, umap_kwargs={"n_neighbors": 5})
        saved = pl.read_parquet(tmp_path / "umap_embedding.parquet")
        assert len(saved) == 20
        assert saved["umap_x"].null_count() == 0
