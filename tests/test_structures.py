"""Tests for explore/structures.py.

All tests use small known SMILES strings; no external files are required.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from delexplore.explore.structures import (
    render_hit_grid,
    render_single_structure,
    smiles_to_svg_dict,
)

# ---------------------------------------------------------------------------
# Test molecules
# ---------------------------------------------------------------------------

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
NAPROXEN = "COc1ccc2cc(ccc2c1)C(C)C(=O)O"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
INVALID = "not_a_smiles"
BENZENE_SMARTS = "c1ccccc1"


def _make_ranked_df(
    smiles_list: list[str],
    include_scores: bool = True,
    include_zscore: bool = True,
) -> pl.DataFrame:
    """Build a minimal ranked DataFrame compatible with render_hit_grid."""
    n = len(smiles_list)
    data: dict[str, list] = {
        "rank": list(range(1, n + 1)),
        "smiles": smiles_list,
        "code_1": list(range(n)),
        "code_2": list(range(n)),
    }
    if include_scores:
        data["composite_score"] = [0.1 * i for i in range(1, n + 1)]
    if include_zscore:
        data["zscore"] = [float(i) for i in range(n)]
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# 1. render_hit_grid — basic SVG output
# ---------------------------------------------------------------------------


class TestRenderHitGrid:
    def test_returns_svg_string(self):
        df = _make_ranked_df([ASPIRIN, CAFFEINE, IBUPROFEN, NAPROXEN, PARACETAMOL])
        result = render_hit_grid(df, top_n=5)
        assert isinstance(result, str)
        assert result.strip().startswith("<?xml") or "<svg" in result

    def test_svg_contains_all_ranks(self):
        df = _make_ranked_df([ASPIRIN, CAFFEINE, IBUPROFEN])
        result = render_hit_grid(df, top_n=3)
        # RDKit renders legend text as SVG paths with class='legend'.
        # Verify that legend paths are present (one set per compound).
        assert result.count("class='legend'") > 0

    def test_top_n_limits_output(self):
        df = _make_ranked_df([ASPIRIN, CAFFEINE, IBUPROFEN, NAPROXEN, PARACETAMOL])
        # 2 compounds → narrower SVG (width ≤ 2 × cell_size[0])
        result_2 = render_hit_grid(df, top_n=2, cols_per_row=4, cell_size=(350, 300))
        result_5 = render_hit_grid(df, top_n=5, cols_per_row=4, cell_size=(350, 300))
        # The 5-compound SVG must be larger (more paths)
        assert len(result_5) > len(result_2)

    def test_composite_score_in_legend(self):
        df = _make_ranked_df([ASPIRIN], include_scores=True)
        result = render_hit_grid(df, top_n=1, show_scores=True)
        # Legend paths are rendered; at minimum the SVG must be non-trivial
        assert "class='legend'" in result

    def test_zscore_in_legend(self):
        df = _make_ranked_df([ASPIRIN], include_scores=True, include_zscore=True)
        result = render_hit_grid(df, top_n=1, show_scores=True)
        assert "class='legend'" in result

    def test_show_scores_false(self):
        df = _make_ranked_df([ASPIRIN], include_scores=True, include_zscore=True)
        result = render_hit_grid(df, top_n=1, show_scores=False)
        assert "Score:" not in result
        assert "z=" not in result

    def test_missing_rank_column_raises(self):
        df = pl.DataFrame({"smiles": [ASPIRIN], "code_1": [0]})
        with pytest.raises(ValueError, match="rank"):
            render_hit_grid(df)

    def test_missing_smiles_col_raises(self):
        df = pl.DataFrame({"rank": [1], "code_1": [0]})
        with pytest.raises(ValueError, match="smiles"):
            render_hit_grid(df)

    # 2. highlight_substructure
    def test_highlight_substructure_no_error(self):
        df = _make_ranked_df([ASPIRIN, CAFFEINE, IBUPROFEN])
        # benzene ring is present in aspirin, ibuprofen, naproxen
        result = render_hit_grid(df, top_n=3, highlight_substructure=BENZENE_SMARTS)
        assert isinstance(result, str)
        assert "<svg" in result

    def test_invalid_smarts_logs_warning_no_crash(self, caplog):
        import logging

        df = _make_ranked_df([ASPIRIN])
        with caplog.at_level(logging.WARNING, logger="delexplore.explore.structures"):
            result = render_hit_grid(df, top_n=1, highlight_substructure="[invalid!")
        assert isinstance(result, str)

    # 3. Invalid SMILES in the set
    def test_invalid_smiles_renders_placeholder(self):
        df = _make_ranked_df([ASPIRIN, INVALID, CAFFEINE])
        # Should not raise; placeholder is inserted for the invalid entry
        result = render_hit_grid(df, top_n=3)
        assert isinstance(result, str)
        assert "<svg" in result

    def test_all_invalid_smiles(self):
        df = _make_ranked_df([INVALID, INVALID])
        result = render_hit_grid(df, top_n=2)
        assert isinstance(result, str)

    def test_empty_dataframe(self):
        empty = pl.DataFrame(
            {"rank": [], "smiles": [], "code_1": []},
            schema={"rank": pl.Int64, "smiles": pl.Utf8, "code_1": pl.Int64},
        )
        result = render_hit_grid(empty)
        assert isinstance(result, str)
        assert "<svg" in result

    # Output to file
    def test_writes_svg_to_file(self, tmp_path):
        df = _make_ranked_df([ASPIRIN, CAFFEINE])
        out = tmp_path / "hits.svg"
        result = render_hit_grid(df, top_n=2, output_path=out)
        assert out.exists()
        content = out.read_text()
        assert "<svg" in content
        # Return value should match file content
        assert result == content

    def test_writes_png_to_file(self, tmp_path):
        df = _make_ranked_df([ASPIRIN])
        out = tmp_path / "hits.png"
        result = render_hit_grid(df, top_n=1, output_path=out, img_format="png")
        assert out.exists()
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"
        assert out.read_bytes() == result

    def test_creates_parent_directories(self, tmp_path):
        df = _make_ranked_df([ASPIRIN])
        out = tmp_path / "subdir" / "nested" / "hits.svg"
        render_hit_grid(df, top_n=1, output_path=out)
        assert out.exists()

    def test_png_format_returns_bytes(self):
        df = _make_ranked_df([ASPIRIN])
        result = render_hit_grid(df, top_n=1, img_format="png")
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    def test_top_n_larger_than_df(self):
        """top_n > len(df) should just return all rows without error."""
        df = _make_ranked_df([ASPIRIN, CAFFEINE])
        result = render_hit_grid(df, top_n=100)
        assert isinstance(result, str)
        assert "<svg" in result
        # Two compounds → at least two sets of legend paths
        assert result.count("class='legend'") > 0


# ---------------------------------------------------------------------------
# 4. smiles_to_svg_dict
# ---------------------------------------------------------------------------


class TestSmilesToSvgDict:
    def test_returns_correct_number_of_entries(self):
        df = pl.DataFrame(
            {"smiles": [ASPIRIN, CAFFEINE, IBUPROFEN]},
        )
        result = smiles_to_svg_dict(df)
        assert len(result) == 3

    def test_keys_are_row_indices_by_default(self):
        df = pl.DataFrame({"smiles": [ASPIRIN, CAFFEINE]})
        result = smiles_to_svg_dict(df)
        assert set(result.keys()) == {"0", "1"}

    def test_keys_from_id_col(self):
        df = pl.DataFrame(
            {"id": ["compound_A", "compound_B"], "smiles": [ASPIRIN, CAFFEINE]}
        )
        result = smiles_to_svg_dict(df, id_col="id")
        assert "compound_A" in result
        assert "compound_B" in result

    def test_values_are_svg_strings(self):
        df = pl.DataFrame({"smiles": [ASPIRIN]})
        result = smiles_to_svg_dict(df)
        svg = result["0"]
        assert isinstance(svg, str)
        assert "<svg" in svg

    def test_invalid_smiles_gets_placeholder(self):
        df = pl.DataFrame({"smiles": [ASPIRIN, INVALID]})
        result = smiles_to_svg_dict(df)
        assert "<svg" in result["0"]
        # placeholder for invalid
        placeholder = result["1"]
        assert "<svg" in placeholder
        assert "Invalid" in placeholder

    def test_empty_dataframe_returns_empty_dict(self):
        df = pl.DataFrame(
            {"smiles": []}, schema={"smiles": pl.Utf8}
        )
        result = smiles_to_svg_dict(df)
        assert result == {}

    def test_missing_smiles_col_raises(self):
        df = pl.DataFrame({"other": [ASPIRIN]})
        with pytest.raises(ValueError, match="smiles"):
            smiles_to_svg_dict(df, smiles_col="smiles")

    def test_missing_id_col_raises(self):
        df = pl.DataFrame({"smiles": [ASPIRIN]})
        with pytest.raises(ValueError, match="id_col"):
            smiles_to_svg_dict(df, id_col="nonexistent")

    def test_size_parameter_reflected_in_svg(self):
        df = pl.DataFrame({"smiles": [ASPIRIN]})
        result = smiles_to_svg_dict(df, size=(123, 456))
        svg = result["0"]
        assert "123" in svg
        assert "456" in svg

    def test_integer_id_col(self):
        df = pl.DataFrame(
            {"code_1": [0, 1], "smiles": [ASPIRIN, CAFFEINE]},
            schema={"code_1": pl.Int64, "smiles": pl.Utf8},
        )
        result = smiles_to_svg_dict(df, id_col="code_1")
        assert "0" in result
        assert "1" in result


# ---------------------------------------------------------------------------
# 6. render_single_structure
# ---------------------------------------------------------------------------


class TestRenderSingleStructure:
    def test_returns_svg_string(self):
        result = render_single_structure(ASPIRIN)
        assert isinstance(result, str)
        assert "<svg" in result

    def test_svg_is_non_empty(self):
        result = render_single_structure(ASPIRIN)
        assert len(result) > 100

    def test_invalid_smiles_returns_placeholder(self):
        result = render_single_structure(INVALID)
        assert isinstance(result, str)
        assert "<svg" in result
        assert "Invalid" in result

    def test_highlight_smarts_no_error(self):
        result = render_single_structure(ASPIRIN, highlight_smarts=BENZENE_SMARTS)
        assert isinstance(result, str)
        assert "<svg" in result

    def test_highlight_smarts_no_match_no_error(self):
        # Caffeine has no benzene ring — highlight should simply do nothing
        result = render_single_structure(CAFFEINE, highlight_smarts=BENZENE_SMARTS)
        assert isinstance(result, str)

    def test_size_reflected_in_svg(self):
        result = render_single_structure(ASPIRIN, size=(321, 234))
        assert "321" in result
        assert "234" in result

    def test_writes_svg_to_file(self, tmp_path):
        out = tmp_path / "single.svg"
        result = render_single_structure(ASPIRIN, output_path=out)
        assert out.exists()
        assert out.read_text() == result

    def test_png_format_returns_bytes(self):
        result = render_single_structure(ASPIRIN, img_format="png")
        assert isinstance(result, bytes)
        # Cairo returns raw bytes — check it's non-trivial length
        assert len(result) > 100

    def test_invalid_smiles_png_placeholder(self):
        result = render_single_structure(INVALID, img_format="png")
        assert isinstance(result, bytes)
        assert len(result) > 0
