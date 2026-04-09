"""Tests for explore/dashboard.py."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from delexplore.explore.dashboard import generate_dashboard

# ---------------------------------------------------------------------------
# Minimal synthetic data
# ---------------------------------------------------------------------------

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def _minimal_df(n: int = 5) -> pl.DataFrame:
    """Ranked DataFrame with code columns and composite_score but no SMILES."""
    return pl.DataFrame({
        "rank":            list(range(1, n + 1)),
        "code_1":          list(range(n)),
        "code_2":          list(range(n)),
        "composite_score": [0.1 * i for i in range(1, n + 1)],
        "zscore":          [float(n - i) for i in range(n)],
        "support_score":   [float(i % 3) for i in range(n)],
    })


def _smiles_df(n: int = 3) -> pl.DataFrame:
    smiles_list = [ASPIRIN, CAFFEINE, IBUPROFEN][:n]
    return pl.DataFrame({
        "rank":            list(range(1, n + 1)),
        "code_1":          list(range(n)),
        "code_2":          list(range(n)),
        "smiles":          smiles_list,
        "composite_score": [0.1 * i for i in range(1, n + 1)],
    })


def _properties_df(n: int = 3) -> pl.DataFrame:
    return pl.DataFrame({
        "code_1":         list(range(n)),
        "code_2":         list(range(n)),
        "mw":             [180.2, 194.2, 206.3][:n],
        "logp":           [1.2, -0.1, 3.5][:n],
        "tpsa":           [63.6, 58.4, 37.3][:n],
        "qed":            [0.55, 0.28, 0.73][:n],
        "lipinski_pass":  [True, True, True][:n],
        "bro5_pass":      [True, False, True][:n],
    })


# ---------------------------------------------------------------------------
# 1. Minimal dashboard → valid HTML
# ---------------------------------------------------------------------------


class TestMinimalDashboard:
    def test_returns_path(self, tmp_path: Path) -> None:
        df = _minimal_df()
        out = tmp_path / "dashboard.html"
        result = generate_dashboard(df, output_path=out, experiment_name="Test")
        assert result == out

    def test_file_created(self, tmp_path: Path) -> None:
        df = _minimal_df()
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_html_structure(self, tmp_path: Path) -> None:
        """Output must be parseable HTML with expected landmarks."""
        df = _minimal_df()
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, output_path=out)
        html = out.read_text(encoding="utf-8")

        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<body" in html
        assert "</body>" in html

    def test_experiment_name_in_title(self, tmp_path: Path) -> None:
        df = _minimal_df()
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, output_path=out, experiment_name="MyExperiment")
        html = out.read_text(encoding="utf-8")
        assert "MyExperiment" in html

    def test_rank_values_present(self, tmp_path: Path) -> None:
        df = _minimal_df(5)
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, output_path=out)
        html = out.read_text(encoding="utf-8")
        # Rank 1 through 5 should appear somewhere in the table
        for r in range(1, 6):
            assert str(r) in html

    def test_no_rank_col_raises(self, tmp_path: Path) -> None:
        df = pl.DataFrame({"code_1": [0, 1], "composite_score": [0.1, 0.2]})
        with pytest.raises(ValueError, match="rank"):
            generate_dashboard(df, output_path=tmp_path / "x.html")

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        df = _minimal_df()
        out = tmp_path / "nested" / "subdir" / "dashboard.html"
        generate_dashboard(df, output_path=out)
        assert out.exists()

    def test_top_n_limits_rows(self, tmp_path: Path) -> None:
        df = _minimal_df(20)
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, top_n=5, output_path=out)
        html = out.read_text(encoding="utf-8")
        # Rank 6 should not appear in a table row (only rank 1-5 are included)
        # Count occurrences of rank-badge content
        badges = re.findall(r'class="rank-badge[^"]*">\s*(\d+)\s*<', html)
        badge_ints = [int(b) for b in badges]
        assert max(badge_ints) <= 5

    def test_score_histogram_section_present(self, tmp_path: Path) -> None:
        df = _minimal_df()
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, output_path=out)
        html = out.read_text(encoding="utf-8")
        assert "hist-bar" in html

    def test_generated_timestamp_present(self, tmp_path: Path) -> None:
        df = _minimal_df()
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, output_path=out)
        html = out.read_text(encoding="utf-8")
        # Template embeds "Generated at" or timestamp in footer
        assert "Generated by DELexplore" in html

    def test_sortable_js_included(self, tmp_path: Path) -> None:
        df = _minimal_df()
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, output_path=out)
        html = out.read_text(encoding="utf-8")
        assert "sortTable" in html

    def test_search_js_included(self, tmp_path: Path) -> None:
        df = _minimal_df()
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, output_path=out)
        html = out.read_text(encoding="utf-8")
        assert "filterTable" in html

    def test_no_external_css_or_js(self, tmp_path: Path) -> None:
        """Dashboard must be self-contained — no external resource links."""
        df = _minimal_df()
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, output_path=out)
        html = out.read_text(encoding="utf-8")
        # No <link href="http..."> or <script src="http...">
        assert not re.search(r'<link[^>]+href=["\']https?://', html)
        assert not re.search(r'<script[^>]+src=["\']https?://', html)


# ---------------------------------------------------------------------------
# 2. SMILES → inline SVG structures
# ---------------------------------------------------------------------------


class TestSmilesDashboard:
    def test_structure_column_present_when_smiles(self, tmp_path: Path) -> None:
        df = _smiles_df()
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, smiles_col="smiles", output_path=out)
        html = out.read_text(encoding="utf-8")
        assert "structure-cell" in html

    def test_svg_embedded_for_valid_smiles(self, tmp_path: Path) -> None:
        df = _smiles_df()
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, smiles_col="smiles", output_path=out)
        html = out.read_text(encoding="utf-8")
        # At least one <svg> block should be present
        assert "<svg" in html

    def test_no_structure_col_when_smiles_missing(self, tmp_path: Path) -> None:
        """DataFrame without smiles column → no structure table column header."""
        df = _minimal_df()
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, smiles_col="smiles", output_path=out)
        html = out.read_text(encoding="utf-8")
        # The <th>Structure</th> header should not appear when there are no structures
        assert "<th>Structure</th>" not in html

    def test_invalid_smiles_does_not_crash(self, tmp_path: Path) -> None:
        df = pl.DataFrame({
            "rank":            [1, 2],
            "code_1":          [0, 1],
            "smiles":          ["invalid_smiles", ASPIRIN],
            "composite_score": [0.1, 0.2],
        })
        out = tmp_path / "dashboard.html"
        # Must not raise
        generate_dashboard(df, smiles_col="smiles", output_path=out)
        assert out.exists()

    def test_none_smiles_does_not_crash(self, tmp_path: Path) -> None:
        df = pl.DataFrame({
            "rank":            [1, 2],
            "code_1":          [0, 1],
            "smiles":          [None, ASPIRIN],
            "composite_score": [0.1, 0.2],
        })
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, smiles_col="smiles", output_path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# 3. Properties → drug-likeness section
# ---------------------------------------------------------------------------


class TestPropertiesDashboard:
    def test_druglikeness_section_when_properties_provided(
        self, tmp_path: Path
    ) -> None:
        df = _smiles_df()
        props = _properties_df()
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, properties_df=props, output_path=out)
        html = out.read_text(encoding="utf-8")
        assert "Drug-Likeness" in html

    def test_no_druglikeness_section_without_properties(
        self, tmp_path: Path
    ) -> None:
        df = _minimal_df()
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, output_path=out)
        html = out.read_text(encoding="utf-8")
        assert "Drug-Likeness" not in html

    def test_lipinski_pass_shown(self, tmp_path: Path) -> None:
        df = _smiles_df()
        props = _properties_df()
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, properties_df=props, output_path=out)
        html = out.read_text(encoding="utf-8")
        assert "Lipinski" in html

    def test_mw_histogram_present(self, tmp_path: Path) -> None:
        df = _smiles_df()
        props = _properties_df()
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, properties_df=props, output_path=out)
        html = out.read_text(encoding="utf-8")
        assert "MW" in html

    def test_summary_card_lipinski_pct(self, tmp_path: Path) -> None:
        """100% lipinski pass → '100%' appears in the summary."""
        df = _smiles_df()
        props = _properties_df()
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, properties_df=props, output_path=out)
        html = out.read_text(encoding="utf-8")
        assert "100%" in html


# ---------------------------------------------------------------------------
# 4. UMAP embedding → UMAP section
# ---------------------------------------------------------------------------


class TestUmapDashboard:
    def test_umap_section_when_embedding_provided(self, tmp_path: Path) -> None:
        df = _minimal_df(5)
        emb = np.random.default_rng(0).normal(size=(5, 2))
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, umap_embedding=emb, output_path=out)
        html = out.read_text(encoding="utf-8")
        assert "Chemical Space" in html

    def test_no_umap_section_without_embedding(self, tmp_path: Path) -> None:
        df = _minimal_df(5)
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, output_path=out)
        html = out.read_text(encoding="utf-8")
        assert "Chemical Space" not in html

    def test_umap_embedded_as_base64_img(self, tmp_path: Path) -> None:
        df = _minimal_df(5)
        emb = np.random.default_rng(0).normal(size=(5, 2))
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, umap_embedding=emb, output_path=out)
        html = out.read_text(encoding="utf-8")
        assert "data:image/svg+xml;base64," in html

    def test_wrong_size_embedding_does_not_crash(self, tmp_path: Path) -> None:
        """Embedding with mismatched rows → UMAP omitted, no crash."""
        df = _minimal_df(5)
        emb = np.random.default_rng(0).normal(size=(3, 2))  # wrong n
        out = tmp_path / "dashboard.html"
        generate_dashboard(df, umap_embedding=emb, output_path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# 5. Combined: all features together
# ---------------------------------------------------------------------------


class TestFullDashboard:
    def test_all_sections_present(self, tmp_path: Path) -> None:
        df = _smiles_df(3)
        emb = np.random.default_rng(0).normal(size=(3, 2))
        props = _properties_df(3)
        out = tmp_path / "dashboard.html"
        generate_dashboard(
            df,
            smiles_col="smiles",
            top_n=3,
            umap_embedding=emb,
            properties_df=props,
            output_path=out,
            experiment_name="Full Test",
        )
        html = out.read_text(encoding="utf-8")

        assert "structure-cell"   in html   # structures
        assert "hist-bar"         in html   # score histogram
        assert "Chemical Space"   in html   # UMAP
        assert "Drug-Likeness"    in html   # properties
        assert "Full Test"        in html   # experiment name

    def test_html_is_large_enough(self, tmp_path: Path) -> None:
        """A dashboard with all features should be at least 10 KB."""
        df = _smiles_df(3)
        emb = np.random.default_rng(0).normal(size=(3, 2))
        props = _properties_df(3)
        out = tmp_path / "dashboard.html"
        generate_dashboard(
            df, smiles_col="smiles", umap_embedding=emb,
            properties_df=props, output_path=out,
        )
        assert out.stat().st_size > 10_000
