"""CLI integration tests for the property pipeline additions.

Tests:
  - `analyse rank --help` shows --library-parquet option
  - `explore properties --help` works
  - `explore render-hits --help` works
  - Full pipeline: ranked DataFrame with SMILES → properties → ranking with
    penalty → render grid (all via the Python API, not subprocess)
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner

from delexplore.cli import main

# ---------------------------------------------------------------------------
# Shared SMILES fixtures
# ---------------------------------------------------------------------------

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
NAPROXEN = "COc1ccc2cc(ccc2c1)C(C)C(=O)O"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
INVALID = "not_a_smiles"


def _make_ranked_df_with_smiles(
    smiles_list: list[str] | None = None,
    n: int = 5,
) -> pl.DataFrame:
    """Build a ranked DataFrame compatible with all three CLI commands."""
    if smiles_list is None:
        smiles_list = [ASPIRIN, CAFFEINE, IBUPROFEN, NAPROXEN, PARACETAMOL][:n]
    n = len(smiles_list)
    return pl.DataFrame(
        {
            "rank": list(range(1, n + 1)),
            "code_1": list(range(n)),
            "code_2": list(range(n)),
            "smiles": smiles_list,
            "composite_score": [0.1 * i for i in range(1, n + 1)],
            "zscore": [float(i) for i in range(n)],
            "fold_enrichment": [float(i + 1) for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# --help tests
# ---------------------------------------------------------------------------


class TestHelpOutput:
    def test_analyse_rank_help_shows_library_parquet(self):
        runner = CliRunner()
        result = runner.invoke(main, ["analyse", "rank", "--help"])
        assert result.exit_code == 0, result.output
        assert "--library-parquet" in result.output

    def test_analyse_rank_help_shows_smiles_col(self):
        runner = CliRunner()
        result = runner.invoke(main, ["analyse", "rank", "--help"])
        assert result.exit_code == 0
        assert "--smiles-col" in result.output

    def test_explore_properties_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["explore", "properties", "--help"])
        assert result.exit_code == 0, result.output
        assert "--hits" in result.output
        assert "--smiles-col" in result.output
        assert "--output" in result.output

    def test_explore_render_hits_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["explore", "render-hits", "--help"])
        assert result.exit_code == 0, result.output
        assert "--hits" in result.output
        assert "--top-n" in result.output
        assert "--output" in result.output
        assert "--highlight" in result.output
        assert "--cols-per-row" in result.output


# ---------------------------------------------------------------------------
# explore properties command
# ---------------------------------------------------------------------------


class TestExploreProperties:
    def test_writes_parquet_and_summary_json(self, tmp_path):
        runner = CliRunner()
        hits_file = tmp_path / "hits.parquet"
        _make_ranked_df_with_smiles().write_parquet(hits_file)
        out_dir = tmp_path / "props_out"

        result = runner.invoke(
            main,
            ["explore", "properties", "--hits", str(hits_file), "--output", str(out_dir)],
        )
        assert result.exit_code == 0, result.output

        # Parquet written
        parquet_files = list(out_dir.glob("properties_*.parquet"))
        assert len(parquet_files) == 1

        # Summary JSON written and parseable
        summary_path = out_dir / "properties_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert "n_compounds" in summary
        assert summary["n_compounds"] == 5
        assert "n_valid_smiles" in summary
        assert "n_macrocycles" in summary
        assert "fraction_lipinski_pass" in summary
        assert "fraction_bro5_pass" in summary
        assert "mw" in summary
        assert "mean" in summary["mw"]

    def test_summary_json_values_reasonable(self, tmp_path):
        runner = CliRunner()
        hits_file = tmp_path / "hits.parquet"
        # All 5 molecules are small, drug-like → lipinski_pass expected True for most
        _make_ranked_df_with_smiles().write_parquet(hits_file)
        out_dir = tmp_path / "props_out"

        runner.invoke(
            main,
            ["explore", "properties", "--hits", str(hits_file), "--output", str(out_dir)],
        )
        summary = json.loads((out_dir / "properties_summary.json").read_text())
        assert summary["n_valid_smiles"] == 5
        assert summary["fraction_lipinski_pass"] is not None
        assert 0.0 <= summary["fraction_lipinski_pass"] <= 1.0

    def test_with_invalid_smiles_in_hits(self, tmp_path):
        runner = CliRunner()
        hits_file = tmp_path / "hits.parquet"
        df = _make_ranked_df_with_smiles([ASPIRIN, INVALID, CAFFEINE])
        df.write_parquet(hits_file)
        out_dir = tmp_path / "props_out"

        result = runner.invoke(
            main,
            ["explore", "properties", "--hits", str(hits_file), "--output", str(out_dir)],
        )
        assert result.exit_code == 0, result.output
        summary = json.loads((out_dir / "properties_summary.json").read_text())
        # 2 of 3 SMILES are valid
        assert summary["n_compounds"] == 3
        assert summary["n_valid_smiles"] == 2

    def test_missing_smiles_col_exits_nonzero(self, tmp_path):
        runner = CliRunner()
        hits_file = tmp_path / "hits.parquet"
        pl.DataFrame({"rank": [1], "code_1": [0]}).write_parquet(hits_file)
        out_dir = tmp_path / "props_out"

        result = runner.invoke(
            main,
            [
                "explore", "properties",
                "--hits", str(hits_file),
                "--smiles-col", "smiles",
                "--output", str(out_dir),
            ],
        )
        assert result.exit_code != 0

    def test_csv_input_accepted(self, tmp_path):
        runner = CliRunner()
        hits_file = tmp_path / "hits.csv"
        _make_ranked_df_with_smiles().write_csv(hits_file)
        out_dir = tmp_path / "props_out"

        result = runner.invoke(
            main,
            ["explore", "properties", "--hits", str(hits_file), "--output", str(out_dir)],
        )
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# explore render-hits command
# ---------------------------------------------------------------------------


class TestExploreRenderHits:
    def test_writes_svg_file(self, tmp_path):
        runner = CliRunner()
        hits_file = tmp_path / "hits.parquet"
        _make_ranked_df_with_smiles().write_parquet(hits_file)
        out_svg = tmp_path / "grid.svg"

        result = runner.invoke(
            main,
            [
                "explore", "render-hits",
                "--hits", str(hits_file),
                "--output", str(out_svg),
                "--top-n", "5",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_svg.exists()
        content = out_svg.read_text()
        assert "<svg" in content

    def test_writes_png_file(self, tmp_path):
        runner = CliRunner()
        hits_file = tmp_path / "hits.parquet"
        _make_ranked_df_with_smiles().write_parquet(hits_file)
        out_png = tmp_path / "grid.png"

        result = runner.invoke(
            main,
            [
                "explore", "render-hits",
                "--hits", str(hits_file),
                "--output", str(out_png),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_png.exists()
        assert out_png.read_bytes()[:4] == b"\x89PNG"

    def test_with_highlight_smarts(self, tmp_path):
        runner = CliRunner()
        hits_file = tmp_path / "hits.parquet"
        _make_ranked_df_with_smiles().write_parquet(hits_file)
        out_svg = tmp_path / "grid.svg"

        result = runner.invoke(
            main,
            [
                "explore", "render-hits",
                "--hits", str(hits_file),
                "--output", str(out_svg),
                "--highlight", "c1ccccc1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_svg.exists()

    def test_missing_rank_col_adds_row_order(self, tmp_path):
        """Hits without a 'rank' column get row order as rank with a warning."""
        runner = CliRunner()
        hits_file = tmp_path / "hits.parquet"
        df = pl.DataFrame({"smiles": [ASPIRIN, CAFFEINE], "code_1": [0, 1]})
        df.write_parquet(hits_file)
        out_svg = tmp_path / "grid.svg"

        result = runner.invoke(
            main,
            [
                "explore", "render-hits",
                "--hits", str(hits_file),
                "--output", str(out_svg),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_svg.exists()

    def test_csv_input_accepted(self, tmp_path):
        runner = CliRunner()
        hits_file = tmp_path / "hits.csv"
        _make_ranked_df_with_smiles().write_csv(hits_file)
        out_svg = tmp_path / "grid.svg"

        result = runner.invoke(
            main,
            [
                "explore", "render-hits",
                "--hits", str(hits_file),
                "--output", str(out_svg),
            ],
        )
        assert result.exit_code == 0, result.output

    def test_cols_per_row_option(self, tmp_path):
        runner = CliRunner()
        hits_file = tmp_path / "hits.parquet"
        _make_ranked_df_with_smiles().write_parquet(hits_file)
        out_svg = tmp_path / "grid.svg"

        result = runner.invoke(
            main,
            [
                "explore", "render-hits",
                "--hits", str(hits_file),
                "--output", str(out_svg),
                "--cols-per-row", "2",
            ],
        )
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Integration: full pipeline via Python API
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    """Exercise the complete path: properties → ranking → render grid."""

    def test_properties_to_ranking_with_penalty(self):
        """compute_properties_for_ranking output plugs into compute_composite_rank."""
        from delexplore.analyse.multilevel import run_multilevel_enrichment
        from delexplore.analyse.rank import compute_composite_rank
        from delexplore.explore.properties import compute_properties_for_ranking

        # Reuse the synthetic_counts fixture via inline fixture call
        # Build a minimal counts DataFrame directly
        import itertools

        code_pairs = list(itertools.product(range(5), range(4)))
        sels = ["target_1", "target_2", "blank_1", "blank_2"]
        rows = []
        for sel in sels:
            is_target = sel.startswith("target")
            for c1, c2 in code_pairs:
                if (c1, c2) == (1, 2) and is_target:
                    count = 300
                elif (c1, c2) == (1, 2):
                    count = 2
                else:
                    count = (c1 + c2 + 1) % 10
                rows.append({"selection": sel, "code_1": c1, "code_2": c2, "count": count})

        counts = pl.DataFrame(rows)
        code_cols = ["code_1", "code_2"]

        ml = run_multilevel_enrichment(
            counts,
            n_cycles=2,
            code_cols=code_cols,
            post_selections=["target_1", "target_2"],
            control_selections=["blank_1", "blank_2"],
        )

        # Build SMILES DataFrame (one SMILES per compound)
        from delexplore.analyse.aggregate import get_level_name

        full_level = get_level_name(tuple(code_cols))
        compound_df = ml[full_level].select(code_cols)
        # Cycle through valid SMILES
        smiles_pool = [ASPIRIN, CAFFEINE, IBUPROFEN, NAPROXEN, PARACETAMOL]
        smiles_list = [smiles_pool[i % len(smiles_pool)] for i in range(len(compound_df))]
        smiles_df = compound_df.with_columns(pl.Series("smiles", smiles_list))

        penalty_df = compute_properties_for_ranking(
            smiles_df, smiles_col="smiles", code_cols=code_cols
        )

        assert "property_penalty" in penalty_df.columns
        assert penalty_df["property_penalty"].null_count() == 0
        # All SMILES are drug-like → penalty should be 1.0
        assert (penalty_df["property_penalty"] == 1.0).all()

        ranked = compute_composite_rank(
            ml, code_cols, properties_df=penalty_df
        )

        assert "rank" in ranked.columns
        assert "composite_score" in ranked.columns
        assert "property_penalty" in ranked.columns
        assert len(ranked) == len(compound_df)
        # Penalties are 1.0 for all → composite_score is purely from agreement/support
        assert (ranked["property_penalty"] == 1.0).all()

    def test_ranking_with_penalty_then_render(self, tmp_path):
        """End-to-end: ranked DataFrame with SMILES → render grid file."""
        from delexplore.explore.structures import render_hit_grid

        ranked = _make_ranked_df_with_smiles()
        out = tmp_path / "final_hits.svg"
        render_hit_grid(ranked, smiles_col="smiles", top_n=5, output_path=out)

        assert out.exists()
        assert "<svg" in out.read_text()

    def test_analyse_rank_no_library_message(self, tmp_path, synthetic_data_dir):
        """analyse rank without --library-parquet prints the expected note."""
        runner = CliRunner()
        out_dir = tmp_path / "rank_out"

        result = runner.invoke(
            main,
            [
                "analyse", "rank",
                "--config", str(synthetic_data_dir / "config.yaml"),
                "--post-group", "protein",
                "--control-group", "no_protein",
                "--output", str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "property penalty not applied" in result.output.lower()

    def test_analyse_rank_with_library_parquet(self, tmp_path, synthetic_data_dir):
        """analyse rank with --library-parquet applies property penalty."""
        from delexplore.io.readers import load_experiment

        runner = CliRunner()

        # Build a library parquet from the synthetic config
        exp = load_experiment(synthetic_data_dir / "config.yaml")
        code_cols = [c for c in exp["counts"].columns if c.startswith("code_")]

        # All unique code combinations in the counts
        all_compounds = (
            exp["counts"]
            .select(code_cols)
            .unique()
        )
        # Assign aspirin SMILES to all (simplified)
        lib_df = all_compounds.with_columns(pl.lit(ASPIRIN).alias("smiles"))
        lib_path = tmp_path / "library.parquet"
        lib_df.write_parquet(lib_path)

        out_dir = tmp_path / "rank_out"
        result = runner.invoke(
            main,
            [
                "analyse", "rank",
                "--config", str(synthetic_data_dir / "config.yaml"),
                "--post-group", "protein",
                "--control-group", "no_protein",
                "--output", str(out_dir),
                "--library-parquet", str(lib_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "property penalty" in result.output.lower()

        # ranked_all.parquet must exist and have a property_penalty column
        ranked = pl.read_parquet(out_dir / "ranked_all.parquet")
        assert "property_penalty" in ranked.columns
        # All penalties should be 1.0 (aspirin passes bRo5)
        assert (ranked["property_penalty"] == 1.0).all()
