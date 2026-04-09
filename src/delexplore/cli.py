"""Click CLI for DELexplore.

Entry point: ``delexplore``

Sub-command groups:
  delexplore qc           — data quality assessment
  delexplore analyse      — enrichment scoring and hit ranking
  delexplore explore      — chemical space visualisation
  delexplore library-assess — library synthesis quality intelligence
"""

import logging
import sys
from pathlib import Path

import click
import polars as pl

from delexplore import __version__

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_experiment_or_exit(config_path: Path) -> dict:
    """Load experiment data; exit with a clean message on failure."""
    from delexplore.io.readers import load_experiment

    try:
        exp = load_experiment(config_path)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"ERROR loading experiment: {exc}", err=True)
        sys.exit(1)

    if exp["counts"].is_empty():
        click.echo(
            "ERROR: No counts data was loaded. Check that counts.txt files exist "
            "under selections/<name>/counts.txt relative to the config.",
            err=True,
        )
        sys.exit(1)

    return exp


def _selections_for_group(meta_df: pl.DataFrame, group: str) -> list[str]:
    """Return selection names belonging to *group*."""
    return (
        meta_df
        .filter(pl.col("group") == group)["selection_name"]
        .to_list()
    )


def _validate_groups(
    meta_df: pl.DataFrame,
    post_group: str,
    control_group: str,
    config_path: Path,
) -> tuple[list[str], list[str]]:
    """Exit with a helpful message if either group has no selections."""
    from delexplore.io.readers import get_selection_metadata  # noqa: F401

    post_sels = _selections_for_group(meta_df, post_group)
    ctrl_sels = _selections_for_group(meta_df, control_group)

    all_groups = sorted(meta_df["group"].drop_nulls().unique().to_list())

    if not post_sels:
        click.echo(
            f"ERROR: No selections found for post group '{post_group}' in {config_path}.\n"
            f"Available groups: {all_groups}",
            err=True,
        )
        sys.exit(1)
    if not ctrl_sels:
        click.echo(
            f"ERROR: No selections found for control group '{control_group}' in {config_path}.\n"
            f"Available groups: {all_groups}",
            err=True,
        )
        sys.exit(1)

    return post_sels, ctrl_sels


def _write_multilevel_parquet(
    result: dict[str, pl.DataFrame],
    output_dir: Path,
    prefix: str,
) -> None:
    """Write one parquet file per level, e.g. ``zscore_mono_code_1.parquet``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for level_name, df in result.items():
        out = output_dir / f"{prefix}_{level_name}.parquet"
        df.write_parquet(out)
        logger.debug("Wrote %s (%d rows)", out, len(df))


def _load_multilevel_parquet(
    input_dir: Path,
    prefix: str,
) -> dict[str, pl.DataFrame]:
    """Reload parquet files written by :func:`_write_multilevel_parquet`."""
    result: dict[str, pl.DataFrame] = {}
    pattern = f"{prefix}_*.parquet"
    for path in sorted(input_dir.glob(pattern)):
        level_name = path.stem[len(prefix) + 1:]  # strip "prefix_"
        result[level_name] = pl.read_parquet(path)
        logger.debug("Loaded %s (%d rows)", path, len(result[level_name]))
    return result


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(__version__, prog_name="delexplore")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def main(verbose: bool) -> None:
    """DELexplore — statistical analysis and hit ranking for DEL screening data."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=level)


# ---------------------------------------------------------------------------
# qc group
# ---------------------------------------------------------------------------


@main.group()
def qc() -> None:
    """Data quality assessment commands."""


@qc.command("assess")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to config.yaml.",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory to write data_quality.json and HTML report.",
)
def qc_assess(config_path: Path, output_dir: Path) -> None:
    """Assess data quality and write data_quality.json + qc_report.html."""
    from delexplore.io.readers import get_selection_metadata
    from delexplore.qc.assess import generate_quality_report

    exp = _load_experiment_or_exit(config_path)
    counts = exp["counts"]
    n_cycles = exp["n_cycles"]
    config = exp["config"]

    meta_df = get_selection_metadata(config)
    # generate_quality_report needs group column; it uses "group" internally
    # The assess module uses metadata_df with selection_name + group columns
    report = generate_quality_report(
        counts_df=counts,
        n_cycles=n_cycles,
        config=config,
        output_dir=output_dir,
        metadata_df=meta_df,
    )

    click.echo(f"QC complete  →  overall quality: {report['overall_quality'].upper()}")
    click.echo(f"  Warnings   : {len(report['warnings'])}")
    if report["warnings"]:
        for w in report["warnings"]:
            click.echo(f"    ⚠  {w}")
    click.echo(f"  Recommended levels: {', '.join(report['recommended_analysis_levels'])}")
    click.echo(f"  Report written to: {output_dir / 'qc_report.html'}")


@qc.command("naive")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to config.yaml.",
)
def qc_naive(config_path: Path) -> None:
    """Assess naive library synthesis quality."""
    click.echo("Not yet implemented: qc naive")


# ---------------------------------------------------------------------------
# analyse group
# ---------------------------------------------------------------------------

# Shared options factory
_ANALYSE_OPTIONS = [
    click.option(
        "--config",
        "config_path",
        required=True,
        type=click.Path(exists=True, path_type=Path),
        help="Path to config.yaml.",
    ),
    click.option(
        "--post-group",
        "post_group",
        required=True,
        help="Group name for the protein/post-selection condition (e.g. 'protein').",
    ),
    click.option(
        "--control-group",
        "control_group",
        required=True,
        help="Group name for the control/blank condition (e.g. 'no_protein').",
    ),
    click.option(
        "--output",
        "output_dir",
        required=True,
        type=click.Path(path_type=Path),
        help="Directory for output parquet files.",
    ),
]


def _add_analyse_options(fn):
    for option in reversed(_ANALYSE_OPTIONS):
        fn = option(fn)
    return fn


@main.group()
def analyse() -> None:
    """Enrichment scoring and hit ranking commands."""


@analyse.command("zscore")
@_add_analyse_options
def analyse_zscore(
    config_path: Path,
    post_group: str,
    control_group: str,
    output_dir: Path,
) -> None:
    """Compute normalized z-score enrichment at all synthon levels (Faver 2019)."""
    from delexplore.analyse.multilevel import run_multilevel_enrichment
    from delexplore.io.readers import get_selection_metadata

    exp = _load_experiment_or_exit(config_path)
    meta_df = get_selection_metadata(exp["config"])
    post_sels, ctrl_sels = _validate_groups(meta_df, post_group, control_group, config_path)
    code_cols = [c for c in exp["counts"].columns if c.startswith("code_")]

    click.echo(
        f"Running z-score enrichment: {len(post_sels)} post × {len(ctrl_sels)} control "
        f"selections, {exp['n_cycles']}-cycle library"
    )

    result = run_multilevel_enrichment(
        counts_df=exp["counts"],
        n_cycles=exp["n_cycles"],
        code_cols=code_cols,
        post_selections=post_sels,
        control_selections=ctrl_sels,
        methods=("zscore",),
    )

    _write_multilevel_parquet(result, output_dir, "zscore")

    click.echo(f"Z-score enrichment written to {output_dir}/  ({len(result)} levels)")
    for level, df in result.items():
        click.echo(f"  {level}: {len(df)} features")


@analyse.command("poisson")
@_add_analyse_options
def analyse_poisson(
    config_path: Path,
    post_group: str,
    control_group: str,
    output_dir: Path,
) -> None:
    """Compute Poisson CI and ML enrichment at all synthon levels (Hou 2023)."""
    from delexplore.analyse.multilevel import run_multilevel_enrichment
    from delexplore.io.readers import get_selection_metadata

    exp = _load_experiment_or_exit(config_path)
    meta_df = get_selection_metadata(exp["config"])
    post_sels, ctrl_sels = _validate_groups(meta_df, post_group, control_group, config_path)
    code_cols = [c for c in exp["counts"].columns if c.startswith("code_")]

    click.echo(
        f"Running Poisson ML enrichment: {len(post_sels)} post × {len(ctrl_sels)} control "
        f"selections, {exp['n_cycles']}-cycle library"
    )

    result = run_multilevel_enrichment(
        counts_df=exp["counts"],
        n_cycles=exp["n_cycles"],
        code_cols=code_cols,
        post_selections=post_sels,
        control_selections=ctrl_sels,
        methods=("poisson_ml",),
    )

    _write_multilevel_parquet(result, output_dir, "poisson")

    click.echo(f"Poisson enrichment written to {output_dir}/  ({len(result)} levels)")
    for level, df in result.items():
        click.echo(f"  {level}: {len(df)} features")


@analyse.command("deseq")
@_add_analyse_options
def analyse_deseq(
    config_path: Path,
    post_group: str,
    control_group: str,
    output_dir: Path,
) -> None:
    """Run PyDESeq2 negative binomial GLM enrichment (mono/disynthon levels).

    Note: trisynthon level is excluded by default for performance.  PyDESeq2
    requires at least 2 replicates per condition.
    """
    from delexplore.analyse.multilevel import run_deseq2_enrichment
    from delexplore.io.readers import get_selection_metadata

    exp = _load_experiment_or_exit(config_path)
    config = exp["config"]
    meta_df = get_selection_metadata(config)
    post_sels, ctrl_sels = _validate_groups(meta_df, post_group, control_group, config_path)
    code_cols = [c for c in exp["counts"].columns if c.startswith("code_")]

    # DESeq2 needs a "condition" column in the metadata
    # Build a filtered metadata frame for just the relevant selections
    all_sels = post_sels + ctrl_sels
    deseq_meta = (
        meta_df
        .filter(pl.col("selection_name").is_in(all_sels))
        .with_columns(
            pl.when(pl.col("group") == post_group)
            .then(pl.lit("protein"))
            .otherwise(pl.lit("no_protein"))
            .alias("condition")
        )
        .select(["selection_name", "condition"])
    )

    n_post = len(post_sels)
    n_ctrl = len(ctrl_sels)
    click.echo(
        f"Running PyDESeq2 enrichment: {n_post} post × {n_ctrl} control selections"
    )
    if n_post < 2 or n_ctrl < 2:
        click.echo(
            "WARNING: PyDESeq2 requires ≥ 2 replicates per condition. "
            f"post={n_post}, control={n_ctrl}. Consider using zscore or poisson instead.",
            err=True,
        )
        if n_post < 2 or n_ctrl < 2:
            sys.exit(1)

    # Filter counts to the relevant selections
    filtered_counts = exp["counts"].filter(pl.col("selection").is_in(all_sels))

    try:
        result = run_deseq2_enrichment(
            counts_df=filtered_counts,
            metadata_df=deseq_meta,
            n_cycles=exp["n_cycles"],
            code_cols=code_cols,
        )
    except ImportError:
        click.echo(
            "ERROR: pydeseq2 is not installed. Install it with:\n"
            "  uv add pydeseq2",
            err=True,
        )
        sys.exit(1)

    _write_multilevel_parquet(result, output_dir, "deseq")

    click.echo(f"DESeq2 enrichment written to {output_dir}/  ({len(result)} levels)")
    for level, df in result.items():
        click.echo(f"  {level}: {len(df)} features")


@analyse.command("rank")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to config.yaml.",
)
@click.option(
    "--post-group",
    "post_group",
    required=True,
    help="Group name for the protein/post-selection condition.",
)
@click.option(
    "--control-group",
    "control_group",
    required=True,
    help="Group name for the control/blank condition.",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory for ranked hit list output.",
)
@click.option(
    "--top-n",
    "top_n",
    default=100,
    show_default=True,
    help="Number of top compounds to export.",
)
@click.option(
    "--input",
    "input_dir",
    default=None,
    type=click.Path(path_type=Path),
    help=(
        "Directory containing pre-computed enrichment parquets "
        "(output of 'analyse zscore' / 'analyse poisson').  "
        "If not supplied, runs multilevel enrichment on the fly."
    ),
)
@click.option(
    "--library-parquet",
    "library_parquet",
    default=None,
    type=click.Path(path_type=Path),
    help=(
        "Path to library.parquet containing SMILES per compound "
        "(columns: code_* + smiles).  When provided, drug-likeness "
        "property penalty is factored into the composite score."
    ),
)
@click.option(
    "--smiles-col",
    "smiles_col",
    default="smiles",
    show_default=True,
    help="Name of the SMILES column in --library-parquet.",
)
def analyse_rank(
    config_path: Path,
    post_group: str,
    control_group: str,
    output_dir: Path,
    top_n: int,
    input_dir: Path | None,
    library_parquet: Path | None,
    smiles_col: str,
) -> None:
    """Produce a consensus hit ranking across all enrichment methods.

    If --input is supplied, loads pre-computed enrichment parquets from that
    directory (looks for both 'zscore_*.parquet' and 'poisson_*.parquet').
    Otherwise runs multi-method enrichment from scratch.

    If --library-parquet is supplied, drug-likeness properties are computed
    from SMILES and factored into the composite score as a property penalty.

    Outputs:
      hits_top{N}.csv          — top-N ranked compounds
      ranked_all.parquet       — all compounds with composite scores
    """
    from delexplore.analyse.multilevel import run_multilevel_enrichment
    from delexplore.analyse.rank import compute_composite_rank, export_hit_list
    from delexplore.io.readers import get_selection_metadata

    exp = _load_experiment_or_exit(config_path)
    meta_df = get_selection_metadata(exp["config"])
    post_sels, ctrl_sels = _validate_groups(meta_df, post_group, control_group, config_path)
    code_cols = [c for c in exp["counts"].columns if c.startswith("code_")]

    if input_dir is not None:
        # Load pre-computed results and merge by level
        click.echo(f"Loading pre-computed enrichment from {input_dir}")
        zscore_res = _load_multilevel_parquet(input_dir, "zscore")
        poisson_res = _load_multilevel_parquet(input_dir, "poisson")

        # Merge zscore and poisson columns into one dict per level
        all_levels = set(zscore_res) | set(poisson_res)
        multilevel: dict[str, pl.DataFrame] = {}
        for level in all_levels:
            base = zscore_res.get(level) or poisson_res.get(level)
            if level in zscore_res and level in poisson_res:
                # Join zscore columns onto poisson base (which has fold_enrichment)
                zscore_cols = [
                    c for c in zscore_res[level].columns
                    if c.startswith("zscore") and c not in (poisson_res[level].columns)
                ]
                if zscore_cols:
                    base = poisson_res[level].join(
                        zscore_res[level].select(code_cols + zscore_cols),
                        on=code_cols,
                        how="left",
                    )
            multilevel[level] = base  # type: ignore[assignment]

        if not multilevel:
            click.echo(
                f"ERROR: No enrichment parquets found in {input_dir}. "
                "Run 'analyse zscore' and/or 'analyse poisson' first.",
                err=True,
            )
            sys.exit(1)
    else:
        click.echo("Running multi-method enrichment (zscore + poisson_ml) from scratch…")
        multilevel = run_multilevel_enrichment(
            counts_df=exp["counts"],
            n_cycles=exp["n_cycles"],
            code_cols=code_cols,
            post_selections=post_sels,
            control_selections=ctrl_sels,
            methods=("zscore", "poisson_ml"),
        )

    # --- Property penalty (optional) ---
    properties_df: pl.DataFrame | None = None
    if library_parquet is not None:
        if not library_parquet.exists():
            click.echo(
                f"ERROR: --library-parquet file not found: {library_parquet}",
                err=True,
            )
            sys.exit(1)
        click.echo(f"Loading library SMILES from {library_parquet}")
        try:
            from delexplore.explore.properties import compute_properties_for_ranking

            lib_df = pl.read_parquet(library_parquet)
            missing_cols = [c for c in code_cols if c not in lib_df.columns]
            if missing_cols:
                click.echo(
                    f"ERROR: Library parquet is missing code columns: {missing_cols}",
                    err=True,
                )
                sys.exit(1)
            if smiles_col not in lib_df.columns:
                click.echo(
                    f"ERROR: Library parquet has no '{smiles_col}' column. "
                    f"Available: {lib_df.columns}",
                    err=True,
                )
                sys.exit(1)

            click.echo(
                f"  Computing drug-likeness properties for {len(lib_df):,} compounds…"
            )
            properties_df = compute_properties_for_ranking(
                lib_df, smiles_col=smiles_col, code_cols=code_cols
            )
            click.echo("  Property penalty computed and will be applied to ranking.")
        except ImportError:
            click.echo(
                "WARNING: RDKit not installed — property penalty skipped. "
                "Install with: pip install rdkit",
                err=True,
            )
    else:
        click.echo("No --library-parquet provided — property penalty not applied.")

    ranked = compute_composite_rank(
        multilevel, code_cols, properties_df=properties_df
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ranked.write_parquet(output_dir / "ranked_all.parquet")

    hits = export_hit_list(ranked, top_n=top_n, output_path=output_dir / f"hits_top{top_n}.csv")

    click.echo(f"Ranked {len(ranked)} compounds  →  {output_dir}/")
    click.echo(f"  Top-{top_n} hit list: hits_top{top_n}.csv")
    click.echo("  Full ranking:        ranked_all.parquet")
    click.echo(f"  Rank 1: {dict(zip(code_cols, [hits[c][0] for c in code_cols]))}  "
               f"composite_score={hits['composite_score'][0]:.4f}")


# ---------------------------------------------------------------------------
# explore group
# ---------------------------------------------------------------------------


@main.group()
def explore() -> None:
    """Chemical space visualisation and clustering commands."""


@explore.command("properties")
@click.option(
    "--hits",
    "hits_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to ranked hits parquet or CSV file.",
)
@click.option(
    "--smiles-col",
    "smiles_col",
    default="smiles",
    show_default=True,
    help="Name of the SMILES column in the hits file.",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory for property output files.",
)
def explore_properties(hits_path: Path, smiles_col: str, output_dir: Path) -> None:
    """Compute drug-likeness and macrocycle properties for hit compounds.

    Outputs:
      properties_{timestamp}.parquet  — full property table
      properties_summary.json         — aggregate statistics
    """
    import json
    from datetime import datetime

    try:
        from delexplore.explore.macrocycle import add_macrocycle_columns
        from delexplore.explore.properties import (
            assess_druglikeness,
            calculate_properties,
        )
    except ImportError:
        click.echo(
            "ERROR: RDKit is required for property calculation. "
            "Install with: pip install rdkit",
            err=True,
        )
        sys.exit(1)

    # Load hits file — support both parquet and CSV
    if hits_path.suffix.lower() == ".parquet":
        hits = pl.read_parquet(hits_path)
    else:
        hits = pl.read_csv(hits_path)

    if smiles_col not in hits.columns:
        click.echo(
            f"ERROR: SMILES column '{smiles_col}' not found. "
            f"Available columns: {hits.columns}",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Computing properties for {len(hits):,} compounds…")

    props = calculate_properties(hits, smiles_col=smiles_col)
    props = assess_druglikeness(props)
    props = add_macrocycle_columns(props, smiles_col=smiles_col)

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parquet_path = output_dir / f"properties_{timestamp}.parquet"
    props.write_parquet(parquet_path)

    # --- Summary statistics ---
    n_total = len(props)
    n_valid = int(props["mw"].drop_nulls().len())
    n_macrocycles = int(props["is_macrocycle"].sum()) if "is_macrocycle" in props.columns else 0
    frac_lipinski = (
        float(props["lipinski_pass"].drop_nulls().mean())
        if "lipinski_pass" in props.columns and props["lipinski_pass"].drop_nulls().len() > 0
        else None
    )
    frac_bro5 = (
        float(props["bro5_pass"].drop_nulls().mean())
        if "bro5_pass" in props.columns and props["bro5_pass"].drop_nulls().len() > 0
        else None
    )

    def _stat(col: str) -> dict:
        series = props[col].drop_nulls() if col in props.columns else pl.Series([])
        if series.len() == 0:
            return {"mean": None, "median": None}
        return {
            "mean": round(float(series.mean()), 3),
            "median": round(float(series.median()), 3),
        }

    summary = {
        "n_compounds": n_total,
        "n_valid_smiles": n_valid,
        "n_macrocycles": n_macrocycles,
        "fraction_lipinski_pass": round(frac_lipinski, 4) if frac_lipinski is not None else None,
        "fraction_bro5_pass": round(frac_bro5, 4) if frac_bro5 is not None else None,
        "mw": _stat("mw"),
        "logp": _stat("logp"),
        "tpsa": _stat("tpsa"),
    }

    summary_path = output_dir / "properties_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    click.echo(f"Properties computed  →  {output_dir}/")
    click.echo(f"  Compounds          : {n_total:,}")
    click.echo(f"  Valid SMILES        : {n_valid:,} ({n_valid/n_total*100:.1f}%)" if n_total else "")
    click.echo(f"  Macrocycles         : {n_macrocycles}")
    if frac_lipinski is not None:
        click.echo(f"  Lipinski pass       : {frac_lipinski*100:.1f}%")
    if frac_bro5 is not None:
        click.echo(f"  bRo5 pass           : {frac_bro5*100:.1f}%")
    mw_s = _stat("mw")
    if mw_s["mean"] is not None:
        click.echo(f"  MW  mean/median     : {mw_s['mean']} / {mw_s['median']}")
    click.echo(f"  Property table      : {parquet_path.name}")
    click.echo("  Summary JSON        : properties_summary.json")


@explore.command("render-hits")
@click.option(
    "--hits",
    "hits_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to ranked hits parquet or CSV file.",
)
@click.option(
    "--smiles-col",
    "smiles_col",
    default="smiles",
    show_default=True,
    help="Name of the SMILES column.",
)
@click.option(
    "--top-n",
    "top_n",
    default=20,
    show_default=True,
    help="Number of top compounds to render.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output file path (.svg or .png).",
)
@click.option(
    "--highlight",
    "highlight",
    default=None,
    help="SMARTS pattern to highlight in all structures.",
)
@click.option(
    "--cols-per-row",
    "cols_per_row",
    default=4,
    show_default=True,
    help="Number of structures per grid row.",
)
def explore_render_hits(
    hits_path: Path,
    smiles_col: str,
    top_n: int,
    output_path: Path,
    highlight: str | None,
    cols_per_row: int,
) -> None:
    """Render a grid image of the top-N ranked hit structures.

    The output format is determined by the file extension of --output
    (.svg for SVG, anything else for PNG).
    """
    try:
        from delexplore.explore.structures import render_hit_grid
    except ImportError:
        click.echo(
            "ERROR: RDKit is required for structure rendering. "
            "Install with: pip install rdkit",
            err=True,
        )
        sys.exit(1)

    suffix = output_path.suffix.lower()
    img_format = "svg" if suffix == ".svg" else "png"

    # Load hits file — support both parquet and CSV
    if hits_path.suffix.lower() == ".parquet":
        hits = pl.read_parquet(hits_path)
    else:
        hits = pl.read_csv(hits_path)

    if smiles_col not in hits.columns:
        click.echo(
            f"ERROR: SMILES column '{smiles_col}' not found. "
            f"Available columns: {hits.columns}",
            err=True,
        )
        sys.exit(1)

    if "rank" not in hits.columns:
        click.echo(
            "WARNING: No 'rank' column found — using row order as rank.",
            err=True,
        )
        hits = hits.with_columns(
            pl.Series("rank", range(1, len(hits) + 1))
        )

    n_render = min(top_n, len(hits))
    click.echo(f"Rendering {n_render} structures ({img_format.upper()})…")

    render_hit_grid(
        hits,
        smiles_col=smiles_col,
        top_n=top_n,
        output_path=output_path,
        img_format=img_format,
        cols_per_row=cols_per_row,
        highlight_substructure=highlight,
    )

    click.echo(f"Structure grid written to {output_path}")


@explore.command("umap")
@click.option(
    "--hits",
    "hits_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to ranked hit list parquet.",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory for UMAP plot outputs.",
)
def explore_umap(hits_path: Path, output_dir: Path) -> None:
    """Compute UMAP projection of compound chemical space."""
    click.echo("Not yet implemented: explore umap")


@explore.command("dashboard")
@click.option(
    "--hits",
    "hits_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to ranked hits parquet or CSV (must have a 'rank' column).",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output HTML file path (e.g. results/dashboard.html).",
)
@click.option(
    "--experiment-name",
    "experiment_name",
    default="DELexplore",
    show_default=True,
    help="Experiment name shown in the dashboard header.",
)
@click.option(
    "--smiles-col",
    "smiles_col",
    default="smiles",
    show_default=True,
    help="Name of the SMILES column in the hits file.",
)
@click.option(
    "--top-n",
    "top_n",
    default=100,
    show_default=True,
    help="Number of top-ranked compounds to include.",
)
@click.option(
    "--properties",
    "properties_path",
    default=None,
    type=click.Path(path_type=Path),
    help=(
        "Path to a properties parquet (output of 'explore properties').  "
        "When supplied, adds drug-likeness section to the dashboard."
    ),
)
@click.option(
    "--umap",
    "umap_path",
    default=None,
    type=click.Path(path_type=Path),
    help=(
        "Path to a UMAP embedding parquet with 'umap_x' and 'umap_y' columns "
        "(output of 'explore umap').  When supplied, adds a chemical space plot."
    ),
)
def explore_dashboard(
    hits_path: Path,
    output_path: Path,
    experiment_name: str,
    smiles_col: str,
    top_n: int,
    properties_path: Path | None,
    umap_path: Path | None,
) -> None:
    """Generate a self-contained interactive HTML hit dashboard.

    The dashboard always includes summary cards and a sortable/searchable hit
    table.  Additional sections are added progressively:

    - Structure SVGs: when the hits file has a SMILES column and RDKit is
      installed.
    - Score histogram: when hits have a composite_score column.
    - UMAP scatter: when --umap is supplied.
    - Drug-likeness: when --properties is supplied.
    """
    import numpy as np

    from delexplore.explore.dashboard import generate_dashboard

    # Load hits
    if hits_path.suffix.lower() == ".parquet":
        hits = pl.read_parquet(hits_path)
    else:
        hits = pl.read_csv(hits_path)

    if "rank" not in hits.columns:
        click.echo(
            "ERROR: hits file must contain a 'rank' column. "
            "Run 'analyse rank' first.",
            err=True,
        )
        sys.exit(1)

    n_total = len(hits)

    # Optional: UMAP embedding
    umap_embedding: np.ndarray | None = None
    if umap_path is not None:
        if not umap_path.exists():
            click.echo(f"ERROR: --umap file not found: {umap_path}", err=True)
            sys.exit(1)
        umap_df = pl.read_parquet(umap_path)
        if "umap_x" not in umap_df.columns or "umap_y" not in umap_df.columns:
            click.echo(
                f"ERROR: UMAP parquet must have 'umap_x' and 'umap_y' columns. "
                f"Available: {umap_df.columns}",
                err=True,
            )
            sys.exit(1)
        # Align embedding to the top-N hits in the same order
        code_cols = [c for c in hits.columns if c.startswith("code_")]
        if code_cols:
            top_hits = hits.sort("rank").head(top_n)
            merged = top_hits.join(
                umap_df.select(code_cols + ["umap_x", "umap_y"]),
                on=code_cols,
                how="left",
            )
            xs = merged["umap_x"].fill_null(0).to_numpy()
            ys = merged["umap_y"].fill_null(0).to_numpy()
            umap_embedding = np.column_stack([xs, ys])
        else:
            click.echo(
                "WARNING: No code_* columns found — cannot align UMAP embedding.",
                err=True,
            )

    # Optional: properties
    properties_df: pl.DataFrame | None = None
    if properties_path is not None:
        if not properties_path.exists():
            click.echo(f"ERROR: --properties file not found: {properties_path}", err=True)
            sys.exit(1)
        properties_df = pl.read_parquet(properties_path)
        click.echo(f"  Properties loaded: {len(properties_df):,} compounds")

    click.echo(
        f"Generating dashboard: top-{top_n} of {n_total:,} compounds  →  {output_path}"
    )

    out = generate_dashboard(
        ranked_df=hits,
        smiles_col=smiles_col,
        top_n=top_n,
        umap_embedding=umap_embedding,
        properties_df=properties_df,
        output_path=output_path,
        experiment_name=experiment_name,
        n_total_compounds=n_total,
    )

    click.echo(f"Dashboard written to {out}")
    click.echo(f"  Open in a browser: file://{out.resolve()}")


@explore.command("cluster")
@click.option(
    "--embedding",
    "embedding_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to UMAP embedding parquet.",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory for cluster output.",
)
def explore_cluster(embedding_path: Path, output_dir: Path) -> None:
    """Cluster compounds in UMAP space with HDBSCAN."""
    click.echo("Not yet implemented: explore cluster")


# ---------------------------------------------------------------------------
# library-assess group
# ---------------------------------------------------------------------------


@main.command("library-assess")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to config.yaml.",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory for synthesis and truncation reports.",
)
def library_assess(config_path: Path, output_dir: Path) -> None:
    """Assess library synthesis quality from naive (blank) selection counts.

    Uses blank/no_protein selections from the config as the naive library.
    Outputs:
      synthesis_yield.json     — per-BB yield stats (CV, Gini, outliers)
      truncation_flags.json    — suspected truncation candidates
      bb_yield_weights.parquet — normalization weights for enrichment analysis
    """
    from delexplore.qc.naive import (
        identify_naive_selections,
        run_naive_qc,
    )

    exp = _load_experiment_or_exit(config_path)
    config = exp["config"]
    counts = exp["counts"]
    n_cycles = exp["n_cycles"]
    code_cols = [c for c in counts.columns if c.startswith("code_")]

    naive_sels = identify_naive_selections(config)
    if not naive_sels:
        click.echo(
            "ERROR: No naive/blank selections found in config. "
            "Check that selections have target='No Protein' or group='no_protein'.",
            err=True,
        )
        sys.exit(1)

    click.echo(
        f"Naive library assessment: {len(naive_sels)} blank selection(s): "
        f"{', '.join(naive_sels)}"
    )

    naive_counts = counts.filter(pl.col("selection").is_in(naive_sels))
    total_reads = int(naive_counts["count"].sum())
    click.echo(f"  Total naive reads: {total_reads:,}")

    result = run_naive_qc(naive_counts, n_cycles, code_cols, output_dir)

    click.echo(f"Library assessment complete  →  {output_dir}/")
    click.echo(f"  Truncation candidates : {result['n_flagged_bbs']}")
    if result["truncation_flags"]:
        for t in result["truncation_flags"][:5]:
            click.echo(f"    ⚠  {t['evidence']}")
        if len(result["truncation_flags"]) > 5:
            click.echo(f"    … and {len(result['truncation_flags']) - 5} more")

    for col in code_cols:
        yld = result["synthesis_yield"].get(col, {})
        n_high = len(yld.get("outliers_high", []))
        n_zero = len(yld.get("outliers_zero", []))
        cv     = yld.get("cv", 0.0)
        gini   = yld.get("gini", 0.0)
        click.echo(
            f"  {col}: CV={cv:.3f}  Gini={gini:.3f}  "
            f"over-represented={n_high}  absent={n_zero}"
        )
