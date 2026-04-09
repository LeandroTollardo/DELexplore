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
def analyse_rank(
    config_path: Path,
    post_group: str,
    control_group: str,
    output_dir: Path,
    top_n: int,
    input_dir: Path | None,
) -> None:
    """Produce a consensus hit ranking across all enrichment methods.

    If --input is supplied, loads pre-computed enrichment parquets from that
    directory (looks for both 'zscore_*.parquet' and 'poisson_*.parquet').
    Otherwise runs multi-method enrichment from scratch.

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

    ranked = compute_composite_rank(multilevel, code_cols)

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
