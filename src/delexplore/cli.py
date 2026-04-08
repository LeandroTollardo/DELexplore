"""Click CLI for DELexplore.

Entry point: ``delexplore``

Sub-command groups:
  delexplore qc           — data quality assessment
  delexplore analyse      — enrichment scoring and hit ranking
  delexplore explore      — chemical space visualisation
  delexplore library-assess — library synthesis quality intelligence
"""

import logging
from pathlib import Path

import click

from delexplore import __version__

logger = logging.getLogger(__name__)


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
    """Assess data quality and write data_quality.json."""
    click.echo("Not yet implemented: qc assess")


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


@main.group()
def analyse() -> None:
    """Enrichment scoring and hit ranking commands."""


@analyse.command("zscore")
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
    help="Directory for output parquet files.",
)
def analyse_zscore(config_path: Path, output_dir: Path) -> None:
    """Compute normalized z-score enrichment (Faver et al. 2019)."""
    click.echo("Not yet implemented: analyse zscore")


@analyse.command("poisson")
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
    help="Directory for output parquet files.",
)
def analyse_poisson(config_path: Path, output_dir: Path) -> None:
    """Compute Poisson CI and ML enrichment (Kuai 2018, Hou 2023)."""
    click.echo("Not yet implemented: analyse poisson")


@analyse.command("deseq")
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
    help="Directory for output parquet files.",
)
def analyse_deseq(config_path: Path, output_dir: Path) -> None:
    """Run PyDESeq2 negative binomial GLM enrichment analysis."""
    click.echo("Not yet implemented: analyse deseq")


@analyse.command("rank")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to config.yaml.",
)
@click.option(
    "--input",
    "input_dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing enrichment parquet files.",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory for ranked hit list output.",
)
def analyse_rank(config_path: Path, input_dir: Path, output_dir: Path) -> None:
    """Produce consensus hit ranking across all enrichment methods."""
    click.echo("Not yet implemented: analyse rank")


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
    "--naive-counts",
    "naive_counts_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to naive library counts parquet or txt.",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory for synthesis and truncation reports.",
)
def library_assess(config_path: Path, naive_counts_path: Path, output_dir: Path) -> None:
    """Assess library synthesis quality from naive library counts."""
    click.echo("Not yet implemented: library-assess")
