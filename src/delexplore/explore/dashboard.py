"""Interactive HTML hit explorer for DEL screening results.

Produces a fully self-contained HTML dashboard (no external JS/CSS
dependencies) from a ranked hit list.  Sections included depend on which
optional inputs are supplied:

- **Summary cards** — always present
- **Sortable hit table** with inline SVG structures — always present;
  structures require RDKit
- **Score histogram** — always present when ``composite_score`` is available
- **UMAP plot** — included when *umap_embedding* is supplied
- **Drug-likeness section** — included when *properties_df* is supplied

Usage
-----
>>> from delexplore.explore.dashboard import generate_dashboard
>>> html_path = generate_dashboard(
...     ranked_df,
...     smiles_col="smiles",
...     top_n=100,
...     output_path=Path("results/dashboard.html"),
...     experiment_name="JP-1 vs L1CAM",
... )
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.rdBase import BlockLogs

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    logger.warning("RDKit not installed — structure SVGs will be placeholders.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not installed — UMAP plot will be omitted.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _placeholder_svg(width: int = 160, height: int = 120, msg: str = "No structure") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="{width}" height="{height}" fill="#f8f8f8" stroke="#ccc" stroke-width="1"/>'
        f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
        f'dominant-baseline="middle" font-family="sans-serif" font-size="11" fill="#aaa">{msg}</text>'
        f'</svg>'
    )


def _smiles_to_svg(smiles: str | None, width: int = 160, height: int = 120) -> str:
    """Render SMILES to an inline SVG string. Returns a placeholder on failure."""
    if not _RDKIT_AVAILABLE:
        return _placeholder_svg(width, height)
    if not smiles:
        return _placeholder_svg(width, height)
    try:
        with BlockLogs():
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return _placeholder_svg(width, height, "Invalid SMILES")
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().bondLineWidth = 1.2
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return _placeholder_svg(width, height)


def _build_histogram(
    values: list[float],
    n_bins: int = 20,
) -> dict[str, Any]:
    """Build histogram bar data for the Jinja2 template."""
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if len(arr) == 0:
        return {}

    counts, edges = np.histogram(arr, bins=n_bins)
    max_count = int(counts.max()) if counts.max() > 0 else 1

    bars = []
    for cnt, left, right in zip(counts, edges[:-1], edges[1:]):
        bars.append({
            "height_pct": round(cnt / max_count * 100, 1),
            "label": f"{left:.3f}–{right:.3f}: {cnt}",
        })

    return {
        "bars": bars,
        "x_min": float(edges[0]),
        "x_max": float(edges[-1]),
    }


def _build_prop_histogram(
    values: list[float],
    title: str,
    color: str,
    n_bins: int = 15,
    fmt: str = ".1f",
) -> dict[str, Any]:
    hist = _build_histogram(values, n_bins=n_bins)
    if not hist:
        return {}
    hist["title"] = title
    hist["color"] = color
    hist["x_min"] = f"{hist['x_min']:{fmt}}"
    hist["x_max"] = f"{hist['x_max']:{fmt}}"
    return hist


def _umap_to_b64_svg(
    embedding: np.ndarray,
    color_values: np.ndarray | None = None,
    color_label: str = "composite_score",
) -> str | None:
    """Render UMAP embedding to a base64-encoded SVG string for inline embedding."""
    if not _MPL_AVAILABLE:
        logger.warning("matplotlib not available — UMAP plot omitted")
        return None

    try:
        import io

        fig, ax = plt.subplots(figsize=(7, 5))
        x = embedding[:, 0]
        y = embedding[:, 1]

        if color_values is not None and len(color_values) == len(x):
            sc = ax.scatter(x, y, c=color_values, cmap="viridis_r",
                            s=8, alpha=0.6, linewidths=0)
            plt.colorbar(sc, ax=ax, label=color_label)
        else:
            ax.scatter(x, y, color="steelblue", s=8, alpha=0.6, linewidths=0)

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Chemical Space (UMAP)")

        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    except Exception as exc:
        logger.warning("UMAP plot generation failed: %s", exc)
        return None


def _extract_prop_stats(
    properties_df: pl.DataFrame,
) -> dict[str, Any] | None:
    """Compute summary statistics from a properties DataFrame."""
    stats: dict[str, Any] = {}

    def _median(col: str) -> float | None:
        if col not in properties_df.columns:
            return None
        s = properties_df[col].drop_nulls()
        return float(s.median()) if len(s) > 0 else None

    def _mean(col: str) -> float | None:
        if col not in properties_df.columns:
            return None
        s = properties_df[col].drop_nulls()
        return float(s.mean()) if len(s) > 0 else None

    def _min(col: str) -> float | None:
        if col not in properties_df.columns:
            return None
        s = properties_df[col].drop_nulls()
        return float(s.min()) if len(s) > 0 else None

    def _max(col: str) -> float | None:
        if col not in properties_df.columns:
            return None
        s = properties_df[col].drop_nulls()
        return float(s.max()) if len(s) > 0 else None

    def _frac_pass(col: str) -> float | None:
        if col not in properties_df.columns:
            return None
        s = properties_df[col].drop_nulls()
        return float(s.mean()) * 100 if len(s) > 0 else None

    mw_med = _median("mw")
    logp_med = _median("logp")
    tpsa_med = _median("tpsa")

    if mw_med is None and logp_med is None and tpsa_med is None:
        return None

    stats["median_mw"]  = mw_med or 0.0
    stats["mean_mw"]    = _mean("mw") or 0.0
    stats["median_logp"] = logp_med or 0.0
    stats["min_logp"]   = _min("logp") or 0.0
    stats["max_logp"]   = _max("logp") or 0.0
    stats["median_tpsa"] = tpsa_med or 0.0
    stats["median_qed"]  = _median("qed")

    lip = _frac_pass("lipinski_pass")
    stats["lipinski_pass_pct"] = lip if lip is not None else 0.0

    bro5 = _frac_pass("bro5_pass")
    stats["bro5_pass_pct"] = bro5

    return stats


def _build_prop_histograms(properties_df: pl.DataFrame) -> list[dict[str, Any]]:
    """Build histogram data for MW, logP, TPSA."""
    hists = []

    specs = [
        ("mw",   "MW (Da)",   "#2563eb", ".0f"),
        ("logp", "logP",      "#7c3aed", ".1f"),
        ("tpsa", "TPSA (Å²)", "#0891b2", ".0f"),
        ("qed",  "QED",       "#16a34a", ".2f"),
    ]
    for col, title, color, fmt in specs:
        if col not in properties_df.columns:
            continue
        vals = properties_df[col].drop_nulls().to_list()
        if not vals:
            continue
        h = _build_prop_histogram(vals, title=title, color=color, fmt=fmt)
        if h:
            hists.append(h)

    return hists


def _score_bar_max(hits: list[dict[str, Any]]) -> float:
    """Return the max composite_score for normalizing bar widths (min → width=80px)."""
    scores = [r["composite_score"] for r in hits if r.get("composite_score") is not None]
    if not scores:
        return 1.0
    return max(scores) or 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_dashboard(
    ranked_df: pl.DataFrame,
    smiles_col: str = "smiles",
    top_n: int = 100,
    umap_embedding: np.ndarray | None = None,
    properties_df: pl.DataFrame | None = None,
    output_path: Path | str = Path("dashboard.html"),
    experiment_name: str = "DELexplore",
    svg_size: tuple[int, int] = (160, 120),
    n_total_compounds: int | None = None,
) -> Path:
    """Generate a self-contained interactive HTML hit dashboard.

    The dashboard always includes summary cards and a sortable hit table.
    Additional sections are added progressively based on available inputs:

    - Structure column: requires RDKit and *smiles_col* in *ranked_df*.
    - Score histogram: requires ``composite_score`` in *ranked_df*.
    - UMAP section: requires *umap_embedding* (numpy array, shape ``(n, 2)``).
    - Drug-likeness section: requires *properties_df* with property columns.

    Args:
        ranked_df: Ranked hit list from
            :func:`~delexplore.analyse.rank.compute_composite_rank`.
            Must contain a ``"rank"`` column.  All columns are included in
            the table; recognised column names are displayed with formatting.
        smiles_col: Name of the SMILES column in *ranked_df*.  Used to
            render inline structure SVGs.  When absent or when RDKit is not
            installed, the structure column is omitted.
        top_n: Number of top-ranked compounds to include in the dashboard.
        umap_embedding: Optional array of shape ``(n_hits, 2)`` — the UMAP
            coordinates for the compounds in *ranked_df* (aligned row-wise).
            When supplied, a chemical space scatter plot is embedded.
        properties_df: Optional DataFrame with property columns (``mw``,
            ``logp``, ``tpsa``, ``qed``, ``lipinski_pass``, ``bro5_pass``).
            Must contain the same code columns as *ranked_df* so that it can
            be joined.  When supplied, a drug-likeness section is added.
        output_path: File path where the HTML dashboard is written.  Parent
            directories are created automatically.
        experiment_name: Experiment name displayed in the page title and
            header.
        svg_size: ``(width, height)`` in pixels for inline structure SVGs.
        n_total_compounds: Total number of compounds in the full ranked list
            (before the *top_n* cutoff), used in the summary header.  When
            ``None``, it defaults to ``len(ranked_df)``.

    Returns:
        Path to the written HTML file.

    Raises:
        ValueError: If *ranked_df* does not contain a ``"rank"`` column.
    """
    if "rank" not in ranked_df.columns:
        raise ValueError(
            "'rank' column not found in ranked_df. "
            "Run compute_composite_rank first."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine how many total compounds before cutoff
    if n_total_compounds is None:
        n_total_compounds = len(ranked_df)

    # Subset to top-N, sorted by rank
    hits_df = ranked_df.sort("rank").head(top_n)

    # Detect available columns
    code_cols = [c for c in hits_df.columns if c.startswith("code_")]
    has_smiles = smiles_col in hits_df.columns and _RDKIT_AVAILABLE
    has_composite_score = "composite_score" in hits_df.columns
    has_zscore = "zscore" in hits_df.columns
    has_support_score = "support_score" in hits_df.columns

    # Join properties if code_cols exist for the join
    has_lipinski = has_mw = has_logp = False
    if properties_df is not None and code_cols:
        prop_join_cols = [c for c in code_cols if c in properties_df.columns]
        if prop_join_cols:
            prop_cols_to_add = [
                c for c in ["lipinski_pass", "mw", "logp", "tpsa", "qed", "bro5_pass"]
                if c in properties_df.columns and c not in hits_df.columns
            ]
            if prop_cols_to_add:
                hits_df = hits_df.join(
                    properties_df.select(prop_join_cols + prop_cols_to_add),
                    on=prop_join_cols,
                    how="left",
                )
        has_lipinski = "lipinski_pass" in hits_df.columns
        has_mw       = "mw"            in hits_df.columns
        has_logp     = "logp"          in hits_df.columns

    # ── Build per-row data ──────────────────────────────────────────────────

    score_values: list[float] = []
    if has_composite_score:
        score_values = [
            v for v in hits_df["composite_score"].to_list()
            if v is not None
        ]

    score_bar_max = max(score_values) if score_values else 1.0

    svg_w, svg_h = svg_size
    rows_data: list[dict[str, Any]] = []
    hit_dicts = hits_df.to_dicts()

    for row in hit_dicts:
        smiles_val = row.get(smiles_col) if has_smiles else None
        svg_str = _smiles_to_svg(smiles_val, svg_w, svg_h) if has_smiles else ""

        score = row.get("composite_score")
        bar_width = int(score / score_bar_max * 80) if score is not None and score_bar_max else 0

        search_key = " ".join(
            str(row.get(c, "")) for c in code_cols + [smiles_col]
        )

        rows_data.append({
            "rank":           row.get("rank"),
            "svg":            svg_str,
            "codes":          {c: row.get(c) for c in code_cols},
            "composite_score": score,
            "zscore":          row.get("zscore"),
            "support_score":   row.get("support_score"),
            "lipinski_pass":   row.get("lipinski_pass"),
            "mw":              row.get("mw"),
            "logp":            row.get("logp"),
            "score_bar_width": bar_width,
            "search_key":      search_key,
        })

    # ── Score stats ─────────────────────────────────────────────────────────

    score_stats: dict[str, float] | None = None
    if score_values:
        arr = np.array(score_values)
        score_stats = {
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "median": float(np.median(arr)),
            "mean":   float(arr.mean()),
        }

    # ── Score histogram ─────────────────────────────────────────────────────

    score_hist = _build_histogram(score_values) if score_values else None

    # ── UMAP ────────────────────────────────────────────────────────────────

    umap_b64: str | None = None
    if umap_embedding is not None and len(umap_embedding) > 0:
        color_vals: np.ndarray | None = None
        if score_values and len(score_values) == len(umap_embedding):
            color_vals = np.array(
                hits_df["composite_score"].to_list(), dtype=float
            )
        umap_b64 = _umap_to_b64_svg(umap_embedding, color_vals)

    # ── Property stats & histograms ─────────────────────────────────────────

    prop_stats: dict[str, Any] | None = None
    prop_histograms: list[dict[str, Any]] = []

    if properties_df is not None:
        # Use properties_df for global stats; join with hits for the table
        prop_stats = _extract_prop_stats(properties_df)
        prop_histograms = _build_prop_histograms(properties_df)
    elif has_mw or has_logp or has_lipinski:
        # Properties were already joined into hits_df — compute stats from there
        prop_stats = _extract_prop_stats(hits_df)
        prop_histograms = _build_prop_histograms(hits_df)

    # ── Render template ─────────────────────────────────────────────────────

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("dashboard.html")

    html = template.render(
        experiment_name=experiment_name,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_total_compounds=n_total_compounds,
        hits=rows_data,
        code_cols=code_cols,
        has_smiles=has_smiles,
        has_composite_score=has_composite_score,
        has_zscore=has_zscore,
        has_support_score=has_support_score,
        has_lipinski=has_lipinski,
        has_mw=has_mw,
        has_logp=has_logp,
        score_stats=score_stats,
        score_hist=score_hist,
        umap_b64=umap_b64,
        prop_stats=prop_stats,
        prop_histograms=prop_histograms,
    )

    output_path.write_text(html, encoding="utf-8")
    logger.info(
        "Dashboard written to %s (%d hits, %d bytes)",
        output_path, len(rows_data), len(html),
    )
    return output_path
