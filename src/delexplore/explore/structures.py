"""Molecular structure rendering for DEL hit compounds.

Generates grid images and individual structure SVGs/PNGs from ranked hit
lists, annotated with enrichment scores.  Output is either returned in
memory or written to a file.

Usage
-----
>>> from delexplore.explore.structures import render_hit_grid, smiles_to_svg_dict
>>> svg = render_hit_grid(ranked_df, top_n=20, output_path="hits.svg")
>>> inline = smiles_to_svg_dict(hits_df, smiles_col="smiles")
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RDKit import (gated — optional dependency)
# ---------------------------------------------------------------------------

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.rdBase import BlockLogs

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    logger.warning(
        "RDKit is not installed. Structure rendering will not be available. "
        "Install with: pip install rdkit"
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_rdkit() -> None:
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for structure rendering. "
            "Install with: pip install rdkit"
        )


def _parse_mol(smiles: str | None):
    """Return an RDKit Mol or None if invalid/missing."""
    if not smiles:
        return None
    with BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    return mol


def _placeholder_svg(width: int, height: int, message: str = "Invalid SMILES") -> str:
    """Return a minimal SVG rectangle with a centred text label."""
    return (
        f'<?xml version="1.0" encoding="iso-8859-1"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'baseProfile="full" width="{width}" height="{height}">'
        f'<rect width="{width}" height="{height}" fill="#f8f8f8" stroke="#cccccc" stroke-width="1"/>'
        f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
        f'dominant-baseline="middle" font-family="sans-serif" '
        f'font-size="14" fill="#999999">{message}</text>'
        f'</svg>'
    )


def _placeholder_png(width: int, height: int, message: str = "Invalid SMILES") -> bytes:
    """Return PNG bytes of a grey placeholder rectangle with text.

    Falls back to a raw 1×1 transparent PNG if Pillow is unavailable.
    """
    try:
        from PIL import Image, ImageDraw as PilDraw, ImageFont

        img = Image.new("RGB", (width, height), color=(248, 248, 248))
        draw = PilDraw.Draw(img)
        draw.rectangle([0, 0, width - 1, height - 1], outline=(204, 204, 204))
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((width // 2, height // 2), message, fill=(153, 153, 153),
                  anchor="mm", font=font)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # Minimal 1×1 transparent PNG as absolute fallback
        return (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
            b'\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
            b'\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        )


def _build_legend(row: dict[str, Any], show_scores: bool) -> str:
    """Build a multi-line legend string for one compound."""
    parts: list[str] = [f"Rank {row['rank']}"]
    if show_scores:
        if "composite_score" in row and row["composite_score"] is not None:
            parts.append(f"Score: {row['composite_score']:.3f}")
        if "zscore" in row and row["zscore"] is not None:
            parts.append(f"z={row['zscore']:.2f}")
    return "\n".join(parts)


def _get_highlight_atoms(mol, smarts: str) -> list[int]:
    """Return atom indices matching *smarts* in *mol*, or empty list."""
    if mol is None:
        return []
    with BlockLogs():
        pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        logger.warning("Invalid SMARTS pattern: %r", smarts)
        return []
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return []
    # Flatten all match atom sets
    seen: set[int] = set()
    for match in matches:
        seen.update(match)
    return sorted(seen)


def _pil_to_bytes(pil_image) -> bytes:
    """Convert a PIL Image to PNG bytes."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def _write_output(data: str | bytes, path: Path, img_format: str) -> None:
    """Write *data* to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if img_format == "svg":
        path.write_text(data, encoding="utf-8")  # type: ignore[arg-type]
    else:
        path.write_bytes(data)  # type: ignore[arg-type]
    logger.info("Structure image written to %s", path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_hit_grid(
    ranked_df: pl.DataFrame,
    smiles_col: str = "smiles",
    top_n: int = 20,
    output_path: Path | str | None = None,
    img_format: str = "svg",
    cols_per_row: int = 4,
    cell_size: tuple[int, int] = (350, 300),
    highlight_substructure: str | None = None,
    show_scores: bool = True,
) -> str | bytes:
    """Render a grid image of the top-N ranked hit structures.

    Compounds are taken from the first *top_n* rows of *ranked_df* (which
    must already be sorted by rank ascending, as produced by
    :func:`~delexplore.analyse.rank.compute_composite_rank`).  Invalid SMILES
    are replaced by a grey placeholder cell so the grid always has exactly
    *top_n* entries (or fewer if the DataFrame has fewer rows).

    Args:
        ranked_df: Hit list DataFrame.  Must contain a ``"rank"`` column and
            *smiles_col*.  Optional enrichment columns ``"composite_score"``
            and ``"zscore"`` are used in legends if present.
        smiles_col: Name of the column holding SMILES strings.
        top_n: Maximum number of compounds to include.
        output_path: If provided, the image is written to this path.  The
            file extension is ignored — the format is controlled by
            *img_format*.
        img_format: ``"svg"`` (default) or ``"png"``.
        cols_per_row: Number of molecule cells per grid row.
        cell_size: ``(width, height)`` in pixels for each molecule cell.
        highlight_substructure: Optional SMARTS pattern.  Matching atoms are
            highlighted in every structure where the pattern is found.
        show_scores: Whether to include enrichment scores in the legend.

    Returns:
        SVG string when ``img_format="svg"``, PNG bytes when
        ``img_format="png"``.

    Raises:
        ImportError: If RDKit is not installed.
        ValueError: If ``"rank"`` or *smiles_col* is absent from *ranked_df*.
    """
    _require_rdkit()

    if "rank" not in ranked_df.columns:
        raise ValueError(
            "'rank' column not found. Run compute_composite_rank first."
        )
    if smiles_col not in ranked_df.columns:
        raise ValueError(
            f"Column '{smiles_col}' not found. Available: {ranked_df.columns}"
        )

    if len(ranked_df) == 0:
        empty: str | bytes = _placeholder_svg(*cell_size, "No compounds") \
            if img_format == "svg" else _placeholder_png(*cell_size, "No compounds")
        if output_path is not None:
            _write_output(empty, Path(output_path), img_format)
        return empty

    subset = ranked_df.sort("rank").head(top_n)
    rows = subset.to_dicts()

    mols: list[Any] = []
    legends: list[str] = []
    highlight_lists: list[list[int]] = []
    placeholder_indices: set[int] = set()

    for idx, row in enumerate(rows):
        smiles = row.get(smiles_col)
        mol = _parse_mol(smiles)

        if mol is None:
            logger.warning(
                "Invalid SMILES at rank %s: %r — using placeholder cell",
                row.get("rank"), smiles,
            )
            mols.append(Chem.MolFromSmiles("C"))  # invisible dummy
            placeholder_indices.add(idx)
            highlight_lists.append([])
            legends.append(f"Rank {row.get('rank', '?')}\n[Invalid SMILES]")
            continue

        mols.append(mol)
        legends.append(_build_legend(row, show_scores))

        if highlight_substructure is not None:
            highlight_lists.append(_get_highlight_atoms(mol, highlight_substructure))
        else:
            highlight_lists.append([])

    use_svg = img_format == "svg"
    highlight_arg = highlight_lists if highlight_substructure is not None else None

    # Apply draw options by patching via drawOptions keyword if supported
    # (MolsToGridImage passes **kwargs to the underlying drawer in some versions)
    grid = Draw.MolsToGridImage(
        mols,
        molsPerRow=cols_per_row,
        subImgSize=cell_size,
        legends=legends,
        highlightAtomLists=highlight_arg,
        useSVG=use_svg,
    )

    if use_svg:
        result: str | bytes = grid  # already a str
    else:
        result = _pil_to_bytes(grid)

    if output_path is not None:
        _write_output(result, Path(output_path), img_format)

    return result


def render_single_structure(
    smiles: str,
    output_path: Path | str | None = None,
    img_format: str = "svg",
    size: tuple[int, int] = (400, 300),
    highlight_smarts: str | None = None,
) -> str | bytes:
    """Render a single molecule structure.

    Intended for use in the interactive dashboard where individual structures
    are rendered on demand.

    Args:
        smiles: A SMILES string.
        output_path: If provided, write the image to this path.
        img_format: ``"svg"`` (default) or ``"png"``.
        size: ``(width, height)`` in pixels.
        highlight_smarts: Optional SMARTS pattern; matching atoms are
            highlighted.

    Returns:
        SVG string or PNG bytes.

    Raises:
        ImportError: If RDKit is not installed.
    """
    _require_rdkit()

    mol = _parse_mol(smiles)
    if mol is None:
        logger.warning("Invalid SMILES for single structure render: %r", smiles)
        result: str | bytes
        if img_format == "svg":
            result = _placeholder_svg(*size)
        else:
            result = _placeholder_png(*size)
        if output_path is not None:
            _write_output(result, Path(output_path), img_format)
        return result

    highlight_atoms: list[int] | None = None
    if highlight_smarts is not None:
        atoms = _get_highlight_atoms(mol, highlight_smarts)
        highlight_atoms = atoms if atoms else None

    w, h = size
    if img_format == "svg":
        drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(w, h)

    drawer.drawOptions().legendFontSize = 16
    drawer.drawOptions().bondLineWidth = 1.5

    if highlight_atoms:
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
    else:
        drawer.DrawMolecule(mol)

    drawer.FinishDrawing()

    if img_format == "svg":
        result = drawer.GetDrawingText()
    else:
        result = drawer.GetDrawingText()  # Cairo returns bytes

    if output_path is not None:
        _write_output(result, Path(output_path), img_format)

    return result


def smiles_to_svg_dict(
    df: pl.DataFrame,
    smiles_col: str = "smiles",
    id_col: str | None = None,
    size: tuple[int, int] = (250, 200),
) -> dict[str, str]:
    """Convert each SMILES in a DataFrame to an inline SVG string.

    Produces a dict suitable for embedding structures in an HTML dashboard.
    Invalid SMILES map to a placeholder SVG rather than raising.

    Args:
        df: Input DataFrame containing *smiles_col*.
        smiles_col: Name of the SMILES column.
        id_col: Column to use as dict keys.  If ``None``, the 0-based row
            index (as a string) is used.
        size: ``(width, height)`` in pixels for each SVG.

    Returns:
        Dict mapping compound ID (or row index string) → SVG string.

    Raises:
        ImportError: If RDKit is not installed.
        ValueError: If *smiles_col* is absent from *df*.
    """
    _require_rdkit()

    if smiles_col not in df.columns:
        raise ValueError(
            f"Column '{smiles_col}' not found. Available: {df.columns}"
        )
    if id_col is not None and id_col not in df.columns:
        raise ValueError(
            f"id_col '{id_col}' not found. Available: {df.columns}"
        )

    if len(df) == 0:
        return {}

    smiles_list = df[smiles_col].to_list()
    ids: list[str]
    if id_col is not None:
        ids = [str(v) for v in df[id_col].to_list()]
    else:
        ids = [str(i) for i in range(len(df))]

    result: dict[str, str] = {}
    w, h = size

    for key, smiles in zip(ids, smiles_list):
        mol = _parse_mol(smiles)
        if mol is None:
            result[key] = _placeholder_svg(w, h)
            continue

        drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
        drawer.drawOptions().legendFontSize = 14
        drawer.drawOptions().bondLineWidth = 1.5
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        result[key] = drawer.GetDrawingText()

    return result
