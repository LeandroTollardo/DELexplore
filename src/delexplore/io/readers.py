"""Readers for DELT-Hit output formats.

Single point of entry for all input data. Handles:
- counts.txt: TSV selection count files (one per selection)
- config.yaml: experiment metadata and library definition
- library.parquet: optional enumerated library with SMILES

Auto-detects cycle count (n_cycles) from the number of code_* columns
present in the counts file.
"""

import logging
from pathlib import Path
from typing import Any

import polars as pl
import yaml

logger = logging.getLogger(__name__)


def read_counts(path: Path) -> pl.DataFrame:
    """Read a DELT-Hit counts TSV file into a Polars DataFrame.

    Detects the number of code columns (code_1, code_2, ..., code_N)
    automatically and attaches n_cycles as a custom attribute via the
    DataFrame's schema metadata comment (accessible via the returned
    object's ``attrs`` dict-like interface — stored as a plain attribute
    on the DataFrame so callers can do ``df.n_cycles``).

    Args:
        path: Path to a counts.txt file (tab-separated, with header).

    Returns:
        DataFrame with columns: code_1, ..., code_N, count, id.
        The returned DataFrame has an extra Python-level attribute
        ``n_cycles`` (int) set on the object for convenience.
    """
    path = Path(path)
    df = pl.read_csv(path, separator="\t")

    code_cols = [c for c in df.columns if c.startswith("code_")]
    n_cycles = len(code_cols)

    logger.info(
        "Loaded %d rows from %s (n_cycles=%d, code_cols=%s)",
        len(df),
        path,
        n_cycles,
        code_cols,
    )

    # Attach n_cycles as a plain Python attribute for callers to inspect.
    df.n_cycles = n_cycles  # type: ignore[attr-defined]
    return df


def read_config(path: Path) -> dict[str, Any]:
    """Read a DELT-Hit config.yaml file.

    Args:
        path: Path to config.yaml.

    Returns:
        Full config dict as parsed by yaml.safe_load.
    """
    path = Path(path)
    with path.open() as fh:
        config: dict[str, Any] = yaml.safe_load(fh)
    logger.info("Loaded config from %s (%d selections)", path, len(config.get("selections", {})))
    return config


def load_experiment(config_path: Path) -> dict[str, Any]:
    """Load a complete experiment: config + all selection count files.

    Searches for counts files relative to the config directory using the
    following priority order for each selection named in the config:
    1. ``{config_dir}/selections/{sel_name}/counts.txt``
    2. ``{config_dir}/{sel_name}/counts.txt``

    Selections whose counts file cannot be found are skipped with a warning.

    Args:
        config_path: Path to the experiment config.yaml.

    Returns:
        Dict with keys:
        - ``"config"``: the full parsed config dict.
        - ``"counts"``: combined Polars DataFrame with a leading
          ``selection`` column plus all code_*, count, id columns.
        - ``"n_cycles"``: int, detected from the first loaded counts file.
    """
    config_path = Path(config_path)
    config_dir = config_path.parent
    config = read_config(config_path)

    selections = config.get("selections", {})
    if not selections:
        logger.warning("No selections found in config %s", config_path)

    frames: list[pl.DataFrame] = []
    n_cycles_detected: int | None = None

    for sel_name in selections:
        candidates = [
            config_dir / "selections" / sel_name / "counts.txt",
            config_dir / sel_name / "counts.txt",
        ]
        counts_path: Path | None = next((p for p in candidates if p.exists()), None)

        if counts_path is None:
            logger.debug("No counts file found for selection %s — skipping", sel_name)
            continue

        sel_df = read_counts(counts_path)

        if n_cycles_detected is None:
            n_cycles_detected = sel_df.n_cycles  # type: ignore[attr-defined]
        elif sel_df.n_cycles != n_cycles_detected:  # type: ignore[attr-defined]
            logger.warning(
                "Selection %s has n_cycles=%d but expected %d",
                sel_name,
                sel_df.n_cycles,  # type: ignore[attr-defined]
                n_cycles_detected,
            )

        sel_df = sel_df.with_columns(pl.lit(sel_name).alias("selection"))
        # Reorder so selection is the first column.
        sel_df = sel_df.select(["selection"] + [c for c in sel_df.columns if c != "selection"])
        frames.append(sel_df)

    if frames:
        combined = pl.concat(frames, how="diagonal")
        n_cycles = n_cycles_detected or 0
        logger.info(
            "Combined %d selections into %d rows (n_cycles=%d)",
            len(frames),
            len(combined),
            n_cycles,
        )
    else:
        combined = pl.DataFrame()
        n_cycles = 0
        logger.warning("No count files were loaded for experiment at %s", config_path)

    return {"config": config, "counts": combined, "n_cycles": n_cycles}


def get_selection_metadata(config: dict[str, Any]) -> pl.DataFrame:
    """Extract per-selection metadata from a parsed config dict.

    Handles optional fields gracefully — fields absent in the config
    are filled with ``None``.

    Args:
        config: Parsed config dict as returned by :func:`read_config`.

    Returns:
        DataFrame with columns:
        selection_name, target, group, date, operator,
        beads, protocol.
    """
    selections: dict[str, Any] = config.get("selections", {})

    rows = []
    for sel_name, sel_info in selections.items():
        if not isinstance(sel_info, dict):
            continue
        target = sel_info.get("target")
        # DELT-Hit uses .nan for blank/no-protein conditions; normalise to None.
        if target != target or str(target).lower() in (".nan", "nan", "none", ""):
            target = None
        rows.append(
            {
                "selection_name": sel_name,
                "target": target,
                "group": sel_info.get("group"),
                "date": str(sel_info.get("date")) if sel_info.get("date") else None,
                "operator": sel_info.get("operator"),
                "beads": sel_info.get("beads"),
                "protocol": sel_info.get("protocol"),
            }
        )

    schema = {
        "selection_name": pl.Utf8,
        "target": pl.Utf8,
        "group": pl.Utf8,
        "date": pl.Utf8,
        "operator": pl.Utf8,
        "beads": pl.Utf8,
        "protocol": pl.Utf8,
    }

    if not rows:
        return pl.DataFrame(schema=schema)

    return pl.DataFrame(rows, schema=schema)
