"""Multi-level synthon aggregation engine.

Aggregates compound counts to mono-, di-, and trisynthon levels for
N-cycle DEL libraries. Each synthon level gets independent enrichment
analysis with its own diversity parameter.
"""

import logging

logger = logging.getLogger(__name__)


def aggregate_to_synthon_level() -> None:
    """Aggregate compound counts to a specified synthon level.

    Not yet implemented.
    """
    raise NotImplementedError("synthon aggregation is not yet implemented")
