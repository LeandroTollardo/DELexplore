"""Consensus hit ranking using multi-method and multi-level scores.

Combines enrichment scores across methods (z-score, Poisson ML, DESeq2)
and synthon levels into a final composite rank with multi-level support scoring.
"""

import logging

logger = logging.getLogger(__name__)


def rank_hits() -> None:
    """Produce consensus hit ranking from multi-method enrichment scores.

    Not yet implemented.
    """
    raise NotImplementedError("consensus hit ranking is not yet implemented")
