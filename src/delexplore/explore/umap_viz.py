"""UMAP projections of compound chemical space.

Uses Morgan fingerprints (radius=2, nBits=2048) with Jaccard distance
and UMAP for dimensionality reduction, colored by enrichment scores.
"""

import logging

logger = logging.getLogger(__name__)


def compute_umap_embedding() -> None:
    """Compute UMAP embedding of compound fingerprints.

    Not yet implemented.
    """
    raise NotImplementedError("UMAP visualization is not yet implemented")
