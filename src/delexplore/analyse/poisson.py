"""Poisson confidence interval and ML enrichment (Kuai 2018, Hou 2023).

Implements exact chi-squared Poisson CI and maximum-likelihood enrichment
fold with continuity correction.
"""

import logging

logger = logging.getLogger(__name__)


def calculate_poisson_ci() -> None:
    """Calculate exact Poisson confidence interval via chi-squared quantiles.

    Not yet implemented.
    """
    raise NotImplementedError("Poisson CI is not yet implemented")


def calculate_poisson_ml_enrichment() -> None:
    """Calculate maximum-likelihood enrichment fold (Hou et al. 2023).

    Not yet implemented.
    """
    raise NotImplementedError("Poisson ML enrichment is not yet implemented")
