"""Data quality assessment and HTML report generation.

Produces data_quality.json consumed by the analysis module.
"""

import logging

logger = logging.getLogger(__name__)


def assess_quality() -> None:
    """Assess DEL data quality and write data_quality.json.

    Not yet implemented.
    """
    raise NotImplementedError("qc assess is not yet implemented")
