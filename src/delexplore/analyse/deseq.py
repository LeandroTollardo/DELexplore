"""PyDESeq2 negative binomial GLM for DEL enrichment analysis.

Runs at mono- and disynthon level for performance. Requires pandas
at the PyDESeq2 boundary; all other data handling uses Polars.
"""

import logging

logger = logging.getLogger(__name__)


def run_deseq2() -> None:
    """Run PyDESeq2 negative binomial GLM enrichment analysis.

    Not yet implemented.
    """
    raise NotImplementedError("DESeq2 analysis is not yet implemented")
