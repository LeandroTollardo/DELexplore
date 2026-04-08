"""Chemistry utilities: fingerprints, property calculation, scaffold decomposition.

All RDKit operations are centralised here to keep chemistry concerns
out of the analysis modules.
"""

import logging

logger = logging.getLogger(__name__)


def smiles_to_fingerprint() -> None:
    """Convert a SMILES string to a Morgan fingerprint bit vector.

    Not yet implemented.
    """
    raise NotImplementedError("fingerprint generation is not yet implemented")


def calculate_properties() -> None:
    """Calculate Lipinski/drug-likeness properties for a compound.

    Not yet implemented.
    """
    raise NotImplementedError("property calculation is not yet implemented")
