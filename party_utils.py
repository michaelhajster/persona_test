"""
Utility functions for party name mapping and validation.
"""

from typing import Dict, Set

# Mapping from internal keys to display labels
PARTY_KEY_TO_LABEL: Dict[str, str] = {
    "cdu_csu": "CDU/CSU",
    "spd": "SPD",
    "gruene": "GRÜNE",
    "fdp": "FDP",
    "linke": "LINKE",
    "afd": "AfD"
}

# Mapping from display labels to internal keys
PARTY_LABEL_TO_KEY: Dict[str, str] = {v: k for k, v in PARTY_KEY_TO_LABEL.items()}

# Set of valid party labels for validation
VALID_PARTY_LABELS: Set[str] = set(PARTY_KEY_TO_LABEL.values())

def party_key_to_label(key: str) -> str:
    """Convert internal party key to display label."""
    return PARTY_KEY_TO_LABEL[key]

def party_label_to_key(label: str) -> str:
    """Convert display label to internal party key."""
    return PARTY_LABEL_TO_KEY[label]

def validate_party_label(label: str) -> bool:
    """Validate if a party label is valid."""
    return label in VALID_PARTY_LABELS 