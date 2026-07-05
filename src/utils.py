"""
utils.py

Small shared helpers used across the other src/ modules. Kept deliberately
minimal — this project's actual logic (cleaning, encoding, training,
inference, plotting) belongs in its own module; this file exists for the
few things genuinely shared across more than one of them.
"""

from pathlib import Path
from typing import Union

import joblib


def get_project_root() -> Path:
    """
    Resolve the repository root from this file's location (src/utils.py -> parent).

    Returns:
        Path to the repository root, assuming the standard
        app/ dataset/ images/ models/ notebooks/ reports/ src/ layout.
    """
    return Path(__file__).resolve().parent.parent


def save_artifact(obj, path: Union[str, Path]) -> None:
    """
    Persist a fitted model or scaler with joblib.

    Args:
        obj: The fitted estimator/transformer to save.
        path: Destination file path (e.g. models/telecom_churn_Project.pkl).
    """
    joblib.dump(obj, path)


def load_artifact(path: Union[str, Path]):
    """
    Load a previously persisted model or scaler.

    Args:
        path: Path to the .pkl file.

    Returns:
        The unpickled object.
    """
    return joblib.load(path)


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a 0-1 fraction (or 0-100 value that's actually already a percentage)
    as a display string. Assumes `value` is already a percentage (0-100), matching
    how this project's accuracy values are stored (e.g. 87.5617, not 0.875617).

    Args:
        value: Percentage value, e.g. 87.5617.
        decimals: Number of decimal places to show.

    Returns:
        Formatted string, e.g. "87.56%".
    """
    return f"{value:.{decimals}f}%"
