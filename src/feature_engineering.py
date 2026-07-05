"""
feature_engineering.py

NOTE ON SCOPE:
The notebook does not construct any derived or engineered features (no binning,
no ratio features, no date/time decomposition). The only feature transformation
that occurs is categorical encoding via pandas get_dummies. This module is
intentionally thin — it reflects what the notebook actually does, rather than
inventing engineered features that don't exist in the original pipeline.

If you want real feature engineering in a future iteration (tenure buckets,
ARPU-to-income ratio, complaint-rate-per-month, etc.), that is a genuine next
step worth adding to Future Improvements — but it would change model inputs
and therefore model performance, which this task explicitly excludes.
"""

from typing import List, Optional

import pandas as pd


def encode_categorical_features(x: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """
    One-hot encode all categorical (object) columns.

    Mirrors the notebook's `pd.get_dummies(x, drop_first=True)` call exactly.

    Args:
        x: Feature DataFrame (target column already separated out).
        drop_first: Whether to drop the first category per column, matching
            the notebook's default to avoid the dummy-variable trap.

    Returns:
        Encoded DataFrame with categorical columns expanded to binary indicators.
    """
    return pd.get_dummies(x, drop_first=drop_first)


def align_features_to_reference(
    x_encoded: pd.DataFrame, reference_columns: List[str]
) -> pd.DataFrame:
    """
    Reindex an encoded feature set to a fixed, known column order.

    This is required at inference time: a single customer's input, once
    one-hot encoded, will not contain every category level seen during
    training. Missing columns are filled with 0 (that category was not
    present in this input); unexpected extra columns are dropped.

    Args:
        x_encoded: Newly encoded feature DataFrame (e.g. one prediction row).
        reference_columns: The exact column order the scaler/model was fit on
            (typically `scaler.feature_names_in_`).

    Returns:
        DataFrame reindexed to reference_columns, missing values filled with 0.
    """
    return x_encoded.reindex(columns=reference_columns, fill_value=0)


def split_features_and_target(
    df: pd.DataFrame, target_column: str = "Churn_Flag"
) -> "tuple[pd.DataFrame, pd.Series]":
    """
    Split a cleaned DataFrame into features (x) and target (y).

    Matches the notebook's `x = df.iloc[:,:-1]; y = df.iloc[:,-1]` — assumes
    the target column is the last column, consistent with the source dataset.

    Args:
        df: Cleaned DataFrame with the target as the final column.
        target_column: Name of the target column, used only for validation.

    Returns:
        Tuple of (feature DataFrame, target Series).
    """
    assert df.columns[-1] == target_column, (
        f"Expected last column to be '{target_column}', got '{df.columns[-1]}'. "
        "The notebook relies on positional slicing (iloc[:, :-1]) — column order matters."
    )
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return x, y
