"""
data_preprocessing.py

Data loading and cleaning functions extracted directly from the project notebook
(Telecom_Churn_Prediction.ipynb, cells covering identifier removal, missing-value
imputation, and duplicate detection).

NOTE ON FIDELITY:
The notebook checks for duplicate records (df.duplicated().sum() -> 538 rows) but
never calls df.drop_duplicates(). The model currently deployed was trained on data
that still contains those 538 duplicate rows. `check_duplicates()` below mirrors
that — it reports duplicates, it does not remove them. Do not add a silent
drop_duplicates() call here: doing so would change the training data and therefore
the model's behavior, which the task explicitly prohibits.
"""

from pathlib import Path
from typing import Union

import pandas as pd


def load_raw_data(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the raw telecom churn dataset from a CSV file.

    Args:
        path: Path to telecom_churn_data.csv.

    Returns:
        Raw, unmodified DataFrame as read from disk.
    """
    return pd.read_csv(path)


def drop_identifier_column(df: pd.DataFrame, id_column: str = "Customer_ID") -> pd.DataFrame:
    """
    Drop the customer identifier column.

    A unique identifier carries no predictive signal and risks the model
    learning row-order or ID-based artifacts instead of genuine patterns.

    Args:
        df: Input DataFrame.
        id_column: Name of the identifier column to drop.

    Returns:
        DataFrame with the identifier column removed.
    """
    return df.drop([id_column], axis=1)


def check_duplicates(df: pd.DataFrame) -> int:
    """
    Count duplicate rows in the dataset without removing them.

    Mirrors the notebook's actual behavior (detection only). See module
    docstring for why this function intentionally does not drop duplicates.

    Args:
        df: Input DataFrame.

    Returns:
        Number of fully duplicated rows.
    """
    return int(df.duplicated().sum())


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Return a per-column count of missing values, sorted descending.

    Args:
        df: Input DataFrame.

    Returns:
        Series indexed by column name, sorted by missing-value count descending.
    """
    return df.isna().sum().sort_values(ascending=False)


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in place of the notebook's cleaning step:
    numerical columns filled with the column median, categorical (object)
    columns filled with the column mode.

    Args:
        df: Input DataFrame (post identifier-drop).

    Returns:
        DataFrame with missing values imputed. Operates on a copy.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["int64", "float64"]):
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def clean_dataset(path: Union[str, Path], id_column: str = "Customer_ID") -> pd.DataFrame:
    """
    Full cleaning pipeline in the exact order the notebook performs it:
    load -> drop identifier -> impute missing values.

    Duplicate rows are intentionally left in place — see module docstring.

    Args:
        path: Path to the raw CSV file.
        id_column: Name of the identifier column to drop.

    Returns:
        Cleaned DataFrame, ready for encoding (see feature_engineering.py).
    """
    df = load_raw_data(path)
    df = drop_identifier_column(df, id_column)
    df = impute_missing_values(df)
    return df
