"""
prediction.py

Inference-time functions: take a single new customer record, apply the exact
same encoding + scaling used in training, and return the deployed Random
Forest model's prediction. This mirrors the logic already implemented and
verified in app.py (see that file's header docstring for the bit-for-bit
reproduction note: accuracy 87.56168037341632%, confusion matrix
[[17248,742],[2056,2449]]).
"""

from pathlib import Path
from typing import Dict, Union

import joblib
import pandas as pd

from feature_engineering import align_features_to_reference, encode_categorical_features


def load_artifacts(model_dir: Union[str, Path]) -> Dict[str, object]:
    """
    Load the trained model and fitted scaler from disk.

    Args:
        model_dir: Directory containing telecom_churn_Project.pkl and Scaler.pkl.

    Returns:
        Dict with keys 'model' and 'scaler'.
    """
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "telecom_churn_Project.pkl")
    scaler = joblib.load(model_dir / "Scaler.pkl")
    return {"model": model, "scaler": scaler}


def prepare_single_input(raw_input: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Encode and align a single new customer row for inference.

    Steps (must match training exactly):
        1. One-hot encode categorical fields.
        2. Reindex to the scaler's known column order (scaler.feature_names_in_),
           filling any category the scaler saw during training but this input
           didn't produce with 0.

    Args:
        raw_input: A single-row (or small batch) DataFrame of raw customer
            fields, in the same shape as the training data minus the target
            column and Customer_ID.
        scaler: The fitted StandardScaler loaded via load_artifacts().

    Returns:
        Encoded, column-aligned DataFrame ready for scaler.transform().
    """
    encoded = encode_categorical_features(raw_input)
    aligned = align_features_to_reference(encoded, list(scaler.feature_names_in_))
    return aligned


def predict_churn(raw_input: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """
    Run the full inference pipeline on new customer input.

    Args:
        raw_input: Raw customer row(s), unencoded.
        model: The loaded RandomForestClassifier.
        scaler: The loaded, fitted StandardScaler.

    Returns:
        DataFrame with columns ['prediction', 'churn_probability'] — prediction
        is 0 (retain) or 1 (churn); churn_probability is model.predict_proba
        for the churn class.
    """
    aligned = prepare_single_input(raw_input, scaler)
    scaled = scaler.transform(aligned)
    prediction = model.predict(scaled)
    probability = model.predict_proba(scaled)[:, 1]
    return pd.DataFrame({"prediction": prediction, "churn_probability": probability})


def risk_level_from_probability(probability: float) -> str:
    """
    Map a churn probability to a human-readable risk tier for business users.

    Thresholds are a reasonable default (Low < 0.33, Medium < 0.66, else High) —
    not derived from the notebook, since the notebook never defines risk bands.
    Adjust these thresholds against real retention-team feedback before treating
    them as calibrated.

    Args:
        probability: Predicted probability of churn (0.0-1.0).

    Returns:
        One of "Low", "Medium", "High".
    """
    if probability < 0.33:
        return "Low"
    if probability < 0.66:
        return "Medium"
    return "High"
