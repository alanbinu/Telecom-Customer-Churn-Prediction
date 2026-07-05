"""
model_training.py

Model training and comparison functions extracted from the notebook's modeling
cells. Hyperparameters match the notebook exactly (mostly library defaults,
with random_state=42 set only where the notebook set it — DecisionTree,
RandomForest, and SVC. LogisticRegression was left at library defaults with
no random_state in the notebook; that is preserved here rather than "fixed",
since changing it would change results and this task excludes retraining.
"""

from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def split_data(
    x: pd.DataFrame, y: pd.Series, test_size: float = 0.30, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train/test split matching the notebook exactly: 70/30, random_state=42.

    Args:
        x: Encoded feature DataFrame.
        y: Target Series.
        test_size: Fraction held out for testing (notebook uses 0.30).
        random_state: Seed for reproducibility (notebook uses 42).

    Returns:
        x_train, x_test, y_train, y_test
    """
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def scale_features(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on the training set and transform both splits.

    Args:
        x_train: Training features.
        x_test: Test features.

    Returns:
        Tuple of (scaled x_train, scaled x_test, fitted scaler instance).
        The fitted scaler should be persisted (joblib.dump) so inference-time
        code can apply the identical transform to new customer input.
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler


def train_all_models(x_train, y_train) -> Dict[str, object]:
    """
    Train all six classifiers evaluated in the notebook, with identical
    hyperparameters (library defaults except where the notebook set
    random_state=42).

    Args:
        x_train: Scaled training features.
        y_train: Training target.

    Returns:
        Dict mapping model name -> fitted estimator.
    """
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(random_state=42),  # notebook never sets probability=True — SVM has no predict_proba here
        "Naive Bayes": GaussianNB(),
    }
    for model in models.values():
        model.fit(x_train, y_train)
    return models


def evaluate_models(models: Dict[str, object], x_test, y_test) -> pd.DataFrame:
    """
    Compute test-set accuracy for each trained model, sorted descending.

    Args:
        models: Dict of model name -> fitted estimator (from train_all_models).
        x_test: Scaled test features.
        y_test: True test labels.

    Returns:
        DataFrame with columns ['Model', 'Accuracy'] (Accuracy as a percentage),
        sorted descending — matches the notebook's comparison table (cell 77).
    """
    rows = []
    for name, model in models.items():
        y_pred = model.predict(x_test)
        rows.append({"Model": name, "Accuracy": accuracy_score(y_test, y_pred) * 100})
    return pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)


def get_feature_importance(rf_model: RandomForestClassifier, feature_names) -> pd.DataFrame:
    """
    Extract and rank Random Forest feature importances.

    Args:
        rf_model: A fitted RandomForestClassifier.
        feature_names: Column names in the same order used for training
            (e.g. x_train.columns if x_train is a DataFrame, or
            scaler.feature_names_in_ if working from the persisted scaler).

    Returns:
        DataFrame with columns ['Feature', 'Importance'], sorted descending.
    """
    importance = pd.DataFrame(
        {"Feature": feature_names, "Importance": rf_model.feature_importances_}
    )
    return importance.sort_values(by="Importance", ascending=False).reset_index(drop=True)
