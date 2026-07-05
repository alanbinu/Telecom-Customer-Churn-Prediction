"""
visualization.py

Chart-generation functions for the project's reporting visuals. These
reproduce the notebook's own plots (churn distribution, correlation heatmap,
feature importance, model comparison, confusion matrix, ROC curve) using a
consistent dark-navy / royal-blue theme so every image in reports/ and the
README looks like it belongs to the same project.
"""

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

NAVY = "#0B1220"
CARD = "#111A2E"
ROYAL = "#3B82F6"
ROYAL_LT = "#60A5FA"
WHITE = "#F1F5F9"
GREY = "#94A3B8"
RED = "#EF4444"


def _style_axes(ax, fig) -> None:
    """Apply the shared dark theme to a matplotlib Axes/Figure pair."""
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=GREY, labelsize=10)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)


def plot_churn_distribution(churn_counts: dict, save_path: str) -> None:
    """
    Bar chart of retained vs. churned customer counts.

    Args:
        churn_counts: Dict like {"Retained": 60027, "Churned": 14956}.
        save_path: File path to save the PNG to.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    total = sum(churn_counts.values())
    labels = [f"{k}\n{v / total * 100:.2f}%" for k, v in churn_counts.items()]
    bars = ax.bar(labels, list(churn_counts.values()), color=[ROYAL, RED], width=0.5)
    for b, v in zip(bars, churn_counts.values()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + total * 0.01,
                f"{v:,}", ha="center", color=WHITE, fontsize=13, fontweight="bold")
    ax.set_title(f"Customer Churn Distribution  (n = {total:,})", fontsize=15,
                 fontweight="bold", pad=20)
    ax.set_ylabel("Customers")
    _style_axes(ax, fig)
    ax.grid(axis="y", color="#1E293B", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(save_path, facecolor=NAVY)
    plt.close()


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, save_path: str) -> None:
    """
    Heatmap of a (typically pre-filtered, top-N) correlation matrix.

    Args:
        corr_matrix: A square correlation DataFrame (e.g. corr[top_features]).
        save_path: File path to save the PNG to.
    """
    fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
    im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", color=GREY, fontsize=8)
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_yticklabels(corr_matrix.index, color=GREY, fontsize=8)
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            ax.text(j, i, f"{corr_matrix.values[i, j]:.2f}", ha="center", va="center",
                    color=WHITE, fontsize=6.5)
    ax.set_title("Correlation Matrix — Top Features vs Churn_Flag", fontsize=14,
                 fontweight="bold", color=WHITE, pad=15)
    fig.patch.set_facecolor(NAVY)
    fig.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(save_path, facecolor=NAVY)
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame, save_path: str, top_n: int = 10) -> None:
    """
    Horizontal bar chart of the top-N Random Forest feature importances.

    Args:
        importance_df: DataFrame with columns ['Feature', 'Importance']
            (output of model_training.get_feature_importance).
        save_path: File path to save the PNG to.
        top_n: Number of top features to display.
    """
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    bars = ax.barh(top["Feature"][::-1], top["Importance"][::-1], color=ROYAL)
    for b, v in zip(bars, top["Importance"][::-1]):
        ax.text(v + 0.001, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center",
                color=WHITE, fontsize=9)
    ax.set_title(f"Random Forest — Top {top_n} Feature Importance", fontsize=15,
                 fontweight="bold", pad=15)
    ax.set_xlabel("Importance Score")
    _style_axes(ax, fig)
    ax.grid(axis="x", color="#1E293B", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(save_path, facecolor=NAVY)
    plt.close()


def plot_model_comparison(accuracy_df: pd.DataFrame, save_path: str) -> None:
    """
    Bar chart comparing test accuracy across all evaluated models.

    Args:
        accuracy_df: DataFrame with columns ['Model', 'Accuracy'] (percentage),
            e.g. output of model_training.evaluate_models().
        save_path: File path to save the PNG to.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    colors = [ROYAL_LT if m == "Random Forest" else ROYAL for m in accuracy_df["Model"]]
    bars = ax.bar(accuracy_df["Model"], accuracy_df["Accuracy"], color=colors)
    for b, v in zip(bars, accuracy_df["Accuracy"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.2f}%", ha="center",
                color=WHITE, fontsize=10, fontweight="bold")
    ax.set_ylim(max(0, accuracy_df["Accuracy"].min() - 15), 100)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Model Comparison — Test Set Accuracy", fontsize=15, fontweight="bold", pad=15)
    plt.xticks(rotation=15, ha="right")
    _style_axes(ax, fig)
    ax.grid(axis="y", color="#1E293B", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(save_path, facecolor=NAVY)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, accuracy: float, save_path: str,
                           class_labels: Sequence[str] = ("No Churn", "Churn")) -> None:
    """
    Render a 2x2 confusion matrix with counts and an accuracy subtitle.

    Args:
        cm: 2x2 confusion matrix array, e.g. sklearn.metrics.confusion_matrix output.
        accuracy: Overall accuracy (0-1) to display in the title.
        save_path: File path to save the PNG to.
        class_labels: Labels for the negative/positive classes.
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Predicted: {c}" for c in class_labels], color=GREY)
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"Actual: {c}" for c in class_labels], color=GREY)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color=WHITE if cm[i, j] < cm.max() * 0.6 else NAVY,
                     fontsize=16, fontweight="bold")
    ax.set_title(f"Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%", fontsize=14,
                 fontweight="bold", color=WHITE, pad=15)
    fig.patch.set_facecolor(NAVY)
    plt.tight_layout()
    plt.savefig(save_path, facecolor=NAVY)
    plt.close()


def plot_roc_curve(y_test, y_proba, save_path: str, model_name: str = "Random Forest") -> None:
    """
    Plot the ROC curve and annotate the AUC.

    Args:
        y_test: True binary labels.
        y_proba: Predicted probability of the positive (churn) class.
        save_path: File path to save the PNG to.
        model_name: Label used in the legend.
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    ax.plot(fpr, tpr, color=ROYAL_LT, linewidth=2.5, label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.15, color=ROYAL)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=15, fontweight="bold", pad=15)
    ax.legend(loc="lower right", facecolor=CARD, edgecolor="none", labelcolor=WHITE, fontsize=11)
    _style_axes(ax, fig)
    ax.grid(color="#1E293B", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(save_path, facecolor=NAVY)
    plt.close()
