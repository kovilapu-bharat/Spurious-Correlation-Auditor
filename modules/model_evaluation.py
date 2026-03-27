"""
Model Evaluation Module
-----------------------
Train selected classifiers and compute standard evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

from modules.data_input import AuditData

# ---------------------------------------------------------------------------
# Supported models
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, Any] = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None  # None for multi-class without OVR
    report_text: str


@dataclass
class ModelReport:
    model_name: str
    model: Any  # trained estimator
    metrics: ModelMetrics
    y_pred: np.ndarray
    y_proba: np.ndarray | None


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def train_model(model_name: str, data: AuditData) -> ModelReport:
    """Train a model from the registry and evaluate on the test set."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    from sklearn.base import clone
    estimator = clone(MODEL_REGISTRY[model_name])
    estimator.fit(data.X_train, data.y_train)

    y_pred = estimator.predict(data.X_test)

    # probabilities
    y_proba = None
    if hasattr(estimator, "predict_proba"):
        y_proba = estimator.predict_proba(data.X_test)

    # --- metrics ---
    n_classes = len(data.class_names)
    average = "binary" if n_classes == 2 else "weighted"

    acc = accuracy_score(data.y_test, y_pred)
    prec = precision_score(data.y_test, y_pred, average=average, zero_division=0)
    rec = recall_score(data.y_test, y_pred, average=average, zero_division=0)
    f1 = f1_score(data.y_test, y_pred, average=average, zero_division=0)

    roc = None
    if y_proba is not None:
        try:
            if n_classes == 2:
                roc = roc_auc_score(data.y_test, y_proba[:, 1])
            else:
                roc = roc_auc_score(
                    data.y_test, y_proba, multi_class="ovr", average="weighted"
                )
        except ValueError:
            roc = None

    report_text = classification_report(data.y_test, y_pred, zero_division=0)

    metrics = ModelMetrics(
        accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=roc,
        report_text=report_text,
    )

    return ModelReport(
        model_name=model_name, model=estimator, metrics=metrics,
        y_pred=y_pred, y_proba=y_proba,
    )


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(report: ModelReport, data: AuditData) -> plt.Figure:
    """Return a styled confusion-matrix figure."""
    cm = confusion_matrix(data.y_test, report.y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {report.model_name}", fontweight="bold", pad=10)
    fig.tight_layout()
    return fig


def plot_roc_curve(report: ModelReport, data: AuditData) -> plt.Figure | None:
    """Return an ROC-curve figure (binary classification only)."""
    if report.y_proba is None or len(data.class_names) != 2:
        return None
    fpr, tpr, _ = roc_curve(data.y_test, report.y_proba[:, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#6C63FF", lw=2, label=f"AUC = {report.metrics.roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {report.model_name}", fontweight="bold", pad=10)
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    return fig
