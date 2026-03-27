"""
Spurious Correlation Detection Module
--------------------------------------
Identifies features that may be driving predictions through spurious
relationships using SHAP, LIME, correlation analysis, and feature perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

from modules.data_input import AuditData
from modules.model_evaluation import ModelReport

# ---------------------------------------------------------------------------
# Subsampling constants — tune these for speed vs fidelity trade-offs
# ---------------------------------------------------------------------------
MAX_SHAP_SAMPLES: int = 1_000       # Max test samples for SHAP computation
MAX_DROP_TEST_SAMPLES: int = 10_000  # Max train samples per feature-drop retrain

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FeatureAnalysis:
    """Analysis result for a single feature."""
    name: str
    shap_importance: float
    target_correlation: float
    mutual_info: float
    drop_accuracy_change: float  # accuracy change when feature is removed
    risk_level: str  # "High", "Medium", "Low"
    reason: str


@dataclass
class SpuriousReport:
    """Full spurious-correlation analysis report."""
    feature_analyses: list[FeatureAnalysis]
    shap_values: np.ndarray | None = None
    overall_risk: str = "Low"
    flagged_count: int = 0


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------

def _subsample(X: np.ndarray, max_n: int, rng_seed: int = 42) -> np.ndarray:
    """Return a random subsample of rows if X exceeds *max_n* rows."""
    if len(X) <= max_n:
        return X
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx]


def _subsample_Xy(
    X: np.ndarray, y: np.ndarray, max_n: int, rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample both X and y together, preserving row alignment."""
    if len(X) <= max_n:
        return X, y
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx], y[idx]


def _compute_shap_importance(report: ModelReport, data: AuditData) -> np.ndarray:
    """Compute mean |SHAP| importance per feature using permutation importance as a robust fallback."""
    try:
        import shap
        model = report.model

        # Subsample test data for SHAP — statistically equivalent rankings
        X_test_sample = _subsample(data.X_test, MAX_SHAP_SAMPLES)

        # Use TreeExplainer for tree-based models
        if hasattr(model, "estimators_") or hasattr(model, "tree_"):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test_sample)
        else:
            # KernelExplainer for others — use a small background sample optimized for speed (25)
            bg = shap.sample(pd.DataFrame(data.X_train, columns=data.feature_names), min(25, len(data.X_train)))
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            # Reduce test samples significantly (from 50 to 15) to prevent hanging Streamlit ops
            shap_vals = explainer.shap_values(X_test_sample[:min(15, len(X_test_sample))])

        # Handle shap.Explanation objects (newer SHAP versions)
        if hasattr(shap_vals, 'values'):
            shap_vals = shap_vals.values

        # shap_vals may be a list (one per class) — average across classes
        if isinstance(shap_vals, list):
            shap_vals = np.mean([np.abs(np.array(sv)) for sv in shap_vals], axis=0)
        else:
            shap_vals = np.abs(np.array(shap_vals))

        # Ensure 2D then mean across samples
        if shap_vals.ndim == 3:
            # Shape (n_samples, n_features, n_classes) — mean over classes then samples
            shap_vals = np.mean(np.abs(shap_vals), axis=2)
        if shap_vals.ndim == 2:
            result = np.mean(shap_vals, axis=0)
        else:
            result = shap_vals

        return result.flatten().astype(float)
    except Exception:
        # Fallback: permutation importance (also subsampled)
        X_test_sub, y_test_sub = _subsample_Xy(
            data.X_test, data.y_test, MAX_SHAP_SAMPLES,
        )
        result = permutation_importance(
            report.model, X_test_sub, y_test_sub,
            n_repeats=10, random_state=42, scoring="accuracy",
        )
        importances = result.importances_mean
        # Normalise to 0-1
        total = importances.sum()
        return (importances / total if total > 0 else importances).astype(float)


def _compute_shap_values_for_plot(report: ModelReport, data: AuditData):
    """Return raw SHAP values for visualisation (may return None on failure)."""
    try:
        import shap
        model = report.model
        if hasattr(model, "estimators_") or hasattr(model, "tree_"):
            explainer = shap.TreeExplainer(model)
            # Subsample for plot — keeps visualisation responsive
            X_plot = _subsample(data.X_test, MAX_SHAP_SAMPLES)
            return explainer.shap_values(X_plot)
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def _compute_target_correlations(data: AuditData) -> np.ndarray:
    """Pearson correlation between each feature and the target (absolute)."""
    correlations = []
    for i in range(data.X_train.shape[1]):
        corr, _ = pearsonr(data.X_train[:, i], data.y_train)
        correlations.append(abs(corr))
    return np.array(correlations)


def _compute_mutual_information(data: AuditData) -> np.ndarray:
    """Mutual information between each feature and the target."""
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(data.X_train, data.y_train, random_state=42)
    return mi


# ---------------------------------------------------------------------------
# Feature perturbation (drop-column test)
# ---------------------------------------------------------------------------

def _feature_drop_test(report: ModelReport, data: AuditData) -> np.ndarray:
    """
    For each feature, retrain the model without it and measure accuracy change.
    A large *drop* in accuracy from removing a low-correlation feature may
    indicate spurious reliance.

    Training is subsampled to MAX_DROP_TEST_SAMPLES for speed on large datasets.
    """
    base_acc = accuracy_score(data.y_test, report.y_pred)
    accuracy_changes = []

    # Subsample training data for faster retraining per feature
    X_train_sub, y_train_sub = _subsample_Xy(
        data.X_train, data.y_train, MAX_DROP_TEST_SAMPLES,
    )

    for i in range(data.X_train.shape[1]):
        X_train_drop = np.delete(X_train_sub, i, axis=1)
        X_test_drop = np.delete(data.X_test, i, axis=1)

        estimator = clone(report.model)
        estimator.fit(X_train_drop, y_train_sub)
        acc_drop = accuracy_score(data.y_test, estimator.predict(X_test_drop))
        accuracy_changes.append(base_acc - acc_drop)

    return np.array(accuracy_changes)


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

def _classify_risk(
    shap_imp: float,
    target_corr: float,
    mi: float,
    acc_change: float,
    shap_threshold: float = 0.5,
    corr_threshold: float = 0.15,
) -> tuple[str, str]:
    """
    Classify a feature's spurious-correlation risk.

    Heuristic:
    - High importance (SHAP rank) + low target correlation → potentially spurious
    """
    # normalise shap_imp to a relative scale later; here we use raw
    if shap_imp > shap_threshold and target_corr < corr_threshold:
        return "High", (
            f"High model importance ({shap_imp:.3f}) but very weak target "
            f"correlation ({target_corr:.3f}) — likely spurious."
        )
    if shap_imp > shap_threshold * 0.5 and target_corr < corr_threshold * 2:
        return "Medium", (
            f"Moderate model importance ({shap_imp:.3f}) with weak target "
            f"correlation ({target_corr:.3f}) — worth investigating."
        )
    return "Low", "Feature importance aligns with its correlation to the target."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_spurious_correlations(
    report: ModelReport,
    data: AuditData,
    run_drop_test: bool = True,
) -> SpuriousReport:
    """
    Run the full spurious-correlation detection pipeline.

    Returns a ``SpuriousReport`` with per-feature analyses and an overall risk.
    """
    shap_importances = _compute_shap_importance(report, data)
    target_correlations = _compute_target_correlations(data)
    mutual_infos = _compute_mutual_information(data)

    if run_drop_test:
        drop_changes = _feature_drop_test(report, data)
    else:
        drop_changes = np.zeros(len(data.feature_names))

    # Normalise SHAP importances to 0-1
    max_shap = shap_importances.max() if shap_importances.max() > 0 else 1.0
    shap_norm = shap_importances / max_shap

    analyses: list[FeatureAnalysis] = []
    for i, fname in enumerate(data.feature_names):
        s_imp = float(shap_norm[i])
        t_corr = float(target_correlations[i])
        m_info = float(mutual_infos[i])
        d_change = float(drop_changes[i])
        risk, reason = _classify_risk(s_imp, t_corr, m_info, d_change)
        analyses.append(
            FeatureAnalysis(
                name=fname,
                shap_importance=s_imp,
                target_correlation=t_corr,
                mutual_info=m_info,
                drop_accuracy_change=d_change,
                risk_level=risk,
                reason=reason,
            )
        )

    # Overall risk
    high_count = sum(1 for a in analyses if a.risk_level == "High")
    med_count = sum(1 for a in analyses if a.risk_level == "Medium")
    flagged = high_count + med_count

    if high_count >= 2:
        overall = "High"
    elif high_count >= 1 or med_count >= 2:
        overall = "Medium"
    else:
        overall = "Low"

    # SHAP values for plots
    shap_vals = _compute_shap_values_for_plot(report, data)

    return SpuriousReport(
        feature_analyses=analyses,
        shap_values=shap_vals,
        overall_risk=overall,
        flagged_count=flagged,
    )


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_importance_vs_correlation(sr: SpuriousReport) -> plt.Figure:
    """Scatter: SHAP importance vs target correlation per feature."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for fa in sr.feature_analyses:
        color = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#2ECC71"}[fa.risk_level]
        ax.scatter(
            fa.target_correlation, fa.shap_importance,
            c=color, s=100, edgecolors="white", zorder=3,
        )
        ax.annotate(
            fa.name, (fa.target_correlation, fa.shap_importance),
            fontsize=8, ha="left", va="bottom", alpha=0.85,
        )

    ax.set_xlabel("Target Correlation (|Pearson r|)", fontsize=11)
    ax.set_ylabel("Normalised SHAP Importance", fontsize=11)
    ax.set_title("Feature Importance vs Target Correlation", fontweight="bold", fontsize=13, pad=10)
    ax.axvline(0.15, ls="--", color="grey", alpha=0.5, label="Low-corr threshold")
    ax.axhline(0.5, ls="--", color="grey", alpha=0.3, label="High-importance threshold")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_shap_summary(sr: SpuriousReport, data: AuditData) -> plt.Figure | None:
    """SHAP beeswarm / bar summary plot."""
    if sr.shap_values is None:
        return None
    try:
        import shap
        fig, ax = plt.subplots(figsize=(10, 7))
        vals = sr.shap_values
        if isinstance(vals, list):
            vals = vals[1] if len(vals) == 2 else vals[0]
        # Use subsampled X_test to match SHAP value dimensions
        X_plot = _subsample(data.X_test, MAX_SHAP_SAMPLES)
        shap.summary_plot(
            vals, features=X_plot,
            feature_names=data.feature_names, show=False,
        )
        fig = plt.gcf()
        fig.tight_layout()
        return fig
    except Exception:
        return None
