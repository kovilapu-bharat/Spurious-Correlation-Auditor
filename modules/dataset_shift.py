"""
Dataset Shift Analysis Module
------------------------------
Detects covariate shift, label shift, and simulated concept drift between
training and test distributions using statistical tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency, entropy
from sklearn.base import clone
from sklearn.metrics import accuracy_score

from modules.data_input import AuditData
from modules.model_evaluation import ModelReport

# ---------------------------------------------------------------------------
# Subsampling constants
# ---------------------------------------------------------------------------
MAX_KS_SAMPLES: int = 10_000    # Max rows per split for KS tests
MAX_DRIFT_SAMPLES: int = 5_000  # Max test samples for concept drift simulation

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FeatureShift:
    """Shift analysis for one feature."""
    name: str
    ks_statistic: float
    ks_pvalue: float
    kl_divergence: float
    shifted: bool  # True if statistically significant shift detected


@dataclass
class ShiftReport:
    """Full dataset shift analysis report."""
    feature_shifts: list[FeatureShift]
    label_shift_pvalue: float
    label_shifted: bool
    concept_drift_score: float  # accuracy drop under perturbation
    overall_risk: str  # "High", "Medium", "Low"
    shifted_feature_count: int


# ---------------------------------------------------------------------------
# Per-feature covariate shift
# ---------------------------------------------------------------------------

def _subsample(X: np.ndarray, max_n: int, rng_seed: int = 42) -> np.ndarray:
    """Return a random subsample of rows if X exceeds *max_n* rows."""
    if len(X) <= max_n:
        return X
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(X), size=max_n, replace=False)
    return X[idx]


def _ks_test_features(data: AuditData, alpha: float = 0.05) -> list[FeatureShift]:
    """Run the two-sample KS test on each feature between train & test."""
    # Subsample for very large datasets — KS test is valid on subsets
    X_train_sub = _subsample(data.X_train, MAX_KS_SAMPLES)
    X_test_sub = _subsample(data.X_test, MAX_KS_SAMPLES)

    results: list[FeatureShift] = []
    for i, fname in enumerate(data.feature_names):
        stat, pval = ks_2samp(X_train_sub[:, i], X_test_sub[:, i])
        kl = _kl_divergence_1d(X_train_sub[:, i], X_test_sub[:, i])
        results.append(
            FeatureShift(
                name=fname,
                ks_statistic=float(stat),
                ks_pvalue=float(pval),
                kl_divergence=float(kl),
                shifted=pval < alpha,
            )
        )
    return results


def _kl_divergence_1d(p_samples: np.ndarray, q_samples: np.ndarray, bins: int = 30) -> float:
    """Compute KL divergence between two 1D sample arrays."""
    lo = min(p_samples.min(), q_samples.min())
    hi = max(p_samples.max(), q_samples.max())
    p_hist, _ = np.histogram(p_samples, bins=bins, range=(lo, hi), density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, range=(lo, hi), density=True)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    # Normalise to probability distributions
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    return float(entropy(p_hist, q_hist))


# ---------------------------------------------------------------------------
# Label shift
# ---------------------------------------------------------------------------

def _label_shift_test(data: AuditData) -> tuple[float, bool]:
    """Chi-squared test for label distribution differences."""
    from collections import Counter
    train_counts = Counter(data.y_train)
    test_counts = Counter(data.y_test)
    all_labels = sorted(set(train_counts.keys()) | set(test_counts.keys()))

    observed = np.array([[train_counts.get(l, 0) for l in all_labels],
                         [test_counts.get(l, 0) for l in all_labels]])

    if observed.shape[1] < 2 or observed.min() == 0:
        return 1.0, False

    _, pval, _, _ = chi2_contingency(observed)
    return float(pval), pval < 0.05


# ---------------------------------------------------------------------------
# Simulated concept drift
# ---------------------------------------------------------------------------

def _simulate_concept_drift(
    report: ModelReport,
    data: AuditData,
    noise_scale: float = 0.3,
    n_features_to_perturb: int = 3,
) -> float:
    """
    Add Gaussian noise to the top-N important features in the test set and
    measure accuracy degradation.  Returns the accuracy drop (positive = worse).
    """
    base_acc = accuracy_score(data.y_test, report.y_pred)

    # Subsample for permutation importance speed
    X_test_sub = _subsample(data.X_test, MAX_DRIFT_SAMPLES)
    y_test_sub = data.y_test[:len(X_test_sub)] if len(data.y_test) > MAX_DRIFT_SAMPLES else data.y_test
    # Re-align y_test_sub if we subsampled
    if len(data.X_test) > MAX_DRIFT_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(data.X_test), size=MAX_DRIFT_SAMPLES, replace=False)
        y_test_sub = data.y_test[idx]

    # Identify top features by permutation importance
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(
        report.model, X_test_sub, y_test_sub,
        n_repeats=5, random_state=42, scoring="accuracy",
    )
    top_indices = np.argsort(perm.importances_mean)[-n_features_to_perturb:]

    X_noisy = data.X_test.copy()
    rng = np.random.default_rng(42)
    for idx in top_indices:
        std = X_noisy[:, idx].std()
        X_noisy[:, idx] += rng.normal(0, noise_scale * std, size=len(X_noisy))

    noisy_acc = accuracy_score(data.y_test, report.model.predict(X_noisy))
    return float(base_acc - noisy_acc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse_dataset_shift(
    report: ModelReport,
    data: AuditData,
    alpha: float = 0.05,
) -> ShiftReport:
    """Run the full dataset-shift analysis pipeline."""
    feature_shifts = _ks_test_features(data, alpha=alpha)
    label_pval, label_shifted = _label_shift_test(data)
    drift_score = _simulate_concept_drift(report, data)

    shifted_count = sum(1 for fs in feature_shifts if fs.shifted)

    # Overall risk
    pct_shifted = shifted_count / max(len(feature_shifts), 1)
    if pct_shifted > 0.4 or drift_score > 0.10:
        overall = "High"
    elif pct_shifted > 0.15 or drift_score > 0.05:
        overall = "Medium"
    else:
        overall = "Low"

    return ShiftReport(
        feature_shifts=feature_shifts,
        label_shift_pvalue=label_pval,
        label_shifted=label_shifted,
        concept_drift_score=drift_score,
        overall_risk=overall,
        shifted_feature_count=shifted_count,
    )


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_shift_comparison(
    data: AuditData,
    feature_index: int,
) -> plt.Figure:
    """Side-by-side KDE plot of train vs test for a single feature."""
    import seaborn as sns

    fname = data.feature_names[feature_index]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(data.X_train[:, feature_index], ax=ax, label="Train", fill=True, alpha=0.4, color="#6C63FF")
    sns.kdeplot(data.X_test[:, feature_index], ax=ax, label="Test", fill=True, alpha=0.4, color="#FF6584")
    ax.set_title(f"Distribution: {fname}", fontweight="bold", fontsize=12, pad=8)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_shift_summary(sr: ShiftReport) -> plt.Figure:
    """Horizontal bar chart of KS statistics for all features."""
    names = [fs.name for fs in sr.feature_shifts]
    stats = [fs.ks_statistic for fs in sr.feature_shifts]
    colors = ["#E74C3C" if fs.shifted else "#2ECC71" for fs in sr.feature_shifts]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.35)))
    ax.barh(names, stats, color=colors, edgecolor="white", height=0.6)
    ax.axvline(0.05, ls="--", color="grey", alpha=0.5, label="Significance hint")
    ax.set_xlabel("KS Statistic", fontsize=11)
    ax.set_title("Covariate Shift per Feature (KS Test)", fontweight="bold", fontsize=13, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
