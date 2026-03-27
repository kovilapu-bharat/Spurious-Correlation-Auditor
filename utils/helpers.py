"""
Shared utility functions and constants for the Model Auditing Application.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend for Streamlit compatibility
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#6C63FF",
    "secondary": "#FF6584",
    "success": "#2ECC71",
    "warning": "#F39C12",
    "danger": "#E74C3C",
    "info": "#3498DB",
    "dark": "#1E1E2F",
    "light": "#F5F6FA",
    "gradient_start": "#6C63FF",
    "gradient_end": "#FF6584",
}

RISK_COLORS = {
    "High": "#E74C3C",
    "Medium": "#F39C12",
    "Low": "#2ECC71",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def styled_metric_fig(
    values: dict[str, float],
    title: str = "Model Metrics",
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """Create a styled horizontal bar chart for metric values (0-1 range)."""
    fig, ax = plt.subplots(figsize=figsize)
    names = list(values.keys())
    vals = list(values.values())
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))

    bars = ax.barh(names, vals, color=colors, edgecolor="white", height=0.55)
    ax.set_xlim(0, 1.05)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    return fig


def correlation_heatmap(
    corr_matrix, title: str = "Feature Correlation", figsize=(10, 8)
) -> plt.Figure:
    """Create a styled correlation heatmap."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def risk_badge(level: str) -> str:
    """Return a coloured emoji + label for a risk level string."""
    mapping = {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}
    return mapping.get(level, level)
