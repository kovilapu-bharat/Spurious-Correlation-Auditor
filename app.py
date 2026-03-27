"""
ML Model Auditing Dashboard
============================
Streamlit application that orchestrates all auditing modules into an
interactive, tab-based interface.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pathlib
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spurious Correlation Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = pathlib.Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from modules.data_input import (
    BUILTIN_DATASETS,
    AuditData,
    load_builtin_dataset,
    load_csv_dataset,
    prepare_data,
)
from modules.model_evaluation import (
    MODEL_REGISTRY,
    ModelReport,
    train_model,
    plot_confusion_matrix,
    plot_roc_curve,
)
from modules.spurious_detection import (
    SpuriousReport,
    detect_spurious_correlations,
    plot_importance_vs_correlation,
    plot_shap_summary,
)
from modules.dataset_shift import (
    ShiftReport,
    analyse_dataset_shift,
    plot_shift_comparison,
    plot_shift_summary,
)
from modules.nlp_explanation import (
    ExplanationReport,
    generate_explanation,
)
from modules.report_generation import (
    generate_text_report,
    generate_pdf_report,
)
from utils.helpers import styled_metric_fig, correlation_heatmap, risk_badge

# ---------------------------------------------------------------------------
# Cached wrappers — avoid recomputation on re-runs / tab switches
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_load_builtin(name: str, **kwargs) -> pd.DataFrame:
    return load_builtin_dataset(name, **kwargs)


@st.cache_data(show_spinner=False)
def _cached_prepare(df_pickle, target_column: str, test_size: float):
    """Cache-friendly wrapper — accepts serialisable inputs."""
    df = pd.DataFrame(df_pickle)
    return prepare_data(df, target_column=target_column, test_size=test_size)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🔍 ML Model Auditor")
    st.caption("Detect spurious correlations & audit model reliability")
    st.divider()

    # --- Dataset ---
    st.markdown("### 📂 Dataset")
    data_source = st.radio(
        "Choose data source",
        ["Built-in dataset", "Upload CSV"],
        index=0,
        label_visibility="collapsed",
    )

    df: pd.DataFrame | None = None
    dataset_name = "Unknown"
    target_col = "target"

    run_nlp_pipeline = False
    run_image_pipeline = False
    
    if data_source == "Built-in dataset":
        dataset_name = st.selectbox(
            "Select dataset",
            list(BUILTIN_DATASETS.keys()),
        )
        st.info(BUILTIN_DATASETS[dataset_name]["description"], icon="ℹ️")
        
        if dataset_name == "IMDB Sentiment":
            run_nlp_pipeline = st.checkbox("Run 10-Step NLP Pipeline (Baseline + Main)", value=True, help="Trains a Logistic Regression baseline and Random Forest main model side-by-side.")
        elif dataset_name == "Waterbirds (Spurious)":
            run_image_pipeline = st.checkbox("Run Image Pipeline (CNN Feature Extraction)", value=True, help="Uses ResNet50 to extract 2048-d image features and trains both Logistic Regression and Random Forest models.")
    else:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        target_col = st.text_input("Target column name", value="target")
        dataset_name = "Uploaded CSV"

    st.divider()

    # --- Model ---
    st.markdown("### 🤖 Model")
    if run_nlp_pipeline or run_image_pipeline:
        st.info("Using Logistic Regression (Baseline) and Random Forest (Main) for Pipeline.", icon="🔒")
        model_name = "Random Forest"
    else:
        model_name = st.selectbox("Select classifier", list(MODEL_REGISTRY.keys()))

    st.divider()

    # --- Options ---
    st.markdown("### ⚙️ Options")
    # Disable drop test by default for NLP/Image pipelines due to massive feature counts (e.g. 2048 resnet features)
    default_drop = not (run_nlp_pipeline or run_image_pipeline)
    run_drop_test = st.checkbox("Run feature drop test", value=default_drop, help="Retrain without each feature to measure reliance. Slower but more thorough.")
    test_size = st.slider("Test split ratio", 0.1, 0.5, 0.25, 0.05)

    st.divider()

    # --- Run button ---
    run_audit = st.button("🚀 Run Audit", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Main area — Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <h1 style='background: linear-gradient(90deg, #6C63FF, #FF6584);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-size: 2.6rem; font-weight: 800;'>
            Spurious Correlation Auditor
        </h1>
        <p style='color: #aaa; font-size: 1.1rem; margin-top: -8px;'>
            Detect spurious correlations · Evaluate reliability · Generate audit reports
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Run the audit pipeline
# ---------------------------------------------------------------------------
if run_audit:
    # 1. Load data
    with st.status("🚀 Running audit pipeline…", expanded=True) as status:
        status.update(label="📂 Loading dataset…")
        try:
            if data_source == "Built-in dataset":
                if dataset_name == "Waterbirds (Spurious)" and run_image_pipeline:
                    df = _cached_load_builtin(dataset_name, extract_images=True)
                else:
                    df = _cached_load_builtin(dataset_name)
            else:
                if uploaded is None:
                    st.error("Please upload a CSV file first.")
                    st.stop()
                df = load_csv_dataset(uploaded, target_col)
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

        # Dataset size warning
        n_rows = len(df)
        if n_rows > 50_000:
            st.warning(
                f"⚠️ Large dataset detected ({n_rows:,} rows). "
                f"SHAP and shift analyses will use subsampling for speed. "
                f"Consider disabling *Feature drop test* for faster results.",
                icon="⏱️",
            )

        # 2. Prepare data
        status.update(label=f"🔧 Preparing data ({n_rows:,} rows)…")
        try:
            audit_data: AuditData = _cached_prepare(
                df.to_dict(), target_column=target_col, test_size=test_size,
            )
        except Exception as e:
            st.error(f"Failed to prepare data: {e}")
            st.stop()

        # 3. Train model(s)
        baseline_report = None
        if run_nlp_pipeline or run_image_pipeline:
            status.update(label="🤖 Training Baseline Model (Logistic Regression)…")
            baseline_report = train_model("Logistic Regression", audit_data)
            status.update(label="🤖 Training Main Model (Random Forest)…")
            model_report: ModelReport = train_model("Random Forest", audit_data)
        else:
            status.update(label=f"🤖 Training {model_name}…")
            model_report: ModelReport = train_model(model_name, audit_data)

        # 4. Spurious detection
        status.update(label="🔍 Analysing spurious correlations…")
        spurious_report: SpuriousReport = detect_spurious_correlations(
            model_report, audit_data, run_drop_test=run_drop_test,
        )

        # 5. Dataset shift
        status.update(label="📈 Analysing dataset shift…")
        shift_report: ShiftReport = analyse_dataset_shift(model_report, audit_data)

        # 6. NLP explanation
        status.update(label="📝 Generating explanation report…")
        explanation: ExplanationReport = generate_explanation(
            model_report, spurious_report, shift_report,
        )

        status.update(label="✅ Audit complete!", state="complete", expanded=False)

    # Save results to session
    st.session_state.update(
        audit_done=True,
        audit_data=audit_data,
        baseline_report=baseline_report,
        model_report=model_report,
        spurious_report=spurious_report,
        shift_report=shift_report,
        explanation=explanation,
        dataset_name=dataset_name,
        df=df,
    )

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if st.session_state.get("audit_done"):
    audit_data: AuditData = st.session_state.audit_data
    model_report: ModelReport = st.session_state.model_report
    spurious_report: SpuriousReport = st.session_state.spurious_report
    shift_report: ShiftReport = st.session_state.shift_report
    explanation: ExplanationReport = st.session_state.explanation
    dataset_name: str = st.session_state.dataset_name
    df: pd.DataFrame = st.session_state.df

    # --- Top-level summary cards ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Reliability Score", f"{explanation.reliability_score:.0f} / 100")
    with c2:
        st.metric("F1 Score", f"{model_report.metrics.f1:.3f}")
    with c3:
        st.metric("Spurious Risk", risk_badge(spurious_report.overall_risk))
    with c4:
        st.metric("Shift Risk", risk_badge(shift_report.overall_risk))

    st.divider()

    # ===================================================================
    #  TABS
    # ===================================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview",
        "🎯 Model Performance",
        "🔍 Spurious Correlations",
        "📈 Dataset Shift",
        "📝 Audit Report",
    ])

    # --- TAB 1: Data Overview ---
    with tab1:
        st.subheader("Dataset Summary")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Samples", len(df))
        col_b.metric("Features", len(audit_data.feature_names))
        col_c.metric("Classes", len(audit_data.class_names))

        st.markdown("#### First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.describe().round(3), use_container_width=True)

        st.markdown("#### Feature Correlation Heatmap")
        if len(audit_data.feature_names) > 50:
            st.info(f"Correlation heatmap omitted for large feature sets ({len(audit_data.feature_names)} features).", icon="ℹ️")
        else:
            feature_df = df[audit_data.feature_names + [audit_data.target_name]]
            fig = correlation_heatmap(feature_df.select_dtypes(include='number').corr())
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("#### Target Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        df[audit_data.target_name].value_counts().plot.bar(ax=ax2, color=["#6C63FF", "#FF6584", "#2ECC71", "#F39C12"])
        ax2.set_title("Class Distribution", fontweight="bold")
        ax2.spines[["top", "right"]].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # --- TAB 2: Model Performance ---
    with tab2:
        baseline_report: ModelReport | None = st.session_state.get("baseline_report", None)
        
        if baseline_report is not None:
            st.subheader(f"Main Model: {model_report.model_name} (vs Baseline: {baseline_report.model_name})")
            
            st.markdown("#### Performance Comparison")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mb = baseline_report.metrics
            mm = model_report.metrics
            mc1.metric("Accuracy", f"{mm.accuracy:.3f}", delta=f"{mm.accuracy - mb.accuracy:.3f}")
            mc2.metric("Precision", f"{mm.precision:.3f}", delta=f"{mm.precision - mb.precision:.3f}")
            mc3.metric("Recall", f"{mm.recall:.3f}", delta=f"{mm.recall - mb.recall:.3f}")
            mc4.metric("F1 Score", f"{mm.f1:.3f}", delta=f"{mm.f1 - mb.f1:.3f}")
        else:
            st.subheader(f"Model: {model_report.model_name}")

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            m = model_report.metrics
            mc1.metric("Accuracy", f"{m.accuracy:.3f}")
            mc2.metric("Precision", f"{m.precision:.3f}")
            mc3.metric("Recall", f"{m.recall:.3f}")
            mc4.metric("F1 Score", f"{m.f1:.3f}")
            mc5.metric("ROC-AUC", f"{m.roc_auc:.3f}" if m.roc_auc else "N/A")

        col_cm, col_roc = st.columns(2)
        with col_cm:
            st.markdown("#### Confusion Matrix (Main)")
            fig_cm = plot_confusion_matrix(model_report, audit_data)
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with col_roc:
            st.markdown("#### ROC Curve (Main)")
            fig_roc = plot_roc_curve(model_report, audit_data)
            if fig_roc:
                st.pyplot(fig_roc)
                plt.close(fig_roc)
            else:
                st.info("ROC curve is only available for binary classification.")

        with st.expander("📄 Classification Report (Main)"):
            st.code(model_report.metrics.report_text)
            
        if baseline_report is not None:
            with st.expander("📄 Classification Report (Baseline)"):
                st.code(baseline_report.metrics.report_text)

    # --- TAB 3: Spurious Correlations ---
    with tab3:
        st.subheader("Spurious Correlation Analysis")

        st.markdown(f"**Overall Risk**: {risk_badge(spurious_report.overall_risk)}")
        st.markdown(f"**Flagged Features**: {spurious_report.flagged_count} / {len(spurious_report.feature_analyses)}")

        # Importance vs Correlation scatter
        st.markdown("#### Feature Importance vs Target Correlation")
        st.caption("Features in the **top-left** (high importance, low correlation) may be spurious.")
        fig_scatter = plot_importance_vs_correlation(spurious_report)
        st.pyplot(fig_scatter)
        plt.close(fig_scatter)

        # SHAP summary
        fig_shap = plot_shap_summary(spurious_report, audit_data)
        if fig_shap:
            st.markdown("#### SHAP Feature Importance")
            st.pyplot(fig_shap)
            plt.close(fig_shap)

        # Detailed table
        st.markdown("#### Feature Analysis Table")
        fa_data = []
        for fa in spurious_report.feature_analyses:
            fa_data.append({
                "Feature": fa.name,
                "SHAP Importance": round(fa.shap_importance, 4),
                "Target Correlation": round(fa.target_correlation, 4),
                "Mutual Information": round(fa.mutual_info, 4),
                "Drop Acc Change": round(fa.drop_accuracy_change, 4),
                "Risk": fa.risk_level,
            })
        fa_df = pd.DataFrame(fa_data)
        st.dataframe(
            fa_df.style.applymap(
                lambda v: "background-color: rgba(231,76,60,0.2)" if v == "High"
                else ("background-color: rgba(243,156,18,0.2)" if v == "Medium" else ""),
                subset=["Risk"],
            ),
            use_container_width=True,
        )

        # Detailed findings per flagged feature
        flagged = [fa for fa in spurious_report.feature_analyses if fa.risk_level in ("High", "Medium")]
        if flagged:
            st.markdown("#### Detailed Findings")
            for fa in flagged:
                icon = "🔴" if fa.risk_level == "High" else "🟡"
                with st.expander(f"{icon} {fa.name} — {fa.risk_level} Risk"):
                    st.write(fa.reason)
                    st.write(f"- **SHAP Importance**: {fa.shap_importance:.4f}")
                    st.write(f"- **Target Correlation**: {fa.target_correlation:.4f}")
                    st.write(f"- **Mutual Information**: {fa.mutual_info:.4f}")
                    st.write(f"- **Accuracy change on removal**: {fa.drop_accuracy_change:+.4f}")

        # --- Image Pipeline Specific Visualization: Background Bias ---
        if "_true_place" in df.columns and flagged:
            st.markdown("### 🖼️ Image Background Bias Visualization")
            st.info("The model was trained on ResNet50 features. Let's check if the top spurious features correlate with the actual background (Water vs. Land).")
            
            # Find the most spurious feature
            top_spurious = flagged[0].name
            
            fig_bias, ax_bias = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                x=df["_true_place"].map({0: "Land", 1: "Water"}), 
                y=df[top_spurious], 
                ax=ax_bias,
                palette=["#F39C12", "#3498DB"]
            )
            ax_bias.set_title(f"Correlation of Spurious '{top_spurious}' with Background", fontweight="bold")
            ax_bias.set_xlabel("True Background (Spurious Factor)")
            ax_bias.set_ylabel(f"Activation of {top_spurious}")
            st.pyplot(fig_bias)
            plt.close(fig_bias)
            
            st.markdown(f"**Insight**: If there is a large difference in the boxed activations above, it means the feature **`{top_spurious}`** (which the model heavily relies on) is actually just acting as a detector for the background, revealing the **shortcut learning** behavior!")

    # --- TAB 4: Dataset Shift ---
    with tab4:
        st.subheader("Dataset Shift Analysis")

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Shifted Features", f"{shift_report.shifted_feature_count} / {len(shift_report.feature_shifts)}")
        sc2.metric("Label Shift", "Detected" if shift_report.label_shifted else "None")
        sc3.metric("Concept Drift Drop", f"{shift_report.concept_drift_score:.1%}")

        st.markdown(f"**Overall Shift Risk**: {risk_badge(shift_report.overall_risk)}")

        # KS summary bar chart
        st.markdown("#### Covariate Shift (KS Test)")
        fig_ks = plot_shift_summary(shift_report)
        st.pyplot(fig_ks)
        plt.close(fig_ks)

        # Distribution comparison for top shifted features
        shifted_feats = [fs for fs in shift_report.feature_shifts if fs.shifted]
        if shifted_feats:
            st.markdown("#### Distribution Comparison (Shifted Features)")
            for fs in shifted_feats[:5]:  # show top 5
                idx = audit_data.feature_names.index(fs.name)
                fig_dist = plot_shift_comparison(audit_data, idx)
                st.pyplot(fig_dist)
                plt.close(fig_dist)

        # Shift details table
        st.markdown("#### Shift Details")
        shift_data = []
        for fs in shift_report.feature_shifts:
            shift_data.append({
                "Feature": fs.name,
                "KS Statistic": round(fs.ks_statistic, 4),
                "p-value": round(fs.ks_pvalue, 4),
                "KL Divergence": round(fs.kl_divergence, 4),
                "Shifted": "⚠️ Yes" if fs.shifted else "✅ No",
            })
        st.dataframe(pd.DataFrame(shift_data), use_container_width=True)

    # --- TAB 5: Audit Report ---
    with tab5:
        st.subheader("Complete Audit Report")

        # Reliability score gauge
        score = explanation.reliability_score
        if score >= 80:
            score_color = "#2ECC71"
            score_label = "✅ Reliable"
        elif score >= 60:
            score_color = "#F39C12"
            score_label = "⚠️ Moderate Concerns"
        else:
            score_color = "#E74C3C"
            score_label = "🚨 Significant Concerns"

        st.markdown(
            f"""
            <div style='text-align:center; padding: 20px; background: rgba(255,255,255,0.05);
                        border-radius: 16px; border: 1px solid {score_color}30; margin-bottom: 20px;'>
                <h1 style='color:{score_color}; font-size: 3.5rem; margin: 0;'>{score:.0f}</h1>
                <p style='color:{score_color}; font-size: 1.2rem; margin: 4px 0;'>{score_label}</p>
                <p style='color: #888;'>Overall Reliability Score (out of 100)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # NLP explanation sections
        for section in explanation.sections:
            severity_icons = {
                "success": "✅",
                "info": "ℹ️",
                "warning": "⚠️",
                "danger": "🚨",
            }
            icon = severity_icons.get(section.severity, "📌")

            severity_colors = {
                "success": "#2ECC71",
                "info": "#3498DB",
                "warning": "#F39C12",
                "danger": "#E74C3C",
            }
            color = severity_colors.get(section.severity, "#888")

            st.markdown(
                f"""
                <div style='padding: 16px 20px; margin: 12px 0; border-radius: 12px;
                            background: rgba(255,255,255,0.04);
                            border-left: 4px solid {color};'>
                    <h4 style='margin: 0 0 8px 0; color: {color};'>{icon} {section.title}</h4>
                    <p style='color: #ccc; line-height: 1.7; margin: 0;'>{section.text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # Download buttons
        st.markdown("#### 📥 Download Reports")
        col_dl1, col_dl2 = st.columns(2)

        text_report = generate_text_report(
            model_report, spurious_report, shift_report, explanation, dataset_name,
        )
        with col_dl1:
            st.download_button(
                "📄 Download Text Report",
                data=text_report,
                file_name="audit_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        pdf_bytes = generate_pdf_report(
            model_report, spurious_report, shift_report, explanation, dataset_name,
        )
        with col_dl2:
            st.download_button(
                "📕 Download PDF Report",
                data=pdf_bytes,
                file_name="audit_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

else:
    # Landing state
    st.markdown(
        """
        <div style='text-align: center; padding: 80px 20px; color: #888;'>
            <p style='font-size: 3rem;'>🔍</p>
            <h3>Welcome to ML Model Auditor</h3>
            <p>Select a dataset and model from the sidebar, then click <b>Run Audit</b> to begin.</p>
            <br>
            <p style='font-size: 0.9rem; color: #666;'>
                This tool analyses machine learning models for spurious correlations,<br>
                evaluates dataset shifts, and generates explainable reliability reports.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
