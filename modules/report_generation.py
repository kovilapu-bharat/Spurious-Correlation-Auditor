"""
Report Generation Module
-------------------------
Compiles all analysis outputs into downloadable PDF and plain-text reports.
"""

from __future__ import annotations

import io
import re
import textwrap
from datetime import datetime

from fpdf import FPDF

from modules.model_evaluation import ModelReport
from modules.spurious_detection import SpuriousReport


def _sanitize_for_pdf(text: str) -> str:
    """Replace Unicode characters unsupported by Helvetica with ASCII equivalents."""
    replacements = {
        "\u2014": "-",   # em dash
        "\u2013": "-",   # en dash
        "\u2192": "->",  # right arrow
        "\u2190": "<-",  # left arrow
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\U0001f534": "[HIGH]",   # red circle emoji
        "\U0001f7e1": "[MEDIUM]", # yellow circle emoji
        "\U0001f7e2": "[LOW]",    # green circle emoji
        "\u2022": "-",   # bullet
        "\u00d7": "x",   # multiplication sign
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Strip any remaining non-latin1 characters
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text
from modules.dataset_shift import ShiftReport
from modules.nlp_explanation import ExplanationReport


# ---------------------------------------------------------------------------
# Plain text report
# ---------------------------------------------------------------------------

def generate_text_report(
    model_report: ModelReport,
    spurious_report: SpuriousReport,
    shift_report: ShiftReport,
    explanation: ExplanationReport,
    dataset_name: str = "Unknown",
) -> str:
    """Build a plain-text audit report string."""
    lines: list[str] = []
    sep = "=" * 70

    lines.append(sep)
    lines.append("       ML MODEL AUDIT REPORT")
    lines.append(f"       Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append("")

    lines.append(f"Dataset        : {dataset_name}")
    lines.append(f"Model          : {model_report.model_name}")
    lines.append(f"Reliability    : {explanation.reliability_score:.0f} / 100")
    lines.append(f"Spurious Risk  : {spurious_report.overall_risk}")
    lines.append(f"Shift Risk     : {shift_report.overall_risk}")
    lines.append("")
    lines.append(sep)
    lines.append("  MODEL PERFORMANCE")
    lines.append(sep)
    m = model_report.metrics
    lines.append(f"  Accuracy   : {m.accuracy:.4f}")
    lines.append(f"  Precision  : {m.precision:.4f}")
    lines.append(f"  Recall     : {m.recall:.4f}")
    lines.append(f"  F1 Score   : {m.f1:.4f}")
    lines.append(f"  ROC-AUC    : {m.roc_auc:.4f}" if m.roc_auc else "  ROC-AUC    : N/A")
    lines.append("")

    lines.append(sep)
    lines.append("  SPURIOUS CORRELATION ANALYSIS")
    lines.append(sep)
    lines.append(f"  Flagged features: {spurious_report.flagged_count}")
    lines.append(f"  Overall risk    : {spurious_report.overall_risk}")
    lines.append("")

    for fa in spurious_report.feature_analyses:
        flag = " ⚠" if fa.risk_level in ("High", "Medium") else ""
        lines.append(f"  [{fa.risk_level:6s}] {fa.name}{flag}")
        lines.append(f"          SHAP importance   : {fa.shap_importance:.4f}")
        lines.append(f"          Target correlation: {fa.target_correlation:.4f}")
        lines.append(f"          Mutual information: {fa.mutual_info:.4f}")
        lines.append(f"          Drop acc change   : {fa.drop_accuracy_change:+.4f}")
        lines.append(f"          {fa.reason}")
        lines.append("")

    lines.append(sep)
    lines.append("  DATASET SHIFT ANALYSIS")
    lines.append(sep)
    lines.append(f"  Shifted features  : {shift_report.shifted_feature_count} / {len(shift_report.feature_shifts)}")
    lines.append(f"  Label shift       : {'Yes' if shift_report.label_shifted else 'No'}")
    lines.append(f"  Concept drift drop: {shift_report.concept_drift_score:.4f}")
    lines.append(f"  Overall risk      : {shift_report.overall_risk}")
    lines.append("")

    for fs in shift_report.feature_shifts:
        flag = " ⚠" if fs.shifted else ""
        lines.append(f"  {fs.name}{flag}")
        lines.append(f"      KS stat: {fs.ks_statistic:.4f}  p-value: {fs.ks_pvalue:.4f}  KL div: {fs.kl_divergence:.4f}")
    lines.append("")

    lines.append(sep)
    lines.append("  NLP EXPLANATION")
    lines.append(sep)
    for section in explanation.sections:
        lines.append(f"\n  >> {section.title}")
        # Wrap long text
        wrapped = textwrap.fill(section.text, width=68, initial_indent="     ", subsequent_indent="     ")
        lines.append(wrapped)
    lines.append("")

    lines.append(sep)
    lines.append(f"  OVERALL RELIABILITY SCORE: {explanation.reliability_score:.0f} / 100")
    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

class _AuditPDF(FPDF):
    """Custom PDF with header/footer for the audit report."""

    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(108, 99, 255)  # primary purple
        self.cell(0, 8, "ML Model Audit Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(108, 99, 255)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 30, 47)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def metric_row(self, label: str, value: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(50, 50, 50)
        self.cell(50, 6, label, new_x="RIGHT")
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")


def generate_pdf_report(
    model_report: ModelReport,
    spurious_report: SpuriousReport,
    shift_report: ShiftReport,
    explanation: ExplanationReport,
    dataset_name: str = "Unknown",
) -> bytes:
    """Build a PDF audit report and return it as bytes."""
    pdf = _AuditPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # --- Title ---
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(108, 99, 255)
    pdf.cell(0, 15, "Model Audit Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Dataset: {dataset_name}  |  Model: {model_report.model_name}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Reliability badge
    score = explanation.reliability_score
    pdf.set_font("Helvetica", "B", 14)
    if score >= 80:
        pdf.set_text_color(46, 204, 113)
    elif score >= 60:
        pdf.set_text_color(243, 156, 18)
    else:
        pdf.set_text_color(231, 76, 60)
    pdf.cell(0, 10, f"Reliability Score: {score:.0f} / 100", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # --- Model Performance ---
    pdf.section_title("1. Model Performance")
    m = model_report.metrics
    pdf.metric_row("Accuracy:", f"{m.accuracy:.4f}")
    pdf.metric_row("Precision:", f"{m.precision:.4f}")
    pdf.metric_row("Recall:", f"{m.recall:.4f}")
    pdf.metric_row("F1 Score:", f"{m.f1:.4f}")
    pdf.metric_row("ROC-AUC:", f"{m.roc_auc:.4f}" if m.roc_auc else "N/A")
    pdf.ln(3)

    # --- Spurious Correlations ---
    pdf.section_title("2. Spurious Correlation Analysis")
    pdf.metric_row("Flagged features:", str(spurious_report.flagged_count))
    pdf.metric_row("Overall risk:", spurious_report.overall_risk)
    pdf.ln(2)

    for fa in spurious_report.feature_analyses:
        if fa.risk_level in ("High", "Medium"):
            pdf.set_font("Helvetica", "B", 10)
            color = (231, 76, 60) if fa.risk_level == "High" else (243, 156, 18)
            pdf.set_text_color(*color)
            pdf.cell(0, 6, f"[{fa.risk_level}] {fa.name}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(50, 50, 50)
            pdf.set_font("Helvetica", "", 9)
            detail = (
                f"SHAP: {fa.shap_importance:.3f} | Correlation: {fa.target_correlation:.3f} | "
                f"MI: {fa.mutual_info:.4f} | Drop change: {fa.drop_accuracy_change:+.3f}"
            )
            pdf.cell(0, 5, detail, new_x="LMARGIN", new_y="NEXT")
            # Clean text for PDF (remove markdown bold markers and Unicode)
            clean_reason = _sanitize_for_pdf(fa.reason.replace("**", ""))
            pdf.multi_cell(0, 5, clean_reason)
            pdf.ln(2)

    # --- Dataset Shift ---
    pdf.section_title("3. Dataset Shift Analysis")
    pdf.metric_row("Shifted features:", f"{shift_report.shifted_feature_count} / {len(shift_report.feature_shifts)}")
    pdf.metric_row("Label shift:", "Yes" if shift_report.label_shifted else "No")
    pdf.metric_row("Concept drift drop:", f"{shift_report.concept_drift_score:.4f}")
    pdf.metric_row("Overall risk:", shift_report.overall_risk)
    pdf.ln(3)

    # --- NLP Explanation ---
    pdf.section_title("4. Detailed Explanation")
    for section in explanation.sections:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 30, 47)
        pdf.cell(0, 7, section.title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        # Clean text for PDF
        clean_text = _sanitize_for_pdf(
            section.text.replace("**", "").replace("🔴", "[HIGH]").replace("🟡", "[MEDIUM]").replace("🟢", "[LOW]")
        )
        pdf.multi_cell(0, 5, clean_text)
        pdf.ln(3)

    return bytes(pdf.output())
