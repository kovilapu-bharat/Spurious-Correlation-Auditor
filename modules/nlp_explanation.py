"""
NLP Explanation Module
-----------------------
Converts numeric analysis results into human-readable natural language
paragraphs using spaCy for linguistic processing and NLTK for synonym
variation and sentence quality.

Uses:
  - spaCy: sentence segmentation, noun-phrase extraction, text processing
  - NLTK:  WordNet synonyms for vocabulary variation, sentence tokenization
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# NLP library imports — spaCy & NLTK
# ---------------------------------------------------------------------------
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize

# Ensure required NLTK data is available
for _pkg in ("punkt_tab", "wordnet", "omw-1.4"):
    nltk.download(_pkg, quiet=True)

import spacy

# Load spaCy English model (small); download if missing
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download as _spacy_download
    _spacy_download("en_core_web_sm")
    _nlp = spacy.load("en_core_web_sm")

from modules.model_evaluation import ModelReport
from modules.spurious_detection import SpuriousReport
from modules.dataset_shift import ShiftReport

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ExplanationSection:
    """One titled paragraph in the explanation report."""
    title: str
    text: str
    severity: str = "info"  # "info", "warning", "danger", "success"


@dataclass
class ExplanationReport:
    sections: list[ExplanationSection] = field(default_factory=list)
    reliability_score: float = 0.0  # 0–100


# ---------------------------------------------------------------------------
# NLP utilities — spaCy & NLTK helpers
# ---------------------------------------------------------------------------

def _synonyms(word: str, pos: str | None = None) -> list[str]:
    """
    Return a list of WordNet synonyms for the given word (NLTK).

    Used to add vocabulary variation in generated explanations so the text
    does not feel overly repetitive.
    """
    wn_pos = None
    if pos in ("JJ", "ADJ"):
        wn_pos = wordnet.ADJ
    elif pos in ("RB", "ADV"):
        wn_pos = wordnet.ADV
    elif pos in ("NN", "NOUN"):
        wn_pos = wordnet.NOUN

    synsets = wordnet.synsets(word, pos=wn_pos) if wn_pos else wordnet.synsets(word)
    lemmas: list[str] = []
    for s in synsets[:3]:
        for lemma in s.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower() and name not in lemmas:
                lemmas.append(name)
    return lemmas


def _vary_adjective(word: str, fallback: str | None = None) -> str:
    """
    Pick a synonym for an adjective using NLTK WordNet.
    Falls back to the original word if no synonyms are found.
    """
    syns = _synonyms(word, pos="ADJ")
    if syns:
        return random.choice(syns[:3])
    return fallback or word


def _extract_noun_phrases(text: str) -> list[str]:
    """
    Use spaCy to extract noun phrases from text.
    Useful for identifying key concepts in generated explanations.
    """
    doc = _nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]


def _segment_sentences(text: str) -> list[str]:
    """
    Use spaCy for sentence segmentation with NLTK fallback.
    Ensures generated paragraphs have well-formed sentence boundaries.
    """
    doc = _nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    if not sentences:
        # Fallback to NLTK tokenizer
        sentences = sent_tokenize(text)
    return sentences


def _refine_text(text: str) -> str:
    """
    Post-process generated text using spaCy and NLTK:
    1. Segment into sentences with spaCy
    2. Validate sentence structure
    3. Rejoin with consistent spacing
    """
    sentences = _segment_sentences(text)
    refined: list[str] = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Ensure sentence ends with punctuation
        if sent and sent[-1] not in ".!?:":
            sent += "."
        # Capitalise first letter
        if sent and sent[0].islower():
            sent = sent[0].upper() + sent[1:]
        refined.append(sent)
    return " ".join(refined)


def _quality_description(quality: str) -> str:
    """Return a varied quality descriptor using NLTK synonyms."""
    variations = {
        "excellent": ["outstanding", "exceptional", "superb"],
        "good": ["solid", "satisfactory", "decent"],
        "moderate": ["average", "fair", "acceptable"],
        "poor": ["weak", "inadequate", "subpar"],
    }
    choices = variations.get(quality, [])
    if choices:
        return random.choice(choices)
    return _vary_adjective(quality, fallback=quality)


# ---------------------------------------------------------------------------
# Template generators (enhanced with NLP processing)
# ---------------------------------------------------------------------------

def _model_performance_section(report: ModelReport) -> ExplanationSection:
    m = report.metrics
    roc_text = f"{m.roc_auc:.3f}" if m.roc_auc is not None else "N/A"

    if m.f1 >= 0.85:
        quality = "excellent"
        severity = "success"
    elif m.f1 >= 0.7:
        quality = "good"
        severity = "info"
    elif m.f1 >= 0.5:
        quality = "moderate"
        severity = "warning"
    else:
        quality = "poor"
        severity = "danger"

    # Use NLTK synonym variation for the quality adjective
    varied_quality = _quality_description(quality)

    text = (
        f"The {report.model_name} model shows {varied_quality} performance on the test set. "
        f"It achieves an accuracy of {m.accuracy:.1%}, precision of {m.precision:.1%}, "
        f"recall of {m.recall:.1%}, and an F1-score of {m.f1:.3f}. "
        f"The ROC-AUC score is {roc_text}. "
    )

    if quality in ("excellent", "good"):
        text += (
            "While these numbers are encouraging, high performance alone does not "
            "guarantee the model has learned meaningful patterns — it may still "
            "rely on spurious correlations present in the training data."
        )
    else:
        text += (
            "The relatively low performance suggests the model struggles to "
            "capture the underlying patterns, which may indicate insufficient "
            "features, noisy data, or distribution issues."
        )

    # Refine text with spaCy sentence processing
    text = _refine_text(text)

    return ExplanationSection(title="Model Performance Summary", text=text, severity=severity)


def _spurious_overview_section(sr: SpuriousReport) -> ExplanationSection:
    if sr.flagged_count == 0:
        text = _refine_text(
            "No potentially spurious correlations were detected. All features "
            "with high model importance also demonstrate meaningful correlation "
            "with the target variable. This is a positive indicator of model "
            "reliability."
        )
        return ExplanationSection(
            title="Spurious Correlation Assessment",
            text=text,
            severity="success",
        )

    high = [fa for fa in sr.feature_analyses if fa.risk_level == "High"]
    med = [fa for fa in sr.feature_analyses if fa.risk_level == "Medium"]

    parts = [
        f"The analysis flagged {sr.flagged_count} feature(s) as potentially "
        f"exhibiting spurious relationships with the model's predictions."
    ]

    if high:
        names = ", ".join(f"'{f.name}'" for f in high)
        parts.append(
            f"\n\n🔴 **High Risk**: {names}. These features have high importance "
            f"in the model but show weak statistical correlation with the target "
            f"variable. The model may be relying on patterns in these features "
            f"that do not reflect genuine causal relationships."
        )

    if med:
        names = ", ".join(f"'{f.name}'" for f in med)
        parts.append(
            f"\n\n🟡 **Medium Risk**: {names}. These features show moderate "
            f"importance with relatively low target correlation. Further "
            f"investigation is recommended to determine if these relationships "
            f"are meaningful."
        )

    severity = "danger" if high else "warning"
    return ExplanationSection(
        title="Spurious Correlation Assessment",
        text=" ".join(parts),
        severity=severity,
    )


def _spurious_detail_sections(sr: SpuriousReport) -> list[ExplanationSection]:
    """One section per flagged feature, enriched with spaCy noun-phrase analysis."""
    sections: list[ExplanationSection] = []
    flagged = [fa for fa in sr.feature_analyses if fa.risk_level in ("High", "Medium")]
    for fa in flagged:
        text = (
            f"Feature '{fa.name}' has a normalised SHAP importance of "
            f"{fa.shap_importance:.3f} but its absolute Pearson correlation with "
            f"the target is only {fa.target_correlation:.3f}. "
            f"Mutual information score: {fa.mutual_info:.4f}. "
        )
        if fa.drop_accuracy_change > 0.01:
            text += (
                f"Removing this feature decreases accuracy by "
                f"{fa.drop_accuracy_change:.1%}, confirming the model relies on it. "
            )
        else:
            text += (
                f"Removing this feature has minimal impact on accuracy "
                f"({fa.drop_accuracy_change:+.1%}), suggesting the model can "
                f"compensate without it. "
            )
        text += fa.reason

        # Extract key concepts using spaCy noun phrases
        key_concepts = _extract_noun_phrases(text)
        if key_concepts and len(key_concepts) > 2:
            concept_summary = ", ".join(key_concepts[:4])
            text += f" Key concepts identified: {concept_summary}."

        text = _refine_text(text)

        sev = "danger" if fa.risk_level == "High" else "warning"
        sections.append(ExplanationSection(
            title=f"Feature Detail: {fa.name}",
            text=text,
            severity=sev,
        ))
    return sections


def _dataset_shift_section(shift: ShiftReport) -> ExplanationSection:
    n = shift.shifted_feature_count
    total = len(shift.feature_shifts)

    if n == 0:
        text = (
            "No significant covariate shift was detected between the training "
            "and test sets. Feature distributions appear consistent."
        )
        severity = "success"
    else:
        names = ", ".join(
            f"'{fs.name}'" for fs in shift.feature_shifts if fs.shifted
        )
        text = (
            f"{n} out of {total} features show statistically significant "
            f"distribution shift between training and test data (KS test, "
            f"α=0.05): {names}. "
        )
        if shift.label_shifted:
            text += (
                "Additionally, the label distribution differs significantly "
                "between train and test sets, indicating label shift. "
            )
        severity = "danger" if n > total * 0.3 else "warning"

    # Concept drift
    text += (
        f"\n\nSimulated concept drift analysis shows an accuracy degradation "
        f"of {shift.concept_drift_score:.1%} when noise is added to the most "
        f"important features. "
    )
    if shift.concept_drift_score > 0.10:
        text += "This suggests the model is sensitive to small input changes, indicating potential fragility."
    elif shift.concept_drift_score > 0.05:
        text += "The model shows moderate sensitivity to input perturbation."
    else:
        text += "The model appears reasonably robust to small input perturbations."

    text = _refine_text(text)

    return ExplanationSection(title="Dataset Shift Analysis", text=text, severity=severity)


def _reliability_section(score: float) -> ExplanationSection:
    if score >= 80:
        verdict = _vary_adjective("strong", fallback="strong")
        text = (
            f"**Overall Reliability Score: {score:.0f}/100** — The model demonstrates "
            f"{verdict} performance with minimal spurious correlation risk and stable "
            f"data distributions. It appears suitable for deployment with standard "
            f"monitoring."
        )
        severity = "success"
    elif score >= 60:
        text = (
            f"**Overall Reliability Score: {score:.0f}/100** — The model shows "
            f"acceptable performance but some concerns were identified. Address "
            f"the flagged features and distribution issues before production use."
        )
        severity = "warning"
    else:
        text = (
            f"**Overall Reliability Score: {score:.0f}/100** — Significant "
            f"concerns were identified regarding model reliability. The model "
            f"may rely on spurious patterns or be sensitive to data changes. "
            f"Thorough review and retraining is recommended."
        )
        severity = "danger"

    return ExplanationSection(title="Reliability Assessment", text=text, severity=severity)


# ---------------------------------------------------------------------------
# Reliability scoring
# ---------------------------------------------------------------------------

def _compute_reliability_score(
    report: ModelReport,
    sr: SpuriousReport,
    shift: ShiftReport,
) -> float:
    """
    Compute a 0–100 reliability score based on:
    - Model performance (F1)          → 40 points
    - Spurious correlation risk       → 35 points
    - Dataset shift severity          → 25 points
    """
    # Performance component (0–40)
    f1 = report.metrics.f1
    perf_score = min(f1 / 0.95, 1.0) * 40  # 0.95+ F1 → full 40 points

    # Spurious component (0–35)
    total_features = len(sr.feature_analyses) or 1
    high_pct = sum(1 for fa in sr.feature_analyses if fa.risk_level == "High") / total_features
    med_pct = sum(1 for fa in sr.feature_analyses if fa.risk_level == "Medium") / total_features
    spurious_penalty = high_pct * 1.0 + med_pct * 0.5
    spur_score = max(0, 1.0 - spurious_penalty) * 35

    # Shift component (0–25)
    shifted_pct = shift.shifted_feature_count / max(len(shift.feature_shifts), 1)
    drift_penalty = min(shift.concept_drift_score / 0.15, 1.0)
    shift_penalty = (shifted_pct + drift_penalty) / 2
    shift_score = max(0, 1.0 - shift_penalty) * 25

    return round(perf_score + spur_score + shift_score, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_explanation(
    model_report: ModelReport,
    spurious_report: SpuriousReport,
    shift_report: ShiftReport,
) -> ExplanationReport:
    """
    Generate a complete NLP explanation report from analysis results.

    Uses spaCy for sentence segmentation / noun-phrase extraction and
    NLTK WordNet for synonym variation to produce natural, varied text.

    Returns an ``ExplanationReport`` with titled sections and a reliability score.
    """
    score = _compute_reliability_score(model_report, spurious_report, shift_report)

    sections: list[ExplanationSection] = [
        _model_performance_section(model_report),
        _spurious_overview_section(spurious_report),
        *_spurious_detail_sections(spurious_report),
        _dataset_shift_section(shift_report),
        _reliability_section(score),
    ]

    return ExplanationReport(sections=sections, reliability_score=score)
