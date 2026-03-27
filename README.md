# Spurious Correlation Auditor: Post-Hoc Audit for ML and NLP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B.svg)](https://streamlit.io/)

A comprehensive framework for auditing Machine Learning and NLP models. This tool identifies **spurious correlations**, evaluates **dataset shifts**, and provides **explainable reliability reports** during the model validation phase.

---

## 🔍 Key Features

- **Spurious Correlation Detection**: Detects features (e.g., image backgrounds, non-causal words) that models rely on for predictions despite having low causal relevance.
- **Explainability (SHAP/XAI)**: Integrated SHAP analysis for both tabular/NLP and image-based data to visualize decision-making.
- **Dataset Shift Analysis**: Identifies covariate shift and label shift using statistical tests (Kolmogorov-Smirnov, KL Divergence).
- **Cross-Domain Auditing**: Unified framework for both Natural Language Processing (IMDB) and Computer Vision (Waterbirds) tasks.
- **Automated Reporting**: Generates high-quality PDF and plain-text audit reports suitable for research papers and industrial compliance.

## 🏗️ Architecture

The system is modularly designed into specific auditing components:
- `modules.data_input`: Robust loaders for built-in and custom datasets.
- `modules.spurious_detection`: Core logic for identifying non-causal shortcuts.
- `modules.model_evaluation`: Metric comparison (Precision, Recall, F1) across baseline and main models.
- `modules.nlp_explanation`: Natural language interpretation of statistical risks.

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher.
- `pip` or any other virtual environment manager.

### Installation
1. Clone the repository:
   ```bash
   git clone <REMOTE_URL>
   cd ml-model-auditor
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
Run the auditing dashboard:
```bash
streamlit run app.py
```

## 📊 Research Application

The output format is specifically structured for **Research Paper presentation**, providing:
- **Model Performance Comparison Tables**
- **SHAP Intensity Heatmaps**
- **Detected Spurious Factor Summaries**
- **Cross-Domain Reliability Benchmarks**

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
