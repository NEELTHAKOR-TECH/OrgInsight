# OrgInsight

OrgInsight is a complete end-to-end analytics and AI project for understanding global organizations at scale. The project analyzes 100,000 organizations across 243 countries and 147 industries, performs data quality checks, SQL-based analytics, machine learning, explainability analysis, and builds an interactive dashboard with AI-generated insights.

This project demonstrates a full data science workflow from raw dataset ingestion to validated reporting and product-style dashboard delivery.

## Table of Contents

- Overview
- Project goals
- Tech stack
- Dataset description
- Repository structure
- Installation and setup
- Run the pipeline
- Open the dashboard
- Project phases and outputs
- Key findings
- Testing and validation
- Screenshots
- License and notes

---

## Overview

OrgInsight answers a practical analytics question: what patterns exist across a synthetic global organization dataset, and can a machine learning pipeline produce interpretable insights from it?

The project is designed to:

- clean and validate a large raw dataset
- generate derived business features like company age and size bands
- compute KPI summaries and distribution metrics
- run SQL analytics using DuckDB
- train and evaluate machine learning models
- explain model behavior using SHAP and permutation importance
- generate an interactive HTML dashboard
- synthesize AI-based explainable findings from validated data

---

## Project Goals

1. Build a robust and reproducible data preparation pipeline.
2. Turn raw organizational data into business-ready analysis features.
3. Create KPI-driven exploratory analytics on organization count, industry, geography, and employee patterns.
4. Test whether employee-size classification can be predicted from metadata features.
5. Evaluate whether model explanations are meaningful or indicate no real predictive signal.
6. Deliver a browser-based dashboard for stakeholders.
7. Provide a professional GitHub-ready documentation set.

---

## Tech Stack

- Python 3.10+
- Pandas
- NumPy
- Scikit-learn
- DuckDB
- Plotly
- Matplotlib / Seaborn
- Joblib
- JSON validation workflow
- Optional AI backends: Ollama or OpenAI-compatible API

---

## Dataset Description

The canonical source file is:

- `data/raw/organizations-100000.csv`

Raw schema:

| Column | Type | Description |
|---|---|---|
| `Index` | integer | Row number |
| `Organization Id` | string | Unique organization ID |
| `Name` | string | Organization name |
| `Website` | string | Website URL |
| `Country` | string | Country code or country name |
| `Description` | string | Free text summary |
| `Founded` | integer | Foundation year |
| `Industry` | string | Industry label |
| `Number of employees` | integer | Employee count |

Key dataset metrics:

- 100,000 organizations
- 243 countries
- 147 industries
- 5 derived size bands
- dynamic company-age calculation based on current year

---

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   └── organizations-100000.csv
│   └── processed/
│       ├── organizations_clean.csv
│       ├── preparation_report.json
│       ├── kpis.json
│       ├── kpis_validation.json
│       ├── sql_analytics.json
│       ├── sql_analytics_validation.json
│       ├── ml_results.json
│       ├── ml_validation.json
│       ├── explainability.json
│       ├── explainability_validation.json
│       ├── rf_model.joblib
│       ├── rf_model_default.joblib
│       ├── scaler.joblib
│       └── encoders.joblib
├── src/
│   ├── data_preparation.py
│   ├── eda_kpi.py
│   ├── sql_analytics.py
│   ├── ml_model.py
│   ├── explainability.py
│   ├── generate_dashboard.py
│   └── ai_insights.py
├── dashboard/
│   └── index.html
├── reports/
│   ├── ai_insights.md
│   └── fig_*.png
├── docs/
│   └── initial-audit.md
├── screenshots/
│   └── dashboard screenshots for README and demo
├── tests/
│   └── test_pipeline.py
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── run_pipeline.py
└── organizations-100000.csv
```

---

## Installation and Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If you want to use AI inference features, configure the environment file:

```bash
copy .env.example .env
```

Then update `.env` with your model endpoint and API key when needed.

---

## Running the Project

### Run the full pipeline

```bash
python run_pipeline.py
```

### Skip retraining and use saved model artifacts

```bash
python run_pipeline.py --skip-ml
```

### Run with Ollama AI backend

```bash
python run_pipeline.py --ai-endpoint http://localhost:11434/api/generate --ai-model llama3
```

### Run AI insights with OpenAI-compatible API

```bash
python src/ai_insights.py --openai
```

Or via environment file injection:

```bash
AI_API_KEY=your_key python src/ai_insights.py --endpoint https://api.openai.com/v1/chat/completions --model gpt-4o --openai
```

### Run tests

```bash
python -m pytest tests/ -v
```

The project includes 155 automated tests covering the pipeline, KPI generation, SQL analytics, model evaluation, explainability, dashboard content, and AI report generation.

---

## Open the Dashboard

Open the generated dashboard in a browser:

```text
dashboard/index.html
```

Or serve it locally:

```bash
python -m http.server 8000 --directory dashboard
```

Then open:

```text
http://localhost:8000/
```

---

## Project Workflow and Phases

### Phase 1 — Audit and project inspection

- review raw structure and data integrity
- identify canonical data source
- establish project scope and validation approach

### Phase 2 — Data preparation

Handled in `src/data_preparation.py`.

Key activities:

- load raw CSV data
- clean invalid or malformed values
- validate schema and null counts
- derive business features such as:
  - `company_age`
  - `founding_decade`
  - `size_band`
  - `broad_sector`
  - `is_name_country_dup`
- generate the cleaned dataset and preparation report

Output files:

- `data/processed/organizations_clean.csv`
- `data/processed/preparation_report.json`

### Phase 3 — EDA and KPI analytics

Handled in `src/eda_kpi.py`.

This phase computes summary metrics, distribution plots, and high-level descriptive statistics across:

- industry distribution
- country distribution
- employee counts
- company age
- founding year trend
- sector summaries

Output files:

- `data/processed/kpis.json`
- `reports/fig_*.png`

### Phase 4 — SQL analytics

Handled in `src/sql_analytics.py`.

The project uses DuckDB to run analytical queries such as:

- dataset totals
- top industries and countries
- sector share analysis
- size-band distributions
- founding trends
- country diversity metrics
- percentiles and enterprise summaries

Output files:

- `data/processed/sql_analytics.json`

### Phase 5 — Machine learning

Handled in `src/ml_model.py`.

The task is a 5-class classification problem to predict employee size band based on metadata features, while excluding direct employee count leakage.

Modeling summary:

- target: employee size band
- features used: `company_age`, `Founded`, `industry_enc`, `country_enc`, `sector_enc`, `is_name_country_dup`
- model candidates: dummy baseline, Random Forest default, Random Forest balanced
- evaluation metrics: accuracy, macro F1, weighted F1, CV stability

Output files:

- `data/processed/ml_results.json`
- `data/processed/rf_model.joblib`
- `data/processed/rf_model_default.joblib`
- `data/processed/scaler.joblib`
- `data/processed/encoders.joblib`

### Phase 6 — Explainability

Handled in `src/explainability.py`.

This stage applies three explainability methods to interpret model outputs:

- Gini impurity feature importance
- permutation importance
- SHAP values

Output files:

- `data/processed/explainability.json`
- `reports/fig_feature_importance.png`
- `reports/fig_permutation_importance.png`
- `reports/fig_shap_importance.png`

### Phase 7 — Dashboard generation

Handled in `src/generate_dashboard.py`.

This creates a browser-based interactive dashboard with sections for:

- overview
- industry analysis
- geography analysis
- founding trends
- machine learning results
- AI insights

Output file:

- `dashboard/index.html`

### Phase 8 — AI insights layer

Handled in `src/ai_insights.py`.

This layer reads valid, previously computed metrics from processed files and passes them to an AI endpoint. It avoids fabricating unsupported values and supports both OpenAI-compatible and Ollama-compatible APIs.

Output file:

- `reports/ai_insights.md`

---

## Key Findings

### Data quality

Validated checks show:

- zero nulls in source columns
- zero duplicate `Organization Id` values
- zero full-row duplicates
- clean numeric conversion for `Founded` and `Number of employees`
- low outlier pressure and stable data quality

### Dataset summary

From processed KPI generation:

| KPI | Value |
|---|---|
| Total organizations | 100,000 |
| Total industries | 147 |
| Total countries | 243 |
| Average employees | 5,004.0 |
| Median employees | 4,998.0 |
| Average company age | 30.3 years |
| Peak founding year | 1998 |
| Top 10 countries share | 5.34% |

### Broad sector distribution

| Sector | Count | Share |
|---|---:|---:|
| Other | 70,070 | 70.07% |
| Technology | 9,536 | 9.54% |
| Healthcare | 5,413 | 5.41% |
| Finance | 5,407 | 5.41% |
| Manufacturing | 4,807 | 4.81% |
| Services | 4,767 | 4.77% |

### Top industries

| Industry | Count |
|---|---:|
| Insurance | 747 |
| Hospital / Health Care | 739 |
| Leisure / Travel | 736 |
| Logistics / Procurement | 732 |
| Non-Profit / Volunteering | 729 |

### Top countries

| Country | Count |
|---|---:|
| Congo | 847 |
| Korea | 810 |
| Lebanon | 477 |
| Iraq | 470 |
| Turkey | 469 |

### Model findings

The machine learning task is informative but the underlying synthetic dataset has limited predictive structure.

- Dummy baseline accuracy: about 49.99%
- Random Forest default accuracy: about 49.25%
- Random Forest balanced accuracy: about 24.68%
- Most feature-target correlations are near zero
- The dataset appears to assign employee counts largely independently of the metadata features

This is a key project outcome: the model does not uncover a strong causal or predictive relationship in the synthetic data, and the explainability findings must be interpreted cautiously.

---

## Validation and Testing

The repository includes validation logic and a full automated test suite.

Validation includes:

- schema checks
- null checks
- duplicate checks
- feature engineering validation
- result file integrity checks
- dashboard file checks
- AI context validation

Automated tests:

```bash
python -m pytest tests/ -v
```

The test suite covers:

- data preparation
- KPI correctness
- SQL analytics queries
- machine learning outcomes
- explainability outputs
- dashboard generation
- AI insight generation

---

## Screenshots

The project includes dashboard and analytics screenshots in the `screenshots` folder.

![Dashboard screenshot 1](screenshots/WhatsApp%20Image%202026-09-24%20at%2010.00.24%20PM.jpeg)

![Dashboard screenshot 2](screenshots/WhatsApp%20Image%202026-09-24%20at%2010.00.25%20PM.jpeg)

![Dashboard screenshot 3](screenshots/WhatsApp%20Image%202026-09-24%20at%2010.00.25%20PM%20(1).jpeg)

![Dashboard screenshot 4](screenshots/WhatsApp%20Image%202026-09-24%20at%2010.00.26%20PM.jpeg)

![Dashboard screenshot 5](screenshots/WhatsApp%20Image%202026-09-24%20at%2010.00.26%20PM%20(1).jpeg)

---

## Notes and Caveats

- The dataset is synthetic and intentionally designed for pattern exploration rather than real-world causal inference.
- Employee count distribution is nearly uniform, which limits strong predictive signal.
- Explainability methods can rank high-cardinality encoded features highly, even when the model has little real predictive power.
- The project validates findings from generated JSON artifacts before using them in AI summaries.

---

## Summary

OrgInsight is a practical, full-stack-style analytics project that combines:

- data engineering
- KPI reporting
- SQL analytics
- predictive modeling
- explainability
- dashboarding
- AI-generated narrative summaries

It is designed as both a documentation-rich project and a demonstration of professional end-to-end data science workflow execution.

---

## License

This project is intended for educational and demonstration purposes. Please check the repository policies and any applicable academic or organizational requirements before reuse in production environments.

- `.env.example` exists and documents `AI_API_KEY`
- `.gitignore` excludes `.env`
- Fallback runs cleanly with empty endpoint and produces valid report
