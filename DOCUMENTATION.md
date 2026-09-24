# OrgInsight Documentation

## 1. Project Overview

OrgInsight is a full-stack data analytics and AI project for analyzing a synthetic global organizations dataset. The project covers data preparation, KPI generation, SQL analytics, machine learning, explainability, dashboard generation, and AI-generated insights.

The dataset contains 100,000 organizations across 243 countries and 147 industries. The project is structured around a reproducible analytics pipeline that transforms raw data into validated insights and a user-friendly dashboard.

---

## 2. Business Objective

The main objective of this project is to explore patterns in organization size, geography, industry, founding trends, and segmentation. It also evaluates whether employee size classification can be predicted from metadata features such as country, industry, company age, and founding year.

The project demonstrates both descriptive analytics and predictive modeling, while keeping the results explainable and transparent.

---

## 3. Dataset

### Source

- `data/raw/organizations-100000.csv`

### Raw Columns

| Column | Type | Description |
|---|---|---|
| Index | integer | Row identifier |
| Organization Id | string | Unique organization identifier |
| Name | string | Organization name |
| Website | string | Organization website |
| Country | string | Country or region label |
| Description | string | Business description |
| Founded | integer | Founding year |
| Industry | string | Industry category |
| Number of employees | integer | Employee count |

### Dataset Summary

- 100,000 rows
- 243 countries
- 147 industries
- Synthetic but analytics-ready dataset
- No direct use of raw data for final reporting without cleaning and validation

---

## 4. Repository Structure

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
│   └── dashboard images
├── tests/
│   └── test_pipeline.py
├── .env.example
├── .gitignore
├── README.md
├── DOCUMENTATION.md
├── requirements.txt
├── run_pipeline.py
└── organizations-100000.csv
```

---

## 5. Project Workflow

The project follows a structured 8-phase workflow:

1. Data audit and validation
2. Data preparation and feature engineering
3. EDA and KPI generation
4. SQL analytics
5. Machine learning
6. Explainability analysis
7. Dashboard creation
8. AI-generated insights

---

## 6. Phase Details

### 6.1 Data Audit and Validation

The project begins with a review of the dataset to confirm structure, completeness, duplicate patterns, and validity of raw inputs.

This stage ensures the data is suitable for downstream processing and identifies any schema issues before modeling.

### 6.2 Data Preparation

The `src/data_preparation.py` module handles:

- loading the raw dataset
- correcting column types
- checking nulls and duplicates
- creating derived fields
- generating a cleaned dataset
- writing validation files

Derived fields include:

- `company_age`
- `founding_decade`
- `size_band`
- `broad_sector`
- `is_name_country_dup`

### 6.3 EDA and KPI Generation

The `src/eda_kpi.py` module computes summary metrics and chart outputs such as:

- employee distribution
- industry concentration
- country distribution
- founding year trends
- company age distribution
- sector breakdown

### 6.4 SQL Analytics

The `src/sql_analytics.py` module performs analytical querying using DuckDB. It extracts patterns such as:

- total organizations by industry and country
- average employees by sector
- company age trends
- size-band distributions
- top countries and industries
- year-over-year founding patterns

### 6.5 Machine Learning

The `src/ml_model.py` module trains machine learning models to predict size band using metadata. The target is the employee-size category.

Main model types:

- dummy baseline
- Random Forest default
- Random Forest balanced

Key idea:

- model should predict whether a company belongs to a known size category
- direct employee count is excluded to prevent leakage

### 6.6 Explainability

The `src/explainability.py` module generates explainability reports using:

- Gini feature importance
- permutation importance
- SHAP values

This helps determine whether model predictions are meaningful or mainly driven by high-cardinality encoded features.

### 6.7 Dashboard Creation

The `src/generate_dashboard.py` module creates an interactive HTML dashboard in `dashboard/index.html`.

It includes sections for:

- overview
- industry insights
- geographic insights
- founding pattern analysis
- ML results
- AI-generated summaries

### 6.8 AI Insights Layer

The `src/ai_insights.py` module reads validated analytical outputs and generates summaries using a supported AI endpoint. This layer is designed to provide contextual understanding without inventing unsupported metrics.

---

## 7. Key Metrics and Findings

The project produces key aggregate metrics such as:

- 100,000 total organizations
- 243 countries
- 147 industries
- average employee count around 5,004
- average company age around 30.3 years
- peak founding year around 1998

### Size Bands

| Size Band | Range | Approx. Share |
|---|---|---|
| Micro | < 50 employees | 0.5% |
| Small | 50–249 | 2.0% |
| Medium | 250–999 | 7.5% |
| Large | 1,000–4,999 | 40.0% |
| Enterprise | 5,000+ | 50.0% |

### Top Industries

- Insurance
- Hospital / Health Care
- Leisure / Travel
- Logistics / Procurement
- Non-Profit / Volunteering

### Top Countries

- Congo
- Korea
- Lebanon
- Iraq
- Turkey

### Modeling Insight

The project finds that the synthetic data has limited real predictive signal for employee-size band classification. While model metrics can be computed and compared, they often remain near baseline or weak in meaningful predictive power. This is an important analytical result rather than a failure of the pipeline.

---

## 8. Validation and Tests

The project includes automated validation tests in `tests/test_pipeline.py`.

The test suite covers:

- data preparation correctness
- KPI validation
- SQL analytics validation
- model artifact expectations
- explainability checks
- dashboard generation
- AI context generation

Run tests:

```bash
python -m pytest tests/ -v
```

---

## 9. How to Run the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run pipeline

```bash
python run_pipeline.py
```

### Run without retraining

```bash
python run_pipeline.py --skip-ml
```

### Launch dashboard locally

```bash
python -m http.server 8000 --directory dashboard
```

Then open:

```text
http://localhost:8000/
```

---

## 10. Environment Settings

Some features use optional AI APIs.

Use the template file:

```bash
copy .env.example .env
```

Set values such as:

- `AI_API_KEY`
- `AI_ENDPOINT`
- `AI_MODEL`

The project is designed so the AI layer only processes validated, trusted metrics and does not invent unsupported values.

---

## 11. Screenshots

The project includes dashboard and analytics screenshots in the `screenshots/` folder. These visuals are useful for documentation and demonstration purposes.

### Dashboard Screenshots

![Dashboard screenshot 1](screenshots/WhatsApp%20Image%202026-09-24%20at%2010.00.24%20PM.jpeg)

![Dashboard screenshot 2](screenshots/WhatsApp%20Image%202026-09-24%20at%2010.00.25%20PM.jpeg)

![Dashboard screenshot 3](screenshots/WhatsApp%20Image%202026-09-24%20at%2010.00.25%20PM%20(1).jpeg)

![Dashboard screenshot 4](screenshots/WhatsApp%20Image%202026-09-24%20at%2010.00.26%20PM.jpeg)

![Dashboard screenshot 5](screenshots/WhatsApp%20Image%202026-09-24%20at%2010.00.26%20PM%20(1).jpeg)

---

## 12. Important Notes

- The dataset is synthetic and designed for educational and analytical experimentation.
- High-cardinality variables like country and industry can appear important in explainability even when the dataset has weak predictive signal.
- The project emphasizes transparent analysis, not unsupported causal claims.
- Generated artifacts in `data/processed/` are validated before being used for AI narrative generation.

---

## 13. Conclusion

OrgInsight is a comprehensive example of a modern analytics workflow: from raw dataset validation to machine learning, explainability, dashboard visualization, and AI-assisted reporting. It is suitable as a project portfolio item and as an example of organized, reproducible data science work.

---

## 14. References

- `README.md`
- `reports/ai_insights.md`
- `docs/initial-audit.md`
- `data/processed/*.json`
- `tests/test_pipeline.py`
