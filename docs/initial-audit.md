# OrgInsight — Initial Project Audit
**Project:** OrgInsight — AI-Powered Organization Analytics & Employee Intelligence  
**Audit Date:** Phase 1 Inspection  
**Status:** Read-only audit. No files were modified.

---

## 1. Project Structure

```
a:\IBM_Bob_Data_Analytics_Project\
├── organizations-100000.csv        ← Root-level copy of raw dataset (present but not the canonical path)
├── README.md                       ← Project documentation
├── requirements.txt                ← Python dependencies
├── run_pipeline.py                 ← Full pipeline orchestrator
├── data/
│   ├── raw/
│   │   └── organizations-100000.csv   ← Canonical raw dataset (preserved, unmodified)
│   └── processed/
│       ├── organizations_clean.csv    ← Cleaned + enriched dataset (100,000 rows × 13 cols)
│       ├── preparation_report.json
│       ├── kpis.json
│       ├── ml_results.json
│       ├── explainability.json
│       ├── sql_analytics.json
│       ├── rf_model.joblib            ← Trained Random Forest model
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
│   └── index.html                  ← Interactive Plotly.js dashboard (6 tabs)
├── reports/
│   ├── ai_insights.md
│   └── fig_*.png  (11 chart files)
├── tests/
│   └── test_pipeline.py            ← 48 automated tests
├── notebooks/                      ← EMPTY — no notebooks present
└── docs/                           ← Created during this audit
```

**Observation:** `organizations-100000.csv` exists at both the workspace root and at `data/raw/`. The root copy is not used by any source module — all code reads from `data/raw/organizations-100000.csv`.

---

## 2. Dataset Inspection

### 2.1 File Paths
| Location | Status |
|---|---|
| `organizations-100000.csv` (root) | Present — not referenced by any `src/` module |
| `data/raw/organizations-100000.csv` | Present — canonical source used by all pipeline code |
| `data/processed/organizations_clean.csv` | Present — generated output |

### 2.2 Raw Dataset Properties

| Property | Value |
|---|---|
| Total rows | 100,000 |
| Total columns | 9 |
| File encoding | UTF-8 |
| Header row | Yes |

### 2.3 Column Names and Data Types

| # | Column | Raw Type (CSV) | Coerced Type | Notes |
|---|---|---|---|---|
| 1 | `Index` | string → int | Int64 | Sequential 1–100,000 |
| 2 | `Organization Id` | string | string | Hex-style IDs, mixed case |
| 3 | `Name` | string | string | Company name; 27,585 non-unique |
| 4 | `Website` | string | string | URL; not used analytically |
| 5 | `Country` | string | string | 243 unique values |
| 6 | `Description` | string | string | Jargon taglines; low analytical value |
| 7 | `Founded` | string → int | Int64 | Range: 1970–2022 |
| 8 | `Industry` | string | string | 147 unique values |
| 9 | `Number of employees` | string → int | Int64 | Range: 1–9,999; mean 5,003.96 |

### 2.4 Missing Values

All columns report **zero null values** and **zero empty strings** in the raw dataset. No imputation was required.

| Column | Null Count | Empty String Count |
|---|---|---|
| All 9 columns | 0 | 0 |

### 2.5 Duplicates

| Check | Result |
|---|---|
| Duplicate `Organization Id` | **0** — all IDs are unique |
| Duplicate `Name` | **27,585** — many companies share a name (e.g., "Smith LLC") |
| Duplicate rows (full) | Not explicitly checked; implied zero given unique IDs |

Name duplicates are expected in synthetic/generated data and do not affect analysis — `Organization Id` is the reliable unique key.

### 2.6 Numerical Fields — Summary Statistics

| Field | Min | Max | Mean | Median |
|---|---|---|---|---|
| `Founded` | 1,970 | 2,022 | — | — |
| `Number of employees` | 1 | 9,999 | 5,003.96 | 4,998 |
| `company_age` (derived) | 2 | 54 | 28.3 | 28.0 |

Employee count is distributed nearly uniformly across 1–9,999 (no strong skew).  
Founding year is distributed nearly uniformly across 1970–2022 (~1,850–1,970 orgs per year).

### 2.7 Categorical Fields — Key Distributions

**Industry** (147 unique values, near-uniform):

| Rank | Industry | Count |
|---|---|---|
| 1 | Insurance | 747 |
| 2 | Hospital / Health Care | 739 |
| 3 | Leisure / Travel | 736 |
| 4 | Logistics / Procurement | 732 |
| 5 | Non - Profit / Volunteering | 729 |

**Country** (243 unique values):

| Rank | Country | Count |
|---|---|---|
| 1 | Congo | 847 |
| 2 | Korea | 810 |
| 3 | Lebanon | 477 |
| 4 | Iraq | 470 |
| 5 | Turkey | 469 |

**Size Band** (derived — EU SME thresholds):

| Band | Threshold | Count | % |
|---|---|---|---|
| Micro | < 50 employees | 492 | 0.5% |
| Small | 50–249 | 1,994 | 2.0% |
| Medium | 250–999 | 7,537 | 7.5% |
| Large | 1,000–4,999 | 39,987 | 40.0% |
| Enterprise | ≥ 5,000 | 49,990 | 50.0% |

**Broad Sector** (derived — keyword-mapped from Industry):

| Sector | Count | % |
|---|---|---|
| Other | 70,070 | 70.1% |
| Technology | 9,536 | 9.5% |
| Healthcare | 5,413 | 5.4% |
| Finance | 5,407 | 5.4% |
| Manufacturing | 4,807 | 4.8% |
| Services | 4,767 | 4.8% |

> Note: "Other" dominates (70%) because keyword matching only covers 5 named sectors. 70 of 147 industry labels don't contain any mapped keyword.

### 2.8 Possible ML Target

The existing pipeline derives `size_band` from `Number of employees` and uses it as the classification target. This is the most natural supervised ML target given the available columns. `Number of employees` itself is excluded from features to prevent data leakage. No regression target (predicting exact employee count) has been implemented.

---

## 3. Existing Components — What Already Works

### 3.1 Data Preparation (`src/data_preparation.py`) ✅
- Loads raw CSV from `data/raw/`
- Validates schema (9 expected columns)
- Checks null counts, empty strings, duplicate IDs
- Coerces `Founded` and `Number of employees` to Int64
- Derives `company_age`, `founding_decade`, `size_band`, `broad_sector`
- Writes `data/processed/organizations_clean.csv` (100,000 rows × 13 columns)
- Writes `data/processed/preparation_report.json`
- Raw file is never modified

### 3.2 EDA & KPI Analytics (`src/eda_kpi.py`) ✅
- Computes 15+ KPI metrics from processed data
- Saves `data/processed/kpis.json`
- Generates 8 matplotlib/seaborn chart PNGs in `reports/`:
  - `fig_size_distribution.png`
  - `fig_top_industries.png`
  - `fig_top_countries.png`
  - `fig_founding_trend.png`
  - `fig_employee_distribution.png`
  - `fig_sector_breakdown.png`
  - `fig_avg_employees_by_sector.png`
  - `fig_decade_vs_employees.png`

### 3.3 SQL Analytics (`src/sql_analytics.py`) ✅
- Uses DuckDB (in-process, no server required)
- Runs 10 analytical queries (Q1–Q10)
- Covers: industry ranking, country ranking, sector overview, size bands, decade trends, cross-industry-country analysis, diversity scoring
- Saves `data/processed/sql_analytics.json`

### 3.4 ML Model (`src/ml_model.py`) ✅
- Task: 5-class classification → predict `size_band`
- Features: `company_age`, `Founded`, `industry_enc`, `country_enc`, `sector_enc`
- Models: Logistic Regression (baseline) + Random Forest (main, 200 trees, depth ≤ 15)
- Evaluation: accuracy, weighted F1, classification report, 5-fold CV
- Trained artifacts saved: `rf_model.joblib`, `scaler.joblib`, `encoders.joblib`
- Results: `data/processed/ml_results.json`

**Actual model performance (from ml_results.json):**

| Model | Accuracy | Weighted F1 | CV Mean | CV Std |
|---|---|---|---|---|
| Logistic Regression | 8.73% | 13.02% | — | — |
| Random Forest | 25.91% | 30.95% | 24.96% | ±0.71% |

Random baseline for 5 classes = 20%. Random Forest beats random by ~6pp.

### 3.5 Model Explainability (`src/explainability.py`) ✅
- Gini feature importances from trained Random Forest
- SHAP TreeExplainer on 2,000-record stratified sample
- Saves `data/processed/explainability.json`
- Generates: `fig_feature_importance_clean.png`, `fig_shap_importance.png`, `fig_shap_enterprise.png`

**Feature importance results:**

| Feature | Gini Importance | SHAP Mean |SHAP|| 
|---|---|---|
| Country (enc) | 34.63% | 0.012679 |
| Industry (enc) | 28.73% | 0.015201 |
| Founded | 15.05% | 0.021776 |
| Company Age | 15.01% | 0.039047 |
| Broad Sector (enc) | 6.58% | 0.012234 |

### 3.6 Interactive Dashboard (`dashboard/index.html`) ✅
- Self-contained HTML, ~30 KB
- Uses Plotly.js 2.27.0 (CDN)
- 6 tabs: Overview, Industries, Geography, Growth Trends, ML Model, AI Insights
- All KPI values hard-baked from real data at generation time (not live-queried)
- AI tab includes a configurable endpoint form for Ollama/OpenAI-compatible APIs

### 3.7 AI Insights Layer (`src/ai_insights.py`) ✅
- CLI tool with `--endpoint`, `--model`, `--output`, `--openai` flags
- Reads only validated JSON artifacts (kpis.json, ml_results.json, explainability.json, sql_analytics.json)
- Falls back to structured Markdown report if no API is configured
- API key read from `AI_API_KEY` environment variable — never hard-coded
- Current output: `reports/ai_insights.md` (structured fallback, no LLM connected)

### 3.8 Pipeline Runner (`run_pipeline.py`) ✅
- Runs all phases in order with timing
- `--skip-ml` flag to bypass ML training
- `--ai-endpoint` and `--ai-model` flags

### 3.9 Tests (`tests/test_pipeline.py`) ✅
- **48 tests, 48 passing** (last run: 2.36s)
- Covers all 7 phases with class-grouped tests
- Validates: schema, nulls, derived columns, KPI values, SQL result completeness, ML metrics, SHAP keys, dashboard content, AI insights file

### 3.10 Documentation (`README.md`) ✅
- Project overview, structure, quick-start commands
- Phase summary table
- Key findings section with actual metrics

---

## 4. What Is Missing

| Gap | Detail | Priority |
|---|---|---|
| `notebooks/` directory is empty | No Jupyter notebooks for exploratory or narrative analysis | Medium |
| No `docs/` directory (before this audit) | No dedicated documentation folder beyond README | Low |
| Dashboard uses CDN Plotly.js | Requires internet connection to render charts | Low |
| AI layer has no live LLM connected | Falls back to structured Markdown — no narrative text generated yet | Medium |
| `broad_sector` "Other" = 70.1% | Keyword mapping leaves 70 of 147 industries uncategorized | Medium |
| No regression ML target | Employee count prediction (regression) not implemented | Low |
| `company_age` hard-coded to 2024 | Will silently become stale as time passes | Low |
| Dashboard KPIs are static | Values baked at generation time; require re-running `generate_dashboard.py` to refresh | Low |
| No `__init__.py` in `src/` | `src` is not a proper Python package; imports work via `sys.path` hack in tests | Low |
| Root-level CSV duplicate | `organizations-100000.csv` exists at root and `data/raw/` — only the latter is used | Low |

---

## 5. Component Reusability Assessment

| Component | Reusable As-Is | Notes |
|---|---|---|
| `data_preparation.py` | ✅ Yes | Clean, modular, importable functions |
| `eda_kpi.py` | ✅ Yes | `compute_kpis()` and `run_eda()` are independent |
| `sql_analytics.py` | ✅ Yes | DuckDB queries are self-contained |
| `ml_model.py` | ✅ Yes | `train_and_evaluate()` is standalone |
| `explainability.py` | ✅ Yes | Loads saved model artifacts |
| `generate_dashboard.py` | ✅ Yes | Pure data-in → HTML-out |
| `ai_insights.py` | ✅ Yes | Works offline with fallback |
| `run_pipeline.py` | ✅ Yes | Orchestrates all phases |
| `tests/test_pipeline.py` | ✅ Yes | All 48 tests pass |

---

## 6. Dependencies (requirements.txt)

| Package | Version Constraint | Used By |
|---|---|---|
| pandas | ≥2.0.0 | data_preparation, eda_kpi, ml_model, explainability |
| numpy | ≥1.24.0 | ml_model, explainability |
| scikit-learn | ≥1.3.0 | ml_model |
| matplotlib | ≥3.7.0 | eda_kpi, ml_model, explainability |
| seaborn | ≥0.12.0 | eda_kpi, ml_model, explainability |
| plotly | ≥5.15.0 | generate_dashboard (CDN only — not imported as Python lib) |
| shap | ≥0.42.0 | explainability |
| duckdb | ≥0.9.0 | sql_analytics |
| jinja2 | ≥3.1.0 | Listed but not actively imported in any src module |
| requests | ≥2.31.0 | Listed but not used (ai_insights uses urllib.request) |
| pytest | ≥7.4.0 | tests/ |

> `jinja2` and `requests` are listed in requirements but not imported by any current source module.

---

## 7. Summary

The project is **functionally complete** across all 8 analytical phases. The pipeline runs end-to-end from raw CSV → cleaned data → KPIs → SQL analytics → ML model → SHAP explainability → interactive dashboard → AI insights report. All 48 automated tests pass.

The dataset is clean (zero nulls, zero structural issues), synthetic in character (near-uniform distributions), and well-suited for organizational analytics. The ML model's modest accuracy (~26%) is an analytically valid and correctly documented finding — it reflects the near-uniform distribution of employee count across industries and geographies in this dataset.

The primary gaps for the **OrgInsight** product vision are:
1. No Jupyter notebooks for narrative/exploratory presentation
2. The `broad_sector` mapping leaves 70% of records in "Other"
3. The AI layer has no live LLM integration tested yet
4. The dashboard title and branding still references the generic project name, not "OrgInsight"
