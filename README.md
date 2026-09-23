# OrgInsight — Global Organizations Analytics + AI Project

A complete academic Data Analytics and AI project analysing **100,000 global organizations** across 243 countries and 147 industries.
Built in 8 phases: data preparation → EDA/KPIs → SQL analytics → ML modelling → explainability → interactive dashboard → AI insights.

---

## Dataset

`data/raw/organizations-100000.csv` — **100,000 rows × 9 columns** (never modified):

| Column | Type | Notes |
|---|---|---|
| `Index` | int | Row number |
| `Organization Id` | string | Unique 15-char hex ID |
| `Name` | string | Company name |
| `Website` | string | URL |
| `Country` | string | 243 unique values |
| `Description` | string | Free text |
| `Founded` | int | 1970–2022 |
| `Industry` | string | 147 unique values |
| `Number of employees` | int | 1–9,999 |

---

## Project Structure

```
├── data/
│   ├── raw/
│   │   └── organizations-100000.csv    # NEVER MODIFIED — canonical source
│   └── processed/
│       ├── organizations_clean.csv     # 100,000 rows × 14 cols (9 raw + 5 derived)
│       ├── preparation_report.json     # Phase 2 quality report
│       ├── kpis.json                   # 60+ KPIs (reference_year dynamic)
│       ├── kpis_validation.json        # 15/15 checks PASSED
│       ├── sql_analytics.json          # 18 DuckDB queries (Q01–Q18)
│       ├── sql_analytics_validation.json
│       ├── ml_results.json             # Dummy + RF Default + RF Balanced
│       ├── ml_validation.json          # 18/18 checks PASSED
│       ├── explainability.json         # Gini MDI + Permutation + SHAP
│       ├── explainability_validation.json
│       ├── rf_model.joblib             # RF Balanced (class_weight='balanced')
│       ├── rf_model_default.joblib     # RF Default
│       ├── scaler.joblib
│       └── encoders.joblib
├── src/
│   ├── data_preparation.py             # Phase 2 — cleaning & enrichment
│   ├── eda_kpi.py                      # Phase 3 — EDA & KPI computation
│   ├── sql_analytics.py                # Phase 4 — DuckDB analytical queries
│   ├── ml_model.py                     # Phase 5 — feature engineering & ML
│   ├── explainability.py               # Phase 6 — Gini MDI + Permutation + SHAP
│   ├── generate_dashboard.py           # Phase 7 — interactive HTML dashboard
│   └── ai_insights.py                  # Phase 8 — AI narrative layer
├── dashboard/
│   └── index.html                      # ← Open in any browser (35 KB, 6 tabs)
├── reports/
│   ├── fig_*.png                       # 23 chart PNGs (EDA + ML + Explainability)
│   └── ai_insights.md                  # Structured findings report
├── tests/
│   └── test_pipeline.py                # 155 automated tests
├── .env.example                        # Environment variable reference (copy to .env)
├── .gitignore
├── run_pipeline.py                     # Full pipeline runner
└── requirements.txt
```

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run full pipeline
```bash
python run_pipeline.py
```

### Skip ML re-training (use existing model artefacts)
```bash
python run_pipeline.py --skip-ml
```

### Run with AI integration (Ollama)
```bash
python run_pipeline.py --ai-endpoint http://localhost:11434/api/generate --ai-model llama3
```

### Run with OpenAI-compatible API
```bash
# Option A — .env file (recommended, key never in shell history)
cp .env.example .env
# Edit .env: set AI_API_KEY, AI_ENDPOINT, AI_MODEL
python src/ai_insights.py --openai

# Option B — inline
AI_API_KEY=sk-... python src/ai_insights.py \
  --endpoint https://api.openai.com/v1/chat/completions \
  --model gpt-4o --openai
```

### Run tests
```bash
python -m pytest tests/ -v
# 155 passed in ~9s
```

### Open dashboard
Open `dashboard/index.html` in any modern browser.

---

## Phase Summary

| Phase | Module | Output |
|---|---|---|
| 2 — Data Preparation | `src/data_preparation.py` | `organizations_clean.csv` (14 cols), `preparation_report.json` |
| 3 — EDA & KPIs | `src/eda_kpi.py` | `kpis.json` (60+ KPIs), 13 EDA chart PNGs |
| 4 — SQL Analytics | `src/sql_analytics.py` | `sql_analytics.json` (18 queries, Q01–Q18) |
| 5 — ML Model | `src/ml_model.py` | `rf_model.joblib`, `ml_results.json` (Dummy + RF Default + RF Balanced) |
| 6 — Explainability | `src/explainability.py` | `explainability.json` (Gini MDI + Permutation + SHAP), 5 plots |
| 7 — Dashboard | `src/generate_dashboard.py` | `dashboard/index.html` (35 KB, 6 tabs, sector filter) |
| 8 — AI Insights | `src/ai_insights.py` | `reports/ai_insights.md` |

---

## Key Findings

### Data Quality (Phase 2)

All results verified from `data/processed/preparation_report.json`:

- **Zero nulls** across all 9 raw columns (pre- and post-processing)
- **Zero full-row duplicates**, zero `Organization Id` duplicates
- **Zero type errors**: Founded and Number of employees coerced cleanly; 0 failures
- **Zero statistical outliers** in employee count (IQR fences: −4,984.5 to 14,995.5; z-score range: −1.73 to +1.73)
- **764 rows** share the same Name+Country — all have unique `Organization Id` values and different Industry/Founded. Flagged via `is_name_country_dup` (not dropped)
- **5 derived columns added**: `company_age`, `founding_decade`, `size_band`, `broad_sector`, `is_name_country_dup`
- `company_age` reference year: **dynamic** (`datetime.date.today().year`, currently 2026)
- Known limitation: `broad_sector = "Other"` covers **70,070 rows (70.1%)** — 70 of 147 industry labels match none of the 5 keyword-based sector mappings

---

### Dataset Overview (Phase 3 — KPIs)

All values from `data/processed/kpis.json`:

| KPI | Value |
|---|---|
| Total organizations | 100,000 |
| Total industries | 147 |
| Total countries | 243 |
| Average employees | 5,004.0 |
| Median employees | 4,998.0 |
| Std dev employees | 2,889.5 |
| Employee skewness | 0.001 (near-symmetric) |
| Employee kurtosis | −1.197 (platykurtic — uniform-like) |
| Employee range | 1 – 9,999 |
| Average company age | 30.3 years |
| Median company age | 30.0 years |
| Company age range | 4 – 56 years (ref year: 2026) |
| Founded range | 1970 – 2022 |
| Peak founding year | 1998 (2,022 orgs) |
| Avg orgs founded per year | ~1,887 |
| Industry HHI | 0.0068 (near-perfectly uniform) |
| Industry count range | 626 – 747 orgs per industry |
| Top 10 countries share | 5.34% (wide geographic spread) |
| Name+Country duplicates | 764 rows flagged |

#### Size Band Distribution

| Band | Employee Range | Count | % |
|---|---|---|---|
| Micro | < 50 | 492 | 0.5% |
| Small | 50 – 249 | 1,994 | 2.0% |
| Medium | 250 – 999 | 7,537 | 7.5% |
| Large | 1,000 – 4,999 | 39,987 | 40.0% |
| Enterprise | 5,000+ | 49,990 | 50.0% |

#### Top 10 Industries (by organization count)

| Industry | Count |
|---|---|
| Insurance | 747 |
| Hospital / Health Care | 739 |
| Leisure / Travel | 736 |
| Logistics / Procurement | 732 |
| Non-Profit / Volunteering | 729 |
| Publishing Industry | 726 |
| Farming | 722 |
| Religious Institutions | 721 |
| Packaging / Containers | 721 |
| Electrical / Electronic Manufacturing | 718 |

#### Top 10 Countries (by organization count)

| Country | Count |
|---|---|
| Congo | 847 |
| Korea | 810 |
| Lebanon | 477 |
| Iraq | 470 |
| Turkey | 469 |
| Austria | 455 |
| Seychelles | 454 |
| Mauritius | 453 |
| Singapore | 452 |
| Israel | 450 |

#### Broad Sector Breakdown

| Sector | Count | % | Avg Employees |
|---|---|---|---|
| Other | 70,070 | 70.07% | 5,014.6 |
| Technology | 9,536 | 9.54% | 5,017.9 |
| Healthcare | 5,413 | 5.41% | 4,918.5 |
| Finance | 5,407 | 5.41% | 4,972.8 |
| Manufacturing | 4,807 | 4.81% | 5,013.1 |
| Services | 4,767 | 4.77% | 4,943.1 |

*Note: Avg employees are near-identical across all sectors (range: 4,918–5,018) — consistent with the uniform employee-count distribution in this synthetic dataset.*

---

### SQL Analytics (Phase 4)

18 DuckDB queries (Q01–Q18) covering:

| Query | Topic |
|---|---|
| Q01 | Dataset totals |
| Q02 | Top 20 industries by count |
| Q03 | Top 20 countries by count |
| Q04 | Sector overview with % share |
| Q05 | Size band distribution |
| Q06 | Founding decade trends |
| Q07 | Top industry per top country |
| Q08 | Top/Bottom 10 industries by avg employees |
| Q09 | Founding trend by year (all 53 years) |
| Q10 | Sector × size band crosstab |
| Q11 | Most industrially diverse countries |
| Q12 | Employee percentile distribution (p10–p90) |
| Q13 | Top/Bottom 10 countries by avg employees |
| Q14 | Year-on-year founding count change |
| Q15 | Name+Country duplicate flag summary |
| Q16 | Industry spread uniformity (HHI) |
| Q17 | Peak and trough founding years |
| Q18 | Enterprise rate by sector |

---

### ML Model (Phase 5)

**Task**: Multi-class classification — predict company size band (5 classes) from 6 features.
**Train/test split**: 80,000 / 20,000 rows (stratified).
**Features**: `company_age`, `Founded`, `industry_enc`, `country_enc`, `sector_enc`, `is_name_country_dup`
**Excluded**: `Number of employees` (directly defines `size_band` — including it is data leakage).

All values from `data/processed/ml_results.json`:

| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC (OvR) |
|---|---|---|---|---|
| Dummy (most-frequent) | 49.99% | — | — | — |
| RF Default | 49.25% | — | — | 0.4953 |
| RF Balanced | 24.68% | 0.1584 | 0.2978 | 0.4955 |

**5-Fold Cross-Validation (RF Balanced)**: 24.91% ± 0.66%
**5-Fold Cross-Validation (RF Default)**: 49.18% ± 0.14%

**Max |Pearson r| (feature vs target)**: 0.003398 — all Spearman p > 0.23

**Finding**: RF Default accuracy equals the dummy baseline; RF Balanced trades overall accuracy for minority-class recall. All feature–target correlations are statistically indistinguishable from zero. This is a genuine null result — the synthetic dataset assigns employee counts independently of all other fields. These are *association* findings — **NOT causal claims**.

---

### Explainability (Phase 6)

Three independent methods applied to the RF Balanced model. All values from `data/processed/explainability.json`:

#### 1. Gini MDI (Mean Decrease in Impurity)

| Feature | Gini MDI % |
|---|---|
| Country (enc) | 33.75% |
| Industry (enc) | 28.16% |
| Year Founded | 15.53% |
| Company Age | 15.50% |
| Broad Sector (enc) | 6.83% |
| Name-Country Dup Flag | 0.22% |

*⚠ Gini MDI overstates high-cardinality encoded features. Country has 243 unique values; Industry has 147.*

#### 2. Permutation Importance — Decisive Test

*Method: sklearn permutation_importance, accuracy scoring, 5 repeats, test set = 20,000 rows*
*Negative/near-zero values = shuffling the feature does not reduce accuracy = no predictive signal.*

| Feature | Mean Accuracy Drop |
|---|---|
| Year Founded | −0.07804 |
| Company Age | −0.07585 |
| Industry (enc) | −0.00565 |
| Broad Sector (enc) | −0.00476 |
| Country (enc) | −0.00176 |
| Name-Country Dup Flag | +0.00017 |

**All values near zero or negative — confirming no genuine predictive signal in any feature.**

#### 3. SHAP (TreeExplainer, mean |SHAP|)

*Sample: 1,697 rows (stratified from test set)*

| Feature | Mean |SHAP| |
|---|---|
| Country (enc) | 0.020374 |
| Industry (enc) | 0.018239 |
| Company Age | 0.012699 |
| Year Founded | 0.012309 |
| Broad Sector (enc) | 0.007187 |
| Name-Country Dup Flag | 0.000178 |

#### Cross-Method Agreement

| Method | Top Feature |
|---|---|
| Gini MDI | Country (enc) |
| Permutation | Name-Country Dup Flag *(noise — all near zero)* |
| SHAP | Country (enc) |

Methods do **not** agree on a single top feature. Gini and SHAP both rank Country (enc) highest due to its 243 unique encoded values; permutation importance, the most reliable measure, shows all features are effectively uninformative.

---

### Dashboard (Phase 7)

`dashboard/index.html` — 35 KB, opens in any modern browser, no server required.

**6 tabs**: Overview · Industry · Geography · Founding Trends · ML Results · AI Insights
**Sector filter** on Overview tab (All Sectors + 6 individual sectors).
All KPI cards and chart values are dynamically loaded from `kpis.json` at generation time — no hard-coded numbers.

---

## AI Layer (Phase 8)

[`src/ai_insights.py`](src/ai_insights.py) feeds only **validated analytical results** to an LLM. It never fabricates metrics.

- API key read from `AI_API_KEY` environment variable — **never hard-coded**
- Copy `.env.example` → `.env` and set your key; `.env` is git-ignored
- Structured fallback report generated automatically when no API endpoint is configured
- Timeout configurable via `--timeout` (default: 120 s); HTTP errors reported with status code
- Supports both Ollama-compatible and OpenAI-compatible endpoints
- `.env` values loaded without requiring `python-dotenv` (built-in parser in `ai_insights.py`)

---

## Tests

**155 automated tests** covering all phases:

```bash
python -m pytest tests/ -v
# 155 passed in ~9s
```

| Test class | Tests | Coverage |
|---|---|---|
| `TestDataPreparation` | 27 | Raw schema, nulls, types, derived columns, size bands, sectors, duplicates, outliers |
| `TestKPIs` | 28 | KPI values, validation file, all charts generated, skewness, HHI, concentrations |
| `TestSQLAnalytics` | 27 | All 18 queries present, row counts, totals, peak year, cross-tab, dup flag |
| `TestMLModel` | 25 | Model files, accuracy bounds, CV stability, correlations, ROC-AUC, plots |
| `TestExplainability` | 21 | Gini sum, SHAP non-negative, permutation present, cross-method, all plots |
| `TestDashboard` | 4 | File exists, size, Plotly present, real KPI values in HTML |
| `TestAIInsights` | 20 | Output content, actual data used, no hard-coded key, `.env.example`, `.gitignore`, fallback |
| **Total** | **155** | |

`TestAIInsights` specifically verifies:
- Report references all 3 explainability methods (Gini MDI, Permutation, SHAP)
- `load_validated_context()` values exactly match source JSON files (no fabrication)
- Leakage guard: employee count not in feature list
- No `sk-...`-style API key pattern in source code
- `.env.example` exists and documents `AI_API_KEY`
- `.gitignore` excludes `.env`
- Fallback runs cleanly with empty endpoint and produces valid report
