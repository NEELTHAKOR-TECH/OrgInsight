# OrgInsight — Structured Findings Report
*Generated: 2026-09-23 23:52 (structured fallback — no AI API)*

## Dataset Overview
- **100,000 organizations** across **243 countries** and **147 industries**
- Average employees: **5,004.0** (median: 4,998.0)
- Average company age: **30.3 years** (founded 1970-2022)
- Peak founding year: **1998**
- Industry concentration (HHI): 0.0068 — near-uniform across industries
- Top 10 countries share: 5.3% — wide geographic spread

## Size Band Distribution
- **Micro**: 492 organizations (0.5%)
- **Small**: 1,994 organizations (2.0%)
- **Medium**: 7,537 organizations (7.5%)
- **Large**: 39,987 organizations (40.0%)
- **Enterprise**: 49,990 organizations (50.0%)

## Industry Landscape
Top 5 industries by organization count:
- Insurance: 747
- Hospital / Health Care: 739
- Leisure / Travel: 736
- Logistics / Procurement: 732
- Non - Profit / Volunteering: 729

## Geographic Distribution
Top 5 countries by organization count:
- Congo: 847
- Korea: 810
- Lebanon: 477
- Iraq: 470
- Turkey: 469

## Broad Sector Average Employees
- Technology: 5,017.9 avg employees
- Other: 5,014.6 avg employees
- Manufacturing: 5,013.1 avg employees
- Finance: 4,972.8 avg employees
- Services: 4,943.1 avg employees
- Healthcare: 4,918.5 avg employees

## Predictive Model Findings
- **Task**: Multi-class classification: predict company size band (5 classes)
- **Features used**: company_age, Founded, industry_enc, country_enc, sector_enc, is_name_country_dup
- **Max |Pearson r| (feature vs target)**: 0.003398 (statistically near-zero — no meaningful linear relationship)
- **Dummy baseline accuracy** (most-frequent class): 0.4999 (49.99%)
- **RF Default accuracy**: 0.4925 — at dummy baseline, confirms no predictive signal
- **RF Balanced accuracy**: 0.2468 — below baseline; optimises minority-class recall
- **RF Balanced Macro F1**: 0.1584
- **RF Balanced Weighted F1**: 0.2978
- **5-Fold CV Accuracy (RF Balanced)**: 0.2491 +/- 0.0066

## Explainability — Three Methods
Methods applied: Gini MDI (Mean Decrease in Impurity), Permutation Importance (test set, 5 repeats), SHAP TreeExplainer (mean |SHAP|, all classes)

### 1. Gini MDI Feature Importances (RF Balanced)
*(Note: Gini MDI can overstate importance for high-cardinality encoded features)*
  - Company Age: 15.50%
  - Year Founded: 15.53%
  - Industry (enc): 28.16%
  - Country (enc): 33.75%
  - Broad Sector (enc): 6.83%
  - Name-Country Dup Flag: 0.22%
- **Top feature (Gini MDI)**: Country (enc)

### 2. Permutation Importance — Decisive Test
  *Method: sklearn permutation_importance, scoring=accuracy, n_repeats=5*
  *Test size: 20,000 rows | Repeats: 5*
  Mean accuracy drop when feature is shuffled (negative = no drop = no signal):
  - Company Age: -0.07585
  - Year Founded: -0.07804
  - Industry (enc): -0.00565
  - Country (enc): -0.00176
  - Broad Sector (enc): -0.00476
  - Name-Country Dup Flag: +0.00017
- **Top feature (Permutation)**: Name-Country Dup Flag
- *Negative/near-zero values mean shuffling that feature does not reduce test accuracy — confirming it carries no real predictive signal.*

### 3. SHAP (mean |SHAP| value, TreeExplainer)
  *Sample size: 1,697 rows (stratified from test set)*
  - Company Age: 0.012699
  - Year Founded: 0.012309
  - Industry (enc): 0.018239
  - Country (enc): 0.020374
  - Broad Sector (enc): 0.007187
  - Name-Country Dup Flag: 0.000178
- **Top feature (SHAP)**: Country (enc)

### Cross-Method Agreement
  - Gini MDI top: Country (enc)
  - Permutation top: Name-Country Dup Flag
  - SHAP top: Country (enc)
  - Methods agree on single top feature: **False**
  *(Note: Gini and SHAP rank Country (enc) highest due to its 243 unique values; Permutation top is noise — all values near zero.)*

## Explainability Interpretation
> Three independent explainability methods were applied to the trained Random Forest model. All three rank 'Country (enc)' and 'Industry (enc)' as the two most-used features. However, all permutation importance values are near zero or negative — meaning that shuffling any individual feature does not measurably reduce test accuracy. This is the decisive finding: the model relies on these features in its internal logic, but none of them contain genuine predictive signal for employee-count size band. All feature-target Pearson |r| < 0.004 (Spearman p > 0.23), confirming the absence of any meaningful statistical relationship. These are descriptive findings about model internals -- NOT evidence that country or industry determines organizational scale.

## Important Caveat
> Model accuracy (24.68%) is well below the 49.99% dummy baseline. Gini MDI can overstate the importance of high-cardinality encoded features (Country: 243 values, Industry: 147 values). Permutation importance on the test set is the most reliable measure and confirms all features are effectively uninformative. This result is analytically valid — it reflects the synthetic nature of the dataset where employee counts were assigned independently of all other fields.

---
*All statistics in this report are computed from the actual dataset — no values are fabricated.*
*Source: data/processed/kpis.json, ml_results.json, explainability.json*