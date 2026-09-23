"""
Tests for data preparation, EDA, SQL analytics, and ML pipeline.
Run with: pytest tests/ -v
"""

import pandas as pd
import numpy as np
import json
import os
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_preparation import load_raw, validate_schema, check_quality, clean_and_enrich


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    return load_raw()


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean_and_enrich(raw_df)


# ── Phase 2: Data Preparation ─────────────────────────────────────────────────

class TestDataPreparation:
    def test_raw_loads_100k_rows(self, raw_df):
        assert len(raw_df) == 100_000, f"Expected 100,000 rows, got {len(raw_df)}"

    def test_schema_valid(self, raw_df):
        result = validate_schema(raw_df)
        assert result["missing_columns"] == [], f"Missing columns: {result['missing_columns']}"

    def test_no_nulls_in_raw(self, raw_df):
        null_counts = raw_df.isnull().sum().to_dict()
        for col, n in null_counts.items():
            assert n == 0, f"Column '{col}' has {n} null values"

    def test_clean_adds_derived_columns(self, clean_df):
        for col in ["company_age", "founding_decade", "size_band", "broad_sector",
                    "is_name_country_dup"]:
            assert col in clean_df.columns, f"Missing derived column: {col}"

    def test_column_count(self, clean_df):
        assert len(clean_df.columns) == 14, \
            f"Expected 14 columns (9 raw + 5 derived), got {len(clean_df.columns)}"

    def test_company_age_positive(self, clean_df):
        assert (clean_df["company_age"] > 0).all(), "Some company ages are <= 0"

    def test_company_age_dynamic_reference_year(self, clean_df):
        """company_age must reflect the current year, not a hard-coded 2024."""
        import datetime
        ref_year = datetime.date.today().year
        expected_max_age = ref_year - 1970
        assert int(clean_df["company_age"].max()) == expected_max_age, \
            f"Max company_age {clean_df['company_age'].max()} does not match {ref_year}-1970={expected_max_age}"

    def test_no_nulls_in_key_columns(self, clean_df):
        key_cols = ["Organization Id", "Name", "Country", "Industry",
                    "Founded", "Number of employees", "company_age",
                    "founding_decade", "size_band", "broad_sector"]
        for col in key_cols:
            n = int(clean_df[col].isnull().sum())
            assert n == 0, f"Post-clean null in '{col}': {n}"

    def test_size_band_no_unknown(self, clean_df):
        valid_bands = {"Micro", "Small", "Medium", "Large", "Enterprise"}
        actual = set(clean_df["size_band"].dropna().unique())
        unknown = actual - valid_bands
        assert not unknown, f"Unknown size bands found: {unknown}"

    def test_size_band_sums_to_100k(self, clean_df):
        total = int(clean_df["size_band"].value_counts().sum())
        assert total == 100_000, f"size_band total {total} != 100,000"

    def test_broad_sector_categories(self, clean_df):
        valid = {"Technology", "Finance", "Healthcare", "Manufacturing", "Services", "Other"}
        actual = set(clean_df["broad_sector"].unique())
        unknown = actual - valid
        assert not unknown, f"Unknown broad sectors: {unknown}"

    def test_founding_decade_alignment(self, clean_df):
        """founding_decade must equal Founded // 10 * 10"""
        computed = (clean_df["Founded"] // 10 * 10).astype("Int64")
        assert (computed == clean_df["founding_decade"]).all()

    def test_size_band_consistent_with_employees(self, clean_df):
        micro = clean_df[clean_df["size_band"] == "Micro"]["Number of employees"]
        assert (micro < 50).all(), "Micro band contains employees >= 50"
        enterprise = clean_df[clean_df["size_band"] == "Enterprise"]["Number of employees"]
        assert (enterprise >= 5000).all(), "Enterprise band contains employees < 5000"

    def test_no_full_duplicate_rows(self, clean_df):
        assert clean_df.duplicated().sum() == 0, "Processed data contains fully duplicate rows"

    def test_org_id_unique(self, clean_df):
        assert clean_df["Organization Id"].duplicated().sum() == 0, \
            "Duplicate Organization Ids in processed data"

    def test_is_name_country_dup_binary(self, clean_df):
        assert clean_df["is_name_country_dup"].isin([0, 1]).all(), \
            "is_name_country_dup contains values other than 0 or 1"

    def test_is_name_country_dup_count(self, clean_df):
        flagged = int(clean_df["is_name_country_dup"].sum())
        assert flagged == 764, \
            f"Expected 764 flagged Name+Country duplicate rows, got {flagged}"

    def test_no_employees_zero_or_negative(self, clean_df):
        assert (clean_df["Number of employees"] >= 1).all(), \
            "Some employee counts are zero or negative"

    def test_no_employee_iqr_outliers(self, clean_df):
        emp = clean_df["Number of employees"].astype(float)
        q1, q3 = emp.quantile(0.25), emp.quantile(0.75)
        iqr = q3 - q1
        outliers = ((emp < q1 - 1.5 * iqr) | (emp > q3 + 1.5 * iqr)).sum()
        assert outliers == 0, f"IQR outliers found in Number of employees: {outliers}"

    def test_founded_in_valid_range(self, clean_df):
        assert (clean_df["Founded"] >= 1900).all(), "Some Founded years are before 1900"
        import datetime
        assert (clean_df["Founded"] <= datetime.date.today().year).all(), \
            "Some Founded years are in the future"

    def test_no_industry_whitespace(self, clean_df):
        issues = (clean_df["Industry"] != clean_df["Industry"].str.strip()).sum()
        assert issues == 0, f"Industry values with leading/trailing whitespace: {issues}"

    def test_no_country_whitespace(self, clean_df):
        issues = (clean_df["Country"] != clean_df["Country"].str.strip()).sum()
        assert issues == 0, f"Country values with leading/trailing whitespace: {issues}"

    def test_report_has_problems_found(self):
        with open("data/processed/preparation_report.json") as f:
            report = json.load(f)
        assert "problems_found" in report, "preparation_report.json missing 'problems_found' key"
        assert len(report["problems_found"]) > 0, "problems_found list is empty"

    def test_report_pipeline_run_date(self):
        with open("data/processed/preparation_report.json") as f:
            report = json.load(f)
        assert "pipeline_run_date" in report, "Report missing pipeline_run_date"

    def test_clean_row_count_preserved(self, raw_df, clean_df):
        assert len(clean_df) == len(raw_df), "Row count changed after cleaning"

    def test_processed_file_exists(self):
        assert os.path.exists("data/processed/organizations_clean.csv"), \
            "Processed CSV not found"

    def test_preparation_report_exists(self):
        assert os.path.exists("data/processed/preparation_report.json"), \
            "Preparation report not found"


# ── Phase 3: KPIs ─────────────────────────────────────────────────────────────

class TestKPIs:
    @pytest.fixture(scope="class")
    def kpis(self):
        with open("data/processed/kpis.json") as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def kpi_validation(self):
        with open("data/processed/kpis_validation.json") as f:
            return json.load(f)

    def test_kpi_file_exists(self):
        assert os.path.exists("data/processed/kpis.json")

    def test_kpi_validation_file_exists(self):
        assert os.path.exists("data/processed/kpis_validation.json")

    def test_kpi_validation_passed(self, kpi_validation):
        assert kpi_validation["validation_passed"], \
            f"KPI validation failed: {kpi_validation['failures']}"

    def test_kpi_validation_all_checks_ran(self, kpi_validation):
        assert kpi_validation["checks_run"] >= 15, \
            f"Expected >= 15 validation checks, got {kpi_validation['checks_run']}"

    def test_total_organizations(self, kpis):
        assert kpis["total_organizations"] == 100_000

    def test_industries_count(self, kpis):
        assert kpis["total_industries"] == 147

    def test_countries_count(self, kpis):
        assert kpis["total_countries"] == 243

    def test_size_band_pcts_sum_to_100(self, kpis):
        total = sum(kpis["size_band_pct"].values())
        assert abs(total - 100.0) < 0.2, f"Size band pcts sum to {total}, not 100"

    def test_size_band_counts_sum_to_100k(self, kpis):
        total = sum(kpis["size_band_counts"].values())
        assert total == 100_000, f"size_band_counts sum to {total}, not 100,000"

    def test_avg_employees_in_range(self, kpis):
        assert 4000 < kpis["avg_employees"] < 6000, \
            f"avg_employees {kpis['avg_employees']} out of expected range"

    def test_median_employees_positive(self, kpis):
        assert kpis["median_employees"] > 0

    def test_employee_skewness_near_zero(self, kpis):
        """Distribution is near-uniform — skewness should be close to 0."""
        assert abs(kpis["employees_skewness"]) < 0.5, \
            f"Employee skewness {kpis['employees_skewness']} unexpectedly large"

    def test_employee_kurtosis_negative(self, kpis):
        """Uniform distribution has negative kurtosis (platykurtic)."""
        assert kpis["employees_kurtosis"] < 0, \
            f"Employee kurtosis {kpis['employees_kurtosis']} should be negative for uniform data"

    def test_avg_company_age_positive(self, kpis):
        assert kpis["avg_company_age"] > 0

    def test_reference_year_current(self, kpis):
        import datetime
        assert kpis["reference_year"] == datetime.date.today().year, \
            "KPI reference_year does not match current year"

    def test_top_industries_has_10(self, kpis):
        assert len(kpis["top_10_industries"]) == 10

    def test_top_countries_has_10(self, kpis):
        assert len(kpis["top_10_countries"]) == 10

    def test_founding_decade_counts_sum(self, kpis):
        total = sum(kpis["founding_decade_counts"].values())
        assert total == 100_000, f"founding_decade_counts sum {total} != 100,000"

    def test_peak_founding_year_in_range(self, kpis):
        assert 1970 <= kpis["peak_founding_year"] <= 2022

    def test_corr_founded_employees_near_zero(self, kpis):
        """Uniform distribution — correlation should be near 0."""
        assert abs(kpis["corr_founded_vs_employees"]) < 0.1, \
            f"Correlation Founded-Employees {kpis['corr_founded_vs_employees']} unexpectedly large"

    def test_industry_hhi_low(self, kpis):
        """147 near-uniform industries — HHI should be near 1/147 = 0.0068."""
        assert kpis["industry_hhi"] < 0.02, \
            f"Industry HHI {kpis['industry_hhi']} too high for near-uniform distribution"

    def test_sector_hhi_present(self, kpis):
        assert "sector_hhi" in kpis

    def test_enterprise_pct_by_sector_present(self, kpis):
        assert "enterprise_pct_by_sector" in kpis
        assert len(kpis["enterprise_pct_by_sector"]) > 0

    def test_avg_employees_by_sector_all_sectors(self, kpis):
        expected_sectors = {"Technology", "Finance", "Healthcare", "Manufacturing",
                            "Services", "Other"}
        actual = set(kpis["avg_employees_by_sector"].keys())
        assert actual == expected_sectors, f"Sectors mismatch: {actual}"

    def test_country_concentration_low(self, kpis):
        """No country should dominate — top 10 should be well under 20%."""
        assert kpis["top10_country_pct"] < 20, \
            f"Top 10 countries = {kpis['top10_country_pct']}% — unexpectedly concentrated"

    def test_name_country_dup_count(self, kpis):
        assert kpis["name_country_dup_count"] == 764

    def test_charts_generated(self):
        expected_charts = [
            "fig_size_distribution.png",
            "fig_top_industries.png",
            "fig_top_countries.png",
            "fig_founding_trend.png",
            "fig_employee_distribution.png",
            "fig_sector_breakdown.png",
            "fig_avg_employees_by_sector.png",
            "fig_decade_vs_employees.png",
            "fig_company_age_distribution.png",
            "fig_industry_avg_employees.png",
            "fig_sector_size_heatmap.png",
            "fig_country_org_distribution.png",
            "fig_founding_decade_bar.png",
        ]
        for chart in expected_charts:
            assert os.path.exists(f"reports/{chart}"), f"Chart not found: {chart}"

    def test_chart_files_not_empty(self):
        for chart in [
            "fig_size_distribution.png", "fig_sector_size_heatmap.png",
            "fig_industry_avg_employees.png", "fig_company_age_distribution.png",
        ]:
            size = os.path.getsize(f"reports/{chart}")
            assert size > 10_000, f"Chart {chart} is suspiciously small ({size} bytes)"


# ── Phase 4: SQL Analytics ─────────────────────────────────────────────────────

class TestSQLAnalytics:
    @pytest.fixture(scope="class")
    def sql(self):
        with open("data/processed/sql_analytics.json") as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def sql_validation(self):
        with open("data/processed/sql_analytics_validation.json") as f:
            return json.load(f)

    def test_sql_file_exists(self):
        assert os.path.exists("data/processed/sql_analytics.json")

    def test_sql_validation_file_exists(self):
        assert os.path.exists("data/processed/sql_analytics_validation.json")

    def test_sql_validation_passed(self, sql_validation):
        assert sql_validation["validation_passed"], \
            f"SQL validation failed: {sql_validation['failures']}"

    def test_sql_validation_checks_ran(self, sql_validation):
        assert sql_validation["checks_run"] >= 32, \
            f"Expected >= 32 checks, got {sql_validation['checks_run']}"

    def test_all_queries_present(self, sql):
        expected_keys = [
            "Q01_dataset_totals",
            "Q02_top_industries_by_count",
            "Q03_top_countries_by_count",
            "Q04_sector_overview",
            "Q05_size_band_distribution",
            "Q06_founding_decade_trends",
            "Q07_top_industry_per_top_country",
            "Q08_top10_industries_by_avg_employees",
            "Q08_bottom10_industries_by_avg_employees",
            "Q09_founding_trend_by_year",
            "Q10_sector_size_crosstab",
            "Q11_most_diverse_countries",
            "Q12_employee_percentile_distribution",
            "Q13_top10_countries_by_avg_employees",
            "Q13_bottom10_countries_by_avg_employees",
            "Q14_year_on_year_founding_change",
            "Q15_name_country_dup_flag_summary",
            "Q16_industry_spread_uniformity",
            "Q17_peak_and_trough_founding_years",
            "Q18_sector_enterprise_rate",
        ]
        for key in expected_keys:
            assert key in sql, f"Missing query result: {key}"

    def test_q01_totals(self, sql):
        r = sql["Q01_dataset_totals"][0]
        assert r["total_rows"] == 100_000
        assert r["unique_org_ids"] == 100_000
        assert r["unique_industries"] == 147
        assert r["unique_countries"] == 243
        assert r["null_employees"] == 0.0
        assert r["null_industry"] == 0.0
        assert r["null_country"] == 0.0

    def test_q02_top_industry(self, sql):
        top = sql["Q02_top_industries_by_count"][0]
        assert top["Industry"] == "Insurance"
        assert top["org_count"] == 747

    def test_q02_returns_20_rows(self, sql):
        assert len(sql["Q02_top_industries_by_count"]) == 20

    def test_q03_top_country(self, sql):
        top = sql["Q03_top_countries_by_count"][0]
        assert top["Country"] == "Congo"
        assert top["org_count"] == 847

    def test_q04_sector_pcts_sum_to_100(self, sql):
        total = sum(r["pct_of_total"] for r in sql["Q04_sector_overview"])
        assert abs(total - 100.0) < 0.15, f"Sector pcts sum to {total}"

    def test_q04_sector_total_rows(self, sql):
        total = sum(r["org_count"] for r in sql["Q04_sector_overview"])
        assert total == 100_000

    def test_q05_size_bands_all_present(self, sql):
        bands = {r["size_band"] for r in sql["Q05_size_band_distribution"]}
        for band in ["Micro", "Small", "Medium", "Large", "Enterprise"]:
            assert band in bands, f"Missing size band: {band}"

    def test_q05_size_band_total(self, sql):
        total = sum(r["org_count"] for r in sql["Q05_size_band_distribution"])
        assert total == 100_000

    def test_q05_micro_count(self, sql):
        micro = next(r for r in sql["Q05_size_band_distribution"] if r["size_band"] == "Micro")
        assert micro["org_count"] == 492

    def test_q05_enterprise_count(self, sql):
        ent = next(r for r in sql["Q05_size_band_distribution"] if r["size_band"] == "Enterprise")
        assert ent["org_count"] == 49990

    def test_q06_decade_total(self, sql):
        total = sum(r["orgs_founded"] for r in sql["Q06_founding_decade_trends"])
        assert total == 100_000

    def test_q06_decade_1970s(self, sql):
        dec = next(r for r in sql["Q06_founding_decade_trends"] if int(r["decade"]) == 1970)
        assert dec["orgs_founded"] == 18863

    def test_q09_all_years_present(self, sql):
        years = {r["year"] for r in sql["Q09_founding_trend_by_year"]}
        for yr in range(1970, 2023):
            assert yr in years, f"Year {yr} missing from founding trend"

    def test_q09_year_total(self, sql):
        total = sum(r["orgs_founded"] for r in sql["Q09_founding_trend_by_year"])
        assert total == 100_000

    def test_q09_peak_year(self, sql):
        peak = max(sql["Q09_founding_trend_by_year"], key=lambda r: r["orgs_founded"])
        assert peak["year"] == 1998
        assert peak["orgs_founded"] == 2022

    def test_q10_crosstab_total(self, sql):
        total = sum(r["org_count"] for r in sql["Q10_sector_size_crosstab"])
        assert total == 100_000

    def test_q12_percentiles(self, sql):
        p = sql["Q12_employee_percentile_distribution"][0]
        assert p["p50"] == 4998.0
        assert p["p25"] == 2508.0
        assert p["p75"] == 7503.0

    def test_q15_dup_flag_count(self, sql):
        dup_row = next((r for r in sql["Q15_name_country_dup_flag_summary"]
                        if int(r["dup_flag"]) == 1), None)
        assert dup_row is not None, "dup_flag=1 row missing"
        assert dup_row["org_count"] == 764

    def test_q16_industry_uniformity(self, sql):
        s = sql["Q16_industry_spread_uniformity"][0]
        assert s["total_industries"] == 147
        assert s["min_orgs_per_industry"] == 626
        assert s["max_orgs_per_industry"] == 747
        # CV < 10% confirms near-uniform distribution
        assert s["coeff_of_variation_pct"] < 10.0, \
            f"CV={s['coeff_of_variation_pct']}% — distribution unexpectedly non-uniform"

    def test_q17_peak_year_is_1998(self, sql):
        top5 = [r for r in sql["Q17_peak_and_trough_founding_years"]
                if r["category"] == "top_5"]
        assert any(r["year"] == 1998 for r in top5), "1998 not in peak founding years"

    def test_q17_trough_includes_2022(self, sql):
        bot5 = [r for r in sql["Q17_peak_and_trough_founding_years"]
                if r["category"] == "bottom_5"]
        assert any(r["year"] == 2022 for r in bot5), \
            "2022 should be in trough years (partial year)"

    def test_q18_sector_enterprise_rates_present(self, sql):
        sectors = {r["broad_sector"] for r in sql["Q18_sector_enterprise_rate"]}
        for s in ["Technology", "Finance", "Healthcare", "Manufacturing", "Services", "Other"]:
            assert s in sectors, f"Sector {s} missing from Q18"


# ── Phase 5: ML Model ─────────────────────────────────────────────────────────

class TestMLModel:
    @pytest.fixture(scope="class")
    def ml(self):
        with open("data/processed/ml_results.json") as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def ml_val(self):
        with open("data/processed/ml_validation.json") as f:
            return json.load(f)

    def test_ml_file_exists(self):
        assert os.path.exists("data/processed/ml_results.json")

    def test_ml_validation_file_exists(self):
        assert os.path.exists("data/processed/ml_validation.json")

    def test_ml_validation_passed(self, ml_val):
        assert ml_val["validation_passed"], \
            f"ML validation failed: {ml_val['failures']}"

    def test_model_files_exist(self):
        assert os.path.exists("data/processed/rf_model.joblib")
        assert os.path.exists("data/processed/scaler.joblib")
        assert os.path.exists("data/processed/encoders.joblib")

    def test_new_model_default_file_exists(self):
        assert os.path.exists("data/processed/rf_model_default.joblib")

    def test_dummy_baseline_near_50pct(self, ml):
        acc = ml["dummy_baseline"]["accuracy"]
        assert abs(acc - 0.4999) < 0.01, \
            f"Dummy accuracy {acc} not near expected ~0.4999"

    def test_rf_default_accuracy_at_baseline(self, ml):
        """RF default has no predictive signal — accuracy must be near dummy baseline."""
        rfd = ml["random_forest_default"]["accuracy"]
        dum = ml["dummy_baseline"]["accuracy"]
        assert abs(rfd - dum) < 0.05, \
            f"RF default acc={rfd:.4f} too far from dummy baseline {dum:.4f}"

    def test_rf_balanced_accuracy_positive(self, ml):
        acc = ml["random_forest_balanced"]["accuracy"]
        assert acc > 0.0, f"RF balanced accuracy {acc} must be positive"

    def test_rf_balanced_better_macro_f1_than_dummy(self, ml):
        """Balanced RF should have better macro F1 than dummy (dummy collapses to one class)."""
        rf_f1 = ml["random_forest_balanced"]["macro_f1"]
        dum_f1 = ml["dummy_baseline"]["macro_f1"]
        assert rf_f1 >= dum_f1, \
            f"RF balanced macro_f1={rf_f1:.4f} should be >= dummy {dum_f1:.4f}"

    def test_feature_importances_sum_to_1(self, ml):
        total = sum(ml["random_forest_balanced"]["feature_importances"].values())
        assert abs(total - 1.0) < 0.01, f"Feature importances sum to {total}, not 1.0"

    def test_6_features_present(self, ml):
        assert len(ml["feature_names"]) == 6, \
            f"Expected 6 features, got {len(ml['feature_names'])}"

    def test_no_employees_in_features(self, ml):
        assert "Number of employees" not in ml["feature_names"], \
            "Number of employees must not appear in features (leakage)"

    def test_target_classes_correct(self, ml):
        assert ml["class_labels"] == ["Micro", "Small", "Medium", "Large", "Enterprise"]

    def test_train_test_sum_100k(self, ml):
        assert ml["train_size"] + ml["test_size"] == 100_000

    def test_cv_std_reasonable(self, ml):
        std = ml["cross_validation"]["rf_default_cv_accuracy_std"]
        assert std < 0.05, f"CV std {std} too high"

    def test_feature_target_correlations_present(self, ml):
        assert "feature_target_correlations" in ml
        assert len(ml["feature_target_correlations"]) == 6

    def test_all_correlations_near_zero(self, ml):
        for feat, stats in ml["feature_target_correlations"].items():
            assert abs(stats["pearson_r"]) < 0.02, \
                f"{feat}: Pearson r={stats['pearson_r']:.6f} unexpectedly large"

    def test_roc_auc_near_random(self, ml):
        auc = ml["random_forest_default"]["roc_auc_ovr_macro"]
        assert auc is not None
        assert 0.48 <= auc <= 0.52, \
            f"ROC-AUC={auc:.4f} should be near 0.5 (random) given no predictive signal"

    def test_confusion_matrix_plot_exists(self):
        assert os.path.exists("reports/fig_confusion_matrix.png")

    def test_confusion_matrix_balanced_plot_exists(self):
        assert os.path.exists("reports/fig_confusion_matrix_balanced.png")

    def test_feature_importance_plot_exists(self):
        assert os.path.exists("reports/fig_feature_importance.png")

    def test_model_comparison_plot_exists(self):
        assert os.path.exists("reports/fig_model_comparison.png")

    def test_class_distribution_plot_exists(self):
        assert os.path.exists("reports/fig_ml_class_distribution.png")

    # Backward-compat keys still present for downstream phases
    def test_backward_compat_random_forest_key(self, ml):
        assert "random_forest" in ml
        assert "accuracy" in ml["random_forest"]
        assert "feature_importances" in ml["random_forest"]

    def test_backward_compat_logistic_regression_key(self, ml):
        assert "logistic_regression" in ml


# ── Phase 6: Explainability ───────────────────────────────────────────────────

class TestExplainability:
    @pytest.fixture(scope="class")
    def exp(self):
        with open("data/processed/explainability.json") as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def exp_val(self):
        with open("data/processed/explainability_validation.json") as f:
            return json.load(f)

    def test_explainability_file_exists(self):
        assert os.path.exists("data/processed/explainability.json")

    def test_explainability_validation_file_exists(self):
        assert os.path.exists("data/processed/explainability_validation.json")

    def test_explainability_validation_passed(self, exp_val):
        assert exp_val["validation_passed"], \
            f"Explainability validation failed: {exp_val['failures']}"

    def test_shap_has_all_6_features(self, exp):
        expected = {
            "Company Age", "Year Founded", "Industry (enc)",
            "Country (enc)", "Broad Sector (enc)", "Name-Country Dup Flag",
        }
        actual = set(exp["shap_mean_abs"].keys())
        assert actual == expected, f"SHAP keys mismatch: {actual}"

    def test_gini_has_all_6_features(self, exp):
        expected = {
            "Company Age", "Year Founded", "Industry (enc)",
            "Country (enc)", "Broad Sector (enc)", "Name-Country Dup Flag",
        }
        actual = set(exp["feature_importances_gini"].keys())
        assert actual == expected, f"Gini importance keys mismatch: {actual}"

    def test_gini_importances_sum_to_1(self, exp):
        total = sum(exp["feature_importances_gini"].values())
        assert abs(total - 1.0) < 0.01, f"Gini importances sum={total:.4f}"

    def test_shap_values_non_negative(self, exp):
        for feat, val in exp["shap_mean_abs"].items():
            assert val >= 0, f"SHAP value for {feat} is negative: {val}"

    def test_shap_values_near_zero(self, exp):
        """SHAP values must be small given near-zero predictive signal."""
        for feat, val in exp["shap_mean_abs"].items():
            assert val < 0.5, f"SHAP value for {feat} = {val:.4f} unexpectedly large"

    def test_top_features_identified(self, exp):
        assert "top_feature_by_importance" in exp
        assert "top_feature_by_shap" in exp
        assert exp["top_feature_by_importance"] in exp["feature_importances_gini"]
        assert exp["top_feature_by_shap"] in exp["shap_mean_abs"]

    def test_interpretation_present(self, exp):
        assert "interpretation" in exp
        assert len(exp["interpretation"]) > 50

    def test_caveat_present(self, exp):
        assert "important_caveat" in exp

    def test_sample_size_recorded(self, exp):
        assert exp.get("sample_size_for_shap", 0) > 0

    def test_three_methods_present(self, exp):
        assert "methods_used" in exp
        assert len(exp["methods_used"]) == 3

    def test_permutation_importance_present(self, exp):
        assert "permutation_importance" in exp
        perm = exp["permutation_importance"]
        assert "mean_accuracy_drop" in perm
        assert "std_accuracy_drop" in perm
        assert perm.get("n_repeats", 0) >= 3
        assert perm.get("test_size", 0) > 0

    def test_permutation_has_all_6_features(self, exp):
        expected = {
            "Company Age", "Year Founded", "Industry (enc)",
            "Country (enc)", "Broad Sector (enc)", "Name-Country Dup Flag",
        }
        actual = set(exp["permutation_importance"]["mean_accuracy_drop"].keys())
        assert actual == expected, f"Permutation keys mismatch: {actual}"

    def test_permutation_no_large_positive_value(self, exp):
        """No feature should have a large positive accuracy drop when permuted.
        Negative values (shuffling improves accuracy) also confirm no real signal."""
        for feat, val in exp["permutation_importance"]["mean_accuracy_drop"].items():
            assert val < 0.05, \
                f"Perm importance for {feat} = {val:.5f} — unexpectedly large positive signal"

    def test_cross_method_agreement_present(self, exp):
        cm = exp.get("cross_method_top_feature", {})
        assert "gini_mdi" in cm
        assert "permutation" in cm
        assert "shap" in cm

    def test_shap_plots_exist(self):
        assert os.path.exists("reports/fig_shap_importance.png")
        assert os.path.exists("reports/fig_shap_enterprise.png")

    def test_feature_importance_clean_plot_exists(self):
        assert os.path.exists("reports/fig_feature_importance_clean.png")

    def test_permutation_importance_plot_exists(self):
        assert os.path.exists("reports/fig_permutation_importance.png")

    def test_method_comparison_plot_exists(self):
        assert os.path.exists("reports/fig_explainability_comparison.png")


# ── Phase 7: Dashboard ─────────────────────────────────────────────────────────

class TestDashboard:
    def test_dashboard_file_exists(self):
        assert os.path.exists("dashboard/index.html")

    def test_dashboard_not_empty(self):
        size = os.path.getsize("dashboard/index.html")
        assert size > 20_000, f"Dashboard file too small ({size} bytes)"

    def test_dashboard_has_plotly(self):
        with open("dashboard/index.html", encoding="utf-8") as f:
            content = f.read()
        assert "plotly" in content.lower()

    def test_dashboard_has_real_kpi_values(self):
        with open("dashboard/index.html", encoding="utf-8") as f:
            content = f.read()
        # Real values from the dataset should appear
        assert "100,000" in content
        assert "147" in content
        assert "243" in content


# ── Phase 8: AI Insights ──────────────────────────────────────────────────────

class TestAIInsights:
    # ── Output file ───────────────────────────────────────────────────────────
    def test_ai_insights_file_exists(self):
        assert os.path.exists("reports/ai_insights.md")

    def test_ai_insights_not_empty(self):
        size = os.path.getsize("reports/ai_insights.md")
        assert size > 500, f"AI insights file too small ({size} bytes)"

    def test_ai_insights_contains_key_facts(self):
        with open("reports/ai_insights.md", encoding="utf-8") as f:
            content = f.read()
        assert "100,000" in content or "100000" in content

    def test_ai_insights_references_validated_industry_count(self):
        """Report must reference the actual industry count (147) from kpis.json."""
        with open("reports/ai_insights.md", encoding="utf-8") as f:
            content = f.read()
        assert "147" in content, "Industry count 147 missing from report"

    def test_ai_insights_references_validated_country_count(self):
        """Report must reference the actual country count (243) from kpis.json."""
        with open("reports/ai_insights.md", encoding="utf-8") as f:
            content = f.read()
        assert "243" in content, "Country count 243 missing from report"

    def test_ai_insights_references_permutation_importance(self):
        """Report must include the permutation importance section (Phase 6 decisive test)."""
        with open("reports/ai_insights.md", encoding="utf-8") as f:
            content = f.read()
        assert "Permutation" in content, "Permutation importance section missing from report"

    def test_ai_insights_references_three_methods(self):
        """All three explainability method names must appear in the report."""
        with open("reports/ai_insights.md", encoding="utf-8") as f:
            content = f.read()
        assert "Gini" in content, "Gini MDI missing from report"
        assert "SHAP" in content, "SHAP missing from report"
        assert "Permutation" in content, "Permutation importance missing from report"

    def test_ai_insights_mentions_dummy_baseline(self):
        """Report must include the dummy-baseline accuracy result."""
        with open("reports/ai_insights.md", encoding="utf-8") as f:
            content = f.read()
        assert "49.99" in content or "49.9" in content or "Dummy" in content or "dummy" in content

    def test_ai_insights_causation_disclaimer(self):
        """Report must contain a causation / correlation disclaimer."""
        with open("reports/ai_insights.md", encoding="utf-8") as f:
            content = f.read()
        has_disclaimer = (
            "NOT" in content
            or "not causal" in content.lower()
            or "causal" in content.lower()
            or "correlation" in content.lower()
        )
        assert has_disclaimer, "No causation disclaimer found in report"

    # ── Context loading — actual data, no fabrication ─────────────────────────
    def test_load_validated_context_uses_actual_kpis(self):
        """load_validated_context() must pull total_organizations from kpis.json."""
        import json
        from src.ai_insights import load_validated_context
        with open("data/processed/kpis.json") as f:
            kpis = json.load(f)
        ctx = load_validated_context()
        assert ctx["dataset"]["total_organizations"] == kpis["total_organizations"]

    def test_load_validated_context_ml_accuracy_matches_ml_results(self):
        """RF balanced accuracy in context must exactly match ml_results.json."""
        import json
        from src.ai_insights import load_validated_context
        with open("data/processed/ml_results.json") as f:
            ml = json.load(f)
        ctx = load_validated_context()
        assert round(ctx["ml_model"]["rf_balanced_accuracy"], 4) == round(
            ml["random_forest_balanced"]["accuracy"], 4
        )

    def test_load_validated_context_includes_permutation_importance(self):
        """Context dict must contain the permutation_importance block from explainability.json."""
        from src.ai_insights import load_validated_context
        ctx = load_validated_context()
        assert "permutation_importance" in ctx["explainability"]
        perm = ctx["explainability"]["permutation_importance"]
        assert "mean_accuracy_drop" in perm
        assert len(perm["mean_accuracy_drop"]) == 6

    def test_load_validated_context_includes_cross_method_agreement(self):
        """Context must include the cross-method top-feature comparison."""
        from src.ai_insights import load_validated_context
        ctx = load_validated_context()
        assert "cross_method_top_feature" in ctx["explainability"]
        cm = ctx["explainability"]["cross_method_top_feature"]
        assert "gini_mdi" in cm
        assert "permutation" in cm
        assert "shap" in cm
        assert "agreement" in cm

    def test_load_validated_context_three_methods_listed(self):
        """Context must list all three explainability method names."""
        from src.ai_insights import load_validated_context
        ctx = load_validated_context()
        methods = ctx["explainability"]["methods_used"]
        assert len(methods) == 3
        joined = " ".join(methods).lower()
        assert "gini" in joined
        assert "permutation" in joined
        assert "shap" in joined

    def test_load_validated_context_no_employee_count_in_features(self):
        """Employee count must NOT appear in feature list (data leakage guard)."""
        from src.ai_insights import load_validated_context
        ctx = load_validated_context()
        features_lower = [f.lower() for f in ctx["ml_model"]["features_used"]]
        leakage_terms = ["number of employees", "employees", "num_employees"]
        for term in leakage_terms:
            assert term not in features_lower, f"Leakage feature '{term}' found in context"

    # ── Security — no hard-coded secrets ─────────────────────────────────────
    def test_no_hardcoded_api_key_in_source(self):
        """ai_insights.py must never contain a hard-coded API key pattern."""
        with open("src/ai_insights.py", encoding="utf-8") as f:
            source = f.read()
        import re
        # Matches sk-... style tokens or Bearer <long-token> literals
        assert not re.search(r'sk-[A-Za-z0-9]{20,}', source), \
            "Hard-coded OpenAI-style API key found in source"
        assert "AI_API_KEY" in source, \
            "AI_API_KEY env-var reference missing from source"

    def test_env_example_exists(self):
        """.env.example must exist so users know which env vars to set."""
        assert os.path.exists(".env.example"), ".env.example missing from project root"

    def test_env_example_documents_ai_api_key(self):
        """AI_API_KEY must be documented in .env.example."""
        with open(".env.example", encoding="utf-8") as f:
            content = f.read()
        assert "AI_API_KEY" in content

    def test_gitignore_excludes_dotenv(self):
        """.gitignore must protect .env from being committed."""
        assert os.path.exists(".gitignore"), ".gitignore missing from project root"
        with open(".gitignore", encoding="utf-8") as f:
            content = f.read()
        assert ".env" in content, ".env not excluded in .gitignore"

    # ── Fallback behaviour ────────────────────────────────────────────────────
    def test_fallback_runs_without_endpoint(self):
        """run() with empty endpoint must produce a valid report without raising."""
        import tempfile
        from src.ai_insights import run
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = run(
                endpoint="",
                model="llama3",
                api_key="",
                output_path=tmp_path,
                use_openai=False,
            )
            assert result is not None
            assert len(result) > 200
            assert "100,000" in result or "100000" in result
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_fallback_output_contains_permutation_section(self):
        """Structured fallback must include the permutation importance section."""
        from src.ai_insights import load_validated_context, generate_structured_fallback
        ctx = load_validated_context()
        report = generate_structured_fallback(ctx)
        assert "Permutation" in report
        assert "mean_accuracy_drop" not in report  # internal key must not leak as-is
        assert "-0.0" in report or "+0.0" in report  # actual drop values rendered

    def test_fallback_output_contains_shap_section(self):
        """Structured fallback must include the SHAP section."""
        from src.ai_insights import load_validated_context, generate_structured_fallback
        ctx = load_validated_context()
        report = generate_structured_fallback(ctx)
        assert "SHAP" in report

    def test_fallback_output_contains_cross_method_section(self):
        """Structured fallback must include the cross-method agreement section."""
        from src.ai_insights import load_validated_context, generate_structured_fallback
        ctx = load_validated_context()
        report = generate_structured_fallback(ctx)
        assert "Cross-Method" in report or "cross_method" in report.lower()
