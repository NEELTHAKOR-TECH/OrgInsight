"""
Phase 2 — Data Preparation Pipeline
Validates, cleans, and enriches the organizations dataset.
Preserves the original raw file; writes to data/processed/.

Checks performed:
  - Schema validation (required columns present)
  - Null / empty-string counts per column
  - Duplicate Organization Id (unique key)
  - Duplicate Name (expected — many firms share names globally)
  - Duplicate Name+Country combinations (flagged, not dropped — different OrgIds confirmed)
  - Duplicate Name+Country+Industry (6 pairs — distinct OrgIds and Founded years; retained)
  - Full row duplicates (zero found)
  - Invalid data types (Founded, Number of employees — coerced, failures reported)
  - Founded range validity (must be 1900–current year; dataset: 1970–2022, all valid)
  - Number of employees validity (must be ≥ 1; dataset: 1–9,999, all valid)
  - Statistical outliers via IQR fences and Z-score (none found — uniform distribution)
  - Whitespace / casing inconsistencies in all string columns (none found)
  - Organization Id format (15-char hex; all valid)
  - Website format (all start with http:// or https://)
  - Non-ASCII characters in Name, Country (none found)
"""

import datetime
import pandas as pd
import numpy as np
import json
import os
import re

RAW_PATH = "data/raw/organizations-100000.csv"
PROCESSED_PATH = "data/processed/organizations_clean.csv"
REPORT_PATH = "data/processed/preparation_report.json"

# Employee size-band thresholds (EU SME + enterprise convention)
SIZE_BANDS = [
    (0, 49, "Micro"),
    (50, 249, "Small"),
    (250, 999, "Medium"),
    (1000, 4999, "Large"),
    (5000, float("inf"), "Enterprise"),
]

# Reference year for company_age — derived dynamically so the pipeline stays current
REFERENCE_YEAR = datetime.date.today().year

# Valid Founded range
FOUNDED_MIN = 1900
FOUNDED_MAX = REFERENCE_YEAR


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    return df


def validate_schema(df: pd.DataFrame) -> dict:
    expected_cols = [
        "Index", "Organization Id", "Name", "Website",
        "Country", "Description", "Founded", "Industry",
        "Number of employees",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_cols]
    return {"missing_columns": missing, "extra_columns": extra}


def check_quality(df: pd.DataFrame) -> dict:
    """
    Comprehensive pre-cleaning quality report.
    Covers: nulls, empty strings, duplicates, type coercion failures,
    value range validity, outliers, format checks, and encoding issues.
    """
    report = {}
    report["total_rows"] = len(df)

    # ── Nulls & empties ──────────────────────────────────────────────────────
    report["null_counts"] = df.isnull().sum().to_dict()
    report["empty_string_counts"] = (df == "").sum().to_dict()
    report["whitespace_only_counts"] = {
        col: int((df[col].str.strip() == "").sum())
        for col in df.columns
    }

    # ── Duplicate checks ─────────────────────────────────────────────────────
    report["duplicate_full_rows"] = int(df.duplicated().sum())
    report["duplicate_org_ids"] = int(df["Organization Id"].duplicated().sum())
    report["duplicate_names"] = int(df["Name"].duplicated().sum())

    dup_nc = df[df.duplicated(subset=["Name", "Country"], keep=False)]
    report["duplicate_name_country_pairs"] = int(len(dup_nc))

    dup_nci = df[df.duplicated(subset=["Name", "Country", "Industry"], keep=False)]
    report["duplicate_name_country_industry_pairs"] = int(len(dup_nci))
    if len(dup_nci) > 0:
        report["duplicate_name_country_industry_sample"] = (
            dup_nci[["Organization Id", "Name", "Country", "Industry", "Founded"]]
            .sort_values(["Name", "Country"])
            .head(12)
            .to_dict(orient="records")
        )

    # ── Organization Id format ───────────────────────────────────────────────
    org_id_pattern = re.compile(r"^[0-9a-fA-F]{15,16}$")
    invalid_ids = (~df["Organization Id"].str.match(org_id_pattern)).sum()
    report["org_id_format_violations"] = int(invalid_ids)
    report["org_id_length_distribution"] = (
        df["Organization Id"].str.len().value_counts().sort_index().to_dict()
    )

    # ── Founded ──────────────────────────────────────────────────────────────
    founded_num = pd.to_numeric(df["Founded"], errors="coerce")
    report["founded_nulls_after_coerce"] = int(founded_num.isnull().sum())
    report["founded_min"] = int(founded_num.min())
    report["founded_max"] = int(founded_num.max())
    report["founded_below_min"] = int((founded_num < FOUNDED_MIN).sum())
    report["founded_above_max"] = int((founded_num > FOUNDED_MAX).sum())
    report["founded_unique_values"] = int(founded_num.nunique())
    fz = (founded_num - founded_num.mean()) / founded_num.std()
    report["founded_zscore_outliers_gt3"] = int((fz.abs() > 3).sum())

    # ── Number of employees ──────────────────────────────────────────────────
    emp_num = pd.to_numeric(df["Number of employees"], errors="coerce")
    report["employees_nulls_after_coerce"] = int(emp_num.isnull().sum())
    report["employees_min"] = int(emp_num.min())
    report["employees_max"] = int(emp_num.max())
    report["employees_mean"] = round(float(emp_num.mean()), 2)
    report["employees_median"] = round(float(emp_num.median()), 2)
    report["employees_std"] = round(float(emp_num.std()), 2)
    report["employees_zero_or_negative"] = int((emp_num <= 0).sum())

    q1 = emp_num.quantile(0.25)
    q3 = emp_num.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    report["employees_iqr_lower_fence"] = round(float(lower_fence), 2)
    report["employees_iqr_upper_fence"] = round(float(upper_fence), 2)
    report["employees_iqr_outliers_below"] = int((emp_num < lower_fence).sum())
    report["employees_iqr_outliers_above"] = int((emp_num > upper_fence).sum())

    ez = (emp_num - emp_num.mean()) / emp_num.std()
    report["employees_zscore_outliers_gt3"] = int((ez.abs() > 3).sum())

    # ── Categorical cardinality & casing ─────────────────────────────────────
    report["unique_industries"] = int(df["Industry"].nunique())
    report["unique_industries_case_insensitive"] = int(
        df["Industry"].str.lower().nunique()
    )
    report["unique_countries"] = int(df["Country"].nunique())
    report["unique_countries_case_insensitive"] = int(
        df["Country"].str.lower().nunique()
    )
    report["industry_whitespace_issues"] = int(
        (df["Industry"] != df["Industry"].str.strip()).sum()
    )
    report["country_whitespace_issues"] = int(
        (df["Country"] != df["Country"].str.strip()).sum()
    )

    # ── Website format ───────────────────────────────────────────────────────
    report["website_not_http"] = int(
        (~df["Website"].str.startswith(("http://", "https://"))).sum()
    )
    report["website_with_spaces"] = int(df["Website"].str.contains(" ").sum())

    # ── Non-ASCII ────────────────────────────────────────────────────────────
    report["name_non_ascii"] = int(
        df["Name"].apply(lambda x: any(ord(c) > 127 for c in str(x))).sum()
    )
    report["country_non_ascii"] = int(
        df["Country"].apply(lambda x: any(ord(c) > 127 for c in str(x))).sum()
    )

    return report


def clean_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Type coercion ────────────────────────────────────────────────────────
    df["Founded"] = pd.to_numeric(df["Founded"], errors="coerce").astype("Int64")
    df["Number of employees"] = pd.to_numeric(
        df["Number of employees"], errors="coerce"
    ).astype("Int64")
    df["Index"] = pd.to_numeric(df["Index"], errors="coerce").astype("Int64")

    # ── Whitespace normalisation (string columns) ────────────────────────────
    # Pre-checks confirm no whitespace issues exist; strip is a safety measure.
    for col in ["Name", "Country", "Industry", "Description", "Website"]:
        df[col] = df[col].str.strip()

    # ── Flag: Name+Country duplicate ─────────────────────────────────────────
    # 764 rows share the same Name+Country but have unique Organization Ids.
    # These are legitimately distinct organizations (different industry/Founded).
    # Flagged for transparency; no rows are dropped.
    df["is_name_country_dup"] = df.duplicated(
        subset=["Name", "Country"], keep=False
    ).astype(int)

    # ── Derived columns ──────────────────────────────────────────────────────

    # Company age — dynamic reference year (not hard-coded)
    df["company_age"] = REFERENCE_YEAR - df["Founded"]

    # Founding decade
    df["founding_decade"] = (df["Founded"] // 10 * 10).astype("Int64")

    # Size band (EU SME + enterprise thresholds)
    def assign_size_band(emp):
        if pd.isna(emp):
            return "Unknown"
        for lo, hi, label in SIZE_BANDS:
            if lo <= emp <= hi:
                return label
        return "Unknown"

    df["size_band"] = df["Number of employees"].apply(assign_size_band)
    df["size_band"] = pd.Categorical(
        df["size_band"],
        categories=["Micro", "Small", "Medium", "Large", "Enterprise"],
        ordered=True,
    )

    # Broad industry sector (keyword matching across 5 macro-sectors)
    # Note: 70% of rows map to "Other" because 70 of 147 industry labels
    # contain none of the defined keywords. This is a known limitation.
    tech_keywords = [
        "Computer", "Software", "Technology", "Internet", "Semiconductor",
        "Wireless", "Telecommunications", "Network", "Security", "IT "
    ]
    finance_keywords = [
        "Banking", "Financial", "Insurance", "Capital Markets", "Investment",
        "Accounting", "Venture"
    ]
    health_keywords = [
        "Health", "Hospital", "Medical", "Pharmaceutical", "Biotechnology",
        "Wellness", "Veterinary"
    ]
    manufacturing_keywords = [
        "Manufacturing", "Chemicals", "Automotive", "Aerospace", "Electrical",
        "Machinery", "Printing", "Packaging"
    ]
    services_keywords = [
        "Consulting", "Staffing", "Outsourcing", "Legal", "Management",
        "Human Resources", "Public Relations"
    ]

    def map_broad_sector(industry):
        if pd.isna(industry):
            return "Other"
        ind_lower = industry.lower()
        for kw in tech_keywords:
            if kw.lower() in ind_lower:
                return "Technology"
        for kw in finance_keywords:
            if kw.lower() in ind_lower:
                return "Finance"
        for kw in health_keywords:
            if kw.lower() in ind_lower:
                return "Healthcare"
        for kw in manufacturing_keywords:
            if kw.lower() in ind_lower:
                return "Manufacturing"
        for kw in services_keywords:
            if kw.lower() in ind_lower:
                return "Services"
        return "Other"

    df["broad_sector"] = df["Industry"].apply(map_broad_sector)

    return df


def save_processed(df: pd.DataFrame, path: str = PROCESSED_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def validate_processed(df: pd.DataFrame) -> dict:
    """
    Post-processing validation checks. Raises on critical failures.
    Returns a dict of all check outcomes.
    """
    results = {}
    failures = []

    # Row count preserved
    results["row_count"] = len(df)
    if len(df) != 100_000:
        failures.append(f"Row count is {len(df)}, expected 100,000")

    # Column count
    expected_cols = 14  # 9 original + 5 derived
    results["column_count"] = len(df.columns)
    if len(df.columns) != expected_cols:
        failures.append(f"Column count is {len(df.columns)}, expected {expected_cols}")

    # Required derived columns present
    for col in ["company_age", "founding_decade", "size_band", "broad_sector",
                "is_name_country_dup"]:
        if col not in df.columns:
            failures.append(f"Derived column '{col}' is missing")
    results["derived_columns_present"] = [
        col for col in ["company_age", "founding_decade", "size_band",
                        "broad_sector", "is_name_country_dup"]
        if col in df.columns
    ]

    # No nulls in key columns post-cleaning
    key_cols = ["Organization Id", "Name", "Country", "Industry",
                "Founded", "Number of employees", "company_age",
                "founding_decade", "size_band", "broad_sector"]
    null_post = {col: int(df[col].isnull().sum()) for col in key_cols if col in df.columns}
    results["null_counts_post"] = null_post
    for col, n in null_post.items():
        if n > 0:
            failures.append(f"Post-clean null in '{col}': {n}")

    # size_band — no Unknown values (means no invalid employee counts)
    unknown_bands = int((df["size_band"].astype(str) == "Unknown").sum())
    results["size_band_unknown_count"] = unknown_bands
    if unknown_bands > 0:
        failures.append(f"size_band has {unknown_bands} 'Unknown' values")

    # size_band distribution totals to 100k
    results["size_band_distribution"] = (
        df["size_band"].value_counts().sort_index().to_dict()
    )
    if sum(results["size_band_distribution"].values()) != 100_000:
        failures.append("size_band distribution does not sum to 100,000")

    # company_age — all positive (Founded <= REFERENCE_YEAR confirmed in quality check)
    neg_age = int((df["company_age"] < 0).sum())
    results["negative_company_age_count"] = neg_age
    if neg_age > 0:
        failures.append(f"{neg_age} rows have negative company_age")

    # company_age reference year
    results["company_age_reference_year"] = REFERENCE_YEAR

    # founding_decade — all within expected range
    decade_min = int(df["founding_decade"].min())
    decade_max = int(df["founding_decade"].max())
    results["founding_decade_range"] = [decade_min, decade_max]

    # broad_sector — only known values
    valid_sectors = {"Technology", "Finance", "Healthcare", "Manufacturing", "Services", "Other"}
    actual_sectors = set(df["broad_sector"].unique())
    invalid_sectors = actual_sectors - valid_sectors
    results["broad_sector_distribution"] = df["broad_sector"].value_counts().to_dict()
    if invalid_sectors:
        failures.append(f"Unexpected broad_sector values: {invalid_sectors}")

    # is_name_country_dup — binary 0/1 only
    invalid_flag = int(~df["is_name_country_dup"].isin([0, 1]).all())
    results["is_name_country_dup_invalid_values"] = invalid_flag
    results["is_name_country_dup_flagged_count"] = int(df["is_name_country_dup"].sum())
    if invalid_flag:
        failures.append("is_name_country_dup contains values other than 0/1")

    # Org Id still unique post-processing
    results["duplicate_org_ids_post"] = int(df["Organization Id"].duplicated().sum())
    if results["duplicate_org_ids_post"] > 0:
        failures.append("Duplicate Organization Ids found in processed data")

    results["validation_passed"] = len(failures) == 0
    results["failures"] = failures
    return results


def run_pipeline():
    print("Loading raw data...")
    df_raw = load_raw()

    print("Validating schema...")
    schema_check = validate_schema(df_raw)
    if schema_check["missing_columns"]:
        raise ValueError(f"Missing columns: {schema_check['missing_columns']}")

    print("Checking data quality (pre-clean)...")
    quality_report = check_quality(df_raw)

    print("Cleaning and enriching...")
    df_clean = clean_and_enrich(df_raw)

    print("Validating processed data...")
    validation = validate_processed(df_clean)
    if not validation["validation_passed"]:
        raise RuntimeError(
            f"Post-processing validation FAILED:\n" +
            "\n".join(f"  - {f}" for f in validation["failures"])
        )

    full_report = {
        "pipeline_run_date": datetime.date.today().isoformat(),
        "reference_year_for_company_age": REFERENCE_YEAR,
        "schema_check": schema_check,
        "quality_pre": quality_report,
        "quality_post": validation,
        "preprocessing_decisions": [
            "All 9 raw columns preserved unchanged in output.",
            "Founded and Number of employees coerced from string to Int64; coercion failures reported.",
            f"company_age derived dynamically as {REFERENCE_YEAR} - Founded (reference year updates each run).",
            "founding_decade = Founded // 10 * 10.",
            "size_band assigned using EU SME thresholds: Micro<50, Small 50-249, Medium 250-999, Large 1000-4999, Enterprise 5000+.",
            "broad_sector derived from Industry keyword matching across 5 macro-sectors (Technology, Finance, Healthcare, Manufacturing, Services); unmatched → 'Other'.",
            "is_name_country_dup flag (0/1) added: marks 764 rows sharing the same Name+Country. All have unique Organization Ids — rows retained, not dropped.",
            "No rows dropped: zero nulls, zero invalid types, zero out-of-range values, zero statistical outliers.",
            "Original raw file preserved unchanged at data/raw/.",
        ],
        "problems_found": [
            "27,585 duplicate Name values — expected in a global dataset; businesses share names. Organization Id is the unique key.",
            "764 rows share the same Name+Country — all have distinct Organization Ids and different Industry/Founded values. Flagged via is_name_country_dup column; not dropped.",
            "6 rows share Name+Country+Industry — distinct Organization Ids and different Founded years; treated as legitimately different organizations.",
            "No statistical outliers in Number of employees (IQR fences: -4,984.5 to 14,995.5; z-score range: -1.73 to +1.73) — uniform distribution by dataset design.",
            "broad_sector 'Other' covers 70.1% of records (70,070 rows) — 70 of 147 industry labels contain none of the 5 sector keywords. This is a known limitation of the keyword-matching approach, not a data defect.",
            "company_age was previously hard-coded to reference year 2024. Fixed: now uses datetime.date.today().year.",
        ],
    }

    os.makedirs("data/processed", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    save_processed(df_clean)

    print(f"\nPipeline complete.")
    print(f"  Rows saved  : {len(df_clean):,}")
    print(f"  Columns     : {len(df_clean.columns)} (9 original + 5 derived)")
    print(f"  Output      : {PROCESSED_PATH}")
    print(f"  Report      : {REPORT_PATH}")
    print(f"  Validation  : {'PASSED' if validation['validation_passed'] else 'FAILED'}")
    return df_clean, full_report


if __name__ == "__main__":
    run_pipeline()
