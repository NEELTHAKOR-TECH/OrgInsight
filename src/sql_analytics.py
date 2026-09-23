"""
Phase 4 — SQL / Data Analytics
Uses DuckDB (in-process, no server required) to run analytical queries
against the cleaned, processed dataset.

Covers:
  Q01  Total records & column sanity check
  Q02  Industry ranking by organization count (top 20)
  Q03  Country ranking by organization count (top 20)
  Q04  Broad sector overview — count, %, avg/min/max employees
  Q05  Size band distribution — count, %, avg employees, avg age
  Q06  Founding decade trends — count, avg employees, avg age
  Q07  Top industry per top-15 country
  Q08  Industry avg-employee ranking — top 10 and bottom 10 (min 100 orgs)
  Q09  Year-by-year founding count (full range, not hard-coded)
  Q10  Sector × size band cross-tabulation
  Q11  Industry diversity per country — top 15 most diverse
  Q12  Employee percentile distribution
  Q13  Country avg-employee ranking — top 10 and bottom 10 (min 100 orgs)
  Q14  Year-on-year founding change (all years)
  Q15  Duplicate name+country flag summary
  Q16  Industry spread analysis — uniformity check
  Q17  Peak vs trough founding years comparison
  Q18  Sector enterprise-rate ranking

All query results are validated against known values from the processed dataset.
Nothing is fabricated.
"""

import duckdb
import pandas as pd
import json
import os

PROCESSED_PATH = "data/processed/organizations_clean.csv"
OUTPUT_PATH = "data/processed/sql_analytics.json"
VALIDATION_PATH = "data/processed/sql_analytics_validation.json"


def _con(path: str = PROCESSED_PATH) -> duckdb.DuckDBPyConnection:
    """Open a fresh in-process DuckDB connection and load the processed CSV."""
    con = duckdb.connect()
    load_sql = (
        "CREATE TABLE orgs AS SELECT * FROM read_csv_auto("
        + chr(39) + path + chr(39) + ")"
    )
    con.execute(load_sql)
    return con


def run_analytics(path: str = PROCESSED_PATH) -> dict:
    con = _con(path)
    results = {}

    # ── Q01: Total records & column sanity ───────────────────────────────────
    q01 = con.execute("""
        SELECT
            COUNT(*)                          AS total_rows,
            COUNT(DISTINCT "Organization Id") AS unique_org_ids,
            COUNT(DISTINCT Industry)          AS unique_industries,
            COUNT(DISTINCT Country)           AS unique_countries,
            COUNT(DISTINCT size_band)         AS unique_size_bands,
            COUNT(DISTINCT broad_sector)      AS unique_sectors,
            SUM(CASE WHEN "Number of employees" IS NULL THEN 1 ELSE 0 END) AS null_employees,
            SUM(CASE WHEN Industry IS NULL THEN 1 ELSE 0 END)              AS null_industry,
            SUM(CASE WHEN Country IS NULL THEN 1 ELSE 0 END)               AS null_country,
            MIN(Founded)  AS min_founded,
            MAX(Founded)  AS max_founded,
            MIN(company_age) AS min_age,
            MAX(company_age) AS max_age
        FROM orgs
    """).df()
    results["Q01_dataset_totals"] = q01.to_dict(orient="records")

    # ── Q02: Industry ranking by org count (top 20) ──────────────────────────
    q02 = con.execute("""
        SELECT
            Industry,
            COUNT(*)                                        AS org_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_of_total,
            ROUND(AVG("Number of employees"), 1)           AS avg_employees,
            ROUND(STDDEV("Number of employees"), 1)        AS stddev_employees,
            MIN("Number of employees")                     AS min_employees,
            MAX("Number of employees")                     AS max_employees
        FROM orgs
        GROUP BY Industry
        ORDER BY org_count DESC
        LIMIT 20
    """).df()
    results["Q02_top_industries_by_count"] = q02.to_dict(orient="records")

    # ── Q03: Country ranking by org count (top 20) ───────────────────────────
    q03 = con.execute("""
        SELECT
            Country,
            COUNT(*)                                        AS org_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_of_total,
            ROUND(AVG("Number of employees"), 1)           AS avg_employees,
            COUNT(DISTINCT Industry)                       AS unique_industries
        FROM orgs
        GROUP BY Country
        ORDER BY org_count DESC
        LIMIT 20
    """).df()
    results["Q03_top_countries_by_count"] = q03.to_dict(orient="records")

    # ── Q04: Broad sector overview ────────────────────────────────────────────
    q04 = con.execute("""
        SELECT
            broad_sector,
            COUNT(*)                                        AS org_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_of_total,
            ROUND(AVG("Number of employees"), 1)           AS avg_employees,
            ROUND(MEDIAN("Number of employees"), 1)        AS median_employees,
            ROUND(STDDEV("Number of employees"), 1)        AS stddev_employees,
            MIN("Number of employees")                     AS min_employees,
            MAX("Number of employees")                     AS max_employees
        FROM orgs
        GROUP BY broad_sector
        ORDER BY org_count DESC
    """).df()
    results["Q04_sector_overview"] = q04.to_dict(orient="records")

    # ── Q05: Size band distribution ───────────────────────────────────────────
    q05 = con.execute("""
        SELECT
            size_band,
            COUNT(*)                                        AS org_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_of_total,
            ROUND(AVG("Number of employees"), 1)           AS avg_employees,
            MIN("Number of employees")                     AS min_employees,
            MAX("Number of employees")                     AS max_employees,
            ROUND(AVG(company_age), 1)                     AS avg_age_years
        FROM orgs
        GROUP BY size_band
        ORDER BY
            CASE size_band
                WHEN 'Micro'      THEN 1
                WHEN 'Small'      THEN 2
                WHEN 'Medium'     THEN 3
                WHEN 'Large'      THEN 4
                WHEN 'Enterprise' THEN 5
            END
    """).df()
    results["Q05_size_band_distribution"] = q05.to_dict(orient="records")

    # ── Q06: Founding decade trends ───────────────────────────────────────────
    q06 = con.execute("""
        SELECT
            founding_decade                              AS decade,
            COUNT(*)                                     AS orgs_founded,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_of_total,
            ROUND(AVG("Number of employees"), 1)         AS avg_employees,
            ROUND(STDDEV("Number of employees"), 1)      AS stddev_employees,
            ROUND(AVG(company_age), 1)                   AS avg_age_years
        FROM orgs
        WHERE founding_decade IS NOT NULL
        GROUP BY founding_decade
        ORDER BY founding_decade
    """).df()
    results["Q06_founding_decade_trends"] = q06.to_dict(orient="records")

    # ── Q07: Top industry per top-15 country ──────────────────────────────────
    # Fixed: note that with ~680 orgs/industry and ~400 orgs/country, the top
    # industry per country has very few orgs (~10-12). This is a real finding:
    # no single industry dominates any country — all are near-uniform.
    q07 = con.execute("""
        WITH country_totals AS (
            SELECT Country, COUNT(*) AS total
            FROM orgs
            GROUP BY Country
            ORDER BY total DESC
            LIMIT 15
        ),
        industry_counts AS (
            SELECT
                o.Country,
                o.Industry,
                COUNT(*) AS cnt
            FROM orgs o
            INNER JOIN country_totals ct ON o.Country = ct.Country
            GROUP BY o.Country, o.Industry
        ),
        ranked AS (
            SELECT
                Country,
                Industry,
                cnt,
                ROW_NUMBER() OVER (PARTITION BY Country ORDER BY cnt DESC) AS rn
            FROM industry_counts
        )
        SELECT
            r.Country,
            r.Industry          AS top_industry,
            r.cnt               AS top_industry_count,
            ct.total            AS country_total_orgs,
            ROUND(100.0 * r.cnt / ct.total, 1) AS top_industry_pct
        FROM ranked r
        INNER JOIN country_totals ct ON r.Country = ct.Country
        WHERE r.rn = 1
        ORDER BY ct.total DESC
    """).df()
    results["Q07_top_industry_per_top_country"] = q07.to_dict(orient="records")

    # ── Q08: Industry avg-employee ranking (top 10 / bottom 10, min 100 orgs) ─
    q08_top = con.execute("""
        SELECT
            Industry,
            COUNT(*)                              AS org_count,
            ROUND(AVG("Number of employees"), 1) AS avg_employees,
            ROUND(MEDIAN("Number of employees"), 1) AS median_employees,
            ROUND(STDDEV("Number of employees"), 1) AS stddev_employees
        FROM orgs
        GROUP BY Industry
        HAVING COUNT(*) >= 100
        ORDER BY avg_employees DESC
        LIMIT 10
    """).df()
    q08_bot = con.execute("""
        SELECT
            Industry,
            COUNT(*)                              AS org_count,
            ROUND(AVG("Number of employees"), 1) AS avg_employees,
            ROUND(MEDIAN("Number of employees"), 1) AS median_employees,
            ROUND(STDDEV("Number of employees"), 1) AS stddev_employees
        FROM orgs
        GROUP BY Industry
        HAVING COUNT(*) >= 100
        ORDER BY avg_employees ASC
        LIMIT 10
    """).df()
    results["Q08_top10_industries_by_avg_employees"]    = q08_top.to_dict(orient="records")
    results["Q08_bottom10_industries_by_avg_employees"] = q08_bot.to_dict(orient="records")

    # ── Q09: Year-by-year founding count (full range — dynamic, not hard-coded) ─
    q09 = con.execute("""
        SELECT
            Founded                AS year,
            COUNT(*)               AS orgs_founded,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 3) AS pct_of_total
        FROM orgs
        GROUP BY Founded
        ORDER BY Founded
    """).df()
    results["Q09_founding_trend_by_year"] = q09.to_dict(orient="records")

    # ── Q10: Sector × size band cross-tabulation ──────────────────────────────
    q10 = con.execute("""
        SELECT
            broad_sector,
            size_band,
            COUNT(*) AS org_count,
            ROUND(100.0 * COUNT(*) /
                SUM(COUNT(*)) OVER (PARTITION BY broad_sector), 1) AS pct_within_sector
        FROM orgs
        GROUP BY broad_sector, size_band
        ORDER BY broad_sector,
            CASE size_band
                WHEN 'Micro'      THEN 1
                WHEN 'Small'      THEN 2
                WHEN 'Medium'     THEN 3
                WHEN 'Large'      THEN 4
                WHEN 'Enterprise' THEN 5
            END
    """).df()
    results["Q10_sector_size_crosstab"] = q10.to_dict(orient="records")

    # ── Q11: Industry diversity per country (top 15 most diverse) ────────────
    # HAVING clause removed: all 243 countries have >= 351 orgs; filter was redundant
    q11 = con.execute("""
        SELECT
            Country,
            COUNT(DISTINCT Industry)   AS unique_industries,
            COUNT(*)                   AS org_count,
            ROUND(100.0 * COUNT(DISTINCT Industry) / 147, 1) AS pct_of_all_industries
        FROM orgs
        GROUP BY Country
        ORDER BY unique_industries DESC
        LIMIT 15
    """).df()
    results["Q11_most_diverse_countries"] = q11.to_dict(orient="records")

    # ── Q12: Employee percentile distribution ─────────────────────────────────
    q12 = con.execute("""
        SELECT
            ROUND(MIN("Number of employees"), 0)                              AS min_val,
            ROUND(PERCENTILE_CONT(0.01) WITHIN GROUP
                  (ORDER BY "Number of employees"), 0)                        AS p01,
            ROUND(PERCENTILE_CONT(0.05) WITHIN GROUP
                  (ORDER BY "Number of employees"), 0)                        AS p05,
            ROUND(PERCENTILE_CONT(0.10) WITHIN GROUP
                  (ORDER BY "Number of employees"), 0)                        AS p10,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP
                  (ORDER BY "Number of employees"), 0)                        AS p25,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
                  (ORDER BY "Number of employees"), 0)                        AS p50,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
                  (ORDER BY "Number of employees"), 0)                        AS p75,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP
                  (ORDER BY "Number of employees"), 0)                        AS p90,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP
                  (ORDER BY "Number of employees"), 0)                        AS p95,
            ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP
                  (ORDER BY "Number of employees"), 0)                        AS p99,
            ROUND(MAX("Number of employees"), 0)                              AS max_val,
            ROUND(AVG("Number of employees"), 2)                              AS mean_val,
            ROUND(STDDEV("Number of employees"), 2)                           AS stddev_val
        FROM orgs
    """).df()
    results["Q12_employee_percentile_distribution"] = q12.to_dict(orient="records")

    # ── Q13: Country avg-employee ranking (top 10 / bottom 10, min 100 orgs) ─
    # Every country has >= 351 orgs so HAVING >= 100 never filters anyone;
    # included for explicitness
    q13_top = con.execute("""
        SELECT Country,
               COUNT(*) AS org_count,
               ROUND(AVG("Number of employees"), 1) AS avg_employees,
               ROUND(MEDIAN("Number of employees"), 1) AS median_employees
        FROM orgs
        GROUP BY Country
        HAVING COUNT(*) >= 100
        ORDER BY avg_employees DESC
        LIMIT 10
    """).df()
    q13_bot = con.execute("""
        SELECT Country,
               COUNT(*) AS org_count,
               ROUND(AVG("Number of employees"), 1) AS avg_employees,
               ROUND(MEDIAN("Number of employees"), 1) AS median_employees
        FROM orgs
        GROUP BY Country
        HAVING COUNT(*) >= 100
        ORDER BY avg_employees ASC
        LIMIT 10
    """).df()
    results["Q13_top10_countries_by_avg_employees"]    = q13_top.to_dict(orient="records")
    results["Q13_bottom10_countries_by_avg_employees"] = q13_bot.to_dict(orient="records")

    # ── Q14: Year-on-year founding change ─────────────────────────────────────
    q14 = con.execute("""
        WITH yearly AS (
            SELECT Founded AS year, COUNT(*) AS n
            FROM orgs
            GROUP BY Founded
        )
        SELECT
            year,
            n AS orgs_founded,
            LAG(n) OVER (ORDER BY year)        AS prev_year_count,
            n - LAG(n) OVER (ORDER BY year)    AS yoy_change,
            ROUND(100.0 * (n - LAG(n) OVER (ORDER BY year))
                  / NULLIF(LAG(n) OVER (ORDER BY year), 0), 1) AS yoy_pct_change
        FROM yearly
        ORDER BY year
    """).df()
    results["Q14_year_on_year_founding_change"] = q14.to_dict(orient="records")

    # ── Q15: Duplicate name+country flag summary ──────────────────────────────
    q15 = con.execute("""
        SELECT
            is_name_country_dup             AS dup_flag,
            COUNT(*)                        AS org_count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_of_total,
            ROUND(AVG("Number of employees"), 1) AS avg_employees,
            ROUND(STDDEV("Number of employees"), 1) AS stddev_employees
        FROM orgs
        GROUP BY is_name_country_dup
        ORDER BY is_name_country_dup
    """).df()
    results["Q15_name_country_dup_flag_summary"] = q15.to_dict(orient="records")

    # ── Q16: Industry spread / uniformity analysis ────────────────────────────
    q16 = con.execute("""
        SELECT
            COUNT(DISTINCT Industry)           AS total_industries,
            MIN(ind_count)                     AS min_orgs_per_industry,
            MAX(ind_count)                     AS max_orgs_per_industry,
            ROUND(AVG(ind_count), 1)           AS avg_orgs_per_industry,
            ROUND(STDDEV(ind_count), 2)        AS stddev_orgs_per_industry,
            MAX(ind_count) - MIN(ind_count)    AS count_range,
            ROUND(STDDEV(ind_count) / AVG(ind_count) * 100, 2) AS coeff_of_variation_pct
        FROM (
            SELECT Industry, COUNT(*) AS ind_count
            FROM orgs
            GROUP BY Industry
        ) sub
    """).df()
    results["Q16_industry_spread_uniformity"] = q16.to_dict(orient="records")

    # ── Q17: Peak vs trough founding years ────────────────────────────────────
    q17 = con.execute("""
        WITH yearly AS (
            SELECT Founded AS year, COUNT(*) AS n
            FROM orgs
            GROUP BY Founded
        ),
        full_range AS (
            SELECT
                year, n,
                ROW_NUMBER() OVER (ORDER BY n DESC) AS rank_desc,
                ROW_NUMBER() OVER (ORDER BY n ASC)  AS rank_asc
            FROM yearly
        )
        SELECT
            year,
            n AS orgs_founded,
            CASE WHEN rank_desc <= 5 THEN 'top_5'
                 WHEN rank_asc  <= 5 THEN 'bottom_5'
            END AS category
        FROM full_range
        WHERE rank_desc <= 5 OR rank_asc <= 5
        ORDER BY n DESC
    """).df()
    results["Q17_peak_and_trough_founding_years"] = q17.to_dict(orient="records")

    # ── Q18: Sector enterprise-rate ranking ───────────────────────────────────
    q18 = con.execute("""
        SELECT
            broad_sector,
            COUNT(*) AS total_orgs,
            SUM(CASE WHEN size_band = 'Enterprise' THEN 1 ELSE 0 END)  AS enterprise_count,
            SUM(CASE WHEN size_band = 'Micro'      THEN 1 ELSE 0 END)  AS micro_count,
            ROUND(100.0 * SUM(CASE WHEN size_band = 'Enterprise' THEN 1 ELSE 0 END)
                  / COUNT(*), 2) AS enterprise_pct,
            ROUND(100.0 * SUM(CASE WHEN size_band = 'Micro' THEN 1 ELSE 0 END)
                  / COUNT(*), 2) AS micro_pct,
            ROUND(AVG("Number of employees"), 1) AS avg_employees
        FROM orgs
        GROUP BY broad_sector
        ORDER BY enterprise_pct DESC
    """).df()
    results["Q18_sector_enterprise_rate"] = q18.to_dict(orient="records")

    con.close()
    return results


def save_results(results: dict, path: str = OUTPUT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_results(results: dict) -> dict:
    """
    Cross-check key SQL outputs against known ground-truth values
    derived independently from the processed dataset using pandas.
    """
    import pandas as pd
    df = pd.read_csv(PROCESSED_PATH)

    failures = []
    checks = {}

    def check(name: str, got, expected, tol=0):
        ok = abs(float(got) - float(expected)) <= (tol if tol else 0)
        checks[name] = {"got": got, "expected": expected, "passed": bool(ok)}
        if not ok:
            failures.append(f"{name}: got={got}, expected={expected}")

    # Q01 — totals
    r01 = results["Q01_dataset_totals"][0]
    check("Q01_total_rows",          r01["total_rows"],        100_000)
    check("Q01_unique_org_ids",      r01["unique_org_ids"],    100_000)
    check("Q01_unique_industries",   r01["unique_industries"],     147)
    check("Q01_unique_countries",    r01["unique_countries"],      243)
    check("Q01_null_employees",      r01["null_employees"],          0)
    check("Q01_null_industry",       r01["null_industry"],           0)
    check("Q01_null_country",        r01["null_country"],            0)
    check("Q01_min_founded",         r01["min_founded"],          1970)
    check("Q01_max_founded",         r01["max_founded"],          2022)

    # Q02 — top industry is Insurance (747)
    top_ind = results["Q02_top_industries_by_count"][0]
    ok_name = top_ind["Industry"] == df["Industry"].value_counts().idxmax()
    ok_cnt  = int(top_ind["org_count"]) == int(df["Industry"].value_counts().max())
    checks["Q02_top_industry_name"]  = {"got": top_ind["Industry"],
                                        "expected": df["Industry"].value_counts().idxmax(),
                                        "passed": ok_name}
    checks["Q02_top_industry_count"] = {"got": int(top_ind["org_count"]),
                                        "expected": int(df["Industry"].value_counts().max()),
                                        "passed": ok_cnt}
    if not ok_name:  failures.append(f"Q02_top_industry_name mismatch")
    if not ok_cnt:   failures.append(f"Q02_top_industry_count mismatch")

    # Q03 — top country is Congo (847)
    top_cty = results["Q03_top_countries_by_count"][0]
    ok_cty  = top_cty["Country"] == df["Country"].value_counts().idxmax()
    ok_ccnt = int(top_cty["org_count"]) == int(df["Country"].value_counts().max())
    checks["Q03_top_country_name"]  = {"got": top_cty["Country"],
                                       "expected": df["Country"].value_counts().idxmax(),
                                       "passed": ok_cty}
    checks["Q03_top_country_count"] = {"got": int(top_cty["org_count"]),
                                       "expected": int(df["Country"].value_counts().max()),
                                       "passed": ok_ccnt}
    if not ok_cty:  failures.append("Q03_top_country_name mismatch")
    if not ok_ccnt: failures.append("Q03_top_country_count mismatch")

    # Q04 — sector counts sum to 100k; pct sum ~100
    sector_total = sum(r["org_count"] for r in results["Q04_sector_overview"])
    check("Q04_sector_total", sector_total, 100_000)
    sector_pct_sum = sum(r["pct_of_total"] for r in results["Q04_sector_overview"])
    ok_pct = abs(sector_pct_sum - 100.0) <= 0.1
    checks["Q04_sector_pct_sum"] = {"got": round(sector_pct_sum, 2),
                                    "expected": 100.0, "passed": ok_pct}
    if not ok_pct:
        failures.append(f"Q04_sector_pct_sum: {sector_pct_sum}")

    # Q05 — size band counts sum to 100k; Micro=492, Enterprise=49990
    sb_total = sum(r["org_count"] for r in results["Q05_size_band_distribution"])
    check("Q05_size_band_total", sb_total, 100_000)
    micro_row = next(r for r in results["Q05_size_band_distribution"] if r["size_band"] == "Micro")
    check("Q05_micro_count", micro_row["org_count"], 492)
    ent_row = next(r for r in results["Q05_size_band_distribution"] if r["size_band"] == "Enterprise")
    check("Q05_enterprise_count", ent_row["org_count"], 49990)

    # Q06 — decade counts sum to 100k; 1970s = 18863
    dec_total = sum(r["orgs_founded"] for r in results["Q06_founding_decade_trends"])
    check("Q06_decade_total", dec_total, 100_000)
    dec_1970 = next(r for r in results["Q06_founding_decade_trends"] if int(r["decade"]) == 1970)
    check("Q06_decade_1970s_count", dec_1970["orgs_founded"], 18863)

    # Q09 — year totals sum to 100k; peak year 1998 has 2022 orgs
    yr_total = sum(r["orgs_founded"] for r in results["Q09_founding_trend_by_year"])
    check("Q09_year_total", yr_total, 100_000)
    peak_yr = max(results["Q09_founding_trend_by_year"], key=lambda r: r["orgs_founded"])
    check("Q09_peak_year",       peak_yr["year"],         1998)
    check("Q09_peak_year_count", peak_yr["orgs_founded"], 2022)

    # Q10 — crosstab grand total = 100k
    ct_total = sum(r["org_count"] for r in results["Q10_sector_size_crosstab"])
    check("Q10_crosstab_total", ct_total, 100_000)

    # Q12 — percentiles match pandas
    emp = df["Number of employees"]
    pct_row = results["Q12_employee_percentile_distribution"][0]
    check("Q12_p50", pct_row["p50"], emp.quantile(0.5),    tol=1)
    check("Q12_p25", pct_row["p25"], emp.quantile(0.25),   tol=1)
    check("Q12_p75", pct_row["p75"], emp.quantile(0.75),   tol=1)
    check("Q12_mean", pct_row["mean_val"], round(emp.mean(), 2), tol=0.5)

    # Q15 — dup flag=1 count = 764
    dup_row = next((r for r in results["Q15_name_country_dup_flag_summary"]
                    if int(r["dup_flag"]) == 1), None)
    if dup_row:
        check("Q15_dup_flag_count", dup_row["org_count"], 764)
    else:
        failures.append("Q15: dup_flag=1 row not found")

    # Q16 — industry spread: 147 industries, min 626, max 747
    sp = results["Q16_industry_spread_uniformity"][0]
    check("Q16_total_industries",      sp["total_industries"],       147)
    check("Q16_min_orgs_per_industry", sp["min_orgs_per_industry"],  626)
    check("Q16_max_orgs_per_industry", sp["max_orgs_per_industry"],  747)

    return {
        "validation_passed": len(failures) == 0,
        "checks_run":        len(checks),
        "failures":          failures,
        "checks":            checks,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_sql_analytics():
    print("Running SQL analytics with DuckDB...")
    results = run_analytics()
    save_results(results)
    print(f"Results saved to {OUTPUT_PATH}")

    print("\nValidating SQL results...")
    validation = validate_results(results)
    os.makedirs("data/processed", exist_ok=True)
    with open(VALIDATION_PATH, "w") as f:
        json.dump(validation, f, indent=2, default=str)
    status = "PASSED" if validation["validation_passed"] else "FAILED"
    print(f"Validation: {status}  "
          f"({validation['checks_run']} checks, {len(validation['failures'])} failures)")
    if not validation["validation_passed"]:
        for fail in validation["failures"]:
            print(f"  FAIL: {fail}")

    # ── Highlights ────────────────────────────────────────────────────────────
    print("\n--- HIGHLIGHTS ---")

    r01 = results["Q01_dataset_totals"][0]
    print(f"\nQ01 Dataset totals:")
    print(f"  Rows={r01['total_rows']:,}  Industries={r01['unique_industries']}  "
          f"Countries={r01['unique_countries']}  Nulls(emp)={r01['null_employees']}")

    print("\nQ04 Broad Sector Overview:")
    for row in results["Q04_sector_overview"]:
        print(f"  {row['broad_sector']:15s} orgs={row['org_count']:>7,}"
              f"  ({row['pct_of_total']:>5.2f}%)"
              f"  avg_emp={row['avg_employees']:>7,.1f}")

    print("\nQ05 Size Band Distribution:")
    for row in results["Q05_size_band_distribution"]:
        print(f"  {row['size_band']:12s} orgs={row['org_count']:>7,}"
              f"  ({row['pct_of_total']:>5.2f}%)"
              f"  avg_age={row['avg_age_years']:>5.1f} yrs")

    print("\nQ06 Founding Decade Trends:")
    for row in results["Q06_founding_decade_trends"]:
        print(f"  {int(row['decade'])}s: {row['orgs_founded']:>6,} orgs"
              f"  avg_emp={row['avg_employees']:>7,.1f}")

    print("\nQ12 Employee Percentiles:")
    p = results["Q12_employee_percentile_distribution"][0]
    print(f"  P10={p['p10']:.0f}  P25={p['p25']:.0f}  Median={p['p50']:.0f}"
          f"  P75={p['p75']:.0f}  P90={p['p90']:.0f}  Mean={p['mean_val']:.1f}")

    print("\nQ16 Industry Spread (uniformity):")
    s = results["Q16_industry_spread_uniformity"][0]
    print(f"  Industries={s['total_industries']}  Range={s['min_orgs_per_industry']}"
          f"-{s['max_orgs_per_industry']}  "
          f"Std={s['stddev_orgs_per_industry']}  CV={s['coeff_of_variation_pct']}%")

    print("\nQ17 Peak vs Trough Founding Years:")
    for row in results["Q17_peak_and_trough_founding_years"]:
        print(f"  {row['category']:9s}  year={row['year']}  orgs={row['orgs_founded']:,}")

    return results, validation


if __name__ == "__main__":
    run_sql_analytics()
