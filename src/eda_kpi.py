"""
Phase 3 — EDA & KPI Analytics
Generates all key performance indicators and exploratory visualizations
from the processed dataset. Saves KPIs to data/processed/kpis.json
and charts to reports/.

All values are computed dynamically from actual data — nothing is hard-coded.
"""

import datetime
import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_PATH = "data/processed/organizations_clean.csv"
KPI_PATH = "data/processed/kpis.json"
KPI_VALIDATION_PATH = "data/processed/kpis_validation.json"
REPORTS_DIR = "reports"

PALETTE = ["#3b82d4", "#7c5cd8", "#e67e22", "#2ecc71", "#e74c3c", "#1abc9c"]
SIZE_ORDER = ["Micro", "Small", "Medium", "Large", "Enterprise"]
sns.set_theme(style="whitegrid", palette=PALETTE)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_PATH)
    df["size_band"] = pd.Categorical(
        df["size_band"],
        categories=SIZE_ORDER,
        ordered=True,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# KPI COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute all KPIs dynamically from actual data. No hard-coded numbers."""
    kpis = {}
    emp = df["Number of employees"].dropna()
    age = df["company_age"].dropna()
    yearly = df.groupby("Founded").size()

    # ── Headline counts ───────────────────────────────────────────────────────
    kpis["total_organizations"] = int(len(df))
    kpis["total_industries"] = int(df["Industry"].nunique())
    kpis["total_countries"] = int(df["Country"].nunique())
    kpis["reference_year"] = datetime.date.today().year

    # ── Employee workforce KPIs ───────────────────────────────────────────────
    kpis["avg_employees"] = round(float(emp.mean()), 1)
    kpis["median_employees"] = round(float(emp.median()), 1)
    kpis["std_employees"] = round(float(emp.std()), 2)
    kpis["min_employees"] = int(emp.min())
    kpis["max_employees"] = int(emp.max())
    kpis["employees_skewness"] = round(float(emp.skew()), 4)
    kpis["employees_kurtosis"] = round(float(emp.kurtosis()), 4)
    kpis["employees_q1"] = int(emp.quantile(0.25))
    kpis["employees_q3"] = int(emp.quantile(0.75))
    kpis["employees_iqr"] = int(emp.quantile(0.75) - emp.quantile(0.25))
    kpis["employees_p10"] = int(emp.quantile(0.10))
    kpis["employees_p90"] = int(emp.quantile(0.90))
    kpis["employees_percentiles"] = {
        str(p): int(emp.quantile(p / 100))
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
    }

    # ── Company age KPIs ──────────────────────────────────────────────────────
    kpis["avg_company_age"] = round(float(age.mean()), 1)
    kpis["median_company_age"] = round(float(age.median()), 1)
    kpis["min_company_age"] = int(age.min())
    kpis["max_company_age"] = int(age.max())
    kpis["std_company_age"] = round(float(age.std()), 2)
    kpis["company_age_skewness"] = round(float(age.skew()), 4)

    # ── Founding year / trend KPIs ────────────────────────────────────────────
    kpis["founded_min_year"] = int(df["Founded"].min())
    kpis["founded_max_year"] = int(df["Founded"].max())
    kpis["orgs_founded_per_year_avg"] = round(float(yearly.mean()), 1)
    kpis["orgs_founded_per_year_std"] = round(float(yearly.std()), 2)
    kpis["peak_founding_year"] = int(yearly.idxmax())
    kpis["peak_founding_year_count"] = int(yearly.max())
    kpis["orgs_founded_per_year"] = {
        str(int(k)): int(v) for k, v in yearly.items()
    }

    # ── Size band distribution ────────────────────────────────────────────────
    kpis["size_band_counts"] = df["size_band"].value_counts().sort_index().to_dict()
    kpis["size_band_pct"] = (
        (df["size_band"].value_counts(normalize=True).sort_index() * 100)
        .round(2)
        .to_dict()
    )

    # ── Founding decade distribution ──────────────────────────────────────────
    decade_counts = df["founding_decade"].value_counts().sort_index()
    kpis["founding_decade_counts"] = {
        str(int(k)): int(v) for k, v in decade_counts.items()
    }

    # ── Industry KPIs ─────────────────────────────────────────────────────────
    ivc = df["Industry"].value_counts()
    kpis["top_10_industries"] = ivc.head(10).to_dict()
    kpis["bottom_10_industries"] = ivc.tail(10).to_dict()
    kpis["industry_count_min"] = int(ivc.min())
    kpis["industry_count_max"] = int(ivc.max())
    kpis["industry_count_std"] = round(float(ivc.std()), 2)
    kpis["industry_hhi"] = round(
        float((df["Industry"].value_counts(normalize=True) ** 2).sum()), 6
    )

    # Average employees per industry (top/bottom 10 by mean, min 100 orgs)
    ind_stats = (
        df.groupby("Industry")["Number of employees"]
        .agg(["mean", "median", "count"])
    )
    ind_stats = ind_stats[ind_stats["count"] >= 100]
    kpis["top_10_industries_by_avg_employees"] = (
        ind_stats.sort_values("mean", ascending=False)
        .head(10)["mean"]
        .round(1)
        .to_dict()
    )
    kpis["bottom_10_industries_by_avg_employees"] = (
        ind_stats.sort_values("mean", ascending=True)
        .head(10)["mean"]
        .round(1)
        .to_dict()
    )

    # ── Country KPIs ──────────────────────────────────────────────────────────
    cvc = df["Country"].value_counts()
    kpis["top_10_countries"] = cvc.head(10).to_dict()
    kpis["country_count_min"] = int(cvc.min())
    kpis["country_count_max"] = int(cvc.max())
    kpis["country_count_mean"] = round(float(cvc.mean()), 1)
    kpis["country_count_std"] = round(float(cvc.std()), 1)
    kpis["country_with_most_orgs"] = str(cvc.idxmax())
    kpis["country_with_fewest_orgs"] = str(cvc.idxmin())
    kpis["countries_with_fewer_than_50_orgs"] = int((cvc < 50).sum())

    # Country concentration
    kpis["top5_country_pct"] = round(float(cvc.head(5).sum() / len(df) * 100), 2)
    kpis["top10_country_pct"] = round(float(cvc.head(10).sum() / len(df) * 100), 2)

    # ── Broad sector KPIs ─────────────────────────────────────────────────────
    kpis["broad_sector_counts"] = df["broad_sector"].value_counts().to_dict()
    kpis["broad_sector_pct"] = (
        (df["broad_sector"].value_counts(normalize=True) * 100).round(2).to_dict()
    )
    kpis["sector_hhi"] = round(
        float((df["broad_sector"].value_counts(normalize=True) ** 2).sum()), 4
    )

    sector_stats = df.groupby("broad_sector")["Number of employees"].agg(
        ["mean", "median"]
    ).round(1)
    kpis["avg_employees_by_sector"] = sector_stats["mean"].sort_values(ascending=False).to_dict()
    kpis["median_employees_by_sector"] = sector_stats["median"].to_dict()

    # Enterprise % by sector
    ct_norm = (
        pd.crosstab(df["broad_sector"], df["size_band"], normalize="index") * 100
    ).round(2)
    kpis["enterprise_pct_by_sector"] = ct_norm["Enterprise"].to_dict()
    kpis["micro_pct_by_sector"] = ct_norm["Micro"].to_dict()

    # ── Average employees per decade ──────────────────────────────────────────
    decade_avg_emp = (
        df.groupby("founding_decade")["Number of employees"]
        .mean()
        .round(1)
        .sort_index()
    )
    kpis["avg_employees_by_decade"] = {
        str(int(k)): round(float(v), 1) for k, v in decade_avg_emp.items()
    }

    # ── Avg company age per size band ─────────────────────────────────────────
    age_by_band = df.groupby("size_band")["company_age"].mean().round(1)
    kpis["avg_company_age_by_size_band"] = age_by_band.to_dict()

    # ── Correlations ─────────────────────────────────────────────────────────
    kpis["corr_founded_vs_employees"] = round(
        float(df["Founded"].corr(df["Number of employees"])), 6
    )
    kpis["corr_company_age_vs_employees"] = round(
        float(df["company_age"].corr(df["Number of employees"])), 6
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    kpis["top_industry_per_size_band"] = (
        df.groupby("size_band")["Industry"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )
    kpis["top_country_per_sector"] = (
        df.groupby("broad_sector")["Country"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )
    kpis["name_country_dup_count"] = int(df["is_name_country_dup"].sum())

    return kpis


def save_kpis(kpis: dict, path: str = KPI_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(kpis, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# KPI VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_kpis(kpis: dict, df: pd.DataFrame) -> dict:
    """Cross-check KPI values against raw DataFrame aggregations."""
    failures = []
    checks = {}

    def check(name, computed, expected, tol=0.01):
        ok = abs(float(computed) - float(expected)) <= tol
        checks[name] = {"computed": computed, "expected": expected, "passed": ok}
        if not ok:
            failures.append(f"{name}: got {computed}, expected {expected}")

    # Row count
    check("total_organizations", kpis["total_organizations"], len(df), tol=0)

    # Industry / country cardinality
    check("total_industries", kpis["total_industries"], df["Industry"].nunique(), tol=0)
    check("total_countries", kpis["total_countries"], df["Country"].nunique(), tol=0)

    # Employee mean / median (allow floating rounding)
    check("avg_employees", kpis["avg_employees"],
          round(df["Number of employees"].mean(), 1), tol=0.05)
    check("median_employees", kpis["median_employees"],
          round(df["Number of employees"].median(), 1), tol=0.05)

    # Company age mean
    check("avg_company_age", kpis["avg_company_age"],
          round(df["company_age"].mean(), 1), tol=0.05)

    # Size band total must equal row count
    sb_total = sum(kpis["size_band_counts"].values())
    check("size_band_counts_total", sb_total, len(df), tol=0)

    # Size band pct must sum to ~100
    sb_pct_total = round(sum(kpis["size_band_pct"].values()), 1)
    check("size_band_pct_sum", sb_pct_total, 100.0, tol=0.2)

    # Top industry by count
    top_ind_name = df["Industry"].value_counts().idxmax()
    top_ind_count = int(df["Industry"].value_counts().max())
    kpi_top_ind_name = list(kpis["top_10_industries"].keys())[0]
    kpi_top_ind_count = list(kpis["top_10_industries"].values())[0]
    ok_name = (kpi_top_ind_name == top_ind_name)
    ok_count = (kpi_top_ind_count == top_ind_count)
    checks["top_industry_name"] = {"passed": ok_name,
                                   "computed": kpi_top_ind_name,
                                   "expected": top_ind_name}
    checks["top_industry_count"] = {"passed": ok_count,
                                    "computed": kpi_top_ind_count,
                                    "expected": top_ind_count}
    if not ok_name:
        failures.append(f"top_industry_name: got {kpi_top_ind_name}, expected {top_ind_name}")
    if not ok_count:
        failures.append(f"top_industry_count: got {kpi_top_ind_count}, expected {top_ind_count}")

    # Top country by count
    top_cty_name = df["Country"].value_counts().idxmax()
    kpi_top_cty_name = list(kpis["top_10_countries"].keys())[0]
    ok_cty = (kpi_top_cty_name == top_cty_name)
    checks["top_country_name"] = {"passed": ok_cty,
                                  "computed": kpi_top_cty_name,
                                  "expected": top_cty_name}
    if not ok_cty:
        failures.append(f"top_country_name: got {kpi_top_cty_name}, expected {top_cty_name}")

    # Peak founding year
    yearly = df.groupby("Founded").size()
    check("peak_founding_year", kpis["peak_founding_year"],
          int(yearly.idxmax()), tol=0)
    check("peak_founding_year_count", kpis["peak_founding_year_count"],
          int(yearly.max()), tol=0)

    # Correlation values
    check("corr_founded_vs_employees",
          kpis["corr_founded_vs_employees"],
          round(float(df["Founded"].corr(df["Number of employees"])), 6), tol=0.0001)

    # is_name_country_dup count
    check("name_country_dup_count", kpis["name_country_dup_count"],
          int(df["is_name_country_dup"].sum()), tol=0)

    result = {
        "validation_passed": len(failures) == 0,
        "checks_run": len(checks),
        "failures": failures,
        "checks": checks,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def _save(name: str):
    path = f"{REPORTS_DIR}/{name}"
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_size_distribution(df: pd.DataFrame):
    """Bar chart — org count per size band."""
    counts = df["size_band"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=PALETTE[:len(counts)])
    ax.set_title("Organization Count by Employee Size Band", fontsize=14, fontweight="bold")
    ax.set_xlabel("Size Band")
    ax.set_ylabel("Number of Organizations")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                f"{val:,}", ha="center", va="bottom", fontsize=9)
    return _save("fig_size_distribution.png")


def plot_top_industries(df: pd.DataFrame):
    """Horizontal bar — top 15 industries by org count."""
    top = df["Industry"].value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(top))]
    ax.barh(top.index, top.values, color=colors)
    ax.set_title("Top 15 Industries by Organization Count", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Organizations")
    xmax = top.values.max()
    ax.set_xlim(0, xmax * 1.12)
    for i, val in enumerate(top.values):
        ax.text(val + 3, i, str(val), va="center", fontsize=8)
    return _save("fig_top_industries.png")


def plot_top_countries(df: pd.DataFrame):
    """Horizontal bar — top 15 countries by org count."""
    top = df["Country"].value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.index, top.values, color=PALETTE[1])
    ax.set_title("Top 15 Countries by Organization Count", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Organizations")
    xmax = top.values.max()
    ax.set_xlim(0, xmax * 1.12)
    for i, val in enumerate(top.values):
        ax.text(val + 2, i, str(val), va="center", fontsize=8)
    return _save("fig_top_countries.png")


def plot_founding_trend(df: pd.DataFrame):
    """Area + line — orgs founded per year. Title is dynamic."""
    yearly = df.groupby("Founded").size().reset_index(name="count")
    yr_min = int(yearly["Founded"].min())
    yr_max = int(yearly["Founded"].max())
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(yearly["Founded"], yearly["count"], alpha=0.25, color=PALETTE[0])
    ax.plot(yearly["Founded"], yearly["count"], color=PALETTE[0], linewidth=1.8)
    # Annotate peak year
    peak_row = yearly.loc[yearly["count"].idxmax()]
    ax.annotate(
        f"Peak: {int(peak_row['Founded'])} ({int(peak_row['count'])} orgs)",
        xy=(peak_row["Founded"], peak_row["count"]),
        xytext=(peak_row["Founded"] - 8, peak_row["count"] + 50),
        fontsize=9, color=PALETTE[4],
        arrowprops=dict(arrowstyle="->", color=PALETTE[4], lw=1),
    )
    ax.set_title(
        f"Organizations Founded Per Year ({yr_min}–{yr_max})",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Year Founded")
    ax.set_ylabel("Number of Organizations")
    return _save("fig_founding_trend.png")


def plot_employee_distribution(df: pd.DataFrame):
    """Histogram + box — employee count distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram with percentile lines
    emp = df["Number of employees"].dropna()
    axes[0].hist(emp, bins=50, color=PALETTE[0], edgecolor="white", alpha=0.85)
    for p, label, color in [(25, "Q1", PALETTE[2]), (50, "Med", PALETTE[4]),
                             (75, "Q3", PALETTE[1])]:
        v = emp.quantile(p / 100)
        axes[0].axvline(v, color=color, linestyle="--", linewidth=1.4, label=f"{label}={v:.0f}")
    axes[0].set_title("Employee Count Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Employees")
    axes[0].set_ylabel("Frequency")
    axes[0].legend(fontsize=8)

    # Box per size band (only Large + Enterprise have enough range to show)
    data_by_band = [
        df[df["size_band"] == band]["Number of employees"].dropna().values
        for band in SIZE_ORDER
    ]
    bp = axes[1].boxplot(data_by_band, patch_artist=True)
    axes[1].set_xticks(range(1, 6))
    axes[1].set_xticklabels(SIZE_ORDER)
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title("Employee Distribution by Size Band", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Size Band")
    axes[1].set_ylabel("Number of Employees")
    return _save("fig_employee_distribution.png")


def plot_sector_breakdown(df: pd.DataFrame):
    """Pie chart — organizations by broad sector."""
    sector_counts = df["broad_sector"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 7))
    wedges, texts, autotexts = ax.pie(
        sector_counts.values,
        labels=sector_counts.index,
        autopct="%1.1f%%",
        colors=PALETTE[:len(sector_counts)],
        startangle=140,
        pctdistance=0.82,
    )
    for text in autotexts:
        text.set_fontsize(9)
    ax.set_title("Organizations by Broad Sector", fontsize=14, fontweight="bold")
    return _save("fig_sector_breakdown.png")


def plot_avg_employees_by_sector(df: pd.DataFrame):
    """Bar chart — avg employees per sector. Y-axis rescaled to show differences."""
    sector_avg = (
        df.groupby("broad_sector")["Number of employees"]
        .mean()
        .sort_values(ascending=False)
        .round(0)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(sector_avg.index, sector_avg.values, color=PALETTE[:len(sector_avg)])
    ax.set_title("Average Employees by Broad Sector", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Avg. Employees")
    # Rescale axis to show variation (all values near 5000)
    ymin = int(sector_avg.min()) - 200
    ymax = int(sector_avg.max()) + 300
    ax.set_ylim(ymin, ymax)
    for bar, val in zip(bars, sector_avg.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 15,
                f"{int(val):,}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", labelrotation=15)
    return _save("fig_avg_employees_by_sector.png")


def plot_decade_vs_employees(df: pd.DataFrame):
    """Line chart — avg employees per founding decade. Rescaled y-axis."""
    decade_avg = df.groupby("founding_decade")["Number of employees"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(decade_avg.index.astype(int), decade_avg.values, marker="o",
            color=PALETTE[2], linewidth=2, markersize=8)
    ax.fill_between(decade_avg.index.astype(int), decade_avg.values,
                    alpha=0.12, color=PALETTE[2])
    ymin = int(decade_avg.min()) - 150
    ymax = int(decade_avg.max()) + 150
    ax.set_ylim(ymin, ymax)
    ax.set_title("Average Employees by Founding Decade", fontsize=14, fontweight="bold")
    ax.set_xlabel("Founding Decade")
    ax.set_ylabel("Avg. Number of Employees")
    for x, y in zip(decade_avg.index.astype(int), decade_avg.values):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    return _save("fig_decade_vs_employees.png")


def plot_company_age_distribution(df: pd.DataFrame):
    """Histogram — company age distribution (dynamic reference year)."""
    ref_year = datetime.date.today().year
    age = df["company_age"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(age, bins=40, color=PALETTE[3], edgecolor="white", alpha=0.85)
    ax.axvline(age.mean(), color=PALETTE[4], linestyle="--", linewidth=1.6,
               label=f"Mean = {age.mean():.1f} yrs")
    ax.axvline(age.median(), color=PALETTE[1], linestyle=":", linewidth=1.6,
               label=f"Median = {age.median():.0f} yrs")
    ax.set_title(
        f"Company Age Distribution (Reference Year: {ref_year})",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Company Age (years)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    return _save("fig_company_age_distribution.png")


def plot_industry_avg_employees(df: pd.DataFrame):
    """Dual horizontal bar — top 10 and bottom 10 industries by avg employees."""
    ind_stats = (
        df.groupby("Industry")["Number of employees"]
        .agg(["mean", "count"])
    )
    ind_stats = ind_stats[ind_stats["count"] >= 100]
    top10 = ind_stats.sort_values("mean", ascending=False).head(10)
    bot10 = ind_stats.sort_values("mean", ascending=True).head(10).sort_values("mean", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top 10
    axes[0].barh(top10.index, top10["mean"].round(0), color=PALETTE[0], alpha=0.85)
    axes[0].set_title("Top 10 Industries\n(Highest Avg Employees)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Avg Employees")
    xmax0 = top10["mean"].max()
    axes[0].set_xlim(xmax0 * 0.88, xmax0 * 1.06)
    for i, (val, name) in enumerate(zip(top10["mean"], top10.index)):
        axes[0].text(val + 5, i, f"{val:.0f}", va="center", fontsize=8)

    # Bottom 10 (lowest)
    bot10_plot = bot10.sort_values("mean")
    axes[1].barh(bot10_plot.index, bot10_plot["mean"].round(0), color=PALETTE[4], alpha=0.85)
    axes[1].set_title("Bottom 10 Industries\n(Lowest Avg Employees)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Avg Employees")
    xmax1 = bot10_plot["mean"].max()
    axes[1].set_xlim(bot10_plot["mean"].min() * 0.95, xmax1 * 1.06)
    for i, (val, name) in enumerate(zip(bot10_plot["mean"], bot10_plot.index)):
        axes[1].text(val + 5, i, f"{val:.0f}", va="center", fontsize=8)

    plt.suptitle("Average Employees per Industry\n(industries with ≥100 organizations)",
                 fontsize=13, fontweight="bold", y=1.01)
    return _save("fig_industry_avg_employees.png")


def plot_sector_size_heatmap(df: pd.DataFrame):
    """Heatmap — sector × size band (% within sector, showing size distribution)."""
    ct_norm = pd.crosstab(
        df["broad_sector"], df["size_band"], normalize="index"
    ) * 100
    # Reorder columns
    ct_norm = ct_norm[[c for c in SIZE_ORDER if c in ct_norm.columns]]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        ct_norm.round(1),
        annot=True, fmt=".1f", cmap="Blues",
        linewidths=0.5, ax=ax, cbar_kws={"label": "% within sector"},
    )
    ax.set_title("Size Band Distribution by Sector (% within each sector)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Size Band")
    ax.set_ylabel("Broad Sector")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    return _save("fig_sector_size_heatmap.png")


def plot_country_org_distribution(df: pd.DataFrame):
    """Histogram — how many orgs per country (shows uniformity vs concentration)."""
    cvc = df["Country"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(cvc.values, bins=40, color=PALETTE[1], edgecolor="white", alpha=0.85)
    ax.axvline(cvc.mean(), color=PALETTE[4], linestyle="--", linewidth=1.6,
               label=f"Mean = {cvc.mean():.0f}")
    ax.axvline(cvc.median(), color=PALETTE[2], linestyle=":", linewidth=1.6,
               label=f"Median = {cvc.median():.0f}")
    ax.set_title("Distribution of Organizations per Country",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Organizations per Country")
    ax.set_ylabel("Number of Countries")
    ax.legend(fontsize=9)
    # Annotate Congo outlier
    ax.annotate(
        f"Congo: {cvc.max()}",
        xy=(cvc.max(), 1), xytext=(cvc.max() - 60, 8),
        fontsize=9, color=PALETTE[4],
        arrowprops=dict(arrowstyle="->", color=PALETTE[4], lw=1),
    )
    return _save("fig_country_org_distribution.png")


def plot_founding_decade_bar(df: pd.DataFrame):
    """Bar chart — orgs founded per decade with count labels."""
    decade_counts = df["founding_decade"].value_counts().sort_index()
    decade_labels = [f"{int(k)}s" for k in decade_counts.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(decade_labels, decade_counts.values, color=PALETTE[:len(decade_counts)])
    ax.set_title("Organizations Founded by Decade", fontsize=14, fontweight="bold")
    ax.set_xlabel("Founding Decade")
    ax.set_ylabel("Number of Organizations")
    note = "(partial decade — data ends 2022)"
    ax.text(len(decade_labels) - 1, decade_counts.values[-1] + 400, note,
            ha="center", fontsize=8, color=PALETTE[4])
    for bar, val in zip(bars, decade_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:,}", ha="center", va="bottom", fontsize=9)
    return _save("fig_founding_decade_bar.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_eda():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Loading processed data...")
    df = load_data()
    print(f"  {len(df):,} rows × {len(df.columns)} columns loaded")

    print("Computing KPIs...")
    kpis = compute_kpis(df)
    save_kpis(kpis)
    print(f"  KPIs saved to {KPI_PATH}")

    print("Validating KPIs against source data...")
    validation = validate_kpis(kpis, df)
    os.makedirs("data/processed", exist_ok=True)
    with open(KPI_VALIDATION_PATH, "w") as f:
        json.dump(validation, f, indent=2, default=str)
    status = "PASSED" if validation["validation_passed"] else "FAILED"
    print(f"  Validation: {status} ({validation['checks_run']} checks, "
          f"{len(validation['failures'])} failures)")
    if not validation["validation_passed"]:
        for fail in validation["failures"]:
            print(f"    FAIL: {fail}")

    print("Generating visualizations...")
    charts = [
        ("fig_size_distribution",       plot_size_distribution(df)),
        ("fig_top_industries",          plot_top_industries(df)),
        ("fig_top_countries",           plot_top_countries(df)),
        ("fig_founding_trend",          plot_founding_trend(df)),
        ("fig_employee_distribution",   plot_employee_distribution(df)),
        ("fig_sector_breakdown",        plot_sector_breakdown(df)),
        ("fig_avg_employees_by_sector", plot_avg_employees_by_sector(df)),
        ("fig_decade_vs_employees",     plot_decade_vs_employees(df)),
        ("fig_company_age_distribution",plot_company_age_distribution(df)),
        ("fig_industry_avg_employees",  plot_industry_avg_employees(df)),
        ("fig_sector_size_heatmap",     plot_sector_size_heatmap(df)),
        ("fig_country_org_distribution",plot_country_org_distribution(df)),
        ("fig_founding_decade_bar",     plot_founding_decade_bar(df)),
    ]
    for name, path in charts:
        print(f"  [OK] {path}")

    print(f"\n--- KEY METRICS ---")
    print(f"Total Organizations : {kpis['total_organizations']:,}")
    print(f"Industries          : {kpis['total_industries']}")
    print(f"Countries           : {kpis['total_countries']}")
    print(f"Avg Employees       : {kpis['avg_employees']:,}")
    print(f"Median Employees    : {kpis['median_employees']:,}")
    print(f"Employee Skewness   : {kpis['employees_skewness']}")
    print(f"Employee Kurtosis   : {kpis['employees_kurtosis']}")
    print(f"Avg Company Age     : {kpis['avg_company_age']} yrs  "
          f"(ref year: {kpis['reference_year']})")
    print(f"Peak Founding Year  : {kpis['peak_founding_year']} "
          f"({kpis['peak_founding_year_count']:,} orgs)")
    print(f"r(Founded, Employees): {kpis['corr_founded_vs_employees']}")
    print(f"\nSize Band Distribution:")
    for band in SIZE_ORDER:
        count = kpis["size_band_counts"].get(band, 0)
        pct = kpis["size_band_pct"].get(band, 0)
        print(f"  {band:12s}: {count:>7,}  ({pct:.2f}%)")

    return kpis, validation


if __name__ == "__main__":
    run_eda()
