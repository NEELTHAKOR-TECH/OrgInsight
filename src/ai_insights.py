"""
Phase 8 — AI Insights Layer
Reads validated analytical results and generates natural-language
executive insights using any OpenAI-compatible or Ollama API.

Usage:
    python src/ai_insights.py                          # structured fallback (no API needed)
    python src/ai_insights.py --endpoint URL           # Ollama-compatible endpoint
    python src/ai_insights.py --endpoint URL --openai  # OpenAI-compatible endpoint
    python src/ai_insights.py --model gpt-4o           # custom model name
    python src/ai_insights.py --output reports/ai_insights.md
    python src/ai_insights.py --timeout 60             # API timeout in seconds

Environment variables (load from .env or set in shell):
    AI_API_KEY   -- bearer token for OpenAI-compatible APIs (never hard-code)
    AI_ENDPOINT  -- overrides --endpoint default
    AI_MODEL     -- overrides --model default

Rules:
  - Only feeds validated metrics from KPIs, SQL, and ML results.
  - Never fabricates statistics or claims.
  - API key read from AI_API_KEY environment variable (never hard-coded).
  - If no API is available, writes the structured findings without AI text.
"""

import argparse
import json
import os
import sys
import datetime

KPI_PATH = "data/processed/kpis.json"
ML_PATH = "data/processed/ml_results.json"
EXPLAINABILITY_PATH = "data/processed/explainability.json"
SQL_PATH = "data/processed/sql_analytics.json"
DEFAULT_OUTPUT = "reports/ai_insights.md"

DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"
DEFAULT_TIMEOUT = 120


def _load_dotenv():
    """
    Load .env into os.environ without requiring python-dotenv.
    Only reads simple KEY=value lines; skips comments and blank lines.
    Shell environment always takes precedence over .env values.
    Safe to call when .env does not exist.
    """
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:   # shell env takes precedence
                os.environ[key] = value


_load_dotenv()


def load_validated_context() -> dict:
    """Load only validated analytical results — no raw data."""
    with open(KPI_PATH) as f:
        kpis = json.load(f)
    with open(ML_PATH) as f:
        ml = json.load(f)
    with open(EXPLAINABILITY_PATH) as f:
        exp = json.load(f)
    with open(SQL_PATH) as f:
        sql = json.load(f)

    # Extract a concise fact set — no invention
    top_industries = list(kpis["top_10_industries"].items())[:5]
    top_countries = list(kpis["top_10_countries"].items())[:5]
    sector_counts = kpis["broad_sector_counts"]
    size_pcts = kpis["size_band_pct"]

    # ── Key ML metrics from Phase 5 ──────────────────────────────────────────
    dum_acc  = ml["dummy_baseline"]["accuracy"]
    rfd_acc  = ml["random_forest_default"]["accuracy"]
    rfd_auc  = ml["random_forest_default"].get("roc_auc_ovr_macro")
    rfb_acc  = ml["random_forest_balanced"]["accuracy"]
    rfb_f1   = ml["random_forest_balanced"]["macro_f1"]
    rfb_wf1  = ml["random_forest_balanced"]["weighted_f1"]
    cv_mean  = ml["cross_validation"]["rf_balanced_cv_accuracy_mean"]
    cv_std   = ml["cross_validation"]["rf_balanced_cv_accuracy_std"]

    # ── Feature correlations ─────────────────────────────────────────────────
    max_r = max(
        abs(s["pearson_r"])
        for s in ml["feature_target_correlations"].values()
    )

    context = {
        "dataset": {
            "total_organizations": kpis["total_organizations"],
            "total_industries": kpis["total_industries"],
            "total_countries": kpis["total_countries"],
            "avg_employees": round(kpis["avg_employees"], 1),
            "median_employees": kpis["median_employees"],
            "avg_company_age_years": round(kpis["avg_company_age"], 1),
            "peak_founding_year": kpis["peak_founding_year"],
            "founded_range": "1970-2022",
            "industry_hhi": kpis["industry_hhi"],
            "top10_country_pct": kpis["top10_country_pct"],
        },
        "size_distribution_pct": kpis["size_band_pct"],
        "size_distribution_counts": kpis["size_band_counts"],
        "top_5_industries": {k: v for k, v in top_industries},
        "top_5_countries": {k: v for k, v in top_countries},
        "broad_sectors": sector_counts,
        "avg_employees_by_sector": kpis["avg_employees_by_sector"],
        "founding_decade_trend": kpis["founding_decade_counts"],
        "ml_model": {
            "task": "Multi-class classification: predict company size band (5 classes)",
            "features_used": ml["feature_names"],
            "classes": ml["class_labels"],
            "IMPORTANT_note": (
                "Employee count excluded from features (it directly defines size_band — "
                "including it would be data leakage). All feature-target correlations near zero."
            ),
            "max_pearson_r_abs": round(max_r, 6),
            "dummy_baseline_accuracy": round(dum_acc, 4),
            "rf_default_accuracy": round(rfd_acc, 4),
            "rf_default_roc_auc_ovr": round(rfd_auc, 4) if rfd_auc else None,
            "rf_balanced_accuracy": round(rfb_acc, 4),
            "rf_balanced_macro_f1": round(rfb_f1, 4),
            "rf_balanced_weighted_f1": round(rfb_wf1, 4),
            "rf_balanced_cv_accuracy": f"{round(cv_mean, 4)} +/- {round(cv_std, 4)}",
            "feature_importances_pct": {
                k: round(v * 100, 2)
                for k, v in ml["random_forest_balanced"]["feature_importances"].items()
            },
            "model_note": ml.get("model_note", ""),
        },
        "explainability": {
            "methods_used": exp["methods_used"],
            "top_feature_gini": exp["top_feature_by_importance"],
            "top_feature_shap": exp["top_feature_by_shap"],
            "feature_importances_gini_pct": {
                k: round(v * 100, 2)
                for k, v in exp["feature_importances_gini"].items()
            },
            "permutation_importance": {
                "method": exp["permutation_importance"]["method"],
                "test_size": exp["permutation_importance"]["test_size"],
                "n_repeats": exp["permutation_importance"]["n_repeats"],
                "mean_accuracy_drop": {
                    k: round(v, 5)
                    for k, v in exp["permutation_importance"]["mean_accuracy_drop"].items()
                },
                "top_feature": exp["permutation_importance"]["top_feature"],
                "note": exp["permutation_importance"]["note"],
            },
            "shap_mean_abs": exp["shap_mean_abs"],
            "shap_sample_size": exp["sample_size_for_shap"],
            "cross_method_top_feature": exp["cross_method_top_feature"],
            "interpretation": exp["interpretation"],
            "important_caveat": exp.get("important_caveat", ""),
        },
    }
    return context


def build_prompt(context: dict) -> str:
    ctx_json = json.dumps(context, indent=2)
    return f"""You are a professional data analyst writing an executive summary for a business audience.

IMPORTANT RULES:
- Use ONLY the validated metrics provided below. Do NOT invent, estimate, or extrapolate any numbers.
- Clearly distinguish correlation from causation. Never say one variable "causes" another.
- Keep the summary focused, clear, and professional.
- Write in 5 paragraphs:
  (1) Dataset overview and scale
  (2) Industry and sector landscape
  (3) Geographic distribution
  (4) Organizational scale insights and founding trends
  (5) Predictive model and explainability findings — be honest about near-zero predictive signal; mention all three
      explainability methods (Gini MDI, Permutation Importance, SHAP); note that permutation importance is the
      decisive test and all values are near-zero or negative; clearly explain what this means analytically without
      fabricating any numbers

VALIDATED ANALYTICAL RESULTS (all metrics computed from actual data):
{ctx_json}

Write the executive summary now:"""


def call_ollama(endpoint: str, model: str, prompt: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Call Ollama-compatible API (stream=False)."""
    import urllib.request
    import urllib.error
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        endpoint, data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"Ollama HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama connection error: {e.reason}") from e
    return data.get("response", json.dumps(data))


def call_openai_compatible(
    endpoint: str, model: str, prompt: str, api_key: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Call OpenAI-compatible chat completions API."""
    import urllib.request
    import urllib.error
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        endpoint, data=payload, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"OpenAI-compatible API HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenAI-compatible API connection error: {e.reason}") from e
    return data["choices"][0]["message"]["content"]


def generate_structured_fallback(context: dict) -> str:
    """
    Generate structured findings without AI when no API is available.
    All values extracted from the validated context dict — nothing fabricated.
    """
    ml  = context["ml_model"]
    exp = context["explainability"]
    ds  = context["dataset"]

    lines = [
        "# OrgInsight — Structured Findings Report",
        f"*Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} (structured fallback — no AI API)*",
        "",
        "## Dataset Overview",
        f"- **{ds['total_organizations']:,} organizations** across "
        f"**{ds['total_countries']} countries** and **{ds['total_industries']} industries**",
        f"- Average employees: **{ds['avg_employees']:,}** (median: {ds['median_employees']:,})",
        f"- Average company age: **{ds['avg_company_age_years']} years** (founded {ds['founded_range']})",
        f"- Peak founding year: **{ds['peak_founding_year']}**",
        f"- Industry concentration (HHI): {ds['industry_hhi']:.4f} — near-uniform across industries",
        f"- Top 10 countries share: {ds['top10_country_pct']:.1f}% — wide geographic spread",
        "",
        "## Size Band Distribution",
    ]
    for band, pct in context["size_distribution_pct"].items():
        cnt = context["size_distribution_counts"].get(band, 0)
        lines.append(f"- **{band}**: {cnt:,} organizations ({pct:.1f}%)")

    lines += [
        "",
        "## Industry Landscape",
        "Top 5 industries by organization count:",
    ]
    for ind, cnt in context["top_5_industries"].items():
        lines.append(f"- {ind}: {cnt:,}")

    lines += [
        "",
        "## Geographic Distribution",
        "Top 5 countries by organization count:",
    ]
    for cty, cnt in context["top_5_countries"].items():
        lines.append(f"- {cty}: {cnt:,}")

    lines += [
        "",
        "## Broad Sector Average Employees",
    ]
    for sector, avg in context["avg_employees_by_sector"].items():
        lines.append(f"- {sector}: {avg:,.1f} avg employees")

    lines += [
        "",
        "## Predictive Model Findings",
        f"- **Task**: {ml['task']}",
        f"- **Features used**: {', '.join(ml['features_used'])}",
        f"- **Max |Pearson r| (feature vs target)**: {ml['max_pearson_r_abs']:.6f} "
        "(statistically near-zero — no meaningful linear relationship)",
        f"- **Dummy baseline accuracy** (most-frequent class): {ml['dummy_baseline_accuracy']:.4f} ({round(ml['dummy_baseline_accuracy']*100, 2)}%)",
        f"- **RF Default accuracy**: {ml['rf_default_accuracy']:.4f} — at dummy baseline, confirms no predictive signal",
        f"- **RF Balanced accuracy**: {ml['rf_balanced_accuracy']:.4f} — below baseline; optimises minority-class recall",
        f"- **RF Balanced Macro F1**: {ml['rf_balanced_macro_f1']:.4f}",
        f"- **RF Balanced Weighted F1**: {ml['rf_balanced_weighted_f1']:.4f}",
        f"- **5-Fold CV Accuracy (RF Balanced)**: {ml['rf_balanced_cv_accuracy']}",
    ]

    lines += [
        "",
        "## Explainability — Three Methods",
        f"Methods applied: {', '.join(exp['methods_used'])}",
        "",
        "### 1. Gini MDI Feature Importances (RF Balanced)",
        "*(Note: Gini MDI can overstate importance for high-cardinality encoded features)*",
    ]
    for feat, imp in exp["feature_importances_gini_pct"].items():
        lines.append(f"  - {feat}: {imp:.2f}%")
    lines.append(f"- **Top feature (Gini MDI)**: {exp['top_feature_gini']}")

    perm = exp["permutation_importance"]
    lines += [
        "",
        "### 2. Permutation Importance — Decisive Test",
        f"  *Method: {perm['method']}*",
        f"  *Test size: {perm['test_size']:,} rows | Repeats: {perm['n_repeats']}*",
        "  Mean accuracy drop when feature is shuffled (negative = no drop = no signal):",
    ]
    for feat, drop in perm["mean_accuracy_drop"].items():
        lines.append(f"  - {feat}: {drop:+.5f}")
    lines.append(f"- **Top feature (Permutation)**: {perm['top_feature']}")
    lines.append(f"- *{perm['note']}*")

    lines += [
        "",
        "### 3. SHAP (mean |SHAP| value, TreeExplainer)",
        f"  *Sample size: {exp['shap_sample_size']:,} rows (stratified from test set)*",
    ]
    for feat, val in exp["shap_mean_abs"].items():
        lines.append(f"  - {feat}: {val:.6f}")
    lines.append(f"- **Top feature (SHAP)**: {exp['top_feature_shap']}")

    cm = exp["cross_method_top_feature"]
    lines += [
        "",
        "### Cross-Method Agreement",
        f"  - Gini MDI top: {cm['gini_mdi']}",
        f"  - Permutation top: {cm['permutation']}",
        f"  - SHAP top: {cm['shap']}",
        f"  - Methods agree on single top feature: **{cm['agreement']}**",
        "  *(Note: Gini and SHAP rank Country (enc) highest due to its 243 unique values; "
        "Permutation top is noise — all values near zero.)*",
        "",
        "## Explainability Interpretation",
        f"> {exp['interpretation']}",
        "",
        "## Important Caveat",
        f"> {exp.get('important_caveat', 'See interpretation above.')}",
        "",
        "---",
        "*All statistics in this report are computed from the actual dataset — no values are fabricated.*",
        "*Source: data/processed/kpis.json, ml_results.json, explainability.json*",
    ]

    return "\n".join(lines)


def run(
    endpoint: str,
    model: str,
    api_key: str,
    output_path: str,
    use_openai: bool,
    timeout: int = DEFAULT_TIMEOUT,
):
    print("Loading validated context...")
    context = load_validated_context()

    prompt = build_prompt(context)
    ai_text = None

    if endpoint:
        print(f"Calling AI API at {endpoint} with model={model} (timeout={timeout}s)...")
        try:
            if use_openai:
                ai_text = call_openai_compatible(endpoint, model, prompt, api_key, timeout)
            else:
                ai_text = call_ollama(endpoint, model, prompt, timeout)
            print("AI response received.")
        except Exception as e:
            print(f"AI API call failed: {e}", file=sys.stderr)
            print("Falling back to structured findings report.")
    else:
        print("No AI endpoint configured — producing structured findings report.")

    if ai_text:
        report = f"""# AI Executive Summary — OrgInsight Analytics
*Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*
*Model: {model} | Based on 100,000 validated organization records*

---

{ai_text}

---

## Source Data (Validated Inputs)

```json
{json.dumps(context, indent=2)}
```
"""
    else:
        report = generate_structured_fallback(context)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate AI insights from validated analytics")
    # Defaults fall back to environment variables set in .env / shell
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("AI_ENDPOINT", ""),
        help="AI API endpoint URL (env: AI_ENDPOINT)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AI_MODEL", DEFAULT_MODEL),
        help="Model name (env: AI_MODEL)",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output markdown file")
    parser.add_argument(
        "--openai", action="store_true",
        help="Use OpenAI-compatible chat completions format",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"API request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    args = parser.parse_args()

    api_key = os.environ.get("AI_API_KEY", "")
    if api_key:
        print("AI_API_KEY environment variable detected.")

    run(
        endpoint=args.endpoint,
        model=args.model,
        api_key=api_key,
        output_path=args.output,
        use_openai=args.openai,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
