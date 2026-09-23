"""
run_pipeline.py — Full project pipeline runner.
Executes all phases in order:
  1. Data Preparation
  2. EDA & KPI Analytics
  3. SQL Analytics
  4. ML Model Training & Evaluation
  5. Model Explainability
  6. Dashboard Generation
  7. AI Insights (structured fallback when no endpoint given)

Run:
    python run_pipeline.py
    python run_pipeline.py --ai-endpoint http://localhost:11434/api/generate
    python run_pipeline.py --ai-endpoint https://api.openai.com/v1/chat/completions --openai
    python run_pipeline.py --skip-ml   (skip ML training — use existing model artefacts)

Environment variables (set in .env or shell):
    AI_API_KEY  — bearer token for OpenAI-compatible APIs
    AI_ENDPOINT — default AI endpoint (overridden by --ai-endpoint)
    AI_MODEL    — default model name  (overridden by --ai-model)
"""

import argparse
import os
import sys
import time


def run_phase(name: str, fn):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = fn()
    elapsed = round(time.time() - t0, 1)
    print(f"  ✓ Completed in {elapsed}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the full analytics pipeline")
    parser.add_argument(
        "--ai-endpoint",
        default=os.environ.get("AI_ENDPOINT", ""),
        help="AI API endpoint for Phase 8 (env: AI_ENDPOINT)",
    )
    parser.add_argument(
        "--ai-model",
        default=os.environ.get("AI_MODEL", "llama3"),
        help="AI model name (env: AI_MODEL)",
    )
    parser.add_argument(
        "--openai", action="store_true",
        help="Use OpenAI-compatible chat completions format for Phase 8",
    )
    parser.add_argument(
        "--ai-timeout", type=int, default=120,
        help="AI API timeout in seconds (default: 120)",
    )
    parser.add_argument("--skip-ml", action="store_true",
                        help="Skip ML training (use if already trained)")
    args = parser.parse_args()

    from src.data_preparation import run_pipeline as phase2
    from src.eda_kpi import run_eda as phase3
    from src.sql_analytics import run_sql_analytics as phase4
    from src.ml_model import train_and_evaluate as phase5
    from src.explainability import run_explainability as phase6
    from src.generate_dashboard import generate_dashboard as phase7
    from src.ai_insights import run as phase8

    print("\n🚀 Global Organizations Analytics Pipeline")
    print("   Dataset: organizations-100000.csv")
    print("   100,000 organizations · 147 industries · 243 countries\n")

    run_phase("Phase 2 — Data Preparation", phase2)
    run_phase("Phase 3 — EDA & KPI Analytics", phase3)
    run_phase("Phase 4 — SQL Analytics (DuckDB)", phase4)

    if not args.skip_ml:
        run_phase("Phase 5 — Feature Engineering & ML", phase5)
        run_phase("Phase 6 — Model Explainability (SHAP)", phase6)
    else:
        print("\n[SKIP] ML phases skipped (--skip-ml flag set)")

    run_phase("Phase 7 — Interactive Dashboard", phase7)

    def _phase8():
        return phase8(
            endpoint=args.ai_endpoint,
            model=args.ai_model,
            api_key=os.environ.get("AI_API_KEY", ""),
            output_path="reports/ai_insights.md",
            use_openai=args.openai,
            timeout=args.ai_timeout,
        )

    run_phase("Phase 8 — AI Insights", _phase8)

    print(f"\n{'='*60}")
    print("  ✅ Pipeline complete!")
    print(f"{'='*60}")
    print("\n  Outputs:")
    print("  - data/processed/organizations_clean.csv")
    print("  - data/processed/kpis.json")
    print("  - data/processed/sql_analytics.json")
    print("  - data/processed/ml_results.json")
    print("  - data/processed/explainability.json")
    print("  - reports/  (8+ chart PNGs)")
    print("  - reports/ai_insights.md")
    print("  - dashboard/index.html  ← Open in browser")
    print()


if __name__ == "__main__":
    main()
