#!/usr/bin/env python3
"""
generate_chap4_tables.py — Regenerate Chapter 4 LaTeX tables from updated result CSVs.

Run this AFTER re-running run_full_ablation.py with the Naive, Moving Average,
R2, and Vanilla Transformer additions, so that combined_summary.csv / llm_summary.csv /
baseline_summary.csv / window_results.csv reflect the new experiment.

This script does NOT edit chap4_results.tex directly. It prints ready-to-paste
LaTeX snippets for the tables/numbers that change when the underlying data changes,
so you can review each one before pasting it in (recommended, since prose sentences
around each table may also reference specific numbers that need matching updates).

Usage:
    python3 generate_chap4_tables.py --results-dir results/

Outputs (printed to stdout, section by section):
    1. tab:rq1_best        -- Best model per currency pair and forecast horizon (RMSE)
    2. RQ1 aggregate numbers -- mean RMSE/DA per model (for Fig 4.1/4.2 captions/prose)
    3. R^2 summary table (NEW) -- mean R^2 per model, to insert as a new table
    4. Naive / Moving Average / Vanilla Transformer summary -- new baseline rows
    5. tab:stat_tests direction counts -- sanity-check numbers against paired t-test output
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_data(results_dir: Path):
    combined = pd.read_csv(results_dir / "combined_summary.csv")
    window = pd.read_csv(results_dir / "window_results.csv")
    combined.columns = [c.strip().lower() for c in combined.columns]
    window.columns = [c.strip().lower() for c in window.columns]
    return combined, window


def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. tab:rq1_best -- best model per (currency, horizon) by RMSE
# ---------------------------------------------------------------------------

def gen_rq1_best_table(combined: pd.DataFrame):
    section("1. tab:rq1_best -- Best Model per Currency Pair and Forecast Horizon")

    rows = []
    for (cur, hor), grp in combined.groupby(["currency", "horizon"]):
        best = grp.loc[grp["rmse"].idxmin()]
        cfg = f" ({best['config']})" if isinstance(best.get("config"), str) and best["config"] not in ("baseline", "nan") else ""
        rows.append({
            "currency": cur, "horizon": hor,
            "model_cfg": f"{best['model']}{cfg}",
            "rmse": best["rmse"], "mae": best["mae"],
            "mape": best["mape"], "da": best["da"],
        })

    rows.sort(key=lambda r: (r["currency"], r["horizon"]))

    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Best Model per Currency Pair and Forecast Horizon (by RMSE)}")
    print(r"  \label{tab:rq1_best}")
    print(r"  \small")
    print(r"  \begin{tabular}{lllrrrr}")
    print(r"    \toprule")
    print(r"    \textbf{Currency} & \textbf{Horizon} & \textbf{Best Model (Config)} & \textbf{RMSE} & \textbf{MAE} & \textbf{MAPE} & \textbf{DA (\%)} \\")
    print(r"    \midrule")
    prev_currency = None
    for r in rows:
        if prev_currency is not None and r["currency"] != prev_currency:
            print(r"    \midrule")
        cur_display = r["currency"][:3] + "/" + r["currency"][3:] if "/" not in r["currency"] else r["currency"]
        print(f"    {cur_display} & h={r['horizon']:<2} & {r['model_cfg']} & "
              f"{r['rmse']:.4f} & {r['mae']:.4f} & {r['mape']:.4f} & {r['da']:.2f} \\\\")
        prev_currency = r["currency"]
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    print("\n[CHECK] Compare each row's Best Model column against the previous version.")
    print("[CHECK] If Naive/MovingAverage/VanillaTransformer wins any cell, the accompanying")
    print("        prose paragraph discussing 'LLM wins 14/15' etc. must be updated to match.")


# ---------------------------------------------------------------------------
# 2. RQ1 aggregate numbers (mean RMSE / DA per model, across all currency-horizon)
# ---------------------------------------------------------------------------

def gen_rq1_aggregates(combined: pd.DataFrame):
    section("2. RQ1 Aggregate Numbers (mean RMSE / DA per model)")

    llm = combined[combined["type"] == "llm"]
    base = combined[combined["type"] == "baseline"]

    # Best config per LLM model (lowest mean RMSE), matching Fig 4.1 methodology
    print("-- LLM models (best config selected by mean RMSE) --")
    for model, grp in llm.groupby("model"):
        best_cfg_rmse = grp.groupby("config")["rmse"].mean().idxmin()
        sub = grp[grp["config"] == best_cfg_rmse]
        print(f"  {model:15s} best_config={best_cfg_rmse:15s} "
              f"mean_RMSE={sub['rmse'].mean():.2f}  mean_DA={sub['da'].mean():.2f}%")

    print("\n-- Baseline models (mean across all currency/horizon) --")
    for model, grp in base.groupby("model"):
        print(f"  {model:20s} mean_RMSE={grp['rmse'].mean():.2f}  mean_DA={grp['da'].mean():.2f}%")

    print("\n[USE] These numbers feed the prose in Section 4.1 (e.g. 'Qwen2.5-7B achieves the")
    print("      lowest mean RMSE of X, compared to Y for the best baseline') and the")
    print("      Abstract / Chapter 5 Summary of Findings, which currently hardcode 26.57 / 95.64.")


# ---------------------------------------------------------------------------
# 3. R^2 summary table (NEW table, not in the original thesis)
# ---------------------------------------------------------------------------

def gen_r2_table(combined: pd.DataFrame):
    section("3. R^2 Summary Table (NEW -- suggested insertion after Fig 4.1/4.2 in RQ1)")

    if "r2" not in combined.columns:
        print("[SKIP] No 'r2' column found in combined_summary.csv yet.")
        print("       This means run_full_ablation.py has not been re-run with the R2 patch,")
        print("       or the output file passed via --results-dir is the OLD one.")
        return

    llm = combined[combined["type"] == "llm"]
    base = combined[combined["type"] == "baseline"]

    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \caption{Mean $R^2$ by Model (higher is better; 0 = no better than predicting the mean)}")
    print(r"  \label{tab:rq1_r2}")
    print(r"  \begin{tabular}{lr}")
    print(r"    \toprule")
    print(r"    \textbf{Model} & \textbf{Mean $R^2$} \\")
    print(r"    \midrule")

    llm_best_r2 = []
    for model, grp in llm.groupby("model"):
        best_cfg = grp.groupby("config")["rmse"].mean().idxmin()
        sub = grp[grp["config"] == best_cfg]
        llm_best_r2.append((model, sub["r2"].mean()))
    for model, r2 in sorted(llm_best_r2, key=lambda x: -x[1]):
        print(f"    {model} & {r2:.4f} \\\\")

    print(r"    \midrule")
    base_r2 = base.groupby("model")["r2"].mean().sort_values(ascending=False)
    for model, r2 in base_r2.items():
        print(f"    {model} & {r2:.4f} \\\\")

    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")

    print("\n[NOTE] Negative R^2 for a baseline means it performs worse than a naive")
    print("       constant-mean predictor -- worth flagging explicitly in the discussion")
    print("       text if it occurs, since it is a stronger and more specific claim than")
    print("       'baseline DA is near random' from the existing Section 4.1 discussion.")


# ---------------------------------------------------------------------------
# 4. New baseline summary: Naive, Moving Average, Vanilla Transformer
# ---------------------------------------------------------------------------

def gen_new_baselines_summary(combined: pd.DataFrame):
    section("4. New Baselines Summary (Naive, Moving Average, Vanilla Transformer)")

    new_models = ["naive", "moving_average", "vanilla_transformer"]
    base = combined[combined["type"] == "baseline"]
    present = [m for m in new_models if m in base["model"].unique()]
    missing = [m for m in new_models if m not in base["model"].unique()]

    if missing:
        print(f"[WARNING] Not found in combined_summary.csv yet: {missing}")
        print("          -> re-run has not happened, or these models errored out silently.")
        print("          -> check the run_full_ablation.py console log for '<MODEL> ERROR:' lines.")

    if not present:
        print("[SKIP] None of the new baseline models are present yet.")
        return

    for model in present:
        sub = base[base["model"] == model]
        print(f"\n-- {model} --")
        print(f"  Overall mean RMSE: {sub['rmse'].mean():.4f}")
        print(f"  Overall mean MAPE: {sub['mape'].mean():.4f}%")
        print(f"  Overall mean DA:   {sub['da'].mean():.2f}%")
        if "r2" in sub.columns:
            print(f"  Overall mean R^2:  {sub['r2'].mean():.4f}")
        print(f"  Wins (lowest RMSE) in how many of 15 currency-horizon cells:")
        win_count = 0
        for (cur, hor), grp in combined[combined["type"] == "baseline"].groupby(["currency", "horizon"]):
            if grp.loc[grp["rmse"].idxmin(), "model"] == model:
                win_count += 1
        print(f"    {win_count} / 15")

    print("\n[USE] Compare naive/moving_average DA against ~50% -- per the strengthened")
    print("      DA discussion already added to chap4_results.tex, this is the single most")
    print("      important number: does the random-walk baseline also sit near 50% DA,")
    print("      confirming the FX-market-is-hard-to-beat framing, or does it do notably")
    print("      better/worse than the other classical baselines?")
    print("[USE] Compare vanilla_transformer vs iTransformer RMSE/DA directly -- this is the")
    print("      concrete evidence needed to fill the placeholder paragraph in")
    print("      chap3_method.tex Section 3.2 (Vanilla Transformer subsection).")


# ---------------------------------------------------------------------------
# 5. tab:stat_tests sanity-check direction counts (paired t-test inputs)
# ---------------------------------------------------------------------------

def gen_stat_tests_direction_counts(combined: pd.DataFrame):
    section("5. RQ1 Direction Counts (for cross-checking against statistical_tests.py output)")

    llm = combined[combined["type"] == "llm"]
    base = combined[combined["type"] == "baseline"]

    # Best baseline per (currency, horizon)
    best_base = (
        base.loc[base.groupby(["currency", "horizon"])["rmse"].idxmin()]
        [["currency", "horizon", "rmse", "da"]]
        .rename(columns={"rmse": "base_rmse", "da": "base_da"})
    )

    better_rmse, better_da, total = 0, 0, 0
    for model in llm["model"].unique():
        for (cur, hor), grp in llm[llm["model"] == model].groupby(["currency", "horizon"]):
            best_llm_row = grp.loc[grp["rmse"].idxmin()]
            match = best_base[(best_base["currency"] == cur) & (best_base["horizon"] == hor)]
            if match.empty:
                continue
            total += 1
            if best_llm_row["rmse"] < match.iloc[0]["base_rmse"]:
                better_rmse += 1
            if best_llm_row["da"] > match.iloc[0]["base_da"]:
                better_da += 1

    print(f"  LLM better on RMSE: {better_rmse} / {total} (model, currency, horizon) triplets")
    print(f"  LLM better on DA:   {better_da} / {total} (model, currency, horizon) triplets")
    print("\n[NOTE] These are simple win-counts, NOT the paired t-test itself.")
    print("       Re-run statistical_tests.py separately on the new window_results.csv")
    print("       to get updated p-values for Table tab:stat_tests; this function is only")
    print("       a quick sanity check that the direction counts move in a sensible way.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results",
                     help="Directory containing combined_summary.csv and window_results.csv")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    combined, window = load_data(results_dir)

    print(f"Loaded combined_summary.csv: {len(combined)} rows")
    print(f"Loaded window_results.csv:   {len(window)} rows")
    print(f"Models present: {sorted(combined['model'].unique())}")

    gen_rq1_best_table(combined)
    gen_rq1_aggregates(combined)
    gen_r2_table(combined)
    gen_new_baselines_summary(combined)
    gen_stat_tests_direction_counts(combined)

    print("\n" + "=" * 78)
    print("DONE. Review each section above, then manually paste the relevant LaTeX")
    print("snippets into chap4_results.tex, updating surrounding prose numbers to match.")
    print("Remember to also re-run statistical_tests.py and dm_test.py separately for")
    print("updated p-values, and rq_horizon_breakdown (Section 4.2/4.3) if horizon-level")
    print("tables were added.")
    print("=" * 78)


if __name__ == "__main__":
    main()