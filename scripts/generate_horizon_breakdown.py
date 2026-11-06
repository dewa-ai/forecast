#!/usr/bin/env python3
"""
generate_horizon_breakdown.py — Per-horizon DM test tables for RQ2/RQ3.

Addresses committee feedback: "results per horizon? where are the h=1,5,10 tables
for RQ2-4?" RQ1 already has a per-currency-per-horizon table (tab:rq1_best); this
script produces the equivalent horizon-level breakdown for RQ2 (news effect) and
RQ3 (prompting strategy effect), which the thesis originally only reported as
pooled statistics across all three horizons (n=45).

Run this any time window_results.csv changes -- e.g. after re-running
run_full_ablation.py with new baselines, R2, or any other modification that
regenerates the LLM window-level results.

Requires dm_test.py (same folder) for the dm_test() function.

Usage:
    python3 generate_horizon_breakdown.py --csv ../results/window_results.csv

Output:
    Prints two ready-to-paste LaTeX tables (RQ2 and RQ4 optional / RQ3) to stdout,
    plus a plain-text summary. Does not edit chap4_results.tex directly -- paste
    the tables into Table~\ref{tab:rq2_horizon} / Table~\ref{tab:rq3_horizon} in
    chap4_results.tex, replacing the existing ones, and re-check the prose
    paragraph immediately below each table still matches the new numbers.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from dm_test import dm_test  # noqa: E402


NEWS_CFG = ["news_only", "full_model"]
NONEWS_CFG = ["baseline_llm", "prompt_only"]
FEW_CFG = ["prompt_only", "full_model"]
ZERO_CFG = ["baseline_llm", "news_only"]


def compute_rq2_horizon(llm: pd.DataFrame, horizon: int, metric_col: str, negate: bool = False):
    """RQ2: news vs no-news, for a given horizon and metric column (window_rmse or window_da)."""
    llm_h = llm[llm["horizon"] == horizon]
    dm_vals = []
    for (mdl, cur), grp in llm_h.groupby(["model", "currency"]):
        news_ser = grp[grp["config"].isin(NEWS_CFG)].sort_values("window")[metric_col].values
        nonews_ser = grp[grp["config"].isin(NONEWS_CFG)].sort_values("window")[metric_col].values
        n = min(len(news_ser), len(nonews_ser))
        if n < 2:
            continue
        if metric_col == "window_rmse":
            a, b = news_ser[:n] ** 2, nonews_ser[:n] ** 2
        else:  # window_da -- higher is better, negate so "positive d" still means "worse with news"
            a, b = -news_ser[:n], -nonews_ser[:n]
        res = dm_test(a, b, h=horizon)
        dm_vals.append({**res, "model": mdl, "currency": cur})
    return pd.DataFrame(dm_vals) if dm_vals else pd.DataFrame()


def compute_rq3_horizon(llm: pd.DataFrame, horizon: int, metric_col: str):
    """RQ3: few-shot vs zero-shot, for a given horizon and metric column."""
    llm_h = llm[llm["horizon"] == horizon]
    dm_vals = []
    for (mdl, cur), grp in llm_h.groupby(["model", "currency"]):
        few_ser = grp[grp["config"].isin(FEW_CFG)].sort_values("window")[metric_col].values
        zero_ser = grp[grp["config"].isin(ZERO_CFG)].sort_values("window")[metric_col].values
        n = min(len(few_ser), len(zero_ser))
        if n < 2:
            continue
        if metric_col == "window_rmse":
            a, b = few_ser[:n] ** 2, zero_ser[:n] ** 2
        else:
            a, b = -few_ser[:n], -zero_ser[:n]
        res = dm_test(a, b, h=horizon)
        dm_vals.append({**res, "model": mdl, "currency": cur})
    return pd.DataFrame(dm_vals) if dm_vals else pd.DataFrame()


def compute_rq4_horizon(llm: pd.DataFrame, horizon: int):
    """RQ4: interaction (news effect under few-shot vs under zero-shot), per horizon, on DA."""
    llm_h = llm[llm["horizon"] == horizon]
    dm_vals = []
    for (mdl, cur), grp in llm_h.groupby(["model", "currency"]):
        def get(cfg):
            return grp[grp["config"] == cfg].sort_values("window")["window_da"].values

        few_news, few_nonews = get("full_model"), get("prompt_only")
        zero_news, zero_nonews = get("news_only"), get("baseline_llm")
        n = min(len(few_news), len(few_nonews), len(zero_news), len(zero_nonews))
        if n < 2:
            continue
        eff_few = few_news[:n] - few_nonews[:n]
        eff_zero = zero_news[:n] - zero_nonews[:n]
        res = dm_test(-eff_few, -eff_zero, h=horizon)
        dm_vals.append({**res, "model": mdl, "currency": cur,
                         "mean_eff_few": float(eff_few.mean()),
                         "mean_eff_zero": float(eff_zero.mean())})
    return pd.DataFrame(dm_vals) if dm_vals else pd.DataFrame()


def summarize(dm_df: pd.DataFrame, direction_label: str, direction_positive: bool):
    """Return (n_pairs, n_sig, direction_count) from a DM result dataframe."""
    if dm_df.empty:
        return 0, 0, 0
    sig = dm_df["p"] < 0.05
    if direction_positive:
        direction_count = int((dm_df["mean_d"] > 0).sum())
    else:
        direction_count = int((dm_df["mean_d"] < 0).sum())
    return len(dm_df), int(sig.sum()), direction_count


def print_rq2_table(llm: pd.DataFrame, horizons: list[int]):
    print(r"\begin{table}[H]")
    print(r"  \centering")
    print(r"  \caption{RQ2 (News Effect) Diebold--Mariano Results by Forecast Horizon}")
    print(r"  \label{tab:rq2_horizon}")
    print(r"  \begin{tabular}{lrrrr}")
    print(r"    \toprule")
    print(r"    \textbf{Horizon} & \textbf{Metric} & \textbf{$n$} & \textbf{$n_{\text{sig}}$} & \textbf{Direction (news worse)} \\")
    print(r"    \midrule")
    rows = []
    for h in horizons:
        dm_rmse = compute_rq2_horizon(llm, h, "window_rmse")
        n, nsig, dirc = summarize(dm_rmse, "worse", direction_positive=True)
        rows.append((h, "RMSE", n, nsig, dirc))
        dm_da = compute_rq2_horizon(llm, h, "window_da")
        n, nsig, dirc = summarize(dm_da, "worse", direction_positive=True)
        rows.append((h, "DA", n, nsig, dirc))
    for h, metric, n, nsig, dirc in rows:
        total = n
        print(f"    $h={h}${' ' if h!=10 else ''} & {metric} & {n} & {nsig} & {dirc}/{total} \\\\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    return rows


def print_rq3_table(llm: pd.DataFrame, horizons: list[int]):
    print(r"\begin{table}[H]")
    print(r"  \centering")
    print(r"  \caption{RQ3 (Prompting Strategy Effect) Diebold--Mariano Results by Forecast Horizon}")
    print(r"  \label{tab:rq3_horizon}")
    print(r"  \begin{tabular}{lrrrr}")
    print(r"    \toprule")
    print(r"    \textbf{Horizon} & \textbf{Metric} & \textbf{$n$} & \textbf{$n_{\text{sig}}$} & \textbf{Direction (few-shot better)} \\")
    print(r"    \midrule")
    rows = []
    for h in horizons:
        dm_rmse = compute_rq3_horizon(llm, h, "window_rmse")
        n, nsig, dirc = summarize(dm_rmse, "better", direction_positive=False)
        rows.append((h, "RMSE", n, nsig, dirc))
        dm_da = compute_rq3_horizon(llm, h, "window_da")
        n, nsig, dirc = summarize(dm_da, "better", direction_positive=False)
        rows.append((h, "DA", n, nsig, dirc))
    for h, metric, n, nsig, dirc in rows:
        total = n
        print(f"    $h={h}$ & {metric} & {n} & {nsig} & {dirc}/{total} \\\\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    return rows


def print_rq4_optional(llm: pd.DataFrame, horizons: list[int]):
    print("\n-- RQ4 per-horizon (optional; pooled result was already only 1/45 significant) --")
    for h in horizons:
        dm_df = compute_rq4_horizon(llm, h)
        if dm_df.empty:
            print(f"  h={h}: no valid pairs")
            continue
        sig = dm_df["p"] < 0.05
        print(f"  h={h}: n={len(dm_df)}, n_sig={int(sig.sum())}, "
              f"mean_eff_few={dm_df['mean_eff_few'].mean():+.4f}, "
              f"mean_eff_zero={dm_df['mean_eff_zero'].mean():+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../results/window_results.csv")
    ap.add_argument("--include-rq4", action="store_true",
                     help="Also print an optional RQ4 per-horizon breakdown (not a LaTeX table)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    llm = df[df["type"] == "llm"].copy()
    horizons = sorted(llm["horizon"].unique().tolist())

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"LLM rows: {len(llm)}, horizons found: {horizons}")

    print("\n" + "=" * 78)
    print("RQ2 Per-Horizon Table (paste into chap4_results.tex, replacing tab:rq2_horizon)")
    print("=" * 78)
    print_rq2_table(llm, horizons)

    print("\n" + "=" * 78)
    print("RQ3 Per-Horizon Table (paste into chap4_results.tex, replacing tab:rq3_horizon)")
    print("=" * 78)
    print_rq3_table(llm, horizons)

    if args.include_rq4:
        print("\n" + "=" * 78)
        print("RQ4 Per-Horizon (optional, plain-text only)")
        print("=" * 78)
        print_rq4_optional(llm, horizons)

    print("\n" + "=" * 78)
    print("DONE. After pasting tables, re-read the prose paragraph immediately below")
    print("each table (discussing which horizon has strongest/weakest significance)")
    print("to confirm it still matches the new numbers -- the pattern may shift once")
    print("new baselines/R2 change nothing here (RQ2/RQ3 only involve LLM configs),")
    print("but a full re-run with new random LLM sampling can still shift these numbers.")
    print("=" * 78)


if __name__ == "__main__":
    main()