#!/usr/bin/env python3
"""
statistical_tests_extended.py — Paired tests for all four RQs.
Unit of observation: (currency, horizon, llm_model) — 45 pairs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

def fmt(p: float) -> str:
    if np.isnan(p): return "n/a"
    if p < 0.001:   return "p<0.001"
    return f"p={p:.3f}"

def paired_test(a, b, label, results_list, rq, metric):
    a, b = np.array(a, float), np.array(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    d = a - b
    n = len(d)
    mean_d = float(np.mean(d))
    t_res = ttest_rel(a, b)
    try:
        w_res = wilcoxon(d)
        w_p = float(w_res.pvalue)
    except Exception:
        w_p = float("nan")
    sig_t = " *" if t_res.pvalue < 0.05 else ""
    sig_w = " *" if w_p < 0.05 else ""
    print(f"  {label}")
    print(f"    n={n}  mean_delta={mean_d:+.4f}")
    print(f"    paired t-test: {fmt(t_res.pvalue)}{sig_t}   Wilcoxon: {fmt(w_p)}{sig_w}")
    results_list.append({
        "rq": rq, "metric": metric, "n": n, "mean_delta": mean_d,
        "t_p": float(t_res.pvalue), "w_p": w_p,
        "sig_t": t_res.pvalue < 0.05, "sig_w": w_p < 0.05
    })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/combined_summary.csv")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = [c.strip().lower() for c in df.columns]
    llm  = df[df["type"] == "llm"].copy()
    base = df[df["type"] == "baseline"].copy()
    results = []

    # ── RQ1: LLM best config vs best baseline ─────────────────────────────────
    print("\n" + "="*70)
    print("RQ1: LLM (best config) vs Best Traditional Baseline")
    print("Unit: (currency, horizon, llm_model) — 45 pairs")
    print("="*70)

    llm_best = llm.loc[llm.groupby(["model","currency","horizon"])["rmse"].idxmin()]
    base_best = base.loc[base.groupby(["currency","horizon"])["rmse"].idxmin(),
                         ["currency","horizon","rmse","da"]]\
                    .rename(columns={"rmse":"base_rmse","da":"base_da"})

    m1 = llm_best.merge(base_best, on=["currency","horizon"], how="inner")
    paired_test(m1["rmse"], m1["base_rmse"],
        "RMSE: LLM vs best baseline  (negative = LLM better)",
        results, "RQ1", "RMSE")
    paired_test(m1["da"], m1["base_da"],
        "DA:   LLM vs best baseline  (positive = LLM better)",
        results, "RQ1", "DA")

    # ── RQ2: News vs No-News ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("RQ2: News Augmentation Effect")
    print("Unit: (model, currency, horizon) — 45 pairs")
    print("="*70)

    news_map = {"baseline_llm":"no_news","prompt_only":"no_news",
                "news_only":"news","full_model":"news"}
    llm["news_label"] = llm["config"].map(news_map)
    rq2 = llm.groupby(["model","currency","horizon","news_label"])[["rmse","da"]]\
              .mean().unstack("news_label")
    rq2.columns = ["_".join(c) for c in rq2.columns]
    rq2 = rq2.dropna().reset_index()

    paired_test(rq2["rmse_news"], rq2["rmse_no_news"],
        "RMSE: news vs no-news  (negative = news better)",
        results, "RQ2", "RMSE")
    paired_test(rq2["da_news"], rq2["da_no_news"],
        "DA:   news vs no-news  (positive = news better)",
        results, "RQ2", "DA")

    # ── RQ3: Few-shot vs Zero-shot ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("RQ3: Prompting Strategy Effect")
    print("Unit: (model, currency, horizon) — 45 pairs")
    print("="*70)

    prompt_map = {"baseline_llm":"zero","news_only":"zero",
                  "prompt_only":"few","full_model":"few"}
    llm["prompt_label"] = llm["config"].map(prompt_map)
    rq3 = llm.groupby(["model","currency","horizon","prompt_label"])[["rmse","da"]]\
              .mean().unstack("prompt_label")
    rq3.columns = ["_".join(c) for c in rq3.columns]
    rq3 = rq3.dropna().reset_index()

    paired_test(rq3["rmse_few"], rq3["rmse_zero"],
        "RMSE: few-shot vs zero-shot  (negative = few-shot better)",
        results, "RQ3", "RMSE")
    paired_test(rq3["da_few"], rq3["da_zero"],
        "DA:   few-shot vs zero-shot  (positive = few-shot better)",
        results, "RQ3", "DA")

    # ── RQ4: Interaction (news effect under few-shot vs zero-shot) on DA ───────
    print("\n" + "="*70)
    print("RQ4: Interaction Effect (news × prompt) on DA")
    print("Unit: (model, currency, horizon) — 45 pairs")
    print("="*70)

    key = ["model","currency","horizon"]
    few_news   = llm[llm["config"]=="full_model"].set_index(key)["da"]
    few_nonews = llm[llm["config"]=="prompt_only"].set_index(key)["da"]
    zero_news  = llm[llm["config"]=="news_only"].set_index(key)["da"]
    zero_nonews= llm[llm["config"]=="baseline_llm"].set_index(key)["da"]

    idx = few_news.index.intersection(few_nonews.index)\
                        .intersection(zero_news.index)\
                        .intersection(zero_nonews.index)

    news_eff_few  = (few_news.loc[idx]  - few_nonews.loc[idx]).values
    news_eff_zero = (zero_news.loc[idx] - zero_nonews.loc[idx]).values

    print(f"  Mean news effect under few-shot:  {news_eff_few.mean():+.4f}")
    print(f"  Mean news effect under zero-shot: {news_eff_zero.mean():+.4f}")
    paired_test(news_eff_few, news_eff_zero,
        "DA interaction: news_effect_fewshot vs news_effect_zeroshot\n"
        "    (positive = news helps more under few-shot than zero-shot, consistent with disordinal interaction)",
        results, "RQ4", "DA_interaction")

    # ── Save & summary ─────────────────────────────────────────────────────────
    out = pd.DataFrame(results)
    outpath = Path(args.csv).parent / "statistical_test_results.csv"
    out.to_csv(outpath, index=False)
    print(f"\n[OK] Saved to {outpath}")
    print("\n── Summary ──")
    print(out[["rq","metric","n","mean_delta","t_p","w_p","sig_t","sig_w"]].to_string(index=False))

if __name__ == "__main__":
    main()