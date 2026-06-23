#!/usr/bin/env python3
"""
dm_test.py — Diebold-Mariano test for all four RQs.

Reads window_results.csv produced by run_full_ablation.py.
Each row is one window-level RMSE observation for a (type, model, config, currency, horizon).

DM test:
    d_t = MSE_A_t - MSE_B_t  (squared error of window-level RMSE as proxy)
    DM  = mean(d) / SE(d)    (t-distributed under H0 with T-1 df)

Harvey-Leybourne-Newbold (1997) small-sample correction is applied.

Usage:
    python3 scripts/dm_test.py --csv results/window_results.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict:
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction.

    Args:
        loss_a, loss_b: per-window loss series (e.g. squared RMSE per window)
        h: forecast horizon (for HLN correction)

    Returns dict with DM statistic, p-value, mean loss differential.
    """
    d  = loss_a - loss_b
    T  = len(d)
    if T < 2:
        return {"dm": np.nan, "p": np.nan, "mean_d": np.nan, "n": T}

    mean_d = float(np.mean(d))
    var_d  = float(np.var(d, ddof=1))
    if var_d == 0:
        return {"dm": np.nan, "p": np.nan, "mean_d": mean_d, "n": T}

    # Standard DM statistic
    dm = mean_d / np.sqrt(var_d / T)

    # HLN correction factor
    correction = np.sqrt(
        (T + 1 - 2 * h + max(0, h * (h - 1) / T)) / T
    )
    dm_hln = dm * correction
    df     = T - 1
    p      = float(2 * t_dist.sf(abs(dm_hln), df=df))

    return {
        "dm_hln": float(dm_hln), "p": p, "mean_d": mean_d,
        "n": T, "df": df,
    }


def fmt(p: float) -> str:
    if np.isnan(p): return "n/a"
    if p < 0.001:   return "p<0.001"
    return f"p={p:.3f}"


def run_rq1(df: pd.DataFrame, results: list):
    """RQ1: Best LLM config vs best baseline, per (currency, horizon, llm_model)."""
    print("\n" + "=" * 70)
    print("RQ1: LLM (best config per model) vs Best Traditional Baseline")
    print("=" * 70)

    llm  = df[df["type"] == "llm"].copy()
    base = df[df["type"] == "baseline"].copy()

    # For each (model, currency, horizon), select best config by mean window_rmse
    best_config = (
        llm.groupby(["model", "currency", "horizon", "config"])["window_rmse"]
        .mean().reset_index()
        .loc[lambda x: x.groupby(["model", "currency", "horizon"])["window_rmse"].transform("min") == x["window_rmse"]]
        .drop_duplicates(["model", "currency", "horizon"])
        [["model", "currency", "horizon", "config"]]
    )

    # Best baseline per (currency, horizon) by mean window_rmse
    best_base_cfg = (
        base.groupby(["currency", "horizon", "model"])["window_rmse"]
        .mean().reset_index()
        .loc[lambda x: x.groupby(["currency", "horizon"])["window_rmse"].transform("min") == x["window_rmse"]]
        .drop_duplicates(["currency", "horizon"])
        .rename(columns={"model": "base_model"})
        [["currency", "horizon", "base_model"]]
    )

    dm_vals = []
    for _, row in best_config.iterrows():
        llm_series = llm[
            (llm["model"]    == row["model"]) &
            (llm["currency"] == row["currency"]) &
            (llm["horizon"]  == row["horizon"]) &
            (llm["config"]   == row["config"])
        ].sort_values("window")["window_rmse"].values

        base_row = best_base_cfg[
            (best_base_cfg["currency"] == row["currency"]) &
            (best_base_cfg["horizon"]  == row["horizon"])
        ]
        if base_row.empty:
            continue
        base_series = base[
            (base["currency"] == row["currency"]) &
            (base["horizon"]  == row["horizon"]) &
            (base["model"]    == base_row.iloc[0]["base_model"])
        ].sort_values("window")["window_rmse"].values

        n = min(len(llm_series), len(base_series))
        if n < 2:
            continue
        res = dm_test(llm_series[:n] ** 2, base_series[:n] ** 2, h=int(row["horizon"]))
        dm_vals.append({**res,
                         "model": row["model"], "currency": row["currency"],
                         "horizon": row["horizon"]})

    if dm_vals:
        dm_df = pd.DataFrame(dm_vals)
        sig = dm_df["p"] < 0.05
        print(f"  Pairs tested: {len(dm_df)}")
        print(f"  Significant (p<0.05): {sig.sum()} / {len(dm_df)}")
        print(f"  Mean DM (HLN): {dm_df['dm_hln'].mean():.3f}")
        print(f"  Direction (negative = LLM better): {(dm_df['mean_d'] < 0).sum()} of {len(dm_df)} pairs")
        results.append({
            "rq": "RQ1", "comparison": "LLM_best_vs_best_baseline", "metric": "RMSE",
            "n_pairs": len(dm_df), "n_sig": int(sig.sum()),
            "mean_dm": float(dm_df["dm_hln"].mean()),
            "mean_loss_diff": float(dm_df["mean_d"].mean()),
        })
        print(dm_df[["model","currency","horizon","n","mean_d","dm_hln","p"]].to_string(index=False))


def run_rq2(df: pd.DataFrame, results: list):
    """RQ2: News vs No-News for each (model, currency, horizon)."""
    print("\n" + "=" * 70)
    print("RQ2: News Augmentation Effect — DM test per (model, currency, horizon)")
    print("=" * 70)

    llm = df[df["type"] == "llm"].copy()
    news_cfg   = ["news_only", "full_model"]
    nonews_cfg = ["baseline_llm", "prompt_only"]

    dm_vals = []
    for (mdl, cur, hor), grp in llm.groupby(["model", "currency", "horizon"]):
        news_ser   = grp[grp["config"].isin(news_cfg)].sort_values("window")["window_rmse"].values
        nonews_ser = grp[grp["config"].isin(nonews_cfg)].sort_values("window")["window_rmse"].values
        n = min(len(news_ser), len(nonews_ser))
        if n < 2:
            continue
        res = dm_test(news_ser[:n] ** 2, nonews_ser[:n] ** 2, h=int(hor))
        dm_vals.append({**res, "model": mdl, "currency": cur, "horizon": hor})

    if dm_vals:
        dm_df = pd.DataFrame(dm_vals)
        sig = dm_df["p"] < 0.05
        print(f"  Pairs tested: {len(dm_df)}")
        print(f"  Significant (p<0.05): {sig.sum()} / {len(dm_df)}")
        print(f"  Direction (positive = news worse): {(dm_df['mean_d'] > 0).sum()} of {len(dm_df)} pairs")
        results.append({
            "rq": "RQ2", "comparison": "news_vs_nonews", "metric": "RMSE",
            "n_pairs": len(dm_df), "n_sig": int(sig.sum()),
            "mean_dm": float(dm_df["dm_hln"].mean()),
            "mean_loss_diff": float(dm_df["mean_d"].mean()),
        })


def run_rq3(df: pd.DataFrame, results: list):
    """RQ3: Few-shot vs Zero-shot for each (model, currency, horizon)."""
    print("\n" + "=" * 70)
    print("RQ3: Prompting Strategy — DM test per (model, currency, horizon)")
    print("=" * 70)

    llm = df[df["type"] == "llm"].copy()
    few_cfg  = ["prompt_only", "full_model"]
    zero_cfg = ["baseline_llm", "news_only"]

    dm_vals = []
    for (mdl, cur, hor), grp in llm.groupby(["model", "currency", "horizon"]):
        few_ser  = grp[grp["config"].isin(few_cfg)].sort_values("window")["window_rmse"].values
        zero_ser = grp[grp["config"].isin(zero_cfg)].sort_values("window")["window_rmse"].values
        n = min(len(few_ser), len(zero_ser))
        if n < 2:
            continue
        res = dm_test(few_ser[:n] ** 2, zero_ser[:n] ** 2, h=int(hor))
        dm_vals.append({**res, "model": mdl, "currency": cur, "horizon": hor})

    if dm_vals:
        dm_df = pd.DataFrame(dm_vals)
        sig = dm_df["p"] < 0.05
        print(f"  Pairs tested: {len(dm_df)}")
        print(f"  Significant (p<0.05): {sig.sum()} / {len(dm_df)}")
        print(f"  Direction (negative = few-shot better): {(dm_df['mean_d'] < 0).sum()} of {len(dm_df)} pairs")
        results.append({
            "rq": "RQ3", "comparison": "fewshot_vs_zeroshot", "metric": "RMSE",
            "n_pairs": len(dm_df), "n_sig": int(sig.sum()),
            "mean_dm": float(dm_df["dm_hln"].mean()),
            "mean_loss_diff": float(dm_df["mean_d"].mean()),
        })


def run_rq4(df: pd.DataFrame, results: list):
    """RQ4: Interaction — DM test on news effect difference under few-shot vs zero-shot."""
    print("\n" + "=" * 70)
    print("RQ4: Interaction Effect (news x prompt) — DM test on DA differentials")
    print("=" * 70)

    llm = df[df["type"] == "llm"].copy()

    dm_vals = []
    for (mdl, cur, hor), grp in llm.groupby(["model", "currency", "horizon"]):
        def get(cfg):
            return grp[grp["config"] == cfg].sort_values("window")["window_da"].values

        few_news    = get("full_model")
        few_nonews  = get("prompt_only")
        zero_news   = get("news_only")
        zero_nonews = get("baseline_llm")

        n = min(len(few_news), len(few_nonews), len(zero_news), len(zero_nonews))
        if n < 2:
            continue

        # News effect under each prompt strategy
        eff_few  = few_news[:n]  - few_nonews[:n]    # positive = news helps few-shot
        eff_zero = zero_news[:n] - zero_nonews[:n]   # positive = news helps zero-shot

        # DM test: is news effect significantly different between few-shot and zero-shot?
        # Using eff_few as "model A" and eff_zero as "model B" (DA: higher is better,
        # so we negate to align with loss framework)
        res = dm_test(-eff_few, -eff_zero, h=int(hor))
        dm_vals.append({**res, "model": mdl, "currency": cur, "horizon": hor,
                         "mean_eff_few": float(eff_few.mean()),
                         "mean_eff_zero": float(eff_zero.mean())})

    if dm_vals:
        dm_df = pd.DataFrame(dm_vals)
        sig = dm_df["p"] < 0.05
        print(f"  Pairs tested: {len(dm_df)}")
        print(f"  Significant interaction (p<0.05): {sig.sum()} / {len(dm_df)}")
        print(f"  Mean news effect under few-shot:  {dm_df['mean_eff_few'].mean():+.4f}")
        print(f"  Mean news effect under zero-shot: {dm_df['mean_eff_zero'].mean():+.4f}")
        results.append({
            "rq": "RQ4", "comparison": "news_effect_fewshot_vs_zeroshot", "metric": "DA",
            "n_pairs": len(dm_df), "n_sig": int(sig.sum()),
            "mean_dm": float(dm_df["dm_hln"].mean()),
            "mean_loss_diff": float(dm_df["mean_d"].mean()),
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/window_results.csv")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Types: {df['type'].value_counts().to_dict()}")

    results = []
    run_rq1(df, results)
    run_rq2(df, results)
    run_rq3(df, results)
    run_rq4(df, results)

    out = pd.DataFrame(results)
    outpath = csv_path.parent / "dm_test_results.csv"
    out.to_csv(outpath, index=False)
    print(f"\n[OK] DM test results saved to {outpath}")
    print("\n── Summary ──")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()