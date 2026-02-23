#!/usr/bin/env python3
"""
analyze_results.py
- Loads results CSVs from a folder (default: results/)
- Prints leaderboard summaries (overall and per-currency/horizon)
- Finds best configs and deltas vs baseline LLM
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd


def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}. Found: {list(df.columns)}")


def _load_csv(path: Path, name: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = _lower_cols(df)

    # Normalize a few common column name variations
    rename_map = {
        "pair": "currency",
        "fx_pair": "currency",
        "symbol": "currency",
        "h": "horizon",
        "horizon_days": "horizon",
        "directional_accuracy": "da",
        "dir_acc": "da",
        "acc": "da",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # If "config" doesn't exist, create it for baseline rows
    if "config" not in df.columns:
        df["config"] = "default"

    # If "model" missing but there is "llm" / "baseline_model", try map
    if "model" not in df.columns:
        if "baseline_model" in df.columns:
            df["model"] = df["baseline_model"]
        elif "llm" in df.columns:
            df["model"] = df["llm"]

    # Coerce types
    for col in ["rmse", "da"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Horizon could be like "short/medium/long" or int days; keep both
    if "horizon" in df.columns:
        df["horizon"] = df["horizon"].astype(str)

    return df


def _leaderboard(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    """Return best rows per group based on metric (rmse min, da max)."""
    _require_cols(df, group_cols + ["model", "config", metric], "leaderboard")
    df2 = df.dropna(subset=[metric]).copy()

    if metric == "rmse":
        idx = df2.groupby(group_cols)[metric].idxmin()
    else:
        idx = df2.groupby(group_cols)[metric].idxmax()

    best = df2.loc[idx].sort_values(group_cols).reset_index(drop=True)
    return best


def _print_top(df: pd.DataFrame, title: str, metric: str, n: int = 10) -> None:
    df2 = df.dropna(subset=[metric]).copy()
    if df2.empty:
        print(f"\n{title}\n(no rows)\n")
        return

    if metric == "rmse":
        df2 = df2.sort_values(metric, ascending=True)
    else:
        df2 = df2.sort_values(metric, ascending=False)

    cols_show = [c for c in ["currency", "horizon", "model", "config", "rmse", "da"] if c in df2.columns]
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(df2[cols_show].head(n).to_string(index=False))


def _delta_vs_baseline_llm(df_llm: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute deltas of each config vs baseline_llm within each (currency,horizon,model).
    Expects config values like baseline_llm/news_only/prompt_only/full_model.
    """
    needed = ["currency", "horizon", "model", "config", "rmse", "da"]
    for c in needed:
        if c not in df_llm.columns:
            return None

    base = df_llm[df_llm["config"].astype(str).str.lower().eq("baseline_llm")].copy()
    if base.empty:
        return None

    base = base.rename(columns={"rmse": "rmse_base", "da": "da_base"})[
        ["currency", "horizon", "model", "rmse_base", "da_base"]
    ]

    merged = df_llm.merge(base, on=["currency", "horizon", "model"], how="left")
    merged["rmse_delta"] = merged["rmse"] - merged["rmse_base"]  # negative is better
    merged["da_delta"] = merged["da"] - merged["da_base"]        # positive is better
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="results", help="Folder containing summary CSVs")
    ap.add_argument("--topk", type=int, default=10, help="Top-K rows to show in leaderboards")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    if not rdir.exists():
        raise FileNotFoundError(f"Results directory not found: {rdir.resolve()}")

    baseline = _load_csv(rdir / "baseline_summary.csv", "baseline_summary.csv")
    llm = _load_csv(rdir / "llm_summary.csv", "llm_summary.csv")
    combined = _load_csv(rdir / "combined_summary.csv", "combined_summary.csv")

    print(f"\nLoaded results from: {rdir.resolve()}")
    for name, df in [("baseline_summary.csv", baseline), ("llm_summary.csv", llm), ("combined_summary.csv", combined)]:
        if df is None:
            print(f"  - {name}: (missing)")
        else:
            print(f"  - {name}: {len(df)} rows, {len(df.columns)} cols")

    # Basic top lists
    if combined is not None:
        _print_top(combined, "TOP overall by RMSE (lower is better)", "rmse", n=args.topk)
        _print_top(combined, "TOP overall by DA (higher is better)", "da", n=args.topk)

        # Best per (currency,horizon)
        best_rmse = _leaderboard(combined, ["currency", "horizon"], metric="rmse")
        best_da = _leaderboard(combined, ["currency", "horizon"], metric="da")

        print("\n" + "=" * 72)
        print("BEST per (currency,horizon) by RMSE")
        print("=" * 72)
        cols_show = [c for c in ["currency", "horizon", "model", "config", "rmse", "da"] if c in best_rmse.columns]
        print(best_rmse[cols_show].to_string(index=False))

        print("\n" + "=" * 72)
        print("BEST per (currency,horizon) by DA")
        print("=" * 72)
        cols_show = [c for c in ["currency", "horizon", "model", "config", "rmse", "da"] if c in best_da.columns]
        print(best_da[cols_show].to_string(index=False))

    # LLM delta analysis
    if llm is not None:
        deltas = _delta_vs_baseline_llm(llm)
        if deltas is None:
            print("\n[LLM] Could not compute deltas vs baseline_llm (missing columns or baseline_llm config).")
        else:
            # Summarize: mean delta per config across all pairs/horizons/models
            grp = deltas.groupby(["model", "config"], dropna=False).agg(
                mean_rmse_delta=("rmse_delta", "mean"),
                median_rmse_delta=("rmse_delta", "median"),
                mean_da_delta=("da_delta", "mean"),
                median_da_delta=("da_delta", "median"),
                n=("rmse_delta", "count"),
            ).reset_index()

            print("\n" + "=" * 72)
            print("LLM DELTAS vs baseline_llm (rmse_delta < 0 is better, da_delta > 0 is better)")
            print("=" * 72)
            print(grp.sort_values(["model", "mean_rmse_delta"]).to_string(index=False))

            # Show best config per model by mean RMSE delta
            best_cfg = grp.sort_values(["model", "mean_rmse_delta"]).groupby("model").head(1)
            print("\nBest LLM config per model (by mean RMSE improvement):")
            print(best_cfg[["model", "config", "mean_rmse_delta", "mean_da_delta", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()