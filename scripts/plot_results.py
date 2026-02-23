#!/usr/bin/env python3
"""
plot_results.py
- Generates simple plots from combined_summary.csv and llm_summary.csv
- Outputs PNG files into results/plots (default)
Notes:
- Uses matplotlib only (no seaborn).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = _lower_cols(df)
    rename_map = {
        "pair": "currency",
        "fx_pair": "currency",
        "horizon_days": "horizon",
        "directional_accuracy": "da",
        "dir_acc": "da",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "config" not in df.columns:
        df["config"] = "default"
    if "model" not in df.columns:
        if "baseline_model" in df.columns:
            df["model"] = df["baseline_model"]
        elif "llm" in df.columns:
            df["model"] = df["llm"]
    for col in ["rmse", "da"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "horizon" in df.columns:
        df["horizon"] = df["horizon"].astype(str)
    return df


def _savefig(outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_overall_model_rmse(combined: pd.DataFrame, outdir: Path) -> None:
    df = combined.dropna(subset=["rmse"]).copy()
    # average RMSE across all currencies/horizons per (model,config)
    agg = df.groupby(["model", "config"], dropna=False)["rmse"].mean().reset_index()
    # pivot for simple bar plot per model with configs as separate bars
    pivot = agg.pivot(index="model", columns="config", values="rmse").sort_index()

    plt.figure()
    pivot.plot(kind="bar")  # default colors
    plt.xlabel("Model")
    plt.ylabel("Mean RMSE")
    plt.title("Mean RMSE by Model/Config (overall)")
    plt.legend(title="Config", fontsize=8)
    _savefig(outdir / "mean_rmse_by_model_config.png")


def plot_overall_model_da(combined: pd.DataFrame, outdir: Path) -> None:
    df = combined.dropna(subset=["da"]).copy()
    agg = df.groupby(["model", "config"], dropna=False)["da"].mean().reset_index()
    pivot = agg.pivot(index="model", columns="config", values="da").sort_index()

    plt.figure()
    pivot.plot(kind="bar")
    plt.xlabel("Model")
    plt.ylabel("Mean DA")
    plt.title("Mean Directional Accuracy by Model/Config (overall)")
    plt.legend(title="Config", fontsize=8)
    _savefig(outdir / "mean_da_by_model_config.png")


def plot_best_per_currency_horizon(combined: pd.DataFrame, metric: str, outdir: Path) -> None:
    df = combined.dropna(subset=[metric]).copy()
    # choose best row per (currency,horizon)
    if metric == "rmse":
        idx = df.groupby(["currency", "horizon"])[metric].idxmin()
        title_metric = "Best (lowest) RMSE"
    else:
        idx = df.groupby(["currency", "horizon"])[metric].idxmax()
        title_metric = "Best (highest) DA"
    best = df.loc[idx].copy()
    # build a label
    best["label"] = best["model"].astype(str) + "::" + best["config"].astype(str)

    # Create a simple categorical heatmap-like grid using text
    currencies = sorted(best["currency"].unique())
    horizons = sorted(best["horizon"].unique())

    # Map to matrix of strings
    mat = [[None for _ in horizons] for _ in currencies]
    for i, cur in enumerate(currencies):
        for j, h in enumerate(horizons):
            row = best[(best["currency"] == cur) & (best["horizon"] == h)]
            mat[i][j] = row["label"].iloc[0] if len(row) else ""

    plt.figure(figsize=(max(8, 2 + 2 * len(horizons)), max(4, 1 + 0.6 * len(currencies))))
    plt.axis("off")
    plt.title(f"{title_metric} per (currency, horizon) — winner label = model::config")

    # Table plot
    table = plt.table(
        cellText=mat,
        rowLabels=currencies,
        colLabels=horizons,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    _savefig(outdir / f"best_{metric}_winners_table.png")


def plot_llm_config_deltas(llm: pd.DataFrame, outdir: Path) -> None:
    # compute deltas vs baseline_llm
    needed = {"currency", "horizon", "model", "config", "rmse", "da"}
    if not needed.issubset(set(llm.columns)):
        return

    base = llm[llm["config"].astype(str).str.lower() == "baseline_llm"].copy()
    if base.empty:
        return
    base = base.rename(columns={"rmse": "rmse_base", "da": "da_base"})[
        ["currency", "horizon", "model", "rmse_base", "da_base"]
    ]
    df = llm.merge(base, on=["currency", "horizon", "model"], how="left")
    df["rmse_delta"] = df["rmse"] - df["rmse_base"]
    df["da_delta"] = df["da"] - df["da_base"]

    # Aggregate deltas by (model,config)
    agg = df.groupby(["model", "config"], dropna=False).agg(
        mean_rmse_delta=("rmse_delta", "mean"),
        mean_da_delta=("da_delta", "mean"),
    ).reset_index()

    # RMSE delta plot
    pivot_rmse = agg.pivot(index="model", columns="config", values="mean_rmse_delta").sort_index()
    plt.figure()
    pivot_rmse.plot(kind="bar")
    plt.axhline(0.0)
    plt.xlabel("LLM Model")
    plt.ylabel("Mean RMSE delta vs baseline_llm (negative is better)")
    plt.title("LLM config impact (RMSE delta)")
    plt.legend(title="Config", fontsize=8)
    _savefig(outdir / "llm_rmse_delta_vs_baseline.png")

    # DA delta plot
    pivot_da = agg.pivot(index="model", columns="config", values="mean_da_delta").sort_index()
    plt.figure()
    pivot_da.plot(kind="bar")
    plt.axhline(0.0)
    plt.xlabel("LLM Model")
    plt.ylabel("Mean DA delta vs baseline_llm (positive is better)")
    plt.title("LLM config impact (DA delta)")
    plt.legend(title="Config", fontsize=8)
    _savefig(outdir / "llm_da_delta_vs_baseline.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--outdir", type=str, default=None, help="Output dir for plots (default: <results-dir>/plots)")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    outdir = Path(args.outdir) if args.outdir else (rdir / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    combined = _load_csv(rdir / "combined_summary.csv")
    llm = _load_csv(rdir / "llm_summary.csv")

    if combined is None:
        raise FileNotFoundError(f"Missing: {rdir / 'combined_summary.csv'}")

    plot_overall_model_rmse(combined, outdir)
    plot_overall_model_da(combined, outdir)
    plot_best_per_currency_horizon(combined, metric="rmse", outdir=outdir)
    if "da" in combined.columns:
        plot_best_per_currency_horizon(combined, metric="da", outdir=outdir)

    if llm is not None:
        plot_llm_config_deltas(llm, outdir)

    print(f"[OK] Plots saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()