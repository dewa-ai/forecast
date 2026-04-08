#!/usr/bin/env python3
"""
plot_rq.py
Generates thesis-ready plots for each Research Question (RQ1–RQ4).

Output structure:
  results/plots_rq/
    rq1_llm_vs_baseline_rmse.png
    rq1_llm_vs_baseline_da.png
    rq1_per_currency_rmse.png
    rq1_per_currency_da.png
    rq2_news_effect_rmse.png
    rq2_news_effect_da.png
    rq3_prompt_effect_rmse.png
    rq3_prompt_effect_da.png
    rq4_interaction_heatmap_rmse.png
    rq4_interaction_heatmap_da.png
    rq4_interaction_grouped_rmse.png
    rq4_interaction_grouped_da.png

Usage:
    python plot_rq.py --results-dir results/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

COLORS = ["#2E86AB", "#E84855", "#F9A620", "#3BB273", "#7B2D8B", "#555555", "#FF6B35"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"  [SKIP] Not found: {path.name}")
        return None
    df = pd.read_csv(path)
    df = _lower_cols(df)
    rename_map = {
        "pair": "currency", "fx_pair": "currency", "symbol": "currency",
        "h": "horizon", "horizon_days": "horizon",
        "directional_accuracy": "da", "dir_acc": "da", "acc": "da",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "config" not in df.columns:
        df["config"] = "default"
    if "model" not in df.columns:
        if "baseline_model" in df.columns:
            df["model"] = df["baseline_model"]
        elif "llm" in df.columns:
            df["model"] = df["llm"]
    for col in ["rmse", "mae", "mape", "da"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "horizon" in df.columns:
        df["horizon"] = df["horizon"].astype(str)
    return df


def _savefig(outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {outpath.name}")


# ── RQ1: LLM vs Baseline ─────────────────────────────────────────────────────

def plot_rq1(combined: pd.DataFrame, outdir: Path) -> None:
    """
    RQ1: Compare LLM models vs baseline models.
    - Overall mean RMSE and DA per model (best config for LLMs)
    - Per-currency mean RMSE
    """
    print("\n[RQ1] LLM vs Baseline...")

    # Identify baseline vs LLM models
    baseline_names = {"arima", "lstm", "xgboost", "itransformer"}

    def model_type(m: str) -> str:
        return "Baseline" if str(m).lower() in baseline_names else "LLM"

    df = combined.copy()
    df["type"] = df["model"].apply(model_type)

    # For LLMs, use best config per (model, currency, horizon) by RMSE
    llm_df = df[df["type"] == "LLM"].copy()
    if not llm_df.empty and "rmse" in llm_df.columns:
        idx = llm_df.groupby(["model", "currency", "horizon"])["rmse"].idxmin()
        llm_best = llm_df.loc[idx]
    else:
        llm_best = llm_df

    base_df = df[df["type"] == "Baseline"].copy()
    plot_df = pd.concat([base_df, llm_best], ignore_index=True)

    # ── Plot 1a: Overall mean RMSE per model ──
    for metric, ascending, ylabel, suffix in [
        ("rmse", True,  "Mean RMSE (lower is better)", "rmse"),
        ("da",   False, "Mean Directional Accuracy % (higher is better)", "da"),
    ]:
        if metric not in plot_df.columns:
            continue

        agg = (plot_df.dropna(subset=[metric])
                      .groupby(["model", "type"])[metric]
                      .mean()
                      .reset_index()
                      .sort_values(metric, ascending=ascending))

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = [COLORS[0] if t == "Baseline" else COLORS[1] for t in agg["type"]]
        bars = ax.bar(agg["model"], agg[metric], color=colors, edgecolor="white", linewidth=0.8)

        # Legend patches
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS[0], label="Baseline"),
            Patch(facecolor=COLORS[1], label="LLM (best config)"),
        ]
        ax.legend(handles=legend_elements, fontsize=9)
        ax.set_xlabel("Model")
        ax.set_ylabel(ylabel)
        ax.set_title(f"RQ1 — Overall {metric.upper()} by Model")
        plt.xticks(rotation=30, ha="right")
        _savefig(outdir / f"rq1_llm_vs_baseline_{suffix}.png")

    # ── Plot 1b: Per-currency mean RMSE (grouped bar: baseline best vs LLM best) ──
    if "currency" not in plot_df.columns or "rmse" not in plot_df.columns:
        return

    # Best model per (type, currency)
    base_best = (base_df.dropna(subset=["rmse"])
                        .groupby(["type", "currency"])["rmse"].min().reset_index())
    llm_best2 = (llm_best.dropna(subset=["rmse"])
                         .groupby(["type", "currency"])["rmse"].min().reset_index())
    comp = pd.concat([base_best, llm_best2])

    currencies = sorted(comp["currency"].unique())
    x = np.arange(len(currencies))
    types = ["Baseline", "LLM"]
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, t in enumerate(types):
        vals = [comp[(comp["type"] == t) & (comp["currency"] == c)]["rmse"].values
                for c in currencies]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + i * width, vals, width, label=t, color=COLORS[i], edgecolor="white")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(currencies, rotation=20, ha="right")
    ax.set_ylabel("Best RMSE (lower is better)")
    ax.set_title("RQ1 — Best RMSE per Currency: Baseline vs LLM")
    ax.legend()
    _savefig(outdir / "rq1_per_currency_rmse.png")

    # ── Plot 1c: Per-currency DA ──
    if "da" not in plot_df.columns:
        return

    base_da = (base_df.dropna(subset=["da"])
                      .groupby(["type", "currency"])["da"].max().reset_index())
    llm_da  = (llm_best.dropna(subset=["da"])
                       .groupby(["type", "currency"])["da"].max().reset_index())
    comp_da = pd.concat([base_da, llm_da])

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, t in enumerate(types):
        vals = [comp_da[(comp_da["type"] == t) & (comp_da["currency"] == c)]["da"].values
                for c in currencies]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(x + i * width, vals, width, label=t, color=COLORS[i], edgecolor="white")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(currencies, rotation=20, ha="right")
    ax.set_ylabel("Best DA % (higher is better)")
    ax.set_title("RQ1 — Best DA per Currency: Baseline vs LLM")
    ax.legend()
    _savefig(outdir / "rq1_per_currency_da.png")


# ── RQ2: News Augmentation Effect ────────────────────────────────────────────

def plot_rq2(llm: pd.DataFrame, outdir: Path) -> None:
    """
    RQ2: Effect of news augmentation.
    Compare no-news configs vs news configs, averaged across prompt types.
    """
    print("\n[RQ2] News augmentation effect...")

    news_map = {
        "baseline_llm": "No News",
        "prompt_only":  "No News",
        "news_only":    "News",
        "full_model":   "News",
    }

    df = llm.copy()
    df["news_group"] = df["config"].astype(str).str.lower().map(news_map)
    df = df.dropna(subset=["news_group"])

    for metric, ascending, ylabel, suffix in [
        ("rmse", True,  "Mean RMSE (lower is better)", "rmse"),
        ("da",   False, "Mean DA % (higher is better)", "da"),
    ]:
        if metric not in df.columns:
            continue

        agg = (df.dropna(subset=[metric])
                 .groupby(["model", "news_group"])[metric]
                 .mean().reset_index())
        if agg.empty:
            continue

        models = sorted(agg["model"].unique())
        groups = ["No News", "News"]
        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, grp in enumerate(groups):
            vals = [agg[(agg["model"] == m) & (agg["news_group"] == grp)][metric].values
                    for m in models]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.bar(x + i * width, vals, width, label=grp,
                   color=COLORS[i], edgecolor="white")

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"RQ2 — News Augmentation Effect on {metric.upper()}")
        ax.legend(title="News")
        _savefig(outdir / f"rq2_news_effect_{suffix}.png")


# ── RQ3: Prompting Strategy Effect ───────────────────────────────────────────

def plot_rq3(llm: pd.DataFrame, outdir: Path) -> None:
    """
    RQ3: Effect of prompting strategy (zero-shot vs few-shot).
    Compare zero-shot configs vs few-shot configs, averaged across news.
    """
    print("\n[RQ3] Prompting strategy effect...")

    prompt_map = {
        "baseline_llm": "Zero-shot",
        "news_only":    "Zero-shot",
        "prompt_only":  "Few-shot",
        "full_model":   "Few-shot",
    }

    df = llm.copy()
    df["prompt_group"] = df["config"].astype(str).str.lower().map(prompt_map)
    df = df.dropna(subset=["prompt_group"])

    for metric, ascending, ylabel, suffix in [
        ("rmse", True,  "Mean RMSE (lower is better)", "rmse"),
        ("da",   False, "Mean DA % (higher is better)", "da"),
    ]:
        if metric not in df.columns:
            continue

        agg = (df.dropna(subset=[metric])
                 .groupby(["model", "prompt_group"])[metric]
                 .mean().reset_index())
        if agg.empty:
            continue

        models = sorted(agg["model"].unique())
        groups = ["Zero-shot", "Few-shot"]
        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, grp in enumerate(groups):
            vals = [agg[(agg["model"] == m) & (agg["prompt_group"] == grp)][metric].values
                    for m in models]
            vals = [v[0] if len(v) else np.nan for v in vals]
            ax.bar(x + i * width, vals, width, label=grp,
                   color=COLORS[i+2], edgecolor="white")

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"RQ3 — Prompting Strategy Effect on {metric.upper()}")
        ax.legend(title="Prompt")
        _savefig(outdir / f"rq3_prompt_effect_{suffix}.png")


# ── RQ4: Interaction Effect ───────────────────────────────────────────────────

def plot_rq4(llm: pd.DataFrame, outdir: Path) -> None:
    """
    RQ4: Interaction effect of news × prompt.
    - Heatmap of 2×2 mean metric values
    - Grouped bar: all 4 configs per model
    """
    print("\n[RQ4] Interaction effect (news × prompt)...")

    config_map = {
        "baseline_llm": ("No News", "Zero-shot"),
        "news_only":    ("News",    "Zero-shot"),
        "prompt_only":  ("No News", "Few-shot"),
        "full_model":   ("News",    "Few-shot"),
    }

    df = llm.copy()
    df["news_label"]   = df["config"].astype(str).str.lower().map(lambda c: config_map.get(c, (None, None))[0])
    df["prompt_label"] = df["config"].astype(str).str.lower().map(lambda c: config_map.get(c, (None, None))[1])
    df = df.dropna(subset=["news_label", "prompt_label"])

    for metric, better, ylabel, suffix in [
        ("rmse", "lower", "Mean RMSE", "rmse"),
        ("da",   "higher", "Mean DA %", "da"),
    ]:
        if metric not in df.columns:
            continue

        # ── Heatmap ──
        pivot = (df.dropna(subset=[metric])
                   .groupby(["news_label", "prompt_label"])[metric]
                   .mean()
                   .unstack("prompt_label"))

        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(5, 4))
        cmap = "RdYlGn_r" if better == "lower" else "RdYlGn"
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, label=ylabel)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)
        ax.set_xlabel("Prompting Strategy")
        ax.set_ylabel("News Augmentation")
        ax.set_title(f"RQ4 — Interaction Effect ({metric.upper()}, {better} is better)")

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=11, fontweight="bold", color="black")

        _savefig(outdir / f"rq4_interaction_heatmap_{suffix}.png")

        # ── Grouped bar: 4 configs per model ──
        config_order = ["baseline_llm", "news_only", "prompt_only", "full_model"]
        config_labels = {
            "baseline_llm": "Baseline LLM\n(no news, zero-shot)",
            "news_only":    "News Only\n(news, zero-shot)",
            "prompt_only":  "Prompt Only\n(no news, few-shot)",
            "full_model":   "Full Model\n(news, few-shot)",
        }

        agg = (llm.dropna(subset=[metric])
                  .groupby(["model", "config"])[metric]
                  .mean().reset_index())

        models = sorted(agg["model"].unique())
        configs = [c for c in config_order if c in agg["config"].unique()]
        x = np.arange(len(models))
        width = 0.8 / len(configs)

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, cfg in enumerate(configs):
            vals = [agg[(agg["model"] == m) & (agg["config"] == cfg)][metric].values
                    for m in models]
            vals = [v[0] if len(v) else np.nan for v in vals]
            offset = (i - len(configs) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=config_labels.get(cfg, cfg),
                   color=COLORS[i], edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel(f"{ylabel} ({better} is better)")
        ax.set_title(f"RQ4 — All Ablation Configs: {metric.upper()} per Model")
        ax.legend(fontsize=8, ncol=2)
        _savefig(outdir / f"rq4_interaction_grouped_{suffix}.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output dir (default: <results-dir>/plots_rq)")
    args = ap.parse_args()

    rdir   = Path(args.results_dir)
    outdir = Path(args.outdir) if args.outdir else (rdir / "plots_rq")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading results from: {rdir.resolve()}")
    combined = _load_csv(rdir / "combined_summary.csv")
    llm      = _load_csv(rdir / "llm_summary.csv")

    if combined is not None:
        plot_rq1(combined, outdir)
    else:
        print("\n[RQ1] SKIP — combined_summary.csv not found")

    if llm is not None:
        plot_rq2(llm, outdir)
        plot_rq3(llm, outdir)
        plot_rq4(llm, outdir)
    else:
        print("\n[RQ2–4] SKIP — llm_summary.csv not found")

    print(f"\nDone. Plots saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()