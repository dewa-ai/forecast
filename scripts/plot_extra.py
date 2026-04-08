#!/usr/bin/env python3
"""
plot_extra.py
Extra plots and tables for thesis Chapter 4 (Experiments & Results).

Generates:
  1. Per-currency metric tables (RMSE, MAE, MAPE, DA) — one CSV per currency
  2. 2x2 interaction effect table  (RQ4)
  3. Per-horizon bar plots (h=1, h=5, h=10) for RMSE and DA

Input files expected in --results-dir:
  - combined_summary.csv   (all models, all configs)
  - llm_summary.csv        (LLM models only, with config column)

Usage:
    python plot_extra.py --results-dir results/
    python plot_extra.py --results-dir results/ --outdir results/plots_extra
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"  [SKIP] File not found: {path}")
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


def _savefig(outpath: Path, dpi: int = 180) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    print(f"  [OK] {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# 0. RQ1 Summary table — best model per (currency, horizon)
# ─────────────────────────────────────────────────────────────────────────────

def export_rq1_summary_table(combined: pd.DataFrame, outdir: Path) -> None:
    """
    For each (currency, horizon), find the best model by RMSE and by DA.
    Output: one compact CSV for use in thesis body (Table 4.x).

    Columns: currency | horizon | best_model_rmse | rmse | mae | mape | da_at_best_rmse
             | best_model_da | da
    """
    needed = {"currency", "horizon", "model", "rmse"}
    if not needed.issubset(set(combined.columns)):
        print("  [SKIP] RQ1 summary — missing required columns")
        return

    metrics = [m for m in ["rmse", "mae", "mape", "da"] if m in combined.columns]
    rows = []

    for (cur, h), grp in combined.groupby(["currency", "horizon"]):
        row = {"currency": cur, "horizon": h}

        # Best by RMSE
        grp_rmse = grp.dropna(subset=["rmse"])
        if not grp_rmse.empty:
            best_rmse_row = grp_rmse.loc[grp_rmse["rmse"].idxmin()]
            row["best_model_rmse"] = f"{best_rmse_row['model']} ({best_rmse_row.get('config', '')})"
            for m in metrics:
                row[m] = round(best_rmse_row[m], 4) if m in best_rmse_row and pd.notna(best_rmse_row[m]) else None

        # Best by DA (separate column)
        if "da" in combined.columns:
            grp_da = grp.dropna(subset=["da"])
            if not grp_da.empty:
                best_da_row = grp_da.loc[grp_da["da"].idxmax()]
                row["best_model_da"] = f"{best_da_row['model']} ({best_da_row.get('config', '')})"
                row["best_da"] = round(best_da_row["da"], 4)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["currency", "horizon"]).reset_index(drop=True)
    out_path = outdir / "tables" / "rq1_summary_best_per_currency_horizon.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  [OK] {out_path}  ({len(df)} rows)")

    # Also print to console
    print("\n  ── RQ1 Summary: Best Model per (Currency, Horizon) ──")
    print(df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Per-currency metric tables
# ─────────────────────────────────────────────────────────────────────────────

def export_per_currency_tables(combined: pd.DataFrame, outdir: Path) -> None:
    """
    For each currency, export a CSV with columns:
      model | config | horizon | RMSE | MAE | MAPE | DA
    Sorted by horizon then RMSE.
    """
    if "currency" not in combined.columns:
        print("  [SKIP] 'currency' column missing from combined_summary.csv")
        return

    metrics = [m for m in ["rmse", "mae", "mape", "da"] if m in combined.columns]
    show_cols = ["model", "config", "horizon"] + metrics

    currencies = sorted(combined["currency"].dropna().unique())
    for cur in currencies:
        sub = combined[combined["currency"] == cur].copy()
        sub = sub.sort_values(["horizon", "rmse"] if "rmse" in sub.columns else ["horizon"])
        sub = sub[[c for c in show_cols if c in sub.columns]]

        out_path = outdir / "tables" / f"table_{cur}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_path, index=False)
        print(f"  [OK] {out_path}  ({len(sub)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 2×2 Interaction effect table (RQ4)
# ─────────────────────────────────────────────────────────────────────────────

def export_interaction_table(llm: pd.DataFrame, outdir: Path) -> None:
    """
    Build 2×2 table: rows = news (No/Yes), cols = prompt (zero/few).
    Values = mean RMSE and mean DA across all currencies, horizons, models.

    Config names expected: baseline_llm, news_only, prompt_only, full_model
    """
    config_map = {
        "baseline_llm": (False, "zero"),
        "news_only":    (True,  "zero"),
        "prompt_only":  (False, "few"),
        "full_model":   (True,  "few"),
    }

    if "config" not in llm.columns:
        print("  [SKIP] 'config' column missing from llm_summary.csv")
        return

    rows = []
    for cfg, (news, prompt) in config_map.items():
        sub = llm[llm["config"].astype(str).str.lower() == cfg]
        if sub.empty:
            continue
        row = {"config": cfg, "news": "Yes" if news else "No", "prompt": prompt}
        for m in ["rmse", "mae", "mape", "da"]:
            if m in sub.columns:
                row[f"mean_{m}"] = round(sub[m].mean(), 4)
        rows.append(row)

    if not rows:
        print("  [SKIP] No matching configs found for interaction table")
        return

    df = pd.DataFrame(rows)

    # Save flat table
    flat_path = outdir / "tables" / "interaction_flat.csv"
    flat_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(flat_path, index=False)
    print(f"  [OK] {flat_path}")

    # Pivot for RMSE
    if "mean_rmse" in df.columns:
        pivot_rmse = df.pivot(index="news", columns="prompt", values="mean_rmse")
        pivot_rmse.index.name = "News \\ Prompt"
        pivot_path = outdir / "tables" / "interaction_2x2_rmse.csv"
        pivot_rmse.to_csv(pivot_path)
        print(f"  [OK] {pivot_path}")

        # Print to console (thesis-friendly)
        print("\n  ── 2×2 Interaction Table (Mean RMSE) ──")
        print(pivot_rmse.to_string())

    # Pivot for DA
    if "mean_da" in df.columns:
        pivot_da = df.pivot(index="news", columns="prompt", values="mean_da")
        pivot_da.index.name = "News \\ Prompt"
        pivot_path = outdir / "tables" / "interaction_2x2_da.csv"
        pivot_da.to_csv(pivot_path)
        print(f"  [OK] {pivot_path}")

        print("\n  ── 2×2 Interaction Table (Mean DA) ──")
        print(pivot_da.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-horizon plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_horizon(combined: pd.DataFrame, outdir: Path) -> None:
    """
    For each horizon (1, 5, 10), produce:
      - bar plot of mean RMSE per (model, config)
      - bar plot of mean DA per (model, config)
    """
    if "horizon" not in combined.columns:
        print("  [SKIP] 'horizon' column missing")
        return

    horizons = sorted(combined["horizon"].dropna().unique())

    for h in horizons:
        sub = combined[combined["horizon"].astype(str) == str(h)]
        if sub.empty:
            continue

        for metric, ascending, ylabel, label, use_log in [
            ("rmse", True,  "Mean RMSE (log scale)", "rmse", True),
            ("da",   False, "Mean DA (%)",            "da",   False),
        ]:
            if metric not in sub.columns:
                continue

            agg = (sub.dropna(subset=[metric])
                      .groupby(["model", "config"], dropna=False)[metric]
                      .mean()
                      .reset_index())

            # For DA: drop rows where value is 0 (missing baseline DA)
            if metric == "da":
                agg = agg[agg[metric] > 0]

            if agg.empty:
                continue

            pivot = agg.pivot(index="model", columns="config", values=metric).sort_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot.plot(kind="bar", ax=ax)

            if use_log:
                ax.set_yscale("log")
                ax.set_ylabel(ylabel)
                # Add minor grid for log scale
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.grid(True, which="both", alpha=0.3)
            else:
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)

            ax.set_xlabel("Model")
            ax.set_title(f"{ylabel} — Horizon h={h}")
            ax.legend(title="Config", fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
            plt.xticks(rotation=30, ha="right")

            out_path = outdir / f"horizon_{h}_{label}.png"
            _savefig(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output dir (default: <results-dir>/plots_extra)")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    outdir = Path(args.outdir) if args.outdir else (rdir / "plots_extra")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading results from: {rdir.resolve()}")
    combined = _load_csv(rdir / "combined_summary.csv")
    llm      = _load_csv(rdir / "llm_summary.csv")

    # 0. RQ1 summary table
    print("\n[0] Exporting RQ1 summary table (best model per currency/horizon)...")
    if combined is not None:
        export_rq1_summary_table(combined, outdir)
    else:
        print("  [SKIP] combined_summary.csv not found")

    # 1. Per-currency tables
    print("\n[1] Exporting per-currency tables...")
    if combined is not None:
        export_per_currency_tables(combined, outdir)
    else:
        print("  [SKIP] combined_summary.csv not found")

    # 2. 2×2 interaction table
    print("\n[2] Building 2×2 interaction effect table...")
    if llm is not None:
        export_interaction_table(llm, outdir)
    else:
        print("  [SKIP] llm_summary.csv not found")

    # 3. Per-horizon plots
    print("\n[3] Generating per-horizon plots...")
    if combined is not None:
        plot_per_horizon(combined, outdir)
    else:
        print("  [SKIP] combined_summary.csv not found")

    print(f"\nDone. Outputs in: {outdir.resolve()}")


if __name__ == "__main__":
    main()