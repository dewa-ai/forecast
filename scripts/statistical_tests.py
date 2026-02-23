#!/usr/bin/env python3
"""
statistical_tests.py
- Runs simple paired tests for LLM configs vs baseline_llm on RMSE and DA
- Uses paired t-test and Wilcoxon signed-rank (if available)
- Outputs CSV summary into results/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

try:
    from scipy.stats import ttest_rel, wilcoxon
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


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
        if "llm" in df.columns:
            df["model"] = df["llm"]
    for col in ["rmse", "da"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "horizon" in df.columns:
        df["horizon"] = df["horizon"].astype(str)
    return df


def _paired_arrays(base: pd.Series, other: pd.Series) -> Tuple[pd.Series, pd.Series]:
    # Align indices and drop NaNs
    df = pd.concat([base, other], axis=1).dropna()
    return df.iloc[:, 0], df.iloc[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="results")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    if not SCIPY_OK:
        raise RuntimeError(
            "scipy is required for statistical tests.\n"
            "Install: pip install scipy"
        )

    rdir = Path(args.results_dir)
    llm = _load_csv(rdir / "llm_summary.csv")
    if llm is None:
        raise FileNotFoundError(f"Missing: {rdir / 'llm_summary.csv'}")

    needed = {"currency", "horizon", "model", "config", "rmse", "da"}
    missing = needed - set(llm.columns)
    if missing:
        raise ValueError(f"llm_summary.csv missing columns: {sorted(missing)}. Found: {list(llm.columns)}")

    # baseline rows
    base = llm[llm["config"].astype(str).str.lower() == "baseline_llm"].copy()
    if base.empty:
        raise ValueError("No rows with config=baseline_llm found in llm_summary.csv")

    # Index by experimental unit for pairing
    key_cols = ["currency", "horizon", "model"]
    base = base.set_index(key_cols)

    results = []
    for cfg in sorted(llm["config"].astype(str).unique()):
        if cfg.lower() == "baseline_llm":
            continue

        other = llm[llm["config"].astype(str) == cfg].set_index(key_cols)

        # Only compare on overlapping units
        overlap = base.index.intersection(other.index)
        if len(overlap) < 5:
            continue

        # RMSE test (lower is better): test whether cfg < baseline
        b_rmse, o_rmse = _paired_arrays(base.loc[overlap, "rmse"], other.loc[overlap, "rmse"])
        # DA test (higher is better): test whether cfg > baseline
        b_da, o_da = _paired_arrays(base.loc[overlap, "da"], other.loc[overlap, "da"])

        # Paired t-test (two-sided). You can interpret direction with mean delta.
        t_rmse = ttest_rel(o_rmse, b_rmse, nan_policy="omit")
        t_da = ttest_rel(o_da, b_da, nan_policy="omit")

        # Wilcoxon (two-sided); may fail if all deltas zero
        try:
            w_rmse = wilcoxon(o_rmse - b_rmse)
            w_da = wilcoxon(o_da - b_da)
            w_rmse_p = float(w_rmse.pvalue)
            w_da_p = float(w_da.pvalue)
        except Exception:
            w_rmse_p = float("nan")
            w_da_p = float("nan")

        rmse_delta = (o_rmse - b_rmse)
        da_delta = (o_da - b_da)

        results.append({
            "config": cfg,
            "n_pairs": int(len(overlap)),
            "mean_rmse_delta": float(rmse_delta.mean()),
            "median_rmse_delta": float(rmse_delta.median()),
            "t_p_rmse": float(t_rmse.pvalue),
            "w_p_rmse": w_rmse_p,
            "mean_da_delta": float(da_delta.mean()),
            "median_da_delta": float(da_delta.median()),
            "t_p_da": float(t_da.pvalue),
            "w_p_da": w_da_p,
        })

    out = pd.DataFrame(results)
    if out.empty:
        print("[WARN] No configs had enough paired samples to test.")
        return

    out = out.sort_values(["mean_rmse_delta", "t_p_rmse"], ascending=[True, True]).reset_index(drop=True)
    outpath = rdir / "llm_stat_tests.csv"
    out.to_csv(outpath, index=False)

    print("=" * 72)
    print("Paired tests vs baseline_llm (rmse_delta<0 better, da_delta>0 better)")
    print("=" * 72)
    print(out.to_string(index=False))
    print(f"\n[OK] Saved: {outpath.resolve()}")
    print(f"Alpha = {args.alpha} (use p-values to judge significance)")

if __name__ == "__main__":
    main()