#!/usr/bin/env python3
"""
plot_raw_fx_series.py — Raw historical FX price line plots for thesis Chapter 3/4.

Addresses committee feedback requesting visualization of raw time series data
(not just aggregated metric plots), to show trend/volatility characteristics
of each currency pair over the full 2016-2026 study period.

Run this on a machine with internet access to Yahoo Finance,
since yfinance cannot be reached from a network-restricted sandbox.

Usage:
    python3 plot_raw_fx_series.py --start 2016-01-01 --end 2026-03-31 --outdir figures/

Output:
    figures/fx_raw_series_all.png       -- 5-panel grid, one per currency pair
    figures/fx_raw_series_<PAIR>.png    -- individual full-size plot per pair (optional, --individual flag)
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Missing dependency: yfinance. Install with: pip install yfinance --break-system-packages")


PAIR_TO_TICKER = {
    "USD/IDR": "IDR=X",
    "USD/EUR": "EUR=X",
    "USD/SGD": "SGD=X",
    "USD/TWD": "TWD=X",
    "USD/AUD": "AUD=X",
}


def fetch_series(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check network access to Yahoo Finance.")
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def plot_grid(series_dict: dict[str, pd.Series], outpath: Path):
    """5-panel grid, one subplot per currency pair, matching thesis figure style."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (pair, s) in zip(axes, series_dict.items()):
        ax.plot(s.index, s.values, linewidth=0.8, color="#1f4e79")
        ax.set_title(pair, fontsize=11, fontweight="bold")
        ax.set_ylabel("Exchange Rate")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=0)

    # Hide the unused 6th subplot (5 pairs in a 3x2 grid)
    axes[-1].axis("off")

    fig.suptitle("Historical FX Closing Prices, January 2016 -- March 2026", fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {outpath}")


def plot_individual(pair: str, s: pd.Series, outpath: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s.index, s.values, linewidth=0.9, color="#1f4e79")
    ax.set_title(f"{pair} Historical Closing Price (Jan 2016 -- Mar 2026)", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Exchange Rate")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-03-31")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--individual", action="store_true",
                     help="Also save one full-size plot per currency pair, not just the grid")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    series_dict = {}
    for pair, ticker in PAIR_TO_TICKER.items():
        print(f"Fetching {pair} ({ticker})...")
        series_dict[pair] = fetch_series(ticker, args.start, args.end)

    plot_grid(series_dict, outdir / "fx_raw_series_all.png")

    if args.individual:
        for pair, s in series_dict.items():
            safe_name = pair.replace("/", "")
            plot_individual(pair, s, outdir / f"fx_raw_series_{safe_name}.png")


if __name__ == "__main__":
    main()