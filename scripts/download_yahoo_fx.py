#!/usr/bin/env python3
"""
python3 scripts/download_yahoo_fx.py --all --start 2016-01-01

use previous date to avoid missing last day data

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency: yfinance. Install with: pip install yfinance", file=sys.stderr)
    raise


PAIR_TO_TICKER = {
    "USDIDR": "IDR=X",
    "USDEUR": "EUR=X",
    "USDSGD": "SGD=X",
    "USDTWD": "TWD=X",
    "USDAUD": "AUD=X"
}


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance can return MultiIndex columns (even for a single ticker).
    Flatten to simple string columns.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Usually like ('Open','IDR=X'), ('Close','IDR=X') -> keep first level
        df.columns = [c[0] if isinstance(c, tuple) and len(c) > 0 else str(c) for c in df.columns]
    else:
        # Sometimes columns can still be tuples in rare cases
        df.columns = [c[0] if isinstance(c, tuple) and len(c) > 0 else c for c in df.columns]
    return df


def download_fx(ticker: str, start: str | None, end: str | None, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",  # helps keep OHLC as first level
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Reset index so date becomes a column
    df = df.reset_index()

    # Flatten columns so we never see tuple columns
    df = _flatten_columns(df)

    # Normalize date column name
    if "Date" in df.columns:
        df.rename(columns={"Date": "date"}, inplace=True)
    elif "Datetime" in df.columns:
        df.rename(columns={"Datetime": "date"}, inplace=True)
    else:
        # fallback: assume first column is date-like
        df.rename(columns={df.columns[0]: "date"}, inplace=True)

    # Rename OHLCV columns to lowercase
    rename_map = {}
    for c in df.columns:
        if not isinstance(c, str):
            continue
        lc = c.lower()
        if lc in {"open", "high", "low", "close", "adj close", "volume"}:
            rename_map[c] = lc.replace(" ", "_")
    df.rename(columns=rename_map, inplace=True)

    # Keep a clean set of columns
    keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Ensure ISO date
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    # Sort just in case
    df = df.sort_values("date").reset_index(drop=True)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", help="Single pair, e.g., USDIDR")
    ap.add_argument("--pairs", nargs="*", help="Multiple pairs, e.g., USDIDR USDEUR")
    ap.add_argument("--all", action="store_true", help="Download all available pairs")
    ap.add_argument("--ticker", help="Direct Yahoo ticker override, e.g., IDR=X")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD")
    ap.add_argument("--interval", default="1d", help="1d, 1h, etc.")
    ap.add_argument("--out_dir", default="data/fx_raw")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str]] = []

    if args.all:
        # Use all pairs from PAIR_TO_TICKER
        jobs = [(pair, ticker) for pair, ticker in PAIR_TO_TICKER.items()]
        print(f"[INFO] Using all {len(jobs)} available pairs")

    if args.ticker:
        jobs.append(("CUSTOM", args.ticker))

    if args.pair:
        p = args.pair.upper()
        tkr = PAIR_TO_TICKER.get(p, "")
        if not tkr:
            print(f"[WARN] No ticker mapping for {p}. Add to PAIR_TO_TICKER or pass --ticker.", file=sys.stderr)
        else:
            jobs.append((p, tkr))

    if args.pairs:
        for p in args.pairs:
            p = p.upper()
            tkr = PAIR_TO_TICKER.get(p, "")
            if not tkr:
                print(f"[WARN] No ticker mapping for {p}. Add to PAIR_TO_TICKER or pass --ticker.", file=sys.stderr)
                continue
            jobs.append((p, tkr))

    if not jobs:
        print("No valid download jobs. Example: --pairs USDIDR USDEUR", file=sys.stderr)
        sys.exit(1)

    for name, tkr in jobs:
        print(f"[INFO] Downloading {name} ({tkr}) ...")
        df = download_fx(tkr, args.start, args.end, args.interval)

        if df.empty:
            print(f"[WARN] Empty result for {name} ({tkr})", file=sys.stderr)
            continue

        out_path = out_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"[OK] Saved {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()