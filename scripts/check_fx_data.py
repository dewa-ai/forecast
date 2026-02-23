#!/usr/bin/env python3
"""
Check FX data quality and forward-fill missing days.

Examples:
  # Check data quality for all pairs
  python scripts/check_fx_data.py --data-dir data/fx_raw
  
  # Forward-fill ALL days (including weekends) - DEFAULT for FX rates
  python scripts/check_fx_data.py --data-dir data/fx_raw --fill --out-dir data/fx_filled
  
  # Forward-fill trading days only (if needed)
  python scripts/check_fx_data.py --data-dir data/fx_raw --fill --trading-days-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


def analyze_completeness(df: pd.DataFrame, pair_name: str) -> dict:
    """Analyze data completeness and return statistics."""
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    start_date = df['date'].min()
    end_date = df['date'].max()
    
    # Calculate expected trading days (exclude weekends)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    trading_days = all_dates[all_dates.dayofweek < 5]  # Mon-Fri only
    
    expected_count = len(trading_days)
    actual_count = len(df)
    missing_count = expected_count - actual_count
    
    # Find gaps (missing trading days)
    df_dates = set(df['date'].dt.date)
    expected_dates = set(trading_days.date)
    missing_dates = sorted(expected_dates - df_dates)
    
    stats = {
        'pair': pair_name,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'actual_rows': actual_count,
        'expected_trading_days': expected_count,
        'missing_trading_days': missing_count,
        'completeness_pct': round((actual_count / expected_count * 100), 2),
        'missing_dates': missing_dates[:10] if missing_dates else []  # Show first 10
    }
    
    return stats


def forward_fill_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing calendar days (including weekends)."""
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create complete date range (all days including weekends)
    start_date = df['date'].min()
    end_date = df['date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create complete dataframe
    complete_df = pd.DataFrame({'date': all_dates})
    
    # Merge with original data
    filled_df = complete_df.merge(df, on='date', how='left')
    
    # Forward fill missing values (using new pandas syntax)
    filled_df = filled_df.ffill()
    
    # Convert date back to string format
    filled_df['date'] = filled_df['date'].dt.strftime('%Y-%m-%d')
    
    return filled_df


def print_quality_report(stats: dict):
    """Print a formatted quality report."""
    print(f"\n{'='*60}")
    print(f"Data Quality Report: {stats['pair']}")
    print(f"{'='*60}")
    print(f"Date Range    : {stats['start_date']} to {stats['end_date']}")
    print(f"Actual Rows   : {stats['actual_rows']:,}")
    print(f"Expected Days : {stats['expected_trading_days']:,} (trading days only)")
    print(f"Missing Days  : {stats['missing_trading_days']:,}")
    print(f"Completeness  : {stats['completeness_pct']}%")
    
    if stats['missing_dates']:
        print(f"\nFirst Missing Dates (up to 10):")
        for date in stats['missing_dates']:
            print(f"  - {date}")
    
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Check FX data quality and optionally forward-fill missing days"
    )
    ap.add_argument("--data-dir", default="data/fx_raw", help="Input directory with CSV files")
    ap.add_argument("--out-dir", default="data/fx_filled", help="Output directory for filled data")
    ap.add_argument("--pair", help="Check specific pair only (e.g., USDIDR)")
    ap.add_argument("--fill", action="store_true", help="Forward-fill missing days (includes weekends by default)")
    ap.add_argument("--trading-days-only", action="store_true", 
                    help="Fill trading days only (Mon-Fri), exclude weekends")
    args = ap.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Find all CSV files
    if args.pair:
        csv_files = list(data_dir.glob(f"{args.pair.upper()}.csv"))
    else:
        csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"[ERROR] No CSV files found in {data_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Found {len(csv_files)} file(s) to process\n")
    
    all_stats = []
    
    for csv_file in csv_files:
        pair_name = csv_file.stem
        
        try:
            df = pd.read_csv(csv_file)
            
            if df.empty or 'date' not in df.columns:
                print(f"[WARN] Skipping {pair_name}: empty or no date column")
                continue
            
            # Analyze completeness
            stats = analyze_completeness(df, pair_name)
            all_stats.append(stats)
            print_quality_report(stats)
            
            # Forward-fill if requested
            if args.fill:
                if args.trading_days_only:
                    # Fill only trading days
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    
                    # Create trading days range
                    start = df['date'].min()
                    end = df['date'].max()
                    all_dates = pd.date_range(start=start, end=end, freq='D')
                    trading_days = all_dates[all_dates.dayofweek < 5]
                    
                    complete_df = pd.DataFrame({'date': trading_days})
                    filled_df = complete_df.merge(df, on='date', how='left')
                    filled_df = filled_df.ffill()
                    filled_df['date'] = filled_df['date'].dt.strftime('%Y-%m-%d')
                    
                    print(f"[INFO] Forward-filled trading days only (Mon-Fri)")
                else:
                    # Default: Fill ALL calendar days (including weekends)
                    filled_df = forward_fill_dates(df)
                    print(f"[INFO] Forward-filled ALL calendar days (including weekends)")
                
                # Save filled data
                out_dir = Path(args.out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{pair_name}.csv"
                filled_df.to_csv(out_path, index=False)
                
                print(f"[OK] Saved filled data ({len(filled_df)} rows) -> {out_path}\n")
        
        except Exception as e:
            print(f"[ERROR] Failed to process {pair_name}: {e}", file=sys.stderr)
            continue
    
    # Summary report
    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY - All Pairs")
        print(f"{'='*60}")
        print(f"{'Pair':<12} {'Rows':<10} {'Missing':<10} {'Complete%':<10}")
        print(f"{'-'*60}")
        for stat in all_stats:
            print(f"{stat['pair']:<12} {stat['actual_rows']:<10} "
                  f"{stat['missing_trading_days']:<10} {stat['completeness_pct']:<10}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()