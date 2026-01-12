#!/usr/bin/env python3
"""
python3 scripts/download_yahoo_news.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency: yfinance. Install with: pip install yfinance", file=sys.stderr)
    raise

try:
    import pandas as pd
except ImportError:
    print("Missing dependency: pandas. Install with: pip install pandas", file=sys.stderr)
    raise


PAIR_TO_TICKER = {
    "USDIDR": "IDR=X",
    "USDEUR": "EUR=X",
    "USDSGD": "SGD=X",
    "USDTWD": "TWD=X",
    "USDAUD": "AUD=X"
}


def download_news(ticker: str, debug: bool = False) -> list[dict]:
    """Download news articles related to the ticker from Yahoo Finance."""
    try:
        tick = yf.Ticker(ticker)
        news = tick.news
        
        if not news:
            return []
        
        # Debug: print first article structure
        if debug and news:
            print(f"[DEBUG] First article keys: {news[0].keys()}")
            print(f"[DEBUG] First article sample: {json.dumps(news[0], indent=2, default=str)[:500]}")
        
        # Clean and structure the news data
        cleaned_news = []
        for article in news:
            # Yahoo Finance structure: article has 'id' and 'content'
            # The actual data is nested in 'content'
            content = article.get('content', {})
            if not content:
                # Fallback to top-level if no content key
                content = article
            
            # Extract fields from content
            title = content.get('title', '')
            
            # Publisher info
            provider = content.get('provider', {})
            if isinstance(provider, dict):
                publisher = provider.get('displayName', '')
            else:
                publisher = str(provider) if provider else ''
            
            # Link/URL
            link = content.get('clickThroughUrl', {}).get('url', '') if isinstance(
                content.get('clickThroughUrl'), dict
            ) else content.get('url', '')
            
            # Published date - try multiple formats
            pub_date = content.get('pubDate', '')
            published_at = ''
            if pub_date:
                try:
                    # Parse ISO format like "2025-12-17T09:50:00.000Z"
                    dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    published_at = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    published_at = pub_date
            
            # Thumbnail
            thumbnail = ''
            thumb = content.get('thumbnail')
            if thumb:
                if isinstance(thumb, dict):
                    resolutions = thumb.get('resolutions', [])
                    if resolutions and len(resolutions) > 0:
                        thumbnail = resolutions[0].get('url', '')
                elif isinstance(thumb, str):
                    thumbnail = thumb
            
            # Related tickers
            finance = content.get('finance', {})
            related = finance.get('stockTickers', []) if isinstance(finance, dict) else []
            
            cleaned_article = {
                'title': title,
                'publisher': publisher,
                'link': link,
                'published_at': published_at,
                'type': content.get('contentType', ''),
                'thumbnail': thumbnail,
                'related_tickers': related,
                'summary': content.get('summary', '')
            }
            cleaned_news.append(cleaned_article)
        
        # Sort by published date (newest first)
        cleaned_news.sort(key=lambda x: x['published_at'], reverse=True)
        
        return cleaned_news
    
    except Exception as e:
        print(f"[ERROR] Failed to download news for {ticker}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return []


def save_news_json(news_data: list[dict], out_path: Path):
    """Save news data to JSON file."""
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(news_data, f, indent=2, ensure_ascii=False)


def save_news_csv(news_data: list[dict], out_path: Path):
    """Save news data to CSV file."""
    if not news_data:
        return
    
    df = pd.DataFrame(news_data)
    # Convert list columns to JSON string for CSV compatibility
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )
    df.to_csv(out_path, index=False, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(
        description="Download news articles from Yahoo Finance for FX pairs"
    )
    ap.add_argument("--pair", help="Single pair, e.g., USDIDR")
    ap.add_argument("--pairs", nargs="*", help="Multiple pairs, e.g., USDIDR USDEUR USDJPY")
    ap.add_argument("--all", action="store_true", help="Download news for all available pairs")
    ap.add_argument("--ticker", help="Direct Yahoo ticker override, e.g., IDR=X")
    ap.add_argument("--out-dir", default="data/fx_news", help="Output directory for news data")
    ap.add_argument(
        "--format", 
        default="json", 
        choices=["json", "csv", "both"], 
        help="Output format (json, csv, or both)"
    )
    ap.add_argument("--debug", action="store_true", help="Print debug information")
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
            print(
                f"[WARN] No ticker mapping for {p}. Add to PAIR_TO_TICKER or use --ticker.", 
                file=sys.stderr
            )
        else:
            jobs.append((p, tkr))

    if args.pairs:
        for p in args.pairs:
            p = p.upper()
            tkr = PAIR_TO_TICKER.get(p, "")
            if not tkr:
                print(
                    f"[WARN] No ticker mapping for {p}. Add to PAIR_TO_TICKER or use --ticker.", 
                    file=sys.stderr
                )
                continue
            jobs.append((p, tkr))

    if not jobs:
        print("No valid download jobs. Example: --pairs USDIDR USDEUR", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Will download news for {len(jobs)} ticker(s)\n")

    for name, tkr in jobs:
        print(f"[INFO] Downloading news for {name} ({tkr}) ...")
        news_data = download_news(tkr, debug=args.debug)
        
        if not news_data:
            print(f"[WARN] No news found for {name} ({tkr})\n", file=sys.stderr)
            continue
        
        # Save in requested format(s)
        if args.format in ["json", "both"]:
            json_path = out_dir / f"{name}_news.json"
            save_news_json(news_data, json_path)
            print(f"[OK] Saved {len(news_data)} articles -> {json_path}")
        
        if args.format in ["csv", "both"]:
            csv_path = out_dir / f"{name}_news.csv"
            save_news_csv(news_data, csv_path)
            print(f"[OK] Saved {len(news_data)} articles -> {csv_path}")
        
        print()


if __name__ == "__main__":
    main()