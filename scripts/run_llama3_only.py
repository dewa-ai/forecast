#!/usr/bin/env python3
"""
run_llama3_only.py — Run only the Llama3-8B LLM stage and merge into existing results.

Use this after a run_full_ablation.py run where Llama3-8B's vLLM server failed
to start (e.g. port conflict) but baseline models + Qwen2.5-7B + Mistral-7B
already completed successfully.

What it does:
    1. Loads the existing ablation_results.json and window_results.csv from
       --results-dir (produced by the earlier run_full_ablation.py run).
    2. Starts only the Llama3-8B vLLM server (using LLAMA3_CONFIG from
       run_full_ablation.py, so the port fix -- 18003 instead of 18000 --
       is picked up automatically).
    3. Runs the same 5 currencies x 3 horizons x 4 ablation configs x 30
       rolling windows for Llama3-8B only.
    4. Merges the new Llama3 rows into the existing results (does not
       duplicate or overwrite Qwen/Mistral/baseline rows) and re-saves
       ablation_results.json + all summary CSVs.

Usage:
    python3 run_llama3_only.py --results-dir ../results/ \\
        --data-dir ../data/fx_filled --news-dir ../data/fx_news

Safety: if Llama3 rows already exist in llm_results (e.g. this script is
accidentally run twice), it skips re-running and warns instead of duplicating.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# Reuse everything from run_full_ablation.py -- config, server helpers, and the
# AblationStudyRunner class itself, so the Llama3 run uses IDENTICAL logic
# (prompt building, windowing, metrics) to the original run. This guarantees
# the merged results are consistent with the existing Qwen/Mistral rows.
from run_full_ablation import (
    AblationStudyRunner,
    LLAMA3_CONFIG,
    start_all_servers,
    stop_all_servers,
)
from llm_api_helper import LLMForecaster


def load_existing(results_dir: Path):
    """Load the existing ablation_results.json produced by the earlier run."""
    results_file = results_dir / "ablation_results.json"
    if not results_file.exists():
        raise FileNotFoundError(
            f"{results_file} not found. This script merges into an existing run; "
            f"if no prior run exists, use run_full_ablation.py instead."
        )
    with open(results_file) as f:
        existing = json.load(f)
    return existing


def load_existing_window_rows(results_dir: Path) -> list[dict]:
    """Load existing window_results.csv rows as a list of dicts, if present."""
    window_file = results_dir / "window_results.csv"
    if not window_file.exists():
        print(f"[WARN] {window_file} not found -- starting with empty window_rows.")
        return []
    df = pd.read_csv(window_file)
    return df.to_dict(orient="records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="../results")
    ap.add_argument("--data-dir", default="../data/fx_filled")
    ap.add_argument("--news-dir", default="../data/fx_news")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    existing = load_existing(results_dir)
    existing_window_rows = load_existing_window_rows(results_dir)

    already_have_llama3 = any(
        r.get("model") == "llama3-8b" for r in existing.get("llm_results", [])
    )
    if already_have_llama3:
        print("[WARN] llama3-8b results already present in ablation_results.json.")
        print("       Skipping to avoid duplicating rows. If you intended to")
        print("       re-run Llama3, remove its rows from ablation_results.json")
        print("       and window_results.csv first, then re-run this script.")
        return

    print(f"Loaded existing results: "
          f"{len(existing['baseline_results'])} baseline rows, "
          f"{len(existing['llm_results'])} LLM rows (Qwen/Mistral), "
          f"{len(existing_window_rows)} window rows.")

    # Build a runner instance to reuse its exact methods (_load_data, _run_llm_single,
    # ablation_configs, currencies, horizons) -- but seed its internal state with the
    # EXISTING results so nothing from the prior run is lost.
    runner = AblationStudyRunner(args.data_dir, args.news_dir, str(results_dir))
    runner.results["baseline_results"] = existing["baseline_results"]
    runner.results["llm_results"] = existing["llm_results"]
    runner.window_rows = existing_window_rows
    # Preserve original start_time for metadata continuity; note the merge time separately.
    runner.results["metadata"]["start_time"] = existing["metadata"].get("start_time")
    runner.results["metadata"]["llama3_merge_time"] = pd.Timestamp.now().isoformat()

    print("\n" + "=" * 60)
    print("RUNNING LLAMA3-8B ONLY (merging into existing results)")
    print(f"  Port: {LLAMA3_CONFIG['llama3-8b']['port']} "
          f"(changed from 18000 to avoid conflict with another user's process)")
    print(f"  GPU:  {LLAMA3_CONFIG['llama3-8b']['gpu_id']}")
    print("=" * 60)

    procs = start_all_servers(LLAMA3_CONFIG)
    if not procs:
        print("ERROR: Llama3-8B server still failed to start. Check logs/llama3-8b.log")
        print("       (e.g. run: cat logs/llama3-8b.log | tail -50)")
        sys.exit(1)

    # llm_api_helper.py's vLLMAPI class hardcodes llama3-8b's port as 18000 by
    # default; override it via env var to match LLAMA3_CONFIG's actual port,
    # so requests actually reach this script's server instead of port 18000
    # (which is occupied by another user's process on this shared server).
    import os
    os.environ["LLAMA3_PORT"] = str(LLAMA3_CONFIG["llama3-8b"]["port"])

    try:
        forecaster = LLMForecaster(backend="vllm")
        llm_model = "llama3-8b"

        print(f"\n{'#'*60}\n# LLM: {llm_model}\n{'#'*60}")

        for currency in runner.currencies:
            data, news = runner._load_data(currency)
            if data is None:
                continue
            for horizon_name, horizon in runner.horizons.items():
                results = runner._run_llm_single(
                    currency, horizon, horizon_name,
                    data, news, llm_model, forecaster)
                runner.results["llm_results"].extend(results)

        runner.save_results()

    finally:
        stop_all_servers(procs)

    runner.generate_summary()

    n_llama3 = sum(1 for r in runner.results["llm_results"] if r.get("model") == "llama3-8b")
    print("\n" + "=" * 60)
    print(f"DONE. Added {n_llama3} llama3-8b result rows.")
    print(f"Total llm_results: {len(runner.results['llm_results'])}")
    print(f"Total window_rows: {len(runner.window_rows)}")
    print(f"All summary CSVs re-saved to {results_dir} (baseline/Qwen/Mistral rows preserved).")
    print("=" * 60)


if __name__ == "__main__":
    main()