#!/usr/bin/env python3
"""
run_baselines_only.py — Re-run only the baseline phase and merge into existing LLM results.

Use this to fix a specific data issue: window_results.csv was found to contain
only LLM rows (5400 rows: llama3-8b/qwen2.5-7b/mistral-7b), with NO baseline
rows, even though ablation_results.json's baseline_results (aggregate metrics)
were intact with 105 rows across 7 baseline models. The root cause is that
window-level per-window rows for baselines were only ever held in the
AblationStudyRunner.window_rows in-memory list, which is NOT part of
self.results and therefore was never persisted to ablation_results.json --
so once window_results.csv got overwritten by a later merge step (e.g.
run_llama3_only.py), the baseline window rows had no durable source to
recover from.

This script re-runs ONLY the baseline phase (ARIMA, LSTM, XGBoost,
iTransformer, VanillaTransformer, Naive, MovingAverage -- all deterministic,
CPU/GPU but no vLLM needed, so this is fast compared to the LLM phase),
producing fresh baseline window rows, then merges them with the EXISTING
llm_results (Qwen/Mistral/Llama3, already complete) to regenerate all
summary CSVs and window_results.csv correctly.

Usage:
    python3 run_baselines_only.py --results-dir ../results/ \\
        --data-dir ../data/fx_filled --news-dir ../data/fx_news

Safety: overwrites baseline_results in ablation_results.json with the
freshly computed values (should be numerically close to the prior run,
since all baseline models are deterministic given the same data and
config). llm_results are preserved unchanged.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_full_ablation import AblationStudyRunner


def load_existing(results_dir: Path):
    results_file = results_dir / "ablation_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"{results_file} not found.")
    with open(results_file) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="../results")
    ap.add_argument("--data-dir", default="../data/fx_filled")
    ap.add_argument("--news-dir", default="../data/fx_news")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    existing = load_existing(results_dir)

    print(f"Loaded existing results: "
          f"{len(existing['baseline_results'])} baseline rows (aggregate, to be replaced), "
          f"{len(existing['llm_results'])} LLM rows (preserved, Qwen/Mistral/Llama3).")

    runner = AblationStudyRunner(args.data_dir, args.news_dir, str(results_dir))

    # Seed llm_results from existing (unchanged) -- baseline_results and
    # window_rows start empty and will be freshly populated below.
    runner.results["llm_results"] = existing["llm_results"]
    runner.results["metadata"]["start_time"] = existing["metadata"].get("start_time")
    import pandas as pd
    runner.results["metadata"]["baselines_rerun_time"] = pd.Timestamp.now().isoformat()

    print("\n" + "=" * 60)
    print("RE-RUNNING BASELINE PHASE ONLY")
    print("  (ARIMA, LSTM, XGBoost, iTransformer, VanillaTransformer, Naive, MovingAverage)")
    print("  LLM results (Qwen/Mistral/Llama3) are preserved unchanged.")
    print("=" * 60)

    runner.run_all_baselines()   # populates runner.results["baseline_results"] AND runner.window_rows

    n_baseline_window_rows = sum(1 for r in runner.window_rows if r.get("type") == "baseline")
    print(f"\nBaseline phase complete: "
          f"{len(runner.results['baseline_results'])} aggregate rows, "
          f"{n_baseline_window_rows} window rows.")

    if n_baseline_window_rows == 0:
        print("[ERROR] No baseline window rows were produced -- something is still wrong.")
        print("        Check _run_baseline_single() in run_full_ablation.py for issues")
        print("        before proceeding (window_results.csv would remain incomplete).")
        sys.exit(1)

    # IMPORTANT: we still need the LLM window rows merged back in, since
    # runner.window_rows currently only has the freshly-computed baseline rows.
    # These were lost from persistent storage the same way baseline rows were,
    # so we reconstruct them from the existing window_results.csv (LLM-only,
    # 5400 rows) rather than re-running any LLM inference.
    window_file = results_dir / "window_results.csv"
    if window_file.exists():
        existing_window_df = pd.read_csv(window_file)
        existing_llm_window_rows = existing_window_df[
            existing_window_df["type"] == "llm"
        ].to_dict(orient="records")
        print(f"Merging in {len(existing_llm_window_rows)} existing LLM window rows "
              f"from {window_file} (not re-run).")
        runner.window_rows = runner.window_rows + existing_llm_window_rows
    else:
        print("[WARN] window_results.csv not found -- LLM window rows will be missing "
              "from the merged output. DM tests for RQ2-4 will not be computable "
              "until the LLM phase is re-run.")

    runner.save_results()
    runner.generate_summary()

    print("\n" + "=" * 60)
    print(f"DONE. Total window_rows: {len(runner.window_rows)} "
          f"({n_baseline_window_rows} baseline + "
          f"{len(runner.window_rows) - n_baseline_window_rows} llm)")
    print(f"Total baseline_results: {len(runner.results['baseline_results'])}")
    print(f"Total llm_results: {len(runner.results['llm_results'])}")
    print(f"All summary CSVs re-saved to {results_dir}.")
    print("=" * 60)


if __name__ == "__main__":
    main()