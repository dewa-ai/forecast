#!/usr/bin/env python3
"""
Complete Ablation Study Runner for FX Forecasting.

Runs all ablation configurations across:
- 5 currencies (USDIDR, USDEUR, USDSGD, USDTWD, USDAUD)
- 3 horizons (1, 5, 10 days)
- 3 LLM models (Llama, Qwen, Mistral) — each on its own GPU, started in parallel
- 4 baseline models (ARIMA, LSTM, XGBoost, iTransformer)

GPU assignment:
  GPU 0 -> Llama3-8B  (port 18000)
  GPU 1 -> Qwen2.5-7B (port 18001)
  GPU 2 -> Mistral-7B (port 18002)

Ablation study map:
  Ablation 1 (News)        : news=False,zero  vs  news=True,zero
  Ablation 2 (Prompt)      : news=False,zero  vs  news=False,few
  Ablation 3 (Interaction) : 2x2 full factorial (news x prompt)
  Ablation 4 (Explain)     : explainability metrics (coherence, grounding, factor coverage)
  Ablation 5 (Corr)        : correlation between explanation quality and prediction accuracy
  TSFL                     : Time Series as Foreign Language (ChatTime-inspired tokenization)

Usage:
    python3 scripts/run_full_ablation.py --output results/
    python3 scripts/run_full_ablation.py --output results/ --skip-explain  # skip abl4&5
    python3 scripts/run_full_ablation.py --output results/ --skip-tsfl     # skip TSFL
"""

import argparse
import json
import os
import subprocess
import sys
import time
import requests
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from baseline_models import ARIMAForecaster, LSTMForecaster, XGBoostForecaster
from itransformer_model import iTransformerForecaster
from llm_prompts import PromptBuilder
from llm_api_helper import LLMForecaster
from evaluation_metrics import calculate_all_metrics
from ts_tokenizer import TSFLPromptBuilder
from explainability_ablation import ExplainabilityAblation


# -- Server config: one model per GPU -----------------------------------------
MEM_UTIL = 0.70  # 70% per GPU -- leaves headroom for other users
DTYPE    = "half"

LLM_SERVER_CONFIG = {
    "llama3-8b": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "port":     18000,
        "gpu_id":   0,
    },
    "qwen2.5-7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "port":     18001,
        "gpu_id":   1,
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "port":     18002,
        "gpu_id":   2,
    },
}


# -- Server helpers -----------------------------------------------------------

def start_all_servers() -> dict:
    """Launch all 3 vLLM servers in parallel, return {model_name: proc} for ready ones."""
    print("\n[SERVERS] Launching all 3 servers in parallel...")
    procs_raw = {}

    for model_name, cfg in LLM_SERVER_CONFIG.items():
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg["gpu_id"])}
        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model",                  cfg["model_id"],
            "--port",                   str(cfg["port"]),
            "--gpu-memory-utilization", str(MEM_UTIL),
            "--dtype",                  DTYPE,
        ]
        Path("logs").mkdir(exist_ok=True)
        log_file = open(f"logs/{model_name}.log", "w")
        procs_raw[model_name] = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
        print(f"  [{model_name}] Launched on GPU {cfg['gpu_id']}, port {cfg['port']}")

    # Poll until all ready (max 5 min)
    print("\n[SERVERS] Waiting for all servers to be ready...")
    ready = {}
    for elapsed in range(10, 310, 10):
        time.sleep(10)
        for model_name, proc in procs_raw.items():
            if model_name in ready:
                continue
            port = LLM_SERVER_CONFIG[model_name]["port"]
            try:
                r = requests.get(f"http://localhost:{port}/v1/models", timeout=3)
                if r.status_code == 200 and "id" in r.text:
                    ready[model_name] = proc
                    print(f"  [{model_name}] Ready after {elapsed}s")
            except Exception:
                pass

        waiting = [m for m in procs_raw if m not in ready]
        if waiting:
            print(f"  [{elapsed}s] Still waiting: {waiting}")
        if len(ready) == len(procs_raw):
            break

    # Kill any that timed out
    for model_name, proc in procs_raw.items():
        if model_name not in ready:
            print(f"  [{model_name}] TIMEOUT -- check logs/{model_name}.log")
            proc.kill()

    print(f"[SERVERS] Ready: {list(ready.keys())}")
    return ready


def stop_all_servers(procs: dict):
    """Gracefully terminate all vLLM server processes."""
    print("\n[SERVERS] Stopping all servers...")
    for model_name, proc in procs.items():
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(f"  [{model_name}] stopped")
    time.sleep(3)


# -- Main runner --------------------------------------------------------------

class AblationStudyRunner:

    def __init__(self, data_dir: str, news_dir: str, output_dir: str):
        self.data_dir   = Path(data_dir)
        self.news_dir   = Path(news_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.currencies      = ["USDIDR", "USDEUR", "USDSGD", "USDTWD", "USDAUD"]
        self.horizons        = {"short": 1, "medium": 5, "long": 10}
        self.llm_models      = list(LLM_SERVER_CONFIG.keys())
        self.baseline_models = ["arima", "lstm", "xgboost", "itransformer"]

        self.ablation_configs = {
            "baseline_llm": {"news": False, "prompt": "zero"},
            "news_only":    {"news": True,  "prompt": "zero"},
            "prompt_only":  {"news": False, "prompt": "few"},
            "full_model":   {"news": True,  "prompt": "few"},
            # Ablation TSFL -- Time Series as Foreign Language
            # Following ChatTime (Wang et al., 2024): normalize to [-0.5,0.5],
            # discretize to 10K bins, serialize as ###value### tokens.
            # 2x2 factorial: news x few-shot
            "tsfl_zero":     {"news": False, "prompt": "tsfl_zero"},
            "tsfl_news":     {"news": True,  "prompt": "tsfl_news"},
            "tsfl_few":      {"news": False, "prompt": "tsfl_few"},
            "tsfl_few_news": {"news": True,  "prompt": "tsfl_few_news"},
        }

        self.results = {
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "currencies": self.currencies,
                "horizons":   self.horizons,
                "models":     {"llm": self.llm_models, "baseline": self.baseline_models},
            },
            "baseline_results": [],
            "llm_results":      [],
        }

    def _load_data(self, currency: str):
        data_file = self.data_dir / f"{currency}.csv"
        if not data_file.exists():
            print(f"  ERROR: {data_file} not found")
            return None, []
        data = pd.read_csv(data_file)
        news = []
        news_file = self.news_dir / f"{currency}_news.json"
        if news_file.exists():
            with open(news_file) as f:
                news = json.load(f)
        print(f"  {currency}: {len(data)} rows, {len(news)} news")
        return data, news

    # -- Baseline phase -------------------------------------------------------

    def run_all_baselines(self):
        print("\n" + "="*60)
        print("PHASE 1: BASELINE MODELS")
        print("="*60)
        for currency in self.currencies:
            data, _ = self._load_data(currency)
            if data is None:
                continue
            for horizon_name, horizon in self.horizons.items():
                results = self._run_baseline_single(currency, horizon, horizon_name, data)
                self.results["baseline_results"].extend(results)
        self.save_results()

    def _run_baseline_single(self, currency, horizon, horizon_name, data):
        print(f"\n[{currency}] {horizon_name} ({horizon}d)")
        split_idx = int(len(data) * 0.8)
        train     = data[:split_idx]["close"].values
        test      = data[split_idx:]["close"].values
        test_size = min(100, len(test))
        results   = []

        for model_name in self.baseline_models:
            try:
                if model_name == "arima":
                    m = ARIMAForecaster(order=(1, 1, 1))
                    m.fit(train); pred = m.predict(horizon=test_size)
                elif model_name == "lstm":
                    m = LSTMForecaster(seq_length=30, epochs=50, batch_size=32)
                    m.fit(train); pred = m.predict(train, horizon=test_size)
                elif model_name == "xgboost":
                    m = XGBoostForecaster(n_lags=30)
                    m.fit(train); pred = m.predict(horizon=test_size)
                elif model_name == "itransformer":
                    m = iTransformerForecaster(
                        seq_len=30, pred_len=horizon,
                        d_model=64, n_heads=4, n_layers=2, epochs=30)
                    m.fit(train); pred = m.predict(train, horizon=test_size)

                if len(pred) > test_size:
                    pred = pred[:test_size]
                elif len(pred) < test_size:
                    pred = np.concatenate([pred, np.full(test_size - len(pred), pred[-1])])

                metrics = calculate_all_metrics(test[:test_size], pred)
                results.append({
                    "currency": currency, "horizon": horizon, "model": model_name,
                    "rmse": float(metrics["rmse"]), "mae": float(metrics["mae"]),
                    "mape": float(metrics["mape"]), "da": float(metrics["da"]),
                })
                print(f"  {model_name.upper()}: RMSE={metrics['rmse']:.4f} DA={metrics['da']:.2f}%")
            except Exception as e:
                print(f"  {model_name.upper()} ERROR: {e}")
        return results

    # -- LLM phase ------------------------------------------------------------

    def run_all_llms(self):
        print("\n" + "="*60)
        print("PHASE 2: LLM ABLATION (parallel -- one model per GPU)")
        print("="*60)

        procs = start_all_servers()
        if not procs:
            print("ERROR: No LLM servers started. Skipping LLM phase.")
            return

        try:
            forecaster = LLMForecaster(backend="vllm")

            for llm_model in self.llm_models:
                if llm_model not in procs:
                    print(f"\nSkipping {llm_model} -- server failed to start")
                    continue

                print(f"\n{'#'*60}\n# LLM: {llm_model}\n{'#'*60}")

                for currency in self.currencies:
                    data, news = self._load_data(currency)
                    if data is None:
                        continue
                    for horizon_name, horizon in self.horizons.items():
                        results = self._run_llm_single(
                            currency, horizon, horizon_name,
                            data, news, llm_model, forecaster)
                        self.results["llm_results"].extend(results)

                self.save_results()

        finally:
            stop_all_servers(procs)

    def _build_prompt(self, config_name, config, context_data, news,
                      currency, horizon, llm_model):
        """Build prompt for a given config. Applies Qwen context reduction (Fix 5)."""
        prompt_type = config["prompt"]

        # Fix 5: Reduce context for Qwen to prevent "overthinking"
        n_examples = 1 if llm_model == "qwen2.5-7b" else 3
        max_news   = 5 if llm_model == "qwen2.5-7b" else 10

        # --- TSFL: Time Series as Foreign Language (ChatTime, Wang et al. 2024) ---
        # 2x2 factorial: tsfl_zero, tsfl_news, tsfl_few, tsfl_few_news
        if prompt_type.startswith("tsfl_"):
            tsfl_builder = TSFLPromptBuilder(currency_pair=currency, horizon=horizon)
            use_news = config["news"]
            use_few  = prompt_type in ("tsfl_few", "tsfl_few_news")
            if use_few and use_news:
                return tsfl_builder.build_few_shot_with_news(
                    context_data, news, lookback_days=30,
                    n_examples=n_examples, max_news=max_news)
            elif use_few and not use_news:
                return tsfl_builder.build_few_shot(
                    context_data, lookback_days=30, n_examples=n_examples)
            elif not use_few and use_news:
                return tsfl_builder.build_with_news(
                    context_data, news, lookback_days=30, max_news=max_news)
            else:
                return tsfl_builder.build_zero_shot(context_data, lookback_days=30)

        # --- Standard LLM prompts ---
        builder = PromptBuilder(currency_pair=currency, horizon=horizon)
        if config["news"] and prompt_type == "few":
            return builder.build_few_shot_with_news(
                context_data, news, lookback_days=30,
                n_examples=n_examples, max_news=max_news)
        elif config["news"] and prompt_type == "zero":
            return builder.build_news_augmented(
                context_data, news, lookback_days=30, max_news=max_news)
        elif not config["news"] and prompt_type == "few":
            return builder.build_few_shot(
                context_data, lookback_days=30, n_examples=n_examples)
        else:
            return builder.build_zero_shot(context_data, lookback_days=30)

    def _run_llm_single(self, currency, horizon, horizon_name,
                        data, news, llm_model, forecaster):
        """
        Rolling window evaluation (Fix 4).

        Slides a 30-day context window across the test period (10 windows,
        step=5 days) so metrics are averaged over multiple market regimes.
        TSFL configs decode ###token### responses via TSFLTokenizer.decode().
        """
        print(f"\n  [{currency}] {horizon_name} ({horizon}d)")

        LOOKBACK  = 30
        N_WINDOWS = 10
        STEP      = 5

        if len(data) < LOOKBACK + (horizon + STEP) * N_WINDOWS:
            N_WINDOWS = 1
            print(f"    [INFO] Short data — using single window")

        results = []

        for config_name, config in self.ablation_configs.items():
            try:
                all_y_true, all_y_pred = [], []
                is_tsfl = config["prompt"].startswith("tsfl_")

                for w in range(N_WINDOWS):
                    window_end   = len(data) - (N_WINDOWS - w) * STEP - horizon
                    if window_end < LOOKBACK:
                        continue
                    context_data = data.iloc[max(0, window_end - 500):window_end]
                    y_true_arr   = data["close"].values[window_end:window_end + horizon]
                    if len(y_true_arr) < horizon:
                        continue

                    last_price = context_data["close"].iloc[-1]
                    prompt     = self._build_prompt(
                        config_name, config, context_data, news,
                        currency, horizon, llm_model)

                    if is_tsfl:
                        # TSFL: LLM outputs ###tokens### — use TSFLTokenizer to decode
                        tsfl_builder = TSFLPromptBuilder(currency_pair=currency, horizon=horizon)
                        raw_response = forecaster.api.call(prompt, model=llm_model, is_tsfl=True)
                        ref_prices   = context_data["close"].values[-LOOKBACK:]
                        pred         = tsfl_builder.decode_response(
                            raw_response, ref_prices, horizon)
                    else:
                        # Standard: LLM outputs JSON {"predictions": [...]}
                        pred = forecaster.predict(
                            prompt, model=llm_model, expected_scale=last_price)
                        pred = pred[:horizon] if len(pred) >= horizon else np.tile(pred, (horizon // len(pred)) + 1)[:horizon]
                        
                    # Fix 6: warn if flat predictions
                    if np.std(pred) < np.std(y_true_arr) * 0.001:
                        print(f"    [WARN] Flat predictions w={w} ({config_name})")

                    all_y_true.extend(y_true_arr)
                    all_y_pred.extend(pred)

                if not all_y_true:
                    raise ValueError("No valid windows")

                metrics = calculate_all_metrics(
                    np.array(all_y_true), np.array(all_y_pred))
                results.append({
                    "currency": currency, "horizon": horizon,
                    "model":    llm_model, "config": config_name,
                    "news":     config["news"], "prompt": config["prompt"],
                    "rmse": float(metrics["rmse"]), "mae":  float(metrics["mae"]),
                    "mape": float(metrics["mape"]), "da":   float(metrics["da"]),
                    "n_windows": N_WINDOWS,
                })
                print(f"    {config_name}: RMSE={metrics['rmse']:.4f} "
                      f"DA={metrics['da']:.2f}% (n={N_WINDOWS})")
            except Exception as e:
                print(f"    {config_name} ERROR: {e}")
        return results

    # -- Explainability phase (Ablation 4 & 5) ---------------------------------

    def run_all_explainability(self, skip: bool = False):
        if skip:
            print("\n[SKIP] Ablation 4 & 5 (--skip-explain flag)")
            return

        print("\n" + "="*60)
        print("PHASE 3: ABLATION 4 & 5 (EXPLAINABILITY)")
        print("="*60)

        procs = start_all_servers()
        if not procs:
            print("ERROR: No LLM servers. Skipping explainability ablation.")
            return

        try:
            forecaster = LLMForecaster(backend="vllm")
            abl = ExplainabilityAblation(forecaster, output_dir=str(self.output_dir))

            for currency in self.currencies:
                data_file = self.data_dir / f"{currency}.csv"
                news_file = self.news_dir / f"{currency}_news.json"
                if not data_file.exists():
                    continue
                data = pd.read_csv(data_file)
                news = json.load(open(news_file)) if news_file.exists() else []

                for horizon_name, horizon in self.horizons.items():
                    available_models = [m for m in self.llm_models if m in procs]
                    abl.run(data, news, currency, horizon, llm_models=available_models)

            print("\n[OK] Ablation 4 & 5 complete. Check results/ablation4_*.csv and ablation5_*.csv")
        finally:
            stop_all_servers(procs)

    # -- Save / summary -------------------------------------------------------

    def save_results(self):
        out = self.output_dir / "ablation_results.json"
        with open(out, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"[SAVED] {out}")

    def generate_summary(self):
        pd.DataFrame(self.results["baseline_results"]).to_csv(
            self.output_dir / "baseline_summary.csv", index=False)
        pd.DataFrame(self.results["llm_results"]).to_csv(
            self.output_dir / "llm_summary.csv", index=False)
        combined = (
            [{**r, "type": "baseline", "config": "baseline"}
             for r in self.results["baseline_results"]] +
            [{**r, "type": "llm"} for r in self.results["llm_results"]]
        )
        pd.DataFrame(combined).to_csv(
            self.output_dir / "combined_summary.csv", index=False)
        print(f"[SAVED] Summary CSVs -> {self.output_dir}")

    def run_all(self, skip_explain: bool = False, skip_tsfl: bool = False):
        start_time = datetime.now()
        print("\n" + "="*60)
        print("STARTING COMPLETE ABLATION STUDY")
        print(f"  GPU 0 -> llama3-8b  (port 18000)")
        print(f"  GPU 1 -> qwen2.5-7b (port 18001)")
        print(f"  GPU 2 -> mistral-7b (port 18002)")
        print(f"  Ablations 1-3: news/prompt/interaction")
        print(f"  Ablation TSFL: Time Series as Foreign Language {'[SKIP]' if skip_tsfl else ''}")
        print(f"  Ablation 4-5:  explainability + correlation {'[SKIP]' if skip_explain else ''}")
        print("="*60)

        # If skip_tsfl, remove TSFL configs from ablation_configs
        if skip_tsfl:
            self.ablation_configs = {
                k: v for k, v in self.ablation_configs.items()
                if not k.startswith("tsfl_")
            }

        self.run_all_baselines()
        self.run_all_llms()
        self.run_all_explainability(skip=skip_explain)

        elapsed = datetime.now() - start_time
        self.results["metadata"]["end_time"] = datetime.now().isoformat()
        self.results["metadata"]["duration_minutes"] = elapsed.total_seconds() / 60
        self.save_results()
        self.generate_summary()

        print("\n" + "="*60)
        print(f"COMPLETED in {elapsed.total_seconds()/60:.1f} minutes")
        print(f"Results -> {self.output_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",     default="data/fx_filled")
    parser.add_argument("--news-dir",     default="data/fx_news")
    parser.add_argument("--output",       default="results")
    parser.add_argument("--skip-explain", action="store_true",
                        help="Skip Ablation 4 & 5 (explainability)")
    parser.add_argument("--skip-tsfl",    action="store_true",
                        help="Skip TSFL (Time Series as Foreign Language) ablation")
    args = parser.parse_args()

    # Kill only this user's stale vLLM servers (if any) before starting new ones
    subprocess.run("pkill -u $USER -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true",
                   shell=True, capture_output=True)
    time.sleep(3)

    runner = AblationStudyRunner(args.data_dir, args.news_dir, args.output)
    runner.run_all(skip_explain=args.skip_explain, skip_tsfl=args.skip_tsfl)


if __name__ == "__main__":
    main()