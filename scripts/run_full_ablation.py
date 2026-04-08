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

LLM architecture: ChatTime (Wang et al., 2024) — time series data is normalized,
discretized, and serialized into ###value### tokens before entering the prompt.
This is the base architecture for all LLM configs.

Ablation study map (2x2 factorial, all using ChatTime tokenization):
  baseline_llm : news=False, zero-shot
  news_only    : news=True,  zero-shot
  prompt_only  : news=False, few-shot
  full_model   : news=True,  few-shot

Usage:
    python3 run_full_ablation.py --output results/
    python3 run_full_ablation.py --output results/ --skip-explain  # skip explainability
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
from llm_api_helper import LLMForecaster
from evaluation_metrics import calculate_all_metrics
from ts_tokenizer import TSFLPromptBuilder


# -- Server config: one model per GPU -----------------------------------------
MEM_UTIL = 0.70  # 70% per GPU
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
            "--max-model-len",          "4096",
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
            # All configs use ChatTime tokenization (###value### format) as base architecture.
            # Ablation varies only: news augmentation (True/False) x prompting strategy (zero/few).
            "baseline_llm": {"news": False, "prompt": "tsfl_zero"},
            "news_only":    {"news": True,  "prompt": "tsfl_zero"},
            "prompt_only":  {"news": False, "prompt": "tsfl_few"},
            "full_model":   {"news": True,  "prompt": "tsfl_few"},
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
        """
        Build ChatTime prompt for a given config.
        All configs use TSFLPromptBuilder (###value### tokenization).
        Ablation varies only news augmentation and zero-shot vs few-shot.
        Applies Qwen context reduction (Fix 5).
        """
        # Fix 5: Reduce context for Qwen to prevent "overthinking"
        n_examples = 1 if llm_model == "qwen2.5-7b" else 3
        max_news   = 5 if llm_model == "qwen2.5-7b" else (3 if llm_model == "mistral-7b" else 10)

        tsfl_builder = TSFLPromptBuilder(currency_pair=currency, horizon=horizon)
        use_news = config["news"]
        use_few  = config["prompt"] == "tsfl_few"

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

    def _run_llm_single(self, currency, horizon, horizon_name,
                        data, news, llm_model, forecaster):
        """
        Rolling window evaluation (Fix 4).

        Slides a 30-day context window across the test period (10 windows,
        step=10 days) so metrics are averaged over multiple market regimes.
        All configs use ChatTime (TSFLPromptBuilder) — responses are decoded
        via TSFLTokenizer.decode().
        """
        print(f"\n  [{currency}] {horizon_name} ({horizon}d)")

        LOOKBACK  = 30
        N_WINDOWS = 10
        STEP      = 10

        if len(data) < LOOKBACK + (horizon + STEP) * N_WINDOWS:
            N_WINDOWS = 1
            print(f"    [INFO] Short data — using single window")

        results = []

        for config_name, config in self.ablation_configs.items():
            try:
                all_y_true, all_y_pred = [], []

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

                    # All configs use ChatTime: LLM outputs ###tokens### → decode via TSFLTokenizer
                    tsfl_builder = TSFLPromptBuilder(currency_pair=currency, horizon=horizon)
                    raw_response = forecaster.api.call(prompt, model=llm_model, is_tsfl=True)
                    ref_prices   = context_data["close"].values[-LOOKBACK:]
                    pred         = tsfl_builder.decode_response(raw_response, ref_prices, horizon)
                        
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

    def run_all(self):
        start_time = datetime.now()
        print("\n" + "="*60)
        print("STARTING COMPLETE ABLATION STUDY")
        print(f"  Architecture : ChatTime (###value### tokenization)")
        print(f"  GPU 0 -> llama3-8b  (port 18000)")
        print(f"  GPU 1 -> qwen2.5-7b (port 18001)")
        print(f"  GPU 2 -> mistral-7b (port 18002)")
        print(f"  Ablation 2x2 : news x prompting strategy")
        print("="*60)

        self.run_all_baselines()
        self.run_all_llms()

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
    parser.add_argument("--data-dir", default="data/fx_filled")
    parser.add_argument("--news-dir", default="data/fx_news")
    parser.add_argument("--output",   default="results")
    args = parser.parse_args()

    # Kill only this user's stale vLLM servers (if any) before starting new ones
    subprocess.run("pkill -u $USER -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true",
                   shell=True, capture_output=True)
    time.sleep(3)

    runner = AblationStudyRunner(args.data_dir, args.news_dir, args.output)
    runner.run_all()


if __name__ == "__main__":
    main()