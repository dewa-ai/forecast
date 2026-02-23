#!/usr/bin/env python3
"""
Complete Ablation Study Runner for FX Forecasting.

Runs all 5 ablation configurations across:
- 5 currencies (USDIDR, USDEUR, USDSGD, USDTWD, USDAUD)
- 3 horizons (1, 5, 10 days)
- 3 LLM models (Llama, Qwen, Mistral)
- 4 baseline models (ARIMA, LSTM, XGBoost, iTransformer)

Total experiments: 5 x 3 x (3x4 baselines + 3 LLMsx4 configs) = 285 experiments

Usage:
    python3 scripts/run_full_ablation.py --output results/
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from baseline_models import ARIMAForecaster, LSTMForecaster, XGBoostForecaster
from itransformer_model import iTransformerForecaster
from llm_prompts import PromptBuilder
from llm_api_helper import LLMForecaster
from evaluation_metrics import calculate_all_metrics, compare_models_dm


class AblationStudyRunner:
    """Run complete ablation study."""
    
    def __init__(self, data_dir: str, news_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.news_dir = Path(news_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.currencies = ["USDIDR", "USDEUR", "USDSGD", "USDTWD", "USDAUD"]
        self.horizons = {
            "short": 1,
            "medium": 5,
            "long": 10
        }
        self.llm_models = ["llama3-8b", "qwen2.5-7b", "mistral-7b"]
        self.baseline_models = ["arima", "lstm", "xgboost", "itransformer"]
        
        # Ablation configurations
        self.ablation_configs = {
            "baseline_llm": {"news": False, "prompt": "zero"},  # Q1 baseline
            "news_only": {"news": True, "prompt": "zero"},      # Q1: news contribution
            "prompt_only": {"news": False, "prompt": "few"},    # Q2: prompting strategy
            "full_model": {"news": True, "prompt": "few"},      # Q3: full model (interaction)
        }
        
        # Results storage
        self.results = {
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "currencies": self.currencies,
                "horizons": self.horizons,
                "models": {
                    "llm": self.llm_models,
                    "baseline": self.baseline_models
                }
            },
            "baseline_results": [],
            "llm_results": []
        }
    
    def run_baseline(self, currency: str, horizon: int, data: pd.DataFrame) -> dict:
        """Run baseline model experiments."""
        print(f"\n{'='*60}")
        print(f"Baseline Models: {currency} (Horizon={horizon})")
        print(f"{'='*60}")
        
        # Train/test split (80/20)
        split_idx = int(len(data) * 0.8)
        train = data[:split_idx]['close'].values
        test = data[split_idx:]['close'].values
        
        # Limit test size for efficiency
        test_size = min(100, len(test))
        
        results = []
        
        for model_name in self.baseline_models:
            print(f"\n[{model_name.upper()}]")
            
            try:
                # Initialize model
                if model_name == "arima":
                    model = ARIMAForecaster(order=(1, 1, 1))
                    model.fit(train)
                    pred = model.predict(horizon=test_size)
                
                elif model_name == "lstm":
                    model = LSTMForecaster(seq_length=30, epochs=20, batch_size=32)
                    model.fit(train)
                    pred = model.predict(train, horizon=test_size)
                
                elif model_name == "xgboost":
                    model = XGBoostForecaster(n_lags=30)
                    model.fit(train)
                    pred = model.predict(horizon=test_size)
                
                elif model_name == "itransformer":
                    model = iTransformerForecaster(
                        seq_len=30, pred_len=horizon,
                        d_model=64, n_heads=4, n_layers=2,
                        epochs=10
                    )
                    model.fit(train)
                    pred = model.predict(train, horizon=test_size)
                
                # Ensure prediction length matches
                if len(pred) > test_size:
                    pred = pred[:test_size]
                elif len(pred) < test_size:
                    pred = np.concatenate([pred, np.full(test_size - len(pred), pred[-1])])
                
                # Evaluate
                y_true = test[:test_size]
                metrics = calculate_all_metrics(y_true, pred)
                
                result = {
                    "currency": currency,
                    "horizon": horizon,
                    "model": model_name,
                    "rmse": float(metrics["rmse"]),
                    "mae": float(metrics["mae"]),
                    "mape": float(metrics["mape"]),
                    "da": float(metrics["da"])
                }
                results.append(result)
                
                print(f"  RMSE: {metrics['rmse']:.4f} | DA: {metrics['da']:.2f}%")
            
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                continue
        
        return results
    
    def run_llm_ablation(self, currency: str, horizon: int, 
                        data: pd.DataFrame, news: list) -> dict:
        """Run LLM ablation experiments."""
        print(f"\n{'='*60}")
        print(f"LLM Ablation: {currency} (Horizon={horizon})")
        print(f"{'='*60}")
        
        # Train/test split
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        test_size = min(100, len(test_data))
        test_subset = test_data[-test_size:]
        
        # Use train + recent test as context
        context_data = pd.concat([train_data.tail(500), test_subset])
        
        results = []
        
        # Initialize LLM forecaster
        try:
            forecaster = LLMForecaster(backend='vllm')
        except Exception as e:
            print(f"ERROR: Could not initialize LLM forecaster: {e}")
            print("Make sure vLLM servers are running!")
            return results
        
        # Run ablation for each LLM model
        for llm_model in self.llm_models:
            print(f"\n[{llm_model.upper()}]")
            
            for config_name, config in self.ablation_configs.items():
                print(f"  Config: {config_name} (news={config['news']}, prompt={config['prompt']})")
                
                try:
                    # Build prompt
                    builder = PromptBuilder(currency_pair=currency, horizon=horizon)
                    
                    if config["news"] and config["prompt"] == "few":
                        prompt = builder.build_few_shot_with_news(
                            context_data, news, lookback_days=30, n_examples=3
                        )
                    elif config["news"] and config["prompt"] == "zero":
                        prompt = builder.build_news_augmented(
                            context_data, news, lookback_days=30
                        )
                    elif not config["news"] and config["prompt"] == "few":
                        prompt = builder.build_few_shot(
                            context_data, lookback_days=30, n_examples=3
                        )
                    else:  # baseline_llm
                        prompt = builder.build_zero_shot(
                            context_data, lookback_days=30
                        )
                    
                    # Get prediction
                    pred_raw = forecaster.predict(prompt, model=llm_model)
                    
                    # Extend to test size
                    if len(pred_raw) < test_size:
                        pred = np.tile(pred_raw, (test_size // len(pred_raw)) + 1)[:test_size]
                    else:
                        pred = pred_raw[:test_size]
                    
                    # Evaluate
                    y_true = test_subset['close'].values[:test_size]
                    metrics = calculate_all_metrics(y_true, pred)
                    
                    result = {
                        "currency": currency,
                        "horizon": horizon,
                        "model": llm_model,
                        "config": config_name,
                        "news": config["news"],
                        "prompt": config["prompt"],
                        "rmse": float(metrics["rmse"]),
                        "mae": float(metrics["mae"]),
                        "mape": float(metrics["mape"]),
                        "da": float(metrics["da"])
                    }
                    results.append(result)
                    
                    print(f"    RMSE: {metrics['rmse']:.4f} | DA: {metrics['da']:.2f}%")
                
                except Exception as e:
                    print(f"    ERROR: {str(e)}")
                    continue
        
        return results
    
    def run_currency(self, currency: str):
        """Run all experiments for one currency."""
        print(f"\n{'#'*60}")
        print(f"# CURRENCY: {currency}")
        print(f"{'#'*60}")
        
        # Load data
        data_file = self.data_dir / f"{currency}.csv"
        if not data_file.exists():
            print(f"ERROR: Data file not found: {data_file}")
            return
        
        data = pd.read_csv(data_file)
        print(f"Loaded {len(data)} rows")
        
        # Load news
        news_file = self.news_dir / f"{currency}_news.json"
        news = []
        if news_file.exists():
            with open(news_file) as f:
                news = json.load(f)
            print(f"Loaded {len(news)} news articles")
        
        # Run experiments for each horizon
        for horizon_name, horizon in self.horizons.items():
            print(f"\n{'*'*60}")
            print(f"* HORIZON: {horizon_name} ({horizon} days)")
            print(f"{'*'*60}")
            
            # Baseline models
            baseline_results = self.run_baseline(currency, horizon, data)
            self.results["baseline_results"].extend(baseline_results)
            
            # LLM ablation
            llm_results = self.run_llm_ablation(currency, horizon, data, news)
            self.results["llm_results"].extend(llm_results)
            
            # Save intermediate results
            self.save_results()
    
    def run_all(self):
        """Run experiments for all currencies."""
        start_time = datetime.now()
        
        print("\n" + "="*60)
        print("STARTING COMPLETE ABLATION STUDY")
        print("="*60)
        print(f"Currencies: {len(self.currencies)}")
        print(f"Horizons: {len(self.horizons)}")
        print(f"Baseline models: {len(self.baseline_models)}")
        print(f"LLM models: {len(self.llm_models)}")
        print(f"Ablation configs: {len(self.ablation_configs)}")
        print("="*60)
        
        for i, currency in enumerate(self.currencies, 1):
            print(f"\n[{i}/{len(self.currencies)}] Processing {currency}...")
            self.run_currency(currency)
        
        # Final save
        self.results["metadata"]["end_time"] = datetime.now().isoformat()
        elapsed = datetime.now() - start_time
        self.results["metadata"]["duration_minutes"] = elapsed.total_seconds() / 60
        
        self.save_results()
        self.generate_summary()
        
        print("\n" + "="*60)
        print("ABLATION STUDY COMPLETED!")
        print("="*60)
        print(f"Duration: {elapsed.total_seconds()/60:.1f} minutes")
        print(f"Results saved to: {self.output_dir}")
    
    def save_results(self):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"ablation_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n[SAVED] {output_file}")
    
    def generate_summary(self):
        """Generate summary CSV."""
        # Baseline summary
        baseline_df = pd.DataFrame(self.results["baseline_results"])
        if not baseline_df.empty:
            baseline_df.to_csv(self.output_dir / "baseline_summary.csv", index=False)
        
        # LLM summary
        llm_df = pd.DataFrame(self.results["llm_results"])
        if not llm_df.empty:
            llm_df.to_csv(self.output_dir / "llm_summary.csv", index=False)
        
        # Combined summary
        combined = []
        
        for result in self.results["baseline_results"]:
            combined.append({
                "type": "baseline",
                "currency": result["currency"],
                "horizon": result["horizon"],
                "model": result["model"],
                "config": "baseline",
                "rmse": result["rmse"],
                "mae": result["mae"],
                "mape": result["mape"],
                "da": result["da"]
            })
        
        for result in self.results["llm_results"]:
            combined.append({
                "type": "llm",
                "currency": result["currency"],
                "horizon": result["horizon"],
                "model": result["model"],
                "config": result["config"],
                "rmse": result["rmse"],
                "mae": result["mae"],
                "mape": result["mape"],
                "da": result["da"]
            })
        
        combined_df = pd.DataFrame(combined)
        combined_df.to_csv(self.output_dir / "combined_summary.csv", index=False)
        
        print(f"\n[SAVED] Summary CSVs to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run complete ablation study")
    parser.add_argument("--data-dir", default="data/fx_filled", help="FX data directory")
    parser.add_argument("--news-dir", default="data/fx_news", help="News directory")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()
    
    runner = AblationStudyRunner(args.data_dir, args.news_dir, args.output)
    runner.run_all()


if __name__ == "__main__":
    main()