#!/usr/bin/env python3
"""
Runs experiments for:
- Ablation 1: News contribution (w/ vs w/o news)
- Ablation 2: Prompting strategy (zero-shot vs few-shot)
- Ablation 3: Interaction effect (news x prompt)

Example:
    python experiment_runner.py --currency USDIDR --horizon 5 --output results/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import numpy as np
import pandas as pd


class ExperimentConfig:
    """Configuration for ablation experiments."""
    
    def __init__(self):
        # Horizons
        self.horizons = {
            'short': 1,
            'medium': 5,
            'long': 10
        }
        
        # Models
        self.llm_models = [
            'llama3-8b',
            'qwen2.5-7b',
            'mistral-7b'
        ]
        
        self.baseline_models = [
            'arima',
            'lstm',
            'xgboost',
            'itransformer'
        ]
        
        # Ablation settings
        self.ablation_configs = {
            # Ablation 1: News contribution
            'news_ablation': [
                {'news': False, 'prompt': 'zero'},  # baseline LLM
                {'news': True, 'prompt': 'zero'},   # news only
            ],
            
            # Ablation 2: Prompting strategy
            'prompt_ablation': [
                {'news': False, 'prompt': 'zero'},  # zero-shot
                {'news': False, 'prompt': 'few'},   # few-shot
            ],
            
            # Ablation 3: Interaction effect (NEWS × PROMPT)
            'interaction_ablation': [
                {'news': False, 'prompt': 'zero'},  # baseline LLM
                {'news': True, 'prompt': 'zero'},   # news only
                {'news': False, 'prompt': 'few'},   # prompt only
                {'news': True, 'prompt': 'few'},    # full model
            ]
        }


class ExperimentRunner:
    """Run and manage forecasting experiments."""
    
    def __init__(self, output_dir: str = 'results'):
        self.config = ExperimentConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize result storage
        self.results = {
            'metadata': {},
            'baseline_results': {},
            'llm_results': {},
            'ablation_results': {}
        }
    
    def run_baseline_experiments(self, data: pd.DataFrame, currency: str):
        """
        Run baseline model experiments.
        
        Args:
            data: DataFrame with 'date' and 'close' columns
            currency: Currency pair name
        """
        print(f"\n{'='*60}")
        print(f"Running Baseline Experiments: {currency}")
        print(f"{'='*60}\n")
        
        for model_name in self.config.baseline_models:
            print(f"Training {model_name.upper()}...")
            
            for horizon_name, horizon in self.config.horizons.items():
                print(f"  Horizon: {horizon_name} (h={horizon})")
                
                # Placeholder for actual model training and prediction
                # In real implementation, call the baseline models here
                result = self._run_baseline_model(data, model_name, horizon)
                
                # Store results
                key = f"{model_name}_{horizon_name}"
                self.results['baseline_results'][key] = result
                
                print(f"    RMSE: {result['rmse']:.4f}, DA: {result['da']:.2f}%")
        
        print(f"\n{'='*60}\n")
    
    def run_llm_ablation_experiments(self, data: pd.DataFrame, news_data: Dict,
                                     currency: str, ablation_type: str = 'all'):
        """
        Run LLM ablation experiments.
        
        Args:
            data: Price data DataFrame
            news_data: Dictionary of news articles
            currency: Currency pair name
            ablation_type: Which ablation to run ('news', 'prompt', 'interaction', 'all')
        """
        print(f"\n{'='*60}")
        print(f"Running LLM Ablation Experiments: {currency}")
        print(f"{'='*60}\n")
        
        ablations_to_run = []
        if ablation_type == 'all':
            ablations_to_run = ['news_ablation', 'prompt_ablation', 'interaction_ablation']
        else:
            ablations_to_run = [f'{ablation_type}_ablation']
        
        for ablation_name in ablations_to_run:
            print(f"\n{ablation_name.upper().replace('_', ' ')}")
            print(f"{'-'*60}")
            
            configs = self.config.ablation_configs[ablation_name]
            
            for llm_model in self.config.llm_models:
                print(f"\nModel: {llm_model}")
                
                for config in configs:
                    news_flag = config['news']
                    prompt_type = config['prompt']
                    
                    config_name = f"news={news_flag}_prompt={prompt_type}"
                    print(f"  Config: {config_name}")
                    
                    for horizon_name, horizon in self.config.horizons.items():
                        # Placeholder for actual LLM prediction
                        result = self._run_llm_model(
                            data, news_data, llm_model, 
                            news_flag, prompt_type, horizon
                        )
                        
                        # Store results
                        key = f"{llm_model}_{ablation_name}_{config_name}_{horizon_name}"
                        self.results['llm_results'][key] = result
                        
                        print(f"    {horizon_name} (h={horizon}): RMSE={result['rmse']:.4f}, DA={result['da']:.2f}%")
        
        print(f"\n{'='*60}\n")
    
    def _run_baseline_model(self, data: pd.DataFrame, model_name: str, horizon: int) -> Dict:
        """
        Placeholder for running a baseline model.
        
        Replace this with actual model implementation.
        """
        # This is a mock implementation
        # In real code, load and run the actual baseline model
        
        # Mock predictions (replace with actual model)
        np.random.seed(42)
        n_test = 100
        y_true = data['close'].values[-n_test:]
        y_pred = y_true + np.random.randn(n_test) * 0.5
        
        # Calculate metrics (using evaluation_metrics.py)
        from evaluation_metrics import calculate_all_metrics
        metrics = calculate_all_metrics(y_true, y_pred)
        
        return {
            'model': model_name,
            'horizon': horizon,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'mape': metrics['mape'],
            'da': metrics['da'],
            'predictions': y_pred.tolist()[:10]  # Store only first 10 for brevity
        }
    
    def _run_llm_model(self, data: pd.DataFrame, news_data: Dict, 
                      model_name: str, use_news: bool, prompt_type: str, 
                      horizon: int) -> Dict:
        """
        Placeholder for running an LLM model.
        
        Replace this with actual LLM API calls.
        """
        # This is a mock implementation
        # In real code:
        # 1. Prepare prompt with/without news
        # 2. Call LLM API (Llama, Qwen, or Mistral)
        # 3. Parse LLM response
        # 4. Calculate metrics
        
        # Mock predictions (replace with actual LLM predictions)
        np.random.seed(hash(model_name + prompt_type) % 10000)
        n_test = 100
        y_true = data['close'].values[-n_test:]
        
        # Simulate: models with news perform slightly better
        noise_scale = 0.4 if use_news else 0.6
        y_pred = y_true + np.random.randn(n_test) * noise_scale
        
        from evaluation_metrics import calculate_all_metrics
        metrics = calculate_all_metrics(y_true, y_pred)
        
        return {
            'model': model_name,
            'horizon': horizon,
            'use_news': use_news,
            'prompt_type': prompt_type,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'mape': metrics['mape'],
            'da': metrics['da'],
            'predictions': y_pred.tolist()[:10]
        }
    
    def save_results(self, currency: str):
        """Save all results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"results_{currency}_{timestamp}.json"
        
        # Add metadata
        self.results['metadata'] = {
            'currency': currency,
            'timestamp': timestamp,
            'horizons': self.config.horizons,
            'models': {
                'llm': self.config.llm_models,
                'baseline': self.config.baseline_models
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        return output_file
    
    def generate_summary_report(self, currency: str) -> pd.DataFrame:
        """Generate summary report of all experiments."""
        rows = []
        
        # Baseline results
        for key, result in self.results['baseline_results'].items():
            rows.append({
                'type': 'baseline',
                'model': result['model'],
                'horizon': result['horizon'],
                'config': 'baseline',
                'rmse': result['rmse'],
                'mae': result['mae'],
                'mape': result['mape'],
                'da': result['da']
            })
        
        # LLM results
        for key, result in self.results['llm_results'].items():
            config_str = f"news={result['use_news']}_prompt={result['prompt_type']}"
            rows.append({
                'type': 'llm',
                'model': result['model'],
                'horizon': result['horizon'],
                'config': config_str,
                'rmse': result['rmse'],
                'mae': result['mae'],
                'mape': result['mape'],
                'da': result['da']
            })
        
        df = pd.DataFrame(rows)
        
        # Save summary
        summary_file = self.output_dir / f"summary_{currency}.csv"
        df.to_csv(summary_file, index=False)
        print(f"✓ Summary saved to: {summary_file}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description="Run FX forecasting experiments")
    parser.add_argument('--currency', type=str, required=True, help='Currency pair (e.g., USDIDR)')
    parser.add_argument('--data-dir', type=str, default='data/fx_filled', help='Data directory')
    parser.add_argument('--news-dir', type=str, default='data/fx_news', help='News directory')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--ablation', type=str, default='all', 
                       choices=['news', 'prompt', 'interaction', 'all'],
                       help='Which ablation to run')
    parser.add_argument('--baseline-only', action='store_true', help='Run only baseline models')
    parser.add_argument('--llm-only', action='store_true', help='Run only LLM models')
    
    args = parser.parse_args()
    
    # Load data
    print(f"\nLoading data for {args.currency}...")
    data_file = Path(args.data_dir) / f"{args.currency}.csv"
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        return
    
    data = pd.read_csv(data_file)
    print(f"✓ Loaded {len(data)} rows of price data")
    
    # Load news
    news_file = Path(args.news_dir) / f"{args.currency}_news.json"
    news_data = {}
    if news_file.exists():
        with open(news_file, 'r') as f:
            news_data = json.load(f)
        print(f"✓ Loaded {len(news_data)} news articles")
    else:
        print(f"Warning: No news file found at {news_file}")
    
    # Initialize experiment runner
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    # Run experiments
    if not args.llm_only:
        runner.run_baseline_experiments(data, args.currency)
    
    if not args.baseline_only:
        runner.run_llm_ablation_experiments(data, news_data, args.currency, args.ablation)
    
    # Save results
    runner.save_results(args.currency)
    
    # Generate summary
    summary_df = runner.generate_summary_report(args.currency)
    print("\nExperiment Summary:")
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("Experiments completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()