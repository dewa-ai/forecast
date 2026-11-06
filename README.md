# Financial Time Series Forecasting with Large Language Models

Code and experiment pipeline for the Master's thesis *"Financial Time Series Forecasting with Large Language Models"* (National Central University, International Graduate Program in AI; supervisor: Prof. Richard Tzong-Han Tsai).

Although the title refers to financial time series in general, the scope of this work is **foreign exchange (FX) rate forecasting**. The study evaluates whether small open-source LLMs, used without fine-tuning, can forecast FX rates, and how news augmentation and prompting strategy affect accuracy.

## Overview

| Item | Setting |
|---|---|
| Task | Daily FX rate forecasting, January 2016 to March 2026 |
| Currency pairs | USD/IDR, USD/TWD, USD/AUD, USD/EUR, USD/SGD |
| Horizons | 1, 5, and 10 days |
| LLMs | Llama3-8B-Instruct, Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3 (served with vLLM, FP16) |
| LLM input | ChatTime tokenization (normalize, discretize, serialize into `###value###` tokens) |
| Ablation | 2x2 factorial design: news augmentation (on/off) x prompting (zero-shot/few-shot) |
| Baselines | ARIMA, LSTM, XGBoost, iTransformer, vanilla Transformer, Naive (random walk), Moving Average (7-day) |
| Metrics | RMSE, MAE, MAPE, Directional Accuracy (DA), R² |
| Significance | Diebold-Mariano test with Harvey-Leybourne-Newbold correction (N = 30 rolling windows per unit); secondary paired t-test for RQ1 |

### Research questions

1. **RQ1:** To what extent do LLMs outperform traditional forecasting baselines in forecasting foreign exchange rates?
2. **RQ2:** Does incorporating financial news into LLM prompts improve foreign exchange rate forecasting accuracy?
3. **RQ3:** Does few-shot prompting improve forecasting performance compared to zero-shot prompting?
4. **RQ4:** Do news integration and few-shot prompting work independently or synergistically in improving forecast performance?

## Headline results

All figures below are computed from `results/combined_summary.csv` and `results/dm_test_results.csv`.

- Each LLM's single best-RMSE configuration beats every baseline on mean RMSE: Qwen2.5-7B reaches 27.48, against 91.81 for the best baseline (XGBoost).
- The LLM achieves a lower loss than the best baseline in 45 of 45 (model, currency, horizon) comparisons, significant at the 5% level in 38 of 45.
- Baseline mean DA clusters around 50% (48.85% to 50.04%); LLM mean DA ranges from 52.19% to 54.25%.
- All three LLMs have a positive mean R²; every baseline has a negative mean R².
- Few-shot prompting is the more reliable driver of improvement. News augmentation shows no consistent benefit under the data constraints described below.

## Repository structure

```
forecast/
├── config.yaml
├── requirements.txt
├── data/
│   ├── fx_raw/              # raw downloads from Yahoo Finance
│   ├── fx_filled/           # cleaned daily series, one CSV per pair (USDIDR.csv, ...)
│   └── fx_news/             # static news snapshots per pair
├── scripts/                 # all code (run commands from this directory)
├── results/                 # CSV/JSON outputs and generated plots
├── figures/                 # figures used in the thesis
└── logs/                    # vLLM server logs
```

### Key scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `run_full_ablation.py` | Full pipeline: baselines, then LLM ablation (starts vLLM servers automatically) |
| `run_baselines_only.py` | Re-run only the seven baselines and merge into existing LLM results |
| `run_llama3_only.py` | Re-run only Llama3-8B and merge into existing results |
| `baseline_models.py` | ARIMA, LSTM, XGBoost, Naive, Moving Average |
| `itransformer_model.py`, `vanilla_transformer_model.py` | Transformer baselines (variate-wise vs. temporal attention) |
| `ts_tokenizer.py`, `llm_prompts.py` | ChatTime tokenization and prompt construction |
| `llm_api_helper.py` | LLM client for the vLLM servers |
| `evaluation_metrics.py` | RMSE, MAE, MAPE, DA, R² |
| `dm_test.py`, `statistical_tests.py` | Diebold-Mariano tests and paired tests |
| `plot_rq.py`, `plot_extra.py`, `plot_raw_fx_series.py` | Figures for RQ1 to RQ4, per-horizon plots, raw price plots |
| `generate_chap4_tables.py`, `generate_horizon_breakdown.py` | LaTeX-ready tables for the results chapter |
| `download_yahoo_fx.py`, `download_yahoo_news.py`, `check_fx_data.py` | Data collection and sanity checks |
| `start_vllm_servers.sh`, `setup_and_run.sh` | Server and environment helpers |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements beyond the Python packages:

- One or more NVIDIA GPUs for vLLM (the reported runs used L40S GPUs; vLLM v0.4.3).
- A Hugging Face account with access to `meta-llama/Meta-Llama-3-8B-Instruct` (gated model).
- Internet access to Yahoo Finance for data download (`yfinance`).

## Reproducing the experiments

All commands are run from `scripts/`.

### 1. Data

```bash
python3 download_yahoo_fx.py --all --start 2016-01-01
python3 download_yahoo_news.py
python3 check_fx_data.py
```

News headlines are a **static snapshot** (9 to 10 articles per pair) collected at download time, not historical news aligned to each forecast window.

### 2. Full experiment

```bash
python3 run_full_ablation.py \
    --output ../results/ \
    --data-dir ../data/fx_filled \
    --news-dir ../data/fx_news
```

Relative paths assume the command is run from `scripts/`. Omitting `--data-dir` and `--news-dir` makes the script look for `data/fx_filled` relative to the current directory.

vLLM servers are configured in `run_full_ablation.py` (`LLM_SERVER_CONFIG`, `LLAMA3_CONFIG`): GPU assignment, model IDs, and ports (Qwen2.5 on 18001, Mistral on 18002, Llama3 on 18003 in the current version). The default Llama3 port in `llm_api_helper.py` can be overridden with the `LLAMA3_PORT` environment variable, which the run scripts set automatically. Adjust GPU IDs and ports to match your machine, particularly on shared servers.

### 3. Partial re-runs

```bash
python3 run_baselines_only.py --results-dir ../results/ --data-dir ../data/fx_filled --news-dir ../data/fx_news
python3 run_llama3_only.py    --results-dir ../results/ --data-dir ../data/fx_filled --news-dir ../data/fx_news
```

Both scripts merge their output into the existing `ablation_results.json` and summary CSVs without re-running the other models.

### 4. Analysis and figures

```bash
python3 statistical_tests.py --results-dir ../results/
python3 dm_test.py --csv ../results/window_results.csv

python3 plot_rq.py    --results-dir ../results/ --outdir ../results/plots_rq/
python3 plot_extra.py --results-dir ../results/ --outdir ../results/plots_extra/
python3 plot_raw_fx_series.py --outdir ../figures/

python3 generate_chap4_tables.py --results-dir ../results/
python3 generate_horizon_breakdown.py --csv ../results/window_results.csv --include-rq4
```

## Outputs (`results/`)

| File | Content |
|---|---|
| `ablation_results.json` | Aggregate results for all models and configurations |
| `baseline_summary.csv`, `llm_summary.csv`, `combined_summary.csv` | Per (currency, horizon, model, config) metrics, including R² |
| `window_results.csv` | Per-window RMSE and DA, with raw `y_true` / `y_pred` (semicolon-separated) |
| `dm_test_results.csv`, `llm_stat_tests.csv`, `statistical_test_results.csv` | Significance test results |
| `plots_rq/`, `plots_extra/` | Generated figures |

## Evaluation protocol

- **LLMs:** no fine-tuning; 30 rolling evaluation windows per (model, currency, horizon, configuration). Sampling temperature starts low and is raised incrementally (up to 0.7) when a response contains no valid `###value###` tokens.
- **Baselines:** fitted once on the first 80% of each series (chronological split) and evaluated over a 330-observation test window. For the Diebold-Mariano tests, per-window metrics are computed on the same 30 rolling-window structure as the LLMs.
- **Aggregate metrics:** baseline aggregates use the full test window, whereas LLM aggregates pool the 30 rolling windows. This remaining difference is discussed in the thesis (Limitations).
- **Reproducibility:** LLM outputs are sampled and will not be bit-identical across reruns; baseline results are deterministic given the same data and configuration.

## Known limitations

- The news dataset is small, static, and not temporally aligned with each forecast window, so RQ2 should be read as a pilot investigation.
- The LLMs' pretraining cutoffs fall in 2024, so only the later part of the 2016 to March 2026 data is guaranteed to be unseen; memorization of earlier history cannot be fully excluded.
- All results are historical backtests and do not establish live trading performance.
- Time series foundation models (e.g., Timer-XL, Chronos) and other Transformer variants (e.g., PatchTST, Autoformer) are not included and are left as future work.


## Acknowledgements

Supervised by Prof. Richard Tzong-Han Tsai, with guidance and support from Connyn and Royce of IISR lab at National Central University. The tokenization approach builds on ChatTime (Wang et al., 2025).
