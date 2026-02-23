# Forecasting

LLM-based exchange rate forecasting (USD/IDR, USD/TWD, USD/EUR, USD/AUD, USD/SGD)
using Yahoo Finance data.

# FX Forecasting Framework - Setup & Usage Guide

## Installation

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install pandas numpy scipy scikit-learn

# Data collection
pip install yfinance

# Statistical models
pip install statsmodels

# Machine learning
pip install xgboost

# Deep learning
pip install torch torchvision  # Or: pip install torch --index-url https://download.pytorch.org/whl/cpu

# Optional: For visualization
pip install matplotlib seaborn
```

## Project Structure

```
forecast/
├── scripts/
│   ├── download_yahoo_fx.py         # Download FX price data
│   ├── download_yahoo_news.py       # Download news articles
│   ├── check_fx_data.py             # Data quality check & fill
│   ├── evaluation_metrics.py        # Metrics (RMSE, MAE, MAPE, DA, DM)
│   ├── baseline_models.py           # ARIMA, LSTM, XGBoost
│   ├── itransformer_model.py        # iTransformer implementation
│   └── experiment_runner.py         # Ablation study runner
├── data/
│   ├── fx_raw/                      # Raw FX data (trading days only)
│   ├── fx_filled/                   # Filled data (all calendar days)
│   └── fx_news/                     # News articles (JSON)
└── results/                         # Experiment results
```

## Quick Start

### Step 1: Download FX Data

```bash
# Download all currency pairs from 2016
python3 scripts/download_yahoo_fx.py --all --start 2016-01-01

# Or specific pairs
python3 scripts/download_yahoo_fx.py --pairs USDIDR USDEUR --start 2016-01-01
```

**Output:** `data/fx_raw/USDIDR.csv`, `data/fx_raw/USDEUR.csv`, etc.

### Step 2: Download News Data

```bash
# Download news for all pairs
python3 scripts/download_yahoo_news.py --all

# Or specific pair
python3 scripts/download_yahoo_news.py --pair USDIDR
```

**Output:** `data/fx_news/USDIDR_news.json`, etc.

### Step 3: Fill Missing Days (Weekends/Holidays)

```bash
# Forward-fill ALL calendar days (including weekends)
python3 scripts/check_fx_data.py --data-dir data/fx_raw --fill --out-dir data/fx_filled

# Check data quality only (no fill)
python3 scripts/check_fx_data.py --data-dir data/fx_raw
```

**Output:** `data/fx_filled/USDIDR.csv` (with weekends filled)

## Running Experiments

### Baseline Models Only

```bash
python3 scripts/experiment_runner.py \
  --currency USDIDR \
  --data-dir data/fx_filled \
  --output-dir results \
  --baseline-only
```

### LLM Ablation Studies

```bash
# Run all ablations
python3 scripts/experiment_runner.py \
  --currency USDIDR \
  --data-dir data/fx_filled \
  --news-dir data/fx_news \
  --output-dir results \
  --ablation all

# Run specific ablation
python3 scripts/experiment_runner.py \
  --currency USDIDR \
  --ablation news  # or 'prompt' or 'interaction'
```

### Full Experiment (Baselines + LLMs)

```bash
python3 scripts/experiment_runner.py \
  --currency USDIDR \
  --data-dir data/fx_filled \
  --news-dir data/fx_news \
  --output-dir results
```

## Using Individual Models

### ARIMA

```python
from baseline_models import ARIMAForecaster
import pandas as pd

# Load data
df = pd.read_csv('data/fx_filled/USDIDR.csv')
data = df['close'].values

# Train/test split
train = data[:int(0.8 * len(data))]
test = data[int(0.8 * len(data)):]

# Train and predict
model = ARIMAForecaster(order=(1, 1, 1))
model.fit(train)
predictions = model.predict(horizon=len(test))

# Evaluate
from evaluation_metrics import calculate_all_metrics
metrics = calculate_all_metrics(test, predictions)
print(metrics)
```

### LSTM

```python
from baseline_models import LSTMForecaster

model = LSTMForecaster(
    seq_length=30,
    hidden_size=50,
    num_layers=2,
    epochs=50,
    batch_size=32
)

model.fit(train)
predictions = model.predict(train, horizon=len(test))
```

### XGBoost

```python
from baseline_models import XGBoostForecaster

model = XGBoostForecaster(
    n_lags=30,
    n_estimators=100,
    learning_rate=0.1
)

model.fit(train)
predictions = model.predict(horizon=len(test))
```

### iTransformer

```python
from itransformer_model import iTransformerForecaster

model = iTransformerForecaster(
    seq_len=96,
    pred_len=24,
    d_model=128,
    n_heads=8,
    n_layers=3,
    epochs=10
)

model.fit(train)
predictions = model.predict(train, horizon=len(test))
```

## Evaluation Metrics

```python
from evaluation_metrics import calculate_all_metrics, diebold_mariano_test

# Calculate all metrics
metrics = calculate_all_metrics(y_true, y_pred)
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"MAPE: {metrics['mape']:.2f}%")
print(f"DA: {metrics['da']:.2f}%")

# Compare two models statistically
dm_stat, p_value = diebold_mariano_test(y_true, pred_model1, pred_model2)
print(f"DM Statistic: {dm_stat:.4f}")
print(f"P-value: {p_value:.4f}")
```

## Ablation Study Design

### Ablation 1: News Contribution
**Q1: Does external news actually help?**

Settings:
- `w/o news`: Baseline LLM (price data only)
- `w/ news`: LLM with news articles

### Ablation 2: Prompting Strategy
**Q2: Does few-shot prompting matter?**

Settings:
- `zero-shot`: No examples in prompt
- `few-shot`: With 3-5 example predictions

### Ablation 3: Interaction Effect (NEWS × PROMPT)
**Q3: Do they help independently or jointly?**

Settings:
- `zero + no`: Baseline LLM
- `zero + yes`: News only
- `few + no`: Prompt only
- `few + yes`: Full model (best)

## Expected Results Structure

After running experiments, you'll get:

```
results/
├── results_USDIDR_20260119_143022.json  # Full results
└── summary_USDIDR.csv                    # Summary table
```

**Summary CSV columns:**
- `type`: baseline or llm
- `model`: Model name (arima, lstm, llama3-8b, etc.)
- `horizon`: 1, 5, or 10 days
- `config`: Model configuration (e.g., news=True_prompt=few)
- `rmse`, `mae`, `mape`, `da`: Evaluation metrics

## Next Steps for Thesis

1. **Run all experiments** for 5 currency pairs × 3 horizons
2. **Analyze results**: Which models perform best? Does news help?
3. **Statistical significance**: Use DM test to compare models
4. **Visualize**: Create plots showing model comparisons
5. **Write up**: Document findings with tables and figures

## Tips

- **Start small**: Test with one currency pair first
- **Monitor GPU**: LSTM and iTransformer benefit from GPU
- **Hyperparameter tuning**: Experiment with different settings
- **News quality**: More recent news = better signal
- **Horizon matters**: Short-term (h=1) vs long-term (h=10) behave differently

## References

- **iTransformer**: Liu et al. (2023) - "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
- **Diebold-Mariano Test**: Diebold & Mariano (1995) - "Comparing Predictive Accuracy"

---