#!/bin/bash
# setup_and_run.sh
# Complete setup and execution script for ablation study

set -e

echo "=========================================="
echo "FX Forecasting - Complete Setup & Run"
echo "=========================================="
echo ""

# Step 1: Check if data exists
echo "[Step 1/6] Checking data..."
if [ ! -d "data/fx_filled" ] || [ ! "$(ls -A data/fx_filled)" ]; then
    echo "  Data not found. Downloading..."
    
    echo "  → Downloading FX price data..."
    python3 scripts/download_yahoo_fx.py --all --start 2016-01-01
    
    echo "  → Downloading news data..."
    python3 scripts/download_yahoo_news.py --all
    
    echo "  → Filling missing days..."
    python3 scripts/check_fx_data.py --data-dir data/fx_raw --fill --out-dir data/fx_filled
else
    echo "  ✓ Data found"
fi

# Step 2: Check dependencies
echo ""
echo "[Step 2/6] Checking Python dependencies..."
python3 -c "
import sys
missing = []
try:
    import numpy
except: missing.append('numpy')
try:
    import pandas
except: missing.append('pandas')
try:
    import torch
except: missing.append('torch')
try:
    import vllm
except: missing.append('vllm')
try:
    import xgboost
except: missing.append('xgboost')
try:
    import statsmodels
except: missing.append('statsmodels')

if missing:
    print('  Missing packages:', ', '.join(missing))
    print('  Run: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('  ✓ All dependencies installed')
"

# Step 3: Check GPU
echo ""
echo "[Step 3/6] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "  ✓ GPU available"
else
    echo "  ⚠ No GPU detected. LLM inference will be slow."
    echo "    Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Step 4: Start vLLM servers
echo ""
echo "[Step 4/6] Starting vLLM servers..."

# Check if servers are already running on our ports (18000-18002)
LLAMA_OK=$(curl -s http://localhost:18000/v1/models | grep -c "id" || true)
QWEN_OK=$(curl -s http://localhost:18001/v1/models | grep -c "id" || true)
MISTRAL_OK=$(curl -s http://localhost:18002/v1/models | grep -c "id" || true)

if [ "$LLAMA_OK" -ge 1 ] && [ "$QWEN_OK" -ge 1 ] && [ "$MISTRAL_OK" -ge 1 ]; then
    echo "  ✓ vLLM servers already running on ports 18000-18002"
else
    echo "  Starting servers (this may take 3-5 minutes)..."

    # Check that start_vllm_servers.sh exists
    if [ ! -f "scripts/start_vllm_servers.sh" ]; then
        echo "  [ERROR] scripts/start_vllm_servers.sh not found!"
        echo "  Please make sure the file exists before running this script."
        exit 1
    fi

    bash scripts/start_vllm_servers.sh
    # Note: start_vllm_servers.sh already waits and verifies all servers are ready.
    # If it exits 0, all servers are up. If it exits 1, something failed.
fi

# Step 5: Run quick test
echo ""
echo "[Step 5/6] Running quick test..."
python3 scripts/test_models.py --currency USDIDR || {
    echo "  ⚠ Some tests failed. Check errors above."
    echo "    Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
}

# Step 6: Run full ablation study
echo ""
echo "[Step 6/6] Starting complete ablation study..."
echo ""
echo "This will run:"
echo "  - 5 currencies (USDIDR, USDEUR, USDSGD, USDTWD, USDAUD)"
echo "  - 3 horizons (1, 5, 10 days)"
echo "  - 4 baseline models (ARIMA, LSTM, XGBoost, iTransformer)"
echo "  - 3 LLM models × 4 configs (Llama, Qwen, Mistral)"
echo "  - Total: ~285 experiments"
echo ""
echo "Estimated time: 50-90 minutes on L40S GPU"
echo ""
echo "Start now? (y/n)"
read -r response

if [ "$response" = "y" ]; then
    echo ""
    echo "=========================================="
    echo "STARTING ABLATION STUDY"
    echo "=========================================="
    echo ""
    
    # Create results directory
    mkdir -p results
    
    # Run ablation study
    python3 scripts/run_full_ablation.py \
        --data-dir data/fx_filled \
        --news-dir data/fx_news \
        --output results \
        | tee results/run_log.txt
    
    echo ""
    echo "=========================================="
    echo "COMPLETED!"
    echo "=========================================="
    echo ""
    echo "Results saved to: results/"
    echo ""
    echo "Generated files:"
    ls -lh results/ | grep -E "\.csv$|\.json$"
    echo ""
    echo "Next steps:"
    echo "  1. Analyze results: python3 scripts/analyze_results.py"
    echo "  2. Generate plots: python3 scripts/plot_results.py"
    echo "  3. Statistical tests: python3 scripts/statistical_tests.py"
else
    echo "Aborted by user"
fi