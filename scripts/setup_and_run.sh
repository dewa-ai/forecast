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
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1 && \
   curl -s http://localhost:8001/v1/models > /dev/null 2>&1 && \
   curl -s http://localhost:8002/v1/models > /dev/null 2>&1; then
    echo "  ✓ vLLM servers already running"
else
    echo "  Starting servers (this may take 3-5 minutes)..."

    if [ ! -f "scripts/start_vllm_servers.sh" ]; then
        echo "  Creating start_vllm_servers.sh..."
        cat > scripts/start_vllm_servers.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting vLLM servers..."
mkdir -p logs

# Llama 3 8B
nohup vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000 --gpu-memory-utilization 0.30 --dtype half \
    > logs/llama3.log 2>&1 &
echo "Started Llama 3 on port 8000"

sleep 5

# Qwen 2.5 7B
nohup vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8001 --gpu-memory-utilization 0.30 --dtype half \
    > logs/qwen.log 2>&1 &
echo "Started Qwen 2.5 on port 8001"

sleep 5

# Mistral 7B
nohup vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --port 8002 --gpu-memory-utilization 0.30 --dtype half \
    > logs/mistral.log 2>&1 &
echo "Started Mistral on port 8002"

echo "All servers starting... wait 3-5 minutes for models to load"
EOF
        chmod +x scripts/start_vllm_servers.sh
    fi

    bash scripts/start_vllm_servers.sh

    # Tunggu dengan loop sampai semua server ready
    echo "  Waiting for all servers to be ready..."
    MAX_WAIT=300  # maksimal tunggu 5 menit
    ELAPSED=0
    LLAMA_OK=0
    QWEN_OK=0
    MISTRAL_OK=0

    while [ $ELAPSED -lt $MAX_WAIT ]; do
        sleep 15
        ELAPSED=$((ELAPSED + 15))

        curl -s http://localhost:8000/v1/models > /dev/null 2>&1 && LLAMA_OK=1 || LLAMA_OK=0
        curl -s http://localhost:8001/v1/models > /dev/null 2>&1 && QWEN_OK=1 || QWEN_OK=0
        curl -s http://localhost:8002/v1/models > /dev/null 2>&1 && MISTRAL_OK=1 || MISTRAL_OK=0

        echo "  [${ELAPSED}s] Llama:${LLAMA_OK} Qwen:${QWEN_OK} Mistral:${MISTRAL_OK}"

        if [ "$LLAMA_OK" = "1" ] && [ "$QWEN_OK" = "1" ] && [ "$MISTRAL_OK" = "1" ]; then
            echo "  ✓ All servers ready!"
            break
        fi
    done

    # Kalau timeout
    if [ "$LLAMA_OK" != "1" ] || [ "$QWEN_OK" != "1" ] || [ "$MISTRAL_OK" != "1" ]; then
        echo "  ✗ Timeout! Server belum ready setelah ${MAX_WAIT}s"
        echo "  Cek log: tail -f logs/llama3.log"
        echo "  Lanjut anyway? (y/n)"
        read -r response
        if [ "$response" != "y" ]; then
            exit 1
        fi
    fi
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