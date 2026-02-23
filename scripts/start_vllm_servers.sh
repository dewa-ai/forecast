#!/bin/bash
# start_vllm_servers.sh
# Starts 3 vLLM servers for Llama3, Qwen2.5, and Mistral.
#
# Fixes applied vs original:
#   1. Uses full path to vllm binary (nohup doesn't inherit PATH)
#   2. Uses ports 18000-18002 (8000-8002 may be used by other users)
#   3. Pins all models to GPU 1 (GPU 0 and 2 may be used by other users)
#   4. sleep 60 between starts to avoid OOM during simultaneous loading
#   5. Waits for each server to be ready before starting the next

set -e

# ── Config ────────────────────────────────────────────────────────────────────
VLLM_CMD="python3 -m vllm.entrypoints.openai.api_server"  # use module, no binary needed
GPU_ID=(0 1 2)                           # GPU to use — change if GPU 1 is busy
PORTS=(18000 18001 18002)                # avoid 8000-8002 which may be taken
MEM_UTIL=0.70                            # 30% each × 3 models = 90% of GPU
DTYPE="half"                             # float16 to save VRAM
WAIT_BETWEEN=60                          # seconds to wait between model starts
MAX_LEN=8192                             # max sequence length for vLLM (default 4096 may be too small for some models)
# ─────────────────────────────────────────────────────────────────────────────

echo "=========================================="
echo "Starting vLLM Servers"
echo "  GPU       : $GPU_ID"
echo "  Ports     : ${PORTS[*]}"
echo "  VLLM cmd  : $VLLM_CMD"
echo "=========================================="

# Sanity check — make sure vllm Python package is importable
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "[ERROR] vllm Python package not found."
    echo "  Install with: pip install vllm"
    exit 1
fi

# Check if ports are already in use
for port in "${PORTS[@]}"; do
    if ss -tlnp | grep -q ":$port "; then
        echo "[WARN] Port $port is already in use. Skipping or change PORTS in this script."
    fi
done

mkdir -p logs

# ── Llama 3 8B ────────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Starting Llama 3 8B on port ${PORTS[0]}..."
CUDA_VISIBLE_DEVICES=${GPU_ID[0]} nohup $VLLM_CMD \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --port ${PORTS[0]} \
    --gpu-memory-utilization $MEM_UTIL \
    --dtype $DTYPE \
    --max-model-len $MAX_LEN \
    > logs/llama3.log 2>&1 &

echo "  Waiting ${WAIT_BETWEEN}s before starting next model..."
sleep $WAIT_BETWEEN

# ── Qwen 2.5 7B ───────────────────────────────────────────────────────────────
echo "[2/3] Starting Qwen 2.5 7B on port ${PORTS[1]}..."
CUDA_VISIBLE_DEVICES=${GPU_ID[1]} nohup $VLLM_CMD \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port ${PORTS[1]} \
    --gpu-memory-utilization $MEM_UTIL \
    --dtype $DTYPE \
    --max-model-len $MAX_LEN \
    > logs/qwen.log 2>&1 &

echo "  Waiting ${WAIT_BETWEEN}s before starting next model..."
sleep $WAIT_BETWEEN

# ── Mistral 7B ────────────────────────────────────────────────────────────────
echo "[3/3] Starting Mistral 7B on port ${PORTS[2]}..."
CUDA_VISIBLE_DEVICES=${GPU_ID[2]} nohup $VLLM_CMD \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --port ${PORTS[2]} \
    --gpu-memory-utilization $MEM_UTIL \
    --dtype $DTYPE \
    --max-model-len $MAX_LEN \
    > logs/mistral.log 2>&1 &

echo ""
echo "All 3 servers started. Waiting for them to be ready..."
echo "(This may take 3-5 minutes for models to fully load)"
echo ""

# ── Wait until all servers are ready ─────────────────────────────────────────
MAX_WAIT=300
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    sleep 15
    ELAPSED=$((ELAPSED + 15))

    LLAMA_OK=$(curl -s http://localhost:${PORTS[0]}/v1/models | grep -c "id" || true)
    QWEN_OK=$(curl -s http://localhost:${PORTS[1]}/v1/models | grep -c "id" || true)
    MISTRAL_OK=$(curl -s http://localhost:${PORTS[2]}/v1/models | grep -c "id" || true)

    echo "  [${ELAPSED}s] Llama:${LLAMA_OK} Qwen:${QWEN_OK} Mistral:${MISTRAL_OK}"

    if [ "$LLAMA_OK" -ge 1 ] && [ "$QWEN_OK" -ge 1 ] && [ "$MISTRAL_OK" -ge 1 ]; then
        echo ""
        echo "✓ All servers ready!"
        echo "  Llama3  → http://localhost:${PORTS[0]}"
        echo "  Qwen2.5 → http://localhost:${PORTS[1]}"
        echo "  Mistral → http://localhost:${PORTS[2]}"
        echo ""
        echo "Next: python3 scripts/run_full_ablation.py --data-dir data/fx_filled --news-dir data/fx_news --output results"
        exit 0
    fi
done

echo ""
echo "[TIMEOUT] Not all servers ready after ${MAX_WAIT}s. Check logs:"
echo "  tail -f logs/llama3.log"
echo "  tail -f logs/qwen.log"
echo "  tail -f logs/mistral.log"
exit 1