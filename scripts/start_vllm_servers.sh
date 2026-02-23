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
