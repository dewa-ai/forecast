#!/usr/bin/env python3
"""
LLM API Helper

Supports:
- vLLM (GPU - Primary, FAST)
- Ollama (CPU/GPU - Backup, FREE)

Example:
    from llm_api_helper import LLMForecaster
    
    # Primary: vLLM (fastest)
    forecaster = LLMForecaster(backend='vllm')
    predictions = forecaster.predict(prompt, model='llama3-8b')
    
    # Backup: Ollama (if vLLM fails)
    forecaster = LLMForecaster(backend='ollama')
    predictions = forecaster.predict(prompt, model='llama3-8b')
"""

import json
import re
import requests
import numpy as np
from typing import Optional


class BaseAPI:
    """Base class for LLM API integrations."""
    
    def call(self, prompt: str, model: str, **kwargs) -> str:
        """Call LLM API and return response."""
        raise NotImplementedError
    
    def parse_predictions(self, response: str) -> np.ndarray:
        """
        Parse LLM response to extract numerical predictions.
        Handles full JSON, partial JSON (prompt ends with '['), raw arrays, fallback.
        """
        text = response.strip()

        # Case 1: prompt ended with {"predictions": [ so LLM continues with just numbers
        partial = re.match(r"^[\d\s.,]+", text)
        if partial:
            nums = re.findall(r"[\d]+\.?[\d]*", partial.group())
            if nums:
                try:
                    return np.array([float(x) for x in nums])
                except Exception:
                    pass

        # Case 2: full JSON {"predictions": [...]}
        try:
            json_match = re.search(r"\{[^{}]*predictions[^{}]*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return np.array(data["predictions"])
        except Exception:
            pass

        # Case 3: find any array of numbers
        try:
            array_match = re.search(r"\[([\d.,\s]+)\]", text)
            if array_match:
                nums = [float(x.strip()) for x in array_match.group(1).split(",") if x.strip()]
                if nums:
                    return np.array(nums)
        except Exception:
            pass

        # Case 4: extract all decimal numbers from response
        all_nums = re.findall(r"\b\d+\.\d+\b", text)
        if all_nums:
            try:
                return np.array([float(x) for x in all_nums[:20]])
            except Exception:
                pass

        raise ValueError("Could not parse predictions from response")


class vLLMAPI(BaseAPI):
    """
    vLLM API integration - PRIMARY (GPU, FASTEST).
    
    FREE - Runs on your L40S GPU.
    Start servers with: bash scripts/start_vllm_servers.sh

    NOTE: Uses ports 18000-18002 (not 8000-8002) to avoid conflicts
    with other users on shared lab GPUs.
    """
    
    def __init__(self):
        """Initialize with multi-model endpoints."""
        # Using 18000-18002 to avoid port conflicts with other lab users
        self.endpoints = {
            'llama3-8b': 'http://localhost:18000',
            'qwen2.5-7b': 'http://localhost:18001',
            'mistral-7b': 'http://localhost:18002'
        }

        # Model ID yang digunakan saat serve vLLM
        self.model_ids = {
            'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
            'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.3'
        }
    
    def call(self, prompt: str, model: str = 'llama3-8b',
             temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """
        Call vLLM API (OpenAI-compatible chat completions).
        
        Args:
            prompt: Input prompt
            model: Model name (llama3-8b, qwen2.5-7b, mistral-7b)
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Model response text
        """
        base_url = self.endpoints.get(model)
        if not base_url:
            raise ValueError(f"Unknown model: {model}. Available: {list(self.endpoints.keys())}")

        model_id = self.model_ids.get(model)

        url = f"{base_url}/v1/chat/completions"
        
        data = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": ["\n\n\n", "###", "Example"]
        }
        
        try:
            response = requests.post(url, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to {model} at {base_url}. "
                f"Start vLLM servers: bash scripts/start_vllm_servers.sh"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {model} timed out.")
        except Exception as e:
            raise RuntimeError(f"vLLM error: {str(e)}")


class OllamaAPI(BaseAPI):
    """
    Ollama API integration - BACKUP (FREE).
    
    FREE - Runs locally on your server.
    Install: curl -fsSL https://ollama.com/install.sh | sh
    Start: ollama serve
    Pull models: ollama pull llama3
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Args:
            base_url: Ollama server URL
        """
        self.base_url = base_url
        
        # Model mappings
        self.models = {
            'llama3-8b': 'llama3',
            'qwen2.5-7b': 'qwen2.5:7b',
            'mistral-7b': 'mistral'
        }
    
    def call(self, prompt: str, model: str = 'llama3-8b',
             temperature: float = 0.1) -> str:
        """
        Call Ollama API.
        
        Args:
            prompt: Input prompt
            model: Model name
            temperature: Sampling temperature
            
        Returns:
            Model response text
        """
        model_id = self.models.get(model, model)
        
        url = f"{self.base_url}/api/generate"
        
        data = {
            "model": model_id,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=data, timeout=180)
            response.raise_for_status()
            result = response.json()
            return result['response']
        
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Install: curl -fsSL https://ollama.com/install.sh | sh"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama error: {str(e)}")


class LLMForecaster:
    """
    Unified LLM forecaster with FREE backends only.
    
    Supports:
    - vLLM (GPU - primary, fastest)
    - Ollama (CPU/GPU - backup, slower but reliable)
    
    Both are 100% FREE!
    """
    
    def __init__(self, backend: str = 'vllm', base_url: Optional[str] = None):
        """
        Args:
            backend: API backend ('vllm' or 'ollama')
            base_url: Base URL for Ollama (default: http://localhost:11434)
        """
        self.backend = backend
        
        if backend == 'vllm':
            self.api = vLLMAPI()
        
        elif backend == 'ollama':
            base_url = base_url or "http://localhost:11434"
            self.api = OllamaAPI(base_url)
        
        else:
            raise ValueError(
                f"Unknown backend: {backend}. "
                f"Available FREE backends: 'vllm', 'ollama'"
            )
    
    def predict(self, prompt: str, model: str = 'llama3-8b', **kwargs) -> np.ndarray:
        """
        Generate predictions using LLM.
        
        Args:
            prompt: Formatted prompt
            model: Model name (llama3-8b, qwen2.5-7b, mistral-7b)
            **kwargs: Additional API parameters
            
        Returns:
            Array of predicted values
        """
        print(f"[{self.backend}/{model}] Calling LLM...")
        
        response = self.api.call(prompt, model=model, **kwargs)
        predictions = self.api.parse_predictions(response)
        
        print(f"✓ Received {len(predictions)} predictions")
        
        return predictions


# Helper function for automatic fallback
def create_forecaster_with_fallback() -> LLMForecaster:
    """
    Create forecaster with automatic fallback.
    
    Tries vLLM first, falls back to Ollama if vLLM is unavailable.
    
    Returns:
        LLMForecaster instance
    """
    try:
        forecaster = LLMForecaster(backend='vllm')
        requests.get('http://localhost:18000/v1/models', timeout=2)
        print("✓ Using vLLM (GPU - fastest)")
        return forecaster
    except:
        pass
    
    try:
        forecaster = LLMForecaster(backend='ollama')
        requests.get('http://localhost:11434', timeout=2)
        print("⚠ vLLM unavailable, using Ollama (slower)")
        return forecaster
    except:
        pass
    
    raise RuntimeError(
        "No LLM backends available!\n"
        "Start vLLM: bash scripts/start_vllm_servers.sh\n"
        "OR install Ollama: curl -fsSL https://ollama.com/install.sh | sh"
    )


# Example usage
if __name__ == "__main__":
    prompt = """You are a financial analyst.
Predict USD/IDR for next 5 days:

2024-01-15: 15420.50
2024-01-16: 15435.20
2024-01-17: 15448.75

Output ONLY: {"predictions": [day1, day2, day3, day4, day5]}"""
    
    print("="*60)
    print("Testing FREE LLM Backends")
    print("="*60)
    
    # Test vLLM
    print("\n[1] Testing vLLM (primary - fastest)")
    try:
        forecaster = LLMForecaster(backend='vllm')
        predictions = forecaster.predict(prompt, model='llama3-8b')
        print(f"Predictions: {predictions}")
        print("✓ vLLM works!")
    except Exception as e:
        print(f"✗ vLLM error: {e}")
    
    # Test Ollama
    print("\n[2] Testing Ollama (backup - free)")
    try:
        forecaster = LLMForecaster(backend='ollama')
        predictions = forecaster.predict(prompt, model='llama3-8b')
        print(f"Predictions: {predictions}")
        print("✓ Ollama works!")
    except Exception as e:
        print(f"✗ Ollama error: {e}")
    
    print("\n" + "="*60)
    print("Setup:")
    print("  vLLM: bash scripts/start_vllm_servers.sh")
    print("  Ollama: curl -fsSL https://ollama.com/install.sh | sh")
    print("="*60)