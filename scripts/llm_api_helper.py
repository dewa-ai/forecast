#!/usr/bin/env python3
"""
LLM API Helper

Supports:
- vLLM (GPU - Primary, FAST)
- Ollama (CPU/GPU - Backup, FREE)

Fixes applied:
  1. System prompt for zero-shot to prevent Mistral MAPE anomaly
  2. parse_predictions with expected_scale filter to reject out-of-range values
  3. predict() accepts expected_scale kwarg and passes it to parser
"""

import json
import re
import requests
import numpy as np
from typing import Optional


class BaseAPI:
    """Base class for LLM API integrations."""

    def call(self, prompt: str, model: str, **kwargs) -> str:
        raise NotImplementedError

    def parse_predictions(self, response: str,
                          expected_scale: Optional[float] = None) -> np.ndarray:
        """
        Parse LLM response to extract numerical predictions.

        Args:
            response: Raw text response from LLM
            expected_scale: Last known price — used to filter out-of-range values.
                            Predictions outside [scale*0.5, scale*2.0] are dropped.

        Returns:
            Array of predicted values
        """
        text = response.strip()

        predictions = None

        # Case 1: prompt ended with {"predictions": [ so LLM continues with just numbers
        partial = re.match(r"^[\d\s.,]+", text)
        if partial:
            nums = re.findall(r"[\d]+\.?[\d]*", partial.group())
            if nums:
                try:
                    predictions = np.array([float(x) for x in nums])
                except Exception:
                    pass

        # Case 2: full JSON {"predictions": [...]}
        if predictions is None:
            try:
                json_match = re.search(r"\{[^{}]*predictions[^{}]*\}", text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    predictions = np.array(data["predictions"])
            except Exception:
                pass

        # Case 3: find any array of numbers
        if predictions is None:
            try:
                array_match = re.search(r"\[([\d.,\s]+)\]", text)
                if array_match:
                    nums = [float(x.strip()) for x in array_match.group(1).split(",")
                            if x.strip()]
                    if nums:
                        predictions = np.array(nums)
            except Exception:
                pass

        # Case 4: extract all decimal numbers from response
        if predictions is None:
            all_nums = re.findall(r"\b\d+\.\d+\b", text)
            if all_nums:
                try:
                    predictions = np.array([float(x) for x in all_nums[:20]])
                except Exception:
                    pass

        if predictions is None:
            raise ValueError("Could not parse predictions from response")

        # Fix 2: filter by expected_scale to reject garbage values
        if expected_scale is not None and expected_scale > 0:
            lo, hi = expected_scale * 0.5, expected_scale * 2.0
            filtered = predictions[(predictions >= lo) & (predictions <= hi)]
            if len(filtered) >= max(1, len(predictions) // 2):
                predictions = filtered
            else:
                print(f"[WARN] Scale filter removed too many values "
                      f"(expected ~{expected_scale:.2f}). Keeping original.")

        return predictions


class vLLMAPI(BaseAPI):
    """
    vLLM API integration - PRIMARY (GPU, FASTEST).

    Uses ports 18000-18002 to avoid conflicts with other lab users.
    """

    def __init__(self):
        self.endpoints = {
            'llama3-8b':  'http://localhost:18000',
            'qwen2.5-7b': 'http://localhost:18001',
            'mistral-7b': 'http://localhost:18002',
        }
        self.model_ids = {
            'llama3-8b':  'meta-llama/Meta-Llama-3-8B-Instruct',
            'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
            'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.3',
        }

    def call(self, prompt: str, model: str = 'llama3-8b',
             temperature: float = 0.3, max_tokens: int = 1000,
             use_system_prompt: bool = True, **kwargs) -> str:
        """
        Call vLLM API (OpenAI-compatible chat completions).

        For TSFL prompts (is_tsfl=True):
          - System prompt instructs ###value### format output (not JSON)
          - Stop tokens cleared so ### tokens are not cut off
          - Retries up to 3 times if response has no ###value### tokens
          - Temperature 0.3 to reduce flat/repetitive outputs

        For standard prompts:
          - System prompt instructs JSON output
          - Stop tokens prevent runaway generation
        """
        is_tsfl = kwargs.pop("is_tsfl", False)

        base_url = self.endpoints.get(model)
        if not base_url:
            raise ValueError(f"Unknown model: {model}. "
                             f"Available: {list(self.endpoints.keys())}")

        model_id = self.model_ids.get(model)
        url = f"{base_url}/v1/chat/completions"

        messages = []
        if use_system_prompt:
            if is_tsfl:
                messages.append({
                    "role": "system",
                    "content": (
                        "You are a time series forecasting assistant. "
                        "You must output ONLY ###value### tokens separated by spaces. "
                        "Each value must be a decimal number between -1.0 and 1.0. "
                        "Each token MUST have a different value — never repeat the same value. "
                        "Do NOT output any explanation, reasoning, or text. "
                        "Output format: ###0.1234### ###0.2345### ###0.3456###"
                    )
                })
            else:
                messages.append({
                    "role": "system",
                    "content": (
                        "You are a JSON-only financial forecasting assistant. "
                        "You must output ONLY a valid JSON object with a 'predictions' key "
                        "containing a list of numbers. "
                        "Do NOT output any explanation, reasoning, markdown, or code blocks. "
                        "Start your response with { and end with }."
                    )
                })
        messages.append({"role": "user", "content": prompt})

        data = {
            "model":       model_id,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stop":        [] if is_tsfl else ["\n\n\n", "###", "Example"],
        }

        max_retries = 3 if is_tsfl else 1
        last_response = ""

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=data, timeout=120)
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                if is_tsfl:
                    if re.findall(r"###([+-]?\d+\.\d+)###", content):
                        return content
                    last_response = content
                    print(f"    [RETRY {attempt+1}/{max_retries}] No ###tokens### in response")
                    data["temperature"] = min(0.7, temperature + 0.2 * (attempt + 1))
                else:
                    return content

            except requests.exceptions.ConnectionError:
                raise ConnectionError(
                    f"Could not connect to {model} at {base_url}. "
                    f"Start vLLM servers: bash scripts/start_vllm_servers.sh"
                )
            except requests.exceptions.Timeout:
                raise TimeoutError(f"Request to {model} timed out.")
            except Exception as e:
                raise RuntimeError(f"vLLM error: {str(e)}")

        return last_response



class OllamaAPI(BaseAPI):
    """Ollama API integration - BACKUP (FREE)."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.models = {
            'llama3-8b':  'llama3',
            'qwen2.5-7b': 'qwen2.5:7b',
            'mistral-7b': 'mistral',
        }

    def call(self, prompt: str, model: str = 'llama3-8b',
             temperature: float = 0.1, **kwargs) -> str:
        model_id = self.models.get(model, model)
        url = f"{self.base_url}/api/generate"

        # Prepend system instruction inline (Ollama doesn't have system role)
        full_prompt = (
            "You are a JSON-only financial forecasting assistant. "
            "Output ONLY a valid JSON object like {\"predictions\": [v1, v2, ...]}. "
            "No explanation, no markdown.\n\n"
            + prompt
        )

        data = {
            "model": model_id,
            "prompt": full_prompt,
            "temperature": temperature,
            "stream": False,
        }

        try:
            response = requests.post(url, json=data, timeout=180)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Install: curl -fsSL https://ollama.com/install.sh | sh"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama error: {str(e)}")


class LLMForecaster:
    """Unified LLM forecaster supporting vLLM and Ollama backends."""

    def __init__(self, backend: str = 'vllm', base_url: Optional[str] = None):
        self.backend = backend
        if backend == 'vllm':
            self.api = vLLMAPI()
        elif backend == 'ollama':
            self.api = OllamaAPI(base_url or "http://localhost:11434")
        else:
            raise ValueError(
                f"Unknown backend: {backend}. "
                "Available: 'vllm', 'ollama'"
            )

    def predict(self, prompt: str, model: str = 'llama3-8b',
                expected_scale: Optional[float] = None, **kwargs) -> np.ndarray:
        """
        Generate predictions using LLM.

        Args:
            prompt: Formatted prompt string
            model: LLM model name
            expected_scale: Last known price for scale-filtering parsed values.
                            Pass context_data['close'].iloc[-1] from the caller.
            **kwargs: Passed to the backend API call (e.g. temperature, max_tokens)

        Returns:
            Array of predicted values
        """
        print(f"[{self.backend}/{model}] Calling LLM...")
        response = self.api.call(prompt, model=model, **kwargs)
        predictions = self.api.parse_predictions(response,
                                                 expected_scale=expected_scale)
        print(f"  Received {len(predictions)} predictions")
        return predictions


def create_forecaster_with_fallback() -> LLMForecaster:
    """Try vLLM first, fall back to Ollama."""
    try:
        requests.get('http://localhost:18000/v1/models', timeout=2)
        print("Using vLLM (GPU - fastest)")
        return LLMForecaster(backend='vllm')
    except Exception:
        pass
    try:
        requests.get('http://localhost:11434', timeout=2)
        print("vLLM unavailable, using Ollama (slower)")
        return LLMForecaster(backend='ollama')
    except Exception:
        pass
    raise RuntimeError(
        "No LLM backends available!\n"
        "Start vLLM: bash scripts/start_vllm_servers.sh\n"
        "OR install Ollama: curl -fsSL https://ollama.com/install.sh | sh"
    )