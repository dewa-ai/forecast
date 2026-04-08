#!/usr/bin/env python3
"""
ts_tokenizer.py
Time Series as Foreign Language (TSFL) — following ChatTime methodology.

Reference:
    Wang et al. (2024). ChatTime: A Unified Multimodal Time Series Foundation
    Model Bridging Numerical and Textual Data. AAAI 2025. arXiv:2412.11376

Core pipeline (from paper Section 3.2):
    1. Normalize  : min-max scale history to [-0.5, 0.5]
    2. Discretize : map to nearest bin centre among 10K uniform bins in [-1, 1]
    3. Serialize  : wrap each value with "###" → "###0.3529### ###0.4999###"
    4. De-serialize / De-normalize: reverse to recover original prices

Usage:
    from ts_tokenizer import TSFLTokenizer, TSFLPromptBuilder

    tok = TSFLTokenizer()
    token_str = tok.encode(prices)
    pred_prices = tok.decode(token_str, ref_prices=prices, horizon=5)

    builder = TSFLPromptBuilder("USDIDR", horizon=5)
    prompt = builder.build_zero_shot(df)
    prompt = builder.build_with_news(df, news_articles)
"""

from __future__ import annotations
import re
from typing import List, Optional
import numpy as np
import pandas as pd

# Constants from ChatTime paper
N_BINS      = 10_000
BIN_EDGES   = np.linspace(-1.0, 1.0, N_BINS + 1)
BIN_CENTRES = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2


class TSFLTokenizer:
    """Encode/decode price series using ChatTime ###value### format."""

    def encode(self, prices: np.ndarray) -> str:
        """prices → '###0.3529###  ###0.4999###  ...' """
        prices = np.asarray(prices, dtype=float)
        p_min, p_max = prices.min(), prices.max()

        # Step 1: normalize to [-0.5, 0.5]
        if p_max == p_min:
            scaled = np.zeros_like(prices)
        else:
            scaled = (prices - p_min) / (p_max - p_min) - 0.5

        # Step 2: discretize to nearest bin centre
        clipped     = np.clip(scaled, -1.0, 1.0)
        bin_indices = np.searchsorted(BIN_EDGES[1:-1], clipped)
        discretized = BIN_CENTRES[bin_indices]

        # Step 3: serialize
        return "  ".join(f"###{v:.4f}###" for v in discretized)

    def encode_with_stats(self, prices: np.ndarray):
        """Returns (token_str, p_min, p_max) — stats needed for decoding."""
        prices = np.asarray(prices, dtype=float)
        p_min, p_max = float(prices.min()), float(prices.max())
        return self.encode(prices), p_min, p_max

    def decode(self, token_str: str, ref_prices: np.ndarray, horizon: int) -> np.ndarray:
        """'###0.xxxx###...' + history stats → original-scale prices."""
        ref_prices = np.asarray(ref_prices, dtype=float)
        p_min, p_max = float(ref_prices.min()), float(ref_prices.max())

        matches = re.findall(r"###([+-]?\d+\.\d+)###", token_str)
        if not matches:
            # Fallback: small random walk from last price instead of flat
            last = ref_prices[-1]
            daily_vol = float(np.std(np.diff(ref_prices))) if len(ref_prices) > 1 else last * 0.001
            rw = np.cumsum(np.random.normal(0, daily_vol, horizon))
            return last + rw

        scaled = np.array([float(m) for m in matches[:horizon]])

        if p_max == p_min:
            prices = np.full(len(scaled), p_min)
        else:
            prices = (scaled + 0.5) * (p_max - p_min) + p_min

        if len(prices) < horizon:
            prices = np.concatenate([prices, np.full(horizon - len(prices), prices[-1])])
        return prices[:horizon]


class TSFLPromptBuilder:
    """
    Build LLM prompts with TSFL-encoded price history.

    Two variants:
      build_zero_shot()  — token history only
      build_with_news()  — token history + news headlines
    """

    def __init__(self, currency_pair: str, horizon: int = 5):
        self.currency_pair  = currency_pair
        self.base_currency  = currency_pair[:3]
        self.quote_currency = currency_pair[3:]
        self.horizon        = horizon
        self.tokenizer      = TSFLTokenizer()

    def build_zero_shot(self, historical_data: pd.DataFrame, lookback_days: int = 30) -> str:
        """Zero-shot TSFL prompt. Mirrors ChatTime Appendix A.1."""
        recent        = historical_data.tail(lookback_days)
        prices        = recent["close"].values
        current_price = float(prices[-1])
        token_str, p_min, p_max = self.tokenizer.encode_with_stats(prices)

        return f"""You are a helpful assistant that performs time series prediction.
The user will provide a sequence encoded as foreign language tokens and you will predict the sequence.

The {self.base_currency}/{self.quote_currency} exchange rate has been normalized and discretized \
following the ChatTime protocol (Wang et al., 2024):
  - Min-max scaled to [-0.5, 0.5] using history window
  - Discretized to nearest bin centre in [-1, 1] (10,000 bins)
  - Serialized with ### mark characters: ###value###

History window: min={p_min:.4f}, max={p_max:.4f}, current={current_price:.4f}

### Instruction:
Please predict the following sequence carefully.
Output EXACTLY {self.horizon} token(s) in the same ###value### format.
Each token MUST have a different value reflecting the trend — do NOT repeat the same value.
Format: ###value### ###value### ... (space-separated, exactly {self.horizon} tokens)

### Input:
{token_str}

### Response:
"""

    def build_with_news(
        self,
        historical_data: pd.DataFrame,
        news_articles: list,
        lookback_days: int = 30,
        max_news: int = 5,
    ) -> str:
        """TSFL prompt + news. Mirrors ChatTime Appendix A.2 (context-guided)."""
        recent        = historical_data.tail(lookback_days)
        prices        = recent["close"].values
        current_price = float(prices[-1])
        token_str, p_min, p_max = self.tokenizer.encode_with_stats(prices)

        sorted_news = sorted(news_articles, key=lambda x: x.get("published_at", ""), reverse=True)[:max_news]
        news_text = "\n".join(
            f"{i}. [{a.get('published_at','')[:10]}] {a.get('title','')}"
            for i, a in enumerate(sorted_news, 1)
        ) or "No recent news available."

        return f"""You are a helpful assistant that performs time series prediction.
The user will provide a sequence encoded as foreign language tokens along with context knowledge, \
and you will predict the sequence.

The {self.base_currency}/{self.quote_currency} exchange rate has been normalized and discretized \
following the ChatTime protocol (Wang et al., 2024):
  - Min-max scaled to [-0.5, 0.5] using history window
  - Discretized to nearest bin centre in [-1, 1] (10,000 bins)
  - Serialized with ### mark characters: ###value###

History window: min={p_min:.4f}, max={p_max:.4f}, current={current_price:.4f}

### Instruction:
Please predict the following sequence carefully. Context knowledge you may consider:
{news_text}

Output EXACTLY {self.horizon} token(s) in the same ###value### format.
Each token MUST have a different value reflecting the trend — do NOT repeat the same value.
Format: ###value### ###value### ... (space-separated, exactly {self.horizon} tokens)

### Input:
{token_str}

### Response:
"""

    def build_few_shot(self, historical_data: pd.DataFrame,
                       lookback_days: int = 30, n_examples: int = 3) -> str:
        """Few-shot TSFL prompt without news."""
        recent        = historical_data.tail(lookback_days)
        prices        = recent["close"].values
        current_price = float(prices[-1])
        token_str, p_min, p_max = self.tokenizer.encode_with_stats(prices)
        examples      = self._build_examples(historical_data, n_examples)

        return f"""You are a helpful assistant that performs time series prediction.
The user will provide a sequence encoded as foreign language tokens and you will predict the sequence.

The {self.base_currency}/{self.quote_currency} exchange rate has been normalized and discretized \
following the ChatTime protocol (Wang et al., 2024):
  - Min-max scaled to [-0.5, 0.5] using history window
  - Discretized to nearest bin centre in [-1, 1] (10,000 bins)
  - Serialized with ### mark characters: ###value###

History window: min={p_min:.4f}, max={p_max:.4f}, current={current_price:.4f}

Here are {n_examples} examples of past predictions:

{examples}

### Instruction:
Please predict the following sequence carefully.
Output EXACTLY {self.horizon} token(s) in the same ###value### format.
Each token MUST have a different value reflecting the trend — do NOT repeat the same value.
Format: ###value### ###value### ... (space-separated, exactly {self.horizon} tokens)

### Input:
{token_str}

### Response:
"""

    def build_few_shot_with_news(self, historical_data: pd.DataFrame,
                                  news_articles: list,
                                  lookback_days: int = 30,
                                  n_examples: int = 3,
                                  max_news: int = 5) -> str:
        """Few-shot TSFL prompt with news (full proposed method)."""
        recent        = historical_data.tail(lookback_days)
        prices        = recent["close"].values
        current_price = float(prices[-1])
        token_str, p_min, p_max = self.tokenizer.encode_with_stats(prices)
        examples      = self._build_examples(historical_data, n_examples)

        sorted_news = sorted(news_articles, key=lambda x: x.get("published_at", ""), reverse=True)[:max_news]
        news_text = "\n".join(
            f"{i}. [{a.get('published_at','')[:10]}] {a.get('title','')}"
            for i, a in enumerate(sorted_news, 1)
        ) or "No recent news available."

        return f"""You are a helpful assistant that performs time series prediction.
The user will provide a sequence encoded as foreign language tokens along with context knowledge, \
and you will predict the sequence.

The {self.base_currency}/{self.quote_currency} exchange rate has been normalized and discretized \
following the ChatTime protocol (Wang et al., 2024):
  - Min-max scaled to [-0.5, 0.5] using history window
  - Discretized to nearest bin centre in [-1, 1] (10,000 bins)
  - Serialized with ### mark characters: ###value###

History window: min={p_min:.4f}, max={p_max:.4f}, current={current_price:.4f}

Here are {n_examples} examples of past predictions:

{examples}

### Instruction:
Please predict the following sequence carefully. Context knowledge you may consider:
{news_text}

Output EXACTLY {self.horizon} token(s) in the same ###value### format.
Each token MUST have a different value reflecting the trend — do NOT repeat the same value.
Format: ###value### ###value### ... (space-separated, exactly {self.horizon} tokens)

### Input:
{token_str}

### Response:
"""

    def _build_examples(self, historical_data: pd.DataFrame, n_examples: int) -> str:
        """Build few-shot examples in ###token### format from historical windows."""
        prices   = historical_data["close"].values
        lookback = 30
        gap      = 10
        examples = []

        for i in range(n_examples):
            start     = i * (lookback + self.horizon + gap)
            hist_end  = start + lookback
            fut_end   = hist_end + self.horizon
            if fut_end > len(prices):
                break

            hist_prices   = prices[start:hist_end]
            future_prices = prices[hist_end:fut_end]

            hist_tokens, p_min, p_max = self.tokenizer.encode_with_stats(hist_prices)
            # encode future using same min/max for consistency
            fut_norm  = np.clip((future_prices - p_min) / (p_max - p_min + 1e-8) - 0.5, -0.5, 0.5)
            fut_disc  = np.round(fut_norm * 10000) / 10000
            fut_tokens = "  ".join(f"###{v:.4f}###" for v in fut_disc)

            examples.append(
                f"Example {i+1}:\n"
                f"### Input:\n{hist_tokens}\n"
                f"### Response:\n{fut_tokens}\n"
            )

        return "\n".join(examples) if examples else "No examples available."


    def decode_response(self, response: str, ref_prices: np.ndarray, horizon: int) -> np.ndarray:
        """Decode LLM ###token### response → original price scale."""
        return self.tokenizer.decode(response, ref_prices, horizon)


if __name__ == "__main__":
    np.random.seed(42)
    prices = 15000 + np.cumsum(np.random.randn(50) * 20)
    dates  = pd.date_range("2024-01-01", periods=50).strftime("%Y-%m-%d").tolist()
    df     = pd.DataFrame({"date": dates, "close": prices})

    tok = TSFLTokenizer()
    encoded = tok.encode(prices[-10:])
    print("Encoded (last 10):", encoded[:100], "...")

    fake_resp = "###0.5023### ###0.5087### ###0.4998### ###0.5112### ###0.5201###"
    decoded = tok.decode(fake_resp, prices[-30:], horizon=5)
    print("Decoded:", decoded)
    print("Last known:", prices[-1])

    builder = TSFLPromptBuilder("USDIDR", horizon=5)
    prompt  = builder.build_zero_shot(df, lookback_days=30)
    print("\nPrompt preview:\n", prompt[:500])