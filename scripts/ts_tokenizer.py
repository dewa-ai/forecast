#!/usr/bin/env python3
"""
ts_tokenizer.py
Time Series as a Foreign Language — inspired by ChatTime / TimesFM approach.

Core idea: Treat each numeric time series value as a "token" in a special vocabulary
that LLMs can read as natural language, enabling LLMs to reason over temporal patterns
using their pretrained language understanding.

Three tokenization strategies:
  1. WordLevel    : Each discretized bin → a vocab word  (e.g. "rise_moderate")
  2. PatchText    : Fixed-length patches → compact string (e.g. "[+0.3%,+0.1%,-0.2%]")
  3. SignalText   : Semantic descriptors  (e.g. "steady climb with low volatility")

Usage:
    from ts_tokenizer import TSTokenizer

    tok = TSTokenizer(strategy="patch", patch_size=5)
    token_str = tok.encode(prices)           # returns string for LLM prompt
    decoded   = tok.decode(token_str, last_price)   # back to approximate floats
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Vocabulary for word-level tokenization
# ---------------------------------------------------------------------------

_DIRECTION = {
    # (sign, magnitude bucket) -> word token
    (-1, 0): "flat_down",
    (-1, 1): "drop_small",
    (-1, 2): "drop_moderate",
    (-1, 3): "drop_large",
    (-1, 4): "drop_extreme",
    (0,  0): "flat",
    (1,  0): "flat_up",
    (1,  1): "rise_small",
    (1,  2): "rise_moderate",
    (1,  3): "rise_large",
    (1,  4): "rise_extreme",
}

# Thresholds for percentage change magnitude (in %)
_MAGNITUDE_THRESHOLDS = [0.05, 0.2, 0.5, 1.5]   # <0.05% → flat, <0.2% → small, etc.


def _magnitude_bucket(pct_change: float) -> int:
    abs_pct = abs(pct_change)
    for i, thresh in enumerate(_MAGNITUDE_THRESHOLDS):
        if abs_pct < thresh:
            return i
    return len(_MAGNITUDE_THRESHOLDS)


def _pct_to_token(pct_change: float) -> str:
    sign = 0 if abs(pct_change) < 1e-8 else (1 if pct_change > 0 else -1)
    mag  = _magnitude_bucket(pct_change)
    return _DIRECTION.get((sign, mag), f"change_{pct_change:+.2f}pct")


# ---------------------------------------------------------------------------
# TSTokenizer
# ---------------------------------------------------------------------------

class TSTokenizer:
    """
    Tokenize a numeric time series into a language-model-friendly string.

    Parameters
    ----------
    strategy : str
        "word"   – word-level vocabulary tokens (most compact)
        "patch"  – patch-level percentage sequences (most faithful)
        "signal" – semantic natural-language description (most readable)
    patch_size : int
        Number of timesteps per patch (used by "patch" and "signal").
    include_absolute : bool
        If True, prefix each patch/word sequence with the anchor price so the
        LLM can reconstruct absolute values.
    """

    STRATEGIES = ("word", "patch", "signal")

    def __init__(
        self,
        strategy: str = "patch",
        patch_size: int = 5,
        include_absolute: bool = True,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")
        self.strategy = strategy
        self.patch_size = patch_size
        self.include_absolute = include_absolute

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, prices: np.ndarray, dates: Optional[List[str]] = None) -> str:
        """
        Encode a 1-D price array into a string suitable for an LLM prompt.

        Parameters
        ----------
        prices : np.ndarray  shape (T,)
        dates  : optional list of date strings, same length as prices

        Returns
        -------
        token_string : str
        """
        prices = np.asarray(prices, dtype=float)
        if len(prices) < 2:
            return str(prices[0]) if len(prices) == 1 else ""

        if self.strategy == "word":
            return self._encode_word(prices, dates)
        elif self.strategy == "patch":
            return self._encode_patch(prices, dates)
        else:
            return self._encode_signal(prices, dates)

    def decode(
        self,
        token_string: str,
        anchor_price: float,
        horizon: int,
    ) -> np.ndarray:
        """
        Approximate inverse: parse LLM-generated token sequence back into prices.

        Works best with "patch" strategy. For "word" and "signal" strategies,
        a rough centre-of-range for each bucket is used.

        Parameters
        ----------
        token_string : str   – LLM output continuation of the token sequence
        anchor_price : float – last known price (t=0 for the prediction window)
        horizon      : int   – how many steps to decode

        Returns
        -------
        prices : np.ndarray  shape (horizon,)
        """
        if self.strategy == "patch":
            return self._decode_patch(token_string, anchor_price, horizon)
        else:
            return self._decode_word_signal(token_string, anchor_price, horizon)

    # ------------------------------------------------------------------
    # Word-level encoding
    # ------------------------------------------------------------------

    def _encode_word(
        self, prices: np.ndarray, dates: Optional[List[str]]
    ) -> str:
        pct_changes = np.diff(prices) / prices[:-1] * 100
        tokens = [_pct_to_token(p) for p in pct_changes]

        if self.include_absolute:
            anchor = f"[anchor:{prices[0]:.4f}]"
            return anchor + " " + " ".join(tokens)
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # Patch-level encoding
    # ------------------------------------------------------------------

    def _encode_patch(
        self, prices: np.ndarray, dates: Optional[List[str]]
    ) -> str:
        """
        Split the series into non-overlapping patches of `patch_size` steps.
        Each patch is represented as a compact bracket string:
            [+0.12%,+0.05%,-0.18%,+0.30%,+0.11%]
        followed by a separator |
        """
        pct_changes = np.diff(prices) / prices[:-1] * 100
        patches = []

        if self.include_absolute:
            patches.append(f"<price:{prices[0]:.4f}>")

        for start in range(0, len(pct_changes), self.patch_size):
            chunk = pct_changes[start : start + self.patch_size]
            # Date label for first step in patch (if available)
            if dates is not None and start + 1 < len(dates):
                label = f"[{dates[start+1]}:"
            else:
                label = "["
            inner = ",".join(f"{v:+.2f}%" for v in chunk)
            patches.append(label + inner + "]")

        return " | ".join(patches)

    # ------------------------------------------------------------------
    # Signal / semantic encoding
    # ------------------------------------------------------------------

    def _encode_signal(
        self, prices: np.ndarray, dates: Optional[List[str]]
    ) -> str:
        """
        Convert each patch into a human-readable semantic phrase:
            "steady climb (+0.4% avg, low vol)"
        This mirrors the "signal text" representation in ChatTime.
        """
        pct_changes = np.diff(prices) / prices[:-1] * 100
        parts = []

        if self.include_absolute:
            parts.append(f"Starting price: {prices[0]:.4f}")

        for start in range(0, len(pct_changes), self.patch_size):
            chunk = pct_changes[start : start + self.patch_size]
            avg   = float(np.mean(chunk))
            vol   = float(np.std(chunk))
            mn    = float(np.min(chunk))
            mx    = float(np.max(chunk))

            # Direction phrase
            if abs(avg) < 0.05:
                direction = "sideways movement"
            elif avg > 0:
                direction = "upward trend" if avg > 0.3 else "mild upward drift"
            else:
                direction = "downward trend" if avg < -0.3 else "mild downward drift"

            # Volatility phrase
            if vol < 0.1:
                vol_desc = "very low volatility"
            elif vol < 0.3:
                vol_desc = "low volatility"
            elif vol < 0.7:
                vol_desc = "moderate volatility"
            else:
                vol_desc = "high volatility"

            # Date context
            if dates is not None and start + 1 < len(dates):
                date_ctx = f"[{dates[start+1]}] "
            else:
                date_ctx = ""

            phrase = (
                f"{date_ctx}{direction} "
                f"(avg {avg:+.2f}%/day, range [{mn:+.2f}%,{mx:+.2f}%], {vol_desc})"
            )
            parts.append(phrase)

        return "; ".join(parts)

    # ------------------------------------------------------------------
    # Decoding helpers
    # ------------------------------------------------------------------

    def _decode_patch(
        self, token_string: str, anchor_price: float, horizon: int
    ) -> np.ndarray:
        """Parse "[+0.12%,-0.05%,...]" patches back to prices."""
        pct_values = re.findall(r"([+-]?\d+\.?\d*)%", token_string)
        pcts = [float(p) / 100.0 for p in pct_values]
        prices = []
        current = anchor_price
        for i in range(horizon):
            if i < len(pcts):
                current = current * (1.0 + pcts[i])
            prices.append(current)
        return np.array(prices)

    def _decode_word_signal(
        self, token_string: str, anchor_price: float, horizon: int
    ) -> np.ndarray:
        """Rough decode: map word tokens or percentage hints back to prices."""
        # First try to extract explicit percentages
        pct_values = re.findall(r"([+-]?\d+\.?\d*)%", token_string)
        if pct_values:
            return self._decode_patch(token_string, anchor_price, horizon)

        # Map word tokens to approximate daily change
        _TOKEN_TO_PCT = {
            "flat": 0.0, "flat_up": 0.02, "flat_down": -0.02,
            "rise_small": 0.1, "rise_moderate": 0.35, "rise_large": 0.75, "rise_extreme": 2.0,
            "drop_small": -0.1, "drop_moderate": -0.35, "drop_large": -0.75, "drop_extreme": -2.0,
        }
        tokens = token_string.strip().split()
        prices = []
        current = anchor_price
        for i in range(horizon):
            if i < len(tokens):
                pct = _TOKEN_TO_PCT.get(tokens[i], 0.0) / 100.0
                current = current * (1.0 + pct)
            prices.append(current)
        return np.array(prices)


# ---------------------------------------------------------------------------
# Prompt builder extension: TSFLPromptBuilder
# ---------------------------------------------------------------------------

class TSFLPromptBuilder:
    """
    Time Series as Foreign Language Prompt Builder.

    Wraps TSTokenizer to build LLM prompts where the historical price series
    is expressed as a token sequence instead of raw numbers.  This mirrors
    the ChatTime paper's key insight: treating temporal patterns as a "language"
    the model can reason over.

    Parameters
    ----------
    currency_pair : str
    horizon       : int   – forecast horizon in days
    strategy      : str   – tokenization strategy ("word", "patch", "signal")
    patch_size    : int   – patch size for patch/signal strategies
    """

    def __init__(
        self,
        currency_pair: str,
        horizon: int = 5,
        strategy: str = "patch",
        patch_size: int = 5,
    ):
        self.currency_pair  = currency_pair
        self.horizon        = horizon
        self.base_currency  = currency_pair[:3]
        self.quote_currency = currency_pair[3:]
        self.tokenizer      = TSTokenizer(
            strategy=strategy,
            patch_size=patch_size,
            include_absolute=True,
        )
        self.strategy = strategy

    def build_tsfl_zero_shot(
        self,
        historical_data: pd.DataFrame,
        lookback_days: int = 30,
    ) -> str:
        """Zero-shot prompt using token-encoded price history."""
        recent = historical_data.tail(lookback_days).copy()
        prices = recent["close"].values
        dates  = recent["date"].tolist() if "date" in recent.columns else None

        token_str = self.tokenizer.encode(prices, dates)
        current_price = float(prices[-1])

        strategy_desc = {
            "word":   "word-level change tokens (e.g. rise_moderate, drop_small)",
            "patch":  "patch-level percentage sequences (e.g. [+0.12%,-0.05%,+0.30%])",
            "signal": "semantic signal phrases (e.g. 'upward trend, low volatility')",
        }[self.strategy]

        example_preds = ", ".join([f"{current_price * (1 + 0.001*i):.4f}" for i in range(1, self.horizon+1)])

        prompt = f"""You are an expert financial analyst who reads time series data encoded as a sequence of tokens.

The {self.base_currency}/{self.quote_currency} exchange rate history has been encoded using {strategy_desc}.
This representation captures temporal patterns in a compact, language-model-friendly format.

Encoded Price History (last {lookback_days} days):
{token_str}

Current Price: {current_price:.4f}

Your task: Analyze the encoded token sequence to identify patterns, trends, and momentum.
Then predict the exchange rate for the next {self.horizon} day(s).

Key pattern types to look for:
- Trend persistence (consecutive rise/drop tokens)
- Mean reversion (after extreme tokens)
- Volatility clustering (many high-magnitude tokens together)
- Momentum signals (accelerating or decelerating changes)

CRITICAL: Output ONLY a JSON object with exactly {self.horizon} predictions. No explanation.
Format: {{"predictions": [{example_preds}]}}

{{"predictions": ["""

        return prompt

    def build_tsfl_few_shot(
        self,
        historical_data: pd.DataFrame,
        lookback_days: int = 30,
        n_examples: int = 3,
    ) -> str:
        """Few-shot prompt with tokenized historical examples."""
        example_size = lookback_days + self.horizon
        examples_text = []

        for i in range(n_examples):
            start = i * (example_size + 5)
            end_hist = start + lookback_days
            end_future = end_hist + self.horizon

            if end_future > len(historical_data):
                break

            hist_slice   = historical_data.iloc[start:end_hist]
            future_slice = historical_data.iloc[end_hist:end_future]

            if len(future_slice) < self.horizon:
                continue

            prices = hist_slice["close"].values
            dates  = hist_slice["date"].tolist() if "date" in hist_slice.columns else None
            token_str = self.tokenizer.encode(prices, dates)
            future_vals = [round(float(v), 4) for v in future_slice["close"].values]

            examples_text.append(
                f"Example {i+1}:\n"
                f"Encoded history: {token_str}\n"
                f"Actual next {self.horizon} day(s): {future_vals}"
            )

        if not examples_text:
            examples_text = ["(no examples available)"]

        # Current window
        current = historical_data.tail(lookback_days)
        prices  = current["close"].values
        dates   = current["date"].tolist() if "date" in current.columns else None
        current_token_str = self.tokenizer.encode(prices, dates)
        current_price = float(prices[-1])
        example_preds = ", ".join([f"{current_price * (1 + 0.001*i):.4f}" for i in range(1, self.horizon+1)])

        prompt = f"""You are an expert financial analyst who reads tokenized time series data.

Study these examples to learn how to map token patterns to future prices:

{chr(10).join(examples_text)}

Now predict for the current window:
Encoded history: {current_token_str}
Current price: {current_price:.4f}

CRITICAL: Output ONLY a JSON object with exactly {self.horizon} predictions. No explanation.
Format: {{"predictions": [{example_preds}]}}

{{"predictions": ["""

        return prompt

    def build_tsfl_with_news(
        self,
        historical_data: pd.DataFrame,
        news_articles: list,
        lookback_days: int = 30,
        max_news: int = 5,
    ) -> str:
        """Token-encoded price + news augmented prompt."""
        recent = historical_data.tail(lookback_days)
        prices = recent["close"].values
        dates  = recent["date"].tolist() if "date" in recent.columns else None
        token_str = self.tokenizer.encode(prices, dates)
        current_price = float(prices[-1])

        # Format news
        news_lines = []
        sorted_news = sorted(news_articles, key=lambda x: x.get("published_at", ""), reverse=True)[:max_news]
        for i, art in enumerate(sorted_news, 1):
            title = art.get("title", "")
            date  = art.get("published_at", "")[:10]
            news_lines.append(f"{i}. [{date}] {title}")
        news_text = "\n".join(news_lines) if news_lines else "No recent news."

        example_preds = ", ".join([f"{current_price * (1 + 0.001*i):.4f}" for i in range(1, self.horizon+1)])

        prompt = f"""You are an expert financial analyst. You receive time series data encoded as tokens AND recent news.

Recent Market News:
{news_text}

Encoded Price History (last {lookback_days} days, patch-level tokens):
{token_str}

Current Price: {current_price:.4f}

Combine the token pattern analysis with news sentiment to forecast the next {self.horizon} day(s).

CRITICAL: Output ONLY a JSON object with exactly {self.horizon} predictions. No explanation.
Format: {{"predictions": [{example_preds}]}}

{{"predictions": ["""

        return prompt

    def decode_response(
        self,
        response: str,
        anchor_price: float,
        horizon: int,
    ) -> np.ndarray:
        """Decode LLM response back to price predictions."""
        return self.tokenizer.decode(response, anchor_price, horizon)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    # Synthetic data
    np.random.seed(42)
    n = 200
    prices = 15000 + np.cumsum(np.random.randn(n) * 20)
    dates  = pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"date": dates, "close": prices})

    print("=" * 60)
    print("TSTokenizer — Time Series as Foreign Language")
    print("=" * 60)

    for strategy in ("word", "patch", "signal"):
        tok = TSTokenizer(strategy=strategy, patch_size=5)
        encoded = tok.encode(prices[-30:], dates[-30:])
        print(f"\n[{strategy.upper()}] encoded (first 200 chars):")
        print(encoded[:200])

    print("\n" + "=" * 60)
    print("TSFLPromptBuilder — sample prompt (patch strategy, first 300 chars)")
    print("=" * 60)
    builder = TSFLPromptBuilder("USDIDR", horizon=5, strategy="patch", patch_size=5)
    prompt = builder.build_tsfl_zero_shot(df, lookback_days=30)
    print(prompt[:300], "...")