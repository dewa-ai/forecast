#!/usr/bin/env python3
"""
LLM Prompt Templates for FX Forecasting.

Provides prompt templates for:
- Zero-shot forecasting (no examples)
- Few-shot forecasting (with examples)
- News-augmented forecasting (with news context)
"""

from __future__ import annotations

import json
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class PromptBuilder:
    """Build prompts for LLM-based FX forecasting."""
    
    def __init__(self, currency_pair: str, horizon: int = 5):
        self.currency_pair = currency_pair
        self.horizon = horizon
        self.seq_length = 30
        self.base_currency = currency_pair[:3]
        self.quote_currency = currency_pair[3:]
    
    def _strict_output_instruction(self) -> str:
        """Return strict JSON output instruction to prevent LLM from writing explanations."""
        example_values = ", ".join([str(round(1000 + i * 0.5, 2)) for i in range(self.horizon)])
        return f"""CRITICAL INSTRUCTIONS:
- Output ONLY a single JSON object, nothing else
- Do NOT write any explanation, reasoning, or code
- Do NOT use markdown, backticks, or code blocks
- The JSON must contain EXACTLY {self.horizon} numbers in the predictions array
- Start your response with {{ and end with }}

Required output format:
{{"predictions": [{example_values}]}}"""

    def build_zero_shot(self, historical_data: pd.DataFrame,
                        lookback_days: int = 30) -> str:
        recent_data = historical_data.tail(lookback_days)
        price_history = self._format_price_history(recent_data)
        
        current_price = recent_data['close'].iloc[-1]
        mean_price = recent_data['close'].mean()
        volatility = recent_data['close'].std()
        trend = "upward" if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0] else "downward"
        
        prompt = f"""You are an expert financial analyst specializing in foreign exchange (FX) forecasting.

Task: Predict the {self.base_currency}/{self.quote_currency} exchange rate for the next {self.horizon} days.

Historical Data (Last {lookback_days} days):
{price_history}

Current Market Statistics:
- Current Rate: {current_price:.4f}
- {lookback_days}-day Average: {mean_price:.4f}
- Volatility (Std Dev): {volatility:.4f}
- Recent Trend: {trend}

{self._strict_output_instruction()}

{{"predictions": ["""
        
        return prompt
    
    def build_few_shot(self, historical_data: pd.DataFrame,
                       lookback_days: int = 30,
                       n_examples: int = 3) -> str:
        examples = self._build_examples(historical_data, n_examples)
        
        current_data = historical_data.tail(lookback_days)
        price_history = self._format_price_history(current_data)
        
        current_price = current_data['close'].iloc[-1]
        mean_price = current_data['close'].mean()
        volatility = current_data['close'].std()
        trend = "upward" if current_data['close'].iloc[-1] > current_data['close'].iloc[0] else "downward"
        
        prompt = f"""You are an expert financial analyst specializing in foreign exchange (FX) forecasting.

Task: Predict the {self.base_currency}/{self.quote_currency} exchange rate for the next {self.horizon} days.

Here are examples of past forecasting scenarios:

{examples}

Now make your prediction for the current data:

Historical Data (Last {lookback_days} days):
{price_history}

Current Market Statistics:
- Current Rate: {current_price:.4f}
- {lookback_days}-day Average: {mean_price:.4f}
- Volatility (Std Dev): {volatility:.4f}
- Recent Trend: {trend}

{self._strict_output_instruction()}

{{"predictions": ["""
        
        return prompt
    
    def build_news_augmented(self, historical_data: pd.DataFrame,
                             news_articles: List[Dict],
                             lookback_days: int = 30,
                             max_news: int = 10) -> str:
        recent_data = historical_data.tail(lookback_days)
        price_history = self._format_price_history(recent_data)
        news_context = self._format_news(news_articles, max_news)
        
        current_price = recent_data['close'].iloc[-1]
        mean_price = recent_data['close'].mean()
        volatility = recent_data['close'].std()
        trend = "upward" if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0] else "downward"
        
        prompt = f"""You are an expert financial analyst specializing in foreign exchange (FX) forecasting.

Task: Predict the {self.base_currency}/{self.quote_currency} exchange rate for the next {self.horizon} days.

Recent News & Market Events:
{news_context}

Historical Data (Last {lookback_days} days):
{price_history}

Current Market Statistics:
- Current Rate: {current_price:.4f}
- {lookback_days}-day Average: {mean_price:.4f}
- Volatility (Std Dev): {volatility:.4f}
- Recent Trend: {trend}

{self._strict_output_instruction()}

{{"predictions": ["""
        
        return prompt
    
    def build_few_shot_with_news(self, historical_data: pd.DataFrame,
                                 news_articles: List[Dict],
                                 lookback_days: int = 30,
                                 n_examples: int = 3,
                                 max_news: int = 10) -> str:
        examples = self._build_examples(historical_data, n_examples)
        
        current_data = historical_data.tail(lookback_days)
        price_history = self._format_price_history(current_data)
        news_context = self._format_news(news_articles, max_news)
        
        current_price = current_data['close'].iloc[-1]
        mean_price = current_data['close'].mean()
        volatility = current_data['close'].std()
        trend = "upward" if current_data['close'].iloc[-1] > current_data['close'].iloc[0] else "downward"
        
        prompt = f"""You are an expert financial analyst specializing in foreign exchange (FX) forecasting.

Task: Predict the {self.base_currency}/{self.quote_currency} exchange rate for the next {self.horizon} days.

Here are examples of past forecasting scenarios:

{examples}

Recent News & Market Events:
{news_context}

Now make your prediction for the current data:

Historical Data (Last {lookback_days} days):
{price_history}

Current Market Statistics:
- Current Rate: {current_price:.4f}
- {lookback_days}-day Average: {mean_price:.4f}
- Volatility (Std Dev): {volatility:.4f}
- Recent Trend: {trend}

{self._strict_output_instruction()}

{{"predictions": ["""
        
        return prompt
    
    def _format_price_history(self, data: pd.DataFrame) -> str:
        lines = []
        for _, row in data.iterrows():
            lines.append(f"{row['date']}: {row['close']:.4f}")
        return "\n".join(lines)
    
    def _format_news(self, news_articles: List[Dict], max_news: int) -> str:
        if not news_articles:
            return "No recent news available."
        
        sorted_news = sorted(
            news_articles,
            key=lambda x: x.get('published_at', ''),
            reverse=True
        )[:max_news]
        
        lines = []
        for i, article in enumerate(sorted_news, 1):
            title     = article.get('title', 'No title')
            date      = article.get('published_at', 'Unknown date')
            summary   = article.get('summary', '')
            publisher = article.get('publisher', 'Unknown source')
            
            if len(summary) > 200:
                summary = summary[:197] + "..."
            
            news_entry = f"{i}. [{date}] {title}\n   Source: {publisher}"
            if summary:
                news_entry += f"\n   Summary: {summary}"
            lines.append(news_entry)
        
        return "\n\n".join(lines)
    
    def _build_examples(self, data: pd.DataFrame, n_examples: int) -> str:
        """Build few-shot examples. Uses early portion of data so current data is untouched."""
        examples = []
        
        # Each example needs seq_length history + horizon future
        example_size = self.seq_length + self.horizon
        min_needed = example_size * n_examples
        
        if len(data) < min_needed + self.seq_length:
            # Reduce n_examples to fit
            n_examples = max(1, (len(data) - self.seq_length) // example_size)
        
        for i in range(n_examples):
            start_idx = i * example_size
            hist_end  = start_idx + self.seq_length
            future_end = hist_end + self.horizon
            
            if future_end > len(data):
                break
            
            hist_window   = data.iloc[start_idx:hist_end]
            future_window = data.iloc[hist_end:future_end]
            
            if len(future_window) < self.horizon:
                continue
            
            hist_str      = self._format_price_history(hist_window.tail(10))
            future_values = future_window['close'].values
            
            example = f"""Example {i+1}:
Historical Data (last 10 days):
{hist_str}

Actual next {self.horizon} day(s): {json.dumps([round(float(v), 4) for v in future_values])}
"""
            examples.append(example)
        
        if not examples:
            return "No examples available."
        
        return "\n".join(examples)