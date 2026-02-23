#!/usr/bin/env python3
"""
LLM Prompt Templates for FX Forecasting.

Provides prompt templates for:
- Zero-shot forecasting (no examples)
- Few-shot forecasting (with examples)
- News-augmented forecasting (with news context)

Example:
    from llm_prompts import PromptBuilder
    
    builder = PromptBuilder(currency_pair="USDIDR", horizon=5)
    prompt = builder.build_zero_shot(historical_data)
    prompt = builder.build_few_shot(historical_data, news_articles)
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
        """
        Args:
            currency_pair: Currency pair name (e.g., "USDIDR", "USDEUR")
            horizon: Forecast horizon in days (1, 5, or 10)
        """
        self.currency_pair = currency_pair
        self.horizon = horizon
        self.seq_length = 30  # Lookback window
        
        # Extract currency names for natural language
        self.base_currency = currency_pair[:3]  # USD
        self.quote_currency = currency_pair[3:]  # IDR
        
    def build_zero_shot(self, historical_data: pd.DataFrame, 
                       lookback_days: int = 30) -> str:
        """
        Build zero-shot prompt (no examples).
        
        Args:
            historical_data: DataFrame with 'date' and 'close' columns
            lookback_days: Number of historical days to include
            
        Returns:
            Formatted prompt string
        """
        # Get last N days of data
        recent_data = historical_data.tail(lookback_days)
        
        # Format historical prices
        price_history = self._format_price_history(recent_data)
        
        # Calculate basic statistics
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

Instructions:
1. Analyze the historical price movements and trends
2. Consider market volatility and recent patterns
3. Provide your forecast for the next {self.horizon} days
4. Output ONLY a JSON array with {self.horizon} predicted values

Output Format:
{{"predictions": [day1_value, day2_value, ..., day{self.horizon}_value]}}

Example Output:
{{"predictions": [15420.50, 15435.20, 15448.75, 15460.30, 15472.85]}}

Your forecast:"""
        
        return prompt
    
    def build_few_shot(self, historical_data: pd.DataFrame,
                      lookback_days: int = 30,
                      n_examples: int = 3) -> str:
        """
        Build few-shot prompt with examples.
        
        Args:
            historical_data: DataFrame with 'date' and 'close' columns
            lookback_days: Number of historical days to include
            n_examples: Number of example predictions to include
            
        Returns:
            Formatted prompt string
        """
        # Get recent data
        recent_data = historical_data.tail(lookback_days + self.horizon * n_examples)
        
        # Build examples from historical data
        examples = self._build_examples(recent_data, n_examples)
        
        # Get current data for prediction
        current_data = historical_data.tail(lookback_days)
        price_history = self._format_price_history(current_data)
        
        current_price = current_data['close'].iloc[-1]
        mean_price = current_data['close'].mean()
        volatility = current_data['close'].std()
        trend = "upward" if current_data['close'].iloc[-1] > current_data['close'].iloc[0] else "downward"
        
        prompt = f"""You are an expert financial analyst specializing in foreign exchange (FX) forecasting.

Task: Predict the {self.base_currency}/{self.quote_currency} exchange rate for the next {self.horizon} days.

Here are {n_examples} examples of past forecasting scenarios:

{examples}

Now, make your prediction based on the current data:

Historical Data (Last {lookback_days} days):
{price_history}

Current Market Statistics:
- Current Rate: {current_price:.4f}
- {lookback_days}-day Average: {mean_price:.4f}
- Volatility (Std Dev): {volatility:.4f}
- Recent Trend: {trend}

Instructions:
1. Learn from the pattern shown in the examples above
2. Analyze the current historical price movements
3. Provide your forecast for the next {self.horizon} days
4. Output ONLY a JSON array with {self.horizon} predicted values

Output Format:
{{"predictions": [day1_value, day2_value, ..., day{self.horizon}_value]}}

Your forecast:"""
        
        return prompt
    
    def build_news_augmented(self, historical_data: pd.DataFrame,
                           news_articles: List[Dict],
                           lookback_days: int = 30,
                           max_news: int = 10) -> str:
        """
        Build prompt with news context.
        
        Args:
            historical_data: DataFrame with 'date' and 'close' columns
            news_articles: List of news article dictionaries
            lookback_days: Number of historical days to include
            max_news: Maximum number of news articles to include
            
        Returns:
            Formatted prompt string
        """
        # Get recent data
        recent_data = historical_data.tail(lookback_days)
        price_history = self._format_price_history(recent_data)
        
        # Format news articles
        news_context = self._format_news(news_articles, max_news)
        
        # Calculate statistics
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

Instructions:
1. Analyze how recent news events may impact the exchange rate
2. Consider the historical price movements and trends
3. Integrate both fundamental (news) and technical (price) analysis
4. Provide your forecast for the next {self.horizon} days
5. Output ONLY a JSON array with {self.horizon} predicted values

Output Format:
{{"predictions": [day1_value, day2_value, ..., day{self.horizon}_value]}}

Example Output:
{{"predictions": [15420.50, 15435.20, 15448.75, 15460.30, 15472.85]}}

Your forecast:"""
        
        return prompt
    
    def build_few_shot_with_news(self, historical_data: pd.DataFrame,
                                news_articles: List[Dict],
                                lookback_days: int = 30,
                                n_examples: int = 3,
                                max_news: int = 10) -> str:
        """
        Build few-shot prompt with news (full model).
        
        Args:
            historical_data: DataFrame with 'date' and 'close' columns
            news_articles: List of news article dictionaries
            lookback_days: Number of historical days to include
            n_examples: Number of example predictions to include
            max_news: Maximum number of news articles to include
            
        Returns:
            Formatted prompt string
        """
        # Get recent data
        recent_data = historical_data.tail(lookback_days + self.horizon * n_examples)
        
        # Build examples
        examples = self._build_examples(recent_data, n_examples)
        
        # Get current data
        current_data = historical_data.tail(lookback_days)
        price_history = self._format_price_history(current_data)
        
        # Format news
        news_context = self._format_news(news_articles, max_news)
        
        # Calculate statistics
        current_price = current_data['close'].iloc[-1]
        mean_price = current_data['close'].mean()
        volatility = current_data['close'].std()
        trend = "upward" if current_data['close'].iloc[-1] > current_data['close'].iloc[0] else "downward"
        
        prompt = f"""You are an expert financial analyst specializing in foreign exchange (FX) forecasting.

Task: Predict the {self.base_currency}/{self.quote_currency} exchange rate for the next {self.horizon} days.

Here are {n_examples} examples of past forecasting scenarios:

{examples}

Recent News & Market Events:
{news_context}

Now, make your prediction based on the current data:

Historical Data (Last {lookback_days} days):
{price_history}

Current Market Statistics:
- Current Rate: {current_price:.4f}
- {lookback_days}-day Average: {mean_price:.4f}
- Volatility (Std Dev): {volatility:.4f}
- Recent Trend: {trend}

Instructions:
1. Learn from the pattern shown in the examples above
2. Analyze how recent news events may impact the exchange rate
3. Consider the historical price movements and trends
4. Integrate fundamental (news) and technical (price) analysis
5. Provide your forecast for the next {self.horizon} days
6. Output ONLY a JSON array with {self.horizon} predicted values

Output Format:
{{"predictions": [day1_value, day2_value, ..., day{self.horizon}_value]}}

Your forecast:"""
        
        return prompt
    
    def _format_price_history(self, data: pd.DataFrame) -> str:
        """Format price history as readable text."""
        lines = []
        for _, row in data.iterrows():
            lines.append(f"{row['date']}: {row['close']:.4f}")
        return "\n".join(lines)
    
    def _format_news(self, news_articles: List[Dict], max_news: int) -> str:
        """Format news articles as readable text."""
        if not news_articles:
            return "No recent news available."
        
        # Sort by date (newest first) and limit
        sorted_news = sorted(
            news_articles, 
            key=lambda x: x.get('published_at', ''), 
            reverse=True
        )[:max_news]
        
        lines = []
        for i, article in enumerate(sorted_news, 1):
            title = article.get('title', 'No title')
            date = article.get('published_at', 'Unknown date')
            summary = article.get('summary', '')
            publisher = article.get('publisher', 'Unknown source')
            
            # Truncate summary if too long
            if len(summary) > 200:
                summary = summary[:197] + "..."
            
            news_entry = f"{i}. [{date}] {title}\n   Source: {publisher}"
            if summary:
                news_entry += f"\n   Summary: {summary}"
            
            lines.append(news_entry)
        
        return "\n\n".join(lines)
    
    def _build_examples(self, data: pd.DataFrame, n_examples: int) -> str:
        """Build few-shot examples from historical data."""
        examples = []
        
        # Need enough data for examples
        min_data_needed = (self.seq_length + self.horizon) * n_examples + self.seq_length
        if len(data) < min_data_needed:
            print(f"[WARN] Not enough data for {n_examples} examples, using what's available")
            n_examples = max(1, (len(data) - self.seq_length) // (self.seq_length + self.horizon))
        
        # Create examples by sliding window with gap
        gap = 10  # Days between examples
        for i in range(n_examples):
            # Calculate indices
            start_idx = i * (self.seq_length + self.horizon + gap)
            hist_end = start_idx + self.seq_length
            future_end = hist_end + self.horizon
            
            # Check bounds
            if future_end > len(data):
                break
            
            # Get historical window
            hist_window = data.iloc[start_idx:hist_end]
            
            # Get actual future values
            future_window = data.iloc[hist_end:future_end]
            
            if len(future_window) < self.horizon:
                continue
            
            # Format example
            hist_str = self._format_price_history(hist_window.tail(10))  # Last 10 days
            future_values = future_window['close'].values
            
            example = f"""Example {i+1}:
Historical Data (last 10 days):
{hist_str}

Actual values for next {self.horizon} days:
{json.dumps([round(float(v), 2) for v in future_values])}
"""
            examples.append(example)
        
        if not examples:
            return "No examples available due to insufficient data."
        
        return "\n".join(examples)


class LLMForecaster:
    """
    Wrapper for calling LLM APIs with prompts.
    
    Supports: Llama 3 8B, Qwen 2.5 7B, Mistral 7B
    """
    
    def __init__(self, model_name: str = "llama3-8b", api_endpoint: str = None):
        """
        Args:
            model_name: Name of LLM model to use
            api_endpoint: API endpoint URL (if using custom deployment)
        """
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        
    def predict(self, prompt: str) -> np.ndarray:
        """
        Call LLM API and parse predictions.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Array of predicted values
        """
        # TODO: Implement actual API call based on your LLM deployment
        # This is a placeholder implementation
        
        # Option 1: Using Hugging Face Inference API
        # response = self._call_huggingface_api(prompt)
        
        # Option 2: Using local deployment (vLLM, Ollama, etc.)
        # response = self._call_local_api(prompt)
        
        # Option 3: Using OpenRouter or other hosted APIs
        # response = self._call_openrouter_api(prompt)
        
        # For now, return placeholder
        print(f"[{self.model_name}] Calling LLM API...")
        print(f"Prompt length: {len(prompt)} characters")
        
        # Mock response (replace with actual API call)
        # response_text = api_call(prompt)
        # predictions = self._parse_response(response_text)
        
        # Placeholder: return random predictions
        import numpy as np
        predictions = np.random.randn(5) * 10 + 15000
        
        return predictions
    
    def _parse_response(self, response_text: str) -> np.ndarray:
        """Parse LLM response to extract predictions."""
        try:
            # Try to find JSON in response
            import re
            json_match = re.search(r'\{.*"predictions".*\}', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                predictions = np.array(data['predictions'])
                return predictions
            else:
                raise ValueError("No JSON found in response")
        
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response: {response_text[:200]}")
            raise


# Example usage
if __name__ == "__main__":
    # Load sample data
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = np.cumsum(np.random.randn(100) * 5) + 15000
    df = pd.DataFrame({'date': dates.strftime('%Y-%m-%d'), 'close': prices})
    
    # Sample news
    news = [
        {
            'title': 'Indonesia Central Bank Holds Rates Steady',
            'published_at': '2024-04-10 10:00:00',
            'publisher': 'Reuters',
            'summary': 'Bank Indonesia maintained its benchmark rate at 6.0%...'
        },
        {
            'title': 'Rupiah Strengthens on Export Growth',
            'published_at': '2024-04-08 14:30:00',
            'publisher': 'Bloomberg',
            'summary': 'Indonesian rupiah gained 0.5% as export data exceeded expectations...'
        }
    ]
    
    # Build prompts
    builder = PromptBuilder(currency_pair="USDIDR", horizon=5)
    
    print("="*60)
    print("ZERO-SHOT PROMPT")
    print("="*60)
    prompt_zero = builder.build_zero_shot(df, lookback_days=30)
    print(prompt_zero)
    
    print("\n" + "="*60)
    print("FEW-SHOT PROMPT")
    print("="*60)
    prompt_few = builder.build_few_shot(df, lookback_days=30, n_examples=2)
    print(prompt_few[:1000] + "...[truncated]")
    
    print("\n" + "="*60)
    print("NEWS-AUGMENTED PROMPT")
    print("="*60)
    prompt_news = builder.build_news_augmented(df, news, lookback_days=30)
    print(prompt_news[:1000] + "...[truncated]")
    
    print("\n" + "="*60)
    print("FULL MODEL (FEW-SHOT + NEWS)")
    print("="*60)
    prompt_full = builder.build_few_shot_with_news(df, news, lookback_days=30, n_examples=2)
    print(prompt_full[:1000] + "...[truncated]")