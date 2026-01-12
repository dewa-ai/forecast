# from __future__ import annotations
# from typing import List

# def build_prompt_numeric_only(
#     context_dates: List[str],
#     context_values: List[float],
#     horizon: int,
#     series_name: str = "USD/IDR",
# ) -> str:
#     lines = []
#     for i, (d, v) in enumerate(zip(context_dates, context_values), start=1):
#         lines.append(f"Day {i} ({d}): {v:.4f}")

#     return f"""You are a careful financial analyst.

# Given the following {series_name} daily exchange rates over the past {len(context_values)} days:
# {chr(10).join(lines)}

# Task:
# Forecast the next {horizon} days.

# Rules:
# - Output ONLY valid JSON.
# - Use keys "Day+1", "Day+2", ..., "Day+{horizon}".
# - Values must be numbers (no strings, no commas).
# - Do not add any commentary outside JSON.

# Example:
# {{"Day+1": 123.45, "Day+2": 124.00}}

# Now produce the forecast JSON:
# """.strip()

# def build_prompt_with_context(
#     context_dates: List[str],
#     context_values: List[float],
#     horizon: int,
#     macro_context: str,
#     series_name: str = "USD/IDR",
# ) -> str:
#     base = build_prompt_numeric_only(context_dates, context_values, horizon, series_name)
#     return f"""{base}

# Additional macroeconomic context (may be relevant, but be cautious):
# {macro_context}

# Re-check consistency with the numerical trend. Output ONLY JSON.
# """.strip()



#-------------------------------------------------------------------------------
# prompts.py - revised
#-------------------------------------------------------------------------------

# import numpy as np

# FEWSHOT_EXAMPLE = """
# Example (few-shot):
# Context (past rates): [100.0, 100.5, 100.2, 100.8]
# Horizon: 1 day
# Answer: 101.1
# """

# def format_series(series: np.ndarray) -> str:
#     # compact
#     vals = [f"{v:.6f}" for v in series.tolist()]
#     return "[" + ", ".join(vals) + "]"

# def build_prompt(pair: str, horizon: int, series: np.ndarray, news_text: str, few_shot: bool) -> str:
#     prompt = f"""You are a careful financial forecasting assistant.

# Task:
# Given the past exchange-rate time series for {pair}, forecast the exchange rate {horizon} day(s) ahead.

# Rules:
# - Output ONLY a single number (float).
# - No extra text.
# - Use the same scale as the input series.

# Context (past rates):
# {format_series(series)}
# Horizon: {horizon}
# """
#     if news_text:
#         prompt += f"\nRelevant macroeconomic news (lookback window):\n{news_text}\n"

#     if few_shot:
#         prompt += "\n" + FEWSHOT_EXAMPLE.strip() + "\n"

#     prompt += "\nAnswer:"
#     return prompt



#-------------------------------------------------------------------------------
# prompts.py - final
#-------------------------------------------------------------------------------

def build_prompt(pair, horizon, series, news_text, few_shot):
    prompt = f"""
You are a financial forecasting assistant.

Task:
Forecast the exchange rate for {pair} {horizon} day(s) ahead.

Rules:
- Output ONE number only.
- No explanation.

Past rates:
{series.tolist()}
"""

    if news_text:
        prompt += f"\nRelevant macroeconomic news:\n{news_text}\n"

    if few_shot:
        prompt += """
Example:
Past rates: [100, 101, 102]
Horizon: 1
Answer: 103
"""

    prompt += "\nAnswer:"
    return prompt
