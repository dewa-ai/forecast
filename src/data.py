from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import numpy as np

@dataclass
class WindowSample:
    idx: int
    context_dates: List[str]
    context_values: List[float]
    target_dates: List[str]
    target_values: List[float]

def load_series(csv_path: str, date_col: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df[[date_col, value_col]].dropna()
    df[value_col] = df[value_col].astype(float)
    return df

def make_windows(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    window_size: int,
    horizon: int,
    step: int = 1,
    start_ratio: float = 0.7,
) -> Tuple[List[WindowSample], List[WindowSample]]:
    dates = df[date_col].dt.strftime("%Y-%m-%d").tolist()
    values = df[value_col].astype(float).tolist()
    n = len(df)

    samples: List[WindowSample] = []
    idx = 0
    for start in range(0, n - window_size - horizon + 1, step):
        c_dates = dates[start : start + window_size]
        c_vals = values[start : start + window_size]
        t_dates = dates[start + window_size : start + window_size + horizon]
        t_vals = values[start + window_size : start + window_size + horizon]
        samples.append(WindowSample(idx=idx, context_dates=c_dates, context_values=c_vals,
                                   target_dates=t_dates, target_values=t_vals))
        idx += 1

    split = int(len(samples) * start_ratio)
    train = samples[:split]
    test = samples[split:]
    return train, test
