#!/usr/bin/env python3
"""
Evaluation metrics for FX forecasting ablation study.
Includes: MAE, RMSE, MAPE, and Directional Accuracy (DA).

DA is computed against the last observed actual value for each forecast window:
    sign(y_t - y_{t-1}) == sign(yhat_t - y_{t-1})
For multi-step horizons, y_{t-1} is the previous actual value inside the same
forecast window, with the first step anchored to the last observed context value.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error. Lower is better."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error. Lower is better."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, expressed as a percentage. Lower is better."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    last_actual: Optional[float] = None,
) -> float:
    """
    Directional Accuracy (DA), returned as a percentage in [0, 100].

    Preferred use for forecasting windows:
        directional_accuracy(y_true_window, y_pred_window, last_actual=context_last_price)

    This implements:
        sign(y_t - y_{t-1}) == sign(yhat_t - y_{t-1})

    If last_actual is not provided, the function falls back to the older behavior
    based on np.diff(y_true) and np.diff(y_pred). This fallback is kept only for
    backward compatibility; thesis experiments should pass last_actual.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        return float("nan")

    # Backward-compatible fallback. Do not use this for final thesis DA.
    if last_actual is None:
        if len(y_true) < 2:
            return float("nan")
        actual_direction = np.sign(np.diff(y_true))
        predicted_direction = np.sign(np.diff(y_pred))
    else:
        prev_actual = np.concatenate([[float(last_actual)], y_true[:-1]])
        actual_direction = np.sign(y_true - prev_actual)
        predicted_direction = np.sign(y_pred - prev_actual)

    mask = actual_direction != 0
    if not np.any(mask):
        return float("nan")

    correct = np.sum(actual_direction[mask] == predicted_direction[mask])
    total = np.sum(mask)
    return float((correct / total) * 100)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    last_actual: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate RMSE, MAE, MAPE, and DA.

    For a single forecast window, pass last_actual to compute DA correctly.
    For multiple rolling windows, compute RMSE/MAE/MAPE on concatenated values
    and compute DA per window using directional_accuracy(..., last_actual=...).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")

    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "da": directional_accuracy(y_true, y_pred, last_actual=last_actual),
    }


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"Evaluation Metrics: {model_name}")
    print(f"{'='*50}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"DA:   {metrics['da']:.2f}%")
    print(f"{'='*50}\n")