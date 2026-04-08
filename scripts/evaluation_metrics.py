#!/usr/bin/env python3
"""
Evaluation metrics for FX forecasting ablation study.
Includes: MAE, RMSE, MAPE, Directional Accuracy (DA).

Example:
    from evaluation_metrics import calculate_all_metrics

    metrics = calculate_all_metrics(y_true, y_pred)
"""

from __future__ import annotations

import numpy as np
from typing import Dict


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    
    Measures average magnitude of errors, giving more weight to large errors.
    Lower is better.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.
    
    Measures average magnitude of errors, robust to outliers.
    Lower is better.
    """
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    
    Scale-independent metric, expressed as percentage.
    Lower is better.
    
    Note: Returns infinity if any y_true is zero.
    """
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return np.inf
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy (DA).

    Measures percentage of correct directional predictions (up/down).
    Critical for trading applications.
    Higher is better. Range: [0, 100]

    Fix 6: If predictions are flat (all same value), DA is explicitly 0.0
    rather than NaN — making it clear the model failed to predict direction.
    """
    if len(y_true) < 2:
        return np.nan

    # Fix 6: flat predictions → DA = 0 (not NaN)
    if np.std(y_pred) == 0:
        return 0.0

    actual_direction    = np.sign(np.diff(y_true))
    predicted_direction = np.sign(np.diff(y_pred))

    mask = actual_direction != 0
    if not np.any(mask):
        return np.nan

    correct = np.sum(actual_direction[mask] == predicted_direction[mask])
    total   = np.sum(mask)

    return (correct / total) * 100



def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics at once.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Dictionary with all metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    metrics = {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'da': directional_accuracy(y_true, y_pred),
    }
    
    return metrics


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



# Example usage
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    n = 100
    y_true = np.cumsum(np.random.randn(n)) + 100
    y_pred1 = y_true + np.random.randn(n) * 0.5
    y_pred2 = y_true + np.random.randn(n) * 0.8
    
    # Calculate metrics for both models
    print("Example: Comparing two models")

    metrics1 = calculate_all_metrics(y_true, y_pred1)
    print_metrics(metrics1, "Model 1 (Better)")

    metrics2 = calculate_all_metrics(y_true, y_pred2)
    print_metrics(metrics2, "Model 2 (Worse)")