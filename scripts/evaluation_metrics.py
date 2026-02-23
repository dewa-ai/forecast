#!/usr/bin/env python3
"""
Includes: RMSE, MAE, MAPE, Directional Accuracy (DA), and Diebold-Mariano (DM) test.

Example:
    from evaluation_metrics import calculate_all_metrics, diebold_mariano_test
    
    metrics = calculate_all_metrics(y_true, y_pred)
    dm_stat, p_value = diebold_mariano_test(y_true, y_pred_model1, y_pred_model2)
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Dict, Tuple


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
    
    Args:
        y_true: Actual values (must have at least 2 values)
        y_pred: Predicted values
    
    Returns:
        Percentage of correct directional predictions
    """
    if len(y_true) < 2:
        return np.nan
    
    # Calculate direction: positive if price goes up, negative if down
    actual_direction = np.sign(np.diff(y_true))
    predicted_direction = np.sign(np.diff(y_pred))
    
    # Count correct predictions (ignore cases where direction is 0)
    mask = actual_direction != 0
    if not np.any(mask):
        return np.nan
    
    correct = np.sum(actual_direction[mask] == predicted_direction[mask])
    total = np.sum(mask)
    
    return (correct / total) * 100


def diebold_mariano_test(
    y_true: np.ndarray, 
    y_pred_1: np.ndarray, 
    y_pred_2: np.ndarray,
    h: int = 1,
    power: int = 2
) -> Tuple[float, float]:
    """
    Diebold-Mariano (DM) test for comparing forecast accuracy.
    
    Tests the null hypothesis that two forecasts have equal accuracy.
    
    Args:
        y_true: Actual values
        y_pred_1: Predictions from model 1
        y_pred_2: Predictions from model 2
        h: Forecast horizon (default: 1)
        power: Power for loss differential (1=MAE, 2=MSE)
    
    Returns:
        dm_stat: DM test statistic
        p_value: Two-tailed p-value
        
    Interpretation:
        - dm_stat > 0: Model 2 is better than Model 1
        - dm_stat < 0: Model 1 is better than Model 2
        - p_value < 0.05: Difference is statistically significant at 5% level
    """
    # Calculate prediction errors
    e1 = y_true - y_pred_1
    e2 = y_true - y_pred_2
    
    # Calculate loss differential
    d = np.abs(e1) ** power - np.abs(e2) ** power
    
    # Mean of loss differential
    d_mean = np.mean(d)
    
    # Variance of loss differential (with autocorrelation correction)
    n = len(d)
    gamma_0 = np.var(d, ddof=1)
    
    # Harvey-Leybourne-Newbold correction for small samples
    dm_stat = d_mean / np.sqrt(gamma_0 / n)
    
    # Small sample correction
    dm_stat = dm_stat * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    
    # Calculate p-value (two-tailed test)
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))
    
    return dm_stat, p_value


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


def compare_models_dm(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    baseline_model: str = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models using Diebold-Mariano test.
    
    Args:
        y_true: Actual values
        predictions: Dict of {model_name: predictions}
        baseline_model: Name of baseline model to compare against (default: first model)
    
    Returns:
        Dictionary of pairwise DM test results
    """
    if baseline_model is None:
        baseline_model = list(predictions.keys())[0]
    
    if baseline_model not in predictions:
        raise ValueError(f"Baseline model '{baseline_model}' not found in predictions")
    
    baseline_pred = predictions[baseline_model]
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Diebold-Mariano Tests (vs {baseline_model})")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'DM Stat':<12} {'p-value':<12} {'Significance':<15}")
    print(f"{'-'*60}")
    
    for model_name, model_pred in predictions.items():
        if model_name == baseline_model:
            continue
        
        dm_stat, p_value = diebold_mariano_test(y_true, baseline_pred, model_pred)
        
        # Determine significance
        if p_value < 0.01:
            sig = "***"
        elif p_value < 0.05:
            sig = "**"
        elif p_value < 0.10:
            sig = "*"
        else:
            sig = "ns"
        
        # Determine better model
        better = model_name if dm_stat > 0 else baseline_model
        
        results[model_name] = {
            'dm_stat': dm_stat,
            'p_value': p_value,
            'better_model': better,
            'significant': p_value < 0.05
        }
        
        print(f"{model_name:<20} {dm_stat:>11.4f} {p_value:>11.4f} {sig:<15}")
    
    print(f"{'='*60}")
    print("Significance: *** p<0.01, ** p<0.05, * p<0.10, ns = not significant")
    print(f"Positive DM stat = {baseline_model} is worse (other model better)")
    print(f"Negative DM stat = {baseline_model} is better")
    print(f"{'='*60}\n")
    
    return results


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
    
    # Diebold-Mariano test
    dm_stat, p_value = diebold_mariano_test(y_true, y_pred1, y_pred2)
    print(f"Diebold-Mariano Test:")
    print(f"  DM Statistic: {dm_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Interpretation: Model 2 is {'significantly ' if p_value < 0.05 else ''}better" 
          if dm_stat > 0 else f"  Interpretation: Model 1 is {'significantly ' if p_value < 0.05 else ''}better")
    
    # Compare multiple models
    predictions = {
        'Model1': y_pred1,
        'Model2': y_pred2,
    }
    compare_models_dm(y_true, predictions, baseline_model='Model1')