# from __future__ import annotations
# from typing import List, Dict
# import numpy as np

# def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     return float(np.mean(np.abs(y_true - y_pred)))

# def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     eps = 1e-9
#     return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)

# def directional_accuracy(last_context: float, y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     # direction compared to last known value (for t+1) then rolling
#     true_dir = np.sign(np.diff(np.concatenate([[last_context], y_true])))
#     pred_dir = np.sign(np.diff(np.concatenate([[last_context], y_pred])))
#     return float(np.mean(true_dir == pred_dir))

# def summarize_metrics(all_rows: List[Dict]) -> Dict[str, float]:
#     y_true = np.concatenate([r["y_true"] for r in all_rows])
#     y_pred = np.concatenate([r["y_pred"] for r in all_rows])
#     last_vals = np.array([r["last_context"] for r in all_rows], dtype=float)

#     # directional accuracy per-sample then average
#     das = []
#     offset = 0
#     for r in all_rows:
#         n = len(r["y_true"])
#         das.append(directional_accuracy(r["last_context"], r["y_true"], r["y_pred"]))
#         offset += n

#     return {
#         "MAE": mae(y_true, y_pred),
#         "RMSE": rmse(y_true, y_pred),
#         "MAPE": mape(y_true, y_pred),
#         "DA": float(np.mean(das)),
#     }


#-------------------------------------------------------------------------------
# eval.py - revised
#-------------------------------------------------------------------------------

# import numpy as np
# from scipy.stats import ttest_1samp

# def rmse(y, yhat):
#     y = np.asarray(y, dtype=float)
#     yhat = np.asarray(yhat, dtype=float)
#     return float(np.sqrt(np.mean((y - yhat) ** 2)))

# def mape(y, yhat):
#     y = np.asarray(y, dtype=float)
#     yhat = np.asarray(yhat, dtype=float)
#     eps = 1e-12
#     return float(np.mean(np.abs((y - yhat) / (np.abs(y) + eps))) * 100.0)

# def directional_accuracy(y, yhat, prev_y):
#     """
#     Direction based on change from prev_y -> y vs prev_y -> yhat.
#     prev_y is the last observed value before the target.
#     """
#     y = np.asarray(y, dtype=float)
#     yhat = np.asarray(yhat, dtype=float)
#     prev_y = np.asarray(prev_y, dtype=float)
#     return float(np.mean(np.sign(y - prev_y) == np.sign(yhat - prev_y)))

# def dm_test_squared_error(y, yhat_a, yhat_b):
#     """
#     Diebold-Mariano test (simple) on loss differential using squared error.
#     Returns (statistic, p_value).
#     """
#     y = np.asarray(y, dtype=float)
#     a = np.asarray(yhat_a, dtype=float)
#     b = np.asarray(yhat_b, dtype=float)
#     e1 = (y - a) ** 2
#     e2 = (y - b) ** 2
#     d = e1 - e2
#     stat, p = ttest_1samp(d, 0.0)
#     return float(stat), float(p)


#-------------------------------------------------------------------------------
# eval.py - final
#-------------------------------------------------------------------------------

import numpy as np
from scipy.stats import ttest_1samp

def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def mape(y, yhat):
    return float(np.mean(np.abs((y - yhat) / y)) * 100)

def directional_accuracy(y, yhat, prev):
    return float(np.mean(np.sign(y - prev) == np.sign(yhat - prev)))

def dm_test(y, base, full):
    d = (y - base)**2 - (y - full)**2
    stat, p = ttest_1samp(d, 0)
    return float(stat), float(p)
