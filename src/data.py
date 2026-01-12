# from __future__ import annotations
# from dataclasses import dataclass
# from typing import List, Tuple
# import pandas as pd
# import numpy as np

# @dataclass
# class WindowSample:
#     idx: int
#     context_dates: List[str]
#     context_values: List[float]
#     target_dates: List[str]
#     target_values: List[float]

# def load_series(csv_path: str, date_col: str, value_col: str) -> pd.DataFrame:
#     df = pd.read_csv(csv_path)
#     df[date_col] = pd.to_datetime(df[date_col])
#     df = df.sort_values(date_col).reset_index(drop=True)
#     df = df[[date_col, value_col]].dropna()
#     df[value_col] = df[value_col].astype(float)
#     return df

# def make_windows(
#     df: pd.DataFrame,
#     date_col: str,
#     value_col: str,
#     window_size: int,
#     horizon: int,
#     step: int = 1,
#     start_ratio: float = 0.7,
# ) -> Tuple[List[WindowSample], List[WindowSample]]:
#     dates = df[date_col].dt.strftime("%Y-%m-%d").tolist()
#     values = df[value_col].astype(float).tolist()
#     n = len(df)

#     samples: List[WindowSample] = []
#     idx = 0
#     for start in range(0, n - window_size - horizon + 1, step):
#         c_dates = dates[start : start + window_size]
#         c_vals = values[start : start + window_size]
#         t_dates = dates[start + window_size : start + window_size + horizon]
#         t_vals = values[start + window_size : start + window_size + horizon]
#         samples.append(WindowSample(idx=idx, context_dates=c_dates, context_values=c_vals,
#                                    target_dates=t_dates, target_values=t_vals))
#         idx += 1

#     split = int(len(samples) * start_ratio)
#     train = samples[:split]
#     test = samples[split:]
#     return train, test


#----------------------------------------------------------------
# src/data.py - revised
#----------------------------------------------------------------   

# import os
# import yaml
# import numpy as np
# import pandas as pd

# def currency_to_pair(currency: str) -> str:
#     # We define base USD vs other currencies.
#     if currency == "USD":
#         # For USD itself, we use USDIDR as a proxy is meaningless.
#         # In practice you should forecast pairs, not single currency.
#         # We'll map USD -> USDEUR? But better: skip "USD" in experiment.
#         return "USDIDR"
#     return f"USD{currency}"

# def load_fx_series(pair: str) -> pd.DataFrame:
#     path = os.path.join("data", "fx", f"{pair}.csv")
#     df = pd.read_csv(path)
#     df["date"] = pd.to_datetime(df["date"])
#     df = df.sort_values("date")
#     return df

# def load_news_df(pair: str) -> pd.DataFrame:
#     path = os.path.join("data", "news", f"{pair}.csv")
#     if not os.path.exists(path):
#         return pd.DataFrame(columns=["datetime", "title", "snippet", "url"])
#     df = pd.read_csv(path)
#     if "datetime" in df.columns:
#         df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
#     else:
#         df["datetime"] = pd.NaT
#     return df

# def select_news_for_date(news_df: pd.DataFrame, date: pd.Timestamp, lookback_days: int, top_k: int) -> str:
#     if news_df.empty:
#         return ""

#     start = date - pd.Timedelta(days=lookback_days)
#     end = date + pd.Timedelta(days=1)

#     sub = news_df[(news_df["datetime"].notna()) & (news_df["datetime"] >= start) & (news_df["datetime"] < end)]
#     if sub.empty:
#         # fallback: if no datetimes, just take latest K
#         sub = news_df.tail(top_k)

#     # Simple relevance scoring: keyword hits (macro + currency codes)
#     keywords = ["inflation", "interest", "rate", "central bank", "fed", "ecb", "rba", "boj", "gdp", "cpi", "jobs", "trade", "risk"]
#     def score_row(r):
#         text = f"{r.get('title','')} {r.get('snippet','')}".lower()
#         return sum(1 for kw in keywords if kw in text)

#     sub = sub.copy()
#     sub["score"] = sub.apply(score_row, axis=1)
#     sub = sub.sort_values(["score", "datetime"], ascending=[False, False]).head(top_k)

#     lines = []
#     for _, r in sub.iterrows():
#         dt = r.get("datetime")
#         dt_s = "" if pd.isna(dt) else pd.Timestamp(dt).strftime("%Y-%m-%d")
#         title = str(r.get("title", "")).strip()
#         snip = str(r.get("snippet", "")).strip()
#         if snip and snip != "nan":
#             lines.append(f"- [{dt_s}] {title} — {snip}")
#         else:
#             lines.append(f"- [{dt_s}] {title}")

#     return "\n".join(lines)

# def build_samples(pair: str, horizon: int, context_window: int, eval_size: int):
#     df = load_fx_series(pair)
#     y = df["price"].to_numpy(dtype=float)
#     dates = df["date"].to_numpy()

#     # build supervised samples: use t-context_window..t-1 to predict t+h
#     X, Y, D = [], [], []
#     for t in range(context_window, len(y) - horizon):
#         series = y[t - context_window:t]
#         target = y[t + horizon]
#         target_date = pd.Timestamp(dates[t])  # anchor date for context end
#         X.append(series)
#         Y.append(target)
#         D.append(target_date)

#     X = np.asarray(X)
#     Y = np.asarray(Y)
#     D = np.asarray(D, dtype="datetime64[ns]")

#     # use last eval_size samples for evaluation
#     if eval_size is not None and eval_size > 0 and len(Y) > eval_size:
#         X = X[-eval_size:]
#         Y = Y[-eval_size:]
#         D = D[-eval_size:]

#     return X, Y, D




#----------------------------------------------------------------
# src/data.py - final
#----------------------------------------------------------------

import os
import numpy as np
import pandas as pd

def load_fx_series(pair: str) -> pd.DataFrame:
    """
    Expect CSV:
    date,price
    """
    path = os.path.join("data", "fx", f"{pair}.csv")
    df = pd.read_csv(path)

    if "price" not in df.columns:
        raise ValueError(f"{pair}.csv MUST contain column 'price'")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "price"]]

def load_news_df(pair: str) -> pd.DataFrame:
    path = os.path.join("data", "news", f"{pair}.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["datetime", "title", "snippet"])

    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        df["datetime"] = pd.NaT
    return df

def select_news(news_df, anchor_date, lookback_days, top_k):
    if news_df.empty:
        return ""

    start = anchor_date - pd.Timedelta(days=lookback_days)
    end = anchor_date + pd.Timedelta(days=1)

    sub = news_df[
        (news_df["datetime"].notna()) &
        (news_df["datetime"] >= start) &
        (news_df["datetime"] < end)
    ]

    if sub.empty:
        sub = news_df.tail(top_k)

    sub = sub.tail(top_k)

    lines = []
    for _, r in sub.iterrows():
        lines.append(f"- {r['title']} {r.get('snippet','')}".strip())
    return "\n".join(lines)

def build_samples(pair, horizon, context_window, eval_size):
    df = load_fx_series(pair)
    prices = df["price"].to_numpy()
    dates = df["date"].to_numpy()

    X, Y, D, PREV = [], [], [], []

    for t in range(context_window, len(prices) - horizon):
        X.append(prices[t-context_window:t])
        Y.append(prices[t + horizon])
        PREV.append(prices[t-1])
        D.append(dates[t])

    X = np.array(X)
    Y = np.array(Y)
    PREV = np.array(PREV)
    D = np.array(D, dtype="datetime64[ns]")

    if eval_size and len(Y) > eval_size:
        X = X[-eval_size:]
        Y = Y[-eval_size:]
        PREV = PREV[-eval_size:]
        D = D[-eval_size:]

    return X, Y, PREV, D

