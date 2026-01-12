# from __future__ import annotations
# import os, json
# from typing import Dict, Any, List
# import yaml
# import numpy as np
# from tqdm import tqdm

# from .utils import set_seed, ensure_dir, extract_json_obj, safe_float
# from .data import load_series, make_windows
# from .prompts import build_prompt_numeric_only
# from .llm import LLMConfig, LLMForecaster
# from .eval import summarize_metrics

# def json_to_forecast(obj: Dict[str, Any], horizon: int) -> List[float] | None:
#     vals = []
#     for k in [f"Day+{i}" for i in range(1, horizon + 1)]:
#         if k not in obj:
#             return None
#         v = safe_float(obj[k])
#         if v is None or not np.isfinite(v):
#             return None
#         vals.append(float(v))
#     return vals

# def run_one_model(cfg: Dict[str, Any], model_entry: Dict[str, str]) -> Dict[str, Any]:
#     data_cfg = cfg["data"]
#     task_cfg = cfg["task"]
#     inf_cfg = cfg["inference"]
#     out_cfg = cfg["output"]

#     df = load_series(data_cfg["csv_path"], data_cfg["date_col"], data_cfg["value_col"])
#     _, test_samples = make_windows(
#         df,
#         date_col=data_cfg["date_col"],
#         value_col=data_cfg["value_col"],
#         window_size=task_cfg["window_size"],
#         horizon=task_cfg["horizon"],
#         step=task_cfg["step"],
#         start_ratio=task_cfg["start_ratio"],
#     )

#     ensure_dir(out_cfg["out_dir"])
#     ensure_dir(out_cfg["cache_dir"])

#     model_name = model_entry["name"]
#     hf_id = model_entry["hf_id"]

#     forecaster = LLMForecaster(
#         LLMConfig(
#             hf_id=hf_id,
#             use_4bit=inf_cfg["use_4bit"],
#             max_new_tokens=inf_cfg["max_new_tokens"],
#             temperature=inf_cfg["temperature"],
#             top_p=inf_cfg["top_p"],
#             repetition_penalty=inf_cfg["repetition_penalty"],
#         )
#     )

#     rows = []
#     per_sample = []

#     for s in tqdm(test_samples, desc=f"Running {model_name}"):
#         cache_key = f"{model_name}_idx{s.idx}_w{task_cfg['window_size']}_h{task_cfg['horizon']}.json"
#         cache_path = os.path.join(out_cfg["cache_dir"], cache_key)

#         if os.path.exists(cache_path):
#             with open(cache_path, "r") as f:
#                 cached = json.load(f)
#             pred_vals = cached.get("pred_vals")
#         else:
#             prompt = build_prompt_numeric_only(
#                 s.context_dates, s.context_values, task_cfg["horizon"], series_name="USD/IDR"
#             )
#             resp = forecaster.generate(prompt)
#             obj = extract_json_obj(resp)
#             pred_vals = json_to_forecast(obj, task_cfg["horizon"]) if obj else None

#             with open(cache_path, "w") as f:
#                 json.dump(
#                     {
#                         "idx": s.idx,
#                         "response": resp,
#                         "parsed": obj,
#                         "pred_vals": pred_vals,
#                         "target_vals": s.target_values,
#                         "target_dates": s.target_dates,
#                     },
#                     f,
#                     indent=2,
#                 )

#         # If parsing fails, skip sample (or you can fallback to naive forecast)
#         if pred_vals is None:
#             continue

#         y_true = np.array(s.target_values, dtype=float)
#         y_pred = np.array(pred_vals, dtype=float)

#         per_sample.append({
#             "idx": s.idx,
#             "target_dates": s.target_dates,
#             "y_true": y_true.tolist(),
#             "y_pred": y_pred.tolist(),
#         })

#         rows.append({
#             "y_true": y_true,
#             "y_pred": y_pred,
#             "last_context": float(s.context_values[-1]),
#         })

#     metrics = summarize_metrics(rows) if rows else {"MAE": None, "RMSE": None, "MAPE": None, "DA": None}

#     out_path = os.path.join(out_cfg["out_dir"], f"results_{model_name}.json")
#     with open(out_path, "w") as f:
#         json.dump({"model": model_name, "hf_id": hf_id, "metrics": metrics, "n_used": len(rows), "samples": per_sample}, f, indent=2)

#     return {"model": model_name, "metrics": metrics, "n_used": len(rows), "out_path": out_path}

# def main():
#     with open("config.yaml", "r") as f:
#         cfg = yaml.safe_load(f)

#     set_seed(cfg["inference"]["seed"])

#     all_results = []
#     for m in cfg["models"]:
#         res = run_one_model(cfg, m)
#         all_results.append(res)

#     # Save summary
#     ensure_dir(cfg["output"]["out_dir"])
#     with open(os.path.join(cfg["output"]["out_dir"], "summary.json"), "w") as f:
#         json.dump(all_results, f, indent=2)

#     print("\n=== Summary ===")
#     for r in all_results:
#         print(r["model"], r["metrics"], "n_used=", r["n_used"], "->", r["out_path"])

# if __name__ == "__main__":
#     main()


#-------------------------------------------------------------------------------
# run_llm_forecast.py - revised
#-------------------------------------------------------------------------------

# import os
# import yaml
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from itertools import product

# from src.data import build_samples, load_news_df, select_news_for_date
# from src.prompts import build_prompt
# from src.llm import HFLLM
# from src.eval import rmse, mape, directional_accuracy, dm_test_squared_error
# from src.utils import ensure_dir, save_json, stable_hash

# def pair_for_currency(currency: str) -> str:
#     # Forecast USD/XXX for non-USD currencies.
#     # If you keep "USD" in config, it maps to USDIDR by default (not meaningful).
#     if currency == "USD":
#         return "USDIDR"
#     return f"USD{currency}"

# def run_one_setting(llm, cfg, currency: str, horizon: int, use_news: bool, few_shot: bool):
#     pair = pair_for_currency(currency)

#     X, Y, D = build_samples(
#         pair=pair,
#         horizon=horizon,
#         context_window=cfg["experiment"]["context_window"],
#         eval_size=cfg["experiment"]["eval_size"],
#     )

#     news_df = load_news_df(pair) if use_news else None

#     preds = []
#     prev_vals = []  # last observed value before target (for DA)

#     cache_dir = os.path.join(cfg["output_dir"], "cache")
#     ensure_dir(cache_dir)

#     for i in tqdm(range(len(Y)), desc=f"{pair} H={horizon} news={use_news} few={few_shot}", leave=False):
#         series = X[i]
#         y_true = float(Y[i])
#         anchor_date = pd.Timestamp(D[i])

#         news_text = ""
#         if use_news:
#             news_text = select_news_for_date(
#                 news_df=news_df,
#                 date=anchor_date,
#                 lookback_days=int(cfg["news"]["lookback_days"]),
#                 top_k=int(cfg["news"]["top_k"]),
#             )

#         prompt = build_prompt(pair=pair, horizon=horizon, series=series, news_text=news_text, few_shot=few_shot)

#         # cache by prompt hash (saves time)
#         key = stable_hash(prompt)
#         cache_path = os.path.join(cache_dir, f"{key}.txt")
#         if os.path.exists(cache_path):
#             y_pred = float(open(cache_path, "r", encoding="utf-8").read().strip())
#         else:
#             y_pred = llm.generate_number(
#                 prompt,
#                 temperature=float(cfg["llm"]["temperature"]),
#                 max_new_tokens=int(cfg["llm"]["max_new_tokens"]),
#             )
#             with open(cache_path, "w", encoding="utf-8") as f:
#                 f.write(str(y_pred))

#         preds.append(y_pred)
#         prev_vals.append(float(series[-1]))

#     preds = np.asarray(preds, dtype=float)
#     Y = np.asarray(Y, dtype=float)
#     prev_vals = np.asarray(prev_vals, dtype=float)

#     metrics = {
#         "RMSE": rmse(Y, preds),
#         "MAPE": mape(Y, preds),
#         "DA": directional_accuracy(Y, preds, prev_vals),
#     }

#     pred_df = pd.DataFrame({
#         "date": pd.to_datetime(D),
#         "y_true": Y,
#         "y_pred": preds,
#         "prev_y": prev_vals,
#     })

#     return metrics, pred_df

# def main():
#     cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

#     # output dirs
#     out_dir = cfg.get("output_dir", "outputs")
#     cfg["output_dir"] = out_dir
#     ensure_dir(out_dir)
#     ensure_dir(os.path.join(out_dir, "predictions"))

#     # init LLM
#     if cfg["llm"]["backend"] != "hf":
#         raise ValueError("This code supports only backend=hf.")
#     llm = HFLLM(cfg["llm"]["model_name"])

#     currencies = cfg["experiment"]["currencies"]
#     horizons = cfg["experiment"]["horizons"]
#     news_opts = cfg["ablation"]["news"]
#     few_opts = cfg["ablation"]["few_shot"]

#     all_rows = []
#     # Store predictions to compute DM: baseline vs full per (currency,horizon)
#     pred_store = {}

#     for currency, horizon in product(currencies, horizons):
#         for use_news, few_shot in product(news_opts, few_opts):
#             metrics, pred_df = run_one_setting(llm, cfg, currency, int(horizon), bool(use_news), bool(few_shot))

#             tag = f"{currency}_H{horizon}_news{int(use_news)}_few{int(few_shot)}"
#             pred_path = os.path.join(out_dir, "predictions", f"{tag}.csv")
#             pred_df.to_csv(pred_path, index=False)

#             row = {
#                 "currency": currency,
#                 "pair": pair_for_currency(currency),
#                 "horizon": int(horizon),
#                 "news": bool(use_news),
#                 "few_shot": bool(few_shot),
#                 **metrics,
#                 "pred_file": pred_path,
#             }
#             all_rows.append(row)

#             pred_store[(currency, int(horizon), bool(use_news), bool(few_shot))] = pred_df

#         # DM test: baseline (zero,no) vs full (few,yes)
#         base = pred_store.get((currency, int(horizon), False, False))
#         full = pred_store.get((currency, int(horizon), True, True))
#         if base is not None and full is not None:
#             y = base["y_true"].to_numpy()
#             yhat_a = base["y_pred"].to_numpy()
#             yhat_b = full["y_pred"].to_numpy()
#             stat, p = dm_test_squared_error(y, yhat_a, yhat_b)
#             all_rows.append({
#                 "currency": currency,
#                 "pair": pair_for_currency(currency),
#                 "horizon": int(horizon),
#                 "news": None,
#                 "few_shot": None,
#                 "RMSE": None,
#                 "MAPE": None,
#                 "DA": None,
#                 "DM_stat_baseline_vs_full": stat,
#                 "DM_p_baseline_vs_full": p,
#             })

#     save_json(all_rows, os.path.join(out_dir, "summary.json"))
#     print(f"Saved: {os.path.join(out_dir, 'summary.json')}")

# if __name__ == "__main__":
#     main()


#-------------------------------------------------------------------------------
# run_llm_forecast.py - final
#-------------------------------------------------------------------------------

import yaml
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm

from src.data import build_samples, load_news_df, select_news
from src.prompts import build_prompt
from src.llm import LLM
from src.eval import rmse, mape, directional_accuracy, dm_test

def main():
    cfg = yaml.safe_load(open("config.yaml"))

    pairs = cfg["experiment"]["pairs"]
    horizons = cfg["experiment"]["horizons"]

    llm = LLM(cfg["llm"]["model_name"])

    results = {}
    rows = []

    for pair, H in product(pairs, horizons):
        for use_news, few in product(cfg["ablation"]["news"], cfg["ablation"]["few_shot"]):
            X, Y, PREV, D = build_samples(
                pair,
                H,
                cfg["experiment"]["context_window"],
                cfg["experiment"]["eval_size"]
            )

            news_df = load_news_df(pair) if use_news else None
            preds = []

            for i in tqdm(range(len(Y)), desc=f"{pair} H={H} news={use_news} few={few}"):
                news_txt = ""
                if use_news:
                    news_txt = select_news(
                        news_df,
                        pd.Timestamp(D[i]),
                        cfg["news"]["lookback_days"],
                        cfg["news"]["top_k"]
                    )

                prompt = build_prompt(pair, H, X[i], news_txt, few)
                pred = llm.predict(prompt,
                                   cfg["llm"]["temperature"],
                                   cfg["llm"]["max_new_tokens"])
                preds.append(pred)

            preds = np.array(preds)
            key = (pair, H, use_news, few)
            results[key] = preds

            rows.append({
                "pair": pair,
                "horizon": H,
                "news": use_news,
                "few_shot": few,
                "RMSE": rmse(Y, preds),
                "MAPE": mape(Y, preds),
                "DA": directional_accuracy(Y, preds, PREV)
            })

        # DM: baseline vs full
        base = results[(pair, H, False, False)]
        full = results[(pair, H, True, True)]
        stat, p = dm_test(Y, base, full)

        rows.append({
            "pair": pair,
            "horizon": H,
            "DM_baseline_vs_full": stat,
            "DM_p": p
        })

    pd.DataFrame(rows).to_csv("outputs/summary.csv", index=False)
    print("DONE → outputs/summary.csv")

if __name__ == "__main__":
    main()
