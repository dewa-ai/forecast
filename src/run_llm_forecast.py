from __future__ import annotations
import os, json
from typing import Dict, Any, List
import yaml
import numpy as np
from tqdm import tqdm

from .utils import set_seed, ensure_dir, extract_json_obj, safe_float
from .data import load_series, make_windows
from .prompts import build_prompt_numeric_only
from .llm import LLMConfig, LLMForecaster
from .eval import summarize_metrics

def json_to_forecast(obj: Dict[str, Any], horizon: int) -> List[float] | None:
    vals = []
    for k in [f"Day+{i}" for i in range(1, horizon + 1)]:
        if k not in obj:
            return None
        v = safe_float(obj[k])
        if v is None or not np.isfinite(v):
            return None
        vals.append(float(v))
    return vals

def run_one_model(cfg: Dict[str, Any], model_entry: Dict[str, str]) -> Dict[str, Any]:
    data_cfg = cfg["data"]
    task_cfg = cfg["task"]
    inf_cfg = cfg["inference"]
    out_cfg = cfg["output"]

    df = load_series(data_cfg["csv_path"], data_cfg["date_col"], data_cfg["value_col"])
    _, test_samples = make_windows(
        df,
        date_col=data_cfg["date_col"],
        value_col=data_cfg["value_col"],
        window_size=task_cfg["window_size"],
        horizon=task_cfg["horizon"],
        step=task_cfg["step"],
        start_ratio=task_cfg["start_ratio"],
    )

    ensure_dir(out_cfg["out_dir"])
    ensure_dir(out_cfg["cache_dir"])

    model_name = model_entry["name"]
    hf_id = model_entry["hf_id"]

    forecaster = LLMForecaster(
        LLMConfig(
            hf_id=hf_id,
            use_4bit=inf_cfg["use_4bit"],
            max_new_tokens=inf_cfg["max_new_tokens"],
            temperature=inf_cfg["temperature"],
            top_p=inf_cfg["top_p"],
            repetition_penalty=inf_cfg["repetition_penalty"],
        )
    )

    rows = []
    per_sample = []

    for s in tqdm(test_samples, desc=f"Running {model_name}"):
        cache_key = f"{model_name}_idx{s.idx}_w{task_cfg['window_size']}_h{task_cfg['horizon']}.json"
        cache_path = os.path.join(out_cfg["cache_dir"], cache_key)

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cached = json.load(f)
            pred_vals = cached.get("pred_vals")
        else:
            prompt = build_prompt_numeric_only(
                s.context_dates, s.context_values, task_cfg["horizon"], series_name="USD/IDR"
            )
            resp = forecaster.generate(prompt)
            obj = extract_json_obj(resp)
            pred_vals = json_to_forecast(obj, task_cfg["horizon"]) if obj else None

            with open(cache_path, "w") as f:
                json.dump(
                    {
                        "idx": s.idx,
                        "response": resp,
                        "parsed": obj,
                        "pred_vals": pred_vals,
                        "target_vals": s.target_values,
                        "target_dates": s.target_dates,
                    },
                    f,
                    indent=2,
                )

        # If parsing fails, skip sample (or you can fallback to naive forecast)
        if pred_vals is None:
            continue

        y_true = np.array(s.target_values, dtype=float)
        y_pred = np.array(pred_vals, dtype=float)

        per_sample.append({
            "idx": s.idx,
            "target_dates": s.target_dates,
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
        })

        rows.append({
            "y_true": y_true,
            "y_pred": y_pred,
            "last_context": float(s.context_values[-1]),
        })

    metrics = summarize_metrics(rows) if rows else {"MAE": None, "RMSE": None, "MAPE": None, "DA": None}

    out_path = os.path.join(out_cfg["out_dir"], f"results_{model_name}.json")
    with open(out_path, "w") as f:
        json.dump({"model": model_name, "hf_id": hf_id, "metrics": metrics, "n_used": len(rows), "samples": per_sample}, f, indent=2)

    return {"model": model_name, "metrics": metrics, "n_used": len(rows), "out_path": out_path}

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["inference"]["seed"])

    all_results = []
    for m in cfg["models"]:
        res = run_one_model(cfg, m)
        all_results.append(res)

    # Save summary
    ensure_dir(cfg["output"]["out_dir"])
    with open(os.path.join(cfg["output"]["out_dir"], "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Summary ===")
    for r in all_results:
        print(r["model"], r["metrics"], "n_used=", r["n_used"], "->", r["out_path"])

if __name__ == "__main__":
    main()
