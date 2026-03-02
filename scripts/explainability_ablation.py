#!/usr/bin/env python3
"""
explainability_ablation.py
Ablation 4 & 5 for LLM FX Forecasting.

Ablation 4 — Explainability
  Q4: Can LLMs provide human-interpretable explanations for FX predictions,
      and do these explanations demonstrate genuine understanding of market dynamics?
  Metrics:
    - news_grounding_rate  : % explanations that cite at least one actual news headline
    - avg_explanation_len  : mean word count of explanations
    - factor_types_cited   : which factor categories appear (monetary, trade, sentiment…)
    - coherence_score      : heuristic sentence-structure quality (0-1)

Ablation 5 — Explanation-Prediction Correlation
  Q5: Does explanation quality correlate with prediction accuracy?
  Metrics:
    - corr(explanation_length, abs_pred_error)
    - corr(news_reference_count, directional_accuracy)
    - corr(confidence_score, abs_pred_error)

Usage:
    from explainability_ablation import ExplainabilityAblation

    abl = ExplainabilityAblation(llm_forecaster, output_dir="results")
    results = abl.run(data, news_articles, currency="USDIDR", horizon=5)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Prompt builder — Explanation prompts
# ---------------------------------------------------------------------------

class ExplainabilityPromptBuilder:
    """
    Builds prompts that ask the LLM to:
      1. Give a prediction (JSON array)
      2. Give a structured explanation immediately after
    """

    # Factor categories to look for in explanations
    FACTOR_CATEGORIES = {
        "monetary_policy":  ["interest rate", "central bank", "rate hike", "rate cut",
                              "monetary", "fed", "boe", "ecb", "bank indonesia", "bi rate"],
        "trade_balance":    ["trade", "export", "import", "current account", "surplus", "deficit"],
        "inflation":        ["inflation", "cpi", "consumer price", "ppi", "deflation"],
        "risk_sentiment":   ["risk", "sentiment", "safe haven", "uncertainty", "confidence",
                              "geopolit", "war", "tension"],
        "economic_growth":  ["gdp", "growth", "recession", "economic", "employment",
                              "unemployment", "jobs"],
        "market_technical": ["trend", "momentum", "support", "resistance", "moving average",
                              "volatility", "overbought", "oversold"],
        "commodity":        ["oil", "gold", "commodity", "crude", "energy", "metal"],
    }

    def __init__(self, currency_pair: str, horizon: int = 5):
        self.currency_pair  = currency_pair
        self.horizon        = horizon
        self.base_currency  = currency_pair[:3]
        self.quote_currency = currency_pair[3:]

    def build_explain_prompt(
        self,
        historical_data: pd.DataFrame,
        news_articles: List[Dict],
        lookback_days: int = 30,
        max_news: int = 10,
    ) -> str:
        """
        Prompt that requests BOTH a prediction AND a structured explanation.
        The explanation must reference specific news items by number.
        """
        recent = historical_data.tail(lookback_days)
        price_lines = "\n".join(
            f"{row['date']}: {row['close']:.4f}"
            for _, row in recent.iterrows()
        )
        current_price = float(recent["close"].iloc[-1])

        # Format news with numbered items so we can check grounding
        news_lines = []
        sorted_news = sorted(news_articles, key=lambda x: x.get("published_at", ""), reverse=True)[:max_news]
        for i, art in enumerate(sorted_news, 1):
            title   = art.get("title", "")
            date    = art.get("published_at", "")[:10]
            summary = art.get("summary", "")
            if len(summary) > 150:
                summary = summary[:147] + "..."
            news_lines.append(f"[{i}] [{date}] {title}" + (f" — {summary}" if summary else ""))
        news_text = "\n".join(news_lines) if news_lines else "No recent news."

        example_preds = ", ".join([f"{current_price * (1 + 0.001*i):.4f}" for i in range(1, self.horizon+1)])

        prompt = f"""You are an expert FX analyst. Analyze the data below and provide:
1. A price forecast
2. A structured explanation citing specific news items

Recent News (numbered for reference):
{news_text}

Historical {self.base_currency}/{self.quote_currency} Rates (last {lookback_days} days):
{price_lines}

Current Rate: {current_price:.4f}

OUTPUT FORMAT — respond with ONLY this JSON structure:
{{
  "predictions": [{example_preds}],
  "explanation": {{
    "summary": "1-2 sentence overall outlook",
    "key_factors": ["factor1", "factor2", "factor3"],
    "news_references": [1, 3],
    "confidence": 0.7,
    "direction": "bullish/bearish/neutral",
    "risk_factors": ["risk1", "risk2"]
  }}
}}

Replace placeholder values with your actual analysis. The news_references array must contain
the numbers of the news items you are citing (e.g. [1, 3] means you are citing news [1] and [3]).

Respond with ONLY the JSON object above, no other text."""

        return prompt


# ---------------------------------------------------------------------------
# Explanation parser
# ---------------------------------------------------------------------------

class ExplanationParser:
    """Extract structured fields from LLM explanation JSON or free text."""

    FACTOR_CATEGORIES = ExplainabilityPromptBuilder.FACTOR_CATEGORIES

    def parse(self, response: str, news_articles: List[Dict]) -> Dict:
        """
        Parse LLM response to extract predictions + explanation fields.

        Returns
        -------
        dict with keys:
            predictions, explanation_text, news_refs, factor_types,
            confidence_score, explanation_len, coherence_score,
            news_grounding_rate (fraction of actual news cited)
        """
        result = {
            "predictions":         [],
            "explanation_text":    "",
            "news_refs":           [],
            "factor_types":        [],
            "confidence_score":    0.5,
            "explanation_len":     0,
            "coherence_score":     0.0,
            "news_grounding_rate": 0.0,
        }

        text = response.strip()

        # --- Try JSON parse ---
        try:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                result["predictions"] = data.get("predictions", [])
                expl = data.get("explanation", {})

                summary     = expl.get("summary", "")
                key_factors = expl.get("key_factors", [])
                news_refs   = expl.get("news_references", [])
                confidence  = float(expl.get("confidence", 0.5))
                risk_factors = expl.get("risk_factors", [])

                full_text = " ".join([summary] + key_factors + risk_factors)
                result["explanation_text"]  = full_text
                result["news_refs"]         = news_refs if isinstance(news_refs, list) else []
                result["confidence_score"]  = max(0.0, min(1.0, confidence))
        except Exception:
            # Fallback: extract predictions from array pattern
            arr = re.search(r"\[([\d.,\s]+)\]", text)
            if arr:
                try:
                    result["predictions"] = [float(x) for x in arr.group(1).split(",") if x.strip()]
                except Exception:
                    pass
            result["explanation_text"] = text

        # --- Explanation length ---
        result["explanation_len"] = len(result["explanation_text"].split())

        # --- News grounding ---
        cited_count = len(result["news_refs"])
        total_news  = len(news_articles)
        if total_news > 0 and cited_count > 0:
            # Check refs are valid numbers pointing to actual news
            valid_refs = [r for r in result["news_refs"] if isinstance(r, int) and 1 <= r <= total_news]
            result["news_grounding_rate"] = len(valid_refs) / total_news
        else:
            # Fallback: check if any news title keyword appears in explanation text
            expl_lower = result["explanation_text"].lower()
            matched = sum(
                1 for art in news_articles
                if any(word.lower() in expl_lower for word in art.get("title", "").split()[:4])
            )
            result["news_grounding_rate"] = matched / max(total_news, 1)

        # --- Factor type detection ---
        expl_lower = result["explanation_text"].lower()
        cited_factors = []
        for category, keywords in self.FACTOR_CATEGORIES.items():
            if any(kw in expl_lower for kw in keywords):
                cited_factors.append(category)
        result["factor_types"] = cited_factors

        # --- Coherence score (heuristic) ---
        result["coherence_score"] = self._coherence_score(result["explanation_text"])

        return result

    def _coherence_score(self, text: str) -> float:
        """
        Heuristic coherence score (0-1) based on:
        - Length (longer = more detailed, up to a point)
        - Presence of causal connectors
        - Number of distinct financial terms
        """
        if not text or len(text) < 10:
            return 0.0

        words = text.lower().split()
        word_count = len(words)

        # Length score (0-0.3): peaks at ~50 words
        len_score = min(0.3, word_count / 50 * 0.3)

        # Causal connectors (0-0.4)
        connectors = ["because", "due to", "driven by", "as a result", "following",
                      "amid", "despite", "given", "reflecting", "supported by", "weighed by"]
        connector_count = sum(1 for c in connectors if c in text.lower())
        connector_score = min(0.4, connector_count * 0.1)

        # Financial term density (0-0.3)
        all_keywords = [kw for kws in self.FACTOR_CATEGORIES.values() for kw in kws]
        fin_count = sum(1 for kw in all_keywords if kw in text.lower())
        fin_score = min(0.3, fin_count * 0.05)

        return round(len_score + connector_score + fin_score, 3)


# ---------------------------------------------------------------------------
# Ablation 4 runner
# ---------------------------------------------------------------------------

class ExplainabilityAblation:
    """
    Run Ablation 4 (Explainability) and Ablation 5 (Explanation-Prediction Correlation).

    Parameters
    ----------
    llm_forecaster : LLMForecaster instance from llm_api_helper.py
    output_dir     : where to save results
    """

    def __init__(self, llm_forecaster, output_dir: str = "results"):
        self.forecaster  = llm_forecaster
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parser      = ExplanationParser()

    def run(
        self,
        data: pd.DataFrame,
        news_articles: List[Dict],
        currency: str,
        horizon: int,
        llm_models: Optional[List[str]] = None,
        lookback_days: int = 30,
    ) -> Dict:
        """
        Run Ablation 4 & 5 for one currency/horizon combination.

        Returns summary dict with all metrics.
        """
        if llm_models is None:
            llm_models = ["llama3-8b", "qwen2.5-7b", "mistral-7b"]

        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        test_data  = data[split_idx:]
        test_size  = min(50, len(test_data))   # fewer calls for explainability
        test_subset = test_data[-test_size:]
        context_data = pd.concat([train_data.tail(500), test_subset])

        prompt_builder = ExplainabilityPromptBuilder(currency, horizon)

        ablation4_rows = []
        ablation5_rows = []

        for llm_model in llm_models:
            print(f"\n  [{currency}|h={horizon}|{llm_model}] Explainability ablation...")

            # We run one prediction per test window (rolling) but limit to
            # at most 20 calls to keep runtime reasonable
            n_calls = min(20, test_size)

            for i in range(n_calls):
                window_end = split_idx + i + lookback_days
                if window_end > len(data):
                    break

                window_data = data.iloc[window_end - lookback_days: window_end]
                future_start = window_end
                future_end   = window_end + horizon
                if future_end > len(data):
                    break

                y_true = data.iloc[future_start:future_end]["close"].values

                try:
                    prompt = prompt_builder.build_explain_prompt(
                        window_data, news_articles, lookback_days, max_news=8
                    )
                    response = self.forecaster.api.call(
                        prompt, model=llm_model, temperature=0.1, max_tokens=600
                    )
                    parsed = self.parser.parse(response, news_articles)

                    # Metrics for Ablation 4
                    ablation4_rows.append({
                        "currency":            currency,
                        "horizon":             horizon,
                        "model":               llm_model,
                        "window_idx":          i,
                        "news_grounding_rate": parsed["news_grounding_rate"],
                        "explanation_len":     parsed["explanation_len"],
                        "coherence_score":     parsed["coherence_score"],
                        "factor_types":        json.dumps(parsed["factor_types"]),
                        "n_factors_cited":     len(parsed["factor_types"]),
                        "confidence_score":    parsed["confidence_score"],
                    })

                    # Metrics for Ablation 5 (only if prediction available)
                    preds = np.array(parsed["predictions"])
                    if len(preds) >= horizon and len(y_true) >= horizon:
                        preds = preds[:horizon]
                        abs_error = float(np.mean(np.abs(y_true - preds)))

                        # Directional accuracy (horizon > 1)
                        if len(y_true) > 1:
                            da = float(np.mean(
                                np.sign(np.diff(y_true)) == np.sign(np.diff(preds))
                            ) * 100)
                        else:
                            da = float("nan")

                        ablation5_rows.append({
                            "currency":            currency,
                            "horizon":             horizon,
                            "model":               llm_model,
                            "window_idx":          i,
                            "explanation_len":     parsed["explanation_len"],
                            "news_grounding_rate": parsed["news_grounding_rate"],
                            "n_factors_cited":     len(parsed["factor_types"]),
                            "coherence_score":     parsed["coherence_score"],
                            "confidence_score":    parsed["confidence_score"],
                            "abs_pred_error":      abs_error,
                            "directional_accuracy": da,
                        })

                    print(f"    [{i}] coherence={parsed['coherence_score']:.2f} "
                          f"grounding={parsed['news_grounding_rate']:.2f} "
                          f"len={parsed['explanation_len']}")

                except Exception as e:
                    print(f"    [{i}] ERROR: {e}")
                    continue

        # Aggregate Ablation 4 metrics
        abl4_df = pd.DataFrame(ablation4_rows)
        abl4_summary = {}
        if not abl4_df.empty:
            abl4_summary = abl4_df.groupby("model").agg(
                avg_news_grounding=("news_grounding_rate", "mean"),
                avg_explanation_len=("explanation_len", "mean"),
                avg_coherence=("coherence_score", "mean"),
                avg_n_factors=("n_factors_cited", "mean"),
                avg_confidence=("confidence_score", "mean"),
                n_samples=("window_idx", "count"),
            ).reset_index().to_dict("records")

        # Ablation 5 correlations
        abl5_df  = pd.DataFrame(ablation5_rows)
        abl5_corr = self._compute_correlations(abl5_df)

        # Save detailed rows
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        abl4_path = self.output_dir / f"ablation4_detail_{currency}_h{horizon}_{timestamp}.csv"
        abl5_path = self.output_dir / f"ablation5_detail_{currency}_h{horizon}_{timestamp}.csv"
        if not abl4_df.empty:
            abl4_df.to_csv(abl4_path, index=False)
        if not abl5_df.empty:
            abl5_df.to_csv(abl5_path, index=False)

        return {
            "currency":        currency,
            "horizon":         horizon,
            "ablation4_rows":  ablation4_rows,
            "ablation4_summary": abl4_summary,
            "ablation5_rows":  ablation5_rows,
            "ablation5_correlations": abl5_corr,
        }

    def _compute_correlations(self, df: pd.DataFrame) -> Dict:
        """Compute Ablation 5 correlations per model."""
        if df.empty:
            return {}

        corr_rows = []
        numeric_cols = ["explanation_len", "news_grounding_rate",
                        "n_factors_cited", "coherence_score", "confidence_score"]
        target_cols  = {"abs_pred_error": "neg", "directional_accuracy": "pos"}

        for model, grp in df.groupby("model"):
            grp = grp.dropna(subset=["abs_pred_error"])
            if len(grp) < 5:
                continue
            row = {"model": model, "n": len(grp)}
            for pred_var in numeric_cols:
                for target, direction in target_cols.items():
                    if pred_var in grp.columns and target in grp.columns:
                        valid = grp[[pred_var, target]].dropna()
                        if len(valid) >= 5:
                            corr = float(valid[pred_var].corr(valid[target]))
                            key = f"corr_{pred_var}_vs_{target}"
                            row[key] = round(corr, 4)
            corr_rows.append(row)

        return corr_rows

    def run_all_currencies(
        self,
        data_dir: str,
        news_dir: str,
        currencies: Optional[List[str]] = None,
        horizons: Optional[Dict] = None,
        llm_models: Optional[List[str]] = None,
    ) -> Dict:
        """Run Ablation 4 & 5 for all currencies and horizons."""
        if currencies is None:
            currencies = ["USDIDR", "USDEUR", "USDSGD", "USDTWD", "USDAUD"]
        if horizons is None:
            horizons = {"short": 1, "medium": 5, "long": 10}

        all_abl4 = []
        all_abl5_corr = []

        for currency in currencies:
            data_path = Path(data_dir) / f"{currency}.csv"
            news_path = Path(news_dir) / f"{currency}_news.json"

            if not data_path.exists():
                print(f"[SKIP] {currency}: data file not found")
                continue

            data  = pd.read_csv(data_path)
            news  = json.load(open(news_path)) if news_path.exists() else []

            for horizon_name, horizon in horizons.items():
                result = self.run(
                    data, news, currency, horizon,
                    llm_models=llm_models,
                )
                all_abl4.extend(result.get("ablation4_rows", []))
                all_abl5_corr.extend(result.get("ablation5_correlations", []))

        # Save global summaries
        if all_abl4:
            abl4_global = pd.DataFrame(all_abl4)
            abl4_global.to_csv(self.output_dir / "ablation4_summary.csv", index=False)
            print(f"\n[SAVED] ablation4_summary.csv ({len(abl4_global)} rows)")

        if all_abl5_corr:
            abl5_global = pd.DataFrame(all_abl5_corr)
            abl5_global.to_csv(self.output_dir / "ablation5_correlations.csv", index=False)
            print(f"[SAVED] ablation5_correlations.csv ({len(abl5_global)} rows)")

        return {
            "ablation4_rows": all_abl4,
            "ablation5_correlations": all_abl5_corr,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Run Ablation 4 & 5 (Explainability)")
    ap.add_argument("--data-dir",  default="data/fx_filled")
    ap.add_argument("--news-dir",  default="data/fx_news")
    ap.add_argument("--output",    default="results")
    ap.add_argument("--currency",  default=None, help="Single currency (default: all)")
    ap.add_argument("--horizon",   type=int, default=None, help="Single horizon days (default: all)")
    ap.add_argument("--backend",   default="vllm", choices=["vllm", "ollama"])
    args = ap.parse_args()

    # Import LLM forecaster
    sys.path.insert(0, str(Path(__file__).parent))
    from llm_api_helper import LLMForecaster

    forecaster = LLMForecaster(backend=args.backend)
    abl = ExplainabilityAblation(forecaster, output_dir=args.output)

    currencies = [args.currency] if args.currency else None
    horizons   = {str(args.horizon): args.horizon} if args.horizon else None

    results = abl.run_all_currencies(
        data_dir=args.data_dir,
        news_dir=args.news_dir,
        currencies=currencies,
        horizons=horizons,
    )

    print("\n" + "=" * 60)
    print("Ablation 4 & 5 completed!")
    print(f"Results saved to: {args.output}")
    print("=" * 60)