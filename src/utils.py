from __future__ import annotations
import os, json, re, random
from typing import Any, Dict, Optional
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract first JSON object from a model response.
    Returns dict or None.
    """
    m = _JSON_RE.search(text)
    if not m:
        return None
    block = m.group(0)
    block = block.strip()

    # Try strict JSON first
    try:
        return json.loads(block)
    except Exception:
        pass

    # Try to fix common issues: single quotes, trailing commas
    block2 = block.replace("'", '"')
    block2 = re.sub(r",\s*}", "}", block2)
    block2 = re.sub(r",\s*]", "]", block2)
    try:
        return json.loads(block2)
    except Exception:
        return None

def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None
