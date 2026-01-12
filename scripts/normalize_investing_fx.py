import os, json
import yaml
import pandas as pd
from dateutil import parser

def parse_number(x: str):
    if x is None:
        return None
    s = str(x).strip()
    s = s.replace(",", "")
    s = s.replace("%", "")
    try:
        return float(s)
    except:
        return None

def parse_date_any(s: str):
    try:
        return parser.parse(s, dayfirst=False, fuzzy=True)
    except:
        return pd.NaT

def main():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    in_dir = "data/fx_raw"
    out_dir = "data/fx"
    os.makedirs(out_dir, exist_ok=True)

    for fn in os.listdir(in_dir):
        if not fn.endswith(".json"):
            continue
        payload = json.load(open(os.path.join(in_dir, fn), "r", encoding="utf-8"))
        pair = payload["pair"]
        rows = payload["rows"]

        # Attempt to map based on common Investing historical columns
        # rows = [ [Date, Price, Open, High, Low, Vol, Change%], ... ]
        df = pd.DataFrame(rows)
        if df.shape[1] < 2:
            print(f"[FX] Skip {pair}: too few columns")
            continue

        df = df.rename(columns={0: "date", 1: "price"})
        df["date"] = df["date"].apply(parse_date_any)
        df["price"] = df["price"].apply(parse_number)

        df = df.dropna(subset=["date", "price"]).sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")

        # Invert rate if configured (e.g. EUR/USD -> USD/EUR)
        invert = bool(cfg["pairs"][pair].get("invert", False))
        if invert:
            df["price"] = 1.0 / df["price"]

        df[["date", "price"]].to_csv(os.path.join(out_dir, f"{pair}.csv"), index=False)
        print(f"[FX] Wrote {pair}: {len(df)} rows")

if __name__ == "__main__":
    main()



#--------------------------------------------------------------
# final
#---------------------------------------------------------------

import os, json
import pandas as pd
from dateutil import parser

IN_DIR = "data/fx_raw"
OUT_DIR = "data/fx"

INVERT = {
    "USDEUR": True,
}

def parse_date(s):
    try:
        return parser.parse(s, fuzzy=True)
    except:
        return pd.NaT

def parse_num(s):
    try:
        return float(str(s).replace(",", "").strip())
    except:
        return None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for fn in os.listdir(IN_DIR):
        if not fn.endswith(".json"):
            continue

        payload = json.load(open(f"{IN_DIR}/{fn}", encoding="utf-8"))
        pair = payload["pair"]
        rows = payload["rows"]

        df = pd.DataFrame(rows)
        if df.shape[1] < 2:
            print(f"[FX] skip {pair}: too few columns")
            continue

        df = df.rename(columns={0: "date", 1: "price"})
        df["date"] = df["date"].apply(parse_date)
        df["price"] = df["price"].apply(parse_num)

        df = df.dropna(subset=["date", "price"]).sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")

        if INVERT.get(pair, False):
            df["price"] = 1.0 / df["price"]

        df[["date", "price"]].to_csv(f"{OUT_DIR}/{pair}.csv", index=False)
        print(f"[FX] normalized {pair} → {OUT_DIR}/{pair}.csv ({len(df)} rows)")

if __name__ == "__main__":
    main()
