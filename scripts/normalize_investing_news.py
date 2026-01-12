# import os, json
# import pandas as pd
# from dateutil import parser

# def parse_time_text(s: str):
#     if not s:
#         return pd.NaT
#     try:
#         return parser.parse(s, fuzzy=True)
#     except:
#         return pd.NaT

# def main():
#     in_dir = "data/news_raw"
#     out_dir = "data/news"
#     os.makedirs(out_dir, exist_ok=True)

#     for fn in os.listdir(in_dir):
#         if not fn.endswith(".json"):
#             continue
#         payload = json.load(open(os.path.join(in_dir, fn), "r", encoding="utf-8"))
#         pair = payload["pair"]
#         items = payload["items"]

#         df = pd.DataFrame(items)
#         if df.empty:
#             print(f"[NEWS] Skip {pair}: empty")
#             continue

#         df["datetime"] = df.get("time_text", "").apply(parse_time_text)
#         # Sometimes time_text is missing; keep rows but mark NaT.
#         df["title"] = df["title"].fillna("").astype(str)
#         df["snippet"] = df.get("snippet", "").fillna("").astype(str)
#         df["url"] = df.get("url", "").fillna("").astype(str)

#         df = df.drop_duplicates(subset=["title", "url"], keep="first")
#         df.to_csv(os.path.join(out_dir, f"{pair}.csv"), index=False)
#         print(f"[NEWS] Wrote {pair}: {len(df)} items")

# if __name__ == "__main__":
#     main()


#--------------------------------------------------------------
# final
#--------------------------------------------------------------

import os, json
import pandas as pd
from dateutil import parser

IN_DIR = "data/news_raw"
OUT_DIR = "data/news"

def parse_time(x):
    try:
        return parser.parse(x, fuzzy=True)
    except:
        return pd.NaT

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for fn in os.listdir(IN_DIR):
        if not fn.endswith(".json"):
            continue

        payload = json.load(open(f"{IN_DIR}/{fn}", encoding="utf-8"))
        pair = payload["pair"]

        df = pd.DataFrame(payload["items"])
        if df.empty:
            continue

        df["datetime"] = df["time_text"].apply(parse_time)
        df = df.drop_duplicates(subset=["title"])
        df = df[["datetime", "title", "url"]]

        df.to_csv(f"{OUT_DIR}/{pair}.csv", index=False)
        print(f"[NEWS] normalized {pair}")

if __name__ == "__main__":
    main()
