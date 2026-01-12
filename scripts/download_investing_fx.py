# import os, json, time
# from datetime import datetime
# import yaml
# from playwright.sync_api import sync_playwright

# def accept_consent_if_present(page):
#     candidates = [
#         "button:has-text('I Accept')",
#         "button:has-text('Accept')",
#         "button:has-text('AGREE')",
#         "button:has-text('Got it')",
#         "button#onetrust-accept-btn-handler",
#     ]
#     for sel in candidates:
#         try:
#             btn = page.locator(sel)
#             if btn.count() > 0 and btn.first.is_visible():
#                 btn.first.click(timeout=1500)
#                 page.wait_for_timeout(500)
#                 return
#         except Exception:
#             pass

# def scrape_fx_rows(page, url: str, row_selector: str):
#     page.goto(url, wait_until="domcontentloaded", timeout=60_000)
#     page.wait_for_timeout(1200)
#     accept_consent_if_present(page)

#     rows = page.locator(row_selector)
#     n = rows.count()

#     out = []
#     for i in range(min(n, 5000)):  # cap safety
#         tr = rows.nth(i)
#         tds = tr.locator("td")
#         if tds.count() < 5:
#             continue
#         # Typical columns: Date, Price, Open, High, Low, Vol., Change%
#         vals = [tds.nth(j).inner_text().strip() for j in range(min(tds.count(), 7))]
#         out.append(vals)
#     return out

# def main():
#     cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
#     out_dir = "data/fx_raw"
#     os.makedirs(out_dir, exist_ok=True)

#     throttle = float(cfg["scraping"]["throttle_seconds"])
#     user_agent = cfg["scraping"]["user_agent"]
#     headless = bool(cfg["scraping"]["headless"])
#     timeout_ms = int(cfg["scraping"]["timeout_ms"])

#     row_selector = cfg["selectors"]["fx_table_rows"]
#     pairs = cfg["pairs"]

#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=headless)
#         context = browser.new_context(user_agent=user_agent, locale=cfg["scraping"]["locale"])
#         page = context.new_page()
#         page.set_default_timeout(timeout_ms)

#         for pair, meta in pairs.items():
#             url = meta["fx_url"]
#             print(f"[FX] Fetching {pair}: {url}")
#             time.sleep(throttle)

#             try:
#                 rows = scrape_fx_rows(page, url, row_selector)
#                 payload = {
#                     "pair": pair,
#                     "url": url,
#                     "fetched_at_utc": datetime.utcnow().isoformat(),
#                     "rows": rows,
#                 }
#                 with open(os.path.join(out_dir, f"{pair}.json"), "w", encoding="utf-8") as f:
#                     json.dump(payload, f, ensure_ascii=False, indent=2)
#                 print(f"  -> saved {len(rows)} rows")
#             except Exception as e:
#                 print(f"  !! failed {pair}: {e}")

#         browser.close()

# if __name__ == "__main__":
#     main()


#--------------------------------------------------------------
# final - v1
#--------------------------------------------------------------

# import os, json, time
# from datetime import datetime
# from playwright.sync_api import sync_playwright

# PAIRS = {
#     "USDIDR": "https://www.investing.com/currencies/usd-idr-historical-data",
#     "USDTWD": "https://www.investing.com/currencies/usd-twd-historical-data",
#     "USDAUD": "https://www.investing.com/currencies/usd-aud-historical-data",
#     "USDEUR": "https://www.investing.com/currencies/eur-usd-historical-data",  # note: EUR/USD
#     "USDSGD": "https://www.investing.com/currencies/usd-sgd-historical-data",
# }

# OUT_DIR = "data/fx_raw"
# THROTTLE = 2.0

# def accept_cookie(page):
#     try:
#         page.locator("button#onetrust-accept-btn-handler").click(timeout=2000)
#     except:
#         pass

# def scrape_fx_table(page, url):
#     page.goto(url, timeout=60000)
#     page.wait_for_timeout(1500)
#     accept_cookie(page)

#     rows = []
#     trs = page.locator("table tbody tr")

#     for i in range(min(trs.count(), 5000)):
#         tr = trs.nth(i)
#         tds = tr.locator("td")
#         if tds.count() < 2:
#             continue

        
#         vals = [tds.nth(j).inner_text().strip() for j in range(min(tds.count(), 7))]
#         rows.append(vals)

#     return rows

# def main():
#     os.makedirs(OUT_DIR, exist_ok=True)

#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=True)
#         page = browser.new_page(
#             user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X)"
#         )

#         for pair, url in PAIRS.items():
#             print(f"[FX] {pair}")
#             time.sleep(THROTTLE)

#             try:
#                 rows = scrape_fx_table(page, url)
#                 payload = {
#                     "pair": pair,
#                     "url": url,
#                     "fetched_at": datetime.utcnow().isoformat(),
#                     "rows": rows
#                 }

#                 with open(f"{OUT_DIR}/{pair}.json", "w", encoding="utf-8") as f:
#                     json.dump(payload, f, ensure_ascii=False, indent=2)

#                 print(f"  saved {len(rows)} rows")
#             except Exception as e:
#                 print(f"  ERROR: {e}")

#         browser.close()

# if __name__ == "__main__":
#     main()


#--------------------------------------------------------------
# final - v2
#---------------------------------------------------------------

import os
import re
import json
import time
from datetime import date, datetime
from typing import List

from playwright.sync_api import sync_playwright

# =========================
# CONFIG
# =========================
PAIRS = {
    "USDIDR": "https://www.investing.com/currencies/usd-idr-historical-data",
    "USDTWD": "https://www.investing.com/currencies/usd-twd-historical-data",
    "USDAUD": "https://www.investing.com/currencies/usd-aud-historical-data",
    "USDEUR": "https://www.investing.com/currencies/eur-usd-historical-data",
    "USDSGD": "https://www.investing.com/currencies/usd-sgd-historical-data",
}

OUT_DIR = "data/fx_raw"
START = date(2015, 1, 1)
END   = date(2025, 12, 22)

THROTTLE = 2.0
AJAX_URL = "https://www.investing.com/instruments/HistoricalDataAjax"

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

# =========================
# HELPERS
# =========================
def mmddyyyy(d: date) -> str:
    return f"{d.month:02d}/{d.day:02d}/{d.year:04d}"

def yearly_chunks(start: date, end: date):
    chunks = []
    for y in range(start.year, end.year + 1):
        cs = date(y, 1, 1)
        ce = date(y, 12, 31)
        if y == start.year:
            cs = start
        if y == end.year:
            ce = end
        chunks.append((cs, ce))
    return chunks

def accept_cookie(page):
    try:
        page.locator("#onetrust-accept-btn-handler").click(timeout=2500)
    except:
        pass

def parse_rows_from_html_table(html: str):
    """
    HistoricalDataAjax returns HTML snippet with <table> rows.
    Output row format (usually):
      [Date, Price, Open, High, Low, Vol, Change%]
    """
    trs = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I)
    rows = []
    for tr in trs:
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S | re.I)
        if len(tds) < 2:
            continue
        clean = []
        for td in tds[:7]:
            td_txt = re.sub(r"<[^>]+>", "", td)
            td_txt = td_txt.replace("&nbsp;", " ")
            td_txt = re.sub(r"\s+", " ", td_txt).strip()
            clean.append(td_txt)
        rows.append(clean)
    return rows

def find_ids_in_obj(obj) -> List[int]:
    target_keys = {"pairId", "pair_id", "currId", "curr_id", "instrumentId", "instrument_id", "id"}
    found = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and k in target_keys:
                    if isinstance(v, int) and v >= 1000:
                        found.append(v)
                    elif isinstance(v, str) and v.isdigit() and int(v) >= 1000:
                        found.append(int(v))
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(obj)
    return found

def extract_curr_id_from_page_state(page, pair: str) -> int:
    """
    Extract curr_id/pairId from SPA state:
    - script#__NEXT_DATA__ (Next.js)
    - fallback: scan scripts
    """
    html = page.content()
    with open(f"debug_{pair}.html", "w", encoding="utf-8") as f:
        f.write(html)

    candidates: List[int] = []

    # Next.js __NEXT_DATA__
    try:
        nxt = page.locator("script#__NEXT_DATA__")
        if nxt.count():
            txt = nxt.first.text_content() or ""
            if txt.strip():
                obj = json.loads(txt)
                candidates += find_ids_in_obj(obj)
                with open(f"debug_{pair}__NEXT_DATA__.json", "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False)
    except:
        pass

    # scan all scripts text (fallback)
    try:
        scripts_text = page.evaluate(
            "() => Array.from(document.scripts).map(s => s.textContent || '').filter(t => t.length > 0)"
        )
        joined = "\n".join(scripts_text)
        for pat in [
            r'"pairId"\s*:\s*(\d+)',
            r'"curr_id"\s*:\s*(\d+)',
            r'"currId"\s*:\s*(\d+)',
        ]:
            candidates += [int(x) for x in re.findall(pat, joined)]
    except:
        pass

    candidates = [c for c in candidates if isinstance(c, int) and c >= 1000]
    if not candidates:
        raise RuntimeError(f"Could not extract curr_id (saved debug_{pair}.html).")

    from collections import Counter
    return Counter(candidates).most_common(1)[0][0]

def fetch_historical_via_inpage_fetch(page, referer_url: str, curr_id: int, st: date, en: date) -> str:
    """
    Call HistoricalDataAjax using browser's fetch() INSIDE the page.
    This often bypasses Cloudflare blocks that hit requests/context.request.
    """
    form = {
        "curr_id": str(curr_id),
        "smlID": "0",
        "header": "null",
        "st_date": mmddyyyy(st),
        "end_date": mmddyyyy(en),
        "interval_sec": "Daily",
        "sort_col": "date",
        "sort_ord": "DESC",
        "action": "historical_data",
    }

    js = """
    async ({ url, referer, form }) => {
      const body = new URLSearchParams(form).toString();

      const resp = await fetch(url, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
          "X-Requested-With": "XMLHttpRequest",
          "Accept": "text/html, */*; q=0.01",
          "Referer": referer
        },
        body
      });

      const text = await resp.text();
      return { status: resp.status, text };
    }
    """
    out = page.evaluate(js, {"url": AJAX_URL, "referer": referer_url, "form": form})
    status = out["status"]
    text = out["text"]

    if status != 200:
        raise RuntimeError(f"In-page fetch failed HTTP {status}: {text[:200]}")
    if "Just a moment" in text or "cf-chl" in text:
        raise RuntimeError("In-page fetch returned Cloudflare challenge page.")

    return text

# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    chunks = yearly_chunks(START, END)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for pair, url in PAIRS.items():
            print(f"\n[FX] {pair}")
            time.sleep(THROTTLE)

            context = browser.new_context(
                locale="en-US",
                user_agent=UA,
                viewport={"width": 1280, "height": 720},
            )
            page = context.new_page()

            try:
                # Open main page (get CF cookies/JS state)
                page.goto(url, wait_until="domcontentloaded", timeout=0)
                page.wait_for_timeout(2500)
                accept_cookie(page)

                # Ensure table exists in DOM (not necessarily visible)
                page.wait_for_selector("table tbody tr", state="attached", timeout=60000)

                curr_id = extract_curr_id_from_page_state(page, pair)
                print(f"  curr_id = {curr_id}")

                all_rows = []
                for st, en in chunks:
                    html = fetch_historical_via_inpage_fetch(page, url, curr_id, st, en)
                    rows = parse_rows_from_html_table(html)
                    print(f"  {st} → {en}: {len(rows)} rows")
                    all_rows.extend(rows)
                    time.sleep(0.4)

                # dedupe by date string
                dedup = {}
                for r in all_rows:
                    if r:
                        dedup[r[0]] = r
                final_rows = list(dedup.values())

                out_path = f"{OUT_DIR}/{pair}.json"
                payload = {
                    "pair": pair,
                    "url": url,
                    "curr_id": curr_id,
                    "range": {"start": START.isoformat(), "end": END.isoformat()},
                    "fetched_at": datetime.utcnow().isoformat(),
                    "rows": final_rows,
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

                print(f"saved {len(final_rows)} unique rows -> {out_path}")

            except Exception as e:
                print(f"ERROR: {e}")

            context.close()

        browser.close()

if __name__ == "__main__":
    main()

