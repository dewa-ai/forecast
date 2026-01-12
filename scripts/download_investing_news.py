# import os, json, time
# from datetime import datetime
# import yaml
# from playwright.sync_api import sync_playwright

# def accept_consent_if_present(page):
#     candidates = [
#         "button:has-text('I Accept')",
#         "button:has-text('Accept')",
#         "button:has-text('AGREE')",
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

# def safe_text(loc, default=""):
#     try:
#         if loc.count() == 0:
#             return default
#         return loc.first.inner_text().strip()
#     except Exception:
#         return default

# def scrape_news(page, url: str, sel_card: str, sel_link: str, sel_time: str, sel_snippet: str, max_items: int = 300):
#     page.goto(url, wait_until="domcontentloaded", timeout=60_000)
#     page.wait_for_timeout(1200)
#     accept_consent_if_present(page)

#     items = []
#     cards = page.locator(sel_card)
#     n = min(cards.count(), max_items)

#     for i in range(n):
#         c = cards.nth(i)
#         link = c.locator(sel_link)
#         href = ""
#         title = ""
#         try:
#             if link.count() > 0:
#                 href = link.first.get_attribute("href") or ""
#                 title = link.first.inner_text().strip()
#         except Exception:
#             pass

#         t = safe_text(c.locator(sel_time), "")
#         snip = safe_text(c.locator(sel_snippet), "")

#         # Normalize href (Investing often uses relative)
#         if href.startswith("/"):
#             href = "https://www.investing.com" + href

#         if title:
#             items.append({
#                 "title": title,
#                 "time_text": t,
#                 "snippet": snip,
#                 "url": href,
#             })

#     return items

# def main():
#     cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

#     out_dir = "data/news_raw"
#     os.makedirs(out_dir, exist_ok=True)

#     throttle = float(cfg["scraping"]["throttle_seconds"])
#     user_agent = cfg["scraping"]["user_agent"]
#     headless = bool(cfg["scraping"]["headless"])
#     timeout_ms = int(cfg["scraping"]["timeout_ms"])

#     sel_card = cfg["selectors"]["news_card"]
#     sel_link = cfg["selectors"]["news_headline_link"]
#     sel_time = cfg["selectors"]["news_time"]
#     sel_snip = cfg["selectors"]["news_snippet"]

#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=headless)
#         context = browser.new_context(user_agent=user_agent, locale=cfg["scraping"]["locale"])
#         page = context.new_page()
#         page.set_default_timeout(timeout_ms)

#         for pair, meta in cfg["pairs"].items():
#             url = meta["news_url"]
#             print(f"[NEWS] Fetching {pair}: {url}")
#             time.sleep(throttle)

#             try:
#                 items = scrape_news(page, url, sel_card, sel_link, sel_time, sel_snip)
#                 payload = {
#                     "pair": pair,
#                     "url": url,
#                     "fetched_at_utc": datetime.utcnow().isoformat(),
#                     "items": items,
#                 }
#                 with open(os.path.join(out_dir, f"{pair}.json"), "w", encoding="utf-8") as f:
#                     json.dump(payload, f, ensure_ascii=False, indent=2)
#                 print(f"  -> saved {len(items)} items")
#             except Exception as e:
#                 print(f"  !! failed {pair}: {e}")

#         browser.close()

# if __name__ == "__main__":
#     main()


#--------------------------------------------------------------
# final
#--------------------------------------------------------------

import os, json, time
from datetime import datetime
from playwright.sync_api import sync_playwright

PAIRS = {
    "USDIDR": "https://www.investing.com/currencies/usd-idr-news",
    "USDTWD": "https://www.investing.com/currencies/usd-twd-news",
    "USDAUD": "https://www.investing.com/currencies/usd-aud-news",
    "USDEUR": "https://www.investing.com/currencies/eur-usd-news",
    "USDSGD": "https://www.investing.com/currencies/usd-sgd-news",
}

OUT_DIR = "data/news_raw"
THROTTLE = 2.0

def accept_cookie(page):
    try:
        page.locator("button#onetrust-accept-btn-handler").click(timeout=2000)
    except:
        pass

def scrape_news(page, url):
    page.goto(url, timeout=60000)
    page.wait_for_timeout(1500)
    accept_cookie(page)

    items = []

    # Investing news cards (robust selector)
    cards = page.locator("article")
    for i in range(min(cards.count(), 300)):
        c = cards.nth(i)

        title = c.locator("a").first.inner_text().strip()
        link = c.locator("a").first.get_attribute("href")
        time_txt = c.locator("time").inner_text() if c.locator("time").count() else ""

        if link and link.startswith("/"):
            link = "https://www.investing.com" + link

        items.append({
            "title": title,
            "time_text": time_txt,
            "url": link
        })

    return items

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X)"
        )

        for pair, url in PAIRS.items():
            print(f"[NEWS] {pair}")
            time.sleep(THROTTLE)

            try:
                items = scrape_news(page, url)
                payload = {
                    "pair": pair,
                    "url": url,
                    "fetched_at": datetime.utcnow().isoformat(),
                    "items": items
                }

                with open(f"{OUT_DIR}/{pair}.json", "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

                print(f"  saved {len(items)} items")

            except Exception as e:
                print(f"  ERROR: {e}")

        browser.close()

if __name__ == "__main__":
    main()
