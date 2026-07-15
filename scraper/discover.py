"""Playwright discovery: navigate pickleball.com results/rankings, intercept every
JSON response, and dump it to disk so we can find the internal data endpoints.

Run: python scraper/discover.py
Output: scraper/discovery/*.json  (one file per page visited + a summary)
"""
import asyncio
import json
import os
import re
from pathlib import Path

from playwright.async_api import async_playwright

OUT = Path(__file__).parent / "discovery"
OUT.mkdir(exist_ok=True)

# Pages worth probing. The results/rankings pages are the ones that hit the
# authenticated data backend.
TARGETS = [
    ("results", "https://pickleball.com/results"),
    ("rankings", "https://pickleball.com/rankings"),
]

# Only log JSON bodies from these hosts (skip analytics/ads noise).
INTERESTING_HOST = re.compile(r"pickleball\.com|majorleague|ppatour|picklewave|forwrd", re.I)


async def probe(page_name, url, browser):
    log = []
    context = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
        )
    )
    page = await context.new_page()

    async def on_response(resp):
        try:
            ct = resp.headers.get("content-type", "")
            if "application/json" not in ct:
                return
            if not INTERESTING_HOST.search(resp.url):
                return
            try:
                body = await resp.json()
            except Exception:
                body = await resp.text()
            log.append({
                "url": resp.url,
                "status": resp.status,
                "method": resp.request.method,
                "req_headers": dict(resp.request.headers),
                "body": body,
            })
        except Exception as e:
            log.append({"url": getattr(resp, "url", "?"), "error": str(e)})

    page.on("response", on_response)

    print(f"[{page_name}] navigating {url}")
    try:
        await page.goto(url, wait_until="networkidle", timeout=60000)
    except Exception as e:
        print(f"[{page_name}] goto warning: {e}")
    # let lazy XHRs fire
    await page.wait_for_timeout(6000)

    # Try to surface any clickable tournament/event links for the next stage.
    links = await page.eval_on_selector_all(
        "a[href]",
        "els => els.map(e => e.getAttribute('href')).filter(Boolean)",
    )
    tourney_links = sorted({
        l for l in links
        if re.search(r"tourn|event|result|match|bracket|draw", l, re.I)
    })

    out = OUT / f"{page_name}.json"
    out.write_text(json.dumps({
        "url": url,
        "json_responses": log,
        "tourney_links": tourney_links,
    }, indent=2, default=str))
    print(f"[{page_name}] captured {len(log)} JSON responses -> {out}")
    print(f"[{page_name}] {len(tourney_links)} candidate tournament links")

    await context.close()
    return {"page": page_name, "n_json": len(log), "links": tourney_links,
            "endpoints": sorted({r['url'].split('?')[0] for r in log if 'url' in r})}


async def main():
    summary = []
    async with async_playwright() as pw:
        exe = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        browser = await pw.chromium.launch(
            headless=True,
            executable_path=exe if os.path.exists(exe) else None,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
            proxy={"server": proxy} if proxy else None,
        )
        for name, url in TARGETS:
            try:
                summary.append(await probe(name, url, browser))
            except Exception as e:
                print(f"[{name}] FAILED: {e}")
                summary.append({"page": name, "error": str(e)})
        await browser.close()
    (OUT / "_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== ENDPOINTS SEEN ===")
    for s in summary:
        print(f"\n## {s.get('page')}")
        for ep in s.get("endpoints", []):
            print("  ", ep)


if __name__ == "__main__":
    asyncio.run(main())
