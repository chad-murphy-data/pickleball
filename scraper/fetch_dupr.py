"""Fetch REAL current DUPR doubles ratings from dupr.com's public rankings.

Why this exists: the per-match "rating" embedded in pickleball.com's BFF
(scraper/extract_ratings.py -> data/platform_ratings.csv) DIVERGED from
real DUPR around 2026-05-22 and got stuck ~0.3-0.7 low (e.g. Ben Johns
frozen at 6.56772 while DUPR.com shows 7.121).  So that field is NOT a
trustworthy DUPR copy anymore.  This grabs the real numbers straight from
the source.

The dupr.com/rankings page is a Webflow site whose ranking tables are a
published CMS collection (no API/auth needed).  The four main tabs are, in
document order: Men's Doubles, Women's Doubles, Men's Singles, Women's
Singles.  We take the two DOUBLES tabs (blocks 0 and 1) and verify each
against a known anchor before trusting it.

Coverage: DUPR only publishes the TOP ~50 per discipline on the public
site.  Players outside the top-50, and any historical/as-of-match rating,
require the authenticated api.dupr.gg (login-gated) and are not available
here.

Run:  python scraper/fetch_dupr.py        # -> data/dupr_doubles.csv
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
URL = "https://www.dupr.com/rankings"
# sanity anchors: block 0 must be men's doubles, block 1 women's doubles
ANCHOR = {0: "Ben Johns", 1: "Anna Leigh Waters"}


def fetch_html() -> str:
    r = httpx.get(URL, follow_redirects=True, timeout=30,
                  headers={"User-Agent": "Mozilla/5.0 (pickles-analytics; polite)"})
    r.raise_for_status()
    return r.text


def parse_blocks(html: str):
    """Return the ordered list-blocks, each a list of (name, rating)."""
    opens = [m.start() for m in re.finditer(r"post_list ranking-collection w-dyn-items", html)]
    bounds = opens + [len(html)]
    blocks = []
    for i in range(len(opens)):
        seg = html[bounds[i]:bounds[i + 1]]
        items = []
        for it in re.split(r"post_item w-dyn-item", seg)[1:]:
            nm = re.search(r'heading-table name">([^<]+)</div>', it)
            rt = re.search(r'heading-table right">([^<]+)</div>', it)
            if nm and rt:
                items.append((nm.group(1).strip(), rt.group(1).strip()))
        blocks.append(items)
    return blocks


def main():
    html = fetch_html()
    blocks = parse_blocks(html)
    if len(blocks) < 2:
        sys.exit(f"unexpected page structure: {len(blocks)} ranking blocks found")
    md, wd = blocks[0], blocks[1]
    for idx, want in ANCHOR.items():
        got = blocks[idx][0][0] if blocks[idx] else "(empty)"
        if got != want:
            sys.exit(f"SAFETY CHECK FAILED: block {idx} first player is {got!r}, "
                     f"expected {want!r} — DUPR may have restructured the page; "
                     f"inspect before trusting.")
    out = DATA / "dupr_doubles.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["discipline", "rank", "name", "dupr"])
        for i, (nm, rt) in enumerate(md):
            w.writerow(["MD", i + 1, nm, rt])
        for i, (nm, rt) in enumerate(wd):
            w.writerow(["WD", i + 1, nm, rt])
    print(f"verified against anchors; wrote {out}")
    print(f"  men's doubles:   {len(md)} players (top {len(md)}), #1 {md[0][0]} {md[0][1]}")
    print(f"  women's doubles: {len(wd)} players (top {len(wd)}), #1 {wd[0][0]} {wd[0][1]}")
    print("  NOTE: public page = top-50 only; below that + history needs the login-gated API.")


if __name__ == "__main__":
    main()
