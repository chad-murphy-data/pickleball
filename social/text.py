"""Post text for a slate: title, markdown body, per-image captions.

    python social/text.py --date 2026-09-06     # prints title + body, writes post.md

Deliberately flat — numbers and names, one framing line, no voice.  The
same file feeds Reddit (post_reddit.py) and any other outlet later.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "social"))
from render import (SITE_URL, deck_round, fmt_pct, oriented, pretty_date,   # noqa: E402
                    round_word, short_event, side_long)
from slate import OUT, parse_date                                        # noqa: E402


def match_line(row: dict) -> str:
    fav, dog, p = oriented(row)
    rd = round_word(row["round"]) if row["round"] else ""
    tail = f" · {row['start']}" if row.get("start") else ""
    if row.get("branches"):
        alts = "; ".join(f"vs {side_long(b)} {fmt_pct(b['p_known'])}" for b in row["branches"])
        return f"- {rd}: **{side_long(fav)}** vs TBD — {alts}{tail}"
    if p is None:
        return f"- {rd}: {side_long(fav)} vs {side_long(dog)} — not priced ({row.get('note') or 'n/a'}){tail}"
    line = f"- {rd}: **{side_long(fav)} {fmt_pct(p)}** vs {side_long(dog)} {fmt_pct(1 - p)}"
    extra = []
    if row.get("modal"):
        a, b = row["modal"].split("-")
        extra.append(f"modal game {a}-{b}" if fav is row["t1"] else f"modal game {b}-{a}")
    if row.get("p_db") is not None:
        pdb = row["p_db"] if fav is row["t1"] else 1 - row["p_db"]
        extra += [f"DreamBreaker {fmt_pct(pdb)}", row["format"]]
    if row.get("start"):
        extra.append(row["start"])
    return line + (f" ({', '.join(extra)})" if extra else "")


def build_text(slate: dict) -> dict:
    d = dt.date.fromisoformat(slate["date"])
    events = [short_event(e) for e in slate["events"]]
    rw = deck_round(slate)
    when = d.strftime("%A %b %-d")
    title = f"{', '.join(events)} — {when} win probabilities" + (f" ({rw.lower()})" if rw else "")

    lines = [f"Pre-match win probabilities for every scheduled pro match on {when}, "
             f"from the PICKLES model. Posted before first serve, not updated after; "
             f"graded on the site.", ""]
    for s in slate["slates"]:
        venue = f" · {s['venue']}" if s.get("venue") else ""
        lines.append(f"**{s['title']}** — {short_event(s['event'])}{venue}")
        lines.append("")
        lines += [match_line(r) for r in s["rows"]]
        lines.append("")
    lines += [f"Method in the last slide. Ratings, the live board and the receipts ledger: "
              f"https://{SITE_URL}/", "",
              "Numbers never show as 0% or 100%: roughly 1 in 100 favorites priced at 99% "
              "still lose, so 98.9% is the ceiling."]
    return {"title": title[:300], "body": "\n".join(lines)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="tomorrow")
    args = ap.parse_args()
    d = str(parse_date(args.date))
    slate = json.loads((OUT / d / "slate.json").read_text())
    t = build_text(slate)
    (OUT / d / "post.md").write_text(f"# {t['title']}\n\n{t['body']}\n")
    print(t["title"]); print(); print(t["body"])


if __name__ == "__main__":
    main()
