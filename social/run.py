"""One command: slate -> slides -> text -> (optionally) Reddit.

    python social/run.py                       # tomorrow, render + text only
    python social/run.py --date today
    python social/run.py --post                # ...and post to r/<SOCIAL_SUBREDDIT>
    python social/run.py --post --dry-run      # everything except the submit

Exit 0 with "nothing scheduled" when the date has no pro matches — the
nightly workflow runs every evening and this is the quiet-day path.  It
also refuses to post a deck in which nothing could be priced (every side
TBD or unrated), so a bracket that isn't published yet never produces an
empty carousel.  Outputs land in social/out/<date>/ (gitignored; the
workflow uploads them as an artifact).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "social"))
import slate as S            # noqa: E402
import render as R           # noqa: E402
from text import build_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="tomorrow", help="YYYY-MM-DD | today | tomorrow")
    ap.add_argument("--post", action="store_true", help="submit to Reddit")
    ap.add_argument("--dry-run", action="store_true", help="with --post: print instead of submit")
    ap.add_argument("--no-cover", action="store_true")
    ap.add_argument("--min-priced", type=int, default=1,
                    help="refuse to post unless at least this many matches carry a number")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    d = S.parse_date(args.date)
    sl = S.build(d)
    out = S.OUT / str(d)
    out.mkdir(parents=True, exist_ok=True)
    (out / "slate.json").write_text(json.dumps(sl, indent=1))
    if not sl["slates"]:
        print(f"{d}: nothing scheduled — no post")
        return 0
    print(f"{d}: {sl['n_matches']} matches, {sl['n_priced']} priced across "
          f"{len(sl['slates'])} bracket(s): {', '.join(sl['events'])}")

    deck = R.render(str(d), keep_html=False, cover=not args.no_cover)
    t = build_text(sl)
    (out / "post.md").write_text(f"# {t['title']}\n\n{t['body']}\n")
    print(f"rendered {len(deck['slides'])} slides + post.md -> {out}")

    if not args.post:
        return 0
    if sl["n_priced"] < args.min_priced:
        print(f"only {sl['n_priced']} priced (< {args.min_priced}) — not posting")
        return 0
    from post_reddit import post
    post(str(d), dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
