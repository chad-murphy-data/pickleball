"""Post a rendered deck to Reddit as a gallery with the slate text as body.

    python social/post_reddit.py --date 2026-09-06 --dry-run   # print what would post
    python social/post_reddit.py --date 2026-09-06             # post it

Auth = a Reddit "script" app (reddit.com/prefs/apps) + the posting
account's login, all from the environment:

    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD
    REDDIT_USER_AGENT   (optional; default below)
    SOCIAL_SUBREDDIT    (default PickleballStats — no r/ prefix)
    SOCIAL_FLAIR_ID     (optional post flair template id)

Idempotent per date: the deck's date is stamped into the post title, and
the poster refuses to submit a second time if the account already has a
post with that title in the subreddit (a re-run of the workflow is safe).
Gallery posts carry body text natively (PRAW 8 `submit(gallery=...,
selftext=...)`); if Reddit ever rejects the text it falls back to a
top-level comment, which a moderator account then pins.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "social"))
from slate import OUT, parse_date       # noqa: E402
from text import build_text             # noqa: E402

DEFAULT_UA = "pickles-forecast-bot/1.0 (r/PickleballStats daily win probabilities)"


def reddit_client():
    import praw
    missing = [k for k in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET",
                           "REDDIT_USERNAME", "REDDIT_PASSWORD") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"missing env: {', '.join(missing)}")
    return praw.Reddit(client_id=os.environ["REDDIT_CLIENT_ID"],
                       client_secret=os.environ["REDDIT_CLIENT_SECRET"],
                       username=os.environ["REDDIT_USERNAME"],
                       password=os.environ["REDDIT_PASSWORD"],
                       user_agent=os.environ.get("REDDIT_USER_AGENT", DEFAULT_UA))


def already_posted(reddit, sub: str, title: str):
    me = reddit.user.me()
    for s in me.submissions.new(limit=50):
        if s.subreddit.display_name.lower() == sub.lower() and s.title == title:
            return s
    return None


def post(date_iso: str, dry_run: bool = False) -> dict:
    out_dir = OUT / date_iso
    slate = json.loads((out_dir / "slate.json").read_text())
    deck = json.loads((out_dir / "deck.json").read_text())
    text = build_text(slate)
    sub = os.environ.get("SOCIAL_SUBREDDIT", "PickleballStats")
    images = [{"path": str(out_dir / s["png"]), "caption": s["caption"][:180]}
              for s in deck["slides"]]
    if dry_run:
        print(f"[dry-run] r/{sub}: {text['title']}")
        for im in images:
            print(f"   {Path(im['path']).name}: {im['caption']}")
        print(); print(text["body"])
        return {"dry_run": True, "title": text["title"], "n_images": len(images)}

    from praw.models import PostMedia
    reddit = reddit_client()
    dup = already_posted(reddit, sub, text["title"])
    if dup:
        print(f"already posted: {dup.shortlink}")
        return {"url": dup.shortlink, "duplicate": True}
    gallery = [{"media": PostMedia(im["path"]), "caption": im["caption"]} for im in images]
    kwargs = {"gallery": gallery, "selftext": text["body"], "send_replies": False}
    if os.environ.get("SOCIAL_FLAIR_ID"):
        kwargs["flair_id"] = os.environ["SOCIAL_FLAIR_ID"]
    subreddit = reddit.subreddit(sub)
    try:
        submission = subreddit.submit(text["title"], **kwargs)
        body_where = "selftext"
    except Exception as e:                                    # noqa: BLE001
        # Reddit occasionally refuses body text on media posts; keep the
        # text visible as a pinned top comment instead
        print(f"gallery with text failed ({e}); retrying without body")
        kwargs.pop("selftext")
        submission = subreddit.submit(text["title"], **kwargs)
        c = submission.reply(text["body"])
        try:
            c.mod.distinguish(sticky=True)
        except Exception:                                     # noqa: BLE001
            pass
        body_where = "comment"
    rec = {"url": submission.shortlink, "id": submission.id, "body": body_where,
           "title": text["title"], "n_images": len(images)}
    (out_dir / "posted.json").write_text(json.dumps(rec, indent=1))
    print(f"posted: {submission.shortlink}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="tomorrow")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    post(str(parse_date(args.date)), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
