"""Resolve exact score formats for ambiguous PPA matches.

getMatchInfosShort doesn't carry the score-format detail (points target,
rally vs side-out). That matters: PPA Challengers play early rounds as a
single game to 15 (side-out), while main draws are best-of-3 to 11 — the
margin scale differs and mislabeling would poison the model.

Rule: fetch the full per-match record (getResultMatchInfos) for
  * every best-of-1 match,
  * every match with a game winner score >= 14 (deuce can't exceed 13-11
    in a clean to-11 game... it can, via win-by-2 — which is exactly why
    we look it up instead of guessing),
  * a deterministic sample of standard Bo3/Bo5 matches per tournament,
    to verify the side-out-to-11 default empirically.

Cached at raw/result_match_infos/{match_uuid}.json — immutable.

Run after harvest.py:  python scraper/enrich_formats.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pb_api import RAW, PBClient

log = logging.getLogger("enrich")

SAMPLE_STANDARD_PER_TOURNAMENT = 8


def needs_lookup(m: dict) -> bool:
    if (m.get("score_format_game_best_out_of") or 3) == 1:
        return True
    for g in ("one", "two", "three", "four", "five"):
        s1 = m.get(f"team_one_game_{g}_score") or 0
        s2 = m.get(f"team_two_game_{g}_score") or 0
        if max(s1, s2) >= 14:
            return True
    return False


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    c = PBClient()
    out = RAW / "result_match_infos"
    out.mkdir(parents=True, exist_ok=True)

    targets: set[str] = set()
    mdir = RAW / "match_infos_short"
    for tdir in sorted(mdir.iterdir()) if mdir.exists() else []:
        standard = []
        for f in sorted(tdir.glob("*.json")):
            for m in json.loads(f.read_text()).get("data") or []:
                muid = m.get("match_uuid")
                if not muid:
                    continue
                if m.get("match_status") != 4:
                    continue
                if needs_lookup(m):
                    targets.add(muid)
                else:
                    standard.append(muid)
        # deterministic verification sample of "standard" matches
        for muid in sorted(standard)[:SAMPLE_STANDARD_PER_TOURNAMENT]:
            targets.add(muid)

    todo = [t for t in sorted(targets) if not (out / f"{t}.json").exists()]
    log.info("format lookups needed: %d (of %d targets; rest cached)", len(todo), len(targets))
    for i, muid in enumerate(todo):
        try:
            body = c._get_json(f"/api/v1/results/getResultMatchInfos?id={muid}")
        except Exception as e:
            log.warning("lookup failed %s: %s", muid, e)
            continue
        (out / f"{muid}.json").write_text(json.dumps(body, indent=1))
        if (i + 1) % 50 == 0:
            log.info("…%d/%d", i + 1, len(todo))
    log.info("done: %d network calls", c.network_calls)


if __name__ == "__main__":
    main()
