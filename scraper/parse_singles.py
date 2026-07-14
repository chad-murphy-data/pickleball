"""Parse PPA pro singles matches from raw/match_infos_short_singles/.

    python scraper/parse_singles.py     # -> data/singles_games.csv

Singles were added to the harvest later than doubles and live in their own
cache dir (see pb_api.match_infos_short_singles). Same payload shape as
doubles match_infos_short, with one player per side. Rules mirror parse.py:
match_status 4 = completed, match_completed_type != 5 = forfeit, UUIDs
lowercased, sanity floor winner >= 11 & win-by-2 when the format is unknown
(PPA singles main draws are best-of-3/5 to 11; Challenger to-15 leaks get
flagged, not silently kept).
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
DATA = ROOT / "data"

log = logging.getLogger("parse_singles")

GAME_ORDINALS = ["one", "two", "three", "four", "five"]
MATCH_COMPLETED_STATUS = 4
NORMAL_COMPLETION = 5


def clean_name(s):
    return " ".join((s or "").split())


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    mdir = RAW / "match_infos_short_singles"
    if not mdir.exists():
        log.error("no %s — run scraper/harvest.py first", mdir)
        return

    titles = {}
    tdir = RAW / "tournaments_on_date"
    for f in sorted(tdir.glob("*.json")) if tdir.exists() else []:
        for t in (json.loads(f.read_text()).get("data") or []):
            titles[t["TournamentID"]] = t["Title"]

    rows, dropped = [], 0
    seen = set()
    for tdir_ in sorted(mdir.iterdir()):
        tid = tdir_.name
        for f in sorted(tdir_.glob("*.json")):
            day = f.stem
            for m in json.loads(f.read_text()).get("data") or []:
                muid = (m.get("match_uuid") or "").lower()
                if not muid or muid in seen:
                    continue
                seen.add(muid)
                title = (m.get("event_title") or "")
                tl = title.lower()
                if "singles" not in tl:
                    continue
                context = ("womens_singles" if "women" in tl else
                           "mens_singles" if "men" in tl else None)
                if context is None or m.get("match_status") != MATCH_COMPLETED_STATUS:
                    dropped += 1
                    continue
                is_forfeit = (m.get("match_completed_type") != NORMAL_COMPLETION
                              or bool(m.get("did_cancel_match")))
                p1 = (m.get("team_one_player_one_uuid") or "").lower()
                p2 = (m.get("team_two_player_one_uuid") or "").lower()
                if not p1 or not p2:
                    dropped += 1
                    continue
                n1 = clean_name(m.get("team_one_player_one_name"))
                n2 = clean_name(m.get("team_two_player_one_name"))
                best_of = m.get("score_format_game_best_out_of") or 3
                draw = title.replace("Singles Pro", "").strip()
                stage = (m.get("round_text") or "").strip()
                for i, o in enumerate(GAME_ORDINALS, start=1):
                    s1 = m.get(f"team_one_game_{o}_score") or 0
                    s2 = m.get(f"team_two_game_{o}_score") or 0
                    if s1 == 0 and s2 == 0:
                        continue
                    winner, margin = max(s1, s2), abs(s1 - s2)
                    flag = ""
                    if not is_forfeit:
                        if winner < 11 or margin < 2:
                            dropped += 1
                            continue
                        if winner > 15:
                            flag = "score>15 — possible to-21/format leak"
                    rows.append({
                        "game_id": f"{muid}:g{i}",
                        "match_id": muid,
                        "event_id": tid,
                        "event_name": titles.get(tid, m.get("tournament_title") or tid),
                        "date": (m.get("match_start") or day)[:10],
                        "context": context,
                        "stage": stage or draw,
                        "p1": p1, "p2": p2,
                        "p1_name": n1, "p2_name": n2,
                        "s1": s1, "s2": s2,
                        "margin": s1 - s2,
                        "game_number": i,
                        "best_of": best_of,
                        "is_forfeit": is_forfeit,
                        "flag": flag,
                    })

    rows.sort(key=lambda r: (r["date"], r["match_id"], r["game_number"]))
    out = DATA / "singles_games.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ["game_id"])
        w.writeheader()
        w.writerows(rows)
    n_matches = len({r["match_id"] for r in rows})
    n_players = len({r["p1"] for r in rows} | {r["p2"] for r in rows})
    log.info("wrote %s: %d games / %d matches / %d players (%d dropped)",
             out, len(rows), n_matches, n_players, dropped)


if __name__ == "__main__":
    main()
