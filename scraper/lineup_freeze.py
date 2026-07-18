"""Auto-freeze matchup forecasts at lineup announcement (Tier 2, kept light).

Watches today's scheduled MLP matchups; the first time a matchup's four
games all have announced lineups (while it is still SCHEDULED — strictly
pre-match), reprices it with the ACTUAL pairings using the same machinery
as make_forecast (values + weakest link + race DP + singles DreamBreaker +
display calibration) and appends one JSON line to
live/lineup_freezes-YYYYMMDD.jsonl:

    {"frozen_at": UTC iso, "matchup_uuid", "date", "start", "event",
     "team1", "team2", "games": [{slot, t1_pair, t2_pair, t1_uuids,
     t2_uuids, p, modal}], "p_db", "tree"}

Companion to live_poller.py on the droplet (run_freeze.sh commits the day's
file). These are timestamped pre-match records with real lineups — the
"lineups official" tier the projected-lineup ledger can be compared
against. Grading them stays a human/analysis step; this job only writes
down what it saw, when it saw it.

Usage:
    python scraper/lineup_freeze.py               # watch until day's slate resolves
    python scraper/lineup_freeze.py --once        # single sweep, print, exit
    python scraper/lineup_freeze.py --interval 90 # seconds between sweeps (min 30)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scraper"))
sys.path.insert(0, str(ROOT / "web"))

from live_poller import TOUR_TZ, is_mlp_league          # noqa: E402
from pb_api import PBClient                             # noqa: E402
from make_forecast import (db_win_prob, load_singles, load_values,  # noqa: E402
                           match_slot, matchup_lineups, matchup_tree,
                           price_game, SLOTS)

log = logging.getLogger("freeze")
OUT = ROOT / "live"


def frozen_already(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            out.add(json.loads(line)["matchup_uuid"])
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def sweep(c: PBClient, today: str, vals, singles, done: set[str], path: Path):
    """One pass over today's MLP matchups; freeze any newly-complete lineups.
    Returns (n_frozen_now, n_awaiting) — awaiting = scheduled without full
    lineups yet (the reason to keep watching)."""
    frozen_now = awaiting = 0
    tls = c._get_json(
        f"/api/v2/results/getTeamLeaguesResultsOnDate?date={today}").get("data") or []
    for tl in tls:
        if not is_mlp_league(tl):
            continue
        for div in tl.get("divisions") or []:
            q = (f"teamLeagueId={tl['uuid']}&organizationId={tl['organizationUuid']}"
                 f"&divisionId={div['divisionUuid']}&seasonId={div['seasonUuid']}"
                 f"&districtId={div['districtUuid']}&date={today}")
            if div.get("matchupGroupUuid"):
                q += f"&matchupGroupUuid={div['matchupGroupUuid']}"
            mus = c._get_json(
                f"/api/v2/results/getTeamLeaguesMatchupsShortOnDivision?{q}").get("data") or []
            for mu in mus:
                status = mu.get("matchupStatus") or ""
                uuid = (mu.get("uuid") or "").lower()
                if not status.startswith("SCHEDULED") or uuid in done:
                    continue
                md = c._get_json(
                    f"/api/v2/results/getResultsMatchupData?matchupId={mu['uuid']}"
                ).get("data") or {}
                lus = matchup_lineups(md)
                if len(lus) < len(SLOTS):
                    awaiting += 1
                    continue
                games, ps = [], []
                priceable = True
                for slot in SLOTS:
                    pair = lus[slot]
                    g = price_game(pair["One"], pair["Two"], vals)
                    if not g:
                        priceable = False
                        break
                    ps.append(g["p"])
                    games.append({
                        "slot": slot, **g,
                        "t1_pair": [vals[u][0] for u in pair["One"]],
                        "t2_pair": [vals[u][0] for u in pair["Two"]],
                        "t1_uuids": pair["One"], "t2_uuids": pair["Two"],
                    })
                if not priceable:
                    awaiting += 1        # untracked player; retry (roster fix?)
                    continue
                r1 = {u for p in lus.values() for u in p["One"]}
                r2 = {u for p in lus.values() for u in p["Two"]}
                p_db = db_win_prob(r1, r2, vals, singles)
                tree = matchup_tree(ps, p_db)
                tree["p_db_win"] = round(p_db, 4)
                rec = {
                    "frozen_at": dt.datetime.now(dt.timezone.utc)
                                   .isoformat(timespec="seconds"),
                    "matchup_uuid": uuid,
                    "date": today,
                    "start": mu.get("plannedStartDate"),
                    "event": tl.get("title"),
                    "team1": mu.get("teamOneTitle"), "team2": mu.get("teamTwoTitle"),
                    "games": games,
                    "tree": {k: round(v, 4) for k, v in tree.items()},
                }
                with path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                done.add(uuid)
                frozen_now += 1
                log.info("FROZE %s vs %s at lineups-official: %s %.0f%%",
                         rec["team1"], rec["team2"],
                         rec["team1"] if tree["p_win"] >= .5 else rec["team2"],
                         100 * max(tree["p_win"], 1 - tree["p_win"]))
    return frozen_now, awaiting


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=90)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.interval < 30:
        raise SystemExit("interval below 30s is impolite; refusing")

    today = dt.datetime.now(TOUR_TZ).date().isoformat()
    OUT.mkdir(exist_ok=True)
    path = OUT / f"lineup_freezes-{today.replace('-', '')}.jsonl"
    done = frozen_already(path)
    vals, singles = load_values(), load_singles()
    c = PBClient()

    idle = 0
    while True:
        try:
            n, awaiting = sweep(c, today, vals, singles, done, path)
            if n:
                idle = 0
            elif not awaiting:
                idle += 1
        except Exception:
            log.exception("sweep failed; continuing")
        if args.once:
            break
        # nothing scheduled without lineups for ~30 min → day resolved
        if idle * args.interval > 1800:
            log.info("no matchups awaiting lineups for 30 min — done for today")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
