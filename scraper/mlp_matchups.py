"""Fetch MLP matchup structure (which games form which matchup, in order)
from the open BFF and write data/mlp_matchups_2026.csv.

games.csv stores MLP games as independent single-game matches with no
matchup id, so anything matchup-shaped (the awards' WPA, team records)
needs this table.  One row per DOUBLES slot of every completed matchup:

  matchup_id, event_id, date, team_one, team_two, game_slot (1-4, played
  order), match_id (joins games.csv), completed_type (5 = played,
  6 = walkover/dead -- absent from games.csv, advances the matchup score
  with nobody credited), winner_side (1/2 relative to team_one/team_two).

DreamBreakers (Singles format) are dropped here like everywhere else.
Responses are cached in raw/mlp_matchups/ (gitignored); re-runs only hit
the network for missing files.  ~1 req/s, same etiquette as harvest.py.

  python scraper/mlp_matchups.py
"""
import csv
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "raw" / "mlp_matchups"
OUT = ROOT / "data" / "mlp_matchups_2026.csv"
BASE = "https://pickleball.com"
SEASON = "2026-01-01"


def get(url: str, path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    req = urllib.request.Request(url, headers={"accept": "application/json",
                                               "user-agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            body = json.load(urllib.request.urlopen(req, timeout=30))
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    time.sleep(1.0)
    return body


def main():
    ev_ids, dates = set(), set()
    for r in csv.DictReader(open(ROOT / "data" / "games.csv")):
        if r["tour"] == "MLP" and r["date"] >= SEASON:
            ev_ids.add(r["event_id"].lower())
            dates.add(r["date"])

    matchups = {}
    for d in sorted(dates):
        day = get(f"{BASE}/api/v2/results/getTeamLeaguesResultsOnDate?date={d}",
                  CACHE / f"date_{d}.json")
        for e in day.get("data") or []:
            if e.get("slug") != "major-league-pickleball":
                continue
            org = e.get("organizationUuid") or e.get("organizationId")
            for dv in e.get("divisions") or []:
                if (dv.get("matchupGroupUuid") or "").lower() not in ev_ids:
                    continue
                q = (f"teamLeagueId={e['uuid']}&organizationId={org}"
                     f"&divisionId={dv['divisionUuid']}&seasonId={dv['seasonUuid']}"
                     f"&districtId={dv['districtUuid']}&date={d}"
                     f"&matchupGroupUuid={dv['matchupGroupUuid']}")
                s = get(f"{BASE}/api/v2/results/getTeamLeaguesMatchupsShortOnDivision?{q}",
                        CACHE / f"short_{e['uuid'][:8]}_{dv['divisionUuid'][:8]}_{d}.json")
                for row in s.get("data") or []:
                    uid = row["uuid"].lower()
                    if "COMPLETED" in (row.get("matchupStatus") or ""):
                        matchups.setdefault(uid, {"date": d,
                                                  "event": dv["matchupGroupUuid"].lower()})

    out = []
    for uid, meta in sorted(matchups.items(), key=lambda kv: kv[1]["date"]):
        m = get(f"{BASE}/api/v2/results/getResultsMatchupData?matchupId={uid}",
                CACHE / f"mu_{uid}.json").get("data") or {}
        if "COMPLETED" not in (m.get("matchupStatus") or ""):
            continue
        rows = [g for g in m.get("matches") or [] if g.get("formatTitle") == "Doubles"]
        rows.sort(key=lambda g: (g.get("matchStart")
                                 or g.get("matchPlannedStart") or "9999"))
        for slot, g in enumerate(rows, 1):
            ctype = g.get("matchCompletedType")
            if ctype == 14:          # scheduled slot that was never contested
                continue
            out.append({
                "matchup_id": uid,
                "event_id": meta["event"],
                "date": meta["date"],
                "team_one": m.get("teamOneTitle") or "",
                "team_two": m.get("teamTwoTitle") or "",
                "game_slot": slot,
                "match_id": (g.get("matchUuid") or "").lower(),
                "completed_type": ctype,
                "winner_side": g.get("winner") or 0,
            })

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    n_mu = len({r['matchup_id'] for r in out})
    print(f"{OUT.name}: {len(out)} game rows, {n_mu} matchups")


if __name__ == "__main__":
    main()
