"""Build current MLP DreamBreaker rosters (2W + 2M per team), UUID-keyed.

For every MLP franchise that played in the window 2026-07-06 .. 2026-07-22
(covers MLP San Diego 7/16-7/19 and the Edward Jones Mid-Season Tournament
7/8-7/12), take the team's MOST RECENT completed matchup and read its
lineups: the WD match's pair = the team's two women, the MD match's pair =
its two men. UUIDs are identity (lowercased); names are display only.

Values ("PICKLE singles score"):
  * ranked  = >= 10 pro singles games -> fit_singles value as-is
  * non-ranked -> corrected imputation  -0.07 + 1.14 * doubles
    (0.28 fitted intercept minus the 0.35 DreamBreaker correction,
     model/db_impute.md)

Exhibition sides (College All-Stars, Team Australia/Canada/Europe) appear
only at the mid-season event and are flagged franchise=0 (excluded from
league simulations by default).

Output: data/db_rosters.csv
Run:    python model/build_db_rosters.py     (cached; network on first run)
"""
from __future__ import annotations

import csv
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scraper"))
from pb_api import PBClient                                   # noqa: E402
from harvest import is_mlp_league                              # noqa: E402

AS_OF = date(2026, 7, 22)
WINDOW_DAYS = 16
IMPUTE_A, IMPUTE_B = -0.07, 1.14      # corrected imputation (db_impute.md)
RANKED_MIN_GAMES = 10
EXHIBITION = {"College All-Stars", "Team Australia", "Team Canada",
              "Team Europe"}
OUT = ROOT / "data" / "db_rosters.csv"


def load_players():
    sv, sg, gen, dbl, names = {}, {}, {}, {}, {}
    with open(ROOT / "data" / "singles_players.csv") as fh:
        for r in csv.DictReader(fh):
            u = r["player_id"].lower()
            sv[u] = float(r["singles_value"]); sg[u] = int(r["singles_games"])
            gen[u] = r["gender"]; names[u] = r["full_name"]
    with open(ROOT / "data" / "v2_players.csv") as fh:
        for r in csv.DictReader(fh):
            u = r["player_id"].lower()
            dbl[u] = float(r["value_now_mean"])
            gen.setdefault(u, r["gender"]); names.setdefault(u, r["full_name"])
    return sv, sg, gen, dbl, names


def discover_matchups(c):
    """All MLP matchups in the window: {matchup_uuid: date}."""
    seen = {}
    for i in range(WINDOW_DAYS + 1):
        d = AS_OF - timedelta(days=i)
        for tl in c.team_leagues_on_date(d):
            if not is_mlp_league(tl):
                continue
            for div in tl.get("divisions") or []:
                try:
                    mus = c.tl_matchups_short(tl, div, d)
                except PermissionError:
                    continue
                for mu in mus or []:
                    u = (mu.get("uuid") or "").lower()
                    if u and (mu.get("matchupStatus") or "") \
                            == "COMPLETED_MATCHUP_STATUS":
                        seen[u] = max(seen.get(u, str(d)), str(d))
    return seen


def main():
    sv, sg, gen, dbl, names = load_players()
    c = PBClient()
    matchups = discover_matchups(c)
    print(f"{len(matchups)} completed MLP matchups in window "
          f"{AS_OF - timedelta(days=WINDOW_DAYS)} .. {AS_OF}")

    # most recent completed matchup per franchise
    latest = {}      # team_uuid -> (date, matchup_uuid, title, event)
    for muid, d in matchups.items():
        md = c.matchup_data(muid, volatile=False)
        if md.get("matchupCompletedType") != "PLAYED_MATCHUP_COMPLETION_TYPE":
            continue
        event = md.get("matchupGroupTitle") or ""
        for tu, tt in (((md.get("teamOneUuid") or "").lower(),
                        md.get("teamOneTitle")),
                       ((md.get("teamTwoUuid") or "").lower(),
                        md.get("teamTwoTitle"))):
            if not tu:
                continue
            if tu not in latest or d > latest[tu][0]:
                latest[tu] = (d, muid, tt, event)

    rows, problems = [], []
    for tu, (d, muid, title, event) in sorted(latest.items(),
                                              key=lambda kv: kv[1][2] or ""):
        md = c.matchup_data(muid, volatile=False)
        side = "One" if (md.get("teamOneUuid") or "").lower() == tu else "Two"
        got = {}
        for m in md.get("matches") or []:
            raw_ab = m.get("matchAbbreviation") or ""
            # bracket rounds suffix the abbreviation (WDG/MDG in the Gold
            # final); mixed games start with MX and are excluded
            ab = ("WD" if raw_ab.startswith("WD")
                  else "MD" if raw_ab.startswith("MD") else None)
            if ab is None or m.get("isTieBreaker"):
                continue
            pair = [(m.get(f"team{side}PlayerOneUuid") or "").lower(),
                    (m.get(f"team{side}PlayerTwoUuid") or "").lower()]
            got[ab] = [u for u in pair if u]
        if sorted(got) != ["MD", "WD"] or len(got["WD"]) != 2 \
                or len(got["MD"]) != 2:
            problems.append(f"{title}: incomplete lineups in {muid[:8]}")
            continue
        for ab, want_g in (("WD", "F"), ("MD", "M")):
            us = sorted(got[ab], key=lambda u: -(
                sv[u] if u in sv and sg[u] >= RANKED_MIN_GAMES
                else (IMPUTE_A + IMPUTE_B * dbl[u]) if u in dbl else -99))
            for rank, u in enumerate(us, 1):
                g = gen.get(u, "?")
                if g != want_g:
                    problems.append(f"{title}: {names.get(u, u)} gender {g} "
                                    f"in {ab}")
                ranked = u in sv and sg.get(u, 0) >= RANKED_MIN_GAMES
                if ranked:
                    vu, src = sv[u], "singles"
                elif u in dbl:
                    vu, src = IMPUTE_A + IMPUTE_B * dbl[u], "imputed"
                else:
                    vu, src = None, "MISSING"
                    problems.append(f"{title}: {names.get(u, u)} has no "
                                    "singles or doubles rating")
                rows.append({
                    "team_uuid": tu, "team_title": title,
                    "franchise": 0 if title in EXHIBITION else 1,
                    "event": event, "lineup_date": d, "matchup_uuid": muid,
                    "player_uuid": u, "full_name": names.get(u, "?"),
                    "gender": want_g, "slot": f"{want_g}{rank}",
                    "singles_games": sg.get(u, 0),
                    "singles_value_raw": f"{sv[u]:.4f}" if u in sv else "",
                    "doubles_value": f"{dbl[u]:.4f}" if u in dbl else "",
                    "value_used": f"{vu:.4f}" if vu is not None else "",
                    "value_source": src,
                })

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    nteams = len({r['team_uuid'] for r in rows})
    print(f"wrote {OUT.name}: {nteams} teams, {len(rows)} player rows")
    for t in sorted({(r['team_title'], r['event'], r['lineup_date'])
                     for r in rows}):
        print(f"  {t[0]:26s} {t[2]}  ({t[1]})")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  !", p)
    else:
        print("no problems: every lineup complete, genders consistent, "
              "every player valued")


if __name__ == "__main__":
    main()
