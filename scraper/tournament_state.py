"""Distill the CURRENT tournament + season state into data/tournament_state.json.

    python scraper/tournament_state.py            # window = today-6 .. today+7

Feeds the site's title-race page (web/build_site.py:build_titlerace):

- MLP "mlp": the FEATURED event weekend — the earliest-starting event in the
  window that still has an unfinished matchup (so Monday morning it advances
  to the next weekend, while a live event always wins), falling back to the
  most recently finished one on a quiet week.  Completed matchups carry
  actual game scores and rally points; still-scheduled matchups of the same
  group carry pool/round metadata (published days ahead, so the pre-event
  sim runs on the real pool draw).  Standings math and the Monte Carlo live
  build-side; this file only records facts.
- MLP "mlp_recap": the most recently finished event when it isn't the
  featured one (the just-completed weekend, shown while the next is up).
- MLP "season": the season-long playoff race — standings points derived
  from each finished city stop's Super Sunday placement matchups, the
  remaining scoring events, the announced playoff structure, and a
  best-lineup matchup tree for every team pair ("matrix") so the build can
  simulate the rest of the season.
- PPA "ppa": every pro DOUBLES division with a main-draw match inside the
  window — per match: round, match number, seeds, players (UUIDs lowercased,
  house rule), winner, per-game scores, format.  "ppa_next": title + dates
  of the next PPA tournament in the lookahead (match lists don't exist until
  a day is underway, so it's a teaser only).

Season points: Super Sunday placement matchups pay 25/18 (matchup #1, the
pool winners' title matchup), 15/12 (#2), 10/8 (#3), 6/4 (#4); the best
team left out of Sunday in each pool takes 1 point; the Mid-Season
Tournament pays a podium bonus (10/6/4).  Standings through
SEASON_BASE_DATE are frozen to the published table (see the constant's
comment for why two 5th-place tiebreak cells can't be derived from match
records); later events derive from data.

Network use: the same polite cached client as the harvester (~1 req/s,
volatile last-3-days refetch).  Runs in the nightly refresh right after the
harvest, so almost everything is already cached; a quiet week costs a
handful of calls.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harvest import SEASON_START, is_mlp_league, is_ppa_tournament  # noqa: E402
from pb_api import MATCHUP_FINAL as FINAL, PBClient    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
log = logging.getLogger("tournament_state")

LOOKBACK = 6          # days behind today that still count as "this event"
LOOKAHEAD = 7         # days ahead: covers the next event weekend fully

SUNDAY_ROUND = "One and Done - Playoff"
SUNDAY_POINTS = {1: (25, 18), 2: (15, 12), 3: (10, 8), 4: (6, 4)}

# Season standings points through MLP Chicago (2026-07-26), reconciled to
# the league's published table (The Dink playoff guide and The Kitchen
# week-9 power rankings agree on every playoff-relevant team).  The Super
# Sunday placement points derive EXACTLY from matchup data (validated
# team-by-team); the Mid-Season podium bonus (champion 10 / runner-up 6 /
# third 4 — StL, NJ, Columbus) and the per-pool 5th-place single points are
# included, but two 5th-place tiebreak cells are NOT derivable from match
# records alone — MLP's published split contradicts head-to-head in one
# event (Dallas) and game win pct in another (St. Louis) — so the
# reconciled totals are frozen here and derivation resumes for events
# AFTER this date (a future ±1 on a 5th-place point is possible and
# harmless: it can only touch teams already out of the race).
SEASON_BASE_DATE = "2026-07-26"
SEASON_BASE_POINTS = {
    "New Jersey 5s": 93, "St. Louis Shock": 93, "Columbus Sliders": 83,
    "Dallas Flash": 72, "Los Angeles Mad Drops": 71,
    "Brooklyn Pickleball Team": 66, "Texas Ranchers": 48,
    "Palm Beach Royals": 44, "Atlanta Bouncers": 36,
    "SoCal Hard Eights": 34, "California Black Bears": 30,
    "Chicago Slice": 27, "Las Vegas Night Owls": 26, "Orlando Squeeze": 25,
    "Miami Pickleball Club": 20, "Utah Black Diamonds": 17,
    "Bay Area Breakers": 15, "Florida Smash": 15,
    "Phoenix Flames": 2, "Carolina Hogs": 2,
}

# Announced 2026 playoff plan (not in the scheduling system yet as of late
# July; the build labels it as announced).  Top 12 by standings points, seeds
# 1-4 skip the First Round; higher seeds choose their opponents each round.
PLAYOFFS = {
    "spots": 12, "byes": 4,
    "rounds": [
        {"name": "First Round", "location": "Dallas",
         "dates": ["2026-08-06", "2026-08-09"]},
        {"name": "Quarterfinals", "location": "Newport Beach",
         "dates": ["2026-08-14", "2026-08-16"]},
        {"name": "Championship Weekend", "location": "New York City",
         "dates": ["2026-08-28", "2026-08-30"]},
    ],
    "note": "announced schedule; playoff matchups not yet published",
}


def mlp_pair_matrix(teams, today, c):
    """Model-rated matchup tree for every unordered team pair, keyed
    "A|B" (sorted titles), tree oriented to the first title.  Reuses the
    forecast machinery — each team's BEST LINEUP from its season roster
    (last-matchup fallback; see make_forecast) — so simulated playoff
    pairings, which can be any cross-pool combination, carry real
    probabilities, not coin flips."""
    sys.path.insert(0, str(ROOT / "web"))
    import make_forecast as mf
    vals, singles = mf.load_values(), mf.load_singles()
    rosters = mf.mlp_rosters(c, today)
    cache, lineups = {}, {}
    for t in teams:
        lineups[t] = mf.projected_lineup_for_team(
            c, t, today, rosters, vals, cache)[0]
    matrix = {}
    for i, a in enumerate(sorted(teams)):
        for b in sorted(teams)[i + 1:]:
            la, lb = lineups.get(a), lineups.get(b)
            if not la or not lb:
                continue
            ps = []
            for slot in mf.SLOTS:
                g = mf.price_game(la.get(slot), lb.get(slot), vals)
                if g:
                    ps.append(g["p"])
            if len(ps) != 4:
                continue
            r1 = {u for pr in la.values() for u in pr}
            r2 = {u for pr in lb.values() for u in pr}
            p_db = mf.db_win_prob(r1, r2, vals, singles)
            tree = mf.matchup_tree(ps, p_db)
            tree["p_db_win"] = p_db
            matrix[f"{a}|{b}"] = {k: round(v, 4) for k, v in tree.items()}
    return matrix


def collect_mlp_events(c: PBClient, d0: date, d1: date) -> dict:
    """Every MLP event weekend with a matchup day in [d0, d1].  Each event
    surfaces as a matchup group on the season-long league's division; group
    by that UUID, falling back to the group TITLE when the UUID hasn't been
    attached to a future date yet (observed 2026-07-22)."""
    events: dict[str, dict] = {}
    d = d0
    while d <= d1:
        for tl in c.team_leagues_on_date(d):
            if not is_mlp_league(tl):
                continue
            for div in tl.get("divisions") or []:
                try:
                    mus = c.tl_matchups_short(tl, div, d)
                except PermissionError:
                    continue
                if not mus:
                    continue
                guid = (div.get("matchupGroupUuid") or "").lower()
                title = (mus[0].get("matchupGroupTitle")
                         or tl.get("title") or "").strip()
                key = guid or f"title:{title}"
                g = events.setdefault(key, {"group": guid, "title": None,
                                            "matchups": {}})
                g["group"] = g["group"] or guid
                for mu in mus:
                    g["title"] = mu.get("matchupGroupTitle") or g["title"]
                    g["matchups"][mu["uuid"].lower()] = {"date": str(d),
                                                         "short": mu}
        d += timedelta(days=1)
    return events


def event_span(g) -> tuple[str, str]:
    ds = sorted(r["date"] for r in g["matchups"].values())
    return ds[0], ds[-1]


def event_is_open(g) -> bool:
    return any((r["short"].get("matchupStatus") or "") not in FINAL
               for r in g["matchups"].values())


def event_rows(c: PBClient, g) -> tuple[list, list]:
    """(completed, remaining) row dicts for one event, detail-enriched."""
    completed, remaining = [], []
    for muid, rec in sorted(g["matchups"].items(), key=lambda kv: kv[1]["date"]):
        mu = rec["short"]
        status = mu.get("matchupStatus") or ""
        if status == "BYE_MATCHUP_STATUS":
            continue
        # detail carries what the short rows lack: pool, bracket stage,
        # matchup number, abbreviations, rally points (volatile until final)
        md = c.matchup_data(muid, volatile=status not in FINAL)
        row = {
            "uuid": muid,
            "date": rec["date"],
            "start": mu.get("plannedStartDate"),
            "team1": mu.get("teamOneTitle"), "team2": mu.get("teamTwoTitle"),
            "abbr1": md.get("teamOneAbbreviation"),
            "abbr2": md.get("teamTwoAbbreviation"),
            "pool": (md.get("poolUuid") or "")[:8],
            "bracket": md.get("inBracketType"),
            "round": md.get("roundText"),
            "mnum": md.get("matchupNumber"),
        }
        if status == "COMPLETED_MATCHUP_STATUS":
            if md.get("matchupCompletedType") != "PLAYED_MATCHUP_COMPLETION_TYPE":
                continue                     # walkover/cancelled: not a result
            row.update({
                "games1": md.get("teamOneScore"), "games2": md.get("teamTwoScore"),
                "pts1": md.get("teamOnePointsEarned"),
                "pts2": md.get("teamTwoPointsEarned"),
                "winner": md.get("winner"),
            })
            completed.append(row)
        elif "SCHEDULED" in status:
            remaining.append(row)
    return completed, remaining


def event_teams(rows) -> list[str]:
    return sorted({r[k] for r in rows for k in ("team1", "team2") if r[k]})


def is_scoring_event(completed, remaining) -> bool:
    """City stops run pool round robins; the Mid-Season double-elimination
    bracket has no RR rows and awards no standings points."""
    return any(r.get("bracket") == "RR" for r in completed + remaining)


def event_points(completed) -> dict[str, int]:
    """Standings points from a FINISHED city stop's rows.  Super Sunday
    placement matchups pay by matchup number; per pool, the best team left
    out of Sunday takes 1 point (5th place), any second one takes 0."""
    points: dict[str, int] = {}
    sunday_teams = set()
    for r in completed:
        if r.get("round") != SUNDAY_ROUND:
            continue
        pay = SUNDAY_POINTS.get(r.get("mnum") or 0)
        if not pay or r.get("winner") not in (1, 2):
            continue
        w, l = (r["team1"], r["team2"]) if r["winner"] == 1 else (r["team2"], r["team1"])
        points[w] = points.get(w, 0) + pay[0]
        points[l] = points.get(l, 0) + pay[1]
        sunday_teams.update((r["team1"], r["team2"]))
    if not points:
        return {}
    # pool records decide the left-out teams' 1/0 split
    pools: dict[str, set] = {}
    rec: dict[str, list] = {}
    for r in completed:
        if r.get("bracket") != "RR" or r.get("winner") not in (1, 2):
            continue
        pools.setdefault(r["pool"], set()).update((r["team1"], r["team2"]))
        for t in (r["team1"], r["team2"]):
            rec.setdefault(t, [0, 0, 0, 0])    # mw, ml, gw, gl
        w, l = (r["team1"], r["team2"]) if r["winner"] == 1 else (r["team2"], r["team1"])
        rec[w][0] += 1
        rec[l][1] += 1
        rec[r["team1"]][2] += r.get("games1") or 0
        rec[r["team1"]][3] += r.get("games2") or 0
        rec[r["team2"]][2] += r.get("games2") or 0
        rec[r["team2"]][3] += r.get("games1") or 0
    def pcts(t):
        mw, ml, gw, gl = rec.get(t, [0, 0, 0, 0])
        return (mw / (mw + ml) if mw + ml else 0.0,
                gw / (gw + gl) if gw + gl else 0.0)
    for pool_teams in pools.values():
        out = sorted((t for t in pool_teams if t not in sunday_teams),
                     key=pcts, reverse=True)
        for i, t in enumerate(out):
            points[t] = points.get(t, 0) + (1 if i == 0 else 0)
    return points


def season_state(c: PBClient, events: dict, rows_by_key: dict) -> dict:
    """Season-long playoff race: standings = the reconciled published base
    through SEASON_BASE_DATE plus derived points for later scoring events;
    records and event cards from data; announced playoff plan."""
    standings: dict[str, dict] = {}

    def row(t):
        return standings.setdefault(t, {"team": t, "points": 0, "events": 0,
                                        "mw": 0, "ml": 0, "gw": 0, "gl": 0})

    for t, p in SEASON_BASE_POINTS.items():
        row(t)["points"] = p
    counted, remaining = [], []
    for key, g in sorted(events.items(), key=lambda kv: event_span(kv[1])[0]):
        completed, sched = rows_by_key[key]
        if not is_scoring_event(completed, sched):
            continue
        first, last = event_span(g)
        if event_is_open(g):
            remaining.append({"event": g["title"], "group": g["group"],
                              "dates": [first, last],
                              "teams": event_teams(completed + sched)})
            continue
        pts = event_points(completed)
        if not pts:
            continue
        title_row = next((r for r in completed
                          if r.get("round") == SUNDAY_ROUND and r.get("mnum") == 1), None)
        champion = None
        if title_row and title_row.get("winner") in (1, 2):
            champion = title_row["team1"] if title_row["winner"] == 1 else title_row["team2"]
        counted.append({"event": g["title"], "group": g["group"],
                        "dates": [first, last], "champion": champion,
                        "points": pts})
        for r in completed:
            if r.get("winner") not in (1, 2):
                continue
            w, l = (r["team1"], r["team2"]) if r["winner"] == 1 else (r["team2"], r["team1"])
            row(w)["mw"] += 1
            row(l)["ml"] += 1
            row(r["team1"])["gw"] += r.get("games1") or 0
            row(r["team1"])["gl"] += r.get("games2") or 0
            row(r["team2"])["gw"] += r.get("games2") or 0
            row(r["team2"])["gl"] += r.get("games1") or 0
        for t, p in pts.items():
            if first > SEASON_BASE_DATE:      # base already covers earlier
                row(t)["points"] += p
            row(t)["events"] += 1
    # published tiebreak order: match win pct, then game win pct
    def order(s):
        mp = s["mw"] / (s["mw"] + s["ml"]) if s["mw"] + s["ml"] else 0.0
        gp = s["gw"] / (s["gw"] + s["gl"]) if s["gw"] + s["gl"] else 0.0
        return (-s["points"], -mp, -gp, s["team"])
    return {"standings": sorted(standings.values(), key=order),
            "events": counted, "remaining": remaining, "playoffs": PLAYOFFS}


def mlp_state(c: PBClient, today: date, events: dict, rows_by_key: dict):
    """(featured, recap) for the window; featured mirrors the historical
    "mlp" shape plus additive dates/status keys."""
    lo, hi = str(today - timedelta(days=LOOKBACK)), str(today + timedelta(days=LOOKAHEAD))
    window = {k: g for k, g in events.items()
              if event_span(g)[1] >= lo and event_span(g)[0] <= hi}
    if not window:
        return None, None
    open_evs = {k: g for k, g in window.items() if event_is_open(g)}
    if open_evs:
        fkey = min(open_evs, key=lambda k: event_span(open_evs[k])[0])
    else:  # quiet week: keep showing the most recently finished event
        fkey = max(window, key=lambda k: event_span(window[k])[1])
    def build(key, with_scores=True):
        g = window[key]
        completed, remaining = rows_by_key[key]
        if not completed and not remaining:
            return None
        first, last = event_span(g)
        status = ("upcoming" if not completed
                  else "final" if not remaining else "live")
        return {"group": g["group"], "event": g["title"],
                "dates": [first, last], "status": status,
                "completed": completed, "remaining": remaining}
    featured = build(fkey)
    recap = None
    final_keys = [k for k in window if k != fkey and not event_is_open(window[k])
                  and event_span(window[k])[1] <= str(today)]
    if featured and final_keys:
        rkey = max(final_keys, key=lambda k: event_span(window[k])[1])
        recap = build(rkey)
        if recap:
            recap.pop("remaining", None)
            recap.pop("status", None)
    return featured, recap


def ppa_state(c: PBClient, today: date):
    """Active PPA pro-doubles divisions with their main-draw matches."""
    tourneys: dict[str, dict] = {}
    for i in range(-LOOKBACK, LOOKAHEAD + 1):
        d = today + timedelta(days=i)
        if d > today:
            continue        # match lists exist only once a day is underway
        try:
            day_ts = c.tournaments_on_date(d)
        except Exception as e:                        # network hiccup: skip day
            log.warning("tournaments_on_date %s failed: %s", d, e)
            continue
        for t in day_ts:
            if not is_ppa_tournament(t):
                continue
            tid, title = t["TournamentID"], t["Title"]
            try:
                pro = [g for g in c.events_flat_group(tid, d)
                       if "pro" in g["group_title"].lower()
                       and "senior" not in g["group_title"].lower()
                       and "junior" not in g["group_title"].lower()]
                if not pro:
                    continue
                events = c.tournament_events_short(tid, pro[0], d)
                doubles = [e for e in events if "doubles" in e["title"].lower()]
                if not doubles:
                    continue
                matches = c.match_infos_short(tid, [e["uuid"] for e in doubles], d)
            except PermissionError:
                continue
            rec = tourneys.setdefault(tid.lower(), {
                "tournament": title, "dates": set(), "divisions": {}})
            rec["dates"].add(str(d))
            for m in matches:
                if not (m.get("round_text") or "").strip():
                    continue                 # consolation / qualifier rounds
                div = rec["divisions"].setdefault(m["event_title"], {})
                div[m["match_uuid"]] = {
                    "round": m.get("round_number"),
                    "round_text": (m.get("round_text") or "").strip(),
                    "match_no": m.get("match_number"),
                    "seed1": m.get("team_one_seed"), "seed2": m.get("team_two_seed"),
                    "best_of": m.get("score_format_game_best_out_of"),
                    "p1": [str(m.get("team_one_player_one_uuid") or "").lower(),
                           str(m.get("team_one_player_two_uuid") or "").lower()],
                    "p2": [str(m.get("team_two_player_one_uuid") or "").lower(),
                           str(m.get("team_two_player_two_uuid") or "").lower()],
                    "n1": [m.get("team_one_player_one_name", "").strip(),
                           m.get("team_one_player_two_name", "").strip()],
                    "n2": [m.get("team_two_player_one_name", "").strip(),
                           m.get("team_two_player_two_name", "").strip()],
                    "winner": m.get("winner"),
                    "completed_type": m.get("match_completed_type"),
                    "scores": [[m.get(f"team_one_game_{n}_score")
                                for n in ("one", "two", "three", "four", "five")],
                               [m.get(f"team_two_game_{n}_score")
                                for n in ("one", "two", "three", "four", "five")]],
                }
    out = []
    for tid, rec in tourneys.items():
        divisions = [{"title": k, "matches": sorted(v.values(),
                      key=lambda m: (m["round"] or 0, m["match_no"] or 0))}
                     for k, v in sorted(rec["divisions"].items()) if v]
        if divisions:
            out.append({"tournament": rec["tournament"], "id": tid,
                        "dates": sorted(rec["dates"]), "divisions": divisions})
    return out


def ppa_next(c: PBClient, today: date):
    """Title + dates of the nearest upcoming PPA tournament in the lookahead
    (a teaser: per-day match lists don't exist until a day is underway)."""
    found: dict[str, dict] = {}
    for i in range(1, LOOKAHEAD + 1):
        d = today + timedelta(days=i)
        try:
            day_ts = c.tournaments_on_date(d)
        except Exception as e:
            log.warning("tournaments_on_date %s failed: %s", d, e)
            continue
        for t in day_ts:
            if not is_ppa_tournament(t):
                continue
            rec = found.setdefault(str(t["TournamentID"]).lower(), {
                "tournament": t["Title"], "id": str(t["TournamentID"]).lower(),
                "dates": []})
            rec["dates"].append(str(d))
    if not found:
        return None
    nearest = min(found.values(), key=lambda r: r["dates"][0])
    return nearest


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    c = PBClient()
    today = date.today()
    events = collect_mlp_events(c, SEASON_START, today + timedelta(days=LOOKAHEAD))
    rows_by_key = {k: event_rows(c, g) for k, g in events.items()}
    featured, recap = mlp_state(c, today, events, rows_by_key)
    season = season_state(c, events, rows_by_key)
    # one best-lineup pair matrix over every season team; the featured event
    # gets its own subset so the historical shape is preserved
    all_teams = sorted({s["team"] for s in season["standings"]}
                       | (set(event_teams(featured["completed"] + featured["remaining"]))
                          if featured else set()))
    matrix = mlp_pair_matrix(all_teams, today, c) if all_teams else {}
    season["matrix"] = matrix
    if featured:
        fteams = set(event_teams(featured["completed"] + featured["remaining"]))
        featured["matrix"] = {k: v for k, v in matrix.items()
                              if set(k.split("|")) <= fteams}
    state = {"generated": str(today),
             "mlp": featured,
             "mlp_recap": recap,
             "ppa": ppa_state(c, today),
             "ppa_next": ppa_next(c, today),
             "season": season}
    (DATA / "tournament_state.json").write_text(json.dumps(state, indent=1))
    mlp = state["mlp"]
    log.info("mlp: %s [%s] (%d done, %d left) | recap: %s | season: %d teams, "
             "%d events counted, %d remaining | ppa: %d live, next: %s",
             mlp["event"] if mlp else "none",
             mlp["status"] if mlp else "-",
             len(mlp["completed"]) if mlp else 0,
             len(mlp["remaining"]) if mlp else 0,
             recap["event"] if recap else "none",
             len(season["standings"]), len(season["events"]),
             len(season["remaining"]), len(state["ppa"]),
             state["ppa_next"]["tournament"] if state["ppa_next"] else "none")


if __name__ == "__main__":
    main()
