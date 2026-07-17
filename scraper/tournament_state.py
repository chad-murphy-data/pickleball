"""Distill the CURRENT tournament state into data/tournament_state.json.

    python scraper/tournament_state.py            # window = today-6 .. today+4

Feeds the site's title-race page (web/build_site.py:build_titlerace):

- MLP: the active matchup group (an event weekend, e.g. "MLP San Diego") —
  completed matchups with actual game scores and rally points earned, plus
  the still-scheduled matchups of the same group.  Standings math and the
  Monte Carlo live build-side; this file only records facts.
- PPA: every pro DOUBLES division with a main-draw match inside the window —
  per match: round, match number, seeds, players (UUIDs lowercased, house
  rule), winner, per-game scores, format.  Seeds let the build reconstruct
  the actual seeded bracket (verified vs the 2026 Portland Challenger:
  semifinal pairings follow the standard 1-vs-4 / 2-vs-3 quarter template).

Network use: the same polite cached client as the harvester (~1 req/s,
volatile last-3-days refetch).  Runs in the nightly refresh right after the
harvest, so almost everything is already cached; a quiet week costs a
handful of calls and writes {"mlp": null, "ppa": []}.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harvest import is_mlp_league, is_ppa_tournament   # noqa: E402
from pb_api import PBClient                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
log = logging.getLogger("tournament_state")

LOOKBACK = 6          # days behind today that still count as "this event"
LOOKAHEAD = 4         # days ahead: rest of the current weekend
FINAL = {"COMPLETED_MATCHUP_STATUS", "BYE_MATCHUP_STATUS"}


def mlp_pair_matrix(teams, today, c):
    """Model-priced matchup tree for every unordered team pair, keyed
    "A|B" (sorted titles), tree oriented to the first title.  Reuses the
    forecast machinery (projected lineups from each team's most recent
    completed matchup) so simulated playoff pairings — which can be any
    cross-pool combination — carry real probabilities, not coin flips."""
    sys.path.insert(0, str(ROOT / "web"))
    import make_forecast as mf
    vals, singles = mf.load_values(), mf.load_singles()
    cache, lineups = {}, {}
    for t in teams:
        lineups[t] = mf.recent_lineup_for_team(c, t, today, cache)[0]
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


def mlp_state(c: PBClient, today: date):
    """The active MLP event, or None on a quiet week.  Each event weekend
    surfaces as its own "team league" (title "MLP San Diego"); the matchup
    group UUID lives on the division record, not the matchup rows."""
    events: dict[str, dict] = {}
    for i in range(-LOOKBACK, LOOKAHEAD + 1):
        d = today + timedelta(days=i)
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
                # one season-long league; the event weekend is the division's
                # matchup group (the league title just echoes that day's event)
                guid = (div.get("matchupGroupUuid") or "").lower()
                if not guid:
                    continue
                g = events.setdefault(guid, {"title": None, "matchups": {}})
                for mu in mus:
                    g["title"] = mu.get("matchupGroupTitle") or g["title"]
                    g["matchups"][mu["uuid"].lower()] = {"date": str(d), "short": mu}
    if not events:
        return None
    # the active event = the one with the most matchups in the window
    guid, g = max(events.items(), key=lambda kv: len(kv[1]["matchups"]))
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
    if not completed and not remaining:
        return None
    teams = sorted({r[k] for r in completed + remaining
                    for k in ("team1", "team2") if r[k]})
    return {"group": guid, "event": g["title"],
            "completed": completed, "remaining": remaining,
            "matrix": mlp_pair_matrix(teams, today, c)}


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


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    c = PBClient()
    today = date.today()
    state = {"generated": str(today),
             "mlp": mlp_state(c, today),
             "ppa": ppa_state(c, today)}
    (DATA / "tournament_state.json").write_text(json.dumps(state, indent=1))
    mlp = state["mlp"]
    log.info("mlp: %s (%d done, %d left) | ppa: %d tournament(s)",
             mlp["event"] if mlp else "none",
             len(mlp["completed"]) if mlp else 0,
             len(mlp["remaining"]) if mlp else 0, len(state["ppa"]))


if __name__ == "__main__":
    main()
