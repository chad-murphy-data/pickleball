"""Build the day-ahead slate: every scheduled pro match on a date, priced.

    python social/slate.py                     # tomorrow (America/Los_Angeles)
    python social/slate.py --date 2026-09-06   # any date
    python social/slate.py --date today

Writes social/out/<date>/slate.json — one record per bracket (MD, WD, MXD,
MS, WS, and MLP matchups) with every match's pre-match win probability, in
the exact numbers the live board would show at rally zero:

- PPA doubles: v2 values + weakest link -> per-point p -> race-to-T DP ->
  display calibration (web/calibration.json), one game at a time from the
  match's real score format (looked up per bracket round, the live proxy's
  rule), then the best-of series tree.  Rally-scoring formats are listed
  but not priced (the race DP is a side-out model).
- PPA singles: singles suite value +- sd, race integrated over both
  players' posterior sd (model/singles_holdout.py "suite+unc"), no doubles
  calibration layer, display floor applied.
- MLP: make_forecast's machinery, one matchup at a time — official
  lineups when published, else each team's best lineup — with the four
  games, the DreamBreaker and the matchup tree.

A side that isn't set yet (Sunday's final while Saturday's semis are still
on) is carried as TBD with no number; it's still worth a line ("winner of
...").  Completed matches on the target date are skipped — this is a
forecast, not a results page.

Verified 2026-09-05 19:34 PT: the BFF already served the next day's
finals list with the decided sides filled in, so an evening run sees
tomorrow's bracket.

Network: the same polite cached client as the harvester for the shared
endpoints; the per-date match list goes straight to the API (not the
harvest cache — that cache is doubles-only and parse.py reads it).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scraper"))
sys.path.insert(0, str(ROOT / "web"))

from harvest import is_mlp_league, is_ppa_tournament          # noqa: E402
from pb_api import PBClient, UA_LIVE                            # noqa: E402
import make_forecast as mf                                      # noqa: E402
from sitelib.race import (calibrate, game_win_prob_uncertain,   # noqa: E402
                          race_dist, sigmoid, team_eta)

log = logging.getLogger("slate")
OUT = ROOT / "social" / "out"
TOUR_TZ = ZoneInfo("America/Los_Angeles")
EPS = mf.CAL["eps"]

BRACKETS = {                       # slide order; key -> display title
    "MD": "Men's Doubles",
    "WD": "Women's Doubles",
    "MXD": "Mixed Doubles",
    "MS": "Men's Singles",
    "WS": "Women's Singles",
    "MLP": "MLP Matchups",
}
ORD = ("one", "two", "three", "four", "five")
FMT_MAX_LOOKUPS = 24


def bracket_of(event_title: str) -> str | None:
    t = (event_title or "").lower()
    if "mixed" in t:
        return "MXD"
    if "singles" in t:
        return "WS" if "women" in t else "MS" if "men" in t else None
    if "doubles" in t:
        return "WD" if "women" in t else "MD" if "men" in t else None
    return None


def display_floor(p: float) -> float:
    """Never 0% / 100% (house rule; live_engine.displayFloor)."""
    return (1 - EPS) * p + EPS / 2


def seq_prob(w1: int, w2: int, need: int, fut: list[float]) -> float:
    """P(side 1 takes the series) given games won so far and the per-game
    probabilities of the remaining games (livepage.seqProb)."""
    if w1 >= need:
        return 1.0
    if w2 >= need:
        return 0.0
    if not fut:
        return 0.5
    p, rest = fut[0], fut[1:]
    return p * seq_prob(w1 + 1, w2, need, rest) + (1 - p) * seq_prob(w1, w2 + 1, need, rest)


def clean_name(m: dict, side: str, n: str) -> str:
    return (m.get(f"team_{side}_player_{n}_name") or "").strip()


def side_of(m: dict, side: str, singles: bool) -> dict:
    ns = ("one",) if singles else ("one", "two")
    uuids = [str(m.get(f"team_{side}_player_{n}_uuid") or "").lower() for n in ns]
    names = [clean_name(m, side, n) for n in ns]
    tbd = not all(uuids) or not all(names)
    return {"uuids": [] if tbd else uuids, "names": [] if tbd else names,
            "seed": m.get(f"team_{side}_seed") or None, "tbd": tbd}


def local_start(m: dict) -> str | None:
    """'10:00 AM EDT' from the BFF's local timestamp (a local wall time
    with a decorative Z) + the abbreviation it ships alongside."""
    s = m.get("local_date_match_planned_start")
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", ""))
    except ValueError:
        return None
    abbr = m.get("timezone_abbreviation") or ""
    return f"{t.strftime('%-I:%M %p')} {abbr}".strip()


# ---- score formats (per bracket round, the live proxy's rule) -------------
def fmt_key(m: dict) -> str:
    if m.get("event_uuid"):
        return "|".join([str(m["event_uuid"]).lower(), m.get("in_bracket_type") or "",
                         str(m.get("pool_id") or "").lower(),
                         str(m.get("round_number") if m.get("round_number") is not None
                             else m.get("round_text") or "")])
    return f"match:{str(m.get('match_uuid')).lower()}"


def fmt_fits(m: dict, fmt: dict) -> bool:
    bo = m.get("score_format_game_best_out_of")
    return not bo or not fmt.get("best_of") or bo == fmt["best_of"]


def match_format(c: PBClient, uuid: str) -> dict | None:
    try:
        body = c._get_json(f"/api/v1/results/getResultMatchInfos?id={uuid}")
    except Exception as e:                                   # noqa: BLE001
        log.warning("format lookup failed %s: %s", uuid, e)
        return None
    d = body.get("data") if isinstance(body, dict) else body
    d = d[0] if isinstance(d, list) and d else d
    if not isinstance(d, dict):
        return None
    mx = [d.get(f"score_format_game_{o}_max") or 0 for o in ORD]
    return {"rally": bool(d.get("is_rally_scoring")),
            "max": mx, "win_by": d.get("score_format_game_one_win_by") or 2,
            "title": d.get("score_format_shorthand") or d.get("score_format_title") or "",
            "best_of": d.get("score_format_game_best_out_of") or len([x for x in mx if x]),
            "planned_start": d.get("local_date_match_planned_start")}


def resolve_formats(c: PBClient, matches: list[dict]) -> dict[str, dict | None]:
    groups: dict[str, list[dict]] = {}
    for m in matches:
        groups.setdefault(fmt_key(m), []).append(m)
    out: dict[str, dict | None] = {}
    lookups = 0
    for key, ms in groups.items():
        grp = None
        if lookups < FMT_MAX_LOOKUPS:
            grp = match_format(c, ms[0]["match_uuid"])
            lookups += 1
        for m in ms:
            u = str(m["match_uuid"]).lower()
            if grp and fmt_fits(m, grp):
                out[u] = grp
            elif grp and lookups < FMT_MAX_LOOKUPS:
                out[u] = match_format(c, m["match_uuid"])
                lookups += 1
            else:
                out[u] = None
    return out


# ---- pricing ----------------------------------------------------------------
def modal_score(p_point: float, T: int) -> str:
    """Most likely final score of one game (the live board's 'modal game')."""
    dist = race_dist(round(p_point, 4), T)
    scores = ([(T, b, pr) for _, b, pr in dist["win_scores"]]
              + [(a, T, pr) for a, _, pr in dist["lose_scores"]])
    a, b, _ = max(scores, key=lambda s: s[2])
    return f"{a}-{b}"


def price_doubles(s1: dict, s2: dict, fmt: dict, vals: dict):
    """(p side 1 wins the series, note, modal first-game score)."""
    try:
        a = [vals[u][1] for u in s1["uuids"]]
        b = [vals[u][1] for u in s2["uuids"]]
    except KeyError:
        return None, "unrated pairing", None
    if len(a) != 2 or len(b) != 2:
        return None, "unrated pairing", None
    eta = team_eta(a[0], a[1], b[0], b[1])
    ts = [t for t in fmt["max"] if t > 0]
    per_game = [calibrate(race_dist(round(sigmoid(eta), 4), t)["p_win"]) for t in ts]
    need = math.ceil(len(ts) / 2)
    return (display_floor(seq_prob(0, 0, need, per_game)), None,
            modal_score(sigmoid(eta), ts[0]))


def price_singles(s1: dict, s2: dict, fmt: dict, singles: dict):
    a, b = singles.get(s1["uuids"][0]), singles.get(s2["uuids"][0])
    if not a or not b:
        return None, "unrated player", None
    eta = a[0] - b[0]
    sd = math.sqrt(a[1] ** 2 + b[1] ** 2)
    ts = [t for t in fmt["max"] if t > 0]
    per_game = [game_win_prob_uncertain(eta, sd, t) for t in ts]
    need = math.ceil(len(ts) / 2)
    return (display_floor(seq_prob(0, 0, need, per_game)), None,
            modal_score(sigmoid(eta), ts[0]))


def load_singles_full() -> dict[str, tuple[float, float]]:
    import csv
    path = ROOT / "data" / "singles_players.csv"
    if not path.exists():
        return {}
    return {r["player_id"]: (float(r["singles_value"]), float(r["singles_sd"]))
            for r in csv.DictReader(path.open())}


# ---- PPA ----------------------------------------------------------------------
def tbd_branches(c: PBClient, m: dict, d: dt.date, known: dict, sg: bool,
                 fmt: dict | None, vals: dict, singles: dict) -> list[dict]:
    """Sunday's final while Saturday's semi is still on: the open side is
    one of the two teams in the pending prior-round match of the same
    event, so price the known side against each.  Prior-round matches are
    read off the previous day's list (same open endpoint)."""
    if not fmt or fmt["rally"] or known["tbd"] or not m.get("event_uuid") \
            or not m.get("round_number"):
        return []
    try:
        body = c._get_json("/api/v1/results/getMatchInfosShort"
                           f"?eventIds={m['event_uuid']}&date={d - dt.timedelta(days=1)}")
    except Exception:                                        # noqa: BLE001
        return []
    prev = (body.get("data") or []) if isinstance(body, dict) else []
    mine = set(known["uuids"])
    out = []
    for pm in prev:
        if pm.get("round_number") != m["round_number"] - 1 or pm.get("match_status") == 4:
            continue
        sides = [side_of(pm, "one", sg), side_of(pm, "two", sg)]
        if any(s["tbd"] for s in sides) or any(set(s["uuids"]) & mine for s in sides):
            continue
        for s in sides:
            p, _, _ = (price_singles(known, s, fmt, singles) if sg
                       else price_doubles(known, s, fmt, vals))
            if p is not None:
                out.append({**s, "p_known": round(p, 4)})
        break                                # one feeder match per open side
    return out


def ppa_slate(c: PBClient, d: dt.date, vals: dict, singles: dict) -> list[dict]:
    """One record per (tournament, bracket) with its scheduled matches."""
    out = []
    for t in c.tournaments_on_date(d):
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
            picks = [e for e in events if re.search(r"doubles|singles", e["title"], re.I)]
            if not picks:
                continue
            body = c._get_json("/api/v1/results/getMatchInfosShort"
                               f"?eventIds={','.join(e['uuid'] for e in picks)}&date={d}")
        except PermissionError:
            continue
        matches = (body.get("data") or []) if isinstance(body, dict) else []
        matches = [m for m in matches if m.get("match_status") != 4
                   and not m.get("did_cancel_match")]
        if not matches:
            continue
        fmts = resolve_formats(c, matches)
        by_bracket: dict[str, list[dict]] = {}
        for m in matches:
            br = bracket_of(m.get("event_title"))
            if not br:
                continue
            sg = br in ("MS", "WS")
            s1, s2 = side_of(m, "one", sg), side_of(m, "two", sg)
            fmt = fmts.get(str(m["match_uuid"]).lower())
            p1, note, modal = None, None, None
            branches = []
            if s1["tbd"] or s2["tbd"]:
                note = "opponent not set yet"
                if not (s1["tbd"] and s2["tbd"]):
                    branches = tbd_branches(c, m, d, s2 if s1["tbd"] else s1, sg,
                                            fmt, vals, singles)
            elif not fmt:
                note = "score format unknown"
            elif fmt["rally"]:
                note = "rally-scoring format, not priced"
            elif sg:
                p1, note, modal = price_singles(s1, s2, fmt, singles)
            else:
                p1, note, modal = price_doubles(s1, s2, fmt, vals)
            by_bracket.setdefault(br, []).append({
                "match_uuid": str(m["match_uuid"]).lower(),
                "round": (m.get("round_text") or "").strip() or
                         (f"R{m['round_number']}" if m.get("round_number") else ""),
                "round_no": m.get("round_number") or 0,
                "match_no": m.get("match_number") or 0,
                "consolation": not (m.get("round_text") or "").strip(),
                "start": local_start(m),
                "format": (fmt or {}).get("title") or
                          (f"best of {m['score_format_game_best_out_of']}"
                           if m.get("score_format_game_best_out_of") else ""),
                "best_of": (fmt or {}).get("best_of") or m.get("score_format_game_best_out_of"),
                "t1": s1, "t2": s2,
                "p1": round(p1, 4) if p1 is not None else None,
                "modal": modal, "note": note, "branches": branches,
            })
        venue = ", ".join(x for x in (t.get("LocationCity"), t.get("LocationState")) if x)
        for br, rows in by_bracket.items():
            rows.sort(key=lambda r: (r["consolation"], r["round_no"], r["match_no"]))
            out.append({"tour": "PPA", "event": title, "event_id": tid.lower(),
                        "venue": venue, "bracket": br, "title": BRACKETS[br],
                        "rows": rows})
    return out


# ---- MLP ----------------------------------------------------------------------
MLP_SLOT_BRACKET = {"WD": "WD", "MD": "MD", "MXD1": "MXD", "MXD2": "MXD"}


def mlp_slate(c: PBClient, d: dt.date, vals: dict, singles_mf: dict) -> list[dict]:
    """MLP matchups scheduled on d: one 'MLP' record (matchup-level) plus
    per-bracket records carrying each matchup's game in that slot."""
    scheduled = []
    for tl in c.team_leagues_on_date(d):
        if not is_mlp_league(tl):
            continue
        for div in tl.get("divisions") or []:
            try:
                mus = c.tl_matchups_short(tl, div, d)
            except PermissionError:
                continue
            for mu in mus:
                if "SCHEDULED" in (mu.get("matchupStatus") or "") \
                        and mu.get("teamOneTitle") and mu.get("teamTwoTitle"):
                    scheduled.append((tl, mu))
    if not scheduled:
        return []
    today = dt.datetime.now(TOUR_TZ).date()
    rosters = mf.mlp_rosters(c, today)
    cache: dict = {}
    matchups, games = [], {b: [] for b in ("WD", "MD", "MXD")}
    for tl, mu in scheduled:
        t1, t2 = mu["teamOneTitle"], mu["teamTwoTitle"]
        official = mf.matchup_lineups(c.matchup_data(mu["uuid"], volatile=True))
        if len(official) >= 3:
            lu1 = {s: p["One"] for s, p in official.items()}
            lu2 = {s: p["Two"] for s, p in official.items()}
            src = "official lineups"
        else:
            lu1, _ = mf.projected_lineup_for_team(c, t1, today, rosters, vals, cache)
            lu2, _ = mf.projected_lineup_for_team(c, t2, today, rosters, vals, cache)
            src = "projected lineups"
        ps = []
        for slot in mf.SLOTS:
            g = mf.price_game((lu1 or {}).get(slot), (lu2 or {}).get(slot), vals)
            if not g:
                continue
            ps.append(g["p"])
            games[MLP_SLOT_BRACKET[slot]].append({
                "match_uuid": f"{mu['uuid']}:{slot}",
                "round": slot, "round_no": 0, "match_no": len(ps),
                "consolation": False, "start": None, "format": "race to 11",
                "t1": {"names": [vals[u][0] for u in lu1[slot]], "uuids": lu1[slot],
                       "seed": None, "tbd": False, "team": t1},
                "t2": {"names": [vals[u][0] for u in lu2[slot]], "uuids": lu2[slot],
                       "seed": None, "tbd": False, "team": t2},
                "p1": g["p"], "modal": g["modal"], "note": None,
            })
        p_db, tree = 0.5, None
        if lu1 and lu2:
            r1 = {u for pr in lu1.values() for u in pr}
            r2 = {u for pr in lu2.values() for u in pr}
            p_db = mf.db_win_prob(r1, r2, vals, singles_mf)
        if len(ps) == 4:
            tree = mf.matchup_tree(ps, p_db)
        matchups.append({
            "match_uuid": mu["uuid"], "round": "Matchup",
            "round_no": 0, "match_no": len(matchups), "consolation": False,
            "start": None, "format": src,          # BFF start stamps carry no tz
            "t1": {"names": [t1], "uuids": [], "seed": None, "tbd": False, "is_team": True},
            "t2": {"names": [t2], "uuids": [], "seed": None, "tbd": False, "is_team": True},
            "p1": round(tree["p_win"], 4) if tree else None,
            "p_db": round(p_db, 4), "modal": None,
            "note": None if tree else "lineup unpriceable",
        })
    tl0 = scheduled[0][0]
    event = tl0.get("title") or "MLP"
    loc = tl0.get("location") if isinstance(tl0.get("location"), dict) else {}
    venue = ", ".join(x for x in (loc.get("city"), loc.get("state")) if x)
    base = {"tour": "MLP", "event": event, "event_id": tl0.get("uuid"), "venue": venue}
    out = [{**base, "bracket": "MLP", "title": BRACKETS["MLP"], "rows": matchups}]
    for br in ("MD", "WD", "MXD"):
        if games[br]:
            out.append({**base, "bracket": br, "title": BRACKETS[br], "rows": games[br]})
    return out


# ---- entry ----------------------------------------------------------------------
def parse_date(s: str | None) -> dt.date:
    today = dt.datetime.now(TOUR_TZ).date()
    if not s or s == "tomorrow":
        return today + dt.timedelta(days=1)
    if s == "today":
        return today
    return dt.date.fromisoformat(s)


def build(d: dt.date) -> dict:
    vals = mf.load_values()
    c = PBClient(ua=UA_LIVE)
    slates = ppa_slate(c, d, vals, load_singles_full()) + mlp_slate(c, d, vals, mf.load_singles())
    order = list(BRACKETS)
    slates.sort(key=lambda s: (s["tour"] != "MLP", s["event"], order.index(s["bracket"])))
    n_rows = sum(len(s["rows"]) for s in slates)
    n_priced = sum(1 for s in slates for r in s["rows"] if r["p1"] is not None)
    return {"date": str(d), "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "events": sorted({s["event"] for s in slates}),
            "n_matches": n_rows, "n_priced": n_priced, "slates": slates}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="tomorrow", help="YYYY-MM-DD | today | tomorrow")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    d = parse_date(args.date)
    slate = build(d)
    out = OUT / str(d)
    out.mkdir(parents=True, exist_ok=True)
    (out / "slate.json").write_text(json.dumps(slate, indent=1))
    for s in slate["slates"]:
        print(f"[{s['tour']}] {s['event']} — {s['title']}")
        for r in s["rows"]:
            a = " / ".join(r["t1"]["names"]) or "TBD"
            b = " / ".join(r["t2"]["names"]) or "TBD"
            p = f"{r['p1']:.0%}" if r["p1"] is not None else f"— ({r['note']})"
            print(f"   {r['round']:<14} {a}  vs  {b}: {p}")
    print(f"wrote {out / 'slate.json'} ({slate['n_priced']}/{slate['n_matches']} priced)")


if __name__ == "__main__":
    main()
