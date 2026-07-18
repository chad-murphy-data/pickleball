"""Price upcoming MLP matchups and write data/forecasts.json for the site.

    python web/make_forecast.py                  # next 7 days -> data/forecasts.json
    python web/make_forecast.py --days 10
    python web/make_forecast.py --commit         # ALSO append pending entries
                                                 # to model/receipts.json (the
                                                 # deliberate pre-match freeze)

Lineups are PROJECTED: each team's most recent completed matchup supplies
its WD/MD/MXD1/MXD2 pairings (the page says so loudly).  Pricing mirrors the
graded Mid-Season methodology: per-game win prob from current v2 values +
weakest link, race-to-11 DP, display calibration applied; the DreamBreaker
is 50/50 by documented convention (singles is outside the doubles model).
P(win matchup) = P(win >=3 of 4 games) + P(2-2) * 0.5.

Network use: same polite cached client as the harvester.  CI runs this
nightly for the page; --commit is reserved for a human deciding to put a
forecast in the permanent ledger BEFORE the matches.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scraper"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harvest import is_mlp_league                          # noqa: E402
from pb_api import PBClient                                # noqa: E402
from tournament_state import ppa_state                     # noqa: E402
from sitelib.race import (calibrate, race_dist, set_calibration, sigmoid,
                          team_eta)                        # noqa: E402

DATA = ROOT / "data"
CAL = json.loads((Path(__file__).resolve().parent / "calibration.json").read_text())
set_calibration(CAL["a"], CAL["b"], CAL["eps"])

SLOTS = ("WD", "MD", "MXD1", "MXD2")
LOOKBACK_DAYS = 60

# DreamBreaker model v2 (fit on all 101 historical DBs, model/db_model.md):
# per-rally logit = K_DB_SINGLES * (mean roster SINGLES value gap). Singles
# values from model/fit_singles.py; players with <10 singles games are
# imputed from doubles via the fitted regression (singles ≈ 0.28 + 1.14*d).
# Singles-gap model beats the doubles proxy by 3.1 nll on the 101 DBs.
K_DB_SINGLES = 0.42
SINGLES_IMPUTE = (0.28, 1.14)


def load_singles():
    path = DATA / "singles_players.csv"
    if not path.exists():
        return {}
    return {r["player_id"]: (float(r["singles_value"]), int(r["singles_games"]))
            for r in csv.DictReader(path.open())}


def db_win_prob(roster1, roster2, vals, singles):
    """P(team1 wins DreamBreaker) from mean roster singles strength."""
    def s_of(u):
        if u in singles and singles[u][1] >= 10:
            return singles[u][0]
        if u in vals:
            a, b = SINGLES_IMPUTE
            return a + b * vals[u][1]
        return None
    s1 = [s_of(u) for u in roster1]
    s2 = [s_of(u) for u in roster2]
    if not s1 or not s2 or any(v is None for v in s1 + s2):
        return 0.5
    gap = sum(s1) / len(s1) - sum(s2) / len(s2)
    p = race_dist(round(sigmoid(K_DB_SINGLES * gap), 4), 21)["p_win"]
    eps = CAL["eps"]
    return min(max(p, eps / 2), 1 - eps / 2)


def load_values():
    """(uuid -> (name, per-point-logit value), field 25th-pct floor value).

    The floor mirrors ppa_bracket_panel: PPA Challenger draws are full of
    players below the model's 60-game threshold, so a missing player is filled
    with the field's 25th-percentile value (and the match flagged) rather than
    dropped — otherwise most Challenger matches would show NOT RATED."""
    vals, dyn = {}, []
    for r in csv.DictReader((DATA / "v2_players.csv").open()):
        vals[r["player_id"]] = (r["full_name"], float(r["value_now_mean"]))
        if r.get("dynamic") in ("1", "True", "true"):
            dyn.append(float(r["value_now_mean"]))
    dyn.sort()
    floor = dyn[len(dyn) // 4] if dyn else 0.0
    return vals, floor


def bo_win(p, best_of):
    """Match win prob from a single-game win prob (mirrors build_site.bo_win)."""
    if best_of == 3:
        return p * p * (3 - 2 * p)
    if best_of == 5:
        return p ** 3 * (10 - 15 * p + 6 * p * p)
    return p


def price_ppa_match(m, vals, floor_value):
    """Price a single upcoming PPA doubles match (pair vs pair).  Returns None
    for byes, TBD opponents, or matches already played."""
    if m.get("winner"):                       # already decided
        return None
    p1, p2 = m.get("p1") or [], m.get("p2") or []
    if not (len(p1) == 2 and len(p2) == 2 and all(p1) and all(p2)):
        return None                           # bye / opponent not yet determined
    best_of = m.get("best_of") or 3

    def pair_vals(pair):
        out, unrated = [], False
        for u in pair:
            if u in vals:
                out.append(vals[u][1])
            else:
                out.append(floor_value); unrated = True
        return out, unrated

    va, u1 = pair_vals(p1)
    vb, u2 = pair_vals(p2)
    eta = team_eta(va[0], va[1], vb[0], vb[1])
    T = 11 if best_of > 1 else 15             # Challenger single games go to 15
    dist = race_dist(round(sigmoid(eta), 4), T)
    p_game = calibrate(dist["p_win"])
    eps = CAL["eps"]
    p_match = min(max(bo_win(p_game, best_of), eps / 2), 1 - eps / 2)
    scores = ([(T, b, pr) for _, b, pr in dist["win_scores"]]
              + [(a, T, pr) for a, _, pr in dist["lose_scores"]])
    modal = max(scores, key=lambda s: s[2])
    return {
        "round": m.get("round_text"),
        "t1_pair": [n for n in (m.get("n1") or []) if n],
        "t2_pair": [n for n in (m.get("n2") or []) if n],
        "p": round(p_match, 4), "modal": f"{modal[0]}-{modal[1]}",
        "best_of": best_of, "unrated": u1 or u2,
    }


def match_slot(m):
    """WD/MD/MXD1/MXD2 (finals use WDG/MDG etc., hence startswith)."""
    ab = (m.get("matchAbbreviation") or "").upper()
    for s in ("MXD1", "MXD2", "WD", "MD"):
        if ab.startswith(s):
            return s
    return None


def matchup_lineups(md):
    """slot -> {teamOne: [uuid, uuid], teamTwo: [...]} from a completed matchup."""
    out = {}
    for m in md.get("matches") or []:
        if m.get("isTieBreaker"):
            continue
        slot = match_slot(m)
        if slot is None:
            continue
        pair = {}
        for side in ("One", "Two"):
            us = [str(m.get(f"team{side}Player{pn}Uuid") or "").lower()
                  for pn in ("One", "Two")]
            if not all(us):
                pair = None
                break
            pair[side] = us
        if pair:
            out[slot] = pair
    return out


def recent_lineup_for_team(c, team_title, before, cache):
    """Walk back from `before` to find the team's most recent completed
    matchup; return (slot -> [uuid, uuid], source_date)."""
    if team_title in cache:
        return cache[team_title]
    d = before
    for _ in range(LOOKBACK_DAYS):
        d -= timedelta(days=1)
        for tl in c.team_leagues_on_date(d):
            if not is_mlp_league(tl):
                continue
            for div in tl.get("divisions") or []:
                try:
                    mus = c.tl_matchups_short(tl, div, d)
                except PermissionError:
                    continue
                for mu in mus:
                    if mu.get("matchupStatus") != "COMPLETED_MATCHUP_STATUS":
                        continue
                    for side, title_key in (("One", "teamOneTitle"),
                                            ("Two", "teamTwoTitle")):
                        if mu.get(title_key) != team_title:
                            continue
                        md = c.matchup_data(mu["uuid"], volatile=False)
                        lus = matchup_lineups(md)
                        if len(lus) >= 3:          # a real played fixture
                            got = ({s: p[side] for s, p in lus.items()}, str(d))
                            cache[team_title] = got
                            return got
    cache[team_title] = (None, None)
    return None, None


def price_game(pair_a, pair_b, vals):
    """Calibrated win prob + modal score for pair_a vs pair_b (race to 11)."""
    if not pair_a or not pair_b:
        return None
    try:
        va = [vals[u][1] for u in pair_a]
        vb = [vals[u][1] for u in pair_b]
    except KeyError:
        return None
    eta = team_eta(va[0], va[1], vb[0], vb[1])
    dist = race_dist(round(sigmoid(eta), 4), 11)
    p = calibrate(dist["p_win"])
    scores = ([(11, b, pr) for _, b, pr in dist["win_scores"]]
              + [(a, 11, pr) for a, _, pr in dist["lose_scores"]])
    modal = max(scores, key=lambda s: s[2])
    return {"p": round(p, 4), "modal": f"{modal[0]}-{modal[1]}",
            "margin": round(dist["exp_margin"], 2)}


def matchup_tree(game_ps, p_db=0.5):
    """Outcome distribution over 4 independent games + 50/50 DreamBreaker."""
    dist = [1.0]
    for p in game_ps:
        nxt = [0.0] * (len(dist) + 1)
        for w, pr in enumerate(dist):
            nxt[w + 1] += pr * p
            nxt[w] += pr * (1 - p)
        dist = nxt
    p40, p31, p22, p13, p04 = dist[4], dist[3], dist[2], dist[1], dist[0]
    # never 0%/100%, matchup level included: the catastrophic tail (injury,
    # lineup shock) applies at least as strongly to a whole matchup as to a
    # game, so the same empirical eps floor clamps the headline number
    eps = CAL["eps"]
    p_win = min(max(p40 + p31 + p22 * p_db, eps / 2), 1 - eps / 2)
    return {"p_win": p_win, "p_40": p40, "p_31": p31,
            "p_db": p22, "p_13": p13, "p_04": p04}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--commit", action="store_true",
                    help="append pending forecast entries to model/receipts.json")
    args = ap.parse_args()

    vals, floor_value = load_values()
    singles = load_singles()
    c = PBClient()
    today = date.today()
    lineup_cache = {}
    forecasts = []

    for i in range(args.days + 1):
        d = today + timedelta(days=i)
        for tl in c.team_leagues_on_date(d):
            if not is_mlp_league(tl):
                continue
            for div in tl.get("divisions") or []:
                try:
                    mus = c.tl_matchups_short(tl, div, d)
                except PermissionError:
                    continue
                for mu in mus:
                    st = mu.get("matchupStatus") or ""
                    if "SCHEDULED" not in st:
                        continue
                    t1, t2 = mu.get("teamOneTitle"), mu.get("teamTwoTitle")
                    if not t1 or not t2:
                        continue
                    lu1, src1 = recent_lineup_for_team(c, t1, today, lineup_cache)
                    lu2, src2 = recent_lineup_for_team(c, t2, today, lineup_cache)
                    games, ps = [], []
                    for slot in SLOTS:
                        g = price_game((lu1 or {}).get(slot), (lu2 or {}).get(slot), vals)
                        pj = None
                        if g:
                            ps.append(g["p"])
                            pj = {
                                "slot": slot, **g,
                                "t1_pair": [vals[u][0] for u in lu1[slot]],
                                "t2_pair": [vals[u][0] for u in lu2[slot]],
                            }
                        games.append(pj)
                    p_db = 0.5
                    if lu1 and lu2:
                        r1 = {u for pr in lu1.values() for u in pr}
                        r2 = {u for pr in lu2.values() for u in pr}
                        p_db = db_win_prob(r1, r2, vals, singles)
                    tree = matchup_tree(ps, p_db) if len(ps) == 4 else None
                    if tree:
                        tree["p_db_win"] = round(p_db, 4)
                    forecasts.append({
                        "date": str(d),
                        "start": mu.get("plannedStartDate"),
                        "event": tl.get("title"),
                        "team1": t1, "team2": t2,
                        "lineups_from": {"team1": src1, "team2": src2},
                        "games": games,
                        "tree": {k: round(v, 4) for k, v in tree.items()} if tree else None,
                    })
                    print(f"{d} {t1} vs {t2}: "
                          + (f"{tree['p_win']:.0%}" if tree else "unpriceable"))

    # ---- PPA pro-doubles: price the unplayed matches in each live draw ----
    # ppa_state only returns a tournament's matches once its day is underway
    # (the BFF publishes the draw then), so these appear when play starts.
    ppa_forecasts = []
    try:
        for t in ppa_state(c, today):
            for div in t["divisions"]:
                for m in div["matches"]:
                    pj = price_ppa_match(m, vals, floor_value)
                    if not pj:
                        continue
                    pj.update(date=str(today), start=None,
                              event=t["tournament"], division=div["title"])
                    ppa_forecasts.append(pj)
        for f in ppa_forecasts:
            print(f"PPA {f['event']} {f['round']}: "
                  f"{'/'.join(n.split()[-1] for n in f['t1_pair'])} vs "
                  f"{'/'.join(n.split()[-1] for n in f['t2_pair'])}: {f['p']:.0%}")
    except Exception as e:                          # network/parse hiccup: MLP still ships
        print(f"PPA pricing skipped: {e}")

    out = {
        "generated": str(today),
        "note": "MLP: projected lineups (each team's most recent completed "
                "matchup), DreamBreaker 50/50 by convention. PPA: unplayed "
                "matches in each live pro-doubles draw (pairs are known once the "
                "day is underway). Calibrated probabilities throughout.",
        "forecasts": forecasts,
        "ppa_forecasts": ppa_forecasts,
    }
    (DATA / "forecasts.json").write_text(json.dumps(out, indent=1))
    print(f"wrote data/forecasts.json ({len(forecasts)} MLP matchups, "
          f"{len(ppa_forecasts)} PPA matches)")

    if args.commit and forecasts:
        rj = json.loads((ROOT / "model" / "receipts.json").read_text())
        stamp = str(today)
        items = []
        for f in forecasts:
            if not f["tree"]:
                continue
            items.append({
                "label": f"{f['date']} {f['team1']} over {f['team2']} (projected lineups)",
                "prob": f["tree"]["p_win"], "outcome": None, "result": None,
                "grade": "PENDING", "brier": None,
            })
        rj["entries"].append({
            "id": f"auto-forecast-{stamp}",
            "title": f"MLP matchup forecasts committed {stamp}",
            "committed": stamp, "graded": None, "status": "pending",
            "grade_after": None,
            "source": "web/make_forecast.py (data/forecasts.json snapshot)",
            "model": "v2 current values, projected lineups, DB 50/50, calibrated",
            "outcome_summary": None, "items": items,
            "footnote": "Committed via make_forecast --commit before the matches; "
                        "grade against final matchup results.",
        })
        rj["updated"] = stamp
        (ROOT / "model" / "receipts.json").write_text(json.dumps(rj, indent=1))
        print(f"appended {len(items)} pending calls to model/receipts.json")


if __name__ == "__main__":
    main()
