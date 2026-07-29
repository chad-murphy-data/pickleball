"""Simulate a whole MLP event weekend: groups, crossover playoff, final places.

    python web/make_event_forecast.py                 # the next/current event
    python web/make_event_forecast.py --sims 200000
    python web/make_event_forecast.py --date 2026-07-30

Writes data/event_forecast.json (machine-readable, for the site / design) and
model/event_forecast.md (the human brief).

FORMAT (reverse-engineered from the BFF and validated against all eight
2026 MLP events — Dallas, Columbus, St. Louis, Austin, St. Petersburg, New
York, San Diego, Chicago; see the docstring block in `place_rule` below):

  - 11-12 teams split into two round-robin GROUPS (pools of 5 or 6); every
    team plays every other team in its group once.  All four games of a
    round-robin matchup are played even at 3-0 (only the ONE-AND-DONE
    playoff stops at three); 2-2 goes to a DreamBreaker.
  - Group rank = matchup wins, then head-to-head for a two-way tie, then
    game differential, then rally-point differential.
  - PLAYOFF is a single round of rank-vs-rank crossovers between the two
    groups: A1 v B1, A2 v B2, A3 v B3, A4 v B4.  Winners take places 1, 3,
    5, 7; losers take 2, 4, 6, 8.  Teams finishing 5th/6th in their group
    do not play a playoff matchup and share places 9-10 / 11-12.

Pricing is the graded make_forecast methodology unchanged: per-game win
probability from current v2 values + weakest link, race-to-11 DP, display
calibration, P(matchup) = P(win >=3 of 4) + P(2-2) * P(DreamBreaker).
Lineups follow the same three-tier ladder (official > best lineup > last
matchup), so these numbers agree with the slate page's per-matchup prices.

Completed matchups in the group are treated as FACT — the simulator seeds
standings from them and only rolls the remaining slate, so this stays
correct if re-run mid-event.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scraper"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harvest import is_mlp_league                              # noqa: E402
from pb_api import MATCHUP_FINAL, PBClient                     # noqa: E402
import make_forecast as mf                                     # noqa: E402
from sitelib.race import race_dist, sigmoid, team_eta          # noqa: E402

log = logging.getLogger("event_forecast")
DATA = ROOT / "data"
GROUP_LABELS = ("A", "B")
PLAYOFF_DEPTH = 4          # ranks 1-4 of each group cross over
PLAYED = "PLAYED_MATCHUP_COMPLETION_TYPE"


# ---------------------------------------------------------------- discovery

def find_event(c: PBClient, today: date, back: int, ahead: int):
    """The active MLP matchup group = the one with the most matchups in the
    window.  Returns (group_uuid, title, [matchup rows])."""
    events: dict[str, dict] = {}
    for i in range(-back, ahead + 1):
        d = today + timedelta(days=i)
        for tl in c.team_leagues_on_date(d):
            if not is_mlp_league(tl):
                continue
            for div in tl.get("divisions") or []:
                try:
                    mus = c.tl_matchups_short(tl, div, d)
                except PermissionError:
                    continue
                guid = (div.get("matchupGroupUuid") or "").lower()
                if not guid or not mus:
                    continue
                g = events.setdefault(guid, {"title": None, "mus": {}})
                for mu in mus:
                    g["title"] = mu.get("matchupGroupTitle") or g["title"]
                    g["mus"][mu["uuid"].lower()] = (str(d), mu)
    if not events:
        return None
    guid, g = max(events.items(), key=lambda kv: len(kv[1]["mus"]))
    rows = []
    for muid, (d, mu) in sorted(g["mus"].items(), key=lambda kv: kv[1][0]):
        status = mu.get("matchupStatus") or ""
        if status == "BYE_MATCHUP_STATUS":
            continue
        md = c.matchup_data(muid, volatile=status not in MATCHUP_FINAL)
        t1, t2 = md.get("teamOneTitle"), md.get("teamTwoTitle")
        if not t1 or not t2:
            continue                      # unfilled playoff placeholder
        rows.append({
            "uuid": muid, "date": d, "start": mu.get("plannedStartDate"),
            "team1": t1, "team2": t2,
            "pool": (md.get("poolUuid") or "")[:8],
            "bracket": md.get("inBracketType") or "?",
            "round": md.get("roundText"), "mnum": md.get("matchupNumber") or 0,
            "done": (status == "COMPLETED_MATCHUP_STATUS"
                     and md.get("matchupCompletedType") == PLAYED),
            "games1": md.get("teamOneScore"), "games2": md.get("teamTwoScore"),
            "pts1": md.get("teamOnePointsEarned"),
            "pts2": md.get("teamTwoPointsEarned"),
        })
    return guid, g["title"], rows


# ------------------------------------------------------------------ pricing

def score_cdfs(dist, T):
    """(winner-score cdf, loser-score cdf) as [(cum, a, b)] in the pricing
    orientation (first team listed first).  The win-by-2 branch is lumped at
    T-2, matching race_dist's convention that the winner's margin is 2."""
    def cdf(rows):
        tot = sum(r[2] for r in rows) or 1.0
        acc, out = 0.0, []
        for a, b, pr in rows:
            acc += pr / tot
            out.append((acc, a, b))
        return out
    win = [(T, b, pr) for _, b, pr in dist["win_scores"]]
    win.append((T, T - 2, dist["p_ot_win"]))
    lose = [(a, T, pr) for a, _, pr in dist["lose_scores"]]
    lose.append((T - 2, T, dist["p_ot"] - dist["p_ot_win"]))
    return cdf(win), cdf(lose)


def db_dist(p_db):
    """Race-to-21 rally distribution whose win prob matches p_db (inverted
    so simulated DreamBreaker SCORES are consistent with the DB model's
    probability, which comes from the singles-value gap)."""
    lo, hi = 1e-3, 1 - 1e-3
    for _ in range(40):
        mid = (lo + hi) / 2
        if race_dist(round(mid, 4), 21)["p_win"] < p_db:
            lo = mid
        else:
            hi = mid
    return race_dist(round((lo + hi) / 2, 4), 21)


def price_pairs(teams, lineups, vals, singles):
    """(a, b) -> everything the simulator needs, keyed on the sorted pair."""
    out = {}
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            la, lb = lineups.get(a), lineups.get(b)
            if not la or not lb:
                continue
            games, ok = [], True
            for slot in mf.SLOTS:
                priced = mf.price_game(la.get(slot), lb.get(slot), vals)
                if not priced:
                    ok = False
                    break
                va = [vals[u][1] for u in la[slot]]
                vb = [vals[u][1] for u in lb[slot]]
                eta = team_eta(va[0], va[1], vb[0], vb[1])
                d = race_dist(round(sigmoid(eta), 4), 11)
                win, lose = score_cdfs(d, 11)
                games.append({"slot": slot, "p": priced["p"], "win": win,
                              "lose": lose, "modal": priced["modal"]})
            if not ok:
                continue
            r1 = {u for pr in la.values() for u in pr}
            r2 = {u for pr in lb.values() for u in pr}
            p_db = mf.db_win_prob(r1, r2, vals, singles)
            dbw, dbl = score_cdfs(db_dist(p_db), 21)
            tree = mf.matchup_tree([g["p"] for g in games], p_db)
            out[(a, b)] = {"games": games, "p_db": p_db, "db_win": dbw,
                           "db_lose": dbl, "tree": tree}
    return out


# --------------------------------------------------------------- simulation

def draw(cdf, u):
    for cum, a, b in cdf:
        if u <= cum:
            return a, b
    return cdf[-1][1], cdf[-1][2]


def place_rule(order_a, order_b):
    """Rank-vs-rank crossover: (rank, teamA, teamB) for the playoff round.

    Validated against every 2026 event: the four playoff matchups always
    pair equal group ranks, and the two rank-1 teams always meet.  The
    reverse-engineered group order (wins > head-to-head > game diff > rally
    diff) is what makes all eight events' pairings reproduce exactly --
    e.g. Chicago's Slice-over-Bouncers (head-to-head beats a worse game
    diff) and Austin's three-way 2-3 tie (game diff, then rally points).
    """
    return [(i + 1, order_a[i], order_b[i]) for i in range(PLAYOFF_DEPTH)]


def simulate(pools, priced, seeds, remaining, n, rng):
    teams = [t for ts in pools.values() for t in ts]
    grank = {t: defaultdict(int) for t in teams}
    place = {t: defaultdict(int) for t in teams}
    tot_w = defaultdict(float)
    tot_gd = defaultdict(float)
    tot_pd = defaultdict(float)
    labels = list(pools)

    def pair(a, b):
        if (a, b) in priced:
            return priced[(a, b)], False
        return priced[(b, a)], True

    def play(a, b, playoff=False):
        """(winner, games_a, games_b, pts_a, pts_b)."""
        rec, flip = pair(a, b)
        if playoff:                     # only the headline number matters
            p = rec["tree"]["p_win"]
            return (a if rng.random() < (1 - p if flip else p) else b), 0, 0, 0, 0
        ga = gb = pa = pb = 0
        for g in rec["games"]:
            p = 1 - g["p"] if flip else g["p"]
            a_won = rng.random() < p
            # cdfs are in the priced orientation; `flip` means `a` is the
            # SECOND team there, so a win for `a` reads off the lose-cdf.
            cdf = g["win"] if (a_won != flip) else g["lose"]
            s1, s2 = draw(cdf, rng.random())
            if flip:
                s1, s2 = s2, s1
            pa += s1
            pb += s2
            ga += a_won
            gb += not a_won
        if ga == gb == 2:
            p = 1 - rec["p_db"] if flip else rec["p_db"]
            a_won = rng.random() < p
            cdf = rec["db_win"] if (a_won != flip) else rec["db_lose"]
            s1, s2 = draw(cdf, rng.random())
            if flip:
                s1, s2 = s2, s1
            pa += s1
            pb += s2
            ga += a_won
            gb += not a_won
        return (a if ga > gb else b), ga, gb, pa, pb

    for _ in range(n):
        W, GD, PD = defaultdict(int), defaultdict(int), defaultdict(int)
        H = {}
        for a, b, w, ga, gb, pa, pb in seeds:
            W[w] += 1
            GD[a] += ga - gb
            GD[b] += gb - ga
            PD[a] += pa - pb
            PD[b] += pb - pa
            H[(a, b)] = 1 if w == a else -1
            H[(b, a)] = -H[(a, b)]
        for a, b in remaining:
            w, ga, gb, pa, pb = play(a, b)
            W[w] += 1
            GD[a] += ga - gb
            GD[b] += gb - ga
            PD[a] += pa - pb
            PD[b] += pb - pa
            H[(a, b)] = 1 if w == a else -1
            H[(b, a)] = -H[(a, b)]
        orders = {}
        for lab, ts in pools.items():
            order = sorted(ts, key=lambda t: (-W[t], -GD[t], -PD[t], rng.random()))
            out, i = [], 0
            while i < len(order):            # head-to-head resolves 2-way ties
                j = i
                while j + 1 < len(order) and W[order[j + 1]] == W[order[i]]:
                    j += 1
                block = order[i:j + 1]
                if len(block) == 2 and H.get((block[1], block[0])) == 1:
                    block.reverse()
                out.extend(block)
                i = j + 1
            orders[lab] = out
            for r, t in enumerate(out, 1):
                grank[t][r] += 1
                tot_w[t] += W[t]
                tot_gd[t] += GD[t]
                tot_pd[t] += PD[t]
        oa, ob = orders[labels[0]], orders[labels[1]]
        for rank, a, b in place_rule(oa, ob):
            win, lose = (a, b) if play(a, b, playoff=True)[0] == a else (b, a)
            place[win][2 * rank - 1] += 1
            place[lose][2 * rank] += 1
        for o in (oa, ob):                    # 5th/6th share 9-10 / 11-12
            for i in range(PLAYOFF_DEPTH, len(o)):
                place[o[i]][2 * i + 1] += 1

    return grank, place, tot_w, tot_gd, tot_pd


# ------------------------------------------------------------------- output

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=200_000)
    ap.add_argument("--days", type=int, default=5, help="lookahead window")
    ap.add_argument("--back", type=int, default=4, help="lookback window")
    ap.add_argument("--date", type=date.fromisoformat, default=None)
    ap.add_argument("--seed", type=int, default=20260730)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    today = args.date or date.today()
    c = PBClient()
    found = find_event(c, today, args.back, args.days)
    if not found:
        log.warning("no MLP event in the window")
        return
    guid, title, rows = found
    rr = [r for r in rows if r["bracket"] == "RR"]
    pools_by_uuid = defaultdict(set)
    for r in rr:
        pools_by_uuid[r["pool"]].add(r["team1"])
        pools_by_uuid[r["pool"]].add(r["team2"])
    if len(pools_by_uuid) != 2:
        log.warning("expected 2 groups, found %d — aborting", len(pools_by_uuid))
        return
    ordered = sorted(pools_by_uuid.items(), key=lambda kv: kv[0])
    pools = {GROUP_LABELS[i]: sorted(v) for i, (_, v) in enumerate(ordered)}
    uuid_of = {GROUP_LABELS[i]: k for i, (k, _) in enumerate(ordered)}
    teams = [t for ts in pools.values() for t in ts]
    log.info("%s: %d teams, groups %s", title,
             len(teams), {k: len(v) for k, v in pools.items()})

    vals, singles = mf.load_values(), mf.load_singles()
    rosters = mf.mlp_rosters(c, today)
    lineups, sources, cache = {}, {}, {}
    for t in teams:
        lu, src = mf.projected_lineup_for_team(c, t, today, rosters, vals, cache)
        lineups[t], sources[t] = lu, src
        log.info("  %-28s %s (roster %d)", t, src, len(rosters.get(t) or ()))
    priced = price_pairs(teams, lineups, vals, singles)
    log.info("priced %d of %d pairs", len(priced), len(teams) * (len(teams) - 1) // 2)

    seeds, remaining, unpriced = [], [], []
    for r in rr:
        a, b = r["team1"], r["team2"]
        if r["done"] and r["games1"] is not None:
            w = a if r["games1"] > r["games2"] else b
            seeds.append((a, b, w, r["games1"], r["games2"],
                          r["pts1"] or 0, r["pts2"] or 0))
        elif (a, b) in priced or (b, a) in priced:
            remaining.append((a, b))
        else:
            unpriced.append((a, b))
    if unpriced:
        log.warning("%d unpriced matchups dropped: %s", len(unpriced), unpriced)

    rng = random.Random(args.seed)
    n = args.sims
    grank, place, tw, tgd, tpd = simulate(pools, priced, seeds, remaining, n, rng)

    def exp_place(pl):
        # 9 and 11 are tier keys covering {9,10} and {11,12}
        return sum((k + 0.5 if k in (9, 11) else k) * v / n for k, v in pl.items())

    out = {
        "generated": str(date.today()), "event": title, "group_uuid": guid,
        "sims": n, "pools": pools, "pool_uuids": uuid_of,
        "played": len(seeds), "scheduled": len(remaining),
        "slate": [], "teams": {},
    }
    for r in sorted(rr, key=lambda r: (r["date"], r["start"] or "")):
        a, b = r["team1"], r["team2"]
        rec = priced.get((a, b)) or priced.get((b, a))
        flip = (a, b) not in priced
        p1 = None
        if rec:
            p1 = 1 - rec["tree"]["p_win"] if flip else rec["tree"]["p_win"]
        out["slate"].append({
            "date": r["date"], "start": r["start"], "group": next(
                (g for g, ts in pools.items() if a in ts), "?"),
            "team1": a, "team2": b, "p_team1": round(p1, 4) if p1 else None,
            "done": r["done"], "games1": r["games1"], "games2": r["games2"],
        })
    for t in teams:
        gr = {str(k): v / n for k, v in sorted(grank[t].items())}
        pl = {str(k): v / n for k, v in sorted(place[t].items())}
        out["teams"][t] = {
            "group": next(g for g, ts in pools.items() if t in ts),
            "lineup_source": sources[t],
            "lineup": {s: [[u, vals[u][0], round(vals[u][1], 4)] for u in pr]
                       for s, pr in (lineups[t] or {}).items()},
            "exp_wins": round(tw[t] / n, 3), "exp_game_diff": round(tgd[t] / n, 2),
            "exp_rally_diff": round(tpd[t] / n, 1),
            "group_rank": {k: round(v, 5) for k, v in gr.items()},
            "p_group_win": round(gr.get("1", 0.0), 5),
            "p_playoff": round(sum(v for k, v in gr.items()
                                   if int(k) <= PLAYOFF_DEPTH), 5),
            "place": {k: round(v, 5) for k, v in pl.items()},
            "p_title": round(pl.get("1", 0.0), 5),
            "p_final": round(pl.get("1", 0.0) + pl.get("2", 0.0), 5),
            "p_podium": round(sum(v for k, v in pl.items() if int(k) <= 3), 5),
            "exp_place": round(exp_place(place[t]), 3),
        }
    path = DATA / "event_forecast.json"
    path.write_text(json.dumps(out, indent=1))
    log.info("wrote %s", path)

    for g, ts in pools.items():
        print(f"\nGROUP {g}")
        for t in sorted(ts, key=lambda t: -out["teams"][t]["exp_wins"]):
            v = out["teams"][t]
            print(f"  {t:<26} W {v['exp_wins']:.2f}  P(1st) {v['p_group_win']:.3f}"
                  f"  P(top{PLAYOFF_DEPTH}) {v['p_playoff']:.3f}")
    print("\nFINAL PLACE")
    for t in sorted(teams, key=lambda t: out["teams"][t]["exp_place"]):
        v = out["teams"][t]
        print(f"  {t:<26} exp {v['exp_place']:5.2f}  title {v['p_title']:.3f}"
              f"  podium {v['p_podium']:.3f}")


if __name__ == "__main__":
    main()
