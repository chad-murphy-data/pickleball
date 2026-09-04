"""value_cap/phase1_value_model.py -- Phase 1 first cut: expected-ties-won
value (V) per player, dyad- and role-aware.

    python value_cap/phase1_value_model.py

V(player) = P(a reference roster with this player wins a tie against a
mirror-image replacement-level opponent) minus 0.5 -- the same reference
roster with a replacement-level player of the same gender in their spot
plays the mirror opponent to an exact coin flip by construction, so 0.5
is the true zero point.

Reuses the production engine exactly (web/sitelib/race.py: GAMMA,
team_eta, game_win_prob_uncertain, race_dist; the K_DB_SINGLES
DreamBreaker-gap model from web/make_forecast.py / model/db_model.md) --
no new win-probability math here. P(win matchup) = P(win >=3 of 4 games)
+ P(2-2) * P(DB), same formula make_forecast.py prices real matchups with.

Two lineups are drawn independently from the same 6-player roster, which
is the whole point: the regular-discipline foursome (WD/MD/MXD1/MXD2) is
the top 2W+2M by DOUBLES value, and the DreamBreaker foursome is the top
2W+2M by SINGLES value -- confirmed against the real 2026 meta (see
phase0_bench_value.md): teams choose their own DB foursome, not
constrained to that tie's other four starters. A player can qualify for
one lineup, both, or neither -- that's what lets a doubles-middling,
singles-elite player (Chris Haworth, Brooklyn, 2026) carry real V through
the DB channel alone, with no special-casing needed. V_regular and V_db
below isolate each channel by forcing the player out of the other
lineup's selection pool.

First-cut assumptions, open to revision (see phase0_bench_value.md --
same open items, not new ones):
  - Replacement level = the Nth-best available player of that gender by
    DOUBLES value, N = (real 2026 franchise count) x 3 = 20 x 3 = 60th.
    20 excludes the four non-franchise entries in mlp_matchups_2026.csv
    (College All-Stars, Team Australia/Canada/Europe). Confirm team count
    for next season.
  - The replacement PERSON is an actual player (the 60th-ranked by
    doubles value) standing in for all 5 filler slots at once -- their
    real singles value comes along for the ride as the replacement DB
    contribution too, rather than maintaining a separate synthetic
    singles-replacement level or 5 distinct replacement individuals.
  - Rosterable pool = 20+ tracked games (same filter as Phase 0's
    talent-cliff read). Includes some players who aren't actually
    available for 2026 (e.g. retired) -- a real Phase 1 should
    cross-check against actual 2026 rosters the way build_site.py does.
  - Combined game uncertainty = quadrature sum of the 4 contributing
    players' own value_now_sd, ignoring the weakest-link term's effect on
    variance and any cross-player covariance. Fine for a first pass.
  - No injury/absence draw yet (that's the phase 2/3 monkey wrench from
    the design discussion) -- this V assumes the player always plays
    whichever role helps most, every tie.

Don't read the output as a finished valuation. It's a first pass to see
whether the shape looks right, per the project's own working rule:
nothing here was tuned to make any strategy look dominant or not.
"""
from __future__ import annotations

import csv
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import (GAMMA, game_win_prob_uncertain, race_dist,  # noqa: E402
                          set_calibration, sigmoid, team_eta)

DATA = ROOT / "data"
CAL = json.loads((ROOT / "web" / "calibration.json").read_text())
set_calibration(CAL["a"], CAL["b"], CAL["eps"])

SINGLES_IMPUTE = json.loads((ROOT / "model" / "singles_model.json").read_text())["impute"]
K_DB_SINGLES = 0.42          # model/db_model.md v2 fit
MIN_GAMES = 20               # rosterable-pool filter, matches Phase 0
N_TEAMS = 20                 # real 2026 franchises; excludes All-Star/Team X entries
T_GAME = 11                  # MLP regular-discipline games: sideout_11


def load_doubles():
    out = {}
    for r in csv.DictReader((DATA / "v2_players.csv").open()):
        if int(r["games"]) < MIN_GAMES or r["gender"] not in ("M", "F"):
            continue
        out[r["player_id"]] = {
            "name": r["full_name"], "gender": r["gender"],
            "v": float(r["value_now_mean"]), "sd": float(r["value_now_sd"]),
        }
    return out


def load_singles():
    return {r["player_id"]: float(r["singles_value"])
            for r in csv.DictReader((DATA / "singles_players.csv").open())}


def singles_of(pid, doubles, singles):
    if pid in singles:
        return singles[pid]
    imp = SINGLES_IMPUTE[doubles[pid]["gender"]]
    return imp["a"] + imp["b"] * doubles[pid]["v"]


def best2(pids, key):
    return sorted(pids, key=lambda u: -key(u))[:2]


def mixed_split(women, men, value_of):
    """Same weakest-link pairing rule as make_forecast.best_lineup."""
    def pair_v(a, b):
        va, vb = value_of(a), value_of(b)
        return va + vb + GAMMA * abs(va - vb)
    w1, w2 = women
    m1, m2 = men
    if pair_v(w1, m1) + pair_v(w2, m2) >= pair_v(w1, m2) + pair_v(w2, m1):
        return (w1, m1), (w2, m2)
    return (w1, m2), (w2, m1)


def game_prob(pair_a, pair_b, doubles):
    v = lambda u: doubles[u]["v"]
    sd = lambda u: doubles[u]["sd"]
    eta = team_eta(v(pair_a[0]), v(pair_a[1]), v(pair_b[0]), v(pair_b[1]))
    combined_sd = sum(sd(u) ** 2 for u in pair_a + pair_b) ** 0.5
    return game_win_prob_uncertain(eta, combined_sd, T_GAME)


def db_prob(db_a, db_b, doubles, singles):
    s = lambda u: singles_of(u, doubles, singles)
    gap = sum(s(u) for u in db_a) / len(db_a) - sum(s(u) for u in db_b) / len(db_b)
    return race_dist(round(sigmoid(K_DB_SINGLES * gap), 4), 21)["p_win"]


def tie_win_prob(roster_a, roster_b, doubles, singles, exclude_from_db=None):
    """roster_* = 6 player_ids (3M+3W). Regular lineup = top2W+2M by doubles
    value; DB lineup = top2W+2M by singles value -- drawn independently.
    exclude_from_db drops one roster_a player_id from the DB lineup's
    candidate pool, to isolate that player's non-DB (regular) value."""
    def split(roster):
        women = [u for u in roster if doubles[u]["gender"] == "F"]
        men = [u for u in roster if doubles[u]["gender"] == "M"]
        return women, men

    wa, ma = split(roster_a)
    wb, mb = split(roster_b)

    rw = best2(wa, lambda u: doubles[u]["v"])
    rm = best2(ma, lambda u: doubles[u]["v"])
    ow = best2(wb, lambda u: doubles[u]["v"])
    om = best2(mb, lambda u: doubles[u]["v"])

    (wA1, mA1), (wA2, mA2) = mixed_split(rw, rm, lambda u: doubles[u]["v"])
    (wB1, mB1), (wB2, mB2) = mixed_split(ow, om, lambda u: doubles[u]["v"])

    game_ps = [
        game_prob(tuple(rw), tuple(ow), doubles),
        game_prob(tuple(rm), tuple(om), doubles),
        game_prob((wA1, mA1), (wB1, mB1), doubles),
        game_prob((wA2, mA2), (wB2, mB2), doubles),
    ]

    p_wins3plus = p_2_2 = 0.0
    for outcome in product([0, 1], repeat=4):
        p = 1.0
        for won, pg in zip(outcome, game_ps):
            p *= pg if won else (1 - pg)
        wins = sum(outcome)
        if wins >= 3:
            p_wins3plus += p
        elif wins == 2:
            p_2_2 += p

    db_pool_w = [u for u in wa if u != exclude_from_db]
    db_pool_m = [u for u in ma if u != exclude_from_db]
    s = lambda u: singles_of(u, doubles, singles)
    db_a = best2(db_pool_w, s) + best2(db_pool_m, s)
    db_b = best2(wb, s) + best2(mb, s)
    p_db = db_prob(db_a, db_b, doubles, singles)

    return p_wins3plus + p_2_2 * p_db


def main():
    doubles = load_doubles()
    singles = load_singles()

    replacement = {}
    for g in ("M", "F"):
        pool = sorted((u for u in doubles if doubles[u]["gender"] == g),
                      key=lambda u: -doubles[u]["v"])
        if len(pool) < N_TEAMS * 3:
            raise SystemExit(f"only {len(pool)} tracked {g} players, need {N_TEAMS*3}")
        replacement[g] = pool[N_TEAMS * 3 - 1]

    opp_roster = [replacement["F"]] * 3 + [replacement["M"]] * 3
    baseline = tie_win_prob(opp_roster, opp_roster, doubles, singles)
    assert abs(baseline - 0.5) < 1e-9, f"replacement-vs-replacement should be 0.5, got {baseline}"

    rows = []
    for pid, info in doubles.items():
        g = info["gender"]
        other = "F" if g == "M" else "M"
        roster = [pid] + [replacement[g]] * 2 + [replacement[other]] * 3

        v_total = tie_win_prob(roster, opp_roster, doubles, singles) - 0.5
        v_reg = tie_win_prob(roster, opp_roster, doubles, singles,
                              exclude_from_db=pid) - 0.5
        rows.append({
            "player_id": pid, "full_name": info["name"], "gender": g,
            "doubles_value": round(info["v"], 4),
            "singles_value": round(singles_of(pid, doubles, singles), 4),
            "V_total": round(v_total, 5), "V_regular": round(v_reg, 5),
            "V_db": round(v_total - v_reg, 5),
        })

    rows.sort(key=lambda r: -r["V_total"])
    out_path = ROOT / "value_cap" / "player_value.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"replacement M (rank {N_TEAMS*3}): {doubles[replacement['M']]['name']} "
          f"v={doubles[replacement['M']]['v']:.3f}")
    print(f"replacement F (rank {N_TEAMS*3}): {doubles[replacement['F']]['name']} "
          f"v={doubles[replacement['F']]['v']:.3f}")
    print(f"wrote {len(rows)} rows to {out_path}\n")

    for g in ("M", "F"):
        print(f"--- top 10 V_total, {g} ---")
        for r in [r for r in rows if r["gender"] == g][:10]:
            print(f"{r['full_name']:25s} V={r['V_total']:+.4f} "
                  f"(reg {r['V_regular']:+.4f} / db {r['V_db']:+.4f})  "
                  f"doubles={r['doubles_value']:+.3f} singles={r['singles_value']:+.3f}")
        print()

    print("--- top 10 V_db (DB-channel value, isolated) ---")
    for r in sorted(rows, key=lambda r: -r["V_db"])[:10]:
        print(f"{r['full_name']:25s} V_db={r['V_db']:+.4f}  V_total={r['V_total']:+.4f}  "
              f"doubles={r['doubles_value']:+.3f} singles={r['singles_value']:+.3f}")
    print()

    for r in rows:
        if "Haworth" in r["full_name"]:
            print(f"Haworth: V_total={r['V_total']:+.4f}  "
                  f"V_regular={r['V_regular']:+.4f}  V_db={r['V_db']:+.4f}")


if __name__ == "__main__":
    main()
