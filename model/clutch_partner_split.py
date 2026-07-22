"""Clutch for one player, split by whether a given PARTNER was on their team.

Motivating question (2026-07-22): Max Freeman's clutch — is it his, or is it
Ben Johns? Freeman plays most of his big matches alongside the GOAT, so his
"clutch on his own serve" might be riding a team that's ahead in every big
moment. This slices Freeman's serving rallies into WITH-partner vs
WITHOUT-partner and recomputes clutch on each, on the exact same footing as
model/big_points.py (same leverage DP, same within-game standardization,
same permutation null) so the z's are directly comparable to
data/clutch_players.csv.

It rebuilds the clutch rally records with match ids retained (big_points
aggregates them away), then partitions ONE player's records by whether the
named partner shared their side in that match.

Run:  python model/clutch_partner_split.py                      # Freeman w/o Johns
      python model/clutch_partner_split.py --player "Name" --partner "Name"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "web"))
import spec_shootout as sx          # noqa: E402
import big_points as bp             # noqa: E402
from sitelib.winprob import serve_probs, _table   # noqa: E402


def build_recs(games, players, chem):
    """Reproduce big_points.clutch()'s per-rally records, but keep the
    match id and side membership so we can slice by partner. Returns
    recs = [(server_uuid, match, levz, resid)] and match_sides =
    {match: (frozenset teamA_uuids, frozenset teamB_uuids)}."""
    raw = ROOT / "raw" / "match_logs"
    train = [g for g in games if sx.RECENT <= g["date"] < sx.SPLIT]
    by_match = defaultdict(list)
    for g in train:
        by_match[g["match"]].append(g)

    def eta_of(g):
        if any(u not in players for u in g["us"]):
            return None
        v = [players[u]["v"] for u in g["us"]]
        c1 = chem.get(frozenset((players[g["us"][0]]["name"],
                                 players[g["us"][1]]["name"])), 0.0)
        c2 = chem.get(frozenset((players[g["us"][2]]["name"],
                                 players[g["us"][3]]["name"])), 0.0)
        return (v[0] + v[1] + sx.GAMMA_V2 * abs(v[0] - v[1])
                - v[2] - v[3] - sx.GAMMA_V2 * abs(v[2] - v[3]) + c1 - c2)

    k_league = 0.443
    recs = []
    match_sides = {}
    for m, gs in by_match.items():
        p = raw / m[:2] / f"{m}.json"
        if not p.exists():
            continue
        body = json.loads(p.read_text())
        rows = body.get("data") if isinstance(body, dict) else body
        if not rows:
            continue
        sides = (frozenset(u.lower() for u in gs[0]["us"][:2]),
                 frozenset(u.lower() for u in gs[0]["us"][2:]))
        match_sides[m] = sides
        games_by_no = {g["gn"]: (g["s1"], g["s2"]) for g in gs}
        eta_by_no = {g["gn"]: eta_of(g) for g in gs}
        T_by_no = {g["gn"]: g["T"] for g in gs}
        seq = bp.reconstruct_states(rows, sides, games_by_no)
        if not seq:
            continue
        by_game = defaultdict(list)
        for s in seq:
            by_game[s[0]].append(s)
        for gno, rallies in by_game.items():
            e = eta_by_no.get(gno)
            if e is None:
                continue
            kA, kB = serve_probs(e, k_league)
            T = T_by_no[gno]
            V = _table(round(kA, 6), round(kB, 6), T, T + 40)
            levs, srv_expect = [], []
            for (_, a, b, state, su, side, won) in rallies:
                levs.append(bp.leverage_of(V, T, a, b, state, side == 0))
                srv_expect.append(kA if side == 0 else kB)
            levs = np.array(levs)
            if levs.std() < 1e-9:
                continue
            levz = (levs - levs.mean()) / levs.std()
            for (rly, lz, ke) in zip(rallies, levz, srv_expect):
                _, a, b, state, su, side, won = rly
                recs.append((su, m, float(lz), float(won) - ke))
    return recs, match_sides


def clutch_of(obs, seed=bp.SEED, n_perm=bp.N_PERM):
    """clutch, null_sd, z for a list of (levz, resid) pairs — identical to
    big_points, so directly comparable to data/clutch_players.csv."""
    lz = np.array([o[0] for o in obs])
    res = np.array([o[1] for o in obs])
    cl = float(np.mean(lz * res))
    rng = np.random.default_rng(seed)
    null = np.array([np.mean(rng.permutation(lz) * res) for _ in range(n_perm)])
    sd = float(null.std())
    return dict(n=len(obs), clutch=cl, null_sd=sd,
                z=cl / sd if sd > 0 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", default="Max Freeman")
    ap.add_argument("--partner", default="Ben Johns")
    args = ap.parse_args()

    games = sx.load_games()
    players, chem = sx.load_v2()

    # name -> uuid(s); pick the uuid that actually carries games
    by_name = defaultdict(list)
    for u, d in players.items():
        by_name[d["name"]].append(u)
    def uuid(nm):
        c = by_name.get(nm, [])
        if not c:
            sys.exit(f"no uuid for {nm!r}")
        return max(c, key=lambda u: players[u].get("games", 0))
    pu, ku = uuid(args.player), uuid(args.partner)

    recs, match_sides = build_recs(games, players, chem)

    # which matches had the partner on the player's OWN side?
    partner_matches, opp_matches = set(), set()
    for m, (A, B) in match_sides.items():
        pl = pu.lower(); pk = ku.lower()
        if pl in A and pk in A or pl in B and pk in B:
            partner_matches.add(m)
        elif pl in (A | B) and pk in (A | B):
            opp_matches.add(m)          # both in match but opposite sides

    mine = [(lz, res, m) for (su, m, lz, res) in recs if su == pu.lower()]
    allobs = [(lz, res) for lz, res, m in mine]
    with_p = [(lz, res) for lz, res, m in mine if m in partner_matches]
    without_p = [(lz, res) for lz, res, m in mine if m not in partner_matches]

    n_matches_all = len({m for _, _, m in mine})
    n_matches_with = len({m for _, _, m in mine if m in partner_matches})

    # population z's (same engine) for ranking the without-partner number
    pop = [(r["name"], float(r["z"])) for r in
           csv.DictReader((ROOT / "data" / "clutch_players.csv").open())]

    def rank_of(z):
        zs = sorted((zz for _, zz in pop), reverse=True)
        # where z would slot among the 182
        return sum(1 for x in zs if x > z) + 1, len(zs)

    print(f"CLUTCH SPLIT — {args.player}, with/without {args.partner} as partner")
    print(f"  {args.player} served {len(allobs)} rallies across {n_matches_all} "
          f"train matches ({n_matches_with} with {args.partner}).")
    if opp_matches:
        print(f"  ({len(opp_matches)} matches had {args.partner} as an opponent"
              " — not excluded; 'without' = not-partnered.)")
    print()
    hdr = f"  {'split':22s} {'rallies':>8} {'clutch':>9} {'z':>7}   rank (of 182, by z)"
    print(hdr)
    for label, obs in [(f"all (as in CSV)", allobs),
                       (f"WITH {args.partner}", with_p),
                       (f"WITHOUT {args.partner}", without_p)]:
        if not obs:
            print(f"  {label:22s} {'0':>8}  (no rallies)")
            continue
        r = clutch_of(obs)
        rk, N = rank_of(r["z"])
        note = "" if r["n"] >= bp.MIN_RALLIES_CL else "  (< 300 — noisy)"
        print(f"  {label:22s} {r['n']:>8} {r['clutch']:>+9.4f} {r['z']:>+7.2f}"
              f"   #{rk} of {N}{note}")


if __name__ == "__main__":
    main()
