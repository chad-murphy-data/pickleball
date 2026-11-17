"""value_cap/shapley_value.py -- context-averaged player value (Shapley-style)
and the self-consistent priced pool.

    python value_cap/shapley_value.py [--samples 3000] [--candidates 80]
        -> player_value_shapley.csv  (phi, phi_se, in_pool, pool_rank per player)

Phase 1's V puts each player into a roster of REPLACEMENT players and asks
how much the tie-win probability moves. That is a single, extreme context:
the star's doubles partner is the #60 player, so the weakest-link gap
penalty (race.py GAMMA * |gap|) is at its largest and damps the star most,
and the damping grows with the star's own value. phase2_pricing.md §2-3
shows what that does to prices.

Here the context is the league itself. For player i of gender g, draw a
partial roster of 2 same-gender + 3 other-gender players uniformly from
the priced pool (60 per gender = what 20 teams roster), draw an opponent
roster the same way, and take

    phi_i = E[ P(partial + i beats opp) - P(partial + repl_g beats opp) ]

repl_g = doubles #60 of gender g (Phase 1's replacement person). The same
partial rosters and opponents are used for every player of a gender
(common random numbers), so differences between players are much less
noisy than the per-player se suggests.

The pool is part of the definition, so it is solved for: start from the
top 60 by V_total, compute phi for a wider candidate set (top
--candidates by V_total), re-pick the top 60 by phi, and repeat until the
pool reproduces itself. Players outside the final pool keep the phi they
were measured with in the last iteration (they are priced at the floor
by phase2_pricing.py regardless).

If tie strength were additive in phi, pricing proportional to phi
(alpha = 1) would make equal-strength rosters cost the same by
construction; any alpha != 1 needed after this is genuine curvature, not
a measurement-context artifact.
"""
from __future__ import annotations

import csv
import random
import sys

from phase1_value_model import ROOT, load_doubles, load_singles, tie_win_prob
from pool import POOL_SIZE

REPL_RANK = 60
OUT = ROOT / "value_cap" / "player_value_shapley.csv"
MAX_ITER = 4


def phi_for(candidates, pool_ids, doubles, singles, repl, n_samples, seed):
    """candidates: gender -> [pid]; pool_ids: gender -> [pid] (context).
    Returns pid -> (phi, se, n)."""
    rng = random.Random(seed)
    out = {}
    for g in ("M", "F"):
        og = "F" if g == "M" else "M"
        draws = []
        for _ in range(n_samples):
            same = tuple(rng.sample(pool_ids[g], 2))
            other = tuple(rng.sample(pool_ids[og], 3))
            opp = tuple(rng.sample(pool_ids["M"], 3) + rng.sample(pool_ids["F"], 3))
            draws.append((same, other, opp))
        base = [tie_win_prob(same + (repl[g],) + other, opp, doubles, singles)
                for same, other, opp in draws]
        for pid in candidates[g]:
            diffs = [tie_win_prob(same + (pid,) + other, opp, doubles, singles) - b
                     for (same, other, opp), b in zip(draws, base)
                     if pid not in same and pid not in opp]
            n = len(diffs)
            mean = sum(diffs) / n
            sd = (sum((d - mean) ** 2 for d in diffs) / (n - 1)) ** 0.5
            out[pid] = (mean, sd / n ** 0.5, n)
    return out


def main(n_samples=3000, n_candidates=80, seed=1):
    doubles = load_doubles()
    singles = load_singles()
    ranked = {g: sorted((u for u in doubles if doubles[u]["gender"] == g),
                        key=lambda u: -doubles[u]["v"]) for g in ("M", "F")}
    repl = {g: ranked[g][REPL_RANK - 1] for g in ("M", "F")}

    v_total, name, gender = {}, {}, {}
    for r in csv.DictReader((ROOT / "value_cap" / "player_value.csv").open()):
        v_total[r["player_id"]] = float(r["V_total"])
        name[r["player_id"]] = r["full_name"]
        gender[r["player_id"]] = r["gender"]
    by_v = {g: sorted((u for u in v_total if gender[u] == g), key=lambda u: -v_total[u])
            for g in ("M", "F")}
    candidates = {g: by_v[g][:n_candidates] for g in ("M", "F")}
    pool_ids = {g: by_v[g][:POOL_SIZE] for g in ("M", "F")}

    for it in range(MAX_ITER):
        phi = phi_for(candidates, pool_ids, doubles, singles, repl, n_samples, seed + it)
        new_pool = {g: sorted(candidates[g], key=lambda u: -phi[u][0])[:POOL_SIZE] for g in ("M", "F")}
        churn = {g: (set(new_pool[g]) - set(pool_ids[g]), set(pool_ids[g]) - set(new_pool[g]))
                 for g in ("M", "F")}
        print(f"iteration {it}: pool churn  M +{len(churn['M'][0])}/-{len(churn['M'][1])}  "
              f"F +{len(churn['F'][0])}/-{len(churn['F'][1])}")
        for g in ("M", "F"):
            for u in sorted(churn[g][0], key=lambda u: -phi[u][0]):
                print(f"    in  {g} {name[u]:26s} phi {phi[u][0]:+.4f}  V_total {v_total[u]:+.4f}")
            for u in sorted(churn[g][1], key=lambda u: -phi[u][0]):
                print(f"    out {g} {name[u]:26s} phi {phi[u][0]:+.4f}  V_total {v_total[u]:+.4f}")
        stable = all(not churn[g][0] and not churn[g][1] for g in ("M", "F"))
        pool_ids = new_pool
        if stable:
            print("pool is self-consistent")
            break
    else:
        print(f"WARNING: pool did not stabilise in {MAX_ITER} iterations; using the last one")

    rows = []
    for g in ("M", "F"):
        order = sorted(candidates[g], key=lambda u: -phi[u][0])
        for rank_, u in enumerate(order, 1):
            m, se, n = phi[u]
            rows.append({"player_id": u, "full_name": name[u], "gender": g,
                         "V_total": f"{v_total[u]:.5f}", "phi": f"{m:.5f}",
                         "phi_se": f"{se:.5f}", "n": n,
                         "in_pool": int(u in pool_ids[g]), "pool_rank": rank_})
            if rank_ <= 12 or u in pool_ids[g] and rank_ > POOL_SIZE - 3:
                print(f"{g} #{rank_:2d} {name[u]:26s} V_total {v_total[u]:+.4f}  phi {m:+.4f} ± {se:.4f}")
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} players, {2*POOL_SIZE} in pool)")


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--samples") + 1]) if "--samples" in sys.argv else 3000
    c = int(sys.argv[sys.argv.index("--candidates") + 1]) if "--candidates" in sys.argv else 80
    main(n, c)
