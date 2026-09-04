"""value_cap/shapley_value.py -- context-averaged player value (Shapley-style).

    python value_cap/shapley_value.py [--samples 3000]   -> player_value_shapley.csv

Phase 1's V puts each player into a roster of REPLACEMENT players and asks
how much the tie-win probability moves. That is a single, extreme context:
the star's doubles partner is the #60 player, so the weakest-link gap
penalty (race.py GAMMA * |gap|) is at its largest and damps the star most.
The damping grows with the star's own value, which is a candidate
explanation for why the alpha that reconciles an indifference pair rises
with the anchor's rank (phase2_joint_pool.py).

Here the context is the league itself: for player i of gender g, draw a
partial roster of 2 same-gender + 3 other-gender players uniformly from
the priced pool (top 60 per gender = exactly what 20 teams roster), draw
an opponent roster the same way, and take

    phi_i = E[ P(partial + i beats opp) - P(partial + repl_g beats opp) ]

repl_g = doubles #60 of gender g, same replacement person as Phase 1.
The same partial rosters and opponents are used for every player of a
gender (common random numbers), so differences between players are much
less noisy than the per-player Monte Carlo sd suggests.

If tie strength were additive in phi, pricing proportional to phi
(alpha = 1) would make equal-strength rosters cost the same by
construction; any alpha != 1 needed after this is genuine curvature, not
a measurement-context artifact.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

from phase1_value_model import ROOT, load_doubles, load_singles, tie_win_prob
from phase2_price_model import load_pool

REPL_RANK = 60
OUT = ROOT / "value_cap" / "player_value_shapley.csv"


def main(n_samples=3000, seed=1):
    doubles = load_doubles()
    singles = load_singles()
    pool = load_pool()
    ids = {g: [pid for pid, _, _ in pool[g]] for g in ("M", "F")}
    ranked = {g: sorted((u for u in doubles if doubles[u]["gender"] == g),
                        key=lambda u: -doubles[u]["v"]) for g in ("M", "F")}
    repl = {g: ranked[g][REPL_RANK - 1] for g in ("M", "F")}

    rng = random.Random(seed)
    # common random numbers: one draw set per gender-of-subject
    draws = {}
    for g in ("M", "F"):
        og = "F" if g == "M" else "M"
        rows = []
        for _ in range(n_samples):
            same = rng.sample(ids[g], 2)
            other = rng.sample(ids[og], 3)
            opp = tuple(rng.sample(ids["M"], 3) + rng.sample(ids["F"], 3))
            rows.append((tuple(same), tuple(other), opp))
        draws[g] = rows

    out = []
    for g in ("M", "F"):
        base = []
        for same, other, opp in draws[g]:
            base.append(tie_win_prob(same + (repl[g],) + other, opp, doubles, singles))
        for pid, name, v_total in pool[g]:
            diffs = []
            for (same, other, opp), b in zip(draws[g], base):
                if pid in same or pid in opp:
                    continue      # subject already present; skip the draw
                p = tie_win_prob(same + (pid,) + other, opp, doubles, singles)
                diffs.append(p - b)
            n = len(diffs)
            mean = sum(diffs) / n
            sd = (sum((d - mean) ** 2 for d in diffs) / (n - 1)) ** 0.5
            out.append({"player_id": pid, "full_name": name, "gender": g,
                        "V_total": f"{v_total:.5f}", "phi": f"{mean:.5f}",
                        "phi_se": f"{sd / n ** 0.5:.5f}", "n": n})
            print(f"{name:28s} {g}  V_total {v_total:+.4f}  phi {mean:+.4f} ± {sd/n**0.5:.4f}")

    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {OUT} ({len(out)} players)")


if __name__ == "__main__":
    n = 3000
    if "--samples" in sys.argv:
        n = int(sys.argv[sys.argv.index("--samples") + 1])
    main(n)
