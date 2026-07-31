"""V7: the grinder cohort IS recoverable.

A7 concluded the 10-name defensive-grinder cohort's membership is
"unrecoverable from the record" and reconstructed 17 players / 6,796 obs by
matching SURNAMES only.  But web/insights/wind/index.html names the two
ambiguous ones in full — "Callie Smith" and "Jorja Johnson" — which pins the
cohort to exactly 10 players.  Check whether that cohort reproduces the
published -0.012 /10 mph, p = 0.975, "across 5,134 games".

Deterministic; read-only.
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "model/weather_review"))
from v7_sock_verify import load, slope  # noqa: E402

COHORT = ["Pablo Tellez", "Gabriel Tardio", "Hayden Patriquin", "Rafa Hewett",
          "Federico Staksrud", "Dylan Frazier", "Callie Smith",
          "Catherine Parenteau", "Jorja Johnson", "Jessie Irvine"]


def main():
    players, names, _ = load()
    pool, per = [], []
    for nm in COHORT:
        pid = next((p for p, x in names.items() if x == nm), None)
        if pid is None or pid not in players:
            print(f"MISSING {nm}")
            continue
        rows = players[pid]
        per.append((nm, len(rows), slope([r[0] for r in rows],
                                         [r[1] for r in rows])[0] * 10))
        pool.extend(rows)
    for nm, n, s in sorted(per, key=lambda t: -t[2]):
        print(f"  {nm:24s} n={n:4d}  {s:+.4f}/10mph")
    xs = [r[0] for r in pool]
    ys = [r[1] for r in pool]
    b = slope(xs, ys)[0]
    ug = len({r[3] for r in pool})
    print(f"\npooled obs = {len(pool)}   distinct matches = {ug}")
    print(f"pooled slope {b*10:+.4f} /10 mph")

    for seed in (2222, 7, 99):
        rng = random.Random(seed)
        w = list(xs)
        ge = 0
        for _ in range(2000):
            rng.shuffle(w)
            ge += slope(w, ys)[0] >= b
        print(f"  one-sided perm p (seed {seed}) = {ge/2000:.3f}")

    # cluster (event) permutation for honesty
    ev = defaultdict(list)
    for r in pool:
        ev[r[2]].append(r)
    keys = list(ev.keys())
    blocks = [[x[0] for x in ev[k]] for k in keys]
    resid = [[x[1] for x in ev[k]] for k in keys]
    rng = random.Random(5)
    ge = 0
    N = 2000
    for _ in range(N):
        order = list(range(len(keys)))
        rng.shuffle(order)
        px, py = [], []
        for slot, src in enumerate(order):
            wb = blocks[src]
            for i, y in enumerate(resid[slot]):
                px.append(wb[i % len(wb)])
                py.append(y)
        ge += slope(px, py)[0] >= b
    print(f"  event-cluster perm p = {ge/N:.3f}  ({len(keys)} events)")


if __name__ == "__main__":
    main()
