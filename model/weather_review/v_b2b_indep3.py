"""ADVERSARIAL B2b part 3: what the paired contrast is actually made of."""
from __future__ import annotations

import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v_b2b_indep import (ROOT, rd, labels, context, bin_of, z2, paired,  # noqa
                         paired_cells, boot_correct)


def units(splits, ctx):
    out = []
    for r in splits:
        mid, gn = r["match_id"], int(r["game_number"])
        m = ctx.get(mid)
        if not m:
            continue
        g = m["fmt"].get(gn)
        if g is None:
            continue
        if m["tour"] == "MLP":
            if gn != 1:
                continue
        elif not ((m["best_of"] == 3 and gn == 3) or (m["best_of"] == 5 and gn == 5)):
            continue
        pre = int(r["pa_pre"]) + int(r["pb_pre"])
        post = int(r["pa_post"]) + int(r["pb_post"])
        if pre < 5 or post < 5:
            continue
        sq, noise, z = z2(int(r["pa_pre"]), pre, int(r["pa_post"]), post)
        out.append(dict(ev=m["event"], bin=bin_of(m), wind=m["wind"], z=z,
                        sq=sq, noise=noise, date=m["date"], tour=m["tour"],
                        mid=mid))
    return out


def main():
    L = labels()
    ctx = context(L["published"])
    U = units(rd(ROOT / "data/decider_splits.csv"), ctx)
    cells = paired_cells(U, "windy")
    print("per-event composition of the 15 paired events (published labels)")
    print(f"{'event':10s} {'n_windy':>7s} {'n_calm':>7s} {'d_e':>7s} "
          f"{'windy days':>10s} {'calm days':>9s} {'shared days':>11s}")
    tw = tc = 0
    for e, (t, c) in sorted(cells.items(), key=lambda kv: -len(kv[1][0])):
        wd = {u["date"] for u in U if u["ev"] == e and u["bin"] == "windy"}
        cd = {u["date"] for u in U if u["ev"] == e and u["bin"] == "calm"}
        tw += len(t)
        tc += len(c)
        print(f"{e[:8]:10s} {len(t):7d} {len(c):7d} "
              f"{sum(t)/len(t)-sum(c)/len(c):+7.3f} {len(wd):10d} {len(cd):9d} "
              f"{len(wd & cd):11d}")
    print(f"TOTAL windy={tw} calm={tc}  (the tester's note claims 1,636 calm)")

    W = [u for u in U if u["bin"] == "windy"]
    C = [u for u in U if u["bin"] == "calm"]
    print(f"\nmean binomial noise: windy={statistics.mean(u['noise'] for u in W):.4f} "
          f"calm={statistics.mean(u['noise'] for u in C):.4f} "
          f"all={statistics.mean(u['noise'] for u in U):.4f}")
    print(f"distinct (event,date) cells: windy="
          f"{len({(u['ev'],u['date']) for u in W})} calm="
          f"{len({(u['ev'],u['date']) for u in C})}")

    # --- DAY-paired contrast: same event AND same calendar day
    T, Cc = defaultdict(list), defaultdict(list)
    for u in U:
        k = (u["ev"], u["date"])
        if u["bin"] == "windy":
            T[k].append(u["z"])
        elif u["bin"] == "calm":
            Cc[k].append(u["z"])
    daycells = {k: (T[k], Cc[k]) for k in T if k in Cc}
    if daycells:
        nt = sum(len(t) for t, _ in daycells.values())
        nc = sum(len(c) for _, c in daycells.values())
        # cluster on EVENT even though pairing is on day
        byev = defaultdict(dict)
        for (e, d), v in daycells.items():
            byev[e][(e, d)] = v

        def st(sample):
            vals = sample.values() if isinstance(sample, dict) else sample
            m = {}
            for i, dd in enumerate(vals):
                for k, v in dd.items():
                    m[(i, k)] = v
            return paired(m, "fe")
        est = st(list(byev.values()))
        lo, hi, _ = boot_correct(byev, st, n=4000, seed=1)
        rng = random.Random(5150)
        ge = 0
        N = 20000
        for _ in range(N):
            sh = {}
            for k, (t, c) in daycells.items():
                pool = t + c
                rng.shuffle(pool)
                sh[k] = (pool[:len(t)], pool[len(t):])
            if paired(sh, "fe") >= est:
                ge += 1
        print(f"\nSAME-EVENT-SAME-DAY paired contrast (wind varies by HOUR): "
              f"cells={len(daycells)} events={len(byev)} windy_n={nt} calm_n={nc} "
              f"FE={est:+.3f} [{lo:+.3f},{hi:+.3f}] randp={(ge+1)/(N+1):.4f}")
    else:
        print("\nNo event-day cell contains both a windy and a calm game.")

    # --- ratio-of-means paired (different estimand / aggregation)
    Tn, Cn = defaultdict(lambda: [0.0, 0.0]), defaultdict(lambda: [0.0, 0.0])
    for u in U:
        if u["bin"] == "windy":
            Tn[u["ev"]][0] += u["sq"]
            Tn[u["ev"]][1] += u["noise"]
        elif u["bin"] == "calm":
            Cn[u["ev"]][0] += u["sq"]
            Cn[u["ev"]][1] += u["noise"]
    ks = [e for e in Tn if e in Cn]
    num = den = 0.0
    for e in ks:
        w = 1.0
        num += w * (Tn[e][0] / Tn[e][1] - Cn[e][0] / Cn[e][1])
        den += w
    print(f"paired ratio-of-means (Sum sq / Sum noise), unit weights, "
          f"{len(ks)} events: {num/den:+.3f}")


if __name__ == "__main__":
    main()
