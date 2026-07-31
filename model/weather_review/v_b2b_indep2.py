"""ADVERSARIAL B2b part 2:
  (A) correct cluster bootstrap on every DiD cell the tester reported
  (B) a MATCHED placebo — games 1-2 of the SAME matches that produced the
      decider — which holds the collider (match went the distance) fixed
  (C) a WITHIN-MATCH DiD: y_m = z2(decider) - mean z2(games 1..k) of the same
      match, then windy-vs-calm within event.  Differences out players,
      closeness, day, court, hour; the only thing left is the end switch.
  (D) skewness of the exact randomization null (why wild-t and permutation
      disagree)
"""
from __future__ import annotations

import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from v_b2b_indep import (ROOT, SCRATCH, rd, labels, context, bin_of, z2,  # noqa
                         paired, paired_cells, boot_correct, boot_dedup, fmt)


def rows(splits, ctx):
    """All games with a recoverable switch boundary, tagged switch/no-switch."""
    out = []
    for r in splits:
        mid, gn = r["match_id"], int(r["game_number"])
        m = ctx.get(mid)
        if not m:
            continue
        g = m["fmt"].get(gn)
        if g is None or g["scoring_format"] != "sideout_11":
            continue
        if m["tour"] == "MLP":
            if gn != 1:
                continue
            sw = True
        else:
            sw = (m["best_of"] == 3 and gn == 3) or (m["best_of"] == 5 and gn == 5)
        pre = int(r["pa_pre"]) + int(r["pb_pre"])
        post = int(r["pa_post"]) + int(r["pb_post"])
        if pre < 5 or post < 5:
            continue
        _, _, z = z2(int(r["pa_pre"]), pre, int(r["pa_post"]), post)
        strict = (r.get("seq_ok") == "1" and r.get("boundary_ok") == "1")
        out.append(dict(ev=m["event"], bin=bin_of(m), tour=m["tour"], mid=mid,
                        gn=gn, z=z, sw=sw, strict=strict))
    return out


def did_cells(sw_cells, ns_cells, w="fe"):
    return paired(sw_cells, w) - paired(ns_cells, w)


def did_boot(cs, cn, n=4000, seed=1, dedup=False):
    allev = sorted(set(cs) | set(cn))
    cells = {e: e for e in allev}

    def stat_mult(smp):
        a = {k: cs[e] for k, e in smp.items() if e in cs}
        b = {k: cn[e] for k, e in smp.items() if e in cn}
        if not a or not b:
            return float("nan")
        return paired(a, "fe") - paired(b, "fe")
    if dedup:
        return boot_dedup(cells, lambda c: stat_mult({i: e for i, e in enumerate(c)}),
                          n=n, seed=seed)[:2]
    return boot_correct(cells, stat_mult, n=n, seed=seed)[:2]


def did_rand(cs, cn, est, n=20000, seed=777):
    rng = random.Random(seed)
    ge = 0
    for _ in range(n):
        pa, pb = {}, {}
        for cell, dst in ((cs, pa), (cn, pb)):
            for e, (t, c) in cell.items():
                pool = t + c
                rng.shuffle(pool)
                dst[e] = (pool[:len(t)], pool[len(t):])
        if paired(pa, "fe") - paired(pb, "fe") >= est:
            ge += 1
    return (ge + 1) / (n + 1)


def main():
    reb = rd(SCRATCH / "rebuilt_splits.csv")
    L = labels()

    print("=" * 78)
    print("(A) every DiD cell the tester reported, with a CORRECT cluster boot")
    print("=" * 78)
    print(f"{'labels':14s} {'strict':6s} {'bin':9s} {'DiD':>7s} "
          f"{'correct boot95':>22s} {'dedup(tester) boot95':>22s} {'randp':>7s}")
    for lab in ("published", "corrected_all", "corrected_hi"):
        ctx = context(L[lab])
        R = rows(reb, ctx)
        for strict in (False, True):
            RR = [r for r in R if (r["strict"] or not strict)]
            sw = [r for r in RR if r["sw"]]
            ns = [r for r in RR if not r["sw"] and r["tour"] != "MLP"]
            for b in ("moderate", "windy"):
                cs = paired_cells(sw, b)
                cn = paired_cells(ns, b)
                if not cs or not cn:
                    continue
                est = did_cells(cs, cn)
                lo, hi = did_boot(cs, cn, seed=1)
                dlo, dhi = did_boot(cs, cn, seed=1, dedup=True)
                p = did_rand(cs, cn, est, n=10000)
                print(f"{lab:14s} {str(strict):6s} {b:9s} {est:+7.3f} "
                      f"[{lo:+7.3f},{hi:+7.3f}] [{dlo:+7.3f},{dhi:+7.3f}] {p:7.4f}")

    print()
    print("=" * 78)
    print("(B) MATCHED placebo: games 1-2 of the SAME (3-game) matches only")
    print("=" * 78)
    for lab in ("published", "corrected_all"):
        ctx = context(L[lab])
        R = rows(reb, ctx)
        dec_matches = {r["mid"] for r in R if r["sw"] and r["tour"] == "PPA"}
        sw = [r for r in R if r["sw"]]
        ns_all = [r for r in R if not r["sw"] and r["tour"] != "MLP"]
        ns_match = [r for r in ns_all if r["mid"] in dec_matches]
        for nm, ns in (("all non-deciders", ns_all),
                       ("non-deciders OF DECIDED matches", ns_match)):
            cs, cn = paired_cells(sw, "windy"), paired_cells(ns, "windy")
            est = did_cells(cs, cn)
            lo, hi = did_boot(cs, cn, seed=1)
            p = did_rand(cs, cn, est, n=10000)
            print(f"  {lab:14s} {nm:34s} placebo FE={paired(cn,'fe'):+.3f} "
                  f"(n_windy={sum(len(t) for t,_ in cn.values())}) "
                  f"DiD={est:+.3f} [{lo:+.3f},{hi:+.3f}] randp={p:.4f}")

    print()
    print("=" * 78)
    print("(C) WITHIN-MATCH DiD  y_m = z2(decider) - mean z2(earlier games)")
    print("=" * 78)
    for lab in ("published", "corrected_all"):
        ctx = context(L[lab])
        R = rows(reb, ctx)
        bym = defaultdict(dict)
        for r in R:
            if r["tour"] != "PPA":
                continue
            bym[r["mid"]][r["gn"]] = r
        units = []
        for mid, gs in bym.items():
            dec = [r for r in gs.values() if r["sw"]]
            pre = [r for r in gs.values() if not r["sw"]]
            if len(dec) != 1 or not pre:
                continue
            d = dec[0]
            y = d["z"] - sum(p["z"] for p in pre) / len(pre)
            units.append(dict(ev=d["ev"], bin=d["bin"], z=y))
        cnt = defaultdict(int)
        for u in units:
            cnt[u["bin"]] += 1
        print(f"  {lab}: n_matches={len(units)} bins={dict(cnt)}")
        for b in ("moderate", "windy"):
            cc = paired_cells(units, b)
            if not cc:
                continue
            est = paired(cc, "fe")
            lo, hi, _ = boot_correct(cc, lambda c: paired(c, "fe"), n=4000, seed=1)
            # randomization
            rng = random.Random(31337)
            ge = 0
            N = 20000
            for _ in range(N):
                sh = {}
                for e, (t, c) in cc.items():
                    pool = t + c
                    rng.shuffle(pool)
                    sh[e] = (pool[:len(t)], pool[len(t):])
                if paired(sh, "fe") >= est:
                    ge += 1
            nt = sum(len(t) for t, _ in cc.values())
            print(f"    {b:9s} G={len(cc):3d} n_t={nt:4d} within-match DiD "
                  f"FE={est:+.3f} [{lo:+.3f},{hi:+.3f}] ATT={paired(cc,'att'):+.3f} "
                  f"randp={(ge+1)/(N+1):.4f}")

    print()
    print("=" * 78)
    print("(D) shape of the exact randomization null (published labels, windy)")
    print("=" * 78)
    ctx = context(L["published"])
    R = rows(reb, ctx)
    sw = [r for r in R if r["sw"]]
    cc = paired_cells(sw, "windy")
    for w in ("fe", "att"):
        obs = paired(cc, w)
        rng = random.Random(2024)
        vals = []
        for _ in range(20000):
            sh = {}
            for e, (t, c) in cc.items():
                pool = t + c
                rng.shuffle(pool)
                sh[e] = (pool[:len(t)], pool[len(t):])
            vals.append(paired(sh, w))
        m = statistics.mean(vals)
        sd = statistics.pstdev(vals)
        sk = sum((v - m) ** 3 for v in vals) / len(vals) / sd ** 3
        vals.sort()
        q = lambda p: vals[int(p * len(vals))]
        print(f"  {w}: obs={obs:+.3f} null mean={m:+.4f} sd={sd:.3f} skew={sk:+.2f} "
              f"| null q2.5={q(.025):+.3f} q97.5={q(.975):+.3f} q95={q(.95):+.3f}")
        print(f"       symmetric-normal p would be {1 - 0.5*(1+math_erf((obs-m)/sd/2**.5)):.4f}"
              f"; exact p={sum(1 for v in vals if v>=obs)/len(vals):.4f}")


def math_erf(x):
    import math
    return math.erf(x)


if __name__ == "__main__":
    main()
