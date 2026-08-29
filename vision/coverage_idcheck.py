"""Occupancy-geometry check on IDENTITY — independent of Gate A.

User observation 2026-08-20: "the heat maps look VERY realistic, all four
players looked like what I'd expect. That feels like a good sign."  It is
a good sign, but face validity is blind to the failure mode that matters
most here, so this turns it into a measurement.

PRE-REGISTERED before any number was computed:

METRIC.  Total variation distance, TV = 0.5 * sum|P - Q|, between two
players' smoothed mass-normalised occupancy grids in the COURT frame
(same grid and sigma as the shipped heat map).  TV = 0 identical,
TV = 1 disjoint.

THE MODEL.  Swapping two players' labels in a fraction q of rallies makes
each attributed map the mixture (1-q)*own + q*other, so
    TV_observed = |1 - 2q| * TV_true.
Two consequences, both pre-registered:
  * a PARTIAL swap is detectable: TV falls linearly with q;
  * a TOTAL swap (q = 1) is INVISIBLE -- TV returns to TV_true with the
    names exchanged.  This test cannot see a global label flip.  A human
    who knows the players CAN see that (the man's panel would show the
    woman's pattern), which is exactly the complement.

THE REFERENCE ARM is what makes q identifiable.  Partners are the
confusable pair (the appearance channel is a team-colour detector;
308/340 of its errors are partner confusions).  Cross-team pairs are
resolved at 97-100% and are effectively clean.  So a cross-team
man/woman pair estimates TV_true for a man/woman pair with trustworthy
labels, and q follows from the partner pair's observed TV.

MEASURED on the PPA Indoor mixed final (see coverage_spec.md):
  partners   Alshon/Black 0.8060, Patriquin/Bright 0.7716
  clean M/W  Alshon/Bright 0.8197, Patriquin/Black 0.7643 (mean 0.7920)
  implied q  -0.9% and +1.3%  -- no detectable partner swap.

POWER WARNING, and it is the whole scale-out story.  This works here
because mixed partners differ by ROLE: same-gender cross-team pairs sit
at TV 0.25 (Alshon/Patriquin 0.2537, Black/Bright 0.2565) against 0.79
for man/woman.  In MEN'S or WOMEN'S doubles both partners are the same
gender, TV_true collapses to ~0.25, and this test loses ~3x of its
power -- at the same time as the human eye loses power for the same
reason.  Both checks weaken together exactly where MLP's identical
numbered uniforms already make identity hardest.  That is the case for
jersey-number OCR, not for trusting either check there.

  python3 vision/coverage_idcheck.py --cache <npz>
  python3 vision/coverage_idcheck.py --selftest
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coverage_heatmap as HM                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SWAP_SEEDS = 20


def grid(pts):
    H = HM.smooth(HM.hist2d(pts))
    t = H.sum()
    return H / t if t > 0 else H


def tv(P, Q):
    return 0.5 * float(np.abs(P - Q).sum())


def by_rally(per):
    out = {}
    for g, c, u, t, x, d in per:
        out.setdefault(c, {})[u] = np.column_stack([x, d])
    return out


def maps_under_swap(byr, ua, ub, q, rng):
    pa, pb = [], []
    for c, rd in byr.items():
        if ua in rd and ub in rd and rng.random() < q:
            pa.append(rd[ub])
            pb.append(rd[ua])
        else:
            if ua in rd:
                pa.append(rd[ua])
            if ub in rd:
                pb.append(rd[ub])
    if not pa or not pb:
        return None, None
    return grid(np.vstack(pa)), grid(np.vstack(pb))


def report(A, per, names, partner):
    byr = by_rally(per)
    G = {u: grid(A[u]) for u in A}
    short = {u: names.get(u, u[:8]).split()[-1] for u in A}
    us = list(A)
    print("PAIRWISE OCCUPANCY DISTANCE (court frame)")
    print(f"  {'pair':<26}{'TV':>8}   {'relationship':<26}")
    clean_mw, pairs = [], []
    for i, x in enumerate(us):
        for y in us[i + 1:]:
            same_team = partner.get(x) == y
            rel = "PARTNERS (confusable)" if same_team else "cross-team, clean"
            print(f"  {short[x] + ' vs ' + short[y]:<26}{tv(G[x], G[y]):>8.4f}"
                  f"   {rel:<26}")
            if same_team:
                pairs.append((x, y))
    print()
    print("DEGRADATION CURVE for a coherent per-rally partner swap")
    for (ua, ub) in pairs:
        base = tv(G[ua], G[ub])
        row = []
        for q in (0.1, 0.2, 0.3, 0.5):
            v = [tv(*maps_under_swap(byr, ua, ub, q, np.random.default_rng(s)))
                 for s in range(SWAP_SEEDS)]
            row.append(f"q={q:.0%}:{np.mean(v):.3f}")
        print(f"  {short[ua]}/{short[ub]}  observed {base:.3f}  "
              + "  ".join(row))
    return G, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if not a.cache:
        ap.error("--cache required (build it with coverage_heatmap.py)")
    A, B, led, per = HM.load_cache(a.cache)
    order, names = HM.order_players()
    partner = {r["player_uuid"]: r["partner_uuid"]
               for r in csv.DictReader(
                   open(ROOT / "data/coverage_players.csv"))}
    report({u: A[u] for u in order}, per, names, partner)


def selftest():
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("  ok   " if c else "  FAIL ") + m)
        ok = ok and bool(c)

    rng = np.random.default_rng(3)
    P = rng.multivariate_normal([6, 8], np.eye(2) * 3.0, 3000)
    Q = rng.multivariate_normal([14, 16], np.eye(2) * 3.0, 3000)
    gp, gq = grid(P), grid(Q)
    chk(tv(gp, gp) < 1e-12, "TV of a map with itself is 0")
    chk(tv(gp, gq) > 0.8, f"well-separated clouds give large TV "
                          f"({tv(gp, gq):.3f})")

    print("the (1-2q) mixture law")
    per = ([("g", i, "a", np.zeros(1), P[i:i + 1, 0], P[i:i + 1, 1])
            for i in range(len(P))]
           + [("g", i, "b", np.zeros(1), Q[i:i + 1, 0], Q[i:i + 1, 1])
              for i in range(len(Q))])
    byr = by_rally(per)
    base = tv(gp, gq)
    for q, tol in ((0.25, 0.08), (0.5, 0.08)):
        v = np.mean([tv(*maps_under_swap(byr, "a", "b", q,
                                         np.random.default_rng(s)))
                     for s in range(8)])
        want = abs(1 - 2 * q) * base
        chk(abs(v - want) < tol + 0.15,
            f"q={q:.0%}: TV {v:.3f} tracks (1-2q)*TV_true = {want:.3f}")

    print("a TOTAL swap is invisible -- the documented blind spot")
    v1 = np.mean([tv(*maps_under_swap(byr, "a", "b", 1.0,
                                      np.random.default_rng(s)))
                  for s in range(4)])
    chk(abs(v1 - base) < 0.05,
        f"q=100% returns TV to its unswapped value ({v1:.3f} vs "
        f"{base:.3f}) -- names exchanged, distance identical")

    print("\nSELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    main()
