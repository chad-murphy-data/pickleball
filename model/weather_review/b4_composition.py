"""B4 part 3, addendum — is the H4 interaction's EVENT COMPOSITION problem
fatal to the estimator, or only variance-inflating?

    python model/weather_review/b4_composition.py

Phase-2 test B2a found that inside the corrected outdoor pool the
web-verified-outdoor events give d ≈ +0.114 [+0.016, +0.168] and the
unaudited events d ≈ −0.111 [−0.179, −0.025] — a 0.225 gap at z ≈ 4,
bigger than the effect under test. Three questions, three tests:

  1. HOW BIG is the between-event dispersion of the interaction slope?
     Fit d_e per event, DerSimonian–Laird τ (excess of the observed spread
     over sampling noise). τ is the honest "how much do events disagree".
  2. Is a 0.225 arm gap REMARKABLE? Permute the event→arm assignment
     (arm sizes fixed at the real verified/unaudited counts), recompute
     the gap 5,000 times, and read off where 0.225 falls. If random splits
     of the same events routinely produce ±0.2 gaps, the B2a split is not
     evidence of an audit-related bias — it is evidence that the
     coefficient is event-noisy.
  3. Does the gap SURVIVE event fixed effects? Event FE identifies d only
     from within-event wind variation, so cross-event composition cannot
     move it. If the arm gap persists under FE, the arms genuinely differ
     in their within-event slopes (heterogeneity, not composition).

Stdlib only, seeded. Writes model/weather_review/b4_composition.md
"""
from __future__ import annotations

import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(HERE))
import b2b_lib as L  # noqa: E402
from b4_speccurve import build_grams, fit_spec, game_rows, read_csv, solve  # noqa: E402

OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def pooled(grams, ns, events, xcols=(0, 1, 2, 3), ycol=5, demean=False):
    sub = {e: grams[e] for e in events if e in grams}
    sns = {e: ns[e] for e in sub}
    res = fit_spec(sub, sns, list(xcols), ycol, demean)
    if not res:
        return None
    beta, V, N, K = res
    i = list(xcols).index(3)
    return beta[i], math.sqrt(max(V[i][i], 0)), N, K


def main():
    say("# B4 addendum — event composition vs the H4 interaction\n")
    rows = game_rows()
    arms = L.label_arms()
    grams, ns = build_grams(rows, "hour_pub", "sustained", "all")
    corrected = {e for e, G in grams.items()
                 if arms["corrected_all"].get(e) == "outdoor"}
    ov = {r["event_id"]: r for r in read_csv(ROOT / "data/venue_overrides.csv")}
    verified = {e for e in corrected if e in ov}
    unaudited = corrected - verified
    geo = {r["event_id"]: r for r in read_csv(ROOT / "data/event_geo.csv")}

    gv = sum(ns[e] for e in verified)
    gu = sum(ns[e] for e in unaudited)
    say(f"Corrected outdoor pool: {len(corrected)} events / "
        f"{gv+gu} games — {len(verified)} web-verified ({gv} games), "
        f"{len(unaudited)} unaudited-heuristic ({gu} games).\n")

    dv = pooled(grams, ns, verified)
    du = pooled(grams, ns, unaudited)
    dall = pooled(grams, ns, corrected)
    say("| arm | events | games | d (skill×wind, share outcome) ± CR1 se |")
    say("|---|---|---|---|")
    for nm, r_, k in (("web-verified outdoor", dv, verified),
                      ("unaudited heuristic outdoor", du, unaudited),
                      ("both (corrected pool)", dall, corrected)):
        say(f"| {nm} | {r_[3]} | {r_[2]} | {r_[0]:+.4f} ± {r_[1]:.4f} |")
    gap = dv[0] - du[0]
    say(f"\nObserved arm gap (verified − unaudited) = **{gap:+.4f}** "
        f"(B2a reported ≈ +0.225 on its own variant).\n")

    # ---------------- 1. between-event dispersion -----------------------
    say("## 1. How much do individual EVENTS disagree about d?\n")
    say("*(within-event fits use CLASSICAL OLS standard errors — the "
        "cluster-robust sandwich is degenerate with one cluster.)*\n")

    def event_fit(e):
        """d_e and its classical SE from the event's own Gram."""
        G, n = grams[e], ns[e]
        if n < 60:
            return None
        wsd2 = G[2][2] / n - (G[0][2] / n) ** 2
        if wsd2 < 0.04:          # < 0.2 mph of within-event wind variation
            return None
        cols = [0, 1, 2, 3]
        A = [[G[i][j] for j in cols] for i in cols]
        c = [G[i][5] for i in cols]
        beta = solve(A, c)
        if beta is None:
            return None
        rss = G[5][5] - sum(b * ci for b, ci in zip(beta, c))
        s2 = rss / (n - 4)
        from b4_speccurve import inv as _inv
        Ai = _inv(A)
        if Ai is None or Ai[3][3] <= 0:
            return None
        return beta[3], math.sqrt(s2 * Ai[3][3]), n

    per = []
    for e in corrected:
        r_ = event_fit(e)
        if r_:
            per.append((e, r_[0], r_[1], r_[2]))
    ws = [1 / s ** 2 for _, _, s, _ in per]
    mu = sum(w * d for w, (_, d, _, _) in zip(ws, per)) / sum(ws)
    Q = sum(w * (d - mu) ** 2 for w, (_, d, _, _) in zip(ws, per))
    k = len(per)
    C = sum(ws) - sum(w * w for w in ws) / sum(ws)
    tau2 = max(0.0, (Q - (k - 1)) / C)
    say(f"{k} events with ≥60 games and an estimable within-event d. "
        f"Fixed-effect mean {mu:+.4f}. Cochran Q = {Q:.1f} on {k-1} df "
        f"(expected {k-1} under no true heterogeneity) → "
        f"**τ = {math.sqrt(tau2):.4f}** (between-event sd of the true "
        f"interaction slope).")
    sds = sorted(s for _, _, s, _ in per)
    say(f"\nMedian within-event standard error {sds[len(sds)//2]:.3f} — a "
        f"single event pins d only to about ±{1.96*sds[len(sds)//2]:.2f}. "
        f"Between-event heterogeneity τ = {math.sqrt(tau2):.3f} is the "
        f"number that matters: it is the sd of the per-event slope, and it "
        f"is {math.sqrt(tau2)/0.072:.0f}× the effect size under debate "
        f"(the binned −2.0 pp windy drift maps to d ≈ −0.072).")

    # ---------------- 2. permutation of the arm assignment --------------
    say("\n## 2. Is a gap of that size remarkable? (event-label permutation)\n")
    pool = sorted(corrected)
    nv = len(verified)
    rng = random.Random(20260731)
    null = []
    for _ in range(5000):
        shuf = pool[:]
        rng.shuffle(shuf)
        a, b = shuf[:nv], shuf[nv:]
        ra, rb = pooled(grams, ns, a), pooled(grams, ns, b)
        if ra and rb:
            null.append(ra[0] - rb[0])
    null.sort()
    n = len(null)
    q = lambda p: null[min(n - 1, int(p * n))]
    bigger = sum(1 for v in null if abs(v) >= abs(gap)) / n
    say(f"5,000 random re-labellings of which events are 'verified' "
        f"({nv} of {len(pool)}), same estimator:")
    say(f"\n| null gap sd | 2.5% | 50% | 97.5% | P(|random gap| ≥ observed "
        f"{abs(gap):.3f}) | P(|random gap| ≥ 0.225) |")
    say("|---|---|---|---|---|---|")
    m = sum(null) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in null) / (n - 1))
    p225 = sum(1 for v in null if abs(v) >= 0.225) / n
    say(f"| {sd:.4f} | {q(0.025):+.4f} | {q(0.5):+.4f} | {q(0.975):+.4f} "
        f"| {bigger:.3f} | {p225:.3f} |")

    # size-stratified version: the audit was not random — it covered the
    # events that supply labels (bigger, tour-flagship), so match on size.
    order = sorted(pool, key=lambda e: ns[e])
    strata = [order[i::3] for i in range(3)]
    nv_str = [sum(1 for e in s if e in verified) for s in strata]
    null2 = []
    for _ in range(5000):
        a = []
        for s, k_ in zip(strata, nv_str):
            ss = s[:]
            rng.shuffle(ss)
            a.extend(ss[:k_])
        aset = set(a)
        b = [e for e in pool if e not in aset]
        ra, rb = pooled(grams, ns, a), pooled(grams, ns, b)
        if ra and rb:
            null2.append(ra[0] - rb[0])
    null2.sort()
    n2 = len(null2)
    m2 = sum(null2) / n2
    sd2 = math.sqrt(sum((v - m2) ** 2 for v in null2) / (n2 - 1))
    p2 = sum(1 for v in null2 if abs(v) >= abs(gap)) / n2
    say(f"\nSize-stratified permutation (events split into 3 size terciles, "
        f"verified-count held fixed inside each — the audit was not random, "
        f"it targeted the bigger label-supplying events): null sd {sd2:.4f}, "
        f"95% band [{null2[int(0.025*n2)]:+.4f}, {null2[int(0.975*n2)]:+.4f}], "
        f"P(|random gap| ≥ {abs(gap):.3f}) = {p2:.3f}.")

    say("\n### Same split, by tour and by outcome (is it an audit effect or "
        "an event-type effect?)\n")
    tour_of = {}
    for g in read_csv(ROOT / "data/games.csv"):
        tour_of.setdefault(g["event_id"], g["tour"])
    say("| subset | verified d | unaudited d | gap |")
    say("|---|---|---|---|")
    for nm, keep in (("all events", lambda e: True),
                     ("PPA events only", lambda e: tour_of.get(e) == "PPA"),
                     ("events ≥300 games",
                      lambda e: ns[e] >= 300),
                     ("events <300 games", lambda e: ns[e] < 300)):
        v = [e for e in verified if keep(e)]
        u = [e for e in unaudited if keep(e)]
        if len(v) < 4 or len(u) < 4:
            continue
        rv, ru = pooled(grams, ns, v), pooled(grams, ns, u)
        say(f"| {nm} ({len(v)}v/{len(u)}u events) | {rv[0]:+.4f} ± {rv[1]:.4f} "
            f"| {ru[0]:+.4f} ± {ru[1]:.4f} | {rv[0]-ru[0]:+.4f} |")

    # ---------------- 3. does the gap survive event FE? -----------------
    say("\n## 3. Does the arm gap survive EVENT fixed effects?\n")
    say("| arm | d, no FE | d, event FE (within-event wind variation only) |")
    say("|---|---|---|")
    for nm, k_ in (("web-verified", verified), ("unaudited", unaudited),
                   ("pool", corrected)):
        a = pooled(grams, ns, k_)
        b = pooled(grams, ns, k_, xcols=(1, 2, 3), demean=True)
        say(f"| {nm} | {a[0]:+.4f} ± {a[1]:.4f} | {b[0]:+.4f} ± {b[1]:.4f} |")
    a1 = pooled(grams, ns, verified, xcols=(1, 2, 3), demean=True)
    a2 = pooled(grams, ns, unaudited, xcols=(1, 2, 3), demean=True)
    say(f"\nGap under event FE = **{a1[0]-a2[0]:+.4f}** vs {gap:+.4f} "
        f"without. Event FE removes all BETWEEN-event variation, so a gap "
        f"that survives is within-event heterogeneity, not composition.")

    # ---------------- 4. the 3 most influential events on the gap -------
    say("\n## 4. Which events move the arm gap most (leave-one-out)?\n")
    infl = []
    for e in corrected:
        vv = verified - {e}
        uu = unaudited - {e}
        r1, r2 = pooled(grams, ns, vv), pooled(grams, ns, uu)
        if r1 and r2:
            infl.append((abs((r1[0] - r2[0]) - gap), e, (r1[0] - r2[0]) - gap))
    infl.sort(reverse=True)
    say("| event | games | arm | Δ gap when dropped |")
    say("|---|---|---|---|")
    for _, e, d in infl[:5]:
        nm = geo.get(e, {}).get("event_name", e)
        arm = "verified" if e in verified else "unaudited"
        say(f"| {nm} ({geo.get(e,{}).get('first_date','')}) | {ns[e]} "
            f"| {arm} | {d:+.4f} |")

    (HERE / "b4_composition.md").write_text("\n".join(OUT) + "\n")
    print("\nwrote model/weather_review/b4_composition.md")


if __name__ == "__main__":
    main()
