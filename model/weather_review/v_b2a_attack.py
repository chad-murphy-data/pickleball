"""ADVERSARIAL VERIFICATION of B2a, part 2 — attacks on the two load-bearing
claims, using per-EVENT sufficient statistics (X'X, X'y) so that any cluster
resample / jackknife / permutation is an exact 4x4 solve and 100k resamples
are cheap.

  (a) paired arm-a -> arm-c change in outdoor d, claimed +0.039 [+0.004,+0.080]
  (b) verified-outdoor d = +0.114 vs unaudited d = -0.111 (gap 0.225, z ~ 4)

    python model/weather_review/v_b2a_attack.py
"""
from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v_b2a_verify import build_games, labels, rd, within_event_d  # noqa: E402


# ---------------------------------------------------------------- suff stats
class Suff:
    """Per-event X'X / X'y for y ~ 1 + skill + w + skill*w."""

    def __init__(self, games):
        self.S, self.t, self.n = {}, {}, {}
        by = defaultdict(list)
        for r in games:
            by[r["ev"]].append(r)
        for ev, rs in by.items():
            X = np.array([[1.0, r["skill"], r["w"], r["sw"]] for r in rs])
            y = np.array([r["y"] for r in rs])
            self.S[ev] = X.T @ X
            self.t[ev] = X.T @ y
            self.n[ev] = len(rs)

    def d(self, evs):
        S = np.zeros((4, 4)); t = np.zeros(4); n = 0
        for e in evs:
            S += self.S[e]; t += self.t[e]; n += self.n[e]
        if n < 50:
            return None
        try:
            return float(np.linalg.solve(S, t)[3])
        except np.linalg.LinAlgError:
            return None

    def N(self, evs):
        return sum(self.n[e] for e in evs)


def boot_ci(suff, evs, nboot, seed):
    rng = random.Random(seed)
    evs = list(evs)
    v = []
    for _ in range(nboot):
        s = [rng.choice(evs) for _ in evs]
        x = suff.d(s)
        if x is not None:
            v.append(x)
    v.sort()
    return v[int(.025 * len(v))], v[int(.975 * len(v))], float(np.std(v, ddof=1))


def jack_se(suff, evs):
    evs = list(evs)
    vals = [suff.d([e for e in evs if e != k]) for k in evs]
    vals = np.array([x for x in vals if x is not None])
    G = len(vals)
    return math.sqrt((G - 1) / G * float(np.sum((vals - vals.mean()) ** 2))), vals


def main():
    games = build_games()
    geo, ov, arms = labels()
    suff = Suff(games)
    P = print

    ev_all = list(suff.S)
    def pool(arm, setting):
        return [e for e in ev_all if arms[arm].get(e) == setting]

    def cls(e):
        o = ov.get(e)
        return f"{geo[e]}->{o['setting'] if o else '(unaudited)'}"

    # sanity: suff-stat solve must reproduce part 1
    P("suff-stat check: arm a outdoor d = %+.4f, arm c outdoor d = %+.4f"
      % (suff.d(pool("a", "outdoor")), suff.d(pool("c", "outdoor"))))

    # =============== CLAIM (a): the paired label delta ======================
    P("\n" + "=" * 72)
    P("CLAIM (a): paired arm-a minus arm-c change in OUTDOOR d")
    P("=" * 72)
    a_out, c_out = set(pool("a", "outdoor")), set(pool("c", "outdoor"))
    a_out_l, c_out_l = list(a_out), list(c_out)
    pt = suff.d(a_out_l) - suff.d(c_out_l)
    P(f"point: {suff.d(a_out_l):+.4f} - ({suff.d(c_out_l):+.4f}) = {pt:+.4f}")

    def paired(evs):
        da, dc = suff.d([e for e in evs if e in a_out]), suff.d([e for e in evs if e in c_out])
        return None if da is None or dc is None else da - dc

    # seed sensitivity at the tester's settings (800 resamples)
    P("\n seed sensitivity of the 800-resample percentile CI (tester's design):")
    lows = []
    for seed in (4242, 1, 2, 3, 7, 11, 99, 123, 2024, 31337):
        rng = random.Random(seed)
        v = []
        for _ in range(800):
            s = [rng.choice(ev_all) for _ in ev_all]
            x = paired(s)
            if x is not None:
                v.append(x)
        v.sort()
        lo, hi = v[int(.025 * len(v))], v[int(.975 * len(v))]
        lows.append(lo)
        P(f"   seed {seed:6d}: [{lo:+.4f}, {hi:+.4f}]  {'EXCLUDES 0' if lo > 0 else 'includes 0'}")
    P(f"   -> lower bound ranges {min(lows):+.4f} .. {max(lows):+.4f}; "
      f"{sum(1 for l in lows if l > 0)}/10 seeds exclude zero")

    # high-precision CI
    rng = random.Random(20260731)
    v = []
    for _ in range(20000):
        s = [rng.choice(ev_all) for _ in ev_all]
        x = paired(s)
        if x is not None:
            v.append(x)
    v.sort()
    lo, hi = v[int(.025 * len(v))], v[int(.975 * len(v))]
    frac0 = sum(1 for x in v if x <= 0) / len(v)
    P(f"\n 20,000-resample paired CI: {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
      f"bootstrap P(delta<=0) = {frac0:.3f}  (two-sided p ~ {2*frac0:.3f})")
    jse, _ = jack_se(suff, ev_all)   # jackknife of pooled d is not the paired stat
    # jackknife of the PAIRED statistic
    jv = np.array([x for x in (paired([e for e in ev_all if e != k]) for k in ev_all)
                   if x is not None])
    G = len(jv)
    pjse = math.sqrt((G - 1) / G * float(np.sum((jv - jv.mean()) ** 2)))
    P(f" jackknife SE of the paired delta over {G} events: {pjse:.4f}  "
      f"z = {pt/pjse:.2f}  normal p = {2*(1-0.5*(1+math.erf(abs(pt/pjse)/math.sqrt(2)))):.3f}")

    # decompose: relabel-only vs drop-only
    P("\n decomposition of the a->c outdoor move (point estimates):")
    core = [e for e in a_out_l if e in c_out]
    removed = [e for e in a_out_l if e not in c_out]
    added = [e for e in c_out_l if e not in a_out]
    P(f"   arm a outdoor      d={suff.d(a_out_l):+.4f} n={suff.N(a_out_l)}")
    P(f"   drop removed only  d={suff.d(core):+.4f} n={suff.N(core)}   "
      f"(change {suff.d(core)-suff.d(a_out_l):+.4f})")
    P(f"   + added flips = c  d={suff.d(c_out_l):+.4f} n={suff.N(c_out_l)}   "
      f"(change {suff.d(c_out_l)-suff.d(core):+.4f})")
    P("   removed events by class: " + ", ".join(
        f"{k}={sum(1 for e in removed if cls(e)==k)}"
        for k in sorted({cls(e) for e in removed})))

    # arm e: relabel ONLY (mixed/unknown keep heuristic label, nothing dropped)
    arm_e = {}
    for e in ev_all:
        o = ov.get(e)
        if o is None or o["setting"] in ("mixed", "unknown") or o["confidence"] == "low":
            arm_e[e] = geo[e]
        else:
            arm_e[e] = o["setting"]
    e_out = [e for e in ev_all if arm_e[e] == "outdoor"]
    e_in = [e for e in ev_all if arm_e[e] == "indoor"]
    P(f"\n arm e (relabel only, NOTHING dropped): outdoor d={suff.d(e_out):+.4f} "
      f"n={suff.N(e_out)}  indoor d={suff.d(e_in):+.4f} n={suff.N(e_in)}  "
      f"diff {suff.d(e_out)-suff.d(e_in):+.4f}")
    lo_e, hi_e, _ = boot_ci(suff, e_out, 4000, 5)
    P(f"   arm e outdoor CI [{lo_e:+.4f},{hi_e:+.4f}]")
    pe = suff.d(a_out_l) - suff.d(e_out)
    ve = []
    rng = random.Random(77)
    for _ in range(8000):
        s = [rng.choice(ev_all) for _ in ev_all]
        da = suff.d([e for e in s if e in a_out])
        de = suff.d([e for e in s if arm_e[e] == "outdoor"])
        if da is not None and de is not None:
            ve.append(da - de)
    ve.sort()
    P(f"   paired a - e change: {pe:+.4f} "
      f"[{ve[int(.025*len(ve))]:+.4f},{ve[int(.975*len(ve))]:+.4f}]  "
      f"(this is the pure RELABELLING effect, no sample restriction)")

    # =============== CLAIM (b): the audit-status split ======================
    P("\n" + "=" * 72)
    P("CLAIM (b): within corrected-outdoor pool, verified vs unaudited")
    P("=" * 72)
    sub = {"unaudited": [e for e in c_out_l if cls(e) == "outdoor->(unaudited)"],
           "verified-outdoor": [e for e in c_out_l if cls(e) == "outdoor->outdoor"],
           "newly-outdoor": [e for e in c_out_l if cls(e) == "indoor->outdoor"]}
    for k, evs in sub.items():
        d = suff.d(evs)
        lo_, hi_, se_ = boot_ci(suff, evs, 4000, 9)
        js, jvals = jack_se(suff, evs)
        P(f"  {k:17s} ev={len(evs):3d} n={suff.N(evs):6d} d={d:+.4f} "
          f"boot [{lo_:+.4f},{hi_:+.4f}] bse={se_:.4f} jackSE={js:.4f} "
          f"LOO range [{jvals.min():+.4f},{jvals.max():+.4f}]")
    gap = suff.d(sub["verified-outdoor"]) - suff.d(sub["unaudited"])
    P(f"\n  observed gap verified - unaudited = {gap:+.4f}")

    # permutation: is AUDIT STATUS special, or is this generic between-event
    # heterogeneity?  Randomly reassign the 48 audited/unaudited labels among
    # the 48 heuristic-outdoor events, holding the (15, 33) split sizes fixed.
    hpool = sub["unaudited"] + sub["verified-outdoor"]
    rng = random.Random(4242)
    k_ver = len(sub["verified-outdoor"])
    perm = []
    for _ in range(20000):
        s = rng.sample(hpool, k_ver)
        rest = [e for e in hpool if e not in set(s)]
        a_, b_ = suff.d(s), suff.d(rest)
        if a_ is not None and b_ is not None:
            perm.append(a_ - b_)
    perm = np.array(perm)
    p_two = float(np.mean(np.abs(perm) >= abs(gap)))
    P(f"  permutation over event labels ({k_ver} vs {len(hpool)-k_ver}, "
      f"{len(perm)} draws): sd={perm.std():.4f}, "
      f"|gap| >= observed in {p_two:.4f} of draws")
    P(f"    -> permutation null sd of a random 15/33 split of THIS pool is "
      f"{perm.std():.4f}; observed {gap:+.4f} is {gap/perm.std():.2f} sd")

    # size-matched permutation (match games, not events)
    n_ver = suff.N(sub["verified-outdoor"])
    perm2 = []
    rng = random.Random(555)
    tries = 0
    while len(perm2) < 5000 and tries < 400000:
        tries += 1
        s, tot = [], 0
        pool_ = hpool[:]
        rng.shuffle(pool_)
        for e in pool_:
            if tot + suff.n[e] <= n_ver * 1.15:
                s.append(e); tot += suff.n[e]
            if tot >= n_ver * 0.85:
                break
        if not (n_ver * 0.85 <= tot <= n_ver * 1.15):
            continue
        rest = [e for e in hpool if e not in set(s)]
        a_, b_ = suff.d(s), suff.d(rest)
        if a_ is not None and b_ is not None:
            perm2.append(a_ - b_)
    perm2 = np.array(perm2)
    P(f"  game-size-matched permutation ({len(perm2)} draws): sd={perm2.std():.4f}, "
      f"p={float(np.mean(np.abs(perm2) >= abs(gap))):.4f}")

    # max-gap-of-three-groups permutation (the tester picked the 2 extremes of 3)
    sizes = [len(sub[k]) for k in ("verified-outdoor", "unaudited", "newly-outdoor")]
    allc = list(c_out)
    rng = random.Random(31337)
    maxg = []
    for _ in range(10000):
        p_ = allc[:]
        rng.shuffle(p_)
        g1 = p_[:sizes[0]]; g2 = p_[sizes[0]:sizes[0]+sizes[1]]; g3 = p_[sizes[0]+sizes[1]:]
        ds = [suff.d(g) for g in (g1, g2, g3)]
        if any(x is None for x in ds):
            continue
        maxg.append(max(ds) - min(ds))
    maxg = np.array(maxg)
    obs_max = max(suff.d(sub[k]) for k in sub) - min(suff.d(sub[k]) for k in sub)
    P(f"  max-minus-min over the THREE sub-pools: observed {obs_max:+.4f}; "
      f"permutation p = {float(np.mean(maxg >= obs_max)):.4f} "
      f"(null median {np.median(maxg):.4f})")

    # confounds of the split
    P("\n  confound audit of the two sub-pools:")
    by = defaultdict(list)
    for r in games:
        by[r["ev"]].append(r)
    for k in ("verified-outdoor", "unaudited"):
        rows = [r for e in sub[k] for r in by[e]]
        yrs = defaultdict(int)
        for r in rows:
            yrs[r["date"][:4]] += 1
        fmts = defaultdict(int)
        for r in rows:
            fmts[r["fmt"]] += 1
        P(f"   {k:17s} MLP={100*sum(1 for r in rows if r['tour']=='MLP')/len(rows):4.1f}% "
          f"wind mean={np.mean([r['wind'] for r in rows]):.2f} "
          f"sd={np.std([r['wind'] for r in rows]):.2f} "
          f"P(>=14)={100*np.mean([r['wind']>=14 for r in rows]):.1f}% "
          f"|skill| mean={np.mean([abs(r['skill']) for r in rows]):.3f}")
        P(f"      years {dict(sorted(yrs.items()))}")
        P(f"      formats {dict(sorted(fmts.items(), key=lambda kv:-kv[1]))}")

    # gap after dropping MLP, and PPA-only
    for lab, filt in (("PPA only", lambda r: r["tour"] == "PPA"),):
        s2 = Suff([r for r in games if filt(r)])
        try:
            g2 = s2.d(sub["verified-outdoor"]) - s2.d(sub["unaudited"])
            P(f"   gap {lab}: {g2:+.4f} "
              f"(verified {s2.d(sub['verified-outdoor']):+.4f}, "
              f"unaudited {s2.d(sub['unaudited']):+.4f})")
        except KeyError:
            P(f"   gap {lab}: n/a")

    # =============== is d a between-event statistic at all? =================
    P("\n" + "=" * 72)
    P("IS d IDENTIFIED WITHIN EVENTS OR BETWEEN THEM?")
    P("=" * 72)
    for lab, evs in (("arm a outdoor", pool("a", "outdoor")),
                     ("arm c outdoor", pool("c", "outdoor")),
                     ("arm a indoor", pool("a", "indoor")),
                     ("arm c indoor", pool("c", "indoor"))):
        rows = [r for e in evs for r in by[e]]
        d_w, se_w, G, n = within_event_d(rows)
        P(f"  {lab:14s} WITHIN-event d = {d_w:+.4f} +/- {se_w:.4f} "
          f"[{d_w-1.96*se_w:+.4f},{d_w+1.96*se_w:+.4f}]  ({G} events, {n} games)"
          f"   vs pooled {suff.d(evs):+.4f}")

    # between-event: per-event skill loading b_e vs event mean wind
    P("\n  between-event route: per-event b_e (y ~ 1+skill) vs event mean wind")
    for lab, evs in (("arm a outdoor", pool("a", "outdoor")),
                     ("arm c outdoor", pool("c", "outdoor"))):
        pts = []
        for e in evs:
            rows = by[e]
            if len(rows) < 100:
                continue
            X = np.array([[1.0, r["skill"]] for r in rows])
            y = np.array([r["y"] for r in rows])
            try:
                bb = np.linalg.solve(X.T @ X, X.T @ y)
            except np.linalg.LinAlgError:
                continue
            pts.append((np.mean([r["wind"] for r in rows]) / 10.0, float(bb[1]), len(rows)))
        w_ = np.array([p[0] for p in pts]); b_ = np.array([p[1] for p in pts])
        n_ = np.array([p[2] for p in pts], float)
        W = np.column_stack([np.ones(len(w_)), w_])
        sl = np.linalg.solve(W.T @ (W * n_[:, None]), W.T @ (b_ * n_))
        P(f"   {lab}: {len(pts)} events >=100 games; sd(b_e)={b_.std():.3f} "
          f"(pooled b~1.05); WLS slope of b_e on wind/10 = {sl[1]:+.4f} "
          f"vs pooled d = {suff.d(evs):+.4f}")


if __name__ == "__main__":
    main()
