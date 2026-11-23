"""A no-ball PRIOR on "did this flight bounce?" (owner asks, 2026-09-04).

THE FRAMING IS THE OWNER'S and it is the important part: *"every pair of
contacts has to make a call -- bounce or not."*  Today bounce is the
RESIDUAL of turn detection, so the denominator is however many wobbles the
tracker produced.  Recast as one binary call per flight, the panel is
finite and the score is well defined: N contacts -> N-1 flights -> N-1
answers.  Base rate P(bounce) = 0.422 (57 of 135) -- a real classification
problem, not a rare event.

Two owner hypotheses tested on that panel, plus the double-bounce rule.

1  "SLOW SHOTS ARE MORE LIKELY TO BOUNCE THAN FAST SHOTS"  -- true in the
   raw data and USELESS once duration is known.  Bounced flights run
   20.7 mph vs 23.5; AUC 0.597.  But stratify on flight duration and speed
   collapses to chance or below in every stratum (0.433 / 0.486 / 0.360),
   and its coefficient flips sign alongside dt.  Slow shots bounce more
   because slow shots take longer.  DURATION is the real variable:
   AUC 0.809, P(bounce) 0.09 -> 0.44 -> 0.72 across duration terciles.

2  "IF THE RECEIVING TEAM IS AT THE BASELINE THERE IS ALMOST CERTAINLY A
   BOUNCE"  -- CONFIRMED, and it is the strongest single feature found.
   By the receiver's distance from the net at their contact:

       0-7 ft  (kitchen)   n=11   P(bounce) 0.09
       7-11 ft             n=50             0.24
       11-15 ft            n=29             0.31
       15-19 ft            n=13             0.46
       19+ ft  (baseline)  n=32             0.91

   And unlike speed it is NOT duration in disguise -- within duration
   strata its AUC is 0.988 / 0.862 / 0.742.  Depth alone AUC 0.806;
   duration alone 0.809; the two together 0.910.

3  The DOUBLE-BOUNCE RULE is exact, not a prior: flight 0 (serve->return)
   reads 0.89 and flight 1 (return->third) reads 1.00, against 0.34 for
   flight 2+.  Flight 0 should be 1.00 -- the one miss in nine is a
   labelling/matching artifact, and is a useful data-quality probe.

WHY THIS MATTERS MORE THAN A TRACKER TWEAK: receiver depth and contact
timing are both measured WITHOUT THE BALL, from pose and the contact
stream, which are solved channels (homography 0.06 ft).  So this is a
prior available on every flight, including the ones where the ball is
never found.

CAVEAT: these are in-sample reads on 135 flights, and both features are
partly CONSEQUENCES of a bounce (a bounce adds travel time; a deep
receiver has more time).  That is fine for a prior -- the claim is
predictive, not causal -- but the numbers are not a held-out score.

    python3 vision/ballsearch/bounce_prior.py
"""
import sys
import numpy as np
sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, "/home/user/pickleball/vision/ballsearch")
import pathfirst as pf
from rally_stats import players
from geom_speed import contacts, clicks, hitter_track, foot_xy, FT_S_TO_MPH

NET_Y = 22.0
RAL = [2, 3, 4, 5, 6, 7, 9, 10, 17]
rows = []
for r in RAL:
    try:
        ctx = pf.context(r)
        P, z = ctx["P"], np.load(ctx["c"]["npz"])
        pls = players(ctx)
        cl = clicks(r)
        cs = contacts(r)
        for s in cs:
            s["tid"] = hitter_track(ctx, pls, s["t"], cl)
            s["xy"] = foot_xy(z, P, s["tid"], s["t"]) if s["tid"] is not None else None
        bt = [float(s["ts"]) for s in ctx["c"]["h_segs"]
              if s and s.get("ok") and s["kind"] == "bounce"]
    except Exception as e:
        print(f"r{r}: skipped ({type(e).__name__})")
        continue
    k = 0
    for a, b in zip(cs, cs[1:]):
        if a["type"] == "whiff" or b["type"] == "whiff":
            continue
        if a["xy"] is None or b["xy"] is None:
            continue
        dt = b["t"] - a["t"]
        if not (0.10 <= dt <= 3.0):
            continue
        rows.append(dict(rally=r, k=k, dt=dt,
                         v=float(np.linalg.norm(b["xy"] - a["xy"])) / dt,
                         depth=abs(float(b["xy"][1]) - NET_Y),   # receiver
                         hdepth=abs(float(a["xy"][1]) - NET_Y),  # hitter
                         bounce=any(a["t"] < x < b["t"] for x in bt)))
        k += 1

y = np.array([w["bounce"] for w in rows])
d = np.array([w["depth"] for w in rows])
k = np.array([w["k"] for w in rows])
dt = np.array([w["dt"] for w in rows])

def auc(s, l):
    x, l = np.asarray(s, float), np.asarray(l, bool)
    if not l.any() or l.all(): return float("nan")
    o = np.argsort(x); rk = np.empty(len(x)); rk[o] = np.arange(1, len(x)+1)
    return float((rk[l].sum() - l.sum()*(l.sum()+1)/2) / (l.sum()*(~l).sum()))

print(f"\nPANEL {len(rows)} flights, base rate {y.mean():.3f}")
print(f"  AUC receiver depth from net -> bounce : {auc(d, y):.3f}")
print(f"  AUC flight duration         -> bounce : {auc(dt, y):.3f}")

print("\nP(bounce) BY RECEIVER DEPTH FROM THE NET  (kitchen line = 7 ft, baseline = 22)")
for lo, hi in [(0,7),(7,11),(11,15),(15,19),(19,99)]:
    m = (d >= lo) & (d < hi)
    if m.sum() < 3: continue
    lab = f"{lo}-{hi if hi<99 else '+'} ft"
    print(f"  {lab:>10} n={m.sum():3d}  P(bounce)={y[m].mean():.2f}")

print("\nP(bounce) BY FLIGHT INDEX  (0 = serve->return, 1 = return->third)")
for i in range(5):
    m = k == i
    if m.sum() < 3: continue
    lab = {0: " serve->return", 1: " return->third"}.get(i, "")
    print(f"  flight {i}  n={m.sum():3d}  P(bounce)={y[m].mean():.2f}{lab}")
m = k >= 2
print(f"  flight 2+ n={m.sum():3d}  P(bounce)={y[m].mean():.2f}")

print("\nDEPTH WITHIN DURATION STRATA (does depth add beyond dt?)")
qs = np.quantile(dt, [0, .33, .66, 1.0])
for lo, hi in zip(qs, qs[1:]):
    m = (dt >= lo) & (dt <= hi)
    if m.sum() < 8 or y[m].all() or not y[m].any(): continue
    print(f"  dt {lo:.2f}-{hi:.2f}s n={m.sum():3d} P={y[m].mean():.2f} "
          f"AUC(depth)={auc(d[m], y[m]):.3f}")

from sklearn.linear_model import LogisticRegression
for nm, X in (("dt", dt.reshape(-1,1)),
              ("depth", d.reshape(-1,1)),
              ("dt+depth", np.column_stack([dt, d])),
              ("dt+depth+idx01", np.column_stack([dt, d, (k <= 1).astype(float)]))):
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    lr = LogisticRegression(max_iter=2000).fit(Xs, y)
    print(f"  {nm:15s} in-sample AUC {auc(lr.predict_proba(Xs)[:,1], y):.3f} "
          f"coefs {np.round(lr.coef_[0],3)}")
