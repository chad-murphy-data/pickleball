"""ADVERSARIAL follow-ups to B1: leverage, within-event identification,
and whether the reported 'event fixed effects' specification does anything.
"""
from __future__ import annotations
import csv, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sitelib.race import sigmoid, team_eta, game_win_prob  # noqa
from heat_test import load, build_games, ols, demean  # noqa
from heat_robust import add_h, symmetrize  # noqa


def rd(p):
    with open(ROOT / p) as f:
        return list(csv.DictReader(f))


setting_audit, setting_heur, hourly, start, v2 = load()
rows, meta, per_match = build_games(setting_audit, hourly, start, v2)
by = defaultdict(list)
for r in rows:
    by[r["setting"]].append(r)
geo = {r["event_id"]: r for r in rd("data/event_geo.csv")}

print("### CHECK 1 — is the tester's 'event FE' demeaning a no-op after "
      "symmetrization?")
rs = add_h(by["outdoor"])
sym = []
for r in symmetrize(rs):
    q = dict(r)
    q["hr"] = (r["hour"] - 14.0) / 6.0
    q["shr"] = q["skill"] * q["hr"]
    q["shr2"] = q["skill"] * q["hr"] ** 2
    q["cell"] = r["ev"]
    sym.append(q)
dm = demean(sym, ["y", "skill", "sh", "shr", "shr2"])
maxshift = max(abs(a[k] - b[k]) for a, b in zip(sym, dm)
               for k in ("y", "skill", "sh", "shr", "shr2"))
print(f"  max |value after demean - before| over all rows/cols = {maxshift:.3e}"
      "   -> demeaning changes NOTHING; the 'event FE' is vacuous.")
b_fe = ols(dm, "y", ["skill", "sh", "shr", "shr2"])
b_nofe = ols(sym, "y", ["skill", "sh", "shr", "shr2"])
b_plain = ols(sym, "y", ["skill", "sh"])
print(f"  d with 'FE'+hour ctrls   = {b_fe[2]:+.4f}")
print(f"  d with hour ctrls, no FE = {b_nofe[2]:+.4f}   (identical)")
print(f"  d, odd spec, NO hour ctrls = {b_plain[2]:+.4f}  <- the primary")
print("  => the move from +0.0165 to +0.029 is caused entirely by the "
      "skill x hour / skill x hour^2 terms, not by event fixed effects.\n")


# ---- genuine within-event identification of the odd interaction --------
def odd_fit(rs, tempkey="T", within_event=False, nboot=3000, seed=17):
    cl = defaultdict(list)
    for r in rs:
        cl[r["ev"]].append(r)
    keys = list(cl)
    suf = []
    for k in keys:
        s = np.array([r["skill"] for r in cl[k]])
        h = np.array([(r[tempkey] - 75.0) / 10.0 for r in cl[k]])
        if within_event:
            h = h - h.mean()
        y = np.array([r["y"] for r in cl[k]])
        X = np.column_stack([s, s * h])
        suf.append(np.concatenate([(X.T @ X).ravel(), X.T @ y]))
    suf = np.array(suf)
    f = lambda t: np.linalg.solve(t[:4].reshape(2, 2), t[4:])
    pt = f(suf.sum(0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(nboot, len(keys)))
    out = np.array([f(t) for t in suf[idx].sum(1)])
    return pt, np.percentile(out[:, 1], [2.5, 97.5]), len(keys), suf, keys


print("### CHECK 2 — genuine WITHIN-EVENT identification of d "
      "(h centred inside each event)")
for name in ("outdoor", "indoor"):
    for we in (False, True):
        pt, ci, nk, _, _ = odd_fit(by[name], within_event=we)
        print(f"  {name:8s} {'within-event' if we else 'pooled      '} "
              f"n={len(by[name]):6d} ev={nk:3d} b={pt[0]:.3f} "
              f"d={pt[1]:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]")
print()

# ---- leverage: jackknife the primary by event --------------------------
print("### CHECK 3 — event jackknife of the primary d (audited outdoor)")
pt, ci, nk, suf, keys = odd_fit(by["outdoor"])
tot = suf.sum(0)
f = lambda t: np.linalg.solve(t[:4].reshape(2, 2), t[4:])
jk = []
for i, k in enumerate(keys):
    jk.append((f(tot - suf[i])[1], k, sum(1 for r in by["outdoor"]
                                          if r["ev"] == k)))
jk.sort()
print(f"  full-sample d = {pt[1]:+.4f}; jackknife range "
      f"[{jk[0][0]:+.4f}, {jk[-1][0]:+.4f}]")
for d_, k, n_ in jk[:3] + jk[-3:]:
    g = geo.get(k, {})
    print(f"    drop n={n_:5d}  d -> {d_:+.4f}   {g.get('venue','')[:55]}")
print()

# ---- leverage on the CONTINUOUS edge statistic -------------------------
print("### CHECK 4 — continuous favourite-edge slope: jackknife + "
      "within-event version")


def edge_slope(rs, within_event=False, nboot=3000, seed=23, jack=False):
    cl = defaultdict(list)
    for r in rs:
        cl[r["ev"]].append(r)
    keys = list(cl)
    suf = []
    for k in keys:
        x = np.array([(r["T"] - 75.0) / 10.0 for r in cl[k]])
        if within_event:
            x = x - x.mean()
        eta = np.array([r["eta"] for r in cl[k]])
        p = np.array([max(game_win_prob(e), 1 - game_win_prob(e))
                      for e in eta])
        won = np.array([1.0 if (r["y"] > 0) == (r["eta"] >= 0) else 0.0
                        for r in cl[k]])
        y = won - p
        suf.append(np.array([len(x), x.sum(), (x * x).sum(), y.sum(),
                             (x * y).sum()]))
    suf = np.array(suf)

    def fit(t):
        n, sx, sxx, sy, sxy = t
        den = n * sxx - sx * sx
        return (n * sxy - sx * sy) / den if den else float("nan")
    pt = fit(suf.sum(0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(nboot, len(keys)))
    ss = suf[idx].sum(1)
    n, sx, sxx, sy, sxy = ss.T
    sl = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    res = (pt, np.percentile(sl, [2.5, 97.5]), len(keys))
    if jack:
        tot = suf.sum(0)
        j = sorted((fit(tot - suf[i]), keys[i], int(suf[i][0]))
                   for i in range(len(keys)))
        return res + (j,)
    return res


for name in ("outdoor", "indoor"):
    for we in (False, True):
        pt, ci, nk = edge_slope(by[name], within_event=we)
        print(f"  {name:8s} {'within-event' if we else 'pooled      '} "
              f"slope={pt:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] per +10F "
              f"(ev={nk})")
pt, ci, nk, j = edge_slope(by["outdoor"], jack=True)
print(f"  outdoor pooled jackknife range [{j[0][0]:+.4f}, {j[-1][0]:+.4f}]")
for d_, k, n_ in j[:3] + j[-3:]:
    print(f"    drop n={n_:5d}  slope -> {d_:+.4f}   "
          f"{geo.get(k,{}).get('venue','')[:55]}")
print()

# ---- leave-one-event-out on the 92F+ binned drift ----------------------
print("### CHECK 5 — leave-one-event-out on the 92F+ minus <70F drift "
      "(audited outdoor) — the retraction statistic")
rs = by["outdoor"]
recs = []
for r in rs:
    p = game_win_prob(abs(r["eta"]))
    recs.append({"ev": r["ev"], "T": r["T"], "p": p,
                 "won": 1.0 if (r["y"] > 0) == (r["eta"] >= 0) else 0.0})


def drift(sub):
    hot = [r for r in sub if r["T"] >= 92]
    cold = [r for r in sub if r["T"] < 70]
    if len(hot) < 30:
        return None, len(hot)
    eh = sum(r["won"] for r in hot) / len(hot) - sum(r["p"] for r in hot) / len(hot)
    ec = sum(r["won"] for r in cold) / len(cold) - sum(r["p"] for r in cold) / len(cold)
    return eh - ec, len(hot)


full, nh = drift(recs)
print(f"  full sample: drift = {full:+.4f} on {nh} hot games")
hotev = sorted({r["ev"] for r in recs if r["T"] >= 92})
for ev in hotev:
    d_, n_ = drift([r for r in recs if r["ev"] != ev])
    nn = sum(1 for r in recs if r["ev"] == ev and r["T"] >= 92)
    lab = f"{d_:+.4f}" if d_ is not None else "n/a (<30 hot games left)"
    print(f"    drop event with {nn:4d} hot games -> drift {lab:>24s}   "
          f"{geo.get(ev,{}).get('venue','')[:45]}")
