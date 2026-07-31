"""ADVERSARIAL VERIFICATION of B5, part 3: event FE done with explicit dummies,
wild-cluster inference, month-x-year FE, and the DiD."""
from __future__ import annotations
import csv, math, sys, datetime as dt
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
exec(open(Path(__file__).resolve().parent / "v_b5_verify.py").read().split("print(\"=== SAMPLE")[0])

OG = [r for r in G if r["lab_c"] == "outdoor"]; IG = [r for r in G if r["lab_c"] == "indoor"]

def enrich(rows):
    out = []
    for r in rows:
        y, m, d = (int(x) for x in r["date"].split("-"))
        rr = dict(r); rr["t10"] = r["temp"]/10.
        rr["days"] = (dt.date(y, m, d)-dt.date(2024, 1, 1)).days/365.
        rr["qual"] = 1. if "qual" in r["stage"].lower() else 0.
        rr["hour_c"] = (r["hour"]-14)/6.
        rr["mlp"] = 1. if r["tour"] == "MLP" else 0.
        rr["sk2"] = r["skill"]*abs(r["skill"]); rr["mo"] = m; rr["yr"] = y
        rr["my"] = f"{y}-{m:02d}"
        for k in ("t10", "days", "qual", "hour_c", "night", "sust", "mlp", "sk2"):
            rr["sk_x_"+k] = r["skill"]*rr[k]
        out.append(rr)
    return out
O, I = enrich(OG), enrich(IG)

def fit(rows, xkeys, ykey="share", absorb=None, wkey="w"):
    """WLS with optional absorbed factor (explicit dummies) + CR1 by event."""
    cols = [np.ones(len(rows))] + [np.array([float(r[k]) for r in rows]) for k in xkeys]
    names = ["const"]+list(xkeys)
    if absorb:
        levs = sorted({r[absorb] for r in rows})[1:]
        for L in levs:
            cols.append(np.array([1.0 if r[absorb] == L else 0.0 for r in rows]))
            names.append(f"{absorb}={L}")
    X = np.column_stack(cols)
    y = np.array([r[ykey] for r in rows]); w = np.array([r[wkey] for r in rows])
    cl = np.array([r["ev"] for r in rows])
    A = np.linalg.pinv(X.T@(X*w[:, None])); beta = A@(X.T@(w*y)); u = y-X@beta
    ks = defaultdict(list)
    for i, c in enumerate(cl):
        ks[c].append(i)
    meat = np.zeros((X.shape[1],)*2)
    for c, ix in ks.items():
        ix = np.array(ix); s = (X[ix]*(w[ix]*u[ix])[:, None]).sum(0); meat += np.outer(s, s)
    Gn = len(ks); N, k = X.shape
    V = (Gn/(Gn-1))*((N-1)/max(N-k, 1))*A@meat@A
    se = np.sqrt(np.abs(np.diag(V)))
    return dict(zip(names, zip(beta, se))), Gn, N, (X, y, w, cl, names)

def rep(tag, res, nm, Gn, N):
    b, se = res[nm]; z = b/se
    p = 2*(1-.5*(1+math.erf(abs(z)/math.sqrt(2))))
    print(f"{tag:60s} {b:+.4f} [{b-1.96*se:+.4f},{b+1.96*se:+.4f}] p={p:.3f} (G={Gn},n={N})")
    return b, se, p

def wild(pack, target, B=1500, seed=5):
    X, y, w, cl, names = pack
    j = names.index(target)
    A = np.linalg.pinv(X.T@(X*w[:, None])); beta = A@(X.T@(w*y)); u = y-X@beta
    ks = defaultdict(list)
    for i, c in enumerate(cl):
        ks[c].append(i)
    idxs = [np.array(v) for v in ks.values()]
    Gn = len(idxs); N, k = X.shape; c1 = (Gn/(Gn-1))*((N-1)/max(N-k, 1))
    def se_j(ud):
        meat = np.zeros((k, k))
        for ix in idxs:
            s = (X[ix]*(w[ix]*ud[ix])[:, None]).sum(0); meat += np.outer(s, s)
        return math.sqrt(abs((c1*A@meat@A)[j, j]))
    t_obs = beta[j]/se_j(u)
    keep = [i for i in range(k) if i != j]; Xr = X[:, keep]
    Ar = np.linalg.pinv(Xr.T@(Xr*w[:, None])); br = Ar@(Xr.T@(w*y))
    fr = Xr@br; ur = y-fr
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(B):
        v = rng.choice([-1.0, 1.0], size=Gn); us = ur.copy()
        for m, ix in enumerate(idxs):
            us[ix] = ur[ix]*v[m]
        ys = fr+us; bs = A@(X.T@(w*ys)); uu = ys-X@bs
        if abs(bs[j]/se_j(uu)) >= abs(t_obs)-1e-12:
            cnt += 1
    return t_obs, (cnt+1)/(B+1)

BASE = ["skill", "sust", "t10", "sk_x_sust", "sk_x_t10"]
CLEAN = BASE + ["night", "hour_c", "qual", "sk2", "sk_x_night", "sk_x_hour_c", "sk_x_qual"]

print("=== J. EVENT FE with EXPLICIT DUMMIES (not FWL demeaning) + wild cluster ===")
for tag, rows in (("outdoor", O), ("indoor", I)):
    res, Gn, N, pk = fit(rows, BASE, absorb="ev")
    b, se, p = rep(f" event-FE base [{tag}]", res, "sk_x_t10", Gn, N)
    if tag == "outdoor":
        t, pw = wild(pk, "sk_x_t10"); print(f"    wild-cluster p = {pw:.3f} (t={t:+.2f}); their FWL boot said +0.0294 p=0.029")
    res, Gn, N, pk = fit(rows, CLEAN, absorb="ev")
    rep(f" event-FE + within-event controls [{tag}]", res, "sk_x_t10", Gn, N)
    if tag == "outdoor":
        t, pw = wild(pk, "sk_x_t10"); print(f"    wild-cluster p = {pw:.3f}; their CLEAN said +0.0318 p=0.029")

print("\n=== K. month-x-year FE interacted with skill (kills ANY calendar/staleness) ===")
for tag, rows in (("outdoor", O), ("indoor", I)):
    levs = sorted({r["my"] for r in rows})[1:]
    rr = []
    for r in rows:
        d = dict(r)
        for L in levs:
            d[f"my_{L}"] = 1. if r["my"] == L else 0.
            d[f"sk_my_{L}"] = r["skill"]*d[f"my_{L}"]
        rr.append(d)
    xs = BASE + [f"my_{L}" for L in levs] + [f"sk_my_{L}" for L in levs]
    res, Gn, N, pk = fit(rr, xs)
    rep(f" skill x temp | skill x (month-year) FE [{tag}] ({len(levs)+1} cells)", res, "sk_x_t10", Gn, N)

print("\n=== L. DiD (out - in), reproduce the reported +0.054 ===")
pooled = []
for tag, rows in (("outdoor", O), ("indoor", I)):
    o = 1.0 if tag == "outdoor" else 0.0
    for r in rows:
        d = dict(r); d["out"] = o
        for k in CLEAN:
            d["out_x_"+k] = o*r[k]
        pooled.append(d)
res, Gn, N, pk = fit(pooled, CLEAN+["out_x_"+k for k in CLEAN], absorb="ev")
rep(" DiD out x skill x temp (event FE, clean controls)", res, "out_x_sk_x_t10", Gn, N)
print("   reported +0.0540 [+0.0140,+0.0952] p=0.011")
t, pw = wild(pk, "out_x_sk_x_t10"); print(f"   wild-cluster p = {pw:.3f}")
# DiD with month-year skill controls in BOTH arms
for tag in ("with skill x month FE in both arms",):
    levs = sorted({r["mo"] for r in pooled})[1:]
    rr = []
    for r in pooled:
        d = dict(r)
        for m in levs:
            d[f"m{m}"] = 1. if r["mo"] == m else 0.
            d[f"sk_m{m}"] = r["skill"]*d[f"m{m}"]
        rr.append(d)
    xs = CLEAN+["out_x_"+k for k in CLEAN]+[f"m{m}" for m in levs]+[f"sk_m{m}" for m in levs]
    res, Gn, N, _ = fit(rr, xs, absorb="ev")
    rep(f" DiD, {tag}", res, "out_x_sk_x_t10", Gn, N)

print("\n=== M. is the outdoor signal driven by within- or between-event temp? ===")
grp = defaultdict(list)
for r in O:
    grp[r["ev"]].append(r)
rows = []
for ev, rs in grp.items():
    mu = sum(x["temp"] for x in rs)/len(rs)
    for r in rs:
        d = dict(r)
        d["tw"] = (r["temp"]-mu)/10.       # within-event deviation
        d["tb"] = mu/10.                    # event mean
        d["sk_x_tw"] = r["skill"]*d["tw"]; d["sk_x_tb"] = r["skill"]*d["tb"]
        rows.append(d)
res, Gn, N, _ = fit(rows, ["skill", "sust", "tw", "tb", "sk_x_sust", "sk_x_tw", "sk_x_tb"])
rep(" skill x WITHIN-event temp deviation", res, "sk_x_tw", Gn, N)
rep(" skill x BETWEEN-event mean temp", res, "sk_x_tb", Gn, N)

print("\n=== N. indoor arm: is its negative reading pure calendar? ===")
for lbl, xs in (("raw", BASE), ("+ skill x days", BASE+["days", "sk_x_days"])):
    res, Gn, N, _ = fit(I, xs)
    rep(f" indoor skill x temp, {lbl}", res, "sk_x_t10", Gn, N)
res, Gn, N, _ = fit(I, BASE, absorb="ev")
rep(" indoor skill x temp, event FE", res, "sk_x_t10", Gn, N)
print(f" indoor temp sd = {np.std([r['temp'] for r in I]):.1f}F vs outdoor "
      f"{np.std([r['temp'] for r in O]):.1f}F; indoor within-event temp sd = ", end="")
g2 = defaultdict(list)
for r in I:
    g2[r["ev"]].append(r["temp"])
print(f"{np.std([t-sum(v)/len(v) for v in g2.values() for t in v]):.1f}F")

print("\n=== O. indoor 12-test false-positive count (CR1 route) ===")
OM = [r for r in M if r["lab_c"] == "outdoor"]; IM = [r for r in M if r["lab_c"] == "indoor"]
CH = [("gustiness", ["sust", "gustiness"]), ("gust", ["gust"]), ("wet", ["sust", "wet"]),
      ("cold", ["sust", "cold"]), ("swirl", ["sust", "swirl"]), ("night", ["sust", "night"])]
ps_out, ps_in = {}, {}
for ch, xk in CH:
    for tag, mr, gr, store in (("outdoor", OM, O, ps_out), ("indoor", IM, I, ps_in)):
        sub = [r for r in mr if all(r.get(k) is not None for k in xk)]
        res, Gn, N, _ = fit(sub, xk, ykey="serve_rate")
        b, se = res[ch]; z = b/se
        store["S:"+ch] = 2*(1-.5*(1+math.erf(abs(z)/math.sqrt(2))))
        sub = [dict(r, **{"sk_x_"+k: r["skill"]*r[k] for k in xk})
               for r in gr if all(r.get(k) is not None for k in xk)]
        res, Gn, N, _ = fit(sub, ["skill"]+xk+["sk_x_"+k for k in xk])
        b, se = res["sk_x_"+ch]; z = b/se
        store["F:"+ch] = 2*(1-.5*(1+math.erf(abs(z)/math.sqrt(2))))
print(" indoor p<0.05:", sum(1 for v in ps_in.values() if v < .05),
      " p<0.10:", sum(1 for v in ps_in.values() if v < .10),
      " (reported 1 and 3);", {k: round(v, 3) for k, v in sorted(ps_in.items(), key=lambda x: x[1])[:4]})
print(" outdoor p<0.05:", sum(1 for v in ps_out.values() if v < .05),
      " p<0.10:", sum(1 for v in ps_out.values() if v < .10), ";",
      {k: round(v, 3) for k, v in sorted(ps_out.items(), key=lambda x: x[1])[:4]})
def holm(d):
    it = sorted(d.items(), key=lambda kv: kv[1]); m = len(it); adj = {}; pr = 0.
    for i, (k, p) in enumerate(it):
        a = max(min(1., (m-i)*p), pr); pr = a; adj[k] = a
    return adj
print(" Holm on the 12 outdoor pre-registered: min adj p =",
      f"{min(holm(ps_out).values()):.3f} (reported 0.606)")
