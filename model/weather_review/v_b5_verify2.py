"""ADVERSARIAL VERIFICATION of B5, part 2 -- attack the temperature thread and
audit the reported secondary numbers."""
from __future__ import annotations
import csv, math, sys, datetime as dt
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
exec(open(Path(__file__).resolve().parent / "v_b5_verify.py").read().split("print(\"=== SAMPLE")[0])

OM = [r for r in M if r["lab_c"] == "outdoor"]; IM = [r for r in M if r["lab_c"] == "indoor"]
OG = [r for r in G if r["lab_c"] == "outdoor"]; IG = [r for r in G if r["lab_c"] == "indoor"]

def design(rows, xkeys, ykey, wkey="w"):
    X = np.column_stack([np.ones(len(rows))] + [np.array([float(r[k]) for r in rows]) for k in xkeys])
    y = np.array([r[ykey] for r in rows]); w = np.array([r[wkey] for r in rows])
    cl = np.array([r["ev"] for r in rows]); return X, y, w, cl

def wls_cr1(rows, xkeys, ykey, wkey="w"):
    X, y, w, cl = design(rows, xkeys, ykey, wkey)
    XtWX = X.T@(X*w[:, None]); Ainv = np.linalg.pinv(XtWX)
    beta = Ainv@(X.T@(w*y)); u = y - X@beta
    keys = defaultdict(list)
    for i, c in enumerate(cl):
        keys[c].append(i)
    meat = np.zeros_like(XtWX)
    for c, idx in keys.items():
        idx = np.array(idx); s = (X[idx]*(w[idx]*u[idx])[:, None]).sum(0); meat += np.outer(s, s)
    Gn = len(keys); N, k = X.shape
    V = (Gn/(Gn-1))*((N-1)/(N-k))*Ainv@meat@Ainv
    se = np.sqrt(np.abs(np.diag(V)))
    return dict(zip(["const"]+list(xkeys), zip(beta, se))), Gn, N

def rep(tag, res, name, Gn, N, extra=""):
    b, se = res[name]; z = b/se
    p = 2*(1-.5*(1+math.erf(abs(z)/math.sqrt(2))))
    print(f"{tag:56s} {b:+.4f} [{b-1.96*se:+.4f},{b+1.96*se:+.4f}] p={p:.3f} (G={Gn},n={N}) {extra}")
    return b, se, p

def enrich(rows):
    out = []
    for r in rows:
        y, m, d = (int(x) for x in r["date"].split("-"))
        doy = dt.date(y, m, d).timetuple().tm_yday
        rr = dict(r)
        rr["t10"] = r["temp"]/10.
        rr["seas"] = math.cos(2*math.pi*(doy-200)/365.)
        rr["days"] = (dt.date(y, m, d)-dt.date(2024, 1, 1)).days/365.
        rr["qual"] = 1. if "qual" in r["stage"].lower() else 0.
        rr["hour_c"] = (r["hour"]-14)/6.
        rr["mlp"] = 1. if r["tour"] == "MLP" else 0.
        rr["sk2"] = r["skill"]*abs(r["skill"])
        rr["mo"] = m
        rr["yr"] = y
        for k in ("t10", "seas", "days", "qual", "hour_c", "night", "sust", "mlp", "gustiness"):
            rr["sk_x_"+k] = r["skill"]*rr[k]
        out.append(rr)
    return out

O, I = enrich(OG), enrich(IG)

print("=== A. composition of the two arms ===")
for tag, rows in (("outdoor", O), ("indoor", I)):
    print(f" {tag}: MLP frac {sum(r['mlp'] for r in rows)/len(rows):.3f}, "
          f"qual frac {sum(r['qual'] for r in rows)/len(rows):.3f}, "
          f"temp mean {sum(r['temp'] for r in rows)/len(rows):.1f}F "
          f"sd {np.std([r['temp'] for r in rows]):.1f}, "
          f"years {sorted({r['yr'] for r in rows})}")
tt = np.array([r["t10"] for r in O]); ss = np.array([r["seas"] for r in O])
print(f" outdoor corr(temp, seasonal cos wave) = {np.corrcoef(tt,ss)[0,1]:+.3f}  "
      "(the 'horse race' is between two near-duplicates)")
tt = np.array([r["t10"] for r in I]); ss = np.array([r["seas"] for r in I])
print(f" indoor  corr(temp, seasonal cos wave) = {np.corrcoef(tt,ss)[0,1]:+.3f}")

print("\n=== B. binned favourite edge by temp: reproduce + decompose ===")
TB = [(-50, 55), (55, 65), (65, 75), (75, 85), (85, 92), (92, 150)]
print(" arm      bin      n     edge(pp)   mean|skill|   edge/|skill| (=b-1)")
for tag, rows in (("outdoor", O), ("indoor", I)):
    for lo, hi in TB:
        sub = [r for r in rows if lo <= r["temp"] < hi]
        if len(sub) < 40:
            print(f" {tag:8s} {lo}-{hi}: n={len(sub)} too thin")
            continue
        ed = sum((1 if r["skill"] >= 0 else -1)*r["share"] - abs(r["skill"]) for r in sub)/len(sub)
        mk = sum(abs(r["skill"]) for r in sub)/len(sub)
        print(f" {tag:8s} {lo if lo>-50 else '<':>4}-{hi if hi<150 else '+':<4} {len(sub):6d}  "
              f"{100*ed:+7.2f}    {mk:.4f}      {ed/mk:+.4f}")
print(" reported outdoor: +0.43/+1.05/+1.42/+1.55/+1.71/+1.83 ; indoor +2.04/+3.62/+1.60/+1.04/+0.61/+1.70")

print("\n=== C. LEAVE-ONE-EVENT-OUT on the HEADLINE coefficient (skill x temp/10) ===")
XS = ["skill", "sust", "t10", "sk_x_sust", "sk_x_t10"]
def blocks(rows, xkeys, ykey):
    X, y, w, cl = design(rows, xkeys, ykey)
    acc = {}
    p = X.shape[1]
    for i in range(len(rows)):
        c = cl[i]
        if c not in acc:
            acc[c] = [np.zeros((p, p)), np.zeros(p), 0]
        acc[c][0] += w[i]*np.outer(X[i], X[i]); acc[c][1] += w[i]*y[i]*X[i]; acc[c][2] += 1
    return acc
acc = blocks(O, XS, "share"); keys = list(acc)
Gt = sum(acc[k][0] for k in keys); bt = sum(acc[k][1] for k in keys)
j = XS.index("sk_x_t10")+1
full = np.linalg.solve(Gt, bt)[j]
jk = sorted((np.linalg.solve(Gt-acc[k][0], bt-acc[k][1])[j], k, acc[k][2]) for k in keys)
print(f" full {full:+.4f}; LOEO range [{jk[0][0]:+.4f}, {jk[-1][0]:+.4f}] over {len(jk)} events; "
      f"sign flip: {'YES' if jk[0][0]*jk[-1][0] < 0 else 'no'}")
print(f"   most-negative-pull event drop -> {jk[-1][0]:+.4f} (n={jk[-1][2]}), "
      f"most-positive-pull drop -> {jk[0][0]:+.4f} (n={jk[0][2]})")
# drop the 5 most influential
infl = sorted(keys, key=lambda k: -abs(np.linalg.solve(Gt-acc[k][0], bt-acc[k][1])[j]-full))
for n in (1, 3, 5):
    Gd = Gt - sum(acc[k][0] for k in infl[:n]); bd = bt - sum(acc[k][1] for k in infl[:n])
    print(f"   drop {n} most influential events -> {np.linalg.solve(Gd, bd)[j]:+.4f}")

print("\n=== D. IS IT THE CALENDAR? skill x month-of-year fixed effects ===")
print("  (12 skill x month dummies absorb ANY annual sawtooth; temp then identified")
print("   only off WITHIN-MONTH temperature variation -- much stronger than a cos wave)")
for tag, rows in (("outdoor", O), ("indoor", I)):
    months = sorted({r["mo"] for r in rows})[1:]
    rr = []
    for r in rows:
        d = dict(r)
        for m in months:
            d[f"m{m}"] = 1. if r["mo"] == m else 0.
            d[f"sk_x_m{m}"] = r["skill"]*d[f"m{m}"]
        rr.append(d)
    xs = XS + [f"m{m}" for m in months] + [f"sk_x_m{m}" for m in months]
    res, Gn, N = wls_cr1(rr, xs, "share")
    rep(f" skill x temp | skill x month FE [{tag}]", res, "sk_x_t10", Gn, N)
    # + year x skill too
    yrs = sorted({r["yr"] for r in rows})[1:]
    for d in rr:
        for yy in yrs:
            d[f"y{yy}"] = 1. if d["yr"] == yy else 0.
            d[f"sk_x_y{yy}"] = d["skill"]*d[f"y{yy}"]
    xs2 = xs + [f"y{yy}" for yy in yrs] + [f"sk_x_y{yy}" for yy in yrs]
    res, Gn, N = wls_cr1(rr, xs2, "share")
    rep(f" skill x temp | skill x (month + year) FE [{tag}]", res, "sk_x_t10", Gn, N)

print("\n=== E. tour confound: MLP is 24% of the hottest bin, 1% of the cold bins ===")
for tag, rows in (("outdoor", O), ("indoor", I)):
    res, Gn, N = wls_cr1(rows, XS+["mlp", "sk_x_mlp"], "share")
    rep(f" skill x temp + skill x MLP [{tag}]", res, "sk_x_t10", Gn, N)
    sub = [r for r in rows if r["mlp"] == 0]
    res, Gn, N = wls_cr1(sub, XS, "share")
    rep(f" skill x temp, PPA only [{tag}]", res, "sk_x_t10", Gn, N)

print("\n=== F. per-year replication (independent) ===")
for tag, rows in (("outdoor", O), ("indoor", I)):
    for yy in (2024, 2025, 2026):
        sub = [r for r in rows if r["yr"] == yy]
        if len(sub) < 500:
            continue
        res, Gn, N = wls_cr1(sub, XS, "share")
        rep(f" skill x temp {yy} [{tag}]", res, "sk_x_t10", Gn, N)

print("\n=== G. label arms (independent) ===")
for key, lbl in (("lab_c", "corrected"), ("lab_p", "published"), ("lab_h", "high-conf only")):
    for arm in ("outdoor", "indoor"):
        sub = enrich([r for r in G if r.get(key) == arm])
        if len(sub) < 500:
            continue
        res, Gn, N = wls_cr1(sub, XS, "share")
        rep(f" skill x temp [{lbl} / {arm}]", res, "sk_x_t10", Gn, N)

print("\n=== H. audit of secondary reported numbers ===")
# swirl
for tag, rows in (("outdoor", OM), ("indoor", IM)):
    sub = [r for r in rows if r.get("swirl") is not None]
    res, Gn, N = wls_cr1(sub, ["sust", "swirl"], "serve_rate")
    rep(f" S swirl [{tag}]", res, "swirl", Gn, N, "(reported out -0.0023 [-0.0045,+0.0000] p=0.050)")
sw = np.array([r["swirl"]*30 for r in OG if r.get("swirl") is not None])
su = np.array([r["sust"]*10 for r in OG if r.get("swirl") is not None])
print(f"   corr(swirl,sust) outdoor = {np.corrcoef(sw,su)[0,1]:+.3f} (reported -0.461)")
for lbl, lo in (("sust>=8", 8), ("sust>=12", 12)):
    sub = [r for r in OM if r.get("swirl") is not None and r["sust"]*10 >= lo]
    res, Gn, N = wls_cr1(sub, ["sust", "swirl"], "serve_rate")
    rep(f" S swirl [outdoor, {lbl}]", res, "swirl", Gn, N)
# other channels
for ch, xk in (("gust", ["gust"]), ("wet", ["sust", "wet"]), ("cold", ["sust", "cold"]),
               ("night", ["sust", "night"])):
    for tag, rows in (("outdoor", OM), ("indoor", IM)):
        sub = [r for r in rows if all(r.get(k) is not None for k in xk)]
        res, Gn, N = wls_cr1(sub, xk, "serve_rate")
        rep(f" S {ch} [{tag}]", res, ch, Gn, N)
# attenuation r
both = []
for r in rd(ROOT/"data/match_times.csv"):
    if r["start_local"] and r["planned_start_local"]:
        wa = H.get((r["event_id"], r["start_local"][:13])); wp = H.get((r["event_id"], r["planned_start_local"][:13]))
        if wa and wp and wa[1] is not None and wp[1] is not None:
            both.append((wa[1]-wa[0], wp[1]-wp[0], wa[0], wp[0], wa[2], wp[2]))
A = np.array(both)
print(f"\n attenuation on n={len(both)} (reported 13242): gustiness r={np.corrcoef(A[:,0],A[:,1])[0,1]:+.3f} "
      f"(rep +0.874), sustained r={np.corrcoef(A[:,2],A[:,3])[0,1]:+.3f} (rep +0.904), "
      f"temp r={np.corrcoef(A[:,4],A[:,5])[0,1]:+.3f} (rep +0.981)")
# what share of the ANALYSIS sample uses planned times?
for tag, rows in (("match rows", M), ("outdoor matches", OM), ("outdoor games", OG)):
    print(f"   {tag}: planned-time share = {1-sum(r['actual'] for r in rows)/len(rows):.3f}")

print("\n=== I. exposure percentiles (reported vs recomputed, outdoor games) ===")
def pct(v, q):
    v = sorted(v); return v[min(len(v)-1, int(q*len(v)))]
for k, sc in (("sust", 10), ("gust", 10), ("gustiness", 10), ("swirl", 30)):
    v = [r[k]*sc for r in OG if r.get(k) is not None]
    print(f"  {k}: p10 {pct(v,.1):.1f} p50 {pct(v,.5):.1f} p90 {pct(v,.9):.1f} "
          f"p99 {pct(v,.99):.1f} max {max(v):.1f}")
tv = [r["temp"] for r in OG]
print(f"  temp: p10 {pct(tv,.1):.1f} p50 {pct(tv,.5):.1f} p90 {pct(tv,.9):.1f} p99 {pct(tv,.99):.1f} max {max(tv):.1f}")
print(f"  n gust>=25 {sum(1 for r in OG if r['gust']*10>=25)} (rep 2839); "
      f"n temp<60 {sum(1 for r in OG if r['temp']<60)} (rep 2393); "
      f"n night {sum(1 for r in OG if r['night'])} (rep 3453)")
