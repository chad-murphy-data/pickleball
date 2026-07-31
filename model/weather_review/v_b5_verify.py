"""ADVERSARIAL VERIFICATION of B5 (gusts/rain/cold/swirl/day-night).

Independent re-build of the dataset from the committed CSVs (does NOT import
b5_channels.py or b2b_lib.py), independent estimators:
  * analytic CR1 cluster-robust sandwich SEs (vs their percentile cluster boot)
  * WILD cluster bootstrap (Rademacher, null imposed) -- the right small-G tool
  * game-1-only re-fit (kills the best-of-3 decider collider)
  * match-level collapse (kills within-match double counting)
  * binary game-win logit (different outcome scale)
  * |skill| composition by temperature bin (binned-table artifact check)
  * seed sensitivity of their own bootstrap
"""
from __future__ import annotations
import csv, math, sys, datetime as dt
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "web"))
from sitelib.race import sigmoid, team_eta  # noqa

def rd(p):
    with open(p) as f:
        return list(csv.DictReader(f))

def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

# ---------------------------------------------------------------- labels
geo = {r["event_id"]: r["setting"] for r in rd(ROOT / "data/event_geo.csv")}
ov = rd(ROOT / "data/venue_overrides.csv")
lab_pub, lab_corr, lab_hi = {}, {}, {}
for e, s in geo.items():
    lab_pub[e] = s if s in ("indoor", "outdoor") else None
    lab_corr[e] = lab_pub[e]
for r in ov:
    e, s, c = r["event_id"], r["setting"], r["confidence"]
    lab_corr[e] = s if s in ("indoor", "outdoor") else None
    if c == "high" and s in ("indoor", "outdoor"):
        lab_hi[e] = s

# ---------------------------------------------------------------- weather
H = {}
for r in rd(ROOT / "data/event_weather_hourly.csv"):
    H[(r["event_id"], r["local_time"][:13])] = (
        fnum(r["windspeed_10m"]), fnum(r["windgusts_10m"]), fnum(r["temperature_2m"]),
        fnum(r["precipitation"]), fnum(r["winddirection_10m"]))

def shift(k, d):
    return (dt.datetime.strptime(k, "%Y-%m-%dT%H") + dt.timedelta(hours=d)).strftime("%Y-%m-%dT%H")

MH = {}
for r in rd(ROOT / "data/match_times.csv"):
    ts, act = r["start_local"], 1
    if not ts:
        ts, act = r["planned_start_local"], 0
    if ts:
        MH[r["match_id"]] = (ts[:13], act)

v2 = {r["player_id"]: float(r["value_now_mean"]) for r in rd(ROOT / "data/v2_players.csv")}
rally = {r["match_id"]: r for r in rd(ROOT / "data/match_rally_summary.csv")
         if r["discipline"] == "doubles" and int(r["n_rallies"]) >= 20}

by_match = defaultdict(list)
for g in rd(ROOT / "data/games.csv"):
    if g["is_dreambreaker"] == "True" or g["is_forfeit"] == "True":
        continue
    by_match[g["match_id"]].append(g)

def circ_sd(a):
    if len(a) < 3:
        return None
    c = sum(math.cos(math.radians(x)) for x in a) / len(a)
    s = sum(math.sin(math.radians(x)) for x in a) / len(a)
    R = min(max(math.hypot(c, s), 1e-9), 1.0)
    return math.degrees(math.sqrt(-2 * math.log(R)))

M, G = [], []   # match rows (serve rate), game rows (share)
for mid, gs in by_match.items():
    g0 = gs[0]; ev = g0["event_id"]
    if mid not in MH:
        continue
    hk, act = MH[mid]
    w = H.get((ev, hk))
    if not w or w[0] is None or w[1] is None:
        continue
    sust, gust, temp, _, _ = w
    pr = [H.get((ev, shift(hk, d)), (None,)*5)[3] for d in (-2, -1, 0)]
    pr = [x for x in pr if x is not None]
    prcp3 = sum(pr) if pr else None
    dirs = [H.get((ev, shift(hk, d)), (None,)*5)[4] for d in (-1, 0, 1, 2)]
    dirs = [x for x in dirs if x is not None]
    sw = circ_sd(dirs)
    hour = int(hk[11:13])
    base = dict(ev=ev, mid=mid, tour=g0["tour"], date=g0["date"], actual=act,
                sust=sust/10., gustiness=(gust-sust)/10., gust=gust/10.,
                temp=temp, t10=(temp/10. if temp is not None else None),
                cold=(max(0., 60.-temp)/10. if temp is not None else None),
                wet=((1. if (prcp3 or 0) > .01 else 0.) if prcp3 is not None else None),
                swirl=(sw/30. if sw is not None else None),
                night=1. if hour >= 17 else 0., hour=hour,
                lab_c=lab_corr.get(ev), lab_p=lab_pub.get(ev), lab_h=lab_hi.get(ev),
                best_of=int(g0["best_of"] or 0))
    if mid in rally:
        rs = rally[mid]
        m = dict(base); m["n_rallies"] = int(rs["n_rallies"])
        m["w"] = float(rs["n_rallies"])
        m["serve_rate"] = int(rs["n_points"]) / int(rs["n_rallies"])
        M.append(m)
    ngames = len(gs)
    for g in gs:
        vals = [v2.get(g[k]) for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if not all(v is not None for v in vals):
            continue
        s1, s2 = int(g["t1_score"]), int(g["t2_score"])
        if s1 + s2 < 11:
            continue
        r = dict(base)
        r["skill"] = sigmoid(team_eta(*vals)) - .5
        r["share"] = s1/(s1+s2) - .5
        r["win"] = 1.0 if s1 > s2 else 0.0
        r["w"] = 1.0
        r["gnum"] = int(g["game_number"])
        r["ngames_match"] = ngames
        r["stage"] = g["stage"]
        G.append(r)

OM = [r for r in M if r["lab_c"] == "outdoor"]; IM = [r for r in M if r["lab_c"] == "indoor"]
OG = [r for r in G if r["lab_c"] == "outdoor"]; IG = [r for r in G if r["lab_c"] == "indoor"]

print("=== SAMPLE (independent rebuild) ===")
print(f"matches w/ logs: outdoor {len(OM)} (events {len({r['ev'] for r in OM})}) "
      f"| indoor {len(IM)} (events {len({r['ev'] for r in IM})})")
print(f"games w/ v2:     outdoor {len(OG)} (events {len({r['ev'] for r in OG})}) "
      f"| indoor {len(IG)} (events {len({r['ev'] for r in IG})})")
print(f"reported:        8738 / 2666 matches, 24718 / 7127 games, 68-69 / 28-30 events")

gusty = [r for r in OG if r["gust"]*10 >= 25]
hid = sum(1 for r in gusty if r["sust"]*10 < 14)
print(f"concealment: {hid}/{len(gusty)} = {100*hid/len(gusty):.1f}% (reported 1713/2839 = 60%)")
a = np.array([r["gustiness"] for r in OG]); b = np.array([r["sust"] for r in OG])
print(f"corr(sust,gustiness) outdoor = {np.corrcoef(a,b)[0,1]:+.3f} (reported +0.710)")

# ---------------------------------------------------------------- estimators
def design(rows, xkeys, ykey, wkey="w"):
    X = np.column_stack([np.ones(len(rows))] + [np.array([r[k] for r in rows]) for k in xkeys])
    y = np.array([r[ykey] for r in rows])
    w = np.array([r[wkey] for r in rows])
    cl = np.array([r["ev"] for r in rows])
    return X, y, w, cl

def wls_cr1(rows, xkeys, ykey, wkey="w"):
    """OLS/WLS point estimate + CR1 cluster-robust sandwich SE."""
    X, y, w, cl = design(rows, xkeys, ykey, wkey)
    XtWX = X.T @ (X * w[:, None])
    beta = np.linalg.solve(XtWX, X.T @ (w * y))
    u = y - X @ beta
    Ainv = np.linalg.inv(XtWX)
    keys = {}
    for i, c in enumerate(cl):
        keys.setdefault(c, []).append(i)
    meat = np.zeros_like(XtWX)
    for c, idx in keys.items():
        idx = np.array(idx)
        s = (X[idx] * (w[idx]*u[idx])[:, None]).sum(0)
        meat += np.outer(s, s)
    Gn = len(keys); N, k = X.shape
    c1 = (Gn/(Gn-1)) * ((N-1)/(N-k))
    V = c1 * Ainv @ meat @ Ainv
    se = np.sqrt(np.diag(V))
    names = ["const"] + list(xkeys)
    return dict(zip(names, zip(beta, se))), Gn, N, (X, y, w, cl, beta, u, Ainv)

def wild_cluster_p(rows, xkeys, ykey, target, wkey="w", B=2000, seed=11):
    """Wild cluster bootstrap-t, Rademacher, NULL IMPOSED (WCR).  Small-G safe."""
    X, y, w, cl = design(rows, xkeys, ykey, wkey)
    names = ["const"] + list(xkeys)
    j = names.index(target)
    # unrestricted
    XtWX = X.T @ (X*w[:, None]); Ainv = np.linalg.inv(XtWX)
    beta = Ainv @ (X.T @ (w*y))
    u = y - X@beta
    ukeys = {}
    for i, c in enumerate(cl):
        ukeys.setdefault(c, []).append(i)
    cids = list(ukeys); idxs = [np.array(ukeys[c]) for c in cids]
    def crse(Xd, wd, ud, Ad):
        meat = np.zeros((Xd.shape[1], Xd.shape[1]))
        for ix in idxs:
            s = (Xd[ix]*(wd[ix]*ud[ix])[:, None]).sum(0)
            meat += np.outer(s, s)
        Gn = len(idxs); N, k = Xd.shape
        V = (Gn/(Gn-1))*((N-1)/(N-k)) * Ad@meat@Ad
        return np.sqrt(np.diag(V))
    t_obs = beta[j]/crse(X, w, u, Ainv)[j]
    # restricted fit (drop target)
    keep = [i for i in range(X.shape[1]) if i != j]
    Xr = X[:, keep]
    Ar = np.linalg.inv(Xr.T@(Xr*w[:, None]))
    br = Ar @ (Xr.T@(w*y))
    ur = y - Xr@br
    fitted_r = Xr@br
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(B):
        v = rng.choice([-1.0, 1.0], size=len(idxs))
        ustar = ur.copy()
        for m, ix in enumerate(idxs):
            ustar[ix] = ur[ix]*v[m]
        ystar = fitted_r + ustar
        bstar = Ainv @ (X.T@(w*ystar))
        us = ystar - X@bstar
        ts = bstar[j]/crse(X, w, us, Ainv)[j]
        if abs(ts) >= abs(t_obs) - 1e-12:
            cnt += 1
    return t_obs, (cnt+1)/(B+1)

def add_ix(rows, chans):
    out = []
    for r in rows:
        rr = dict(r)
        for c in chans:
            rr["sk_x_"+c] = r["skill"]*r[c]
        out.append(rr)
    return out

def rep(tag, res, name, Gn, N, extra=""):
    b, se = res[name]
    lo, hi = b-1.96*se, b+1.96*se
    z = b/se
    p = 2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))
    print(f"{tag:52s} {b:+.4f} [{lo:+.4f},{hi:+.4f}] se={se:.4f} p={p:.3f} "
          f"(G={Gn}, n={N}) {extra}")
    return b, se, p

print("\n=== 1. PRIMARY: gustiness -> serve rate (independent CR1 sandwich) ===")
for tag, rows in (("outdoor", OM), ("indoor", IM)):
    res, Gn, N, _ = wls_cr1(rows, ["sust", "gustiness"], "serve_rate")
    rep(f"S gustiness [{tag}] WLS/CR1", res, "gustiness", Gn, N)
    # unweighted variant (different aggregation)
    for r in rows:
        r["w1"] = 1.0
    res2, Gn2, N2, _ = wls_cr1(rows, ["sust", "gustiness"], "serve_rate", wkey="w1")
    rep(f"S gustiness [{tag}] UNWEIGHTED/CR1", res2, "gustiness", Gn2, N2)
t, p = wild_cluster_p(OM, ["sust", "gustiness"], "serve_rate", "gustiness")
print(f"  wild-cluster (null imposed) outdoor: t={t:+.3f}  p={p:.3f}  (their boot p=0.073)")

print("\n=== 2. skill x gustiness (favourite compression) ===")
for tag, rows in (("outdoor", OG), ("indoor", IG)):
    rr = add_ix(rows, ["sust", "gustiness"])
    res, Gn, N, _ = wls_cr1(rr, ["skill", "sust", "gustiness", "sk_x_sust", "sk_x_gustiness"], "share")
    rep(f"F skill x gustiness [{tag}] CR1", res, "sk_x_gustiness", Gn, N)

print("\n=== 3. POST-HOC skill x temp -- reproduce base spec ===")
for tag, rows in (("outdoor", OG), ("indoor", IG)):
    rr = add_ix(rows, ["sust", "t10"])
    res, Gn, N, _ = wls_cr1(rr, ["skill", "sust", "t10", "sk_x_sust", "sk_x_t10"], "share")
    rep(f"skill x temp/10F [{tag}] CR1", res, "sk_x_t10", Gn, N,
        extra="(reported +0.0305 [+0.0040,+0.0586] / indoor -0.0146..-0.0201)")
rr = add_ix(OG, ["sust", "t10"])
t, p = wild_cluster_p(rr, ["skill", "sust", "t10", "sk_x_sust", "sk_x_t10"], "share", "sk_x_t10")
print(f"  wild-cluster (null imposed) outdoor: t={t:+.3f}  p={p:.3f}  (their boot p=0.022)")

print("\n=== 4. COLLIDER CHECK: best-of-3 game selection ===")
# composition: does the mix of 1/2/3-game matches shift with temperature?
print("  temp bin | games | frac from 3-game matches | frac MLP | mean|skill| | mean n_pts")
TB = [(-50, 55), (55, 65), (65, 75), (75, 85), (85, 92), (92, 150)]
for lo, hi in TB:
    sub = [r for r in OG if lo <= r["temp"] < hi]
    if not sub:
        continue
    f3 = sum(1 for r in sub if r["ngames_match"] >= 3)/len(sub)
    fm = sum(1 for r in sub if r["tour"] == "MLP")/len(sub)
    mk = sum(abs(r["skill"]) for r in sub)/len(sub)
    print(f"   {lo if lo>-50 else '<':>4}-{hi if hi<150 else '+':<4} | {len(sub):6d} | "
          f"{f3:.3f} | {fm:.3f} | {mk:.4f}")
print("  -> if mean|skill| is flat the binned 'edge' table is not a composition artifact;")
print("     if frac-3-game moves, the pooled skill slope is contaminated by the collider.")

print("\n  GAME 1 ONLY (no best-of-3 selection at all):")
for tag, rows in (("outdoor", OG), ("indoor", IG)):
    sub = [r for r in rows if r["gnum"] == 1]
    rr = add_ix(sub, ["sust", "t10"])
    res, Gn, N, _ = wls_cr1(rr, ["skill", "sust", "t10", "sk_x_sust", "sk_x_t10"], "share")
    rep(f"skill x temp [{tag}, game 1 only]", res, "sk_x_t10", Gn, N)
sub = [r for r in OG if r["gnum"] == 1]
rr = add_ix(sub, ["sust", "t10"])
t, p = wild_cluster_p(rr, ["skill", "sust", "t10", "sk_x_sust", "sk_x_t10"], "share", "sk_x_t10")
print(f"  wild-cluster game-1 outdoor: t={t:+.3f}  p={p:.3f}")

print("\n  MATCH-LEVEL COLLAPSE (mean share over the match's games, one row/match):")
for tag, rows in (("outdoor", OG), ("indoor", IG)):
    agg = defaultdict(list)
    for r in rows:
        agg[r["mid"]].append(r)
    coll = []
    for mid, rs in agg.items():
        r0 = dict(rs[0])
        r0["share"] = sum(x["share"] for x in rs)/len(rs)
        coll.append(r0)
    rr = add_ix(coll, ["sust", "t10"])
    res, Gn, N, _ = wls_cr1(rr, ["skill", "sust", "t10", "sk_x_sust", "sk_x_t10"], "share")
    rep(f"skill x temp [{tag}, match-collapsed]", res, "sk_x_t10", Gn, N)

print("\n=== 5. DIFFERENT OUTCOME: game win (linear prob) skill x temp ===")
for tag, rows in (("outdoor", OG), ("indoor", IG)):
    rr = add_ix(rows, ["sust", "t10"])
    for r in rr:
        r["winc"] = r["win"] - 0.5
    res, Gn, N, _ = wls_cr1(rr, ["skill", "sust", "t10", "sk_x_sust", "sk_x_t10"], "winc")
    rep(f"skill x temp on WIN [{tag}]", res, "sk_x_t10", Gn, N)

print("\n=== 6. SEED SENSITIVITY of their own percentile cluster bootstrap ===")
def their_boot(rows, xkeys, ykey, target, seed, nboot=2000):
    X, y, w, cl = design(rows, xkeys, ykey)
    names = ["const"]+list(xkeys); j = names.index(target)
    acc = {}
    p = X.shape[1]
    for i in range(len(rows)):
        c = cl[i]
        if c not in acc:
            acc[c] = [np.zeros((p, p)), np.zeros(p)]
        acc[c][0] += w[i]*np.outer(X[i], X[i]); acc[c][1] += w[i]*y[i]*X[i]
    keys = list(acc)
    Gs = np.stack([acc[k][0] for k in keys]); bs = np.stack([acc[k][1] for k in keys])
    rng = np.random.default_rng(seed)
    ii = rng.integers(0, len(keys), size=(nboot, len(keys)))
    d = []
    for i in range(nboot):
        try:
            d.append(np.linalg.solve(Gs[ii[i]].sum(0), bs[ii[i]].sum(0))[j])
        except np.linalg.LinAlgError:
            pass
    d = np.sort(np.array(d))
    frac = float((d <= 0).mean())
    return d[int(.025*len(d))], d[int(.975*len(d))], max(2*min(frac, 1-frac), 1/len(d))
rr = add_ix(OG, ["sust", "t10"])
for sd in (20260731, 1, 99, 12345, 777):
    lo, hi, pv = their_boot(rr, ["skill", "sust", "t10", "sk_x_sust", "sk_x_t10"], "share", "sk_x_t10", sd)
    print(f"  seed {sd:>9}: skill x temp CI [{lo:+.4f},{hi:+.4f}] p={pv:.3f}")

print("\n=== 7. within-event: is 'temp' just hour-of-day? ===")
grp = defaultdict(list)
for r in OG:
    grp[r["ev"]].append(r)
num = den = 0.0
xs, ys = [], []
for ev, rs in grp.items():
    if len(rs) < 5:
        continue
    mt = sum(r["temp"] for r in rs)/len(rs); mh = sum(r["hour"] for r in rs)/len(rs)
    for r in rs:
        xs.append(r["hour"]-mh); ys.append(r["temp"]-mt)
xs, ys = np.array(xs), np.array(ys)
print(f"  within-event corr(hour, temp) outdoor = {np.corrcoef(xs, ys)[0,1]:+.3f} "
      f"(temp dev sd {ys.std():.1f}F, hour dev sd {xs.std():.1f}h)")
