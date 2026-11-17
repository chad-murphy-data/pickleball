"""Which FLIGHTS does the incumbent miss entirely, and what do they look
like? A flight = the clicks between two consecutive contacts. Per flight:
n clicks, S share, median click speed, duration, and the tracker's hit
rate -- both as the scorer counts it today and with the two scoring
artifacts removed (interpolate to the click's own time; apply the
per-rally sub-frame phase fit on V clicks only).

AUTOPSY ONLY on r9/r10 -- nothing here tunes anything.
"""
import csv, json, sys
sys.path.insert(0, "/home/user/pickleball/vision/ballsearch")
sys.path.insert(0, "/home/user/pickleball/vision")
import numpy as np
import pathfirst as pf, corridor_dp as cdp

DV = "/home/user/pickleball/data/vision"
CON = f"{DV}/contact_labels_chicago0725.csv"
PC = json.loads(pf.TUNE_JSON.read_text())
PHASE = {}


def contacts(rally):
    man, pre = [], []
    for r in csv.DictReader(open(CON)):
        if int(r["rally_cum"]) != rally or r.get("contact", "1") == "0":
            continue
        t = float(r["t_refined_s"] or r["t_tap_s"])
        (man if r["source"] in ("manual", "divergent") else pre).append(t)
    return sorted(man or pre)


def fit_phase(ctx, tr, t0):
    fs = np.array(sorted(tr)); P = np.array([tr[f] for f in fs], float)
    def med(dl):
        ds = []
        for t, tx, ty, v in ctx["truth"]:
            if v != "V": continue
            fq = (t - t0) * 60 + dl
            i = np.searchsorted(fs, fq)
            if i == 0 or i >= len(fs) or fs[i] - fs[i-1] > 2: continue
            w = (fq - fs[i-1]) / (fs[i] - fs[i-1])
            px = P[i-1]*(1-w) + P[i]*w
            d = np.hypot(px[0]-tx, px[1]-ty)
            if d < 30: ds.append(d)
        return np.median(ds) if len(ds) > 20 else np.nan
    g = np.arange(-2.0, 2.01, 0.05)
    v = [med(d) for d in g]
    return float(g[int(np.nanargmin(v))]) if np.any(~np.isnan(v)) else 0.0


def run(rally):
    ctx = pf.context(rally); cdp.W_P_SOFT = 25.0
    res = pf.run(ctx, PC["p_seed"], PC["s_min"], int(PC["gap"]))
    tr, t0 = res["track"], ctx["t0"]
    fs = np.array(sorted(tr)); P = np.array([tr[f] for f in fs], float)
    dl = fit_phase(ctx, tr, t0); PHASE[rally] = dl

    rows = [r for r in csv.DictReader(open(f"{DV}/ball_path_r{rally}.csv")) if r["x"]]
    F = np.array([int(r["frame"]) for r in rows]); T = np.array([float(r["t_s"]) for r in rows])
    X = np.array([float(r["x"]) for r in rows]); Y = np.array([float(r["y"]) for r in rows])
    V = [r["vis"] for r in rows]
    spd = {}
    for i in range(len(rows)):
        d = n = 0
        for j in (i-1, i+1):
            if 0 <= j < len(rows) and 1 <= abs(F[j]-F[i]) <= 2:
                d += np.hypot(X[j]-X[i], Y[j]-Y[i])/abs(F[j]-F[i]); n += 1
        spd[i] = d/n if n else np.nan

    def hit_old(t, tx, ty):
        f = int(round((t-t0)*60))
        p = tr.get(f) or tr.get(f-1) or tr.get(f+1)
        return None if p is None else float(np.hypot(p[0]-tx, p[1]-ty))

    def hit_new(t, tx, ty):
        fq = (t-t0)*60 + dl
        i = np.searchsorted(fs, fq)
        if i == 0 or i >= len(fs) or fs[i]-fs[i-1] > 2:
            return hit_old(t, tx, ty)
        w = (fq-fs[i-1])/(fs[i]-fs[i-1]); px = P[i-1]*(1-w)+P[i]*w
        return float(np.hypot(px[0]-tx, px[1]-ty))

    cs = contacts(rally)
    bounds = [-1e9] + cs + [1e9]
    out = []
    for k in range(len(bounds)-1):
        a, b = bounds[k], bounds[k+1]
        idx = [i for i in range(len(rows)) if a <= T[i] < b and V[i] in ("V", "S")]
        if len(idx) < 3: continue
        ho = [hit_old(T[i], X[i], Y[i]) for i in idx]
        hn = [hit_new(T[i], X[i], Y[i]) for i in idx]
        out.append(dict(rally=rally, t=T[idx[0]], n=len(idx),
                        s=sum(V[i] == "S" for i in idx)/len(idx),
                        dur=T[idx[-1]]-T[idx[0]],
                        spd=float(np.nanmedian([spd[i] for i in idx])),
                        old=sum(1 for d in ho if d is not None and d <= 12)/len(idx),
                        new=sum(1 for d in hn if d is not None and d <= 12)/len(idx)))
    return out


ALL = []
for R in [int(a) for a in sys.argv[1:]]:
    o = run(R); ALL += o
    zo = [f for f in o if f["old"] == 0]; zn = [f for f in o if f["new"] == 0]
    print(f"r{R:2d} phase {PHASE[R]:+.2f}  flights {len(o)}  "
          f"zero-hit as-scored {len(zo)}  zero-hit corrected {len(zn)}")
    for f in o:
        mark = "  <-- ZERO" if f["new"] == 0 else ("  <-- zero(as-scored only)" if f["old"] == 0 else "")
        print(f"     t {f['t']:7.2f}  n {f['n']:3d}  S {f['s']:.0%}  dur {f['dur']:.2f}s  "
              f"spd {f['spd']:5.1f} px/f  hit {f['old']:.2f} -> {f['new']:.2f}{mark}")

print("\n=== what predicts a zero-hit flight (corrected scoring) ===")
z = [f for f in ALL if f["new"] == 0]; nz = [f for f in ALL if f["new"] > 0]
for key, lab in (("n", "clicks"), ("s", "S share"), ("dur", "duration s"), ("spd", "px/frame")):
    a = np.median([f[key] for f in z]) if z else float("nan")
    b = np.median([f[key] for f in nz])
    print(f"  {lab:12s} zero-hit {a:7.2f}   hit {b:7.2f}   (n {len(z)} vs {len(nz)})")
