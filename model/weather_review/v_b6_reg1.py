"""VERIFIER 1 — reg-1 (H4) on corrected labels: point, cluster bootstrap with a
different seed, AND an analytic cluster-robust sandwich SE (independent route).
Also the binned favourite obs-pred drift that the -2.0pp premise rests on."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v_b6_frame import build, win_prob  # noqa: E402

G = [g for g in build() if g["wind"] is not None]


def reg1(rows):
    s = np.array([min(max(g["p"], 0.16), 0.84) - 0.5 for g in rows])
    w = np.array([g["wind"] / 10.0 for g in rows])
    y = np.array([g["share"] - 0.5 for g in rows])
    X = np.column_stack([np.ones(len(rows)), s, w, s * w])
    return X, y


def reg1_nocl(rows):
    """b6/favorites_wind use the UNCLAMPED skill; check both."""
    s = np.array([g["p"] - 0.5 for g in rows])
    w = np.array([g["wind"] / 10.0 for g in rows])
    y = np.array([g["share"] - 0.5 for g in rows])
    X = np.column_stack([np.ones(len(rows)), s, w, s * w])
    return X, y


def fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def cluster_sandwich(X, y, ev):
    b = fit(X, y)
    r = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    by = defaultdict(list)
    for i, e in enumerate(ev):
        by[e].append(i)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for idxs in by.values():
        ii = np.array(idxs)
        u = X[ii].T @ r[ii]
        meat += np.outer(u, u)
    V = XtXi @ meat @ XtXi
    G_ = len(by)
    V *= G_ / (G_ - 1.0)
    return b, np.sqrt(np.diag(V))


def boot(X, y, ev, nboot=1000, seed=987654):
    by = defaultdict(list)
    for i, e in enumerate(ev):
        by[e].append(i)
    keys = list(by)
    idx = [np.array(by[k]) for k in keys]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nboot):
        pick = rng.integers(0, len(keys), len(keys))
        ii = np.concatenate([idx[j] for j in pick])
        out.append(fit(X[ii], y[ii]))
    return np.array(out)


for labkey, labname in (("setting", "corrected"), ("heur", "heuristic")):
    for setting in ("outdoor", "indoor"):
        rows = [g for g in G if g[labkey] == setting]
        ev = [g["event"] for g in rows]
        for tag, mk in (("clamped", reg1), ("unclamped", reg1_nocl)):
            X, y = mk(rows)
            b, se = cluster_sandwich(X, y, ev)
            d = b[3]
            print(f"{labname:9s} {setting:7s} {tag:9s} n={len(rows):6d} "
                  f"b={b[1]:+.4f} d={d:+.4f} sandwich SE={se[3]:.4f} "
                  f"wald95=[{d-1.96*se[3]:+.4f},{d+1.96*se[3]:+.4f}]", end="")
            if tag == "unclamped":
                B = boot(X, y, ev, seed=987654)
                lo, hi = np.percentile(B[:, 3], [2.5, 97.5])
                print(f"  boot95(seed987654)=[{lo:+.4f},{hi:+.4f}]"
                      f" nclust={len(set(ev))}")
            else:
                print()

# ---- binned favourite obs-pred drift -------------------------------------
print("\n--- favourite obs-pred by wind bin (game level) ---")
for labkey, labname in (("heur", "heuristic"), ("setting", "corrected")):
    rows = [g for g in G if g[labkey] == "outdoor"]
    print(f"[{labname}] n={len(rows)}")
    for lo_, hi_ in ((0, 8), (8, 14), (14, 20), (20, 99)):
        sub = [g for g in rows if lo_ <= g["wind"] < hi_]
        if not sub:
            continue
        pf = []
        fw = []
        for g in sub:
            p1 = win_prob(g["p"], g["T"])
            pfav = max(p1, 1 - p1)
            favwon = (g["won"]) == (g["p"] >= 0.5)
            pf.append(pfav)
            fw.append(1.0 if favwon else 0.0)
        pf = np.array(pf)
        fw = np.array(fw)
        # cluster bootstrap SE over events
        evs = [g["event"] for g in sub]
        by = defaultdict(list)
        for i, e in enumerate(evs):
            by[e].append(i)
        keys = list(by)
        idx = [np.array(by[k]) for k in keys]
        rng = np.random.default_rng(4242)
        dr = []
        for _ in range(600):
            pick = rng.integers(0, len(keys), len(keys))
            ii = np.concatenate([idx[j] for j in pick])
            dr.append(fw[ii].mean() - pf[ii].mean())
        lo, hi = np.percentile(dr, [2.5, 97.5])
        print(f"  {lo_:2d}-{hi_:2d} n={len(sub):6d} meanwind={np.mean([g['wind'] for g in sub]):5.2f} "
              f"pred={pf.mean():.3f} obs={fw.mean():.3f} "
              f"edge={fw.mean()-pf.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")
