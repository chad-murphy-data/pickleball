"""click_diag — what the owner's clicks are, measured without the tracker.

Written 2026-09-03 to answer the owner's question "should S clicks be
ignore-zones?  they ARE the ball, they're just less precise".  Three
label-side diagnostics, no model, no tuning, nothing here selects a path:

  precision   leave-one-out local-quadratic residual.  A ball in flight is
              quadratic in (x, y) vs frame, so fit the NEIGHBOURING clicks
              and predict the held-out one.  Windows straddling a contact
              are dropped (the arc is not one quadratic there).  This is
              immune to the tracker, the emission model and any timing
              offset, so it measures the hand and only the hand.

  appearance  per-click pixel speed from adjacent clicked frames, next to
              patch statistics under the click (yellow-pixel share,
              bright-blob elongation).  Tests the owner's speed-vs-colour
              hypothesis on the label side.  CAVEAT: over a 21x21 patch
              the ball is a handful of pixels and the court dominates, so
              the shape/saturation half of that hypothesis is NOT settled
              here -- segmenting ball from background is the detection
              problem itself.  Blur is also co-determined by depth and
              exposure (see CLAUDE.md, the confounded "fast mode").

  phase       one scalar per rally: the sub-frame offset between the click
              grid and the clip's frame grid, fit on V clicks ONLY by
              minimising median |track_interp(f+delta) - click|.  Every
              rally measured so far comes out NEGATIVE (-0.2 to -0.7
              frames at 60 fps): the recorded click times run a few ms
              late, consistent with the browser seeking to the frame at or
              before currentTime.  DIAGNOSTIC ONLY.  It is fit against the
              track, so it must never be folded into a scorer that then
              re-grades -- a shipped correction has to measure each clip's
              true cut offset against the full match video instead.

Usage:  python3 click_diag.py precision 2 3 4 6 7 17
        python3 click_diag.py appearance 3
        python3 click_diag.py phase 3 6 7 17
"""
import csv
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, "/home/user/pickleball/vision")

DV = Path("/home/user/pickleball/data/vision")
CON = DV / "contact_labels_chicago0725.csv"


def rows_for(rally):
    return [r for r in csv.DictReader(open(DV / f"ball_path_r{rally}.csv"))
            if r["x"]]


def contacts(rally):
    man, pre = [], []
    for r in csv.DictReader(open(CON)):
        if int(r["rally_cum"]) != rally or r.get("contact", "1") == "0":
            continue
        t = float(r["t_refined_s"] or r["t_tap_s"])
        (man if r["source"] in ("manual", "divergent") else pre).append(t)
    return sorted(man or pre)


def click_speed(F, X, Y):
    """px per 30 fps click frame, central difference over adjacent clicks."""
    out = np.full(len(F), np.nan)
    for i in range(len(F)):
        d = n = 0.0
        for j in (i - 1, i + 1):
            if 0 <= j < len(F) and 1 <= abs(F[j] - F[i]) <= 2:
                d += np.hypot(X[j] - X[i], Y[j] - Y[i]) / abs(F[j] - F[i])
                n += 1
        if n:
            out[i] = d / n
    return out


def precision(rallies):
    tot = {"V": [], "S": []}
    for R in rallies:
        rows = rows_for(R)
        F = np.array([int(r["frame"]) for r in rows])
        T = np.array([float(r["t_s"]) for r in rows])
        X = np.array([float(r["x"]) for r in rows])
        Y = np.array([float(r["y"]) for r in rows])
        V = [r["vis"] for r in rows]
        cs = contacts(R)
        res = {"V": [], "S": []}
        for i in range(len(rows)):
            if V[i] not in ("V", "S"):
                continue
            if any(T[i] - 4 / 30 < c < T[i] + 4 / 30 for c in cs):
                continue
            m = (np.abs(F - F[i]) <= 4) & (np.arange(len(F)) != i)
            if m.sum() < 4:
                continue
            A = np.vstack([(F[m] - F[i]) ** 2, F[m] - F[i], np.ones(m.sum())]).T
            cx = np.linalg.lstsq(A, X[m], rcond=None)[0]
            cy = np.linalg.lstsq(A, Y[m], rcond=None)[0]
            res[V[i]].append(float(np.hypot(cx[2] - X[i], cy[2] - Y[i])))
        line = f"r{R:2d}: "
        for k in ("V", "S"):
            a = np.array(res[k])
            tot[k] += list(a)
            if len(a):
                line += (f"{k} n={len(a):3d} med {np.median(a):4.1f} "
                         f"p75 {np.percentile(a, 75):4.1f}   ")
        print(line)
    print()
    for k in ("V", "S"):
        a = np.array(tot[k])
        print(f"POOLED {k}: n={len(a)}  median {np.median(a):.2f} px  "
              f"p75 {np.percentile(a, 75):.2f}  p90 {np.percentile(a, 90):.2f}  "
              f"frac<=6px {np.mean(a <= 6):.2f}  frac<=10px {np.mean(a <= 10):.2f}")


def appearance(rallies):
    import cv2
    import pathfirst as pf
    tot = {"V": [], "S": [], "I": []}
    for R in rallies:
        rows = [r for r in csv.DictReader(open(DV / f"ball_path_r{R}.csv"))
                if r["x"]]
        F = np.array([int(r["frame"]) for r in rows])
        T = np.array([float(r["t_s"]) for r in rows])
        X = np.array([float(r["x"]) for r in rows])
        Y = np.array([float(r["y"]) for r in rows])
        V = [r["vis"] for r in rows]
        sp = click_speed(F, X, Y)
        t0 = pf.context(R)["t0"]
        cap = cv2.VideoCapture(str(SP / f"r{R}_clip.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        want = {}
        for i in range(len(rows)):
            want.setdefault(int(round((T[i] - t0) * fps)), []).append(i)
        k = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            for i in want.get(k, ()):
                h, w = img.shape[:2]
                xi, yi = int(round(X[i])), int(round(Y[i]))
                p = img[max(0, yi - 10):min(h, yi + 11),
                        max(0, xi - 10):min(w, xi + 11)]
                if p.size == 0:
                    continue
                hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
                H, S, Vv = (hsv[..., 0].astype(float), hsv[..., 1].astype(float),
                            hsv[..., 2].astype(float))
                yellow = float(np.mean((H >= 20) & (H <= 45) & (S > 60)))
                m = Vv >= (Vv.max() + np.median(Vv)) / 2.0
                ys, xs = np.nonzero(m)
                if len(xs) >= 4:
                    c = np.cov(np.vstack([xs - xs.mean(), ys - ys.mean()]))
                    ev = np.sort(np.linalg.eigvalsh(c))[::-1]
                    el = float(np.sqrt(ev[0] / ev[1])) if ev[1] > 1e-6 else np.nan
                else:
                    el = np.nan
                if V[i] in tot:
                    tot[V[i]].append((sp[i], yellow, el))
            k += 1
        cap.release()
        print(f"rally {R} sampled")
    for k in ("V", "S", "I"):
        a = np.array([r for r in tot[k] if r[0] == r[0]])
        if not len(a):
            continue
        print(f"  {k}: n={len(a):4d}  speed {np.median(a[:, 0]):5.1f} px/f   "
              f"yellow {np.median(a[:, 1]):.3f}   elong {np.nanmedian(a[:, 2]):.2f}")


def phase(rallies):
    import json
    import pathfirst as pf
    import corridor_dp as cdp
    pc = json.loads(pf.TUNE_JSON.read_text())
    for R in rallies:
        ctx = pf.context(R)
        cdp.W_P_SOFT = 25.0
        res = pf.run(ctx, pc["p_seed"], pc["s_min"], int(pc["gap"]))
        tr, t0 = res["track"], ctx["t0"]
        fs = np.array(sorted(tr))
        P = np.array([tr[f] for f in fs], float)

        def med(dl, kinds):
            ds = []
            for t, tx, ty, v in ctx["truth"]:
                if v not in kinds:
                    continue
                fq = (t - t0) * 60 + dl
                i = np.searchsorted(fs, fq)
                if i == 0 or i >= len(fs) or fs[i] - fs[i - 1] > 2:
                    continue
                w = (fq - fs[i - 1]) / (fs[i] - fs[i - 1])
                px = P[i - 1] * (1 - w) + P[i] * w
                d = float(np.hypot(px[0] - tx, px[1] - ty))
                if d < 30:
                    ds.append(d)
            return np.median(ds) if len(ds) > 20 else np.nan

        g = np.arange(-2.0, 2.01, 0.05)
        best = float(g[int(np.nanargmin([med(d, ("V",)) for d in g]))])
        print(f"r{R:2d}  delta {best:+.2f} frames@60 ({best / 60 * 1000:+.1f} ms)   "
              f"median |err|  V {med(0.0, ('V',)):.2f}->{med(best, ('V',)):.2f}   "
              f"S {med(0.0, ('S',)):.2f}->{med(best, ('S',)):.2f}")


if __name__ == "__main__":
    mode, rs = sys.argv[1], [int(a) for a in sys.argv[2:]]
    {"precision": precision, "appearance": appearance, "phase": phase}[mode](rs)
