"""3D court — lift the labeled 2D ball path into court coordinates.

Chain (all inputs already on record, no video needed):
  1. CAMERA: DLT from the 11 clicked landmarks
     (data/vision/court_landmarks_chicago0725.csv — 8 court-plane
     intersections + 3 net verticals). The plane points alone are a
     homography; the post tops/tape are what make rays in the AIR
     resolvable. Reports reprojection residuals + recovered camera
     position as sanity.
  2. BALL: piecewise-ballistic fits of the oracle-passed 2D path
     (data/vision/ball_path_r1.csv) between the frame-exact impacts
     (state labels). Each inter-impact interval is fit as ONE arc
     (X(t) = p0 + v0 t + 0.5 g t^2, g = -32.174 ft/s^2) or TWO arcs
     joined at a z=0 BOUNCE (candidate bounce times = the oracle
     detector's interior direction-change events); whichever
     reprojects better. V clicks weight 1.0, S (smear) 0.5; I
     (inferred) excluded — never fit on inferred points (the
     seen/inferred rule).
  3. CHECKS (free ground truth, no new labels):
     - the serve segment must contain a bounce, landing in the
       correct service box (double-bounce rule);
     - every net crossing's height at y=22 vs the 34-36 in tape;
     - contact heights physically plausible and near the hitter;
     - reprojection RMS per segment.
  4. --viewer: self-contained orbitable HTML (canvas, no libraries) —
     court + net + the 3D ball path, animated.

Frame: court.py convention — x in [0,20] ft from the left sideline,
y=0 FAR baseline, y=44 NEAR, net y=22, z up, feet.

Usage:
    python3 vision/court3d.py                 # solve + fit + checks
    python3 vision/court3d.py --viewer       # also write court3d_r1.html
    python3 vision/court3d.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "vision"
LANDMARKS = DATA / "court_landmarks_chicago0725.csv"
BALL = DATA / "ball_path_r1.csv"
STATE = DATA / "state_labels_chicago0725.csv"
OUT_HTML = DATA / "court3d_r1.html"

G = -32.174          # ft/s^2
NET_Y = 22.0
TAPE_FT = 34 / 12.0  # center tape height
END_TRIM_S = 0.06    # drop clicks this close to an impact (paddle frame)
W_VIS = {"V": 1.0, "S": 0.5}


# ------------------------------------------------------------- camera


def dlt(X3, x2):
    """P (3x4) from n>=6 correspondences, Hartley-normalized."""
    X3, x2 = np.asarray(X3, float), np.asarray(x2, float)
    mx, sx = x2.mean(0), x2.std(0).mean()
    T = np.array([[1/sx, 0, -mx[0]/sx], [0, 1/sx, -mx[1]/sx], [0, 0, 1]])
    mX, sX = X3.mean(0), X3.std(0).mean()
    U = np.eye(4) / sX
    U[3, 3] = 1.0
    U[:3, 3] = -mX / sX
    A = []
    for (Xw, xw) in zip(X3, x2):
        Xh = U @ np.append(Xw, 1.0)
        xh = T @ np.append(xw, 1.0)
        A.append(np.concatenate([Xh, np.zeros(4), -xh[0] * Xh]))
        A.append(np.concatenate([np.zeros(4), Xh, -xh[1] * Xh]))
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Pn = Vt[-1].reshape(3, 4)
    P = np.linalg.inv(T) @ Pn @ U
    return P / np.linalg.norm(P[2, :3])


def project(P, X):
    X = np.atleast_2d(X)
    u = (P @ np.hstack([X, np.ones((len(X), 1))]).T).T
    return u[:, :2] / u[:, 2:3]


def camera_center(P):
    return -np.linalg.inv(P[:, :3]) @ P[:, 3]


def load_landmarks(path=LANDMARKS):
    X3, x2, names = [], [], []
    for r in csv.DictReader(open(path)):
        X3.append([float(r["X_ft"]), float(r["Y_ft"]), float(r["Z_ft"])])
        x2.append([float(r["px_x"]), float(r["px_y"])])
        names.append(r["name"])
    return np.array(X3), np.array(x2), names


# ---------------------------------------------------------- ball fits


def load_ball(path=BALL):
    out = []
    for r in csv.DictReader(open(path)):
        if r["vis"] in W_VIS and r["x"]:
            out.append((float(r["t_s"]), float(r["x"]), float(r["y"]),
                        W_VIS[r["vis"]]))
    return out


def load_impacts(path=STATE, rally=1):
    imps, dead = [], None
    for r in csv.DictReader(open(path)):
        if int(r["rally_cum"]) != rally:
            continue
        if r["kind"] == "impact":
            imps.append((float(r["t_s"]), r["player"]))
        elif r["kind"] == "point_dead":
            dead = float(r["t_s"])
    imps.sort()
    return [t for t, _ in imps], dead


def load_hitters(path=STATE, rally=1):
    out = []
    for r in csv.DictReader(open(path)):
        if int(r["rally_cum"]) == rally and r["kind"] == "impact":
            out.append((float(r["t_s"]), r["player"]))
    return [h for _, h in sorted(out)]


def seg_endpoints(seg, t0, t1):
    a0, _, th0 = seg["arcs"][0]
    aL, _, thL = seg["arcs"][-1]
    return (arc_pos(th0, [t0 - a0])[0], arc_pos(thL, [t1 - aL])[0])


def arc_pos(theta, tau):
    """theta = (p0[3], v0[3], [k]); positions at times tau (n,).
    With 7 params: linear-drag flight, a = g - k v  ->
    x(t) = p0 + vt t + (v0 - vt)(1 - e^{-kt})/k,  vt = g/k.
    A pickleball is a wiffle ball; drag is not optional for the floaty
    shots (pure parabolas misfit drops/dinks by ~8-10 px)."""
    p0, v0 = theta[:3], theta[3:6]
    tau = np.asarray(tau, float)[:, None]
    if len(theta) >= 7:
        k = min(max(abs(float(theta[6])), 1e-4), 3.0)   # physical range
        vt = np.array([0.0, 0.0, G / k])
        return p0 + vt * tau + (v0 - vt) * (-np.expm1(-k * tau)) / k
    acc = np.array([0.0, 0.0, 0.5 * G])
    return p0 + v0 * tau + acc * tau ** 2


def _residuals(theta, P, obs, t0, extra):
    tau = np.array([o[0] for o in obs]) - t0
    X = arc_pos(theta, tau)
    px = project(P, X)
    w = np.array([o[3] for o in obs])[:, None]
    res = ((px - np.array([[o[1], o[2]] for o in obs])) * w).ravel()
    if len(theta) >= 7:
        # mild zero-centered prior on drag: off unless the data insists
        # (kills the depth<->drag degeneracy on clean parabolic flights)
        res = np.concatenate([res, [2.0 * abs(float(theta[6]))]])
    # weak containment prior: the ball is being played on a court. A
    # depth-degenerate arc can match the pixels while racing away from
    # the camera; penalize excursions outside a generous court volume.
    ts = np.linspace(0, max(float(tau.max()), 1e-3), 5)
    Xs = arc_pos(theta, ts)
    lo = np.array([-10.0, -10.0, -2.0])
    hi = np.array([30.0, 54.0, 30.0])
    exc = np.maximum(lo - Xs, 0) + np.maximum(Xs - hi, 0)
    res = np.concatenate([res, 0.5 * exc.ravel()])
    return np.concatenate([res, extra(theta)]) if extra else res


def fit_arc(P, obs, t0, theta0=None, extra=None, iters=60):
    """Gauss-Newton, numeric Jacobian. obs = [(t, px, py, w)]."""
    th = np.array(theta0 if theta0 is not None
                  else [10.0, 22.0, 3.0, 0.0, 0.0, 5.0, 0.4], float)
    lam = 1e-3
    r = _residuals(th, P, obs, t0, extra)
    npar = len(th)
    for _ in range(iters):
        J = np.empty((len(r), npar))
        for j in range(npar):
            d = np.zeros(npar)
            d[j] = 1e-4
            J[:, j] = (_residuals(th + d, P, obs, t0, extra) - r) / 1e-4
        H = J.T @ J + lam * np.eye(npar)
        step = np.linalg.solve(H, -J.T @ r)
        r2 = _residuals(th + step, P, obs, t0, extra)
        if np.sum(r2**2) < np.sum(r**2):
            th, r, lam = th + step, r2, max(lam * 0.5, 1e-6)
            if np.linalg.norm(step) < 1e-8:
                break
        else:
            lam *= 4
            if lam > 1e6:
                break
    n_px = sum(1 for _ in obs) * 2
    rms = float(np.sqrt(np.mean(r[:n_px] ** 2)))
    return th, rms


def fit_best(P, obs, t0, inits, extra=None):
    """Multi-start GN — the drag term makes single-init GN fall into
    local minima on real segments; several cheap starts fix it."""
    best = None
    for th0 in inits:
        th, rms = fit_arc(P, obs, t0, theta0=th0, extra=extra)
        if best is None or rms < best[1]:
            best = (th, rms)
    return best


def default_inits():
    g = [10.0, 22.0, 3.0, 0.0, 0.0, 5.0]
    return [np.array(g + [0.3]), np.array(g + [1.0]),
            np.array([10.0, 22.0, 3.0, 0.0, -20.0, 8.0, 0.3]),
            np.array([10.0, 22.0, 3.0, 0.0, 20.0, 8.0, 0.3])]


def fit_segment(P, obs, t0, t1, events):
    """One arc, or two arcs joined at a z=0 bounce at an interior
    detector event; pick by pixel RMS. Returns dict."""
    single, rms1 = fit_best(P, obs, t0, default_inits())
    best = {"kind": "arc", "arcs": [(t0, t1, single)], "rms": rms1}
    # candidate bounce times: detector events, REFINED on a local grid
    # (the event time is quantized to click frames and can sit a few
    # frames off the true floor contact)
    cands = set()
    for ts in events:
        if t0 + 0.12 < ts < t1 - 0.12:
            for dt in np.arange(-0.167, 0.168, 1 / 30):
                t = round(ts + dt, 3)
                if t0 + 0.12 < t < t1 - 0.12:
                    cands.add(t)
    for ts in sorted(cands):
        if False:
            continue
        o1 = [o for o in obs if o[0] <= ts]
        o2 = [o for o in obs if o[0] >= ts]
        if len(o1) < 4 or len(o2) < 4:
            continue
        pen = 8.0  # px per ft of bounce-height violation
        a1, r1 = fit_best(P, o1, t0, [single] + default_inits(),
                          extra=lambda th: pen * np.array(
                              [arc_pos(th, [ts - t0])[0][2]]))
        xy = arc_pos(a1, [ts - t0])[0]
        if not (-5 <= xy[0] <= 25 and -3 <= xy[1] <= 47):
            continue          # diverged arc — no bounce claim from it
        a2, r2 = fit_best(
            P, o2, ts,
            [np.concatenate([[xy[0], xy[1], 0.0],
                             a1[3:6] * [1, 1, -0.6], a1[6:7]])]
            + default_inits(),
            extra=lambda th: pen * np.concatenate(
                [[th[2]], th[:2] - xy[:2]]))
        rms = float(np.sqrt((r1**2 * len(o1) + r2**2 * len(o2))
                            / (len(o1) + len(o2))))
        plausible = -2 <= xy[0] <= 22 and -1 <= xy[1] <= 45
        if plausible and rms < best["rms"] - 0.5:
            best = {"kind": "bounce", "ts": ts, "bounce_xy": xy[:2],
                    "arcs": [(t0, ts, a1), (ts, t1, a2)], "rms": rms}
    return best


def sample_path(seg, step=1 / 60):
    pts = []
    for (a, b, th) in seg["arcs"]:
        tt = np.arange(a, b, step)
        X = arc_pos(th, tt - a)
        pts += [[float(t), *map(float, x)] for t, x in zip(tt, X)]
    return pts


# ------------------------------------------------------------ checks


def net_crossings(seg):
    out = []
    for (a, b, th) in seg["arcs"]:
        tt = np.linspace(0, b - a, 200)
        X = arc_pos(th, tt)
        s = np.sign(X[:, 1] - NET_Y)
        for i in np.where(np.diff(s) != 0)[0]:
            f = (NET_Y - X[i, 1]) / (X[i + 1, 1] - X[i, 1] + 1e-12)
            z = X[i, 2] + f * (X[i + 1, 2] - X[i, 2])
            x = X[i, 0] + f * (X[i + 1, 0] - X[i, 0])
            out.append((float(a + tt[i]), float(x), float(z)))
    return out


def run(landmarks=LANDMARKS, ball=BALL, state=STATE, viewer=False,
        out_html=OUT_HTML):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from make_ball_audit import detect_events

    X3, x2, names = load_landmarks(landmarks)
    P = dlt(X3, x2)
    reproj = np.linalg.norm(project(P, X3) - x2, axis=1)
    C = camera_center(P)
    print(f"CAMERA: reprojection median {np.median(reproj):.2f}px, "
          f"max {reproj.max():.2f}px ({names[int(reproj.argmax())]})")
    print(f"  camera center ({C[0]:.1f}, {C[1]:.1f}, {C[2]:.1f}) ft "
          f"— expect high, behind a baseline")

    obs_all = load_ball(ball)
    impacts, dead = load_impacts(state)
    hitters = load_hitters(state)
    events = detect_events([(t, x, y) for t, x, y, w in obs_all
                            if w == 1.0])
    bounds = impacts + ([dead] if dead else [])
    segs, contact_z = [], []
    seg_obs = {}
    for k in range(len(bounds) - 1):
        t0, t1 = bounds[k], bounds[k + 1]
        obs = [o for o in obs_all
               if t0 + END_TRIM_S <= o[0] <= t1 - END_TRIM_S]
        if len(obs) < 5:
            segs.append(None)
            continue
        seg_obs[k] = obs
        seg = fit_segment(P, obs, t0, t1, events)
        # plausibility gate: sampled path must stay near the court and
        # under a sane speed, else the segment is DEGENERATE — reported
        # but never drawn or measured
        pts = np.array([p[1:] for p in sample_path(seg)])
        v = np.diff(pts, axis=0) * 60.0
        ok = (seg["rms"] < 8.0
              and pts[:, 0].min() > -15 and pts[:, 0].max() < 35
              and pts[:, 1].min() > -15 and pts[:, 1].max() < 60
              and pts[:, 2].min() > -3 and pts[:, 2].max() < 40
              and (np.linalg.norm(v, axis=1).max() < 176 if len(v)
                   else True))
        seg["ok"] = bool(ok)
        segs.append(seg)


    # ---- pass 2 (user priors, 2026-08-30): every shot CROSSES THE
    # NET (hitter sides voted from pass-1 good fits), and arc k's end
    # = arc k+1's start = the same paddle (consensus contact anchors).
    # Two refit sweeps; the anchors are what fix contact heights and
    # rescue depth-degenerate arcs.
    votes = {}
    for k, seg in enumerate(segs):
        if seg and seg["ok"]:
            y0 = seg_endpoints(seg, bounds[k], bounds[k + 1])[0][1]
            votes.setdefault(hitters[k], []).append(
                1.0 if y0 > NET_Y else -1.0)
    side = {h: (1.0 if sum(v) > 0 else -1.0) for h, v in votes.items()}
    for _sweep in range(2):
        cons = [None] * len(bounds)
        for k, seg in enumerate(segs):
            if seg and seg["ok"]:
                p0, p1 = seg_endpoints(seg, bounds[k], bounds[k + 1])
                for idx, pt in ((k, p0), (k + 1, p1)):
                    cons[idx] = (pt if cons[idx] is None
                                 else (cons[idx] + pt) / 2)
        for k, seg in enumerate(segs):
            if seg is None:
                continue
            t0, t1 = bounds[k], bounds[k + 1]
            s_h = side.get(hitters[k])
            s_n = side.get(hitters[k + 1]) if k + 1 < len(hitters) else None
            c0 = cons[k] if cons[k] is not None else None
            c1 = cons[k + 1] if k + 1 < len(cons) else None

            def make_extra(base, at_start, at_end, dur):
                def extra(th):
                    parts = list(base(th)) if base else []
                    if at_start:
                        pS = arc_pos(th, [0.0])[0]
                        if s_h:
                            parts.append(3.0 * max(
                                0.0, 0.5 - s_h * (pS[1] - NET_Y)))
                        if c0 is not None:
                            parts += list(1.5 * (pS - c0))
                    if at_end:
                        pE = arc_pos(th, [dur])[0]
                        if s_n:
                            parts.append(3.0 * max(
                                0.0, 0.5 - s_n * (pE[1] - NET_Y)))
                        if c1 is not None:
                            parts += list(1.5 * (pE - c1))
                    return np.asarray(parts, float)
                return extra

            obs = seg_obs[k]
            if seg["kind"] == "arc":
                th, rms = fit_best(
                    P, obs, t0,
                    [seg["arcs"][0][2], default_inits()[0]],
                    extra=make_extra(None, True, True, t1 - t0))
                seg2 = {"kind": "arc", "arcs": [(t0, t1, th)],
                        "rms": rms}
            else:
                ts = seg["ts"]
                o1 = [o for o in obs if o[0] <= ts]
                o2 = [o for o in obs if o[0] >= ts]
                pen = 8.0
                a1, r1 = fit_best(
                    P, o1, t0, [seg["arcs"][0][2], default_inits()[0]],
                    extra=make_extra(
                        lambda th: pen * np.array(
                            [arc_pos(th, [ts - t0])[0][2]]),
                        True, False, ts - t0))
                xy = arc_pos(a1, [ts - t0])[0]
                a2, r2 = fit_best(
                    P, o2, ts, [seg["arcs"][-1][2], default_inits()[0]],
                    extra=make_extra(
                        lambda th: pen * np.concatenate(
                            [[th[2]], th[:2] - xy[:2]]),
                        False, True, t1 - ts))
                rms = float(np.sqrt(
                    (r1**2 * len(o1) + r2**2 * len(o2))
                    / (len(o1) + len(o2))))
                seg2 = {"kind": "bounce", "ts": ts,
                        "bounce_xy": arc_pos(a1, [ts - t0])[0][:2],
                        "arcs": [(t0, ts, a1), (ts, t1, a2)],
                        "rms": rms}
            pts = np.array([p[1:] for p in sample_path(seg2)])
            v = np.diff(pts, axis=0) * 60.0
            seg2["ok"] = bool(
                seg2["rms"] < 8.0
                and pts[:, 0].min() > -15 and pts[:, 0].max() < 35
                and pts[:, 1].min() > -15 and pts[:, 1].max() < 60
                and pts[:, 2].min() > -3 and pts[:, 2].max() < 40
                and (np.linalg.norm(v, axis=1).max() < 176
                     if len(v) else True))
            if seg2["ok"] or not seg["ok"]:
                segs[k] = seg2
    # consensus contact heights from the anchored fits
    cons = [None] * len(bounds)
    for k, seg in enumerate(segs):
        if seg and seg["ok"]:
            p0, p1 = seg_endpoints(seg, bounds[k], bounds[k + 1])
            for idx, pt in ((k, p0), (k + 1, p1)):
                cons[idx] = (pt if cons[idx] is None
                             else (cons[idx] + pt) / 2)
    contact_z = [(bounds[k], float(c[2]))
                 for k, c in enumerate(cons[:len(impacts)])
                 if c is not None]
    sides_str = ", ".join(f"{h.split()[-1]}:{'near' if v>0 else 'far'}"
                          for h, v in side.items())
    print(f"\nsides (voted from pass 1): {sides_str}")

    print("\nSEGMENTS (impact -> next):")
    for k, seg in enumerate(segs):
        t0 = bounds[k]
        if seg is None:
            print(f"  {t0:6.2f}  (too few visible clicks — skipped)")
            continue
        b = (f" BOUNCE@({seg['bounce_xy'][0]:.1f},"
             f"{seg['bounce_xy'][1]:.1f})" if seg["kind"] == "bounce"
             else "")
        d = "" if seg["ok"] else "  DEGENERATE (not drawn)"
        print(f"  {t0:6.2f}  rms {seg['rms']:5.1f}px  {seg['kind']}{b}{d}")

    print("\nCHECK 1 — serve segment (double-bounce rule):")
    s0 = segs[0]
    if s0 and s0["kind"] == "bounce":
        x, y = s0["bounce_xy"]
        ok = 0 <= x <= 20 and (0 <= y <= 15 or 29 <= y <= 44)
        print(f"  bounce at ({x:.1f}, {y:.1f}) ft — "
              f"{'INSIDE a service-box region' if ok else 'OUT (!)'} ")
    else:
        print("  no bounce found in serve segment (!)")

    print("CHECK 2 — net crossings vs the 34in tape:")
    lo_cross = 0
    for k, seg in enumerate(segs):
        if seg is None or not seg["ok"]:
            continue
        for (t, x, z) in net_crossings(seg):
            flag = "" if z > TAPE_FT - 0.3 else "  LOW (!)"
            if z <= TAPE_FT - 0.3:
                lo_cross += 1
            print(f"  t={t:6.2f}  clearance z={z:4.1f} ft at x={x:4.1f}"
                  f"{flag}")
    print("CHECK 3 — contact heights:")
    zs = [z for _, z in contact_z]
    n_bad = sum(1 for z in zs if not (0.0 <= z <= 9.0))
    print(f"  {len(zs)} measurable contacts, z range "
          f"[{min(zs):.1f}, {max(zs):.1f}] ft, "
          f"median {np.median(zs):.1f} ft, implausible: {n_bad}")

    if viewer:
        path = []
        for seg in segs:
            if seg and seg["ok"]:
                path += sample_path(seg)
        write_viewer(path, impacts, out_html)
        print(f"\nwrote {out_html} — orbitable 3D rally")
    return P, segs


# ------------------------------------------------------------- viewer


def write_viewer(path, impacts, out):
    # canvas orbit viewer written as a template below (kept out of the
    # docstring for size); PATH = [[t,x,y,z]...]
    html = _viewer_html(json.dumps([[round(v, 3) for v in p]
                                    for p in path]),
                        json.dumps([round(t, 3) for t in impacts]))
    Path(out).write_text(html)


def _viewer_html(path_json, impacts_json):
    return r"""<!doctype html><html><head><meta charset="utf-8">
<title>rally 1 in 3D</title>
<style>body{margin:0;background:#0c0c10;color:#ddd;font:13px system-ui;overflow:hidden}
#hud{position:fixed;left:10px;top:8px}input[type=range]{width:340px}</style>
</head><body><canvas id="c"></canvas>
<div id="hud"><b>rally 1 — 3D</b> · drag orbits · wheel zooms · space plays<br>
<input id="tt" type="range" min="0" max="1000" value="0"> <span id="tl"></span></div>
<script>
const PATH = """ + path_json + r""";
const IMPACTS = """ + impacts_json + r""";
const cv = document.getElementById("c"), g = cv.getContext("2d");
const tt = document.getElementById("tt"), tl = document.getElementById("tl");
let az = -2.4, el = 0.5, zoom = 13, playing = false;
const T0 = PATH[0][0], T1 = PATH[PATH.length-1][0];
let tcur = T0;
function proj(x, y, z){
  const ca=Math.cos(az), sa=Math.sin(az), ce=Math.cos(el), se=Math.sin(el);
  const X=(x-10)*ca-(y-22)*sa, Y=(x-10)*sa+(y-22)*ca;
  const u=X, v=-z*ce - Y*se, depth=Y*ce - z*se;
  return [cv.width/2 + u*zoom, cv.height/2 + v*zoom + depth*0.0];
}
function line(a, b, col, w){
  const p=proj(...a), q=proj(...b);
  g.strokeStyle=col; g.lineWidth=w||1;
  g.beginPath(); g.moveTo(...p); g.lineTo(...q); g.stroke();
}
function draw(){
  cv.width=innerWidth; cv.height=innerHeight;
  g.fillStyle="#0c0c10"; g.fillRect(0,0,cv.width,cv.height);
  const L="#3b6ea5";
  // court lines (court.py's ten segments)
  const S=[[[0,0],[20,0]],[[0,44],[20,44]],[[0,0],[0,44]],[[20,0],[20,44]],
           [[0,15],[20,15]],[[0,29],[20,29]],[[10,0],[10,15]],[[10,29],[10,44]]];
  S.forEach(s=>line([s[0][0],s[0][1],0],[s[1][0],s[1][1],0],L,1.5));
  // net
  for(let x=-1;x<=21;x+=1){const h=2.833+0.167*Math.abs(x-10)/11;
    line([x,22,0],[x,22,h],"#777",1);}
  line([-1,22,3],[21,22,2.84],"#ccc",2); line([21,22,2.84],[21,22,3],"#ccc",2);
  // ball path up to tcur (trail) + full faint
  g.globalAlpha=0.25;
  for(let i=1;i<PATH.length;i++)
    line(PATH[i-1].slice(1),PATH[i].slice(1),"#e8c44a",1);
  g.globalAlpha=1;
  let last=null;
  for(let i=1;i<PATH.length && PATH[i][0]<=tcur;i++){
    line(PATH[i-1].slice(1),PATH[i].slice(1),"#ffd94a",2); last=PATH[i];}
  if(last){const p=proj(last[1],last[2],last[3]);
    g.fillStyle="#ffd94a"; g.beginPath();
    g.arc(p[0],p[1],5,0,7); g.fill();
    // ground shadow
    const s=proj(last[1],last[2],0); g.fillStyle="#0008";
    g.beginPath(); g.ellipse(s[0],s[1],5,2,0,0,7); g.fill();}
  tl.textContent=tcur.toFixed(2)+"s";
}
function step(){
  if(playing){ tcur += (T1-T0)/600; if(tcur>T1) tcur=T0;
    tt.value = 1000*(tcur-T0)/(T1-T0); }
  draw(); requestAnimationFrame(step);
}
let drag=null;
cv.onmousedown=e=>drag=[e.clientX,e.clientY];
window.onmouseup=()=>drag=null;
window.onmousemove=e=>{ if(drag){ az+=(e.clientX-drag[0])*0.008;
  el=Math.max(0.05,Math.min(1.5,el+(e.clientY-drag[1])*0.005));
  drag=[e.clientX,e.clientY]; } };
cv.onwheel=e=>{ zoom=Math.max(4,Math.min(40,zoom*(e.deltaY<0?1.1:0.9)));
  e.preventDefault(); };
tt.oninput=()=>{ tcur=T0+(T1-T0)*tt.value/1000; };
document.onkeydown=e=>{ if(e.key===" "){playing=!playing;e.preventDefault();} };
step();
</script></body></html>"""


# ----------------------------------------------------------- selftest


def selftest():
    rng = np.random.default_rng(3)
    # synthetic camera behind the near baseline, high
    C = np.array([10.0, 75.0, 22.0])
    look = np.array([10.0, 20.0, 2.0]) - C
    zc = look / np.linalg.norm(look)
    xc = np.cross(zc, [0, 0, 1.0]); xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    Rm = np.stack([xc, yc, zc])
    K = np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]])
    Ptrue = K @ np.hstack([Rm, (-Rm @ C)[:, None]])
    lms = [(0,0,0),(20,0,0),(0,44,0),(20,44,0),(0,15,0),(20,15,0),
           (0,29,0),(20,29,0),(-1,22,3),(21,22,3),(10,22,2.833)]
    X3 = np.array(lms, float)
    x2 = project(Ptrue, X3) + rng.normal(0, 0.5, (len(lms), 2))
    P = dlt(X3, x2)
    err = np.linalg.norm(project(P, X3) - x2, axis=1)
    assert np.median(err) < 1.5, err
    Chat = camera_center(P)
    assert np.linalg.norm(Chat - C) < 3.0, Chat
    # ballistic arc recovery
    th_true = np.array([4.0, 40.0, 2.5, 3.0, -30.0, 12.0])
    tt = np.arange(0, 0.8, 1/30)
    obs = [(float(t), *project(P, arc_pos(th_true, [t]))[0]
            + rng.normal(0, 1.0, 2), 1.0) for t in tt]
    th, rms = fit_arc(P, obs, 0.0)
    X_true = arc_pos(th_true, tt)
    X_fit = arc_pos(th, tt)
    err3d = np.linalg.norm(X_true - X_fit, axis=1)
    # honest monocular precision: ~1px click noise at ~55ft range plus
    # the depth<->drag degeneracy gives ft-scale error at arc ENDS; the
    # interior (where bounces and net crossings live) is what placement
    # uses, so that is what gets the tight bound
    mid = err3d[len(err3d)//3 : 2*len(err3d)//3]
    # measured information content of this geometry (900px focal,
    # camera ~60ft out, 1px noise): ~1.3ft median for the PURE
    # ballistic fit too — this is the medium, not the drag term
    assert np.median(err3d) < 1.5, np.median(err3d)
    assert mid.max() < 2.0, mid.max()
    assert err3d.max() < 3.5, err3d.max()
    assert rms < 2.5, rms
    # bounce segment: down, bounce at z=0, up — split must beat single
    thA = np.array([6.0, 38.0, 3.0, 2.0, -20.0, -8.0])
    tb = None
    for t in np.arange(0.01, 1.0, 0.001):
        if arc_pos(thA, [t])[0][2] <= 0:
            tb = t; break
    pb = arc_pos(thA, [tb])[0]
    thB = np.array([pb[0], pb[1], 0.0, 2.0, -20.0, 9.0])
    obs = []
    for t in np.arange(0, tb, 1/30):
        obs.append((float(t), *project(P, arc_pos(thA, [t]))[0]
                    + rng.normal(0, 1, 2), 1.0))
    for t in np.arange(tb, tb + 0.6, 1/30):
        obs.append((float(t), *project(P, arc_pos(thB, [t - tb]))[0]
                    + rng.normal(0, 1, 2), 1.0))
    seg = fit_segment(P, obs, 0.0, tb + 0.6, [tb])
    assert seg["kind"] == "bounce", seg["kind"]
    assert abs(seg["bounce_xy"][0] - pb[0]) < 0.8
    assert abs(seg["bounce_xy"][1] - pb[1]) < 0.8
    cr = net_crossings(seg)
    print(f"selftest OK — camera {np.linalg.norm(Chat-C):.1f}ft off, "
          f"arc err median {np.median(err3d):.2f}/max {err3d.max():.2f}ft, "
          f"bounce recovered at ({seg['bounce_xy'][0]:.1f},"
          f"{seg['bounce_xy'][1]:.1f}) vs ({pb[0]:.1f},{pb[1]:.1f}), "
          f"{len(cr)} net crossings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", default=str(LANDMARKS))
    ap.add_argument("--ball", default=str(BALL))
    ap.add_argument("--state", default=str(STATE))
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--out", default=str(OUT_HTML))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    run(a.landmarks, a.ball, a.state, viewer=a.viewer, out_html=a.out)


if __name__ == "__main__":
    main()
