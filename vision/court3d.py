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
PLAYERS = DATA / "player_positions_r1.csv"
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


def load_players(path=PLAYERS):
    """{player: (t[], x[], y[])} from the committed positions CSV,
    or None if it does not exist."""
    if not Path(path).exists():
        return None
    from collections import defaultdict
    d = defaultdict(list)
    for r in csv.DictReader(open(path)):
        d[r["player"]].append((float(r["t_s"]), float(r["x_ft"]),
                               float(r["y_ft"])))
    return {k: tuple(np.array(c) for c in zip(*sorted(v)))
            for k, v in d.items()}


def player_at(players, name, t, tol=0.3):
    if not players or name not in players:
        return None
    ts, xs, ys = players[name]
    i = int(np.abs(ts - t).argmin())
    if abs(ts[i] - t) > tol:
        return None
    return np.array([xs[i], ys[i]])


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


def arc_vel(theta, tau):
    """Velocity at time tau (central difference on arc_pos)."""
    e = 1e-3
    p = arc_pos(theta, [tau - e, tau + e])
    return (p[1] - p[0]) / (2 * e)


BOUNCE_GRID_S = 1 / 15.0    # model-selection search spacing. 2026-09-01
                            # (fix #1 after the r10 grade): candidate
                            # bounce times are no longer ONLY detector
                            # events — the fit searches an interior
                            # grid, so a bounce the 2D stream never saw
                            # (occluded behind the near player, exactly
                            # where dink bounces live) is still found
                            # when the arcs demand it. Events remain
                            # extra candidates with fine refinement.
REST_PEN = 1.5              # restitution: a bounce dissipates energy —
                            # post-bounce speed must not exceed
                            # pre-bounce (user observation 2026-09-01),
                            # and the vertical velocity must flip up.
BOUNCE_VZ_GATES = False     # hard vz-in/vz-out gates: PREMISE
                            # FALSIFIED and gates OFF (2026-09-01).
                            # They were calibrated against the label
                            # V-shape instrument's read ("r7 all
                            # volleys, r10 human 13 vs 8 = phantoms")
                            # — then the OWNER eyeball-verified ALL
                            # FIVE disputed r10 bounces as REAL
                            # (short-hops at the kitchen line, one
                            # occluded): r10 truth = 13, the fitter
                            # was right, the V-shape margin just
                            # cannot see short-hops and serve/return
                            # bounces. The gates cut a real human
                            # bounce (13->12) and collapsed tracked
                            # recovery (8->3). Kept as a flag with
                            # this record so they are not re-armed
                            # against a lower-bound truth instrument.
BOUNCE_VZ_IN = -3.0         # ft/s (only if BOUNCE_VZ_GATES)
BOUNCE_VZ_OUT = 2.0         # ft/s (only if BOUNCE_VZ_GATES)
BOUNCE_MARGIN = 0.8         # px rms the split must win by (was 0.5;
                            # the interior grid multiplies candidate
                            # comparisons, so the acceptance margin
                            # carries the multiplicity burden)


# BOUNCE CORRIDOR (owner rule, 2026-09-04; measured in ballsearch/infront.py)
#
# "the ball always bounces in front of the person making contact" -- and it
# is stronger than that: over 57 human-solved bounces the bounce sits in a
# NARROW CORRIDOR along the line from the receiver to the previous hitter.
# In front 98% (56/57); lateral offset from that line median 1.37 ft, p95
# 3.28, p99 6.75; front/separation p99 1.00.
#
# The gate this replaces was `-2 <= x <= 22 and -1 <= y <= 45`, i.e. the
# whole court plus a margin, which rules out almost nothing.  Bounds below
# are the measured p99 plus margin, NOT chosen: they keep 56/57 = 98%, the
# single loss being r17's known hitter-attribution error.
#
# Only a bounce is constrained this way.  A CONTACT is at paddle height and
# this project maps to z=0, where height reads as depth -- contacts sit
# "behind the feet" 54% of the time and carry no such structure.
CORRIDOR_LAT = 8.0      # ft off the receiver->hitter line (p99 6.75)
CORRIDOR_BACK = -2.0    # ft behind the receiver still allowed (p01 -1.65)
CORRIDOR_PAST = 1.15    # multiples of the separation, past the receiver


def in_corridor(xy, corridor):
    """Is a candidate bounce point in the receiver->hitter corridor?"""
    if corridor is None:
        return True
    hit, rec = corridor
    if hit is None or rec is None:
        return True
    hit = np.asarray(hit, float)[:2]
    rec = np.asarray(rec, float)[:2]
    d = hit - rec
    n = float(np.hypot(*d))
    if n < 1e-6:
        return True
    u = d / n
    w = np.asarray(xy, float)[:2] - rec
    front = float(w @ u)
    lat = abs(float(u[0] * w[1] - u[1] * w[0]))
    return (lat <= CORRIDOR_LAT and front >= CORRIDOR_BACK
            and front <= CORRIDOR_PAST * n)


def fit_segment(P, obs, t0, t1, events, corridor=None):
    """One arc, or two arcs joined at a z=0 bounce; candidates = an
    interior search grid UNION refined detector events; pick by pixel
    RMS with a 0.5 px acceptance margin + floor/plausibility/
    restitution constraints. Returns dict.

    corridor: optional (hitter_xy, receiver_xy) court positions bounding
    this flight.  When given, a candidate bounce must also lie in the
    BOUNCE CORRIDOR (see CORRIDOR_LAT below) rather than merely on the
    court.  None keeps the old court-wide test."""
    single, rms1 = fit_best(P, obs, t0, default_inits())
    best = {"kind": "arc", "arcs": [(t0, t1, single)], "rms": rms1}
    cands = set()
    t = t0 + 0.12 + BOUNCE_GRID_S
    while t < t1 - 0.12:
        cands.add(round(t, 3))
        t += BOUNCE_GRID_S
    for ts in events:
        if t0 + 0.12 < ts < t1 - 0.12:
            for dt in np.arange(-0.167, 0.168, 1 / 30):
                t = round(ts + dt, 3)
                if t0 + 0.12 < t < t1 - 0.12:
                    cands.add(t)
    for ts in sorted(cands):
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
        v_in = arc_vel(a1, ts - t0)
        sp_in = float(np.linalg.norm(v_in))

        def extra2(th, xy=xy, sp_in=sp_in):
            sp_out = float(np.linalg.norm(th[3:6]))
            return pen * np.concatenate(
                [[th[2]], th[:2] - xy[:2],
                 [REST_PEN * max(0.0, sp_out - sp_in),
                  REST_PEN * max(0.0, -th[5])]])

        a2, r2 = fit_best(
            P, o2, ts,
            [np.concatenate([[xy[0], xy[1], 0.0],
                             a1[3:6] * [1, 1, -0.6], a1[6:7]])]
            + default_inits(),
            extra=extra2)
        rms = float(np.sqrt((r1**2 * len(o1) + r2**2 * len(o2))
                            / (len(o1) + len(o2))))
        plausible = (-2 <= xy[0] <= 22 and -1 <= xy[1] <= 45
                     and in_corridor(xy[:2], corridor))
        if BOUNCE_VZ_GATES:
            if float(v_in[2]) > BOUNCE_VZ_IN:
                continue      # not falling in — a volley, not a bounce
            if float(arc_vel(a2, 0.0)[2]) < BOUNCE_VZ_OUT:
                continue      # not rebounding — phantom split
        if plausible and rms < best["rms"] - BOUNCE_MARGIN:
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
        out_html=OUT_HTML, dump_show=False):
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
    players = load_players()
    print(f"player positions: "
          f"{'loaded (' + str(len(players)) + ' players)' if players else 'none'}")
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
            # PLAYER-GEOMETRY anchors (user prior): the contact happens
            # at the hitter's paddle; her FEET give exact depth. Hinge:
            # free within 3 ft reach, then 2.5 px/ft; z into paddle
            # range [0.3, 8.5].
            pa0 = player_at(players, hitters[k], t0)
            pa1 = (player_at(players, hitters[k + 1], t1)
                   if k + 1 < len(hitters) else None)

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
                        if pa0 is not None:
                            d = float(np.hypot(pS[0] - pa0[0],
                                               pS[1] - pa0[1]))
                            parts.append(2.5 * max(0.0, d - 3.0))
                            parts.append(2.0 * max(0.0, 0.3 - pS[2]))
                            parts.append(2.0 * max(0.0, pS[2] - 8.5))
                    if at_end:
                        pE = arc_pos(th, [dur])[0]
                        if s_n:
                            parts.append(3.0 * max(
                                0.0, 0.5 - s_n * (pE[1] - NET_Y)))
                        if c1 is not None:
                            parts += list(1.5 * (pE - c1))
                        if pa1 is not None:
                            d = float(np.hypot(pE[0] - pa1[0],
                                               pE[1] - pa1[1]))
                            parts.append(2.5 * max(0.0, d - 3.0))
                            parts.append(2.0 * max(0.0, 0.3 - pE[2]))
                            parts.append(2.0 * max(0.0, pE[2] - 8.5))
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

    n_cross = 0, 0
    n_ok = sum(1 for x in segs if x and x["ok"])
    n_x = 0
    for k, seg in enumerate(segs):
        if seg and seg["ok"]:
            ys = np.array([pnt[2] for pnt in sample_path(seg)])
            if ys.min() < NET_Y < ys.max():
                n_x += 1
    print(f"CHECK 4 — net-crossing spans: {n_x}/{n_ok} drawn segments "
          f"cross (labels say all must)")
    if viewer:
        path = []
        for seg in segs:
            if seg and seg["ok"]:
                path += sample_path(seg)
        write_viewer(path, impacts, out_html, players)
        print(f"\nwrote {out_html} — orbitable 3D rally")
    if dump_show:
        types = {}
        for r in csv.DictReader(open(DATA / "contact_labels_chicago0725.csv")):
            if r["rally_cum"] == "1" and r.get("contact", "1") == "1":
                tt = float(r["t_refined_s"] or r["t_tap_s"])
                types[min(impacts, key=lambda i: abs(i - tt))] = r["shot_type"]
        TEAMS = {"Allyce Jones": "UTA", "Etta Tuionetoa": "UTA",
                 "Emma Nelson": "CHI", "Ting Chieh Wei": "CHI"}
        show = {
            "meta": {"rally": 1, "event": "MLP Chicago — Group B",
                     "division": "Women's Doubles",
                     "teams": {"UTA": "Utah Black Diamonds",
                               "CHI": "Chicago Slice"},
                     "camera": [round(float(v), 2) for v in C]},
            "path": [], "players": {}, "impacts": [], "bounces": [],
            "crossings": [], "dead": dead,
        }
        for k, seg in enumerate(segs):
            if seg and seg["ok"]:
                show["path"] += [[round(v, 3) for v in p]
                                 for p in sample_path(seg)]
                for (tt, x, zz) in net_crossings(seg):
                    show["crossings"].append(
                        {"t": round(tt, 2), "x": round(x, 1),
                         "z": round(zz, 2)})
                if seg["kind"] == "bounce":
                    show["bounces"].append(
                        {"t": round(seg["ts"], 2),
                         "x": round(float(seg["bounce_xy"][0]), 1),
                         "y": round(float(seg["bounce_xy"][1]), 1)})
        if players:
            for name, (ts, xs, ys) in players.items():
                show["players"][name] = [
                    [round(float(ts[i]), 2), round(float(xs[i]), 2),
                     round(float(ys[i]), 2)]
                    for i in range(0, len(ts), 2)]
        for k, tt in enumerate(impacts):
            show["impacts"].append(
                {"t": round(tt, 3), "hitter": hitters[k],
                 "team": TEAMS.get(hitters[k], "?"),
                 "type": types.get(tt, "shot")})
        out = DATA / "rally1_show.json"
        out.write_text(json.dumps(show))
        print(f"wrote {out} ({out.stat().st_size//1024} KB show data)")
    return P, segs


# ------------------------------------------------------------- viewer


def write_viewer(path, impacts, out, players=None):
    # canvas orbit viewer; PATH = [[t,x,y,z]...], PLAYERS embedded at
    # 10 fps as {name: [[t,x,y]...]}
    pl = {}
    if players:
        for name, (ts, xs, ys) in players.items():
            pl[name.split()[-1]] = [
                [round(float(ts[i]), 2), round(float(xs[i]), 2),
                 round(float(ys[i]), 2)]
                for i in range(0, len(ts), 2)]
    html = _viewer_html(json.dumps([[round(v, 3) for v in p]
                                    for p in path]),
                        json.dumps([round(t, 3) for t in impacts]),
                        json.dumps(pl))
    Path(out).write_text(html)


def _viewer_html(path_json, impacts_json, players_json="{}"):
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
const PLAYERS = """ + players_json + r""";
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
  // players: interpolated floor position at tcur, stick + name
  const TEAM = {Jones:"#e05c5c",Tuionetoa:"#e05c5c",
                Nelson:"#5ca8e0",Wei:"#5ca8e0"};
  for(const nm in PLAYERS){
    const tr = PLAYERS[nm];
    let i = tr.findIndex(r=>r[0]>=tcur);
    if(i<0) i=tr.length-1; if(i===0) i=1;
    const a=tr[i-1], b=tr[i]||a;
    const f=(b[0]>a[0])?(tcur-a[0])/(b[0]-a[0]):0;
    const x=a[1]+(b[1]-a[1])*Math.max(0,Math.min(1,f));
    const y=a[2]+(b[2]-a[2])*Math.max(0,Math.min(1,f));
    const col=TEAM[nm]||"#999";
    line([x,y,0],[x,y,5.3],col,3);
    const h=proj(x,y,5.8); g.fillStyle=col;
    g.beginPath(); g.arc(h[0],h[1],4,0,7); g.fill();
    g.font="11px system-ui"; g.fillText(nm,h[0]+7,h[1]-4);
  }
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
    ap.add_argument("--dump-show", action="store_true",
                    help="write data/vision/rally1_show.json for the "
                         "PICKLES Replay artifact")
    ap.add_argument("--out", default=str(OUT_HTML))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    run(a.landmarks, a.ball, a.state, viewer=a.viewer, out_html=a.out,
        dump_show=a.dump_show)


if __name__ == "__main__":
    main()
