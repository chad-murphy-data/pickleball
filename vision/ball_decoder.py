"""Ball decoder — gated component 2 of ball_gate.md: candidates -> path.

Turns ~40 motion candidates/frame (ball_candidates.py; mostly junk by
design) into ONE physical ball path with self-discovered turns.

Method: Viterbi over CANDIDATE-PAIR states (an edge a->b carries a
velocity, so transitions can price acceleration):
- edges link candidates <= GAP_MAX frames apart within a speed gate;
- transition cost = capped ||dv|| (capping is what lets the path TURN
  at real contacts without being torn apart by the smoothness prior)
  + per-skipped-frame penalty - per-covered-frame coverage bonus;
- best terminal edge backtracked -> visited points; gaps interpolated
  linearly (ballistic refinement is the 3D stage's job).

Turns are discovered by the oracle battery's own detect_events on the
decoded path — the same frozen instrument that scores them, run on
visited (non-interpolated) points only.

TRAIN-side scoring (rallies 6-7, iteration unrestricted): checks 1-2
of the gate — V/S hit rate vs the user's ball pass, and the
human-matched turns comparison per Amendment 1. Rally 8 stays sealed:
same --graded-run guard as the candidate stage.

Usage:
    python3 vision/ball_decoder.py --rally 6            # decode + score
    python3 vision/ball_decoder.py --rally 6 --dump PATH.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_ball_audit import detect_events, score_events, load_impacts  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "vision"
SEALED = {8}

FPS = 60.0
GAP_MAX = 21            # frames (0.35 s) — the battery's own blind-gap limit
VMAX = 3200.0           # px/s hard speed gate
SKIP_PEN = 6.0          # per skipped frame
ACCEL_SCALE = 60.0      # px/s of |dv| per unit cost
TURN_CAP = 40.0         # max transition cost — a real contact costs this
COVER_BONUS = 14.0      # per covered frame
SUCC_MAX = 8            # successors kept per node per gap step
SLOW_FLOOR = 180.0      # px/s under which an edge is junk-suspicious
SLOW_PEN = 30.0         # max slow-edge cost (scorebug/crowd/lazy limbs
                        # run <150 px/s; the ball's median is ~330)
BODY_PEN = 0.0          # MEASURED HARMFUL 2026-08-31 and disabled: the
                        # true ball sits inside a whole-body box on 36-41%
                        # of its visible frames (near players are huge), so
                        # the tax hits the ball as hard as the limbs. Kept
                        # as a knob for a future wrist-radius variant only.
BODY_MARGIN = 12.0      # px expansion of keypoint-extent boxes
KPT_CONF = 0.3
COURT_PEN = 35.0        # per endpoint OUTSIDE the projected court volume
COURT_MARGIN = 45.0     # px slack around the projected hull
LOB_FT = 16.0           # court volume height for the hull
ANCHOR_R = 60.0         # px radius of a hitter-chain anchor's influence
ANCHOR_DT = 0.12        # s time window of an anchor
ANCHOR_BONUS = 2.0      # edge-cost discount near an anchor. MEASURED
                        # 2026-08-31: at 10.0 bad anchors GLUE the path to
                        # wrists (r6 V 60%->7%) while good ones help (r7
                        # 65->67); anchors carry TIME information, so the
                        # turn waiver does the work and position pull stays
                        # nearly nil — the over-steer lesson, in numbers
ANCHOR_TURN_FACTOR = 0.3  # turning near a predicted contact is cheap


def court_hull():
    """Convex hull (pixels) of the court footprint plus the footprint
    lifted to LOB_FT — the region a real ball can occupy. Uses the
    one-time court calibration (licensed gate input)."""
    from court3d import load_landmarks, dlt, project
    X3, x2, _ = load_landmarks()
    P = dlt(X3, x2)
    corners = []
    for z in (0.0, LOB_FT):
        for cx, cy in ((0, 0), (20, 0), (20, 44), (0, 44)):
            corners.append([cx, cy, z])
    pts = project(P, np.array(corners, dtype=float)).astype(np.float32)
    import cv2
    return cv2.convexHull(pts.reshape(-1, 1, 2))


def out_of_court_flags(byf, hull):
    import cv2
    flags = {}
    for f, cands in byf.items():
        flags[f] = [cv2.pointPolygonTest(hull, (float(x), float(y)), True)
                    < -COURT_MARGIN for x, y in cands]
    return flags


def body_boxes(pose_npz):
    """frame-time -> list of expanded person boxes from ALL tracks
    (automated: no track filtering, no identities — junk tracks included,
    they only ever add caution)."""
    z = np.load(pose_npz)
    t, kpt, kpc = z["t"], z["kpt"], z["kpc"]
    out = defaultdict(list)
    for i in range(len(t)):
        conf = kpc[i] >= KPT_CONF
        if conf.sum() < 3:
            continue
        xs, ys = kpt[i][conf, 0], kpt[i][conf, 1]
        b = (xs.min() - BODY_MARGIN, ys.min() - BODY_MARGIN,
             xs.max() + BODY_MARGIN, ys.max() + BODY_MARGIN)
        if (b[2] - b[0]) * (b[3] - b[1]) > 0.20 * 1280 * 720:
            continue        # scattered-keypoint junk row (state-audit lesson)
        out[round(float(t[i]) * FPS)].append(b)
    return out


def in_body_flags(byf, t0, boxes):
    flags = {}
    for f, cands in byf.items():
        bl = boxes.get(round((t0 + f / FPS) * FPS), [])
        flags[f] = [any(x0 <= x <= x1 and y0 <= y <= y1
                        for x0, y0, x1, y1 in bl) for x, y in cands]
    return flags


def load_candidates(rally):
    byf = defaultdict(list)
    t0 = None
    for r in csv.DictReader(gzip.open(DATA / f"ball_candidates_r{rally}.csv.gz", "rt")):
        f = int(r["frame"])
        byf[f].append((float(r["x"]), float(r["y"])))
        if t0 is None:
            t0 = float(r["t_s"]) - f / FPS
    return byf, t0


def anchor_flags(byf, t0, anchors):
    """Per-candidate: near a hitter-chain predicted contact in both
    space and time (the recursive coupling, priced not forced)."""
    out = {}
    for f, cands in byf.items():
        t = t0 + f / FPS
        near = [(x, y) for ta, xa, ya in anchors
                if abs(t - ta) <= ANCHOR_DT for x, y in [(xa, ya)]]
        out[f] = [any(math.hypot(x - xa, y - ya) <= ANCHOR_R
                      for xa, ya in near) for x, y in cands]
    return out


def decode(byf, flags=None, oflags=None, aflags=None):
    frames = sorted(byf)
    nodes = [(f, i) for f in frames for i in range(len(byf[f]))]
    nid = {n: k for k, n in enumerate(nodes)}
    pos = np.array([byf[f][i] for f, i in nodes])
    nframe = np.array([f for f, _ in nodes])
    nbody = np.array([bool(flags[f][i]) if flags else False
                      for f, i in nodes])
    ncourt = np.array([bool(oflags[f][i]) if oflags else False
                       for f, i in nodes])
    nanchor = np.array([bool(aflags[f][i]) if aflags else False
                        for f, i in nodes])

    # edges: a -> b within gap and speed gates
    by_frame_ids = defaultdict(list)
    for k, (f, _) in enumerate(nodes):
        by_frame_ids[f].append(k)
    edges = []           # (a, b, d)
    e_in = defaultdict(list)
    for a in range(len(nodes)):
        fa = nframe[a]
        got_close = False
        for d in range(1, GAP_MAX + 1):
            if got_close and d > 3:
                break
            cand = by_frame_ids.get(fa + d, ())
            if not cand:
                continue
            dx = pos[list(cand)] - pos[a]
            dist = np.hypot(dx[:, 0], dx[:, 1])
            vmax_d = VMAX * d / FPS
            order = np.argsort(dist)[:SUCC_MAX]
            for j in order:
                if dist[j] > vmax_d:
                    continue
                b = cand[j]
                ei = len(edges)
                edges.append((a, b, d))
                e_in[b].append(ei)
                if d <= 3:
                    got_close = True
    if not edges:
        return []

    # Viterbi over edges
    E = len(edges)
    ea = np.array([e[0] for e in edges])
    eb = np.array([e[1] for e in edges])
    ed = np.array([e[2] for e in edges])
    vel = (pos[eb] - pos[ea]) / (ed / FPS)[:, None]
    speed = np.hypot(vel[:, 0], vel[:, 1])
    slow = SLOW_PEN * np.clip(1.0 - speed / SLOW_FLOOR, 0.0, 1.0)
    body = BODY_PEN * (nbody[ea].astype(float) + nbody[eb]) / 2.0
    court = COURT_PEN * (ncourt[ea].astype(float) + ncourt[eb]) / 2.0
    anch = -ANCHOR_BONUS * (nanchor[ea].astype(float) + nanchor[eb]) / 2.0
    base = SKIP_PEN * (ed - 1) - COVER_BONUS * ed + slow + body + court + anch
    score = base.astype(float).copy()          # start-anywhere
    prev = np.full(E, -1)
    # process edges grouped by arrival frame so predecessors are final
    edge_ids_by_arrival = defaultdict(list)
    for ei in range(E):
        edge_ids_by_arrival[nframe[eb[ei]]].append(ei)
    best_in = {}                                # node -> (score, edge)
    for f in sorted(edge_ids_by_arrival):
        for ei in edge_ids_by_arrival[f]:
            a = ea[ei]
            if a in best_in:
                s0, pe = best_in[a]
                dv = vel[ei] - vel[pe]
                tc = min(math.hypot(dv[0], dv[1]) / ACCEL_SCALE, TURN_CAP)
                if nanchor[a]:
                    tc *= ANCHOR_TURN_FACTOR
                cand_score = s0 + base[ei] + tc
                if cand_score < score[ei]:
                    score[ei] = cand_score
                    prev[ei] = pe
            b = eb[ei]
            if b not in best_in or score[ei] < best_in[b][0]:
                best_in[b] = (score[ei], ei)
    ei = int(np.argmin(score))
    path_nodes = []
    while ei >= 0:
        path_nodes.append(int(eb[ei]))
        if prev[ei] < 0:
            path_nodes.append(int(ea[ei]))
            break
        ei = int(prev[ei])
    path_nodes.reverse()
    return [(int(nframe[k]), float(pos[k, 0]), float(pos[k, 1]))
            for k in path_nodes]


def refine_arcs(visited, t0):
    """Stage 2: piecewise-quadratic arc refit between the path's own
    discovered turns (trimmed least squares per segment). Returns a
    30 fps (t, x, y) path — same sampling as the human labels, so the
    frozen battery sees comparable inputs."""
    pts = [(t0 + f / FPS, x, y) for f, x, y in visited]
    evs = detect_events(pts)
    lo, hi = pts[0][0], pts[-1][0]
    bounds = [lo] + [e for e in evs if lo < e < hi] + [hi]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = [p for p in pts if a <= p[0] <= b]
        tt = np.arange(a, b, 1 / 30.0)
        if len(seg) >= 6 and b - a > 0.12:
            T = np.array([p[0] for p in seg]) - a
            X = np.array([p[1] for p in seg])
            Y = np.array([p[2] for p in seg])
            for _ in range(2):          # trimmed refit
                cx = np.polyfit(T, X, 2)
                cy = np.polyfit(T, Y, 2)
                rx = np.abs(np.polyval(cx, T) - X)
                ry = np.abs(np.polyval(cy, T) - Y)
                keep = (rx + ry) <= np.percentile(rx + ry, 80)
                if keep.sum() >= 6:
                    T, X, Y = T[keep], X[keep], Y[keep]
            for t in tt:
                out.append((t, float(np.polyval(cx, t - a)),
                            float(np.polyval(cy, t - a))))
        else:                            # too thin: keep raw points
            for t in tt:
                near = min(seg, key=lambda p: abs(p[0] - t)) if seg else None
                if near:
                    out.append((t, near[1], near[2]))
    return out


def to_per_frame(visited, t0):
    """Visited points + linear interpolation across gaps <= GAP_MAX."""
    out = {}
    for i, (f, x, y) in enumerate(visited):
        out[f] = (x, y, True)
        if i + 1 < len(visited):
            f2, x2, y2 = visited[i + 1]
            for g in range(f + 1, f2):
                a = (g - f) / (f2 - f)
                out[g] = (x + a * (x2 - x), y + a * (y2 - y), False)
    return {f: (t0 + f / FPS, x, y, vis) for f, (x, y, vis) in out.items()}


def score_train(rally, per_frame, visited, t0, refined=None):
    # gate panel: first physical contact to 0.5 s after the last
    imps_panel, _ = load_impacts(rally=rally)
    p_lo, p_hi = imps_panel[0], imps_panel[-1] + 0.5
    labels = [r for r in csv.DictReader(open(DATA / f"ball_path_r{rally}.csv"))
              if r["x"] and p_lo <= float(r["t_s"]) <= p_hi]
    print(f"--- rally {rally}: decoded {len(visited)} visited points; "
          f"gate panel {p_lo:.2f}-{p_hi:.2f}s")
    for cls, tol in (("V", 25.0), ("S", 40.0)):
        hit = tot = 0
        for r in labels:
            if r["vis"] != cls:
                continue
            tot += 1
            f0 = round((float(r["t_s"]) - t0) * FPS)
            best = min((math.hypot(per_frame[g][1] - float(r["x"]),
                                   per_frame[g][2] - float(r["y"]))
                        for g in (f0 - 1, f0, f0 + 1) if g in per_frame),
                       default=1e9)
            hit += int(best <= tol)
        bar = " (gate check 1 bar: 70%)" if cls == "V" else ""
        print(f"  {cls} hit rate: {100*hit/max(tot,1):.1f}% ({hit}/{tot}){bar}")

    # check 2, human-matched (Amendment 1): tracker vs human, same battery
    imps, dead = load_impacts(rally=rally)
    span = (imps[0] - 1.0, dead)
    trk_pts = refined if refined is not None else [
        (t0 + f / FPS, x, y) for f, x, y in visited]
    hum_pts = [(float(r["t_s"]), float(r["x"]), float(r["y"]))
               for r in labels if r["vis"] == "V"]
    res = {}
    for name, pts in (("tracker", trk_pts), ("human", hum_pts)):
        evs = detect_events(pts)
        obs, p95, pct, med = score_events(evs, imps, span)
        res[name] = (obs, pct)
        print(f"  turns[{name:7s}]: recall {100*obs:.1f}% "
              f"at null pct {100*pct:.0f} (null median {100*med:.1f}, "
              f"95th {100*p95:.1f})")
    ok = (res["tracker"][0] >= res["human"][0]
          and res["tracker"][1] >= res["human"][1])
    print(f"  CHECK 2 human-matched: {'PASS' if ok else 'FAIL'} "
          f"(tracker vs human: recall {100*res['tracker'][0]:.0f} vs "
          f"{100*res['human'][0]:.0f}, pct {100*res['tracker'][1]:.0f} vs "
          f"{100*res['human'][1]:.0f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rally", type=int, required=True)
    ap.add_argument("--serve", type=float, default=None,
                    help="serve pin (VOD s) — licensed gate input; trims "
                         "decode start. Train default: first contact tap.")
    ap.add_argument("--pose", help="pose npz for body-box costs (automated channel)")
    ap.add_argument("--anchors", help="hitter-chain anchors CSV (the missile coupling)")
    ap.add_argument("--dump")
    ap.add_argument("--graded-run", action="store_true")
    a = ap.parse_args()
    if a.rally in SEALED and not a.graded_run:
        raise SystemExit(f"rally {a.rally} is SEALED — refusing")
    byf, t0 = load_candidates(a.rally)
    serve = a.serve
    if serve is None and a.rally not in SEALED:
        serve = load_impacts(rally=a.rally)[0][0]
    if serve is not None:
        f_min = round((serve - 0.3 - t0) * FPS)
        byf = {f: c for f, c in byf.items() if f >= f_min}
    flags = None
    if a.pose and BODY_PEN > 0:
        flags = in_body_flags(byf, t0, body_boxes(a.pose))
    oflags = out_of_court_flags(byf, court_hull())
    aflags = None
    if a.anchors:
        anc = [(float(r["t_s"]), float(r["wrist_x"]), float(r["wrist_y"]))
               for r in csv.DictReader(open(a.anchors))]
        aflags = anchor_flags(byf, t0, anc)
    visited = decode(byf, flags, oflags, aflags)
    refined = refine_arcs(visited, t0)
    # per-frame positions from the refit (check 1 uses these too)
    per_frame = {}
    for t, x, y in refined:
        per_frame[round((t - t0) * FPS)] = (t, x, y, True)
    if a.rally not in SEALED:
        score_train(a.rally, per_frame, visited, t0, refined)
    if a.dump:
        with open(a.dump, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "t_s", "x", "y", "observed"])
            for fr in sorted(per_frame):
                t, x, y, vis = per_frame[fr]
                w.writerow([fr, round(t, 3), round(x, 1), round(y, 1),
                            int(vis)])
        print(f"dumped {a.dump}")


if __name__ == "__main__":
    main()
