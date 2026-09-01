"""Check 3 of ball_gate.md — REPLICATION: court3d on the tracked path
vs court3d on the human path, same anchor policy on both sides.

The gate's deliverable check: the tracker exists to replicate the 3D
reconstruction WITHOUT hand labels. Both sides here are lifted with
court3d's fit machinery (DLT camera, ballistic-drag arcs, bounce
splits) under the SAME fully-automated person policy:

  hitter anchors: the hitter chain's predicted contacts carry a pose
  TRACK id; that track's ankle midpoint, pushed through the z=0
  homography of the one-time court calibration, gives the hitter's
  floor position — the paddle-depth prior court3d pass 2 needs. No
  names, no track_assign clicks, no tap anchors on either side.

Sides differ only in their 2D evidence:
  human side  — the user's ball pass (V weight 1.0, S 0.5) with the
                contact taps as arc bounds (it is the reference);
  tracked side — the decoder's visited points, with its OWN turn
                events split into contact bounds (turns matched to a
                hitter-chain anchor) and interior bounce candidates
                (unmatched turns).

Scored per the frozen bars:
  - median 3D distance between time-matched impact points <= 3.0 ft
  - every drawn tracked arc satisfies the net-crossing check
  - tracked bounce count within +/-1 of the human reconstruction

TRAIN harness: rallies 6/7 free; rally 8 SEALED (--graded-run only).

Usage:
    python3 vision/ball_replicate.py --rally 7 --npz pose/r0007.npz \
        --anchors anchors_r7.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_ball_audit import detect_events, load_impacts  # noqa: E402
import court3d as c3  # noqa: E402
import ball_decoder as bdec  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "vision"
SEALED = set()       # r10 spent 2026-09-01 (graded MIDDLE, now train); next seal = r20 when labeled

NET_Y = 22.0
END_TRIM_S = 0.06
MATCH_S = 0.25          # turn <-> anchor and tracked <-> human bound match
LANK, RANK = 15, 16     # COCO ankles
KPT_CONF = 0.3
IMPACT_BAR_FT = 3.0     # frozen gate bar


# ------------------------------------------------- automated person floor


def floor_homography(P):
    """pixel -> court (x,y) on the z=0 plane."""
    H = P[:, [0, 1, 3]]
    return np.linalg.inv(H)


def track_floor(npz_path, P):
    """{track: (t[], x_ft[], y_ft[])} from ankle midpoints — automated,
    every track kept (junk tracks are simply never referenced by an
    anchor)."""
    z = np.load(npz_path)
    Hi = floor_homography(P)
    out = {}
    for tid in set(z["track"].tolist()):
        m = np.where(z["track"] == tid)[0]
        t, kpt, kpc = z["t"][m], z["kpt"][m], z["kpc"][m]
        ts, xs, ys = [], [], []
        for i in range(len(m)):
            pts = [kpt[i, a] for a in (LANK, RANK) if kpc[i, a] >= KPT_CONF]
            if not pts:
                continue
            px, py = np.mean(pts, axis=0)
            g = Hi @ np.array([px, py, 1.0])
            x, y = g[0] / g[2], g[1] / g[2]
            if -8 <= x <= 28 and -8 <= y <= 52:
                ts.append(float(t[i])); xs.append(x); ys.append(y)
        if ts:
            o = np.argsort(ts)
            out[tid] = (np.array(ts)[o], np.array(xs)[o], np.array(ys)[o])
    return out


def floor_at(floors, tid, t, tol=0.3):
    if tid not in floors:
        return None
    ts, xs, ys = floors[tid]
    i = int(np.abs(ts - t).argmin())
    if abs(ts[i] - t) > tol:
        return None
    return np.array([xs[i], ys[i]])


def turn_angles(points, events, win=0.2):
    """Turn angle at each event time, same in/out-segment geometry as
    detect_events (nearest visible neighbors over a short window)."""
    pts = sorted(points)
    out = {}
    for e in events:
        w = [p for p in pts if abs(p[0] - e) <= win]
        pre = [p for p in w if p[0] < e]
        post = [p for p in w if p[0] > e]
        ang = 0.0
        if len(pre) >= 2 and len(post) >= 2:
            v1 = (pre[-1][1] - pre[0][1], pre[-1][2] - pre[0][2])
            v2 = (post[-1][1] - post[0][1], post[-1][2] - post[0][2])
            n1, n2 = math.hypot(*v1), math.hypot(*v2)
            if n1 > 1 and n2 > 1:
                cosv = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1])
                                     / (n1 * n2)))
                ang = math.degrees(math.acos(cosv))
        out[e] = ang
    return out


def load_anchors(path):
    return [(float(r["t_s"]), int(r["track"]), float(r["wrist_x"]),
             float(r["wrist_y"]))
            for r in csv.DictReader(open(path))]


def bound_anchor_positions(bounds, anchors, floors):
    """Per bound: hitter floor position (or None) via the nearest
    hitter-chain anchor within MATCH_S — the shared automated policy."""
    out = []
    for tb in bounds:
        cand = [(abs(tb - a[0]), a) for a in anchors
                if abs(tb - a[0]) <= MATCH_S]
        if cand:
            _, (ta, tid, _, _) = min(cand, key=lambda c: c[0])
            out.append(floor_at(floors, tid, ta))
        else:
            out.append(None)
    return out


# ------------------------------------------------------- reconstruction


def reconstruct(P, obs_all, bounds, events, panchors):
    """court3d's two-pass fit, generalized: arbitrary bounds, player
    anchors supplied per bound (automated), sides derived from the
    anchors' own floor positions. Returns (segs, cons) — cons[k] is the
    consensus 3D point at bound k (the impact points check 3 compares).
    Adapted from court3d.run pass 2 (the r1-validated instrument)."""
    segs, seg_obs = [], {}
    for k in range(len(bounds) - 1):
        t0, t1 = bounds[k], bounds[k + 1]
        obs = [o for o in obs_all
               if t0 + END_TRIM_S <= o[0] <= t1 - END_TRIM_S]
        if len(obs) < 5:
            segs.append(None)
            continue
        seg_obs[k] = obs
        seg = c3.fit_segment(P, obs, t0, t1, events)
        seg["ok"] = _plausible(seg)
        segs.append(seg)

    # sides straight from the automated anchors (no voting needed: the
    # hitter's feet say which half she is on)
    sides = [None if pa is None else (1.0 if pa[1] > NET_Y else -1.0)
             for pa in panchors]

    for _sweep in range(2):
        cons = [None] * len(bounds)
        for k, seg in enumerate(segs):
            if seg and seg["ok"]:
                p0, p1 = c3.seg_endpoints(seg, bounds[k], bounds[k + 1])
                for idx, pt in ((k, p0), (k + 1, p1)):
                    cons[idx] = (pt if cons[idx] is None
                                 else (cons[idx] + pt) / 2)
        for k, seg in enumerate(segs):
            if seg is None:
                continue
            t0, t1 = bounds[k], bounds[k + 1]
            s_h, s_n = sides[k], (sides[k + 1]
                                  if k + 1 < len(sides) else None)
            c0 = cons[k]
            c1 = cons[k + 1] if k + 1 < len(cons) else None
            pa0, pa1 = panchors[k], (panchors[k + 1]
                                     if k + 1 < len(panchors) else None)

            def make_extra(base, at_start, at_end, dur):
                def extra(th):
                    parts = list(base(th)) if base else []
                    if at_start:
                        pS = c3.arc_pos(th, [0.0])[0]
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
                        pE = c3.arc_pos(th, [dur])[0]
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
                th, rms = c3.fit_best(
                    P, obs, t0,
                    [seg["arcs"][0][2], c3.default_inits()[0]],
                    extra=make_extra(None, True, True, t1 - t0))
                seg2 = {"kind": "arc", "arcs": [(t0, t1, th)], "rms": rms}
            else:
                ts = seg["ts"]
                o1 = [o for o in obs if o[0] <= ts]
                o2 = [o for o in obs if o[0] >= ts]
                pen = 8.0
                a1, r1 = c3.fit_best(
                    P, o1, t0, [seg["arcs"][0][2], c3.default_inits()[0]],
                    extra=make_extra(
                        lambda th: pen * np.array(
                            [c3.arc_pos(th, [ts - t0])[0][2]]),
                        True, False, ts - t0))
                xy = c3.arc_pos(a1, [ts - t0])[0]
                a2, r2 = c3.fit_best(
                    P, o2, ts, [seg["arcs"][-1][2], c3.default_inits()[0]],
                    extra=make_extra(
                        lambda th: pen * np.concatenate(
                            [[th[2]], th[:2] - xy[:2]]),
                        False, True, t1 - ts))
                rms = float(np.sqrt(
                    (r1 ** 2 * len(o1) + r2 ** 2 * len(o2))
                    / (len(o1) + len(o2))))
                seg2 = {"kind": "bounce", "ts": ts,
                        "bounce_xy": c3.arc_pos(a1, [ts - t0])[0][:2],
                        "arcs": [(t0, ts, a1), (ts, t1, a2)], "rms": rms}
            seg2["ok"] = _plausible(seg2)
            if seg2["ok"] or not seg["ok"]:
                segs[k] = seg2

    cons = [None] * len(bounds)
    for k, seg in enumerate(segs):
        if seg and seg["ok"]:
            p0, p1 = c3.seg_endpoints(seg, bounds[k], bounds[k + 1])
            for idx, pt in ((k, p0), (k + 1, p1)):
                cons[idx] = (pt if cons[idx] is None
                             else (cons[idx] + pt) / 2)
    return segs, cons


def _plausible(seg):
    pts = np.array([p[1:] for p in c3.sample_path(seg)])
    if not len(pts):
        return False
    v = np.diff(pts, axis=0) * 60.0
    return bool(seg["rms"] < 8.0
                and pts[:, 0].min() > -15 and pts[:, 0].max() < 35
                and pts[:, 1].min() > -15 and pts[:, 1].max() < 60
                and pts[:, 2].min() > -3 and pts[:, 2].max() < 40
                and (np.linalg.norm(v, axis=1).max() < 176
                     if len(v) else True))


# --------------------------------------------------------------- sides

CLAIM_R = 130.0   # px: a claimed turn must sit near the claiming
                  # anchor's wrist — a contact happens AT the paddle
                  # (good anchors measure 39-73 px from the ball; junk
                  # anchors at fake swings are far from it). Train
                  # iteration 2026-09-01 after r10's graded MIDDLE:
                  # 73 anchors claimed ~30 turns on a 26-contact rally,
                  # stealing real bounces into contact bounds.


def track_sides(floors):
    """{track: -1/+1 team side} from the automated FLOOR positions
    (median court y vs the net) — the same channel check 3's
    reconstruction already trusts for anchor sides. The npz 'side'
    field is NOT team side (measured degenerate on r0010: all four
    player tracks +1, which collapsed the alternation prune to a
    single bound on the first full-iteration run)."""
    out = {}
    for tid, (ts, xs, ys) in floors.items():
        out[int(tid)] = 1 if float(np.median(ys)) > NET_Y else -1
    return out


def claim_bounds(turns, angs, refined, anchors, sides=None):
    """Anchor -> turn claiming with the two 2026-09-01 upgrades:
    (a) SPATIAL GATE — the claimed turn's path position must lie
    within CLAIM_R of the claiming anchor's wrist; (b) ALTERNATION
    PRUNE — consecutive claimed bounds must alternate the claiming
    track's team side (side alternation is EXACT in this sport: 0
    violations / 229 labeled contacts); where two consecutive claims
    share a side, the weaker-angle one is demoted back to the bounce
    pool. One anchor still claims at most one turn (largest angle in
    MATCH_S), per the r7 wobble lesson."""
    pts = sorted(refined)

    def path_at(e):
        near = min(pts, key=lambda p: abs(p[0] - e))
        return near[1], near[2]

    claims = {}                       # turn e -> (angle, track)
    for ta, tid, wx, wy in anchors:
        cand = []
        for e in turns:
            if abs(e - ta) > MATCH_S:
                continue
            px, py = path_at(e)
            if math.hypot(px - wx, py - wy) > CLAIM_R:
                continue
            cand.append((angs[e], e))
        if cand:
            ang, e = max(cand)
            if e not in claims or ang > claims[e][0]:
                claims[e] = (ang, int(tid))
    seq = []                          # (e, angle, side)
    for e in sorted(claims):
        ang, tid = claims[e]
        side = (sides or {}).get(tid)
        if (seq and side is not None and seq[-1][2] is not None
                and side == seq[-1][2]):
            if ang > seq[-1][1]:
                seq[-1] = (e, ang, side)
            continue                  # weaker same-side claim -> bounce
        seq.append((e, ang, side))
    return sorted(e for e, _, _ in seq)


def tracked_side(rally, anchors, floors, serve, end, sides=None):
    """Decode -> turns -> (bounds, bounce events) -> reconstruction
    inputs. Everything automated; serve/end are the licensed window."""
    byf, t0 = bdec.load_candidates(rally)
    f_min = round((serve - 0.3 - t0) * bdec.FPS)
    f_max = round((end + 0.3 - t0) * bdec.FPS)
    byf = {f: c for f, c in byf.items() if f_min <= f <= f_max}
    oflags = bdec.out_of_court_flags(byf, bdec.court_hull())
    aflags = bdec.anchor_flags(
        byf, t0, [(t, x, y) for t, _, x, y in anchors])
    visited = bdec.decode(byf, None, oflags, aflags)
    pts = [(t0 + f / bdec.FPS, x, y) for f, x, y in visited]
    # turns come from the arc-REFINED path — the same stream check 2
    # scores (raw visited points carry wobble that scrambles both the
    # event set and the claim angles); the raw points stay the fit
    # evidence below (real detections, not resampled arcs)
    refined = bdec.refine_arcs(visited, t0)
    turns = [e for e in detect_events(refined)
             if serve - 0.3 <= e < end - 0.05]
    angs = turn_angles(refined, turns)
    # one anchor claims at most ONE turn (the LARGEST-ANGLE one within
    # MATCH_S): an anchor is a predicted contact, one contact makes one
    # turn, and a real shot reverses the ball (measured on r7: real
    # contacts turn 101-171 deg, the path wobbles hugging the same
    # anchors turn 42-66 deg — nearest-in-time picked the wobble twice)
    matched = claim_bounds(turns, angs, refined, anchors, sides)
    claimed = set(matched)
    # TRIED AND REJECTED 2026-08-31: promoting unclaimed turns >= 90 deg
    # to contact bounds (to rescue r6's missile-missed 148.6 contact).
    # It swallowed r7's real bounces into bounds — the crossing
    # demotion cannot give a bounce back once its segment is split —
    # and flipped r7's check 3 to FAIL (bounces 0 vs 2, median 3.10).
    # A contact bound requires a hitter-chain anchor, full stop; a
    # missed anchor degrades to a longer segment, not a fake contact.
    bounce_evs = [e for e in turns if e not in claimed]
    bounds = matched + [end]
    obs = [(t, x, y, 1.0) for t, x, y in pts]
    return obs, bounds, bounce_evs


def human_side(rally, end):
    obs = []
    for r in csv.DictReader(open(DATA / f"ball_path_r{rally}.csv")):
        if r["x"] and r["vis"] in c3.W_VIS:
            obs.append((float(r["t_s"]), float(r["x"]), float(r["y"]),
                        c3.W_VIS[r["vis"]]))
    imps, _ = load_impacts(rally=rally)
    bounds = list(imps) + [end]
    evs = detect_events([(t, x, y) for t, x, y, w in obs if w == 1.0])
    return obs, bounds, evs


# --------------------------------------------------------------- score


def crossing_demotion(P, obs, bounds, evs, floors, anchors, rounds=3):
    """Reconstruct; demote any claimed contact whose FOLLOWING drawn
    segment never crosses the net (every real shot crosses — court3d's
    own physics prior, label-free) to an interior bounce candidate;
    refit until stable. A spurious contact bound splits a real flight
    in two, and the crossing can only live in one of the pieces."""
    for _ in range(rounds):
        pa = bound_anchor_positions(bounds, anchors, floors)
        segs, cons = reconstruct(P, obs, bounds, evs, pa)
        demote = None
        for k, seg in enumerate(segs):
            if seg is None or not seg["ok"] or k == 0:
                continue
            ys = np.array([p[2] for p in c3.sample_path(seg)])
            if not (ys.min() < NET_Y < ys.max()):
                demote = k
                break
        if demote is None:
            return segs, cons, bounds, evs
        evs = sorted(evs + [bounds[demote]])
        bounds = bounds[:demote] + bounds[demote + 1:]
    pa = bound_anchor_positions(bounds, anchors, floors)
    segs, cons = reconstruct(P, obs, bounds, evs, pa)
    return segs, cons, bounds, evs


def compare(rally, trk, hum, P, floors, anchors):
    t_obs, t_bounds, t_evs = trk
    h_obs, h_bounds, h_evs = hum
    t_segs, t_cons, t_bounds, t_evs = crossing_demotion(
        P, t_obs, t_bounds, t_evs, floors, anchors)
    print(f"tracked side: {len(t_bounds)-1} segments "
          f"(bounds at {', '.join(f'{b:.2f}' for b in t_bounds)})")
    print(f"human side:   {len(h_bounds)-1} segments")
    h_pa = bound_anchor_positions(h_bounds, anchors, floors)
    h_segs, h_cons = reconstruct(P, h_obs, h_bounds, h_evs, h_pa)

    for name, segs in (("tracked", t_segs), ("human", h_segs)):
        ok = sum(1 for s in segs if s and s["ok"])
        bo = sum(1 for s in segs if s and s["ok"] and s["kind"] == "bounce")
        rms = [s["rms"] for s in segs if s and s["ok"]]
        print(f"  {name:8s}: {ok}/{len(segs)} segments ok "
              f"({bo} bounces), rms median "
              f"{np.median(rms):.1f}px" if rms else f"  {name}: no fits")

    # impact points: time-match tracked bounds to human bounds
    dists, used = [], set()
    for k, tb in enumerate(t_bounds[:-1]):
        if t_cons[k] is None:
            continue
        m = [(abs(tb - hb), j) for j, hb in enumerate(h_bounds[:-1])
             if j not in used and abs(tb - hb) <= MATCH_S
             and h_cons[j] is not None]
        if m:
            _, j = min(m)
            used.add(j)
            d = float(np.linalg.norm(t_cons[k] - h_cons[j]))
            dists.append(d)
            print(f"  impact {tb:7.2f}s <-> {h_bounds[j]:7.2f}s : "
                  f"3D dist {d:.2f} ft")
    n_h = sum(1 for j, hb in enumerate(h_bounds[:-1])
              if h_cons[j] is not None)
    med = float(np.median(dists)) if dists else float("inf")

    # net-crossing check on drawn tracked arcs (court3d check: every
    # segment between contacts must cross the net)
    n_ok = n_x = 0
    for seg in t_segs:
        if seg and seg["ok"]:
            n_ok += 1
            ys = np.array([p[2] for p in c3.sample_path(seg)])
            if ys.min() < NET_Y < ys.max():
                n_x += 1

    t_b = sum(1 for s in t_segs if s and s["ok"] and s["kind"] == "bounce")
    h_b = sum(1 for s in h_segs if s and s["ok"] and s["kind"] == "bounce")

    print(f"\nCHECK 3 — replication vs the human reconstruction:")
    print(f"  matched impact points: {len(dists)}/{n_h} human impacts, "
          f"median 3D dist {med:.2f} ft (bar <= {IMPACT_BAR_FT})")
    print(f"  net crossings: {n_x}/{n_ok} drawn tracked segments cross")
    print(f"  bounces: tracked {t_b} vs human {h_b} (bar +/-1)")
    ok = (dists and med <= IMPACT_BAR_FT and n_x == n_ok
          and abs(t_b - h_b) <= 1)
    print(f"  CHECK 3: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rally", type=int, required=True)
    ap.add_argument("--npz", required=True,
                    help="pose npz covering the rally (automated channel)")
    ap.add_argument("--anchors", required=True,
                    help="hitter-chain anchors CSV (with track column)")
    ap.add_argument("--serve", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--graded-run", action="store_true")
    a = ap.parse_args()
    if a.rally in SEALED and not a.graded_run:
        raise SystemExit(f"rally {a.rally} is SEALED — refusing")
    serve, end = a.serve, a.end
    if a.rally not in SEALED:
        imps, dead = load_impacts(rally=a.rally)
        serve = serve if serve is not None else imps[0]
        end = end if end is not None else dead
    X3, x2, _ = c3.load_landmarks()
    P = c3.dlt(X3, x2)
    floors = track_floor(a.npz, P)
    anchors = load_anchors(a.anchors)
    trk = tracked_side(a.rally, anchors, floors, serve, end,
                       track_sides(floors))
    hum = human_side(a.rally, end)
    compare(a.rally, trk, hum, P, floors, anchors)


if __name__ == "__main__":
    main()
