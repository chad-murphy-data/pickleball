"""Fit-level lab: run check-3 scoring on cached pipelines under a
claiming config, with an optional ARC-INTERSECTION read-out — impact
time refined to where the adjacent fitted arcs agree best, instead of
the nominal bound time (the r9 fast-shot fix candidate). Refinement
applies to BOTH sides (tracked and human) — the check stays
apples-to-apples.

Usage: python3 fit_lab.py <rally> [--kc K] [--kv V] [--ix]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import ball_replicate as br           # noqa: E402
import court3d as c3                  # noqa: E402
from claim_lab import (load, paddle_series, apply_config,   # noqa: E402
                       claim_production, claim_bounds_veto,
                       claim_bounds_veto2, claim_v3)
from hole_audit import turn_events    # noqa: E402

MATCH_S = br.MATCH_S
NET_Y = br.NET_Y
IMPACT_BAR_FT = br.IMPACT_BAR_FT


def refine_impacts(segs, bounds, cons, win=0.25, grid=1 / 240):
    """cons[k] moved to the adjacent arcs' best-agreement point."""
    out = list(cons)
    for k in range(1, len(bounds) - 1):
        L, R = segs[k - 1], segs[k] if k < len(segs) else None
        if not (L and L.get("ok") and R and R.get("ok")):
            continue
        aL, _, thL = L["arcs"][-1]
        aR, _, thR = R["arcs"][0]
        tt = np.arange(bounds[k] - win, bounds[k] + win, grid)
        pL = c3.arc_pos(thL, tt - aL)
        pR = c3.arc_pos(thR, tt - aR)
        d = np.linalg.norm(pL - pR, axis=1)
        i = int(np.argmin(d))
        if d[i] < 3.0:
            out[k] = 0.5 * (pL[i] + pR[i])
    return out


def score_c3(c, bounds, bevs, ix=False, quiet=False):
    P, floors = c["P"], c["floors"]
    anchors = br.dedupe_anchors(c["anchors"], c["zs"],
                                br.track_sides(floors), c["turns"])
    pts = c["visited"]
    t0 = c["t0"]
    obs = [(t0 + f / 60.0, x, y, 1.0) for f, x, y in pts]
    t_segs, t_cons, t_bounds, t_evs = br.crossing_demotion(
        P, obs, bounds + [c["end"]], bevs, floors, anchors)
    h_obs, h_bounds, h_evs = c["hum"]
    h_segs, h_cons = c["h_segs"], c["h_cons"]
    if ix:
        t_cons = refine_impacts(t_segs, t_bounds, t_cons)
        h_cons = refine_impacts(h_segs, h_bounds, h_cons)

    dists, used = [], set()
    pairs = []
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
            pairs.append((tb, h_bounds[j], d))
    n_h = sum(1 for j, hb in enumerate(h_bounds[:-1])
              if h_cons[j] is not None)
    med = float(np.median(dists)) if dists else float("inf")
    n_ok = n_x = 0
    for seg in t_segs:
        if seg and seg["ok"]:
            n_ok += 1
            ys = np.array([p[2] for p in c3.sample_path(seg)])
            if ys.min() < NET_Y < ys.max():
                n_x += 1
    t_b = sum(1 for s in t_segs if s and s["ok"] and s["kind"] == "bounce")
    h_b = sum(1 for s in h_segs if s and s["ok"] and s["kind"] == "bounce")
    ok = (dists and med <= IMPACT_BAR_FT and n_x == n_ok
          and abs(t_b - h_b) <= 1)
    if not quiet:
        for tb, hb, d in pairs:
            print(f"    impact {tb:7.2f} <-> {hb:7.2f} : {d:.2f} ft")
    print(f"  matched {len(dists)}/{n_h}  med {med:.2f} ft  "
          f"crossings {n_x}/{n_ok}  bounces {t_b}v{h_b}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return dict(matched=len(dists), n_h=n_h, med=med, nx=n_x,
                nok=n_ok, tb=t_b, hb=h_b, ok=ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rally", type=int)
    ap.add_argument("--kc", type=float, default=None)
    ap.add_argument("--kv", type=float, default=None)
    ap.add_argument("--ix", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--veto2", type=float, default=None,
                    help="bounce-geometry veto with pathdist th")
    ap.add_argument("--veto", choices=["next", "drop"], default=None)
    ap.add_argument("--v3", type=str, default=None,
                    help="FAR,NEAR e.g. 0.7,0.5 (9.9 = veto off)")
    ap.add_argument("--oracle", action="store_true",
                    help="DIAGNOSTIC: bounds = hand-timestamped taps "
                         "(fit-stage ceiling; never a production config)")
    ap.add_argument("--hybrid", type=str, default=None,
                    help="ZLO,RTH pose-admission widening, e.g. 0.5,0.6")
    ap.add_argument("--app", type=float, default=None,
                    help="add close-approach anchors at this rth "
                         "(union with production pose+blur set)")
    a = ap.parse_args()
    c = load(a.rally)
    if a.app is not None:
        from approach_lab import approach_events
        from claim_lab import paddle_at
        series = paddle_series(c["npz"])
        evs = [e for e in approach_events(c, series, a.app)
               if c["serve"] - 0.3 <= e[0] < c["end"]]
        anc = list(c["anchors"])
        zs = list(c["zs"])
        n_app = 0
        for t, rel, tid in evs:
            p = paddle_at(series, tid, t, tol=0.12)
            if p is None:
                continue
            anc.append((t, tid, p[0], p[1], p[0], p[1]))
            zs.append(3.0 - rel)
            n_app += 1
        dd = br.dedupe_anchors(anc, zs, br.track_sides(c["floors"]),
                               c["turns"])
        bounds = br.claim_bounds(c["turns"], c["angs"], c["timing_ref"],
                                 dd)
        bevs = [e for e in c["turns"] if e not in set(bounds)]
        print(f"  [approach rth={a.app}: +{n_app} anchors]")
    elif a.hybrid is not None:
        from admit_lab import peak_rel
        from hole_audit import pose_peaks_all
        zlo, rth = (float(x) for x in a.hybrid.split(","))
        series = paddle_series(c["npz"])
        peaks = [p for p in pose_peaks_all(c["npz"])
                 if c["serve"] - 0.3 <= p[0] < c["end"]]
        new = []
        for p in peaks:
            t, z, tid = p[0], p[1], int(p[2])
            if z >= 1.2 or not (z >= zlo
                                and peak_rel(c, series, t, tid) <= rth):
                continue
            if any(abs(t - x[0]) <= 0.15 and int(x[1]) == tid
                   for x in c["anchors"]):
                continue
            new.append((t, tid, p[3], p[4], p[5], p[6], z))
        anchors2 = list(c["anchors"]) + [x[:6] for x in new]
        zs2 = list(c["zs"]) + [x[6] for x in new]
        dd = br.dedupe_anchors(anchors2, zs2,
                               br.track_sides(c["floors"]), c["turns"])
        bounds = br.claim_bounds(c["turns"], c["angs"], c["timing_ref"],
                                 dd)
        bevs = [e for e in c["turns"] if e not in set(bounds)]
        print(f"  [hybrid z>={zlo} rel<={rth}: +{len(new)} anchors]")
    elif a.oracle:
        bounds = sorted(c["imps"])
        bevs = [e for e in c["turns"]
                if all(abs(e - b) > 0.15 for b in bounds)]
        print(f"  [oracle: {len(bounds)} tap-time bounds]")
    elif a.v3 is not None:
        far, near = (float(x) for x in a.v3.split(","))
        series = paddle_series(c["npz"])
        anchors, _ = claim_production(c)
        snap = turn_events(c["timing_ref"], 20.0)
        bounds, na = claim_v3(c, series, anchors, th_far=far,
                              th_near=near, snap_evs=snap)
        bevs = [e for e in c["turns"] if e not in set(bounds)]
        print(f"  [v3 far={far} near={near}: {na} orphan bounds added]")
    elif a.veto2 is not None:
        series = paddle_series(c["npz"])
        anchors, _ = claim_production(c)
        bounds = claim_bounds_veto2(c, series, anchors, th=a.veto2)
        bevs = [e for e in c["turns"] if e not in set(bounds)]
    elif a.veto is not None:
        anchors, _ = claim_production(c)
        bounds = claim_bounds_veto(c["turns"], c["angs"],
                                   c["timing_ref"], anchors, mode=a.veto)
        bevs = [e for e in c["turns"] if e not in set(bounds)]
    elif a.kc is None and a.kv is None:
        anchors, matched = claim_production(c)
        bounds = matched
        bevs = [e for e in c["turns"] if e not in set(matched)]
    else:
        series = paddle_series(c["npz"])
        _, bounds, bevs = apply_config(c, series, a.kc, a.kv)
    print(f"rally {a.rally}: {len(bounds)} bounds, {len(bevs)} "
          f"bounce events{' [ix]' if a.ix else ''}")
    score_c3(c, bounds, bevs, ix=a.ix, quiet=a.quiet)


if __name__ == "__main__":
    main()
