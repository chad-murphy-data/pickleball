"""Claiming lab (stage B): iterate bound-claiming logic on cached
pipelines. Cheap metrics first (no 3D fits):

  - bound recall vs taps (+/-0.15 s)
  - extra bounds (claimed, no tap nearby) = fake contacts
  - false-claimed HUMAN BOUNCES (human fit's bounce times claimed as
    contacts — the metric that killed naive alternation pruning)

Configs:
  P0  production loose claiming (baseline)
  P1k P0 + proximity-claim: unclaimed turn within k*h of any track's
      paddle point -> contact bound
  P2k P1 + proximity-veto: anchor-claimed turn farther than KV*h from
      the claiming track's paddle -> unclaim (fake swing)
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import ball_replicate as br           # noqa: E402
import hitter_chain as hc             # noqa: E402

SP = Path(__file__).resolve().parent
MATCH_S = br.MATCH_S


def load(rally):
    with open(SP / f"c3_cache_r{rally}.pkl", "rb") as f:
        return pickle.load(f)


def paddle_series(npz_path):
    """per track: sorted (t, px, py, h) with NaNs dropped."""
    z = np.load(npz_path)
    tids = sorted(set(z["track"].tolist()),
                  key=lambda k: -(z["track"] == k).sum())[:4]
    out = {}
    for tid in tids:
        t, sp, re, wx, wy, pxa, pya = hc.track_signals(z, tid)
        m = np.where(z["track"] == tid)[0]
        box = z["box"][m]
        h = np.maximum(box[:, 3] - box[:, 1], 20.0)
        order = np.argsort(t)
        t, pxa, pya, h = t[order], pxa[order], pya[order], h[order]
        good = ~np.isnan(pxa)
        out[tid] = (t[good], pxa[good], pya[good], h[good])
    return out


def paddle_at(series, tid, tq, tol=0.06):
    t, px, py, h = series[tid]
    if not len(t):
        return None
    i = int(np.argmin(np.abs(t - tq)))
    if abs(t[i] - tq) > tol:
        return None
    return px[i], py[i], h[i]


def path_at(pts, e):
    near = min(pts, key=lambda p: abs(p[0] - e))
    return near[1], near[2]


def claim_production(c):
    anchors = br.dedupe_anchors(c["anchors"], c["zs"],
                                br.track_sides(c["floors"]), c["turns"])
    matched = br.claim_bounds(c["turns"], c["angs"], c["timing_ref"],
                              anchors)
    return anchors, matched


def prox_min(series, pos, tq):
    """min over tracks of dist(pos, paddle)/h; returns (rel, tid)."""
    best = (np.inf, None)
    for tid in series:
        p = paddle_at(series, tid, tq)
        if p is None:
            continue
        rel = float(np.hypot(pos[0] - p[0], pos[1] - p[1]) / p[2])
        if rel < best[0]:
            best = (rel, tid)
    return best


def apply_config(c, series, k_claim=None, k_veto=None):
    anchors, matched = claim_production(c)
    turnset = list(c["turns"])
    tr = c["timing_ref"]
    claimed = set(matched)
    # veto: claimed turn far from EVERY track's paddle -> unclaim
    if k_veto is not None:
        keep = set()
        for e in claimed:
            rel, _ = prox_min(series, path_at(tr, e), e)
            if rel <= k_veto:
                keep.add(e)
        claimed = keep
    # claim: unclaimed turn near any paddle -> contact
    if k_claim is not None:
        for e in turnset:
            if e in claimed:
                continue
            rel, _ = prox_min(series, path_at(tr, e), e)
            if rel <= k_claim:
                claimed.add(e)
    bounds = sorted(claimed)
    bevs = [e for e in turnset if e not in claimed]
    return anchors, bounds, bevs


def human_bounce_times(c):
    """bounce times from the cached human fit (truth-grade on r10)."""
    out = []
    for s in c["h_segs"]:
        if s and s.get("ok") and s.get("kind") == "bounce":
            out.append(float(s["ts"]))
    return out


def score(c, bounds, hb_times):
    imps = c["imps"]
    used = set()
    rec = 0
    extras = []
    for b in bounds:
        m = [(abs(b - t), i) for i, t in enumerate(imps)
             if i not in used and abs(b - t) <= 0.15]
        if m:
            _, i = min(m)
            used.add(i)
            rec += 1
        else:
            extras.append(b)
    fb = sum(1 for b in extras
             if any(abs(b - t) <= 0.15 for t in hb_times))
    return rec, len(imps), extras, fb


def main():
    rallies = [int(x) for x in sys.argv[1:]] or [7, 9, 10, 6]
    for rally in rallies:
        c = load(rally)
        series = paddle_series(c["npz"])
        hb = human_bounce_times(c)
        print(f"== rally {rally}: {len(c['imps'])} taps, "
              f"{len(c['turns'])} turns, human bounces {len(hb)}")
        cfgs = [("P0", None, None)]
        cfgs += [(f"P1 k={k}", k, None) for k in (0.35, 0.5, 0.7)]
        cfgs += [(f"P2 k={k} v={v}", k, v)
                 for k in (0.5,) for v in (0.9, 1.3)]
        for name, kc, kv in cfgs:
            anchors, bounds, bevs = apply_config(c, series, kc, kv)
            rec, n, extras, fb = score(c, bounds, hb)
            print(f"  {name:14s}: bounds {len(bounds)}  recall {rec}/{n}"
                  f"  extras {len(extras)} (of which {fb} sit on human"
                  f" bounces)")


if __name__ == "__main__":
    main()


# --------------------------------------------- bounce-geometry veto

def turn_v01(pts, e):
    """(vx0, vy0, vx1, vy1) at the sample nearest e (±2-sample geometry,
    mirroring detect_events)."""
    i = int(np.argmin([abs(p[0] - e) for p in pts]))
    if i < 2 or i > len(pts) - 3:
        return None
    x, y = pts[i][1], pts[i][2]
    return (x - pts[i - 2][1], y - pts[i - 2][2],
            pts[i + 2][1] - x, pts[i + 2][2] - y)


def bounce_like(pts, e):
    """True when the turn keeps horizontal direction (bounce signature:
    75% of bounce-turns vs 23% of contact-turns, 4-rally truth)."""
    v = turn_v01(pts, e)
    if v is None:
        return False
    vx0, vy0, vx1, vy1 = v
    return (abs(vx0) > 1 and abs(vx1) > 1
            and np.sign(vx0) == np.sign(vx1))


def bounce_like2(pts, e, vy_up=False):
    v = turn_v01(pts, e)
    if v is None:
        return False
    vx0, vy0, vx1, vy1 = v
    keepx = (abs(vx0) > 1 and abs(vx1) > 1
             and np.sign(vx0) == np.sign(vx1))
    return keepx and (vy1 < 0 if vy_up else True)


def claim_bounds_veto2(c, series, anchors, th=0.7, vy_up=False,
                       mode="next"):
    """veto = bounce-like turn AND far anchor (pathdist > th) — the
    conjunction protects real contacts with keepx geometry."""
    tr = c["timing_ref"]
    turns, angs = c["turns"], c["angs"]
    claimed = set()
    for a in anchors:
        ta = a[0]
        far = anchor_reldist(c, series, a) > th
        cand = sorted(((angs[e], e) for e in turns
                       if abs(e - ta) <= br.MATCH_S), reverse=True)
        for _, e in cand:
            if far and bounce_like2(tr, e, vy_up):
                if mode == "next":
                    continue
                break
            claimed.add(e)
            break
    return sorted(claimed)


def claim_bounds_veto(turns, angs, tr, anchors, mode="next"):
    """production loose claiming + bounce-geometry veto.
    mode 'next': vetoed turn -> claim next-largest non-bounce-like turn
    in window; 'drop': vetoed turn -> claim nothing."""
    claimed = set()
    for a in anchors:
        ta = a[0]
        cand = sorted(((angs[e], e) for e in turns
                       if abs(e - ta) <= br.MATCH_S), reverse=True)
        for _, e in cand:
            if bounce_like(tr, e):
                if mode == "next":
                    continue
                break
            claimed.add(e)
            break
    return sorted(claimed)


def claim_v3(c, series, anchors, th_far=0.7, th_near=0.5,
             snap_evs=None, vy_up=False):
    """P0 loose claiming + conjunction veto (far anchor AND bounce-like
    turn) + anchor-time bounds for GOOD-pathdist anchors whose window
    holds no turn at all. Returns (bounds, n_added)."""
    tr = c["timing_ref"]
    turns, angs = c["turns"], c["angs"]
    claimed = set()
    orphans = []                      # (reldist, ta) good anchors, no turn
    for a in anchors:
        ta = a[0]
        r = anchor_reldist(c, series, a)
        cand = sorted(((angs[e], e) for e in turns
                       if abs(e - ta) <= br.MATCH_S), reverse=True)
        if not cand:
            if r <= th_near:
                orphans.append((r, ta))
            continue
        for _, e in cand:
            if r > th_far and bounce_like2(tr, e, vy_up):
                continue
            claimed.add(e)
            break
    bounds = sorted(claimed)
    n_add = 0
    for r, ta in sorted(orphans):
        bt = snap_time(ta, snap_evs or [])
        if all(abs(bt - b) >= MIN_BOUND_SEP for b in bounds):
            bounds.append(bt)
            bounds.sort()
            n_add += 1
    return bounds, n_add


# ------------------------------------------------------- claiming v2

TH_PATH = 0.6          # trusted-anchor gate: ball path within this many
                       # body-heights of the claiming track's paddle
SNAP_S = 0.12          # snap anchor-time bounds to a >=20-deg event
MIN_BOUND_SEP = 0.25


def anchor_reldist(c, series, a):
    ta, tid = a[0], int(a[1])
    px, py = path_at(c["timing_ref"], ta)
    p = paddle_at(series, tid, ta, tol=0.12)
    if p is None:
        return float("inf")
    return float(np.hypot(px - p[0], py - p[1]) / p[2])


def dedupe_v2(c, series):
    """production cluster rules; rank = ball-at-paddle first, z second."""
    anchors, zs = c["anchors"], c["zs"]
    sides = br.track_sides(c["floors"])
    rels = [anchor_reldist(c, series, a) for a in anchors]

    def rank(z, r):
        return (-r if np.isfinite(r) else -99.0, z)

    keep = []
    items = sorted(zip(zs, rels, anchors), key=lambda x: x[2][0])
    for z, r, a in items:
        t, tid = a[0], int(a[1])
        side = sides.get(tid)
        drop = False
        for j, (z2, r2, a2) in enumerate(keep):
            t2, tid2 = a2[0], int(a2[1])
            side2 = sides.get(tid2)
            close_t = abs(t - t2) <= br.DEDUP_SAME_T
            close_side = (side is not None and side == side2
                          and abs(t - t2) <= br.DEDUP_SIDE_T)
            if close_t or close_side:
                if rank(z, r) > rank(z2, r2):
                    keep[j] = (z, r, a)
                drop = True
                break
        if not drop:
            keep.append((z, r, a))
    kept = [(a, r) for _, r, a in keep]
    return kept


def snap_time(t, snap_evs):
    m = [e for e in snap_evs if abs(e - t) <= SNAP_S]
    return min(m, key=lambda e: abs(e - t)) if m else t


def claim_v2(c, series, th=TH_PATH, fill=True, snap_evs=None,
             peaks=None, verbose=False, policy="closed"):
    """policy: 'closed' = inf reldist vetoed; 'open' = inf trusted
    (veto only anchors MEASURED far from the paddle); 'off' = no gate."""
    kept = dedupe_v2(c, series)
    tsides = br.track_sides(c["floors"])
    turns, angs = c["turns"], c["angs"]
    bounds = []               # (t, side, how)
    for a, r in kept:
        veto = (r > th if policy == "closed"
                else (np.isfinite(r) and r > th) if policy == "open"
                else False)
        if veto:
            continue
        ta, tid = a[0], int(a[1])
        cand = [(angs[e], e) for e in turns if abs(e - ta) <= MATCH_S]
        if cand:
            _, e = max(cand)
            bt, how = e, "turn"
        else:
            bt, how = snap_time(ta, snap_evs or []), "anchor"
        bounds.append((bt, tsides.get(tid), how))
    # collapse near-duplicate bounds
    bounds.sort()
    ded = []
    for b in bounds:
        if ded and abs(b[0] - ded[-1][0]) < MIN_BOUND_SEP:
            if b[2] == "turn" and ded[-1][2] != "turn":
                ded[-1] = b
            continue
        ded.append(b)
    bounds = ded
    n_fill = 0
    if fill and peaks is not None:
        for _round in range(2):
            add = []
            for i in range(1, len(bounds)):
                t0, s0, _ = bounds[i - 1]
                t1, s1, _ = bounds[i]
                gap = t1 - t0
                same = s0 is not None and s0 == s1
                if not (same or gap > 1.7):
                    continue
                want = None
                if same and s0 is not None:
                    want = -s0 if isinstance(s0, (int, float)) else None
                cands = []
                for p in peaks:
                    tp, zp, tidp = p[0], p[1], int(p[2])
                    if not (t0 + MIN_BOUND_SEP < tp < t1 - MIN_BOUND_SEP):
                        continue
                    rel = anchor_reldist(c, series, (tp, tidp, 0, 0,
                                                     p[5], p[6]))
                    if rel > th:
                        continue
                    if want is not None and tsides.get(tidp) != want:
                        continue
                    cands.append((rel, tp, tidp))
                if cands:
                    rel, tp, tidp = min(cands)
                    add.append((snap_time(tp, snap_evs or []),
                                tsides.get(tidp), "fill"))
                    n_fill += 1
            if not add:
                break
            bounds = sorted(bounds + add)
    if verbose:
        for t, s, how in bounds:
            print(f"    bound {t:7.2f} side {s} [{how}]")
    bts = [b[0] for b in bounds]
    bevs = [e for e in turns
            if all(abs(e - b) > 1e-9 for b in bts)]
    return bts, bevs, n_fill
