"""pathfirst — the path-first ball tracker (pre-registered in
pathfirst_gate.md, 2026-09-01; read that first).

No contact guess anywhere. Flights are found FROM the candidate blobs:
three p-strong blobs at three times pin a drag-free ballistic arc (six
linear equations), the arc is scored by how many blobs lie along its
projection, survivors are refit with drag and GROWN frame by frame
until support runs out, and the grown span's ends ARE the contacts (or
bounces). The track is the projected arc at every frame of every
selected flight.

Usage:
  python3 pathfirst.py tune              # r6/r7 grid under the frozen rule -> pathfirst_tune.json
  python3 pathfirst.py grade <rally>     # one shot vs the incumbent (refuses r9/r10 without a live verdict)
  python3 pathfirst.py grade <rally> --p-seed .. --s-min .. --gap ..   # train rallies only
  python3 pathfirst.py selftest          # planted arc through synthetic blobs
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import court3d as c3                                        # noqa: E402
import corridor_dp as cdp                                   # noqa: E402
import spaghetti as spag                                    # noqa: E402
from claim_lab import load, paddle_series                   # noqa: E402
from corridor_lab import (load_truth, prod_contacts, corridors,  # noqa
                          decode_recall)
import geom_fix                                             # noqa: E402

SP = Path(__file__).parent
TUNE_JSON = SP / "pathfirst_tune.json"
G = c3.G
FPS = 60.0

# ---- frozen structure (pathfirst_gate.md) ----
N_SEED = 4
DS = (12, 24, 40)               # frames between the outer seeds
MID_OFF = (-2, 0, 2)
R_SUP = 8.0                     # support kernel radius (px)
R_GROW = 10.0                   # join radius while growing (px)
P_LO, P_HI = 0.2, 0.8           # kernel appearance term 0.2 + 0.8 p
BODY_FACT = 0.3
NEAR_BODY = 16.0
N_PROBE = 8                     # random probes per frame (baseline)
CAP_PER_FRAME = 40              # candidates kept per frame (top by p)
REFIT_EVERY = 6
OVERLAP_SEL = 3                 # frames two selected flights may share
NMS_OVERLAP = 0.6
NMS_PX = 12.0
Z_LO, Z_HI = -0.5, 12.0
SP_LO, SP_HI = 10.0, 110.0
XY_LO = np.array([-10.0, -10.0])
XY_HI = np.array([30.0, 54.0])
BOUNCE_Z = 0.3
BOUNCE_DT, BOUNCE_DXY = 0.15, 2.0
W_BOOK = 0.5                    # weak launch-state prior (r6/r7 book)
NULL_SEED = 20260901
GRID_PSEED = (0.4, 0.6)
GRID_SMIN = (4.0, 6.0, 8.0)
GRID_GAP = (6, 12)
EVAL_RALLIES = (9, 10)
INC_TRAIN = dict(h12=205, have=329)     # incumbent prod r6+r7 (softdp)

_BOOK = np.array([(l["speed"], l["loft"]) for l in spag.PRIOR["launches"]
                  if l["rally"] in (6, 7) and l["fwd"] > 2.0
                  and 12.0 <= l["speed"] <= 105.0])


# ----------------------------------------------------------- context

def context(rally):
    pxs = "_x" if rally in (6, 7) else ""
    c = load(rally)
    series = paddle_series(c["npz"])
    truth = load_truth(rally)
    t0 = c["t0"]
    f_lo = int((c["serve"] - 0.4 - t0) * 60)
    f_hi = int((c["end"] + 0.2 - t0) * 60)
    cc = spag.cands_cached(rally, f_lo, f_hi, 14, "cc", lrn=True, pxs=pxs)
    body = cdp.body_points(c, f_lo, f_hi)
    frames = {}
    for f, cs in cc.items():
        arr = np.asarray([(c_[0], c_[1], c_[4]) for c_ in cs], float)
        if not len(arr):
            continue
        barr = body.get(f)
        if barr is not None and len(barr):
            d = np.hypot(arr[:, 0:1] - barr[None, :, 0],
                         arr[:, 1:2] - barr[None, :, 1]).min(axis=1)
            nb = d <= NEAR_BODY
        else:
            nb = np.zeros(len(arr), bool)
        order = np.argsort(-arr[:, 2])[:CAP_PER_FRAME]
        frames[f] = (arr[order], nb[order])
    arms = (("prod", corridors(c, series, prod_contacts(c, series, 0.5))),
            ("oracle", corridors(c, series, list(c["imps"]))))
    return dict(rally=rally, c=c, truth=truth, t0=t0, cc=cc, pxs=pxs,
                body=body, frames=frames, f_lo=f_lo, f_hi=f_hi,
                dec=decode_recall(c, truth), arms=arms, P=c["P"],
                series=series)


# ------------------------------------------------ kernel + baseline

def kernel(d, p, nb):
    return (np.clip(1.0 - d / R_SUP, 0.0, None) * (P_LO + P_HI * p)
            * np.where(nb, BODY_FACT, 1.0))


def frame_base(frames, seed=NULL_SEED):
    """what a random pixel (inside the frame's candidate bounding box)
    collects — subtracted so junk-dense frames support nothing."""
    rng = np.random.default_rng(seed)
    base = {}
    for f, (arr, nb) in frames.items():
        lo, hi = arr[:, :2].min(0), arr[:, :2].max(0)
        q = rng.uniform(lo, np.maximum(hi, lo + 1), (N_PROBE, 2))
        d = np.hypot(q[:, None, 0] - arr[None, :, 0],
                     q[:, None, 1] - arr[None, :, 1])
        base[f] = float(kernel(d, arr[None, :, 2], nb[None, :])
                        .max(axis=1).mean())
    return base


def support(P, theta_batch, f_start, f_end, frames, base, t0):
    """support of a batch of 6-param (drag-free) arcs sharing a frame
    span. theta (n,6) with tau measured from frame f_start."""
    n = theta_batch.shape[0]
    fs = np.arange(f_start, f_end + 1)
    tau = (fs - f_start) / FPS
    p0, v0 = theta_batch[:, None, :3], theta_batch[:, None, 3:6]
    acc = np.array([0.0, 0.0, 0.5 * G])
    X = p0 + v0 * tau[None, :, None] + acc * (tau[None, :, None] ** 2)
    px = c3.project(P, X.reshape(-1, 3)).reshape(n, len(fs), 2)
    tot = np.zeros(n)
    nsup = np.zeros(n, int)
    for j, f in enumerate(fs):
        fr = frames.get(int(f))
        if fr is None:
            continue
        arr, nb = fr
        d = np.hypot(px[:, j, 0:1] - arr[None, :, 0],
                     px[:, j, 1:2] - arr[None, :, 1])
        s = kernel(d, arr[None, :, 2], nb[None, :]).max(axis=1)
        tot += s - base.get(int(f), 0.0)
        nsup += s > 0.15
    return tot, nsup, X


# ------------------------------------------------ 3-point arc solve

def solve_arcs(P, pts, taus):
    """pts (n,3,2) image points at taus (3,) -> theta (n,6) drag-free
    arcs (p0, v0), tau from taus[0]-frame. Rows of the DLT-style
    system: (P1 - u P3).X~ = 0, (P2 - v P3).X~ = 0."""
    n = pts.shape[0]
    acc = np.array([0.0, 0.0, 0.5 * G])
    A = np.zeros((n, 6, 6))
    b = np.zeros((n, 6))
    for i in range(3):
        tau = taus[i]
        u, v = pts[:, i, 0], pts[:, i, 1]
        for k, (row, coord) in enumerate(((P[0], u), (P[1], v))):
            r = row[None, :] - coord[:, None] * P[2][None, :]   # (n,4)
            rx, rw = r[:, :3], r[:, 3]
            A[:, 2 * i + k, :3] = rx
            A[:, 2 * i + k, 3:] = rx * tau
            b[:, 2 * i + k] = -(rx @ acc * tau ** 2 + rw)
    with np.errstate(all="ignore"):
        try:
            th = np.linalg.solve(A, b[..., None])[..., 0]
        except np.linalg.LinAlgError:
            th = np.full((n, 6), np.nan)
            for i in range(n):
                try:
                    th[i] = np.linalg.solve(A[i], b[i])
                except np.linalg.LinAlgError:
                    pass
    return th


def plausible(theta, span_s):
    """physical launch state + court volume over the span."""
    p0, v0 = theta[:, :3], theta[:, 3:6]
    ok = np.all(np.isfinite(theta), axis=1)
    sp = np.linalg.norm(v0, axis=1)
    ok &= (sp >= SP_LO) & (sp <= SP_HI)
    ts = np.linspace(0, span_s, 5)
    acc = np.array([0.0, 0.0, 0.5 * G])
    X = p0[:, None, :] + v0[:, None, :] * ts[None, :, None] \
        + acc * ts[None, :, None] ** 2
    ok &= (X[:, :, 2] >= Z_LO).all(1) & (X[:, :, 2] <= Z_HI).all(1)
    ok &= (X[:, :, :2] >= XY_LO).all((1, 2)) & (X[:, :, :2] <= XY_HI).all((1, 2))
    return ok


def book_pen(theta):
    v0 = theta[3:6]
    sp = float(np.linalg.norm(v0))
    loft = float(np.degrees(np.arctan2(v0[2], np.hypot(v0[0], v0[1]))))
    dm = np.min(np.hypot((sp - _BOOK[:, 0]) / 8.0, (loft - _BOOK[:, 1]) / 10.0))
    return W_BOOK * max(0.0, dm - 1.0)


# ------------------------------------------------------ hypotheses

def seeds(frames, p_seed):
    out = {}
    for f, (arr, nb) in frames.items():
        m = (arr[:, 2] >= p_seed) & (~nb)
        if not m.any():
            continue
        a = arr[m]
        out[f] = a[np.argsort(-a[:, 2])[:N_SEED], :2]
    return out


def hypotheses(ctx, p_seed):
    """all plausible 3-seed arcs with their support: list of dicts
    (theta6, f_start, f_end, sup, nsup)."""
    P, frames, t0 = ctx["P"], ctx["frames"], ctx["t0"]
    base = frame_base(frames)
    sd = seeds(frames, p_seed)
    hyps = []
    for f1 in sorted(sd):
        s1 = sd[f1]
        for D in DS:
            f3 = f1 + D
            if f3 not in sd:
                continue
            s3 = sd[f3]
            for off in MID_OFF:
                f2 = f1 + D // 2 + off
                if f2 not in sd:
                    continue
                s2 = sd[f2]
                i1, i2, i3 = np.meshgrid(np.arange(len(s1)), np.arange(len(s2)),
                                         np.arange(len(s3)), indexing="ij")
                pts = np.stack([s1[i1.ravel()], s2[i2.ravel()], s3[i3.ravel()]],
                               axis=1)
                taus = np.array([0.0, (f2 - f1) / FPS, D / FPS])
                th = solve_arcs(P, pts, taus)
                ok = plausible(th, D / FPS)
                if not ok.any():
                    continue
                th = th[ok]
                fa, fb = f1 - D // 2, f3 + D // 2
                # re-base tau to fa
                dt = (fa - f1) / FPS
                acc = np.array([0.0, 0.0, G])
                p0 = th[:, :3] + th[:, 3:6] * dt + 0.5 * acc * dt ** 2
                v0 = th[:, 3:6] + acc * dt
                th2 = np.hstack([p0, v0])
                sup, nsup, _ = support(P, th2, fa, fb, frames, base, t0)
                for k in range(len(th2)):
                    hyps.append(dict(theta=th2[k], fa=fa, fb=fb,
                                     sup=float(sup[k]), nsup=int(nsup[k]),
                                     f1=f1, f3=f3))
    return hyps, base


def nms(hyps, s_min):
    keep = []
    hs = sorted([h for h in hyps if h["sup"] >= s_min],
                key=lambda h: -h["sup"])
    for h in hs:
        dup = False
        for k in keep:
            ov = min(h["fb"], k["fb"]) - max(h["fa"], k["fa"]) + 1
            if ov <= 0:
                continue
            if ov / min(h["fb"] - h["fa"] + 1, k["fb"] - k["fa"] + 1) < NMS_OVERLAP:
                continue
            fs = np.arange(max(h["fa"], k["fa"]), min(h["fb"], k["fb"]) + 1)
            if _agree(h, k, fs):
                dup = True
                break
        if not dup:
            keep.append(h)
    return keep


def _proj6(P, theta, fa, fs):
    tau = (np.asarray(fs) - fa) / FPS
    acc = np.array([0.0, 0.0, 0.5 * G])
    X = theta[:3] + theta[3:6] * tau[:, None] + acc * tau[:, None] ** 2
    return c3.project(P, X)


def _agree(h, k, fs, P=None):
    P = P if P is not None else _agree.P
    a = _proj6(P, h["theta"], h["fa"], fs)
    b = _proj6(P, k["theta"], k["fa"], fs)
    return float(np.median(np.hypot(*(a - b).T))) <= NMS_PX


# ------------------------------------------------ grow + refine

def arc_px(P, theta, t_ref, f, t0):
    X = c3.arc_pos(theta, [t0 + f / FPS - t_ref])
    return c3.project(P, X)[0]


def refit(P, obs, t_ref, theta0):
    th, rms = c3.fit_arc(P, obs, t_ref, theta0=theta0)
    return th, rms


def grow(ctx, h, gap):
    """refit with drag on inliers, then extend both ways."""
    P, frames, t0 = ctx["P"], ctx["frames"], ctx["t0"]
    t_ref = t0 + h["fa"] / FPS
    # start drag at ~0: the hypothesis is a drag-free solve, and seeding
    # k=0.3 bent the projected path off its own inliers before any refit
    # (selftest caught it: 5 inliers instead of 61, refit went wild).
    theta = np.concatenate([h["theta"], [1e-4]])
    # inliers over the initial span
    obs = {}
    for f in range(h["fa"], h["fb"] + 1):
        fr = frames.get(f)
        if fr is None:
            continue
        arr, nb = fr
        q = arc_px(P, theta, t_ref, f, t0)
        d = np.hypot(arr[:, 0] - q[0], arr[:, 1] - q[1])
        j = int(np.argmin(d))
        if d[j] <= R_SUP:
            obs[f] = (t0 + f / FPS, arr[j, 0], arr[j, 1],
                      float(arr[j, 2]) * (BODY_FACT if nb[j] else 1.0))
    if len(obs) < 6:
        return None
    theta, rms = refit(P, list(obs.values()), t_ref, theta)
    lo, hi = min(obs), max(obs)
    for direction in (+1, -1):
        f = (hi if direction > 0 else lo)
        miss = 0
        joined = 0
        while miss < gap:
            f += direction
            if f < ctx["f_lo"] - 30 or f > ctx["f_hi"] + 30:
                break
            fr = frames.get(f)
            q = arc_px(P, theta, t_ref, f, t0)
            X = c3.arc_pos(theta, [t0 + f / FPS - t_ref])[0]
            if X[2] < Z_LO - 0.5 or not (0 <= q[0] <= 1280 and 0 <= q[1] <= 720):
                break
            if fr is None:
                miss += 1
                continue
            arr, nb = fr
            d = np.hypot(arr[:, 0] - q[0], arr[:, 1] - q[1])
            j = int(np.argmin(d))
            if d[j] <= R_GROW:
                obs[f] = (t0 + f / FPS, arr[j, 0], arr[j, 1],
                          float(arr[j, 2]) * (BODY_FACT if nb[j] else 1.0))
                miss = 0
                joined += 1
                if joined % REFIT_EVERY == 0:
                    theta, rms = refit(P, list(obs.values()), t_ref, theta)
            else:
                miss += 1
        if joined % REFIT_EVERY:
            theta, rms = refit(P, list(obs.values()), t_ref, theta)
    fa, fb = min(obs), max(obs)
    wsum = float(sum(o[3] for o in obs.values()))
    return dict(theta=theta, t_ref=t_ref, fa=fa, fb=fb, n=len(obs),
                w=wsum, rms=rms, pen=book_pen(theta[:6]),
                density=(wsum - book_pen(theta[:6])) / max(1, fb - fa + 1))


def select(flights):
    chosen = []
    for fl in sorted(flights, key=lambda x: -x["density"]):
        if all(min(fl["fb"], k["fb"]) - max(fl["fa"], k["fa"]) + 1 <= OVERLAP_SEL
               for k in chosen):
            chosen.append(fl)
    return sorted(chosen, key=lambda x: x["fa"])


def track_of(ctx, chosen):
    P, t0 = ctx["P"], ctx["t0"]
    track = {}
    for fl in chosen:
        for f in range(fl["fa"], fl["fb"] + 1):
            q = arc_px(P, fl["theta"], fl["t_ref"], f, t0)
            track[f] = (float(q[0]), float(q[1]))
    return track


def boundaries(ctx, chosen):
    """(t, kind) at flight ends: 'contact' or 'bounce'."""
    t0 = ctx["t0"]
    out = []
    for i, fl in enumerate(chosen):
        ta = t0 + fl["fa"] / FPS
        tb = t0 + fl["fb"] / FPS
        za = c3.arc_pos(fl["theta"], [ta - fl["t_ref"]])[0]
        zb = c3.arc_pos(fl["theta"], [tb - fl["t_ref"]])[0]
        kind_b = "contact"
        if i + 1 < len(chosen):
            nx = chosen[i + 1]
            tn = t0 + nx["fa"] / FPS
            zn = c3.arc_pos(nx["theta"], [tn - nx["t_ref"]])[0]
            if (zb[2] <= BOUNCE_Z and tn - tb <= BOUNCE_DT
                    and np.hypot(*(zn[:2] - zb[:2])) <= BOUNCE_DXY):
                kind_b = "bounce"
        out.append((ta, "contact" if i == 0 or out[-1][1] != "bounce"
                    else "bounce-out"))
        out.append((tb, kind_b))
    return out


def run(ctx, p_seed, s_min, gap, _cache={}):
    key = (ctx["rally"], p_seed)
    if key not in _cache:
        _agree.P = ctx["P"]
        hyps, base = hypotheses(ctx, p_seed)
        _cache[key] = hyps
    hyps = _cache[key]
    kept = nms(hyps, s_min)
    flights = [g for g in (grow(ctx, h, gap) for h in kept) if g]
    chosen = select(flights)
    return dict(track=track_of(ctx, chosen), chosen=chosen,
                n_hyp=len(hyps), n_kept=len(kept), n_fl=len(flights))


# --------------------------------------------------------- grading

def displaced(track, rng):
    dx = rng.uniform(160, 240) * rng.choice([-1, 1])
    dy = rng.uniform(80, 140) * rng.choice([-1, 1])
    return {f: (x + dx, y + dy) for f, (x, y) in track.items()}


def timeshift(track, ctx, rng):
    if not track:
        return {}
    s = int(round(rng.uniform(2, 4) * FPS)) * rng.choice([-1, 1])
    lo, hi = ctx["f_lo"], ctx["f_hi"]
    n = hi - lo + 1
    return {lo + ((f - lo + s) % n): xy for f, xy in track.items()}


def contact_metrics(ctx, times, tag):
    imps = np.array(ctx["c"]["imps"], float)
    if not len(times) or not len(imps):
        print(f"    {tag}: no boundaries")
        return
    ts = np.array(times, float)
    dt = np.abs(imps[:, None] - ts[None, :]).min(axis=1)
    print(f"    {tag:14s} n={len(ts):3d}  |dt| to oracle contacts: median "
          f"{np.median(dt):.3f} s, within 0.10 s {np.mean(dt <= 0.10):.2f}"
          f" ({int((dt <= 0.10).sum())}/{len(imps)})")


def tune():
    ctxs = [context(r) for r in (6, 7)]
    for ctx in ctxs:
        print(f"rally {ctx['rally']}: {len(ctx['truth'])} clicks, decode@12 "
              f"{sum(ctx['dec'])}/{len(ctx['dec'])}, p-cache '{ctx['pxs']}'")
    inc_prec = INC_TRAIN["h12"] / INC_TRAIN["have"]
    print(f"INCUMBENT prod r6+r7: {INC_TRAIN['h12']} @ {inc_prec:.3f}")
    grid = []
    for p_seed in GRID_PSEED:
        for s_min in GRID_SMIN:
            for gap in GRID_GAP:
                tot = dict(h12=0, have=0, added=0)
                per = []
                for ctx in ctxs:
                    res = run(ctx, p_seed, s_min, gap)
                    h12, have, added = geom_fix.grade(res["track"], ctx["truth"],
                                                      ctx["t0"], ctx["dec"])
                    for k, v in zip(("h12", "have", "added"), (h12, have, added)):
                        tot[k] += v
                    per.append(f"r{ctx['rally']} {h12}/{have} "
                               f"(hyp {res['n_hyp']} kept {res['n_kept']} "
                               f"fl {res['n_fl']} sel {len(res['chosen'])})")
                prec = tot["h12"] / max(1, tot["have"])
                cell = dict(p_seed=p_seed, s_min=s_min, gap=gap, prec=prec, **tot)
                grid.append(cell)
                print(f"p_seed={p_seed} s_min={s_min:g} gap={gap:2d}  total r@12 "
                      f"{tot['h12']:4d}  prec@12 {prec:.3f}  ADDED {tot['added']}"
                      f"  | " + "  ".join(per), flush=True)
    ok = [g for g in grid if g["prec"] >= inc_prec and g["h12"] > INC_TRAIN["h12"]]
    rule = ("max total r@12 over r6+r7 s.t. pooled prec@12 >= incumbent prod "
            "(205 @ 0.623); ties larger s_min, smaller gap, larger p_seed; "
            "none -> dead")
    out = dict(incumbent=INC_TRAIN, grid=grid, rule=rule)
    if ok:
        best = sorted(ok, key=lambda g: (-g["h12"], -g["s_min"], g["gap"],
                                         -g["p_seed"]))[0]
        out.update(dead=False, p_seed=best["p_seed"], s_min=best["s_min"],
                   gap=best["gap"])
        print(f"VERDICT: p_seed={best['p_seed']} s_min={best['s_min']:g} "
              f"gap={best['gap']} ({best['h12']} @ {best['prec']:.3f}) — "
              f"freeze and one-shot r9/r10")
    else:
        out.update(dead=True)
        print("VERDICT: no cell beats the incumbent prod arm under the rule — "
              "path-first DEAD, do not run r9/r10")
    TUNE_JSON.write_text(json.dumps(out, indent=1))
    print("wrote", TUNE_JSON)


def run_grade(rally, cell):
    ctx = context(rally)
    truth, t0, dec = ctx["truth"], ctx["t0"], ctx["dec"]
    print(f"rally {rally}: {len(truth)} V/S clicks, decode@12 {sum(dec)}/"
          f"{len(dec)}; cell {cell} (p-cache '{ctx['pxs']}')")
    cdp.W_P_SOFT = 25.0
    for arm, cors in ctx["arms"]:
        inc = cdp.build_track(ctx["cc"], cors, t0, body=ctx["body"])
        cdp.score(inc, truth, t0, dec, f"inc-{arm}")
    res = run(ctx, cell["p_seed"], cell["s_min"], cell["gap"])
    tr = res["track"]
    print(f"  path-first: hyp {res['n_hyp']} kept {res['n_kept']} flights "
          f"{res['n_fl']} selected {len(res['chosen'])}")
    cdp.score(tr, truth, t0, dec, "path-first")
    for vis in ("V", "S"):
        tt = [x for x in truth if x[3] == vis]
        dd = [d for x, d in zip(truth, dec) if x[3] == vis]
        cdp.score(tr, tt, t0, dd, f"  pf[{vis}]")
    rng = np.random.default_rng(NULL_SEED)
    cdp.score(displaced(tr, rng), truth, t0, dec, "null-disp")
    cdp.score(timeshift(tr, ctx, rng), truth, t0, dec, "null-tshift")
    # strata on the incumbent prod geometry
    prod_cors = ctx["arms"][0][1]
    st = geom_fix.strata(ctx, prod_cors, ctx["cc"])
    inc = cdp.build_track(ctx["cc"], prod_cors, t0, body=ctx["body"])
    print("    stratum (incumbent prod geometry)   n   inc   pf")
    for s in ("cand", "nocand", "outwin", "nocor"):
        idx = [i for i, x in enumerate(st) if x == s]
        hi = sum(1 for i in idx if _hit(inc, truth[i], t0))
        hp = sum(1 for i in idx if _hit(tr, truth[i], t0))
        print(f"    {s:8s} {len(idx):4d} {hi:5d} {hp:5d}")
    # contacts
    bd = boundaries(ctx, res["chosen"])
    print(f"  boundaries: {len(bd)} ({sum(1 for _, k in bd if k == 'bounce')} bounces)")
    contact_metrics(ctx, [t for t, k in bd if k.startswith("contact")], "path-first")
    contact_metrics(ctx, prod_contacts(ctx["c"], ctx["series"], 0.5), "prod detector")
    print("-- flights: span | n w rms | density | launch")
    for fl in res["chosen"]:
        v0 = c3.arc_vel(fl["theta"], 0.0)
        sp = np.linalg.norm(v0)
        loft = np.degrees(np.arctan2(v0[2], np.hypot(v0[0], v0[1])))
        print(f"   {t0 + fl['fa'] / FPS:7.2f}-{t0 + fl['fb'] / FPS:7.2f} "
              f"{(fl['fb'] - fl['fa'] + 1) / FPS:4.2f}s | {fl['n']:3d} "
              f"{fl['w']:5.1f} {fl['rms']:4.1f} | {fl['density']:.2f} | "
              f"{sp:5.1f} ft/s loft {loft:5.1f} k {fl['theta'][6]:.2f}")


def _hit(track, x, t0):
    t, tx, ty, vis = x
    f = int(round((t - t0) * 60))
    p = track.get(f) or track.get(f - 1) or track.get(f + 1)
    return p is not None and np.hypot(p[0] - tx, p[1] - ty) <= cdp.R_MAIN


def selftest():
    """a planted drag-free arc through synthetic blobs must be found."""
    ctx = context(6)
    P, t0 = ctx["P"], ctx["t0"]
    rng = np.random.default_rng(1)
    theta = np.array([6.0, 30.0, 3.0, 2.0, -35.0, 12.0])
    fa, fb = 200, 260
    frames = {}
    for f in range(150, 320):
        junk = np.column_stack([rng.uniform(200, 1100, 20), rng.uniform(100, 650, 20),
                                rng.uniform(0.05, 0.6, 20)])
        if fa <= f <= fb:
            q = _proj6(P, theta, fa, [f])[0] + rng.normal(0, 1.5, 2)
            junk = np.vstack([junk, [q[0], q[1], 0.9]])
        frames[f] = (junk, np.zeros(len(junk), bool))
    ctx2 = dict(ctx, frames=frames, f_lo=150, f_hi=320)
    _agree.P = P
    res = run(ctx2, 0.6, 4.0, 6)
    tr = res["track"]
    truth = [(t0 + f / FPS, *_proj6(P, theta, fa, [f])[0], "V")
             for f in range(fa, fb + 1)]
    h = sum(1 for x in truth if _hit(tr, x, t0))
    print(f"selftest: hyps {res['n_hyp']} flights {res['n_fl']} selected "
          f"{len(res['chosen'])}; planted arc hit {h}/{len(truth)} frames")
    assert h >= 0.9 * len(truth), "planted arc not recovered"
    print("selftest OK")


def main():
    if sys.argv[1] == "tune":
        tune()
    elif sys.argv[1] == "selftest":
        selftest()
    elif sys.argv[1] == "grade":
        rally = int(sys.argv[2])
        over = {}
        args = sys.argv[3:]
        for i in range(0, len(args), 2):
            over[args[i].lstrip("-").replace("-", "_")] = float(args[i + 1])
        if rally in EVAL_RALLIES:
            if over:
                raise SystemExit("no knob overrides on the evaluation rallies")
            if not TUNE_JSON.exists() or json.loads(TUNE_JSON.read_text()).get("dead", True):
                raise SystemExit("no live tune verdict — refusing r9/r10")
        if TUNE_JSON.exists() and not json.loads(TUNE_JSON.read_text()).get("dead", True):
            v = json.loads(TUNE_JSON.read_text())
            cell = dict(p_seed=v["p_seed"], s_min=v["s_min"], gap=int(v["gap"]))
        else:
            cell = dict(p_seed=0.6, s_min=6.0, gap=6)
        for k, val in over.items():
            cell[k] = int(val) if k == "gap" else val
        run_grade(rally, cell)


if __name__ == "__main__":
    main()
