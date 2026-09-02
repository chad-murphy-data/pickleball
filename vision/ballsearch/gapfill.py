"""Gap fill by arc extension on top of the adopted path-first track
(gapfill_gate.md).  For each gap between consecutive selected flights, the
two fitted arcs are extended toward each other and the fill switches from
A's arc to B's at the frame where they come closest; a gap whose arcs never
come within D_MEET px is left open.  Nothing is detected, no paddle is
used: the frames only exist through inference and the arcs carry it.

    python3 gapfill.py tune        # r6/r7 cross-fold grid -> gapfill_tune.json
    python3 gapfill.py grade 9     # v1 one shot vs the incumbent (r9/r10) — spent, NOT ADOPTED
    python3 gapfill.py tune2       # v2 (tagged inferred stratum) rule on r6/r7 -> gapfill_tune2.json
    python3 gapfill.py grade2 9    # v2 bars on r9/r10 (owner re-use of the eval rallies, disclosed)
    python3 gapfill.py selftest
    python3 gapfill.py tune3       # v3 HIT-ANCHORED fill of v2's open gaps, r6/r7 -> gapfill_tune3.json
    python3 gapfill.py grade3 9    # v3 bars on r9/r10 (gapfill_gate.md v3)
    python3 gapfill.py selftest3
"""
import json
import sys
from pathlib import Path

import numpy as np

import pathfirst as pf
import corridor_dp as cdp
import geom_fix
import court3d as c3                                        # noqa: E402 (path set by pathfirst)

SP = Path(__file__).parent
TUNE_JSON = SP / "gapfill_tune.json"
GRID_GAPMAX = (0.5, 0.8)
GRID_DMEET = (20.0, 40.0, float("inf"))
INC_TRAIN = dict(h12=263, prec=0.807)
INC_EVAL = {9: dict(h12=537, prec=0.87, f1=0.731), 10: dict(h12=422, prec=0.88, f1=0.675)}
FPS = pf.FPS


def fill(ctx, chosen, gap_max, d_meet):
    """(filled flights, track, inferred-frame set, per-gap rows)."""
    P, t0 = ctx["P"], ctx["t0"]
    fls = [dict(fl) for fl in sorted(chosen, key=lambda fl: fl["fa"])]
    inferred = set()
    rows = []
    for A, B in zip(fls, fls[1:]):
        g = B["fa"] - A["fb"] - 1
        if g < 1:
            continue
        row = dict(ta=t0 + A["fb"] / FPS, tb=t0 + B["fa"] / FPS, dur=g / FPS, filled=False)
        if g / FPS > gap_max:
            row["why"] = "long"
            rows.append(row)
            continue
        fs = np.arange(A["fb"], B["fa"] + 1)
        ea = np.array([pf.arc_px(P, A["theta"], A["t_ref"], f, t0) for f in fs], float)
        eb = np.array([pf.arc_px(P, B["theta"], B["t_ref"], f, t0) for f in fs], float)
        d = np.hypot(*(ea - eb).T)
        k = int(np.argmin(d))
        row["d_c"] = float(d[k])
        row["t_c"] = t0 + fs[k] / FPS
        if d[k] > d_meet:
            row["why"] = "no-meet"
            rows.append(row)
            continue
        f_c = int(fs[k])
        f_c = min(max(f_c, A["fb"]), B["fa"] - 1)     # A keeps >= its own end, B keeps >= 1 frame
        inferred.update(range(A["fb"] + 1, B["fa"]))
        A["fb"], B["fa"] = f_c, f_c + 1
        row["filled"] = True
        rows.append(row)
    track = pf.track_of(ctx, fls)
    return fls, track, inferred, rows


def run(ctx, cell):
    pc = json.loads(pf.TUNE_JSON.read_text())
    assert not pc.get("dead")
    base = pf.run(ctx, pc["p_seed"], pc["s_min"], pc["gap"])
    fls, track, inf, rows = fill(ctx, base["chosen"], cell["gap_max"], cell["d_meet"])
    return dict(base=base, chosen=fls, track=track, inferred=inf, rows=rows)


def cells():
    return [dict(gap_max=g, d_meet=d) for g in GRID_GAPMAX for d in GRID_DMEET]


def nulls(ctx, track, rng):
    h_d = geom_fix.grade(pf.displaced(track, rng), ctx["truth"], ctx["t0"], ctx["dec"])[0]
    h_t = geom_fix.grade(pf.timeshift(track, ctx, rng), ctx["truth"], ctx["t0"], ctx["dec"])[0]
    return h_d, h_t


def tune():
    ctxs = {r: pf.context(r) for r in (6, 7)}
    rows = []
    for cell in cells():
        h12 = have = 0
        nd = nt = 0
        per = []
        for r, ctx in ctxs.items():
            res = run(ctx, cell)
            h, hv, _ = geom_fix.grade(res["track"], ctx["truth"], ctx["t0"], ctx["dec"])
            d_, t_ = nulls(ctx, res["track"], np.random.default_rng(pf.NULL_SEED + r))
            nd, nt = max(nd, d_), max(nt, t_)
            h12 += h
            have += hv
            nf = sum(1 for x in res["rows"] if x["filled"])
            per.append(f"r{r} {h}/{hv} {nf}/{len(res['rows'])} gaps filled")
        prec = h12 / max(1, have)
        rows.append((cell, h12, prec, nd, nt, per))
        print(f"  {cell}  r@12 {h12} prec {prec:.3f}  nulls {nd}/{nt}  {' | '.join(per)}")
    ok = [x for x in rows if x[2] >= INC_TRAIN["prec"] - 0.03 and x[3] <= 3 and x[4] <= 3]
    ok.sort(key=lambda x: (-x[1], x[0]["gap_max"], x[0]["d_meet"]))
    if not ok or ok[0][1] <= INC_TRAIN["h12"]:
        print(f"DEAD: no cell beats the incumbent {INC_TRAIN} under the rule")
        TUNE_JSON.write_text(json.dumps(dict(dead=True)))
        return
    best = ok[0]
    print(f"SELECTED {best[0]}  r@12 {best[1]} prec {best[2]:.3f} (incumbent {INC_TRAIN})")
    out = dict(best[0], train_h12=best[1], train_prec=round(best[2], 3),
               rule="max pooled r@12 on r6+r7 s.t. prec >= inc-0.03 and nulls <= 3; ties smaller gap_max, smaller d_meet")
    if out["d_meet"] == float("inf"):
        out["d_meet"] = "inf"
    TUNE_JSON.write_text(json.dumps(out, indent=1))


def inferred_score(ctx, res):
    """(h12, have, prec, null_disp, null_tshift) on the inferred frames alone."""
    truth, t0, dec = ctx["truth"], ctx["t0"], ctx["dec"]
    inf = {f: xy for f, xy in res["track"].items() if f in res["inferred"]}
    h, hv, _ = geom_fix.grade(inf, truth, t0, dec)
    nd, nt = nulls(ctx, inf, np.random.default_rng(pf.NULL_SEED + ctx["rally"]))
    return h, hv, h / max(1, hv), nd, nt


def tune2():
    """v2 rule (gapfill_gate.md): max pooled inferred r@12 s.t. inferred prec >= 0.5, nulls <= 3."""
    ctxs = {r: pf.context(r) for r in (6, 7)}
    rows = []
    for cell in cells():
        h12 = have = 0
        nd = nt = 0
        per = []
        for r, ctx in ctxs.items():
            res = run(ctx, cell)
            h, hv, p, d_, t_ = inferred_score(ctx, res)
            nd, nt = max(nd, d_), max(nt, t_)
            h12 += h
            have += hv
            per.append(f"r{r} inferred {h}/{hv}")
        prec = h12 / max(1, have)
        rows.append((cell, h12, prec, nd, nt))
        print(f"  {cell}  inferred r@12 {h12} prec {prec:.3f}  nulls {nd}/{nt}  {' | '.join(per)}")
    ok = [x for x in rows if x[2] >= 0.5 and x[3] <= 3 and x[4] <= 3 and x[1] > 0]
    ok.sort(key=lambda x: (-x[1], x[0]["gap_max"], x[0]["d_meet"]))
    out = dict(rule="v2: max pooled inferred r@12 on r6+r7 s.t. inferred prec >= 0.5 and nulls <= 3; ties smaller gap_max, smaller d_meet")
    if not ok:
        print("no cell passes the v2 rule; falling back to the v1 cell")
        out.update(load_cell(), fallback=True)
    else:
        best = ok[0]
        print(f"SELECTED {best[0]}  inferred r@12 {best[1]} prec {best[2]:.3f}")
        out.update(best[0], train_inf_h12=best[1], train_inf_prec=round(best[2], 3))
    if out["d_meet"] == float("inf"):
        out["d_meet"] = "inf"
    (SP / "gapfill_tune2.json").write_text(json.dumps(out, indent=1))


def grade2(rally):
    cell = json.loads((SP / "gapfill_tune2.json").read_text())
    cell = dict(gap_max=float(cell["gap_max"]), d_meet=float(cell["d_meet"]))
    ctx = pf.context(rally)
    res = run(ctx, cell)
    truth, t0, dec = ctx["truth"], ctx["t0"], ctx["dec"]
    inc = INC_EVAL[rally]
    print(f"rally {rally} (v2 product): cell {cell}; incumbent path-first {inc}")
    base = res["base"]["track"]
    tracked = {f: xy for f, xy in res["track"].items() if f not in res["inferred"]}
    same = tracked == base
    h_tr = geom_fix.grade(tracked, truth, t0, dec)[0]
    cdp.score(tracked, truth, t0, dec, "tracked")
    cdp.score(res["track"], truth, t0, dec, "tracked+inf")
    h, hv, prec, nd, nt = inferred_score(ctx, res)
    inf = {f: xy for f, xy in res["track"].items() if f in res["inferred"]}
    cdp.score(inf, truth, t0, dec, "inferred")
    rng = np.random.default_rng(pf.NULL_SEED + rally)
    cdp.score(pf.displaced(inf, rng), truth, t0, dec, "inf-null-disp")
    cdp.score(pf.timeshift(inf, ctx, rng), truth, t0, dec, "inf-null-tsh")
    import events as evm
    ec = json.loads((SP / "events_tune_v3.json").read_text())
    evs = evm.events(ctx, res["chosen"], ec["r_seam"], ec["a_seam"], ec["dt_pair"],
                     ec["off"], d_pair=ec["d_pair"])
    cont, bnc = evm.truth_events(ctx["c"])
    pr_ = evm.prf([e["t"] for e in evs], sorted(cont + bnc))
    f1 = pr_["f1"]
    nf = sum(1 for x in res["rows"] if x["filled"])
    print(f"  gaps {len(res['rows'])}: filled {nf}; inferred frames {len(res['inferred'])}")
    print(f"  events v3 on filled flights: n={len(evs)} recall {pr_['recall']:.3f} prec {pr_['precision']:.3f} "
          f"F1 {f1:.3f} (adopted {inc['f1']})")
    bars = [same and h_tr == inc["h12"], prec >= 0.5, h >= 10, nd <= 3, nt <= 3, f1 >= inc["f1"] - 0.03]
    print(f"  BARS: tracked bit-identical & r@12 {h_tr} == {inc['h12']}: {bars[0]}; inferred prec {prec:.3f} >= 0.5: "
          f"{bars[1]}; inferred r@12 {h} >= 10: {bars[2]}; inferred nulls {nd}/{nt} <= 3: {bars[3] and bars[4]}; "
          f"events F1 {f1:.3f} >= {inc['f1'] - 0.03:.3f}: {bars[5]}  =>  {'PASS' if all(bars) else 'FAIL'}")
    return all(bars)


def product(ctx):
    """THE adopted tagged product (gapfill_gate.md v2): the incumbent path-first
    flights with the arc-extension fill applied, plus the set of inferred frames.
    Consumers must carry the tag (draw dashed, caption) — the tracked half is
    bit-identical to pathfirst.run's track."""
    cell = json.loads((SP / "gapfill_tune2.json").read_text())
    assert not cell.get("dead") and not cell.get("fallback")
    cell = dict(gap_max=float(cell["gap_max"]), d_meet=float(cell["d_meet"]))
    res = run(ctx, cell)
    return dict(chosen=res["chosen"], track=res["track"], inferred=res["inferred"],
                base=res["base"], rows=res["rows"], cell=cell)


def load_cell():
    cell = json.loads(TUNE_JSON.read_text())
    assert not cell.get("dead"), "no live gap-fill cell"
    return dict(gap_max=float(cell["gap_max"]), d_meet=float(cell["d_meet"]))


def grade(rally):
    cell = load_cell()
    ctx = pf.context(rally)
    res = run(ctx, cell)
    truth, t0, dec = ctx["truth"], ctx["t0"], ctx["dec"]
    inc = INC_EVAL[rally]
    print(f"rally {rally}: cell {cell}; incumbent path-first {inc}")
    cdp.score(res["base"]["track"], truth, t0, dec, "path-first")
    cdp.score(res["track"], truth, t0, dec, "gap-fill")
    for vis in "VS":
        tt = [x for x in truth if x[3] == vis]
        dd = [d for x, d in zip(truth, dec) if x[3] == vis]
        cdp.score(res["track"], tt, t0, dd, f"  gf[{vis}]")
    inf_track = {f: xy for f, xy in res["track"].items() if f in res["inferred"]}
    cdp.score(inf_track, truth, t0, dec, "  inferred")
    rng = np.random.default_rng(pf.NULL_SEED + rally)
    cdp.score(pf.displaced(res["track"], rng), truth, t0, dec, "null-disp")
    cdp.score(pf.timeshift(res["track"], ctx, rng), truth, t0, dec, "null-tshift")
    h, hv, _ = geom_fix.grade(res["track"], truth, t0, dec)
    prec = h / max(1, hv)
    nd, nt = nulls(ctx, res["track"], np.random.default_rng(pf.NULL_SEED + rally))
    import events as evm
    ec = json.loads((SP / "events_tune_v3.json").read_text())
    evs = evm.events(ctx, res["chosen"], ec["r_seam"], ec["a_seam"], ec["dt_pair"],
                     ec["off"], d_pair=ec["d_pair"])
    cont, bnc = evm.truth_events(ctx["c"])
    pr_ = evm.prf([e["t"] for e in evs], sorted(cont + bnc))
    rec, pr, f1 = pr_["recall"], pr_["precision"], pr_["f1"]
    nf = sum(1 for x in res["rows"] if x["filled"])
    print(f"  gaps {len(res['rows'])}: filled {nf}, long {sum(1 for x in res['rows'] if x.get('why') == 'long')}, "
          f"no-meet {sum(1 for x in res['rows'] if x.get('why') == 'no-meet')}; inferred frames {len(res['inferred'])}")
    print(f"  events v3 on filled flights: n={len(evs)} recall {rec:.3f} prec {pr:.3f} F1 {f1:.3f} "
          f"(adopted {inc['f1']})")
    bars = [h > inc["h12"], prec >= inc["prec"] - 0.02, nd <= 3, nt <= 3, f1 >= inc["f1"] - 0.03]
    print(f"  BARS: r@12 {h} > {inc['h12']}: {bars[0]}; prec {prec:.3f} >= {inc['prec'] - 0.02:.2f}: "
          f"{bars[1]}; nulls {nd}/{nt} <= 3: {bars[2] and bars[3]}; events F1 {f1:.3f} >= "
          f"{inc['f1'] - 0.03:.3f}: {bars[4]}  =>  {'PASS' if all(bars) else 'FAIL'}")
    print("-- gaps: start end dur | d_c t_c | verdict")
    for x in res["rows"]:
        dc = f"{x['d_c']:5.1f} {x['t_c']:7.2f}" if "d_c" in x else "  ---    ---  "
        print(f"   {x['ta']:7.2f} {x['tb']:7.2f} {x['dur']:4.2f} | {dc} | "
              f"{'filled' if x['filled'] else x['why']}")
    return all(bars)


# ------------------------------------------------------------------ v3
# HIT-ANCHORED fill of the gaps v2 leaves open (gapfill_gate.md v3 -> v3b, owner
# go 2026-09-02; v3's point anchor died on train, v3b anchors a REGION and
# refits the two flights jointly to meet at the contact).  Anchors are PRODUCTION: contact times from the r6/r7
# approach detector (corridor_lab.prod_contacts), hitter = nearest paddle
# proxy, depth = the hitter's floor position.  The owner's imps are never
# read here.  A/B keep their arcs on their own frames; the fill is made of
# extra "pieces" (flights that cover only gap frames), tagged inferred3.
TOL3 = 0.10                 # s: a detected contact this close to the gap anchors it
RMS_MAX3 = 3.0              # px: an anchored refit must still reproduce its own pixels
W_DEPTH3 = 1.0              # px/ft on the anchor's depth (the pixel weight is the grid)
Z_PAD = (0.5, 9.0)          # ft: paddle range the lifted anchor is clamped to
FLOOR_EPS = 0.05            # s: floor search may start just before the gap
R_ANC = 2.0                 # ft: the ball at contact is within a paddle length of the proxy (v3b)
W_MEET = 2.0                # px per px: A' and B' must meet at the contact time (v3b, fixed)
GRID_WANC = (2.0, 6.0)
GRID_BOUNCE = (False, True)
TUNE3_JSON = SP / "gapfill_tune3.json"


def floor_tracks(ctx):
    """{tid: (t[], x[], y[])}: floor position per pose track (ankle midpoint
    through the z=0 homography, else box bottom), 5-sample median."""
    import rally_stats as rs
    z = np.load(ctx["c"]["npz"])
    P = ctx["P"]
    out = {}
    for tid in ctx["series"]:
        m = np.where(z["track"] == tid)[0]
        tt, kpt, kpc, box = z["t"][m], z["kpt"][m], z["kpc"][m], z["box"][m]
        rows = []
        for i in range(len(m)):
            if kpc[i, rs.LANK] >= rs.KP_CONF and kpc[i, rs.RANK] >= rs.KP_CONF:
                uv = (kpt[i, [rs.LANK, rs.RANK], 0].mean(), kpt[i, [rs.LANK, rs.RANK], 1].mean())
            else:
                uv = ((box[i, 0] + box[i, 2]) / 2, box[i, 3])
            xy = rs.ground_point(P, uv)
            rows.append((float(tt[i]), float(xy[0]), float(xy[1])))
        rows.sort()
        arr = np.array(rows)
        if len(arr) >= 5:
            for k in (1, 2):
                arr[:, k] = np.array([np.median(arr[max(0, i - 2):i + 3, k]) for i in range(len(arr))])
        out[tid] = (arr[:, 0], arr[:, 1], arr[:, 2])
    return out


def lift(P, uv, y, zlo=Z_PAD[0], zhi=Z_PAD[1]):
    """image (u,v) at a known depth y -> (x, y, z); z clamped to the paddle
    range with x re-solved from u."""
    a = P[0] - uv[0] * P[2]
    b = P[1] - uv[1] * P[2]
    M = np.array([[a[0], a[2]], [b[0], b[2]]])
    r = -np.array([a[1] * y + a[3], b[1] * y + b[3]])
    x, z = np.linalg.solve(M, r)
    if not zlo <= z <= zhi:
        z = min(max(z, zlo), zhi)
        x = -(a[1] * y + a[2] * z + a[3]) / a[0]
    return np.array([x, y, z], float)


def prod_times(ctx):
    if "_prod" not in ctx:
        ctx["_prod"] = sorted(float(t) for t in pf.prod_contacts(ctx["c"], ctx["series"], 0.5))
    return ctx["_prod"]


def anchors3(ctx, A, B, floors, ta, tb):
    """production anchors inside the gap ±TOL3: dicts(t, uv, X, tid), time-sorted,
    t clipped to the gap, anchors closer than 3 frames merged."""
    from claim_lab import paddle_at
    P, t0, series = ctx["P"], ctx["t0"], ctx["series"]
    pa = np.asarray(pf.arc_px(P, A["theta"], A["t_ref"], A["fb"], t0), float)
    pb = np.asarray(pf.arc_px(P, B["theta"], B["t_ref"], B["fa"], t0), float)
    mid = (pa + pb) / 2
    out = []
    for t in prod_times(ctx):
        if not (ta - TOL3 <= t <= tb + TOL3):
            continue
        best = None
        for tid in series:
            pd = paddle_at(series, tid, t, tol=0.12)
            if pd is None:
                continue
            d = float(np.hypot(pd[0] - mid[0], pd[1] - mid[1]))
            if best is None or d < best[0]:
                best = (d, tid, pd)
        if best is None:
            continue
        _, tid, pd = best
        tt, xx, yy = floors[tid]
        i = int(np.abs(tt - t).argmin())
        if abs(tt[i] - t) > 0.3:
            continue
        X = lift(P, (float(pd[0]), float(pd[1])), float(yy[i]))
        out.append(dict(t=min(max(t, ta), tb), uv=np.array([pd[0], pd[1]], float), X=X, tid=int(tid)))
    out.sort(key=lambda a: a["t"])
    ded = []
    for a in out:
        if ded and a["t"] - ded[-1]["t"] < 3 / FPS:
            continue
        ded.append(a)
    return ded


def _lm(resfn, th0, iters=60):
    """Levenberg–Marquardt with a numeric Jacobian (court3d.fit_arc's loop on any residual)."""
    th = np.array(th0, float)
    lam = 1e-3
    r = resfn(th)
    n = len(th)
    for _ in range(iters):
        J = np.empty((len(r), n))
        for j in range(n):
            d = np.zeros(n)
            d[j] = 1e-4
            J[:, j] = (resfn(th + d) - r) / 1e-4
        H = J.T @ J + lam * np.eye(n)
        step = np.linalg.solve(H, -J.T @ r)
        r2 = resfn(th + step)
        if r2 @ r2 < r @ r:
            th, r, lam = th + step, r2, max(lam * 0.5, 1e-6)
            if np.linalg.norm(step) < 1e-8:
                break
        else:
            lam *= 4
            if lam > 1e6:
                break
    return th, r


def own_obs(P, fl, t0, skip):
    return [(t0 + f / FPS, *map(float, pf.arc_px(P, fl["theta"], fl["t_ref"], f, t0)))
            for f in range(fl["fa"], fl["fb"] + 1) if f not in skip]


def px_per_ft(P, X):
    a = c3.project(P, X[None, :])[0]
    b = c3.project(P, (X + np.array([1.0, 0, 0]))[None, :])[0]
    return max(1e-6, float(np.linalg.norm(b - a)))


def _pix_res(P, th, obs, t_ref):
    tau = np.array([o[0] for o in obs]) - t_ref
    px = c3.project(P, c3.arc_pos(th, tau))
    return (px - np.array([[o[1], o[2]] for o in obs])).ravel()


def _anchor_res(P, th, t_ref, anc, w_anc):
    """region hinge (R_ANC ft around the paddle proxy) + depth, at the contact time."""
    X = c3.arc_pos(th, [anc["t"] - t_ref])[0]
    uv = c3.project(P, X[None, :])[0]
    r_px = R_ANC * px_per_ft(P, anc["X"])
    d = float(np.linalg.norm(uv - anc["uv"]))
    return np.array([w_anc * max(0.0, d - r_px), W_DEPTH3 * (X[1] - anc["X"][1])])


def refit_one(P, fl, anc, t0, w_anc, skip):
    """fl refit to its own pixels + region/depth anchor -> theta, or None (rms > RMS_MAX3)."""
    obs = own_obs(P, fl, t0, skip)
    if len(obs) < 4:
        return None

    def res(th):
        return np.concatenate([_pix_res(P, th, obs, fl["t_ref"]), [2.0 * abs(th[6])],
                               _anchor_res(P, th, fl["t_ref"], anc, w_anc)])

    th, r = _lm(res, fl["theta"])
    rms = float(np.sqrt(np.mean(r[:2 * len(obs)] ** 2)))
    return th if rms <= RMS_MAX3 else None


def refit_pair(P, A, B, anc, t0, w_anc, skip):
    """A and B refit JOINTLY: own pixels each + meet at t_c (W_MEET) + region/depth
    on both -> (thA, thB) or None when either no longer reproduces its pixels."""
    oa, ob = own_obs(P, A, t0, skip), own_obs(P, B, t0, skip)
    if len(oa) < 4 or len(ob) < 4:
        return None
    nA = len(A["theta"])

    def res(th):
        ta_, tb_ = th[:nA], th[nA:]
        XA = c3.arc_pos(ta_, [anc["t"] - A["t_ref"]])[0]
        XB = c3.arc_pos(tb_, [anc["t"] - B["t_ref"]])[0]
        meet = c3.project(P, XA[None, :])[0] - c3.project(P, XB[None, :])[0]
        return np.concatenate([_pix_res(P, ta_, oa, A["t_ref"]), _pix_res(P, tb_, ob, B["t_ref"]),
                               [2.0 * abs(ta_[6]), 2.0 * abs(tb_[6])], W_MEET * meet,
                               _anchor_res(P, ta_, A["t_ref"], anc, w_anc),
                               _anchor_res(P, tb_, B["t_ref"], anc, w_anc)])

    th, r = _lm(res, np.concatenate([A["theta"], B["theta"]]))
    ra = float(np.sqrt(np.mean(r[:2 * len(oa)] ** 2)))
    rb = float(np.sqrt(np.mean(r[2 * len(oa):2 * (len(oa) + len(ob))] ** 2)))
    if ra > RMS_MAX3 or rb > RMS_MAX3:
        return None
    return th[:nA], th[nA:]


def floor_time(fl, t_lo, t_hi, forward=True):
    """forward: first time in (t_lo, t_hi) the arc goes DOWN through z=0;
    backward (B's side): last time it comes UP through z=0."""
    ts = np.arange(t_lo, t_hi, 1 / 240.0)
    if len(ts) < 2:
        return None
    z = c3.arc_pos(fl["theta"], ts - fl["t_ref"])[:, 2]
    hits = [i for i in range(1, len(ts))
            if ((z[i - 1] > 0 >= z[i]) if forward else (z[i - 1] <= 0 < z[i]))]
    if not hits:
        return None
    return float(ts[hits[0]]) if forward else float(ts[hits[-1]])


def bvp(X0, ta, X1, tb):
    """drag-free arc through X0 at ta and X1 at tb: 6-param theta, t_ref = ta."""
    T = tb - ta
    v0 = (np.asarray(X1, float) - np.asarray(X0, float) - np.array([0.0, 0.0, 0.5 * c3.G * T * T])) / T
    return np.concatenate([np.asarray(X0, float), v0])


def piece(parent, theta, t_ref, fa, fb):
    d = dict(parent)
    d.update(theta=np.asarray(theta, float), t_ref=float(t_ref), fa=int(fa), fb=int(fb), n=0, piece=True)
    return d


def fill3(ctx, fls2, inferred2, w_anc, bounce):
    """(flights incl. pieces, track, inferred3 frame set, per-gap rows)."""
    P, t0 = ctx["P"], ctx["t0"]
    if "_floors" not in ctx:
        ctx["_floors"] = floor_tracks(ctx)
    floors = ctx["_floors"]
    fls = [dict(f) for f in sorted(fls2, key=lambda f: f["fa"])]
    out, inf3, rows = [], set(), []
    fr = lambda t: int(round((t - t0) * FPS))
    for i, A in enumerate(fls):
        out.append(A)
        if i + 1 >= len(fls):
            break
        B = fls[i + 1]
        g = B["fa"] - A["fb"] - 1
        if g < 1:
            continue
        ta, tb = t0 + A["fb"] / FPS, t0 + B["fa"] / FPS
        row = dict(ta=ta, tb=tb, dur=g / FPS, filled=0, anchors=0, why="", bounces=0)
        anc = anchors3(ctx, A, B, floors, ta, tb)
        row["anchors"] = len(anc)
        if not anc:
            row["why"] = "no-anchor"
            rows.append(row)
            continue
        pieces = []
        a1, ak = anc[0], anc[-1]
        f_c1 = min(max(fr(a1["t"]), A["fb"]), B["fa"] - 1)
        f_ck = min(max(fr(ak["t"]), A["fb"]), B["fa"] - 1)
        thA = thB = None
        if len(anc) == 1:
            pr = refit_pair(P, A, B, a1, t0, w_anc, inferred2)
            if pr is None:
                row["why"] += "pair-rms "
            else:
                thA, thB = pr
                XcA = c3.arc_pos(thA, [a1["t"] - A["t_ref"]])[0]
                XcB = c3.arc_pos(thB, [a1["t"] - B["t_ref"]])[0]
                a1["Xc"] = ak["Xc"] = (XcA + XcB) / 2
        else:
            thA = refit_one(P, A, a1, t0, w_anc, inferred2)
            thB = refit_one(P, B, ak, t0, w_anc, inferred2)
            if thA is None:
                row["why"] += "A-rms "
            else:
                a1["Xc"] = c3.arc_pos(thA, [a1["t"] - A["t_ref"]])[0]
            if thB is None:
                row["why"] += "B-rms "
            else:
                ak["Xc"] = c3.arc_pos(thB, [ak["t"] - B["t_ref"]])[0]
        # A side: A.fb+1 .. f_c1
        if thA is not None and f_c1 > A["fb"]:
            tbn = floor_time(A, ta - FLOOR_EPS, a1["t"], True) if bounce else None
            if tbn is not None and a1["t"] - tbn >= 2 / FPS:
                fbn = min(max(fr(tbn), A["fb"]), f_c1 - 1)
                Xb = c3.arc_pos(A["theta"], [tbn - A["t_ref"]])[0].copy()
                Xb[2] = 0.0
                if fbn > A["fb"]:
                    pieces.append(piece(A, A["theta"], A["t_ref"], A["fb"] + 1, fbn))
                pieces.append(piece(A, bvp(Xb, tbn, a1["Xc"], a1["t"]), tbn, fbn + 1, f_c1))
                row["bounces"] += 1
            else:
                pieces.append(piece(A, thA, A["t_ref"], A["fb"] + 1, f_c1))
        # between the first and last anchor: volley-to-volley BVP only (stays above the floor)
        if len(anc) >= 2 and thA is not None and thB is not None:
            fa_, fb_ = max(f_c1 + 1, A["fb"] + 1), min(f_ck, B["fa"] - 1)
            if fb_ - fa_ + 1 >= 3:
                th = bvp(a1["Xc"], a1["t"], ak["Xc"], ak["t"])
                zz = c3.arc_pos(th, np.linspace(0, ak["t"] - a1["t"], 20))[:, 2]
                if zz.min() >= 0.0:
                    pieces.append(piece(A, th, a1["t"], fa_, fb_))
                else:
                    row["why"] += "mid-bounce "
        # B side: f_ck+1 .. B.fa-1
        if thB is not None and f_ck + 1 <= B["fa"] - 1:
            tbn = floor_time(B, ak["t"], tb + FLOOR_EPS, False) if bounce else None
            if tbn is not None and tbn - ak["t"] >= 2 / FPS:
                fbn = min(max(fr(tbn), f_ck + 1), B["fa"] - 1)
                Xb = c3.arc_pos(B["theta"], [tbn - B["t_ref"]])[0].copy()
                Xb[2] = 0.0
                pieces.append(piece(B, bvp(ak["Xc"], ak["t"], Xb, tbn), ak["t"], f_ck + 1, fbn))
                if fbn + 1 <= B["fa"] - 1:
                    pieces.append(piece(B, B["theta"], B["t_ref"], fbn + 1, B["fa"] - 1))
                row["bounces"] += 1
            else:
                pieces.append(piece(B, thB, B["t_ref"], f_ck + 1, B["fa"] - 1))
        used = set()
        for pc in pieces:
            frames = set(range(pc["fa"], pc["fb"] + 1))
            assert pc["fa"] > A["fb"] and pc["fb"] < B["fa"] and not (frames & used), (pc["fa"], pc["fb"])
            if pc["fb"] >= pc["fa"]:
                out.append(pc)
                used |= frames
                row["filled"] += len(frames)
        inf3 |= used
        rows.append(row)
    out.sort(key=lambda f: f["fa"])
    return out, pf.track_of(ctx, out), inf3, rows


def cells3():
    return [dict(w_anc=w, bounce=b) for w in GRID_WANC for b in GRID_BOUNCE]


def run3(ctx, cell):
    res2 = product(ctx)
    fls, track, inf3, rows = fill3(ctx, res2["chosen"], res2["inferred"], cell["w_anc"], cell["bounce"])
    return dict(v2=res2, chosen=fls, track=track, inferred=res2["inferred"], inferred3=inf3, rows=rows)


def inferred3_score(ctx, res):
    truth, t0, dec = ctx["truth"], ctx["t0"], ctx["dec"]
    inf = {f: xy for f, xy in res["track"].items() if f in res["inferred3"]}
    h, hv, _ = geom_fix.grade(inf, truth, t0, dec)
    nd, nt = nulls(ctx, inf, np.random.default_rng(pf.NULL_SEED + ctx["rally"]))
    return h, hv, h / max(1, hv), nd, nt


def tune3():
    ctxs = {r: pf.context(r) for r in (6, 7)}
    rows = []
    for cell in cells3():
        h12 = have = 0
        nd = nt = 0
        per = []
        for r, ctx in ctxs.items():
            res = run3(ctx, cell)
            h, hv, p, d_, t_ = inferred3_score(ctx, res)
            nd, nt = max(nd, d_), max(nt, t_)
            h12 += h
            have += hv
            per.append(f"r{r} inferred3 {h}/{hv} ({len(res['inferred3'])} frames, "
                       f"{sum(x['bounces'] for x in res['rows'])} bounces)")
        prec = h12 / max(1, have)
        rows.append((cell, h12, prec, nd, nt))
        print(f"  {cell}  inferred3 r@12 {h12} prec {prec:.3f}  nulls {nd}/{nt}  {' | '.join(per)}")
    ok = [x for x in rows if x[2] >= 0.5 and x[3] <= 3 and x[4] <= 3 and x[1] > 0]
    ok.sort(key=lambda x: (-x[1], x[0]["w_anc"], x[0]["bounce"]))
    out = dict(rule="v3: max pooled inferred3 r@12 on r6+r7 s.t. inferred3 prec >= 0.5 and nulls <= 3; ties smaller w_anc, then bounce off")
    if not ok:
        print("DEAD: no cell passes the v3 rule")
        out["dead"] = True
    else:
        best = ok[0]
        print(f"SELECTED {best[0]}  inferred3 r@12 {best[1]} prec {best[2]:.3f}")
        out.update(best[0], train_inf3_h12=best[1], train_inf3_prec=round(best[2], 3))
    TUNE3_JSON.write_text(json.dumps(out, indent=1))


def load_cell3():
    cell = json.loads(TUNE3_JSON.read_text())
    assert not cell.get("dead"), "no live v3 cell"
    return dict(w_anc=float(cell["w_anc"]), bounce=bool(cell["bounce"]))


def grade3(rally):
    cell = load_cell3()
    ctx = pf.context(rally)
    res = run3(ctx, cell)
    truth, t0, dec = ctx["truth"], ctx["t0"], ctx["dec"]
    inc = INC_EVAL[rally]
    print(f"rally {rally} (v3 hit-anchored fill on the v2 product): cell {cell}")
    v2 = res["v2"]["track"]
    keep = {f: xy for f, xy in res["track"].items() if f not in res["inferred3"]}
    same = keep == v2
    cdp.score(v2, truth, t0, dec, "v2 product")
    cdp.score(res["track"], truth, t0, dec, "v2+inferred3")
    h, hv, prec, nd, nt = inferred3_score(ctx, res)
    inf = {f: xy for f, xy in res["track"].items() if f in res["inferred3"]}
    cdp.score(inf, truth, t0, dec, "inferred3")
    rng = np.random.default_rng(pf.NULL_SEED + rally)
    cdp.score(pf.displaced(inf, rng), truth, t0, dec, "inf3-null-disp")
    cdp.score(pf.timeshift(inf, ctx, rng), truth, t0, dec, "inf3-null-tsh")
    import events as evm
    ec = json.loads((SP / "events_tune_v3.json").read_text())
    evs = evm.events(ctx, res["chosen"], ec["r_seam"], ec["a_seam"], ec["dt_pair"],
                     ec["off"], d_pair=ec["d_pair"])
    cont, bnc = evm.truth_events(ctx["c"])
    pr_ = evm.prf([e["t"] for e in evs], sorted(cont + bnc))
    f1 = pr_["f1"]
    print(f"  open gaps {len(res['rows'])}: anchored {sum(1 for x in res['rows'] if x['anchors'])}, "
          f"frames filled {sum(x['filled'] for x in res['rows'])}, bounces used {sum(x['bounces'] for x in res['rows'])}")
    print(f"  events v3 on v3-filled flights: n={len(evs)} recall {pr_['recall']:.3f} prec {pr_['precision']:.3f} "
          f"F1 {f1:.3f} (adopted {inc['f1']})")
    bars = [same, prec >= 0.5, h >= 10, nd <= 3, nt <= 3, f1 >= inc["f1"] - 0.03]
    print(f"  BARS: v2 frames bit-identical: {bars[0]}; inferred3 prec {prec:.3f} >= 0.5: {bars[1]}; "
          f"inferred3 r@12 {h} >= 10: {bars[2]}; inferred3 nulls {nd}/{nt} <= 3: {bars[3] and bars[4]}; "
          f"events F1 {f1:.3f} >= {inc['f1'] - 0.03:.3f}: {bars[5]}  =>  {'PASS' if all(bars) else 'FAIL'}")
    print("-- open gaps: start end dur | anchors filled bounces | note")
    for x in res["rows"]:
        print(f"   {x['ta'] - t0:6.2f} {x['tb'] - t0:6.2f} {x['dur']:4.2f} | {x['anchors']} {x['filled']:3d} {x['bounces']} | {x['why']}")
    return all(bars)


def product3(ctx):
    """v2 product + the hit-anchored fill (gapfill_gate.md v3).  Only a consumer
    product once the gate records ADOPTED; returns v2's keys plus inferred3."""
    cell = json.loads(TUNE3_JSON.read_text())
    assert cell.get("adopted"), "gapfill v3 is not adopted"
    res = run3(ctx, dict(w_anc=float(cell["w_anc"]), bounce=bool(cell["bounce"])))
    v2 = res["v2"]
    return dict(chosen=res["chosen"], track=res["track"], inferred=res["inferred"],
                inferred3=res["inferred3"], base=v2["base"], rows=v2["rows"], rows3=res["rows"],
                cell=v2["cell"], cell3=cell)


def selftest3():
    """bvp hits both endpoints; lift round-trips; fill3 keeps v2 frames bit-identical,
    fills only gap frames, pieces disjoint."""
    X0, X1 = np.array([3.0, 10.0, 2.0]), np.array([15.0, 30.0, 4.0])
    th = bvp(X0, 1.0, X1, 1.4)
    assert np.allclose(c3.arc_pos(th, [0.0])[0], X0) and np.allclose(c3.arc_pos(th, [0.4])[0], X1)
    ctx = pf.context(6)
    P = ctx["P"]
    X = np.array([7.0, 30.0, 3.5])
    uv = c3.project(P, X[None, :])[0]
    assert np.allclose(lift(P, uv, 30.0), X, atol=1e-6)
    res2 = product(ctx)
    for cell in cells3():
        fls, track, inf3, rows = fill3(ctx, res2["chosen"], res2["inferred"], cell["w_anc"], cell["bounce"])
        for f, xy in res2["track"].items():
            assert track[f] == xy, f
        assert not (inf3 & set(res2["track"])), "inferred3 overlaps v2 frames"
        assert all(f in track for f in inf3)
        print(f"  {cell}: open gaps {len(rows)}, anchored {sum(1 for r in rows if r['anchors'])}, "
              f"inferred3 frames {len(inf3)}, bounces {sum(r['bounces'] for r in rows)}")
    print("selftest3 ok")


def selftest():
    """two synthetic arcs meeting at a known frame: the fill must switch there."""
    ctx = pf.context(6)
    P, t0 = ctx["P"], ctx["t0"]
    pc = json.loads(pf.TUNE_JSON.read_text())
    base = pf.run(ctx, pc["p_seed"], pc["s_min"], pc["gap"])
    ch = sorted(base["chosen"], key=lambda fl: fl["fa"])
    assert len(ch) >= 2
    fls, track, inf, rows = fill(ctx, ch, 10.0, float("inf"))
    # every gap short enough is filled, every filled gap's frames are in the track and tagged
    for A, B in zip(fls, fls[1:]):
        assert B["fa"] - A["fb"] <= 1 or A["fb"] < B["fa"]
    for f in inf:
        assert f in track
    # original frames unchanged
    for f, xy in base["track"].items():
        assert track[f] == xy, f
    # a closed grid leaves everything open
    fls2, track2, inf2, rows2 = fill(ctx, ch, 10.0, 0.0)
    assert not inf2 and track2 == base["track"]
    print(f"selftest ok: {sum(r['filled'] for r in rows)}/{len(rows)} gaps filled, "
          f"{len(inf)} inferred frames, base track preserved, d_meet=0 is a no-op")


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        tune()
    elif sys.argv[1] == "grade":
        r = int(sys.argv[2])
        if r in INC_EVAL and len(sys.argv) > 3:
            sys.exit("no overrides on r9/r10")
        grade(r)
    elif sys.argv[1] == "tune2":
        tune2()
    elif sys.argv[1] == "grade2":
        grade2(int(sys.argv[2]))
    elif sys.argv[1] == "selftest":
        selftest()
    elif sys.argv[1] == "tune3":
        tune3()
    elif sys.argv[1] == "grade3":
        grade3(int(sys.argv[2]))
    elif sys.argv[1] == "selftest3":
        selftest3()
