"""Hand-off seeding on top of the adopted path-first track (handoff_gate.md).

    python3 handoff.py tune            # 16-cell grid on r6/r7 -> handoff_tune.json
    python3 handoff.py grade <rally>   # the one-shot (r9/r10), refuses knobs

Pass 1 is pathfirst.run with its frozen cell. Pass 2 seeds short arcs
only inside hand-off zones: within W frames of a pass-1 flight end and
R_ZONE px of its last tracked point, where the near-body exclusion is
lifted for the first seed. Everything downstream (solve, plausibility,
support, nms, grow, select) is pathfirst's own code, unchanged.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import corridor_dp as cdp                                   # noqa: E402
import geom_fix                                             # noqa: E402
import pathfirst as pf                                      # noqa: E402

SP = Path(__file__).parent
TUNE_JSON = SP / "handoff_tune.json"
DS_HAND = (6, 8, 12)
GRID_RZONE = (40.0, 70.0)
GRID_W = (18, 30)
GRID_PHAND = (0.25, 0.4)
GRID_SMIN = (3.0, 4.0)
INC_TRAIN = dict(h12=263, prec=0.807)
INC_EVAL = {9: dict(h12=537, prec=0.87, f1=0.731), 10: dict(h12=422, prec=0.88, f1=0.675)}
FPS = pf.FPS


def zones(ctx, chosen, w, r_zone):
    """frame -> list of anchor points (x, y) whose zone covers that frame."""
    track = pf.track_of(ctx, chosen)
    z = {}
    for fl in chosen:
        for end, direction in ((fl["fb"], +1), (fl["fa"], -1)):
            anchor = track.get(end)
            if anchor is None:
                continue
            for k in range(1, w + 1):
                f = end + direction * k
                z.setdefault(f, []).append(anchor)
    return z, track


def hand_seeds(frames, zn, p_hand, r_zone):
    """first-seed candidates inside zones: near-body allowed, p >= p_hand."""
    out = {}
    for f, anchors in zn.items():
        fr = frames.get(f)
        if fr is None:
            continue
        arr, nb = fr
        a = np.asarray(anchors)
        d = np.hypot(arr[:, 0:1] - a[None, :, 0], arr[:, 1:2] - a[None, :, 1]).min(axis=1)
        m = (arr[:, 2] >= p_hand) & (d <= r_zone)
        if not m.any():
            continue
        s = arr[m]
        out[f] = s[np.argsort(-s[:, 2])[:pf.N_SEED], :2]
    return out


def hand_hypotheses(ctx, chosen, cell):
    P, frames, t0 = ctx["P"], ctx["frames"], ctx["t0"]
    base = pf.frame_base(frames)
    zn, _ = zones(ctx, chosen, cell["w"], cell["r_zone"])
    first = hand_seeds(frames, zn, cell["p_hand"], cell["r_zone"])
    general = pf.seeds(frames, cell["p_hand"])       # later seeds: ordinary rule
    later = dict(general)
    for f, s in first.items():                        # zone seeds may also be later seeds
        later[f] = np.vstack([later[f], s]) if f in later else s
    hyps = []
    for f1 in sorted(first):
        s1 = first[f1]
        for D in DS_HAND:
            for direction in (+1, -1):
                f3 = f1 + direction * D
                if f3 not in later:
                    continue
                s3 = later[f3]
                for off in pf.MID_OFF:
                    f2 = f1 + direction * (D // 2) + off
                    if f2 not in later or f2 == f1 or f2 == f3:
                        continue
                    s2 = later[f2]
                    fa_, fb_ = (f1, f3) if direction > 0 else (f3, f1)
                    sa, sb = (s1, s3) if direction > 0 else (s3, s1)
                    i1, i2, i3 = np.meshgrid(np.arange(len(sa)), np.arange(len(s2)),
                                             np.arange(len(sb)), indexing="ij")
                    pts = np.stack([sa[i1.ravel()], s2[i2.ravel()], sb[i3.ravel()]], axis=1)
                    taus = np.array([0.0, (f2 - fa_) / FPS, D / FPS])
                    th = pf.solve_arcs(P, pts, taus)
                    ok = pf.plausible(th, D / FPS)
                    if not ok.any():
                        continue
                    th = th[ok]
                    fa, fb = fa_ - D // 2, fb_ + D // 2
                    dt = (fa - fa_) / FPS
                    acc = np.array([0.0, 0.0, pf.G])
                    p0 = th[:, :3] + th[:, 3:6] * dt + 0.5 * acc * dt ** 2
                    v0 = th[:, 3:6] + acc * dt
                    th2 = np.hstack([p0, v0])
                    sup, nsup, _ = pf.support(P, th2, fa, fb, frames, base, t0)
                    for k in range(len(th2)):
                        hyps.append(dict(theta=th2[k], fa=fa, fb=fb, sup=float(sup[k]),
                                         nsup=int(nsup[k]), f1=fa_, f3=fb_))
    return hyps


def run(ctx, cell, _cache={}):
    """pass 1 (frozen) + pass 2 (hand-off). Returns dict(track, chosen, new, base)."""
    pcell = json.loads(pf.TUNE_JSON.read_text())
    assert not pcell.get("dead")
    key = ctx["rally"]
    if key not in _cache:
        _cache[key] = pf.run(ctx, pcell["p_seed"], pcell["s_min"], pcell["gap"])
    base = _cache[key]
    chosen = base["chosen"]
    pf._agree.P = ctx["P"]
    hyps = hand_hypotheses(ctx, chosen, cell)
    kept = pf.nms(hyps, cell["s_min_hand"])
    flights = [g for g in (pf.grow(ctx, h, pcell["gap"]) for h in kept) if g]
    free = [fl for fl in flights
            if all(min(fl["fb"], k["fb"]) - max(fl["fa"], k["fa"]) + 1 <= pf.OVERLAP_SEL
                   for k in chosen)]
    new = pf.select(free)
    allf = sorted(chosen + new, key=lambda x: x["fa"])
    return dict(track=pf.track_of(ctx, allf), chosen=allf, new=new, base=base,
                n_hyp=len(hyps), n_kept=len(kept))


def cells():
    out = []
    for r_zone in GRID_RZONE:
        for w in GRID_W:
            for p_hand in GRID_PHAND:
                for s_min_hand in GRID_SMIN:
                    out.append(dict(r_zone=r_zone, w=w, p_hand=p_hand, s_min_hand=s_min_hand))
    return out


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
            rng = np.random.default_rng(pf.NULL_SEED + r)
            d_, t_ = nulls(ctx, res["track"], rng)
            nd, nt = max(nd, d_), max(nt, t_)
            h12 += h
            have += hv
            per.append(f"r{r} {h}/{hv} +{len(res['new'])}fl")
        prec = h12 / max(1, have)
        rows.append((cell, h12, prec, nd, nt, per))
        print(f"  {cell}  r@12 {h12} prec {prec:.3f}  nulls {nd}/{nt}  {' '.join(per)}")
    ok = [x for x in rows if x[2] >= INC_TRAIN["prec"] - 0.03 and x[3] <= 3 and x[4] <= 3]
    ok.sort(key=lambda x: (-x[1], x[0]["r_zone"], x[0]["w"]))
    if not ok or ok[0][1] <= INC_TRAIN["h12"]:
        print(f"DEAD: no cell beats the incumbent {INC_TRAIN} under the rule")
        TUNE_JSON.write_text(json.dumps(dict(dead=True)))
        return
    best = ok[0]
    print(f"SELECTED {best[0]}  r@12 {best[1]} prec {best[2]:.3f} (incumbent {INC_TRAIN})")
    TUNE_JSON.write_text(json.dumps(dict(best[0], train_h12=best[1], train_prec=round(best[2], 3))))


def grade(rally):
    cell = json.loads(TUNE_JSON.read_text())
    assert not cell.get("dead"), "no live hand-off cell"
    cell = {k: cell[k] for k in ("r_zone", "w", "p_hand", "s_min_hand")}
    ctx = pf.context(rally)
    res = run(ctx, cell)
    truth, t0, dec = ctx["truth"], ctx["t0"], ctx["dec"]
    inc = INC_EVAL[rally]
    print(f"rally {rally}: cell {cell}; incumbent path-first {inc}")
    cdp.score(res["base"]["track"], truth, t0, dec, "path-first")
    cdp.score(res["track"], truth, t0, dec, "hand-off")
    for vis in "VS":
        tt = [x for x in truth if x[3] == vis]
        dd = [d for x, d in zip(truth, dec) if x[3] == vis]
        cdp.score(res["track"], tt, t0, dd, f"  ho[{vis}]")
    rng = np.random.default_rng(pf.NULL_SEED + rally)
    cdp.score(pf.displaced(res["track"], rng), truth, t0, dec, "null-disp")
    cdp.score(pf.timeshift(res["track"], ctx, rng), truth, t0, dec, "null-tshift")
    h, hv, _ = geom_fix.grade(res["track"], truth, t0, dec)
    prec = h / max(1, hv)
    nd, nt = nulls(ctx, res["track"], np.random.default_rng(pf.NULL_SEED + rally))
    # the events layer on the new flights
    import events as evm
    ec = json.loads((SP / "events_tune_v3.json").read_text())
    evs = evm.events(ctx, res["chosen"], ec["r_seam"], ec["a_seam"], ec["dt_pair"],
                     ec["off"], d_pair=ec["d_pair"])
    cont, bnc = evm.truth_events(ctx["c"])
    pr_ = evm.prf([e["t"] for e in evs], sorted(cont + bnc))
    rec, pr, f1 = pr_["recall"], pr_["precision"], pr_["f1"]
    print(f"  new flights {len(res['new'])} (hyp {res['n_hyp']}, kept {res['n_kept']}); "
          f"events on new track: n={len(evs)} recall {rec:.3f} prec {pr:.3f} F1 {f1:.3f} "
          f"(adopted {inc['f1']})")
    bars = [h > inc["h12"], prec >= inc["prec"] - 0.02, nd <= 3, nt <= 3, f1 >= inc["f1"] - 0.03]
    print(f"  BARS: r@12 {h} > {inc['h12']}: {bars[0]}; prec {prec:.3f} >= {inc['prec'] - 0.02:.2f}: "
          f"{bars[1]}; nulls {nd}/{nt} <= 3: {bars[2] and bars[3]}; events F1 {f1:.3f} >= "
          f"{inc['f1'] - 0.03:.3f}: {bars[4]}  =>  {'PASS' if all(bars) else 'FAIL'}")
    print("-- new flights: span | n w rms | density")
    for fl in res["new"]:
        print(f"    {t0 + fl['fa'] / FPS:8.2f}-{t0 + fl['fb'] / FPS:8.2f} {(fl['fb'] - fl['fa'] + 1) / FPS:.2f}s | "
              f"{fl['n']:3d} {fl['w']:5.1f} {fl['rms']:4.1f} | {fl['density']:.2f}")
    return all(bars)


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        tune()
    elif sys.argv[1] == "grade":
        assert len(sys.argv) == 3, "grade takes the rally only; knobs are frozen"
        grade(int(sys.argv[2]))
