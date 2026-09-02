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
"""
import json
import sys
from pathlib import Path

import numpy as np

import pathfirst as pf
import corridor_dp as cdp
import geom_fix

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
