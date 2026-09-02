"""Events (one per change of flight) from the adopted path-first flights.

    python3 events.py tune                 # r6/r7 grid under events_gate.md (writes events_tune.json)
    python3 events.py grade <rally>        # one shot vs RAW / PROD baselines + null (refuses r9/r10 without a live verdict)
    python3 events.py grade 6 --r-seam 16 --a-seam 40 --dt-pair 0.4 --off 0   # train rally, overrides allowed

Track untouched: flights come from pathfirst.run with the frozen cell;
this layer only decides which flight ends are events and when.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import court3d as c3                                        # noqa: E402
import pathfirst as pf                                      # noqa: E402
from corridor_lab import prod_contacts                      # noqa: E402

SP = Path(__file__).parent
TUNE_JSON = SP / "events_tune.json"
FPS = pf.FPS
DT_SEAM = 0.12
TOL = 0.10
BOUNCE_Z = 0.3
NULL_SEED = 20260902
N_NULL = 200
GRID_RSEAM = (8.0, 16.0, 30.0)
GRID_ASEAM = (20.0, 40.0)
GRID_DTPAIR = (0.25, 0.40)
GRID_OFF = (0.0, 0.06)
ADOPTED = {9: 537, 10: 422}       # path-first r@12 on record; asserted in grade


# --------------------------------------------------------- flights

def flights(ctx):
    cell = json.loads(pf.TUNE_JSON.read_text())
    assert not cell.get("dead"), "no live path-first verdict"
    res = pf.run(ctx, cell["p_seed"], cell["s_min"], cell["gap"])
    return res


def proj(P, fl, t, t0):
    """image-plane position of flight fl's arc at absolute time t."""
    X = c3.arc_pos(fl["theta"], [t - fl["t_ref"]])
    return c3.project(P, X)[0]


def vel_px(P, fl, t, t0, e=1 / 120.0):
    a, b = proj(P, fl, t - e, t0), proj(P, fl, t + e, t0)
    return (b - a) / (2 * e)


def seam_stats(ctx, A, B):
    P, t0 = ctx["P"], ctx["t0"]
    tA1 = t0 + A["fb"] / FPS
    tB0 = t0 + B["fa"] / FPS
    dt = tB0 - tA1
    eA = np.linalg.norm(proj(P, A, tB0, t0) - proj(P, B, tB0, t0))
    eB = np.linalg.norm(proj(P, B, tA1, t0) - proj(P, A, tA1, t0))
    vA, vB = vel_px(P, A, tA1, t0), vel_px(P, B, tB0, t0)
    cosang = float(np.dot(vA, vB) / max(1e-9, np.linalg.norm(vA) * np.linalg.norm(vB)))
    ang = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    return dt, float((eA + eB) / 2), ang, tA1, tB0


def closest_time(ctx, A, B, tA1, tB0):
    P, t0 = ctx["P"], ctx["t0"]
    ts = np.arange(tA1 - 0.02, tB0 + 0.02 + 1e-9, 1 / 240.0)
    d = [np.linalg.norm(proj(P, A, t, t0) - proj(P, B, t, t0)) for t in ts]
    return float(ts[int(np.argmin(d))])


def z_at_end(fl, t0):
    return float(c3.arc_pos(fl["theta"], [t0 + fl["fb"] / FPS - fl["t_ref"]])[0][2])


def events(ctx, chosen, r_seam, a_seam, dt_pair, off):
    """list of dicts(t, kind, how) — kind in hit|bounce (secondary), how in
    serve|pair|arrive|depart."""
    t0 = ctx["t0"]
    out = []
    if not chosen:
        return out
    out.append(dict(t=t0 + chosen[0]["fa"] / FPS - off, kind="hit", how="serve"))
    for A, B in zip(chosen, chosen[1:]):
        dt, e, ang, tA1, tB0 = seam_stats(ctx, A, B)
        kind = "bounce" if z_at_end(A, t0) <= BOUNCE_Z else "hit"
        if dt <= DT_SEAM and e <= r_seam and ang <= a_seam:
            continue                                   # one flight, two pieces
        if dt <= dt_pair:
            out.append(dict(t=closest_time(ctx, A, B, tA1, tB0), kind=kind,
                            how="pair", e=e, ang=ang, dt=dt))
        else:
            out.append(dict(t=tA1, kind=kind, how="arrive", dt=dt))
            out.append(dict(t=tB0 - off, kind="hit", how="depart", dt=dt))
    return out


def raw_events(ctx, chosen):
    t0 = ctx["t0"]
    out = []
    for i, fl in enumerate(chosen):
        out.append(dict(t=t0 + fl["fa"] / FPS, kind="hit", how="raw-start"))
        if i + 1 < len(chosen):
            out.append(dict(t=t0 + fl["fb"] / FPS, kind="hit", how="raw-end"))
    return out


# --------------------------------------------------------- truth + metric

def truth_events(c):
    cont = [float(t) for t in c["imps"]]
    bnc = [float(s["ts"]) for s in c["h_segs"] if s["kind"] == "bounce"]
    return sorted(cont), sorted(bnc)


def match(times, truth, tol=TOL):
    """greedy one-to-one by |dt|; returns list of (i_event, j_truth, dt)."""
    pairs = sorted((abs(t - u), i, j) for i, t in enumerate(times)
                   for j, u in enumerate(truth) if abs(t - u) <= tol)
    used_i, used_j, out = set(), set(), []
    for d, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i); used_j.add(j)
        out.append((i, j, times[i] - truth[j]))
    return out


def prf(times, truth):
    m = match(times, truth)
    rec = len(m) / max(1, len(truth))
    prec = len(m) / max(1, len(times))
    f1 = 0.0 if rec + prec == 0 else 2 * rec * prec / (rec + prec)
    return dict(n=len(times), matched=len(m), recall=rec, precision=prec, f1=f1,
                med_dt=float(np.median([abs(x[2]) for x in m])) if m else float("nan"))


def null_f1(times, truth, lo, hi, rng):
    span = hi - lo
    vals = []
    for _ in range(N_NULL):
        s = rng.uniform(2.0, 4.0) * rng.choice((-1, 1))
        sh = [lo + ((t - lo + s) % span) for t in times]
        vals.append(prf(sh, truth)["f1"])
    return float(np.mean(vals)), float(np.std(vals))


def fmt(tag, r):
    return (f"  {tag:14s} n={r['n']:3d} matched={r['matched']:3d} recall {r['recall']:.3f} "
            f"prec {r['precision']:.3f} F1 {r['f1']:.3f} med|dt| {r['med_dt']:.3f}")


# --------------------------------------------------------- tune / grade

def tune():
    ctxs = [pf.context(r) for r in (6, 7)]
    fls = [flights(ctx)["chosen"] for ctx in ctxs]
    truths = [truth_events(ctx["c"]) for ctx in ctxs]
    raw_tot = dict(n=0, matched=0, ntruth=0)
    for ctx, ch, (cont, bnc) in zip(ctxs, fls, truths):
        tr = sorted(cont + bnc)
        r = prf([e["t"] for e in raw_events(ctx, ch)], tr)
        print(f"rally {ctx['rally']}: {len(ch)} flights, truth {len(cont)} contacts + "
              f"{len(bnc)} bounces; RAW " + fmt("", r).strip())
        raw_tot["n"] += r["n"]; raw_tot["matched"] += r["matched"]; raw_tot["ntruth"] += len(tr)
    raw_f1 = pooled_f1(raw_tot)
    print(f"RAW pooled F1 {raw_f1:.3f}")
    grid = []
    for r_seam in GRID_RSEAM:
        for a_seam in GRID_ASEAM:
            for dt_pair in GRID_DTPAIR:
                for off in GRID_OFF:
                    tot = dict(n=0, matched=0, ntruth=0)
                    per = []
                    for ctx, ch, (cont, bnc) in zip(ctxs, fls, truths):
                        tr = sorted(cont + bnc)
                        ev = events(ctx, ch, r_seam, a_seam, dt_pair, off)
                        r = prf([e["t"] for e in ev], tr)
                        tot["n"] += r["n"]; tot["matched"] += r["matched"]; tot["ntruth"] += len(tr)
                        per.append(f"r{ctx['rally']} {r['matched']}/{r['n']} of {len(tr)}")
                    f1 = pooled_f1(tot)
                    cell = dict(r_seam=r_seam, a_seam=a_seam, dt_pair=dt_pair, off=off,
                                f1=f1, **tot)
                    grid.append(cell)
                    print(f"r_seam={r_seam:4.0f} a_seam={a_seam:3.0f} dt_pair={dt_pair:.2f} "
                          f"off={off:.2f}  pooled F1 {f1:.3f}  | " + "  ".join(per), flush=True)
    rule = ("max pooled F1 over r6+r7 (events vs oracle contacts ∪ human bounces, "
            "TOL 0.10); ties smaller r_seam, a_seam, dt_pair, off 0; must beat RAW "
            f"pooled F1 {raw_f1:.3f} else dead")
    best = sorted(grid, key=lambda g: (-g["f1"], g["r_seam"], g["a_seam"], g["dt_pair"],
                                       g["off"]))[0]
    out = dict(raw_f1=raw_f1, grid=grid, rule=rule)
    if best["f1"] > raw_f1:
        out.update(dead=False, **{k: best[k] for k in ("r_seam", "a_seam", "dt_pair", "off")})
        print(f"VERDICT: r_seam={best['r_seam']:g} a_seam={best['a_seam']:g} "
              f"dt_pair={best['dt_pair']} off={best['off']} (F1 {best['f1']:.3f} vs RAW "
              f"{raw_f1:.3f}) — freeze and one-shot r9/r10")
    else:
        out.update(dead=True)
        print("VERDICT: no cell beats RAW — events layer DEAD, do not run r9/r10")
    TUNE_JSON.write_text(json.dumps(out, indent=1))
    print("wrote", TUNE_JSON)


def pooled_f1(tot):
    rec = tot["matched"] / max(1, tot["ntruth"])
    prec = tot["matched"] / max(1, tot["n"])
    return 0.0 if rec + prec == 0 else 2 * rec * prec / (rec + prec)


def grade(rally, cell):
    ctx = pf.context(rally)
    c, t0 = ctx["c"], ctx["t0"]
    res = flights(ctx)
    ch = res["chosen"]
    if rally in ADOPTED:
        import corridor_dp as cdp
        h12, have, _ = __import__("geom_fix").grade(res["track"], ctx["truth"], t0, ctx["dec"])
        assert h12 == ADOPTED[rally], f"track changed: r@12 {h12} != {ADOPTED[rally]}"
        print(f"track check: r@12 {h12} @ {h12 / have:.2f} == adopted")
    cont, bnc = truth_events(c)
    tr = sorted(cont + bnc)
    print(f"rally {rally}: {len(ch)} flights; truth {len(cont)} contacts + {len(bnc)} "
          f"bounces = {len(tr)} events; cell {cell}")
    ev = events(ctx, ch, cell["r_seam"], cell["a_seam"], cell["dt_pair"], cell["off"])
    times = [e["t"] for e in ev]
    r_ev = prf(times, tr)
    r_raw = prf([e["t"] for e in raw_events(ctx, ch)], tr)
    pc = list(prod_contacts(c, ctx["series"], 0.5))
    print(fmt("events", r_ev))
    print(fmt("RAW", r_raw))
    print(fmt("PROD(all)", prf(pc, tr)))
    print(fmt("PROD(cont)", prf(pc, cont)))
    print(fmt("events(cont)", prf(times, cont)))
    rng = np.random.default_rng(NULL_SEED)
    mu, sd = null_f1(times, tr, c["serve"] - 0.4, c["end"] + 0.2, rng)
    print(f"  null F1 (time-shift x{N_NULL}): {mu:.3f} ± {sd:.3f}  -> bar {mu + 3 * sd:.3f}")
    ok = (r_ev["f1"] >= r_raw["f1"] and r_ev["recall"] >= r_raw["recall"] - 0.05
          and r_ev["f1"] > mu + 3 * sd)
    print(f"  BARS: F1 {r_ev['f1']:.3f} >= RAW {r_raw['f1']:.3f}: "
          f"{r_ev['f1'] >= r_raw['f1']}; recall {r_ev['recall']:.3f} >= RAW-0.05 "
          f"{r_raw['recall'] - 0.05:.3f}: {r_ev['recall'] >= r_raw['recall'] - 0.05}; "
          f"> null: {r_ev['f1'] > mu + 3 * sd}  =>  {'PASS' if ok else 'FAIL'}")
    # secondary: types
    m = {i: j for i, j, _ in match(times, tr)}
    bset = set(bnc)
    typed = dict(hit_true=0, hit_false=0, bounce_true=0, bounce_false=0)
    for i, e in enumerate(ev):
        if i in m:
            is_b = tr[m[i]] in bset
            typed[("bounce" if e["kind"] == "bounce" else "hit") + ("_true" if (e["kind"] == "bounce") == is_b else "_false")] += 1
    print(f"  typed (secondary): emitted hit {sum(1 for e in ev if e['kind'] == 'hit')} / "
          f"bounce {sum(1 for e in ev if e['kind'] == 'bounce')}; matched hit right/wrong "
          f"{typed['hit_true']}/{typed['hit_false']}, bounce right/wrong "
          f"{typed['bounce_true']}/{typed['bounce_false']}; ledger {len(cont)}/{len(bnc)}")
    print("-- events: t | how | kind | nearest truth dt (truth kind) | seam e / ang / dt")
    for i, e in enumerate(ev):
        if i in m:
            j = m[i]; note = f"{times[i] - tr[j]:+.3f} ({'B' if tr[j] in bset else 'C'})"
        else:
            note = "  ---  (unmatched)"
        extra = ""
        if "e" in e:
            extra = f"e {e['e']:5.1f} ang {e['ang']:5.1f} dt {e['dt']:.2f}"
        elif "dt" in e:
            extra = f"gap {e['dt']:.2f}"
        print(f"   {e['t']:7.3f} | {e['how']:7s} | {e['kind']:6s} | {note} | {extra}")
    miss = [u for j, u in enumerate(tr) if j not in set(m.values())]
    print("-- truth not matched: " + ", ".join(f"{u:.2f}{'B' if u in bset else 'C'}" for u in miss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("tune", "grade"))
    ap.add_argument("rally", nargs="?", type=int)
    ap.add_argument("--r-seam", type=float)
    ap.add_argument("--a-seam", type=float)
    ap.add_argument("--dt-pair", type=float)
    ap.add_argument("--off", type=float)
    a = ap.parse_args()
    if a.cmd == "tune":
        tune(); return
    over = {k: v for k, v in (("r_seam", a.r_seam), ("a_seam", a.a_seam),
                              ("dt_pair", a.dt_pair), ("off", a.off)) if v is not None}
    if a.rally in pf.EVAL_RALLIES:
        assert not over, "no knob overrides on r9/r10"
        assert TUNE_JSON.exists(), "tune first"
        t = json.loads(TUNE_JSON.read_text())
        assert not t.get("dead"), "events layer is DEAD; grade 9|10 refused"
        cell = {k: t[k] for k in ("r_seam", "a_seam", "dt_pair", "off")}
    else:
        cell = dict(r_seam=16.0, a_seam=40.0, dt_pair=0.40, off=0.0)
        if TUNE_JSON.exists() and not over:
            t = json.loads(TUNE_JSON.read_text())
            if not t.get("dead"):
                cell = {k: t[k] for k in ("r_seam", "a_seam", "dt_pair", "off")}
        cell.update(over)
    grade(a.rally, cell)


if __name__ == "__main__":
    main()
