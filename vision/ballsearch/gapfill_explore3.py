"""TRAIN-ONLY exploration behind gapfill_gate.md §v3c (2026-09-02): why the
hit-anchored fill (v3, v3b) fills nothing, and what any kink rule that uses
only the contact TIME + hitter DEPTH (never the paddle pixel) can reach on
r6/r7.  Two parts, both printed to gapfill_explore3.txt:

  1. geometry — every owner click inside each open gap next to A's forward
     and B's backward arc extrapolation (pixel distance, 3D height);
  2. rule table — the gap filled with a kink at the detector's contact
     time under each candidate rule for the kink point X_c, graded on
     the inferred frames only (r@12 / have / frames, displaced + time-shift
     nulls), with and without the floor-bounce arm.

r9 / r10 are never loaded here.  Diagnostic only; nothing it prints is a
selection — the verdict is written in the gate.

    python3 gapfill_explore3.py > gapfill_explore3.txt
"""
import numpy as np

import pathfirst as pf
import gapfill as gf
import geom_fix
import events as evm
import court3d as c3                                        # noqa: E402 (path set by pathfirst)

FPS = gf.FPS
E_BOUNCE, MU_BOUNCE = 0.75, 0.8                             # vertical restitution, horizontal keep


def ext(fl, t):
    return c3.arc_pos(fl["theta"], [t - fl["t_ref"]])[0]


def bounce_fwd(A, ta, tc):
    """A extrapolated to tc with a floor reflection if its arc crosses z=0 in (ta, tc)."""
    tbn = gf.floor_time(A, ta - gf.FLOOR_EPS, tc, True)
    if tbn is None or tc - tbn < 1 / FPS:
        return ext(A, tc), None
    Xb = ext(A, tbn).copy()
    Xb[2] = 0.0
    v = np.asarray(c3.arc_vel(A["theta"], tbn - A["t_ref"]), float)
    v[2] = -E_BOUNCE * v[2]
    v[:2] *= MU_BOUNCE
    return c3.arc_pos(np.concatenate([Xb, v]), [tc - tbn])[0], (tbn, Xb)


def bounce_bwd(B, tb, tc):
    tbn = gf.floor_time(B, tc, tb + gf.FLOOR_EPS, False)
    if tbn is None or tbn - tc < 1 / FPS:
        return ext(B, tc), None
    Xb = ext(B, tbn).copy()
    Xb[2] = 0.0
    v = np.asarray(c3.arc_vel(B["theta"], tbn - B["t_ref"]), float)
    v[2] = -v[2] / E_BOUNCE
    v[:2] /= MU_BOUNCE
    return c3.arc_pos(np.concatenate([Xb, v]), [tc - tbn])[0], (tbn, Xb)


RULES = ["A-ext", "B-ext", "mean", "mean-y", "chord-t", "chord-y",
         "relift-AB", "relift-A", "relift-B", "relift-mean"]


def xc_rule(rule, P, XA, XB, A, B, ta, tb, tc, yp, side):
    """kink point at tc.  XA / XB = A forward / B backward at tc; yp = hitter floor depth;
    side = which arc owns this anchor when there are two ('A' first, 'B' last)."""
    if rule.startswith("relift"):                   # pixel of an extrapolation, lifted at the hitter's depth
        pa = c3.project(P, XA[None, :])[0]
        pb = c3.project(P, XB[None, :])[0]
        uv = {"relift-AB": pa if side == "A" else pb, "relift-A": pa, "relift-B": pb,
              "relift-mean": (pa + pb) / 2}[rule]
        return gf.lift(P, (float(uv[0]), float(uv[1])), float(yp))
    if rule == "A-ext":
        return XA.copy()
    if rule == "B-ext":
        return XB.copy()
    if rule in ("mean", "mean-y"):
        X = (XA + XB) / 2
    else:                                           # chord-t / chord-y: on the 3D chord A.end -> B.start at tc
        Xa, Xb_ = ext(A, ta), ext(B, tb)
        X = Xa + (Xb_ - Xa) * (tc - ta) / max(1e-6, tb - ta)
    if rule.endswith("-y"):
        X[1] = yp
    return X


def fill_gap(ctx, A, B, anc, rule, bounce):
    P, t0 = ctx["P"], ctx["t0"]
    ta, tb = t0 + A["fb"] / FPS, t0 + B["fa"] / FPS
    fr = lambda t: int(round((t - t0) * FPS))
    a1, ak = anc[0], anc[-1]
    XA, bA = bounce_fwd(A, ta, a1["t"]) if bounce else (ext(A, a1["t"]), None)
    XB, bB = bounce_bwd(B, tb, ak["t"]) if bounce else (ext(B, ak["t"]), None)
    if len(anc) == 1:
        Xc1 = Xck = xc_rule(rule, P, XA, XB, A, B, ta, tb, a1["t"], a1["X"][1], "B")
    else:
        Xc1 = xc_rule(rule, P, XA, XB, A, B, ta, tb, a1["t"], a1["X"][1], "A")
        Xck = xc_rule(rule, P, XA, XB, A, B, ta, tb, ak["t"], ak["X"][1], "B")
    f_c1 = min(max(fr(a1["t"]), A["fb"]), B["fa"] - 1)
    f_ck = min(max(fr(ak["t"]), A["fb"]), B["fa"] - 1)
    pieces = []
    if f_c1 > A["fb"]:                              # A side: A.fb+1 .. f_c1
        if bA is not None:
            tbn, Xb = bA
            fbn = min(max(fr(tbn), A["fb"]), f_c1 - 1)
            if fbn > A["fb"]:
                pieces.append(gf.piece(A, A["theta"], A["t_ref"], A["fb"] + 1, fbn))
            pieces.append(gf.piece(A, gf.bvp(Xb, tbn, Xc1, a1["t"]), tbn, fbn + 1, f_c1))
        else:
            pieces.append(gf.piece(A, gf.bvp(ext(A, ta), ta, Xc1, a1["t"]), ta, A["fb"] + 1, f_c1))
    if len(anc) >= 2:                               # between anchors: BVP kink to kink
        fa_, fb_ = max(f_c1 + 1, A["fb"] + 1), min(f_ck, B["fa"] - 1)
        if fb_ >= fa_:
            pieces.append(gf.piece(A, gf.bvp(Xc1, a1["t"], Xck, ak["t"]), a1["t"], fa_, fb_))
    if f_ck + 1 <= B["fa"] - 1:                     # B side: f_ck+1 .. B.fa-1
        if bB is not None:
            tbn, Xb = bB
            fbn = min(max(fr(tbn), f_ck + 1), B["fa"] - 1)
            pieces.append(gf.piece(B, gf.bvp(Xck, ak["t"], Xb, tbn), ak["t"], f_ck + 1, fbn))
            if fbn + 1 <= B["fa"] - 1:
                pieces.append(gf.piece(B, B["theta"], B["t_ref"], fbn + 1, B["fa"] - 1))
        else:
            pieces.append(gf.piece(B, gf.bvp(Xck, ak["t"], ext(B, tb), tb), ak["t"], f_ck + 1, B["fa"] - 1))
    return [pc for pc in pieces if pc["fb"] >= pc["fa"]]


def open_gaps(ctx):
    res2 = gf.product(ctx)
    if "_floors" not in ctx:
        ctx["_floors"] = gf.floor_tracks(ctx)
    fls = sorted([dict(f) for f in res2["chosen"]], key=lambda f: f["fa"])
    t0 = ctx["t0"]
    out = []
    for A, B in zip(fls, fls[1:]):
        if B["fa"] - A["fb"] - 1 < 1:
            continue
        ta, tb = t0 + A["fb"] / FPS, t0 + B["fa"] / FPS
        out.append((A, B, ta, tb, gf.anchors3(ctx, A, B, ctx["_floors"], ta, tb)))
    return fls, out


def run(ctx, rule, bounce):
    fls, gaps = open_gaps(ctx)
    out, inf3 = list(fls), set()
    for A, B, ta, tb, anc in gaps:
        if not anc:
            continue
        for pc in fill_gap(ctx, A, B, anc, rule, bounce):
            out.append(pc)
            inf3 |= set(range(pc["fa"], pc["fb"] + 1))
    out.sort(key=lambda f: f["fa"])
    track = pf.track_of(ctx, out)
    inf = {f: xy for f, xy in track.items() if f in inf3}
    h, hv, _ = geom_fix.grade(inf, ctx["truth"], ctx["t0"], ctx["dec"])
    nd, nt = gf.nulls(ctx, inf, np.random.default_rng(pf.NULL_SEED + ctx["rally"]))
    return h, hv, len(inf3), nd, nt


def geometry(ctx):
    r, t0, P = ctx["rally"], ctx["t0"], ctx["P"]
    cont, bnc = evm.truth_events(ctx["c"])
    _, gaps = open_gaps(ctx)
    for A, B, ta, tb, anc in gaps:
        print(f"=== r{r} gap f{A['fb']}-{B['fa']} ({ta - t0:.2f}-{tb - t0:.2f} s, {B['fa'] - A['fb'] - 1} frames)"
              f"  A n={A['n']} rms={A['rms']:.2f}  B n={B['n']} rms={B['rms']:.2f}"
              f"  detector anchors {[round(a['t'] - t0, 2) for a in anc]}"
              f"  owner contacts {[round(c - t0, 2) for c in cont if ta - 0.1 <= c <= tb + 0.1]}"
              f"  bounces {[round(b - t0, 2) for b in bnc if ta - 0.1 <= b <= tb + 0.1]}")
        for (t, x, y, vis), d in zip(ctx["truth"], ctx["dec"]):
            if ta - 0.05 <= t <= tb + 0.05:
                ea = np.asarray(pf.arc_px(P, A["theta"], A["t_ref"], (t - t0) * FPS, t0), float)
                eb = np.asarray(pf.arc_px(P, B["theta"], B["t_ref"], (t - t0) * FPS, t0), float)
                za, zb = ext(A, t)[2], ext(B, t)[2]
                print(f"   t={t - t0:.2f} {vis} truth=({x:.0f},{y:.0f})  A_fwd=({ea[0]:.0f},{ea[1]:.0f})"
                      f" d={np.hypot(*(ea - [x, y])):4.0f} z={za:5.1f}   B_bwd=({eb[0]:.0f},{eb[1]:.0f})"
                      f" d={np.hypot(*(eb - [x, y])):4.0f} z={zb:5.1f}")


def main():
    ctxs = {r: pf.context(r) for r in (6, 7)}
    print("PART 1 — geometry of every open gap on train (owner clicks vs the two arcs)")
    for ctx in ctxs.values():
        geometry(ctx)
    print("\nPART 2 — kink-rule table (inferred frames only; kink at the detector's contact time)")
    print("%-12s %-6s | %s" % ("rule", "bounce", "   ".join(f"r{r} r@12/have/frames nulls" for r in ctxs)))
    for rule in RULES:
        for bounce in (False, True):
            cells = [run(ctx, rule, bounce) for ctx in ctxs.values()]
            s = "   ".join(f"{h}/{hv}/{n} {nd}/{nt}" for h, hv, n, nd, nt in cells)
            th, tv = sum(c[0] for c in cells), sum(c[1] for c in cells)
            print("%-12s %-6s | %s   pooled %d/%d prec %.2f" % (rule, bounce, s, th, tv, th / max(1, tv)))


if __name__ == "__main__":
    main()
