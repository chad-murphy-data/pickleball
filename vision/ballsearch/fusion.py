"""fusion.py — the three-part ball model as ONE cost function
(owner go 2026-09-01; HANDOFF.md item 1, depth L3).

The three instruments this fuses, each graded on its own earlier:
  emission   learned per-candidate ball probability p (emission.py,
             trained on r6/r7 owner clicks ONLY);
  spaghetti  physical trail hypotheses — drag-ballistic arcs between
             the two contacts, bounce trails from the 52-shot book,
             null-subtracted support, shot-repertoire prior;
  DP         per-corridor Viterbi chain over motion candidates:
             smoothness (accel) + gaps + endpoints + body-extremity
             cost + W_P_SOFT·(1−p).
"Not an ensemble" (owner): the DP search runs INSIDE the trail
search. Per corridor:
  1. PROPOSE  the spaghetti matcher scores every trail and REFINES the
              top M (M_TOP=8; the M=1 arm is reported alongside). If
              its best strand is under ABSTAIN the corridor falls back
              to the incumbent DP (no trail).
  2. SEARCH   each proposal conditions one DP run: the per-frame
              candidate pool becomes "chord window OR within R_BAND of
              the trail pixel" ranked by distance to the trail (the
              trail defines where to look — lob/bounce excursions the
              chord box cannot see), and taking a candidate costs an
              extra W_TRAIL·min(d_to_trail/R_TRAIL, 1). Everything else
              in the DP (accel, gaps, endpoints, body, emission) is the
              incumbent's, untouched.
  3. CHOOSE   joint cost J_m = DP path cost_m + W_GAP·shot-book prior
              penalty_m (spaghetti's prior_pen, one frame-of-support
              unit = one W_GAP skip). argmin over the M proposals.
  4. BRIDGE   frames the chosen path skipped are filled from the
              chosen trail, FLAGGED as inferred (the -F arm; graded
              separately, it carries the off-frame stratum).
  5. CONFIDENCE (truth-free): J_best minus the same trail-conditioned
              DP on a displaced corridor (anchors + trail shifted ~200
              px) — reported per corridor, never used for selection.

PROTOCOL (written before any number; carries HANDOFF constraints):
  - TRAIN = r6 + r7 with CROSS-FOLD p (p_r{6,7}_cc_14_x.npz). r9/r10
    clicks are evaluation-only: `fusion.py grade 9|10` refuses to run
    until fusion_tune.json holds a frozen choice, and refuses any
    --W/--R override on those rallies.
  - Tuned knobs: W_TRAIL ∈ GRID_W × R_TRAIL ∈ GRID_R. Fixed by design
    (not swept): M_TOP=8, R_BAND=60, SCALE_PRIOR=W_GAP, proposal
    candidates = cc (the HANDOFF's ccL numbers), W_P_SOFT=25 (softdp).
  - SELECTION RULE: over the 4 train panels (r6/r7 × prod/oracle),
    the `fus` arm (chosen DP nodes only, no bridge) — pick the (W, R)
    with the largest total r@12 subject to pooled prec@12 ≥ the
    INCUMBENT's pooled prec@12 (dp-ccS+body, W_P_SOFT=25, no trail);
    ties → smallest W, then smallest R. If no grid point beats the
    incumbent's total r@12 under that constraint, fusion is DEAD:
    record it and do NOT run r9/r10. fus-F and fus1 are reported at
    the chosen point only; they do not vote.
  - Grading = the thread's metric: r@12 over ALL V+S clicks, prec@12 =
    hits / at-click track points, ADDED@12 vs the decode; displaced-
    anchor nulls (same rng seed and draws as spaghetti.py) for fus and
    fus-F; per-click STRATUM (outwin / nocand / cand) reported so the
    geometry misses HANDOFF item 2 named are visible, denominators
    never loosened.

Usage:
  python3 fusion.py tune                 # r6/r7 sweep → fusion_tune.json
  python3 fusion.py grade <rally>        # one-shot graded run (bg it)
  python3 fusion.py selftest             # synthetic corridor, no media
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import corridor_dp as cdp                              # noqa: E402
import spaghetti as spag                               # noqa: E402

SP = Path(__file__).parent
TUNE_JSON = SP / "fusion_tune.json"
W_P_SOFT = 25.0            # frozen incumbent (softdp.py)
M_TOP = 8
SCALE_PRIOR = cdp.W_GAP    # spaghetti prior units -> DP cost units
GRID_W = (3.0, 6.0, 12.0, 25.0)
GRID_R = (8.0, 16.0, 30.0)
NULL_SEED = 20260901
EVAL_RALLIES = (9, 10)
FRAME_W, FRAME_H = 1280, 720


# ------------------------------------------------------------ propose

def propose(P, C, Minv, Hg_inv, c, series, npz, cands, cor, t0, body,
            M=M_TOP, disp=None):
    """spaghetti stage: score every trail, refine the top M, return
    them best-first with pixel tracks and prior penalties. Mirrors
    spaghetti.run_corridor (same pool, null base, refine) but keeps
    the top M instead of the argmax."""
    ta, tb, A, B, wx, wy = cor
    if disp:
        A = (A[0] + disp[0], A[1] + disp[1])
        B = (B[0] + disp[0], B[1] + disp[1])
        cor = (ta, tb, A, B, wx, wy)
    if tb - ta < spag.MIN_COR_S:
        return None
    fa, fb = int(round((ta - t0) * 60)), int(round((tb - t0) * 60))
    fs = list(range(fa, fb + 1))
    times = t0 + np.asarray(fs) / 60.0
    pool = spag.corridor_pool(cands, cor, t0, fa, fb, body)
    if len(pool) < 3:
        return None
    hxy = spag.hitter_xy(c, series, npz, Hg_inv, ta, A)
    trails = (spag.build_direct(C, Minv, ta, tb, A, B, hxy)
              + spag.build_bounce(C, Minv, ta, tb, A, B, hxy,
                                  lats=spag.LATS_REP,
                                  launch_pairs=spag.LIB))
    if not trails:
        return None
    base = spag.frame_base(pool, cor, t0, fs, 91000 + fa)
    px, zmin = spag.trail_pixels(P, trails, ta, times)
    tot, _ = spag.score_trails(px, zmin, spag.prior_pen(trails), pool,
                               fs, base)
    refined = []
    for i in np.argsort(-tot)[:M]:
        tr_i, sc_i = spag.refine(P, C, Minv,
                                 (trails[int(i)], float(tot[int(i)])),
                                 ta, tb, times, pool, fs, hxy, A, B,
                                 base)
        refined.append((tr_i, float(sc_i)))
    refined.sort(key=lambda x: -x[1])
    props = []
    for tr, sc in refined:
        bpx, _ = spag.trail_pixels(P, [tr], ta, times)
        bpx = bpx[0]
        props.append(dict(
            trail=tr, score=sc, pen=float(spag.prior_pen([tr])[0]),
            px={f: (float(bpx[j, 0]), float(bpx[j, 1]))
                for j, f in enumerate(fs)}))
    return dict(cor=cor, fa=fa, fb=fb, props=props,
                abst=refined[0][1] < spag.ABSTAIN)


# ------------------------------------------------------- search/choose

def fuse_corridor(cc, cor, t0, body, prop, M, W, R, disp=None):
    """one corridor: incumbent DP + M trail-conditioned DPs; choose
    by joint cost; bridge skipped frames from the chosen trail."""
    if disp:
        ta, tb, A, B, wx, wy = cor
        cor = (ta, tb, (A[0] + disp[0], A[1] + disp[1]),
               (B[0] + disp[0], B[1] + disp[1]), wx, wy)
    cdp.W_TRAIL, cdp.R_TRAIL = 0.0, R
    inc, cost0 = cdp.dp_path(cc, cor, t0, body=body, return_cost=True)
    rec = dict(cor=cor, inc=inc, cost0=cost0, path=inc, bridge={},
               choice=None, J=cost0, why="dp")
    if prop is None:
        rec["why"] = "noprop"
        return rec
    if prop["abst"] or M <= 0:
        rec["why"] = "abstain"
        return rec
    cdp.W_TRAIL = W
    best = None
    for m, pr in enumerate(prop["props"][:M]):
        path, cost = cdp.dp_path(cc, cor, t0, body=body, trail=pr["px"],
                                 return_cost=True)
        if not path:
            continue
        J = cost + SCALE_PRIOR * pr["pen"]
        if best is None or J < best[0]:
            best = (J, m, path)
    cdp.W_TRAIL = 0.0
    if best is None:
        rec["why"] = "nopath"
        return rec
    J, m, path = best
    px = prop["props"][m]["px"]
    fa, fb = prop["fa"], prop["fb"]
    rec.update(path=path, choice=m, J=J, why="trail",
               bridge={f: px[f] for f in range(fa, fb + 1)
                       if f not in path and f in px})
    return rec


def confidence(cc, t0, body, prop, rec, W, R, disp):
    """truth-free: cost of the SAME trail-conditioned DP on a
    displaced corridor minus the chosen joint cost. inf when the
    displaced DP finds no path at all."""
    if rec["choice"] is None:
        return None
    ta, tb, A, B, wx, wy = rec["cor"]
    cor_d = (ta, tb, (A[0] + disp[0], A[1] + disp[1]),
             (B[0] + disp[0], B[1] + disp[1]), wx, wy)
    pr = prop["props"][rec["choice"]]
    px_d = {f: (x + disp[0], y + disp[1]) for f, (x, y) in pr["px"].items()}
    cdp.W_TRAIL, cdp.R_TRAIL = W, R
    _, cost_n = cdp.dp_path(cc, cor_d, t0, body=body, trail=px_d,
                            return_cost=True)
    cdp.W_TRAIL = 0.0
    if not np.isfinite(cost_n):
        return float("inf")
    return float(cost_n + SCALE_PRIOR * pr["pen"] - rec["J"])


def fuse_rally(ctx, cors, props, M, W, R, disp=None, conf_disp=None):
    """tracks for one arm. props: list aligned with cors (None ok).
    Returns dict(inc, fus, fusF, recs)."""
    inc, fus, fusF, recs = {}, {}, {}, []
    for cor, prop in zip(cors, props):
        rec = fuse_corridor(ctx["cc"], cor, ctx["t0"], ctx["body"], prop,
                            M, W, R, disp)
        if conf_disp is not None and prop is not None:
            rec["conf"] = confidence(ctx["cc"], ctx["t0"], ctx["body"],
                                     prop, rec, W, R, conf_disp)
        inc.update(rec["inc"])
        fus.update(rec["path"])
        fusF.update(rec["path"])
        fusF.update(rec["bridge"])
        recs.append(rec)
    return dict(inc=inc, fus=fus, fusF=fusF, recs=recs)


# ------------------------------------------------------------ context

def context(rally, pxs=""):
    from claim_lab import load, paddle_series
    from corridor_lab import load_truth, decode_recall
    c = load(rally)
    series = paddle_series(c["npz"])
    npz = np.load(c["npz"])
    truth = load_truth(rally)
    t0 = c["t0"]
    P = c["P"]
    C, Minv, Hg_inv = spag.cam(P)
    f_lo = int((c["serve"] - 0.4 - t0) * 60)
    f_hi = int((c["end"] + 0.2 - t0) * 60)
    cdp.W_P_SOFT = W_P_SOFT
    cc = spag.cands_cached(rally, f_lo, f_hi, 14, "cc", lrn=True, pxs=pxs)
    return dict(rally=rally, c=c, series=series, npz=npz, truth=truth,
                t0=t0, P=P, C=C, Minv=Minv, Hg_inv=Hg_inv, cc=cc,
                body=cdp.body_points(c, f_lo, f_hi),
                dec=decode_recall(c, truth))


def proposals(ctx, cors, M=M_TOP, disp=None):
    return [propose(ctx["P"], ctx["C"], ctx["Minv"], ctx["Hg_inv"],
                    ctx["c"], ctx["series"], ctx["npz"], ctx["cc"], cor,
                    ctx["t0"], ctx["body"], M, disp) for cor in cors]


def arms(ctx, rth=0.5):
    from corridor_lab import prod_contacts, corridors
    c, series = ctx["c"], ctx["series"]
    return (("prod", corridors(c, series, prod_contacts(c, series, rth))),
            ("oracle", corridors(c, series, list(c["imps"]))))


def grade(track, truth, t0, dec):
    h12 = have = added = 0
    for (t, tx, ty, vis), d in zip(truth, dec):
        f = int(round((t - t0) * 60))
        p = track.get(f) or track.get(f - 1) or track.get(f + 1)
        if p is None:
            continue
        have += 1
        dd = float(np.hypot(p[0] - tx, p[1] - ty))
        h12 += dd <= cdp.R_MAIN
        added += dd <= cdp.R_MAIN and not d
    return h12, have, added


# --------------------------------------------------------------- tune

def tune():
    """r6/r7 ONLY, cross-fold p. Proposals computed once per panel
    (they do not depend on W/R), then the grid is swept on the DP."""
    panels = []
    for rally in (6, 7):
        ctx = context(rally, pxs="_x")
        print(f"rally {rally}: {len(ctx['truth'])} clicks, decode@12 "
              f"{sum(ctx['dec'])}/{len(ctx['dec'])}")
        for arm, cors in arms(ctx):
            props = proposals(ctx, cors)
            nab = sum(1 for p in props if p is None or p["abst"])
            print(f"  {arm}: {len(cors)} corridors, proposals built, "
                  f"abstain/none {nab}")
            panels.append((ctx, arm, cors, props))

    def total(M, W, R, key):
        tot = dict(h12=0, have=0, added=0)
        per = []
        for ctx, arm, cors, props in panels:
            out = fuse_rally(ctx, cors, props, M, W, R)
            h12, have, added = grade(out[key], ctx["truth"], ctx["t0"],
                                     ctx["dec"])
            tot["h12"] += h12
            tot["have"] += have
            tot["added"] += added
            per.append(f"r{ctx['rally']}-{arm} {h12}/{have}")
        tot["prec"] = tot["h12"] / max(1, tot["have"])
        return tot, per

    rows = []
    inc, per = total(0, 0.0, 16.0, "inc")
    print(f"INCUMBENT   total r@12 {inc['h12']:4d}  prec@12 "
          f"{inc['prec']:.3f}  ADDED {inc['added']:3d}  | "
          + "  ".join(per))
    for W in GRID_W:
        for R in GRID_R:
            t, per = total(M_TOP, W, R, "fus")
            rows.append(dict(W=W, R=R, **t))
            print(f"W={W:4g} R={R:4g}  total r@12 {t['h12']:4d}  prec@12 "
                  f"{t['prec']:.3f}  ADDED {t['added']:3d}  | "
                  + "  ".join(per))
    ok = [r for r in rows if r["prec"] >= inc["prec"]
          and r["h12"] > inc["h12"]]
    verdict = dict(incumbent=inc, grid=rows, rule=(
        "max total r@12 over r6/r7 x prod/oracle (fus arm) s.t. pooled "
        "prec@12 >= incumbent; ties smallest W then R; none -> dead"))
    if not ok:
        print("VERDICT: no (W, R) beats the incumbent under the rule — "
              "fusion DEAD, do not run r9/r10")
        verdict.update(dead=True)
    else:
        best = sorted(ok, key=lambda r: (-r["h12"], r["W"], r["R"]))[0]
        W, R = best["W"], best["R"]
        extra = {}
        for key, M in (("fusF", M_TOP), ("fus1", 1)):
            t, per = total(M, W, R, "fusF" if key == "fusF" else "fus")
            extra[key] = t
            print(f"at choice, {key:5s}: total r@12 {t['h12']:4d}  prec@12"
                  f" {t['prec']:.3f}  ADDED {t['added']:3d}  | "
                  + "  ".join(per))
        print(f"VERDICT: W_TRAIL = {W:g}, R_TRAIL = {R:g} (total r@12 "
              f"{best['h12']} vs incumbent {inc['h12']}, prec "
              f"{best['prec']:.3f} >= {inc['prec']:.3f}) — freeze and "
              f"one-shot r9/r10")
        verdict.update(dead=False, W_TRAIL=W, R_TRAIL=R, M_TOP=M_TOP,
                       at_choice=extra)
    TUNE_JSON.write_text(json.dumps(verdict, indent=1))
    print(f"wrote {TUNE_JSON}")


# -------------------------------------------------------------- grade

def strata(ctx, cors, cc):
    """per-click stratum on the prod corridors: nocor / outwin /
    nocand (no cc candidate within R_MAIN at f±1) / cand."""
    from corridor_lab import window_at
    out = []
    for (t, tx, ty, vis) in ctx["truth"]:
        cor = next((co for co in cors if co[0] <= t <= co[1]), None)
        if cor is None:
            out.append("nocor")
            continue
        cx, cy, wx, wy = window_at(cor, t)
        if abs(tx - cx) > wx or abs(ty - cy) > wy:
            out.append("outwin")
            continue
        f = int(round((t - ctx["t0"]) * 60))
        near = any(np.hypot(c_[0] - tx, c_[1] - ty) <= cdp.R_MAIN
                   for df in (-1, 0, 1) for c_ in cc.get(f + df, ()))
        out.append("cand" if near else "nocand")
    return out


def run_grade(rally, W, R, M):
    pxs = "_x" if rally in (6, 7) else ""
    ctx = context(rally, pxs=pxs)
    truth, t0, dec = ctx["truth"], ctx["t0"], ctx["dec"]
    print(f"rally {rally}: {len(truth)} V/S clicks, decode@12 "
          f"{sum(dec)}/{len(dec)}; fusion W_TRAIL={W:g} R_TRAIL={R:g} "
          f"M={M} R_BAND={cdp.R_BAND:g} W_P_SOFT={W_P_SOFT:g} "
          f"(p-cache suffix '{pxs}')")
    rng = np.random.default_rng(NULL_SEED)
    for arm, cors in arms(ctx):
        props = proposals(ctx, cors)
        cd = (200.0, -110.0)
        out = fuse_rally(ctx, cors, props, M, W, R, conf_disp=cd)
        out1 = fuse_rally(ctx, cors, props, 1, W, R)
        print(f"== {arm}: {len(cors)} corridors")
        cdp.score(out["inc"], truth, t0, dec, "dp-ccS+body")
        cdp.score(out["fus"], truth, t0, dec, "fus")
        cdp.score(out["fusF"], truth, t0, dec, "fus-F")
        cdp.score(out1["fus"], truth, t0, dec, "fus1")
        for vis in ("V", "S"):
            tt = [x for x in truth if x[3] == vis]
            dd = [d for x, d in zip(truth, dec) if x[3] == vis]
            cdp.score(out["fus"], tt, t0, dd, f"  fus[{vis}]")
            cdp.score(out["fusF"], tt, t0, dd, f"  fusF[{vis}]")
        why = {}
        for r in out["recs"]:
            why[r["why"]] = why.get(r["why"], 0) + 1
        print(f"    corridor outcomes: {why}")
        for kk in range(2):
            d = (float(rng.uniform(160, 240)) * rng.choice([-1, 1]),
                 float(rng.uniform(80, 140)) * rng.choice([-1, 1]))
            pn = proposals(ctx, cors, disp=d)
            on = fuse_rally(ctx, cors, pn, M, W, R, disp=d)
            cdp.score(on["fus"], truth, t0, dec, f"null{kk}")
            cdp.score(on["fusF"], truth, t0, dec, f"null{kk}-F")
        # stratified r@12 (prod corridors define the strata)
        st = strata(ctx, cors, ctx["cc"])
        print("    stratum      n   inc   fus  fus-F")
        for name in ("cand", "nocand", "outwin", "nocor"):
            idx = [i for i, s in enumerate(st) if s == name]
            if not idx:
                continue
            tt = [truth[i] for i in idx]
            dd = [dec[i] for i in idx]
            hi = grade(out["inc"], tt, t0, dd)[0]
            hf = grade(out["fus"], tt, t0, dd)[0]
            hF = grade(out["fusF"], tt, t0, dd)[0]
            print(f"    {name:8s} {len(idx):5d} {hi:5d} {hf:5d} {hF:5d}")
        if arm != "prod":
            continue
        h_in = spag.hits(out["inc"], truth, t0)
        h_fu = spag.hits(out["fus"], truth, t0)
        h_fF = spag.hits(out["fusF"], truth, t0)
        print("-- corridors (prod): t-span dur | choice kind/mode sp | "
              "why conf | clicks: inc / fus / fus-F")
        for r in out["recs"]:
            ta, tb = r["cor"][0], r["cor"][1]
            idx = [i for i, (t, *_) in enumerate(truth) if ta <= t <= tb]
            n_ = lambda h: sum(1 for i in idx if h[i] is not None  # noqa
                               and h[i] <= cdp.R_MAIN)
            desc = "  -            "
            if r["choice"] is not None:
                b = None
                for p_ in props:
                    if p_ is not None and p_["cor"] == r["cor"]:
                        b = p_["props"][r["choice"]]["trail"]
                        break
                if b is not None:
                    desc = (f"#{r['choice']} {b['kind'][:3]}/"
                            f"{spag.mode_name(b['sp'], b['loft']):5s} "
                            f"sp{b['sp']:5.1f}")
            conf = r.get("conf")
            cs = ("  -  " if conf is None else
                  (" inf " if not np.isfinite(conf) else f"{conf:5.0f}"))
            print(f"  {ta:7.2f}-{tb:7.2f} {tb-ta:4.2f}s | {desc} | "
                  f"{r['why']:7s} {cs} | {len(idx):3d}: {n_(h_in):3d} /"
                  f"{n_(h_fu):4d} /{n_(h_fF):4d}")


# ------------------------------------------------------------ selftest

def selftest():
    """no media: a synthetic corridor with a planted ballistic trail,
    a junk field and a decoy trail. Checks (1) trail=None DP is bit-
    identical to the incumbent, (2) the trail-conditioned DP with the
    TRUE trail recovers the planted candidates, (3) the chooser picks
    the true trail over the decoy on joint cost, (4) bridge fills the
    skipped frames from the chosen trail."""
    rng = np.random.default_rng(3)
    t0 = 0.0
    fa, fb = 0, 60
    A, B = (200.0, 400.0), (900.0, 300.0)
    truth_px = {}
    for f in range(fa, fb + 1):
        a = f / fb
        truth_px[f] = (A[0] + a * (B[0] - A[0]),
                       A[1] + a * (B[1] - A[1]) - 160.0 * a * (1 - a))
    decoy = {f: (x, y + 90.0) for f, (x, y) in truth_px.items()}
    cands = {}
    for f in range(fa, fb + 1):
        cs = []
        if not (24 <= f <= 28):    # a 5-frame hole (within GAP=6)
            x, y = truth_px[f]
            cs.append((x + rng.normal(0, 1), y + rng.normal(0, 1), 5,
                       30.0, 0.9))
        for _ in range(12):        # junk: random, low p
            cs.append((float(rng.uniform(150, 950)),
                       float(rng.uniform(150, 550)), 4, 20.0,
                       float(rng.uniform(0.02, 0.3))))
        if 10 <= f <= 50:          # a smooth decoy walker, mid p
            x, y = decoy[f]
            cs.append((x, y, 5, 25.0, 0.5))
        cands[f] = cs
    cor = (t0, t0 + fb / 60.0, A, B, 140.0, 170.0)
    cdp.W_P_SOFT = W_P_SOFT
    cdp.W_TRAIL = 0.0
    p_a = cdp.dp_path(cands, cor, t0)
    p_b, cost_b = cdp.dp_path(cands, cor, t0, trail=None, return_cost=True)
    assert p_a == p_b and np.isfinite(cost_b), "trail=None must be identical"
    prop = dict(cor=cor, fa=fa, fb=fb, abst=False, props=[
        dict(trail=dict(kind="direct", sp=40.0, loft=10.0), score=9.0,
             pen=0.0, px=decoy),
        dict(trail=dict(kind="direct", sp=40.0, loft=10.0), score=8.0,
             pen=0.0, px=truth_px)])
    rec = fuse_corridor(cands, cor, t0, None, prop, 2, 12.0, 16.0)
    assert rec["why"] == "trail", rec["why"]
    assert rec["choice"] == 1, f"chooser picked decoy: {rec}"
    hits = sum(1 for f, (x, y) in rec["path"].items()
               if np.hypot(x - truth_px[f][0], y - truth_px[f][1]) <= 4)
    assert hits >= 45, f"true-trail DP recovered only {hits}"
    assert rec["bridge"] and all(f not in rec["path"] for f in rec["bridge"])
    assert all(rec["bridge"][f] == truth_px[f] for f in rec["bridge"])
    # abstain -> incumbent
    prop["abst"] = True
    rec2 = fuse_corridor(cands, cor, t0, None, prop, 2, 12.0, 16.0)
    assert rec2["why"] == "abstain" and rec2["path"] == p_a
    conf = confidence(cands, t0, None, dict(props=prop["props"]), rec,
                      12.0, 16.0, (200.0, -110.0))
    assert conf is None or conf > 0, conf
    print(f"SELFTEST OK: inc {len(p_a)} nodes, fused {len(rec['path'])} "
          f"nodes ({hits} on the planted trail), bridge "
          f"{len(rec['bridge'])} frames, conf {conf:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("tune", "grade", "selftest"))
    ap.add_argument("rally", type=int, nargs="?")
    ap.add_argument("--W", type=float, default=None)
    ap.add_argument("--R", type=float, default=None)
    ap.add_argument("--M", type=int, default=M_TOP)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
        return
    if a.cmd == "tune":
        tune()
        return
    if a.rally is None:
        raise SystemExit("grade needs a rally")
    if a.rally in EVAL_RALLIES:
        if a.W is not None or a.R is not None or a.M != M_TOP:
            raise SystemExit("protocol: no knob overrides on evaluation "
                             "rallies — the tune choice is frozen")
        if not TUNE_JSON.exists():
            raise SystemExit("protocol: run `fusion.py tune` first")
        tj = json.loads(TUNE_JSON.read_text())
        if tj.get("dead"):
            raise SystemExit("protocol: tune verdict is DEAD — r9/r10 "
                             "not run")
        W, R = tj["W_TRAIL"], tj["R_TRAIL"]
    else:
        tj = json.loads(TUNE_JSON.read_text()) if TUNE_JSON.exists() else {}
        W = a.W if a.W is not None else tj.get("W_TRAIL", 12.0)
        R = a.R if a.R is not None else tj.get("R_TRAIL", 16.0)
    run_grade(a.rally, W, R, a.M)


if __name__ == "__main__":
    main()
