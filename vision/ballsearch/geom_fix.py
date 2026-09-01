"""geom_fix — corridor GEOMETRY fix for the incumbent tracker
(HANDOFF next-thread to-do #1, 2026-09-01), tuned on r6/r7 only under
a rule written here BEFORE any number, then one shot on r9/r10.

What geom_lab.py measured on the train rallies (the reason for each
knob; numbers in swing_explore_notes.md):
  * out-of-window clicks are 18-34% of misses, most WITH a candidate
    within 12 px; they are bounces dropping BELOW the chord and lobs
    ABOVE it (vertical overshoot beyond wy: median 31-103 px, p90
    120-273). The chord-box height wy = min(170, 55 + 0.3 L) has no
    duration term and its cap binds on long chords.
      -> WY_CAP, KT:  wy' = min(WY_CAP, 55 + 0.3 L + KT * T)
  * a taller window pushes those excursions (far from the chord by
    construction) out of a CENTER-ranked K=14 pool; ranking the pool
    by learned p admits every train true candidate at K=14.
      -> POOL: "center" (incumbent) | "p" (cdp.POOL_BY_P)
  * endpoint error |A - first click| / |B - last click| is 100-330 px
    on several corridors (contact time / paddle choice), past the DP's
    END_R = 70 anchoring radius AND outside the window at the ends.
      -> EP: the window gains a pad of EP px (x and y) that tapers
         linearly to 0 over the first/last PAD_S = 0.25 s of the
         corridor, and END_R becomes 70 + EP.
Selection (a candidate existed, the chain took another / skipped) is
the largest stratum (38-51%) but the true candidate is already in the
pool 97%+ of the time — that is a DP-cost question (W_P_SOFT vs
W_GAP), out of scope here and already tuned.

PROTOCOL (frozen before numbers): grid WY_CAP {170, 260, 400} x KT
{0, 40, 80} x EP {0, 60, 120} x POOL {center, p}; panel = r6 + r7 x
prod + oracle, cross-fold p, dp-ccS+body W_P_SOFT = 25. Cell (170, 0,
0, center) must reproduce the incumbent 376 @ 0.633 exactly. Rule:
max total r@12 s.t. pooled prec@12 >= incumbent; ties broken by fewest
knobs changed from the incumbent, then smaller WY_CAP, KT, EP, center
before p. None -> DEAD, `grade 9|10` refuses. r9/r10 are graded ONCE
with the frozen cell, same metric, displaced-anchor nulls (seed
20260901), strata table, incumbent alongside.

Usage: python3 geom_fix.py tune | grade <rally> [--cap --kt --ep --pool]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import corridor_dp as cdp                                   # noqa: E402
import spaghetti as spag                                    # noqa: E402
from claim_lab import load, paddle_series                   # noqa: E402
from corridor_lab import (load_truth, prod_contacts, corridors,  # noqa
                          window_at as _window_at, decode_recall)

SP = Path(__file__).parent
TUNE_JSON = SP / "geom_tune.json"
W_P_SOFT = 25.0
END_R0 = cdp.END_R
PAD_S = 0.25
GRID_CAP = (170.0, 260.0, 400.0)
GRID_KT = (0.0, 40.0, 80.0)
GRID_EP = (0.0, 60.0, 120.0)
GRID_POOL = ("center", "p")
INCUMBENT = dict(cap=170.0, kt=0.0, ep=0.0, pool="center")
NULL_SEED = 20260901
EVAL_RALLIES = (9, 10)

_EP = 0.0


def window_pad(cor, t):
    """incumbent window_at plus the endpoint pad (tapered)."""
    cx, cy, wx, wy = _window_at(cor, t)
    if _EP:
        ta, tb = cor[0], cor[1]
        e = min(t - ta, tb - t)
        pad = _EP * max(0.0, 1.0 - e / PAD_S)
        wx, wy = wx + pad, wy + pad
    return cx, cy, wx, wy


def reshape(cors, cap, kt):
    out = []
    for ta, tb, A, B, wx, wy in cors:
        L = float(np.hypot(B[0] - A[0], B[1] - A[1]))
        out.append((ta, tb, A, B, wx, min(cap, 55.0 + 0.3 * L + kt * (tb - ta))))
    return out


def configure(cap, kt, ep, pool):
    global _EP
    _EP = float(ep)
    cdp.END_R = END_R0 + float(ep)
    cdp.POOL_BY_P = (pool == "p")
    cdp.W_P_SOFT = W_P_SOFT
    cdp.window_at = window_pad if ep else _window_at


def context(rally):
    pxs = "_x" if rally in (6, 7) else ""
    c = load(rally)
    series = paddle_series(c["npz"])
    truth = load_truth(rally)
    t0 = c["t0"]
    f_lo = int((c["serve"] - 0.4 - t0) * 60)
    f_hi = int((c["end"] + 0.2 - t0) * 60)
    cc = spag.cands_cached(rally, f_lo, f_hi, 14, "cc", lrn=True, pxs=pxs)
    arms = (("prod", corridors(c, series, prod_contacts(c, series, 0.5))),
            ("oracle", corridors(c, series, list(c["imps"]))))
    return dict(rally=rally, c=c, truth=truth, t0=t0, cc=cc, pxs=pxs,
                body=cdp.body_points(c, f_lo, f_hi),
                dec=decode_recall(c, truth), arms=arms)


def grade(track, truth, t0, dec):
    h12 = have = added = 0
    for (t, tx, ty, vis), d in zip(truth, dec):
        f = int(round((t - t0) * 60))
        p = track.get(f) or track.get(f - 1) or track.get(f + 1)
        if p is None:
            continue
        have += 1
        ok = float(np.hypot(p[0] - tx, p[1] - ty)) <= cdp.R_MAIN
        h12 += ok
        added += ok and not d
    return h12, have, added


def run_cell(ctxs, cap, kt, ep, pool, disp=None):
    configure(cap, kt, ep, pool)
    tot = dict(h12=0, have=0, added=0)
    per = []
    for ctx in ctxs:
        for arm, cors in ctx["arms"]:
            cs = reshape(cors, cap, kt)
            track = cdp.build_track(ctx["cc"], cs, ctx["t0"], disp=disp,
                                    body=ctx["body"])
            h, v, a = grade(track, ctx["truth"], ctx["t0"], ctx["dec"])
            tot["h12"] += h
            tot["have"] += v
            tot["added"] += a
            per.append(f"r{ctx['rally']}-{arm} {h}/{v}")
    configure(**INCUMBENT)
    tot["prec"] = tot["h12"] / max(1, tot["have"])
    return tot, per


def nchanged(cell):
    return sum(1 for k in INCUMBENT if cell[k] != INCUMBENT[k])


def tune():
    ctxs = [context(6), context(7)]
    for ctx in ctxs:
        print(f"rally {ctx['rally']}: {len(ctx['truth'])} clicks, decode@12"
              f" {sum(ctx['dec'])}/{len(ctx['dec'])}, p-cache '{ctx['pxs']}'")
    inc, per = run_cell(ctxs, **INCUMBENT)
    print(f"INCUMBENT                     total r@12 {inc['h12']:4d}  prec@12"
          f" {inc['prec']:.3f}  ADDED {inc['added']:3d}  | " + "  ".join(per))
    assert inc["h12"] == 376 and abs(inc["prec"] - 0.633) < 5e-4, \
        "incumbent cell does not reproduce 376 @ 0.633 — environment drift"
    rows = []
    for cap in GRID_CAP:
        for kt in GRID_KT:
            for ep in GRID_EP:
                for pool in GRID_POOL:
                    cell = dict(cap=cap, kt=kt, ep=ep, pool=pool)
                    t, per = run_cell(ctxs, **cell)
                    rows.append(dict(**cell, **t))
                    print(f"cap={cap:3.0f} kt={kt:2.0f} ep={ep:3.0f} "
                          f"{pool:6s}  total r@12 {t['h12']:4d}  prec@12 "
                          f"{t['prec']:.3f}  ADDED {t['added']:3d}  | "
                          + "  ".join(per))
    ok = [r for r in rows if r["prec"] >= inc["prec"] - 1e-9
          and r["h12"] > inc["h12"]]
    verdict = dict(incumbent=inc, grid=rows, rule=(
        "max total r@12 over r6/r7 x prod/oracle s.t. pooled prec@12 >= "
        "incumbent; ties fewest knobs changed, then smaller cap, kt, ep, "
        "center before p; none -> dead"))
    if not ok:
        print("VERDICT: no cell beats the incumbent under the rule — "
              "geometry fix DEAD, do not run r9/r10")
        verdict.update(dead=True)
    else:
        best = sorted(ok, key=lambda r: (-r["h12"], nchanged(r), r["cap"],
                                         r["kt"], r["ep"],
                                         0 if r["pool"] == "center" else 1))[0]
        print(f"VERDICT: cap={best['cap']:g} kt={best['kt']:g} "
              f"ep={best['ep']:g} pool={best['pool']} (total r@12 "
              f"{best['h12']} vs incumbent {inc['h12']}, prec "
              f"{best['prec']:.3f} >= {inc['prec']:.3f}) — freeze and "
              f"one-shot r9/r10")
        verdict.update(dead=False, cap=best["cap"], kt=best["kt"],
                       ep=best["ep"], pool=best["pool"])
    TUNE_JSON.write_text(json.dumps(verdict, indent=1))
    print(f"wrote {TUNE_JSON}")


def strata(ctx, cors, cc):
    out = []
    for (t, tx, ty, vis) in ctx["truth"]:
        cor = next((co for co in cors if co[0] <= t <= co[1]), None)
        if cor is None:
            out.append("nocor")
            continue
        cx, cy, wx, wy = cdp.window_at(cor, t)
        if abs(tx - cx) > wx or abs(ty - cy) > wy:
            out.append("outwin")
            continue
        f = int(round((t - ctx["t0"]) * 60))
        near = any(np.hypot(c_[0] - tx, c_[1] - ty) <= cdp.R_MAIN
                   for df in (-1, 0, 1) for c_ in cc.get(f + df, ()))
        out.append("cand" if near else "nocand")
    return out


def run_grade(rally, cell):
    ctx = context(rally)
    truth, t0, dec = ctx["truth"], ctx["t0"], ctx["dec"]
    print(f"rally {rally}: {len(truth)} V/S clicks, decode@12 "
          f"{sum(dec)}/{len(dec)}; geometry cell {cell} (p-cache "
          f"'{ctx['pxs']}', W_P_SOFT={W_P_SOFT:g})")
    rng = np.random.default_rng(NULL_SEED)
    for arm, cors in ctx["arms"]:
        print(f"== {arm}: {len(cors)} corridors")
        configure(**INCUMBENT)
        inc = cdp.build_track(ctx["cc"], cors, t0, body=ctx["body"])
        cdp.score(inc, truth, t0, dec, "dp-ccS+body")
        st_inc = strata(ctx, cors, ctx["cc"])
        configure(**cell)
        cs = reshape(cors, cell["cap"], cell["kt"])
        fix = cdp.build_track(ctx["cc"], cs, t0, body=ctx["body"])
        cdp.score(fix, truth, t0, dec, "geom-fix")
        for vis in ("V", "S"):
            tt = [x for x in truth if x[3] == vis]
            dd = [d for x, d in zip(truth, dec) if x[3] == vis]
            cdp.score(fix, tt, t0, dd, f"  fix[{vis}]")
        st_fix = strata(ctx, cs, ctx["cc"])
        for kk in range(2):
            d = (float(rng.uniform(160, 240)) * rng.choice([-1, 1]),
                 float(rng.uniform(80, 140)) * rng.choice([-1, 1]))
            nul = cdp.build_track(ctx["cc"], cs, t0, disp=d,
                                  body=ctx["body"])
            cdp.score(nul, truth, t0, dec, f"null{kk}")
        configure(**INCUMBENT)
        print("    stratum (incumbent geometry)   n   inc   fix"
              "  | (fix geometry)   n")
        for name in ("cand", "nocand", "outwin", "nocor"):
            idx = [i for i, s in enumerate(st_inc) if s == name]
            idx2 = [i for i, s in enumerate(st_fix) if s == name]
            if not idx and not idx2:
                continue
            tt = [truth[i] for i in idx]
            dd = [dec[i] for i in idx]
            hi = grade(inc, tt, t0, dd)[0]
            hf = grade(fix, tt, t0, dd)[0]
            print(f"    {name:8s}                     {len(idx):5d} {hi:5d}"
                  f" {hf:5d}  |               {len(idx2):5d}")
        # per-corridor
        hi_ = spag.hits(inc, truth, t0)
        hf_ = spag.hits(fix, truth, t0)
        print("-- corridors: t-span dur | wy inc/fix | clicks: inc / fix")
        for co, c2 in zip(cors, cs):
            ta, tb = co[0], co[1]
            idx = [i for i, (t, *_) in enumerate(truth) if ta <= t <= tb]
            n_ = lambda h: sum(1 for i in idx if h[i] is not None  # noqa
                               and h[i] <= cdp.R_MAIN)
            print(f"  {ta:7.2f}-{tb:7.2f} {tb - ta:4.2f}s | {co[5]:4.0f}/"
                  f"{c2[5]:4.0f} | {len(idx):3d}: {n_(hi_):3d} / {n_(hf_):3d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("tune", "grade"))
    ap.add_argument("rally", type=int, nargs="?")
    ap.add_argument("--cap", type=float)
    ap.add_argument("--kt", type=float)
    ap.add_argument("--ep", type=float)
    ap.add_argument("--pool", choices=GRID_POOL)
    a = ap.parse_args()
    if a.cmd == "tune":
        tune()
        return
    if a.rally is None:
        raise SystemExit("grade needs a rally")
    over = {k: v for k, v in dict(cap=a.cap, kt=a.kt, ep=a.ep,
                                  pool=a.pool).items() if v is not None}
    if a.rally in EVAL_RALLIES:
        if over:
            raise SystemExit("protocol: no knob overrides on evaluation "
                             "rallies — the tune choice is frozen")
        if not TUNE_JSON.exists():
            raise SystemExit("protocol: run `geom_fix.py tune` first")
        tj = json.loads(TUNE_JSON.read_text())
        if tj.get("dead"):
            raise SystemExit("protocol: tune verdict is DEAD — r9/r10 "
                             "not run")
        cell = {k: tj[k] for k in ("cap", "kt", "ep", "pool")}
    else:
        tj = json.loads(TUNE_JSON.read_text()) if TUNE_JSON.exists() else {}
        cell = {k: over.get(k, tj.get(k, INCUMBENT[k]))
                for k in ("cap", "kt", "ep", "pool")}
    run_grade(a.rally, cell)


if __name__ == "__main__":
    main()
