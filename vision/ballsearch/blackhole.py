"""How big is the "black hole"? (owner question, 2026-09-03)

    "the ball goes into the black hole of [where the] player changes
     direction, off camera, then comes out ... not through tracking
     because it's not there, but through inference?"

Two separate holes get called the same thing, and they have different sizes:

  HUMAN hole   frames where the owner, with a loupe and no time limit,
               could not place the ball cleanly (I = hidden behind a body,
               N = can't place at all).
  MACHINE hole frames the owner DID place (V / S) that the adopted stack
               either misses or never claims.

This script measures both on the same frames, stratified by distance to the
nearest owner-labeled contact -- because the owner's claim is specifically
that the hole sits AT the direction change.

Scoring is the frozen CHECK-1 scorer (gate_checks.py, 25 px V / 40 px S), so
these numbers are comparable to every other graded read in the thread.  It
runs the ADOPTED product (path-first + gap-fill v2), tunes nothing, writes
nothing back, and touches no seal.  r9 / r10 print separately as the spent
evaluation panel.

    python3 blackhole.py
"""
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import pathfirst as pf                                       # noqa: E402
import gapfill                                               # noqa: E402
from gate_checks import TOL                                  # noqa: E402
from geom_speed import contacts                              # noqa: E402

PATHS = Path("/home/user/pickleball/data/vision")
TRAIN = [6, 7, 17]
EVAL = [9, 10]
BANDS = ((0.00, 0.10, "at the contact   (<=0.10 s)"),
         (0.10, 0.25, "just after/before(0.10-0.25)"),
         (0.25, 0.60, "mid-flight       (0.25-0.60)"),
         (0.60, 9.99, "far from contact (>0.60 s)"))


def rally_rows(rally):
    ctx = pf.context(rally)
    res = gapfill.product(ctx)
    track, t0 = res["track"], ctx["t0"]
    cs = [s["t"] for s in contacts(rally) if s["type"] != "whiff"]
    imps = ctx["c"]["imps"]
    lo, hi = imps[0], imps[-1] + 0.5
    tol = dict(TOL)
    rows = []
    with open(PATHS / f"ball_path_r{rally}.csv") as f:
        for r in csv.DictReader(f):
            t = float(r["t_s"])
            if not (lo <= t <= hi):
                continue
            d = min((abs(t - c) for c in cs), default=9.9)
            kind = r["vis"] or "N"
            hit = claim = None
            if r["x"] and kind in tol:
                f0 = round((t - t0) * pf.FPS)
                best = min((math.hypot(track[g][0] - float(r["x"]),
                                       track[g][1] - float(r["y"]))
                            for g in (f0 - 1, f0, f0 + 1) if g in track), default=1e9)
                claim = best < 1e9
                hit = best <= tol[kind]
            rows.append(dict(rally=rally, t=t, d=d, kind=kind, hit=hit, claim=claim))
    return rows


def report(name, rows):
    print(f"\n=== {name}: {len(rows)} owner-judged frames on the gate panel")
    print(f"  {'band':28s} {'frames':>7} {'human hole':>11} {'machine claims':>15} "
          f"{'machine hits':>13}")
    for lo, hi, lab in BANDS:
        b = [r for r in rows if lo <= r["d"] < hi]
        if not b:
            continue
        hole = sum(r["kind"] in ("I", "N") for r in b)
        sc = [r for r in b if r["hit"] is not None]
        cl = sum(r["claim"] for r in sc)
        ht = sum(r["hit"] for r in sc)
        print(f"  {lab:28s} {len(b):7d} {100 * hole / len(b):9.1f}% "
              f"{100 * cl / max(len(sc), 1):13.1f}% {100 * ht / max(len(sc), 1):11.1f}%"
              f"   (n={len(sc)} placed)")
    hole = sum(r["kind"] in ("I", "N") for r in rows)
    sc = [r for r in rows if r["hit"] is not None]
    print(f"  {'ALL':28s} {len(rows):7d} {100 * hole / len(rows):9.1f}% "
          f"{100 * sum(r['claim'] for r in sc) / max(len(sc), 1):13.1f}% "
          f"{100 * sum(r['hit'] for r in sc) / max(len(sc), 1):11.1f}%   (n={len(sc)} placed)")
    # how long are the machine's misses, in a row?
    runs, cur = [], 0
    for r in sorted(rows, key=lambda r: (r["rally"], r["t"])):
        if r["hit"] is False or (r["hit"] is None and r["kind"] in ("I", "N")):
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    if runs:
        a = np.array(runs)
        print(f"  unrecovered runs (machine miss or human hole): n={len(a)} "
              f"median {np.median(a):.0f} fr ({np.median(a) / 30:.2f} s) "
              f"p90 {np.percentile(a, 90):.0f} max {a.max()}")
        print(f"  run-length histogram (30 fps frames): "
              f"{dict(sorted(Counter(a.tolist()).items()))}")


def main():
    tr = []
    for r in TRAIN:
        tr += rally_rows(r)
    report("TRAIN r6+r7+r17", tr)
    ev = []
    for r in EVAL:
        ev += rally_rows(r)
    report("EVAL r9+r10 (spent evaluation; read only)", ev)


if __name__ == "__main__":
    main()
