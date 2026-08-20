#!/usr/bin/env python3
"""Score ranked cell-calls on a packed grid against an ANSWER_KEY_LOC.csv.

WHY THIS EXISTS (2026-08-20). The arm6m round was scored with free
calls and no null, and that is not a measurement: 75 calls across 9
windows at a +/-0.5s tolerance covers most of the drawable video, so a
caller who names cells at random scores about what I scored. The rule
below fixes the two defects.

  1. A PLACEMENT NULL. Calls stay free and unbudgeted -- telling the
     caller how many contacts a window holds is the count leak that had
     to be disclosed on 2026-08-19, so it is not on the table. Instead
     every arm is scored against a permutation null that makes the SAME
     NUMBER OF CALLS the caller made, placed uniformly at random on
     distinct cells. Calling every cell then earns 100% recall and zero
     lift, which is the correct verdict on that strategy.

  2. TWO TOLERANCES. +/-0.5s is the project's metric of record and the
     only number comparable to the pose pipeline's 45.7%, the 3x3 VLM's
     93% and the tracker's 67%, so it stays the headline. But at 0.15s
     cell spacing it is +/-3.3 cells, i.e. one call covers a fifth of a
     6x6 window -- so a sharp read at +/-1 cell is reported beside it.

Matching is greedy one-to-one by |dt| (a call spends itself on one
contact), which is what `matched` means in ball_track.py's scoring.

Calls file: CSV with window,rank,cell  (cell 1-based, rank 1 = most
confident). Ranks let precision-at-k be read off later; they do not
affect recall.
"""
import argparse, csv, random
from collections import defaultdict

STEP_S = 0.15


def load_key(path):
    out = {}
    for r in csv.DictReader(open(path)):
        offs = [float(x) for x in r["offsets_s"].split("|") if x]
        out[r["window"]] = {
            "offsets": offs,
            "grid": int(r["grid"]),
            "cells": int(r["grid"]) ** 2,
            "hitters": [h for h in r.get("hitters", "").split("|") if h],
            "paces": [p for p in r.get("paces", "").split("|") if p],
            "marked": r.get("marked_cells", "") or "",
        }
    return out


def load_calls(path):
    out = defaultdict(list)
    for r in csv.DictReader(open(path)):
        out[r["window"]].append((int(r["rank"]), int(r["cell"])))
    return {w: [c for _rk, c in sorted(v)] for w, v in out.items()}


def match(call_cells, offsets, tol):
    """greedy one-to-one; returns (n_matched, [(cell, offset)] pairs)."""
    pairs = sorted(
        ((abs((c - 1) * STEP_S - o), i, j)
         for i, c in enumerate(call_cells) for j, o in enumerate(offsets)
         if abs((c - 1) * STEP_S - o) <= tol))
    uc, uo, got = set(), set(), []
    for _d, i, j in pairs:
        if i in uc or j in uo:
            continue
        uc.add(i); uo.add(j)
        got.append((call_cells[i], offsets[j]))
    return len(got), got


def score(calls, key, tol):
    hit = tru = cal = 0
    per_window = {}
    for w, k in key.items():
        cc = calls.get(w, [])
        m, got = match(cc, k["offsets"], tol)
        hit += m; tru += len(k["offsets"]); cal += len(cc)
        per_window[w] = (m, len(k["offsets"]), len(cc), got)
    return hit, tru, cal, per_window


def null(calls, key, tol, draws, seed):
    """same call COUNT per window, uniformly placed on distinct cells."""
    rng = random.Random(seed)
    rates = []
    for _ in range(draws):
        hit = tru = 0
        for w, k in key.items():
            n = len(calls.get(w, []))
            cc = rng.sample(range(1, k["cells"] + 1), min(n, k["cells"]))
            m, _ = match(cc, k["offsets"], tol)
            hit += m; tru += len(k["offsets"])
        rates.append(hit / tru if tru else 0.0)
    rates.sort()
    return rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls")
    ap.add_argument("--key")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    key, calls = load_key(a.key), load_calls(a.calls)

    for label, tol in (("+/-0.5s (metric of record)", 0.5),
                       ("+/-1 cell (0.15s, sharp)", STEP_S)):
        hit, tru, cal, pw = score(calls, key, tol)
        rates = null(calls, key, tol, a.draws, a.seed)
        obs = hit / tru if tru else 0.0
        pct = 100.0 * sum(1 for r in rates if r < obs) / len(rates)
        mean = sum(rates) / len(rates)
        lo, hi = rates[int(.025 * len(rates))], rates[int(.975 * len(rates))]
        print(f"\n{label}")
        print(f"  recall     {hit}/{tru} = {obs:.1%}   "
              f"precision {hit}/{cal} = {hit / cal:.1%}" if cal else "")
        print(f"  null       {mean:.1%}  [{lo:.1%}, {hi:.1%}] "
              f"at the same {cal} calls")
        print(f"  LIFT       {obs - mean:+.1%}   observed sits at the "
              f"{pct:.1f}th percentile of the null")
        if pct < 95:
            print("  -> NOT distinguishable from placing the same number "
                  "of calls at random")

    # miss decomposition: was the ball marked where I missed?
    anymark = any(key[w]["marked"] for w in key)
    if anymark:
        _h, _t, _c, pw = score(calls, key, 0.5)
        found = {w: {round(o, 3) for _c, o in pw[w][3]} for w in pw}
        bins = defaultdict(int)
        for w, k in key.items():
            mk = k["marked"]
            for o in k["offsets"]:
                cell = int(round(o / STEP_S))
                near = any(mk[c] == "1" for c in
                           range(max(cell - 1, 0), min(cell + 2, len(mk))))
                bins[(round(o, 3) in found[w], near)] += 1
        print("\nmiss decomposition (tracker mark within +/-1 cell of "
              "the contact)")
        for f in (True, False):
            for n in (True, False):
                print(f"  called={str(f):5s} marked={str(n):5s}  "
                      f"{bins[(f, n)]:3d}")
        tb = bins[(False, False)]; rb = bins[(False, True)]
        if tb + rb:
            print(f"  -> of {tb + rb} misses, {rb} were MARKED "
                  f"(reading-bound) and {tb} unmarked (tracker-bound)")


def selftest():
    import os, tempfile
    d = tempfile.mkdtemp()
    kp, cp = os.path.join(d, "k.csv"), os.path.join(d, "c.csv")
    with open(kp, "w") as fh:
        fh.write("window,rally_cum,t0_s,span_s,grid,markers,n_contacts,"
                 "offsets_s,hitters,paces,marked_cells\n")
        # contacts at cells 1, 11, 21, 31 (0.00, 1.50, 3.00, 4.50 s)
        fh.write("w01.png,1,0,5.25,6,1,4,0.00|1.50|3.00|4.50,a|b|c|d,"
                 "fast|fast|slow|slow," + "1" * 36 + "\n")

    def run(cells):
        with open(cp, "w") as fh:
            fh.write("window,rank,cell\n")
            for i, c in enumerate(cells, 1):
                fh.write(f"w01.png,{i},{c}\n")
        key, calls = load_key(kp), load_calls(cp)
        h, t, c, _ = score(calls, key, STEP_S)
        rates = null(calls, key, STEP_S, 2000, 1)
        obs = h / t
        pct = 100.0 * sum(1 for r in rates if r < obs) / len(rates)
        return obs, sum(rates) / len(rates), pct

    obs, nul, pct = run([1, 11, 21, 31])
    assert obs == 1.0 and pct > 99, f"perfect caller: {obs} pct {pct}"
    print(f"  perfect 4 calls : recall {obs:.0%} null {nul:.0%} "
          f"pct {pct:.0f}  OK")

    obs, nul, pct = run(list(range(1, 37)))
    assert obs == 1.0, "call-everything should still recall 100%"
    assert abs(obs - nul) < 1e-9, "call-everything must have ZERO lift"
    assert pct < 95, "call-everything must not look significant"
    print(f"  all 36 cells    : recall {obs:.0%} null {nul:.0%} "
          f"lift {obs - nul:+.0%}  OK  <- the arm6m failure mode")

    obs, nul, pct = run([2, 9, 17, 25, 33])
    print(f"  5 near-misses   : recall {obs:.0%} null {nul:.0%} "
          f"pct {pct:.0f}")
    assert pct < 95, "a near-miss caller should not clear the null"
    print("selftest: ALL OK")


if __name__ == "__main__":
    main()
