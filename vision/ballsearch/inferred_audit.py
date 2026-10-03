"""inferred_audit — where does gap fill GUESS, and did it need to?

Owner callout 2026-09-03: "at one point I think we inferred ball position
near the paddle because of so many S clicks -- want to make sure S's are
being counted there. An S is better than a guess."

Gap fill v2 emits "inferred" frames: positions with no track behind them.
This asks two questions of every one of them.

  1. Did the owner see the ball there?  Cross the inferred frames against
     the click path by kind (V / S / I / none), and report how far the
     guess landed from the click that was actually sitting there.  Also
     counts the ones within 0.20 s of a contact -- the "near the paddle"
     half of the callout.

  2. Was there a candidate to take?  Split those clicks on decode@12:
     TRUE means the candidate decoder had a usable blob and path-first
     declined it (a selection failure); FALSE means the detector had
     nothing (an emission failure).  The split says which half of the
     stack to work on, and it is different for V and S clicks.

Measured 2026-09-03 over r2/r3/r4/r6/r7/r17: 414 inferred frames, 78% of
them on a frame the owner could see and click, 32% within 0.20 s of a
contact.  Of the guessed-at clicks, 45% HAD a candidate -- and among S
clicks that rises to 57/85 = 67%, against 92/243 = 38% for V.  So on
streaks the tracker most often walks past evidence already in hand, which
is what the owner's rule is pointing at.

Clicks are ground truth, not a run-time input, so "count the S there"
cannot mean "use the click".  It means the S stratum carries recoverable
evidence that the current training rule (S = ignore zone, never a
positive) forbids the emission model from learning.

Usage:  python3 inferred_audit.py 2 3 4 6 7 17
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, "/home/user/pickleball/vision")

import pathfirst as pf          # noqa: E402
import gapfill as gf            # noqa: E402
import corridor_dp as cdp       # noqa: E402

DV = Path("/home/user/pickleball/data/vision")
CON = DV / "contact_labels_chicago0725.csv"
NEAR_CONTACT = 0.20             # s: "near the paddle"


def contacts(rally):
    man, pre = [], []
    for r in csv.DictReader(open(CON)):
        if int(r["rally_cum"]) != rally or r.get("contact", "1") == "0":
            continue
        t = float(r["t_refined_s"] or r["t_tap_s"])
        (man if r["source"] in ("manual", "divergent") else pre).append(t)
    return np.array(sorted(man or pre))


def main(rallies):
    pc = json.loads(pf.TUNE_JSON.read_text())
    g2 = json.loads((SP / "gapfill_tune2.json").read_text())
    g2 = dict(gap_max=float(g2["gap_max"]), d_meet=float(g2["d_meet"]))
    kinds = {"V": 0, "S": 0, "I": 0, "none": 0}
    cand = {}
    for R in rallies:
        ctx = pf.context(R)
        cdp.W_P_SOFT = 25.0
        pf.run(ctx, pc["p_seed"], pc["s_min"], int(pc["gap"]))
        res = gf.run(ctx, g2)
        inf, t0 = set(res["inferred"]), ctx["t0"]

        clicked = {}
        for r in csv.DictReader(open(DV / f"ball_path_r{R}.csv")):
            if not r["x"]:
                continue
            f = int(round((float(r["t_s"]) - t0) * 60))
            clicked[f] = (r["vis"], float(r["x"]), float(r["y"]))
        dec_by_f = {}
        for (t, x, y, v), d in zip(ctx["truth"], ctx["dec"]):
            dec_by_f[int(round((t - t0) * 60))] = d

        cs = contacts(R)
        n = {"V": 0, "S": 0, "I": 0, "none": 0}
        off = {"V": [], "S": []}
        near = 0
        for f in sorted(inf):
            hit = clicked.get(f) or clicked.get(f - 1) or clicked.get(f + 1)
            k = hit[0] if hit else "none"
            n[k] = n.get(k, 0) + 1
            if hit and k in ("V", "S"):
                gx, gy = res["track"][f]
                off[k].append(float(np.hypot(gx - hit[1], gy - hit[2])))
                d = dec_by_f.get(f)
                if d is None:
                    d = dec_by_f.get(f - 1, dec_by_f.get(f + 1))
                if d is not None:
                    key = (k, "candidate existed" if d else "no candidate")
                    cand[key] = cand.get(key, 0) + 1
            if len(cs) and np.min(np.abs(cs - (t0 + f / 60.0))) <= NEAR_CONTACT:
                near += 1
        for k in kinds:
            kinds[k] += n[k]
        tot = max(sum(n.values()), 1)
        print(f"r{R:2d}: {sum(n.values())} inferred frames -> "
              f"V {n['V']} ({n['V']/tot:.0%})  S {n['S']} ({n['S']/tot:.0%})  "
              f"I {n['I']}  no click {n['none']}   "
              f"within {NEAR_CONTACT}s of a contact: {near}")
        for k in ("V", "S"):
            if off[k]:
                a = np.array(off[k])
                print(f"      guess vs the {k} click that was there: "
                      f"median {np.median(a):5.1f} px  within 12px "
                      f"{np.mean(a <= 12):.0%}  n={len(a)}")

    tot = max(sum(kinds.values()), 1)
    print(f"\nPOOLED {tot} inferred frames: "
          + "  ".join(f"{k} {v} ({v/tot:.0%})" for k, v in kinds.items()))
    print(f"  -> {(kinds['V']+kinds['S'])/tot:.0%} of the guesses land where the "
          f"owner could see the ball; {kinds['S']/tot:.0%} on a STREAK.")
    ct = max(sum(cand.values()), 1)
    print("\ncould the guess have been avoided?")
    for k in sorted(cand):
        print(f"  {k[0]} clicks, {k[1]:16s}: {cand[k]:3d}  ({cand[k]/ct:.0%})")
    for kind in ("V", "S"):
        e = cand.get((kind, "candidate existed"), 0)
        m = e + cand.get((kind, "no candidate"), 0)
        if m:
            print(f"  {kind}: {e}/{m} = {e/m:.0%} had a candidate path-first declined")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
