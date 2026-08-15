"""Gate A, step 1 result: can a human find the ball at all?

Gate A's plan was label -> fine-tune -> score detector recall on the fast
stratum, kill below 0.8.  Step 1 answered the question before step 2 ran,
in a way the gate did not anticipate: on 416 sampled frames the labeler —
5x loupe, blink-compare against the previous frame, no time limit —
could not locate the ball in 48% of them.

That number needs one correction before it means anything, and the
labeler supplied it: "a decent chunk of the can't-find were between
points."  True.  Windows for rallies 17+ are built as t0 = t1 - duration,
and the referee log's duration includes a ~6 s pre-serve lead (measured
on the 16 hand-marked rallies), which the 1.2 s head-pad barely dents.
So the raw 52% mixes live play with dead time.

Binning by seconds-since-window-open separates them cleanly and the curve
is exactly the shape the dead-time story predicts — 14% in the first 3 s,
rising to a plateau once play starts.  The plateau IS the measurement:

    in-play findability 64.1%  [59%, 69%]   n=306

Whole CI below Gate A's 0.8 kill line.  Two things make it, if anything,
generous: the `fast` stratum proxy (division x tempo) did not produce a
detectable difficulty split (z = -0.62 vs random), so this is overall
findability and true fast-shot frames are likely harder; and some misses
are the broadcast framing the ball out, which the branch already records
as a permanent bias that "binds even a perfect detector."

Why human findability bounds the program rather than merely describing
it: ground truth can only be created where a human can see the ball.  On
36% of in-play frames there is nothing to train on and nothing to score
against, so a model's claims there are unfalsifiable — and unverifiable
machine labels are precisely what poisoned the earlier fine-tune (42%
kitchen-band vs 14% base).  Even granting a perfect detector on every
findable frame, ~64% of ball positions come back, and the misses
concentrate where the ball is fast or occluded: at the contacts, which
is what the whole program wanted.

    python3 vision/ball_visibility.py
"""
from __future__ import annotations

import argparse
import csv
import math

D = "data/vision"
PRE_SERVE_S = 6.0        # measured lead baked into the log's duration column
PLAY_SPAN = 1.3          # s per shot, for the marked-rally in-play split
KILL = 0.80              # Gate A: below this, the wall is physical


def wilson(v, n):
    if not n:
        return 0.0, 0.0, 0.0
    p = v / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - hw, c + hw


def line(lab, g):
    v = sum(int(r["visible"]) for r in g)
    p, lo, hi = wilson(v, len(g))
    print(f"  {lab:<34} {v:>3}/{len(g):<3} = {p:5.1%}  [{lo:.0%}, {hi:.0%}]")
    return p, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=f"{D}/ball_labels_chicago0725.csv")
    ap.add_argument("--windows", default=f"{D}/rally_windows_chicago0725_v4.csv")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.labels)))
    W = {int(r["rally_cum"]): r for r in csv.DictReader(open(a.windows))}
    elapsed = {r["frame_file"]: float(r["t_video"])
               - float(W[int(r["rally_cum"])]["t0s"]) for r in rows}

    print(f"{len(rows)} frames labeled by hand\n")
    print("RAW, uncorrected — mixes live play with pre-serve dead time:")
    line("all frames", rows)
    for s in ("fast", "random"):
        line(f"  stratum {s}", [r for r in rows if r["stratum"] == s])

    print("\nVisibility vs seconds since the window opened (rallies 17+):")
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 16), (16, 999)):
        g = [r for r in rows if int(r["rally_cum"]) > 16
             and lo <= elapsed[r["frame_file"]] < hi]
        if g:
            p, _, _ = wilson(sum(int(r["visible"]) for r in g), len(g))
            print(f"  {lo:>2}-{hi if hi < 999 else '+':<4}s {len(g):>4} frames  "
                  f"{p:5.1%}  {'#' * int(p * 40)}")

    print("\nIN-PLAY ONLY (past the pre-serve lead, or a serve-pinned rally):")
    g = [r for r in rows if elapsed[r["frame_file"]] >= PRE_SERVE_S
         or int(r["rally_cum"]) <= 16]
    p, lo, hi = line("human findability", g)

    print(f"\nGate A kill line: {KILL:.0%}.  Upper end of the CI: {hi:.0%}.")
    print("VERDICT: " + ("the wall is PHYSICAL — the ball is not recoverable "
                         "from this footage often enough to build ground "
                         "truth, said with a measurement."
                         if hi < KILL else
                         "inconclusive — CI crosses the kill line."))


if __name__ == "__main__":
    main()
