"""Measured shot categories for the 3D replay (user taxonomy, 2026-08-31).

The user renamed the display categories to three speed-defined
buckets — "dink" (slow, often bounces in the kitchen), "speed-up"
(the FIRST fast shot in a sequence), "hand battle" (consistent fast
shots) — with an explicit worry: hard dinking must not register as
attacking. This classifier derives them from PHYSICS (per-shot
outgoing speed off the court3d path) rather than hand labels, with a
battle-state hysteresis, and on rally 1 it reproduces the structure
of the hand labels exactly: both labeled speed-ups are the two rule
speed-ups; smash/counter spans map to hand battle; the one 28.6 mph
firm dink that goes nowhere stays a dink.

Rules (frozen 2026-08-31, receipts in swing_explore_notes.md):
  - outgoing speed = median 3D path speed 0.05-0.30 s after contact;
  - FAST >= 24 mph (rally-1 distribution gap: dinks 13-21, attacks
    25-48); HOT >= 30 mph;
  - serve/return keep their structural names;
  - from calm, a fast shot is a SPEED-UP only if it is HOT or the
    next shot is also fast (a lone moderate-fast push with no
    follow-up is a firm DINK — the hard-dinking guard);
  - while the battle is live, fast shots are HAND BATTLE, and a
    single soft block is absorbed (still hand battle) if the next
    shot is fast; two consecutive slow shots end the battle;
  - the pressure meter ("who's winning?") in the replay counts ONLY
    speed-up/hand-battle shots, weighted (mph - 20) with a 2 s decay
    — it is attack heat, not a calibrated win probability.

Usage:
    python3 vision/shot_categories.py               # rally1_show.json
    python3 vision/shot_categories.py --show PATH.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "vision"
FAST_MPH = 24.0
HOT_MPH = 30.0
FT_S_TO_MPH = 0.681818


def out_speeds(show):
    P = np.array(show["path"])
    V = []
    for i in range(1, len(P)):
        dt = P[i, 0] - P[i - 1, 0]
        if 0 < dt < 0.1:
            V.append((P[i, 0],
                      float(np.linalg.norm(P[i, 1:] - P[i - 1, 1:]) / dt)))

    def out(t):
        w = [s for tt, s in V if t + 0.05 <= tt <= t + 0.30]
        return round(float(np.median(w)) * FT_S_TO_MPH, 1) if w else None
    return [out(im["t"]) for im in show["impacts"]]


def classify(speeds):
    """Categories for a rally's impact list given per-shot mph
    (None = unmeasurable; inherits battle state or defaults dink)."""
    cats, state = [], "SLOW"
    n = len(speeds)
    for k in range(n):
        m = speeds[k]
        fast = m is not None and m >= FAST_MPH
        nxt = speeds[k + 1] if k + 1 < n else None
        nxt_fast = nxt is not None and nxt >= FAST_MPH
        if k == 0:
            cats.append("serve")
            continue
        if k == 1:
            cats.append("return")
            continue
        if state == "SLOW":
            if fast and (m >= HOT_MPH or nxt_fast):
                cats.append("speed-up")
                state = "FAST"
            else:
                cats.append("dink")
        else:
            if fast or nxt_fast or m is None:
                cats.append("hand battle")
            else:
                cats.append("dink")
                state = "SLOW"
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", default=str(DATA / "rally1_show.json"))
    a = ap.parse_args()
    show = json.load(open(a.show))
    sp = out_speeds(show)
    cats = classify(sp)
    for k, im in enumerate(show["impacts"]):
        mm = f"{sp[k]:5.1f}" if sp[k] else "  n/a"
        print(f"{k+1:>2} {im['t']:7.2f} {im['hitter'].split()[-1]:<12} "
              f"{mm} mph  {cats[k]:<12} (label: {im.get('type','?')})")


if __name__ == "__main__":
    main()
