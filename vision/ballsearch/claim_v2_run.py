"""Measure claiming v2 vs production (P0) on cheap metrics.

Per rally: bound recall vs taps, extras, false-claimed human bounces,
plus the pathdist AUC replication (real vs fake anchors).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import ball_replicate as br                        # noqa: E402
from claim_lab import (load, paddle_series, claim_production,   # noqa: E402
                       human_bounce_times, score, claim_v2,
                       dedupe_v2, anchor_reldist, claim_bounds_veto,
                       claim_bounds_veto2)
from hole_audit import turn_events, pose_peaks_all  # noqa: E402


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p < n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main():
    rallies = [int(x) for x in sys.argv[1:]] or [7, 9]
    for rally in rallies:
        c = load(rally)
        series = paddle_series(c["npz"])
        hb = human_bounce_times(c)
        imps = c["imps"]

        # pathdist separation on deduped anchors: real = near a tap
        kept = dedupe_v2(c, series)
        real, fake = [], []
        for a, r in kept:
            if not np.isfinite(r):
                continue
            (real if any(abs(a[0] - t) <= 0.15 for t in imps)
             else fake).append(r)
        print(f"== rally {rally}: {len(imps)} taps, "
              f"{len(c['turns'])} turns, {len(kept)} deduped anchors")
        print(f"   pathdist: real n={len(real)} med="
              f"{np.median(real):.2f}h  fake n={len(fake)} med="
              f"{(np.median(fake) if fake else float('nan')):.2f}h  "
              f"AUC {auc(real, fake):.3f}")

        # P0 baseline vs bounce-geometry veto
        anchors, matched = claim_production(c)
        for name, bounds in [
                ("P0        ", matched),
                ("veto next ", claim_bounds_veto(
                    c["turns"], c["angs"], c["timing_ref"], anchors,
                    mode="next")),
                ("veto drop ", claim_bounds_veto(
                    c["turns"], c["angs"], c["timing_ref"], anchors,
                    mode="drop"))] + [
                (f"v2 th={th}{'^' if vu else ' '}  ",
                 claim_bounds_veto2(c, series, anchors, th=th, vy_up=vu))
                for th in (0.5, 0.7, 1.0) for vu in (False, True)]:
            rec, n, extras, fb = score(c, bounds, hb)
            print(f"   {name}: bounds {len(bounds):2d}  recall {rec}/{n}"
                  f"  extras {len(extras)} ({fb} on human bounces)")


if __name__ == "__main__":
    main()
