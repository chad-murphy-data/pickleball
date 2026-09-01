"""Anchor-quality diagnostic: where do fakes out-z real contacts?

For each local rally: run the production anchor pipeline (pose peaks
+ blur gap-fill), match to owner taps at +/-0.15 s, and report the z
ORDERING quality — matched (real) vs unmatched (fake) z distributions,
top-K composition, and which real contacts are missed entirely.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
import hitter_chain as hc                     # noqa: E402
from make_ball_audit import load_impacts      # noqa: E402

SP = Path("/tmp/claude-0/-home-user-pickleball/"
          "3678254a-7688-518f-baf9-d64243c70ab4/scratchpad")
DATA = Path("/home/user/pickleball/data/vision")

RALLIES = {
    6:  (SP / "r0006.npz", SP / "r6_clip.mp4", 144.80),
    7:  (SP / "r0007.npz", SP / "r7_clip.mp4", 164.50),
    9:  (SP / "r0009.npz", SP / "r9_clip.mp4", 251.00),
    10: (SP / "r0010.npz", SP / "r10_clip.mp4", 292.70),
}


def anchors_for(rally, npz, clip, offset):
    z = np.load(npz)
    picked = hc.predict_contacts(str(npz), float(z["t"].min()),
                                 float(z["t"].max()))
    extra = hc.blur_gap_fill(str(npz), str(clip), offset, picked)
    ev = [(e[0], e[1], "pose") for e in picked]
    ev += [(e[0], e[1], "blur") for e in extra]
    return sorted(ev)


def main():
    for rally, (npz, clip, offset) in RALLIES.items():
        imps, dead = load_impacts(rally=rally)
        ev = anchors_for(rally, npz, clip, offset)
        in_rally = [e for e in ev if imps[0] - 0.2 <= e[0] <= dead]
        used = set()
        matched = []
        missed = []
        for t0 in imps:
            m = [(abs(t0 - e[0]), i) for i, e in enumerate(in_rally)
                 if i not in used and abs(t0 - e[0]) <= 0.15]
            if m:
                d, i = min(m)
                used.add(i)
                matched.append((t0, in_rally[i]))
            else:
                missed.append(t0)
        fakes = [e for i, e in enumerate(in_rally) if i not in used]
        mz = [e[1] for _, e in matched]
        fz = [e[1] for e in fakes]
        print(f"== rally {rally}: {len(imps)} taps | anchors in-rally "
              f"{len(in_rally)} | recall {len(matched)}/{len(imps)} | "
              f"fakes {len(fakes)}")
        if mz:
            print(f"   matched z: med {np.median(mz):.2f} "
                  f"[{np.percentile(mz,10):.2f},{np.percentile(mz,90):.2f}]")
        if fz:
            print(f"   fake    z: med {np.median(fz):.2f} "
                  f"[{np.percentile(fz,10):.2f},{np.percentile(fz,90):.2f}]")
        # top-K composition: walking down by z, what fraction is real?
        srt = sorted(in_rally, key=lambda e: -e[1])
        real_ts = {e[0] for _, e in matched}
        for K in (10, 20, 30):
            if K > len(srt):
                break
            top = srt[:K]
            nreal = sum(1 for e in top if e[0] in real_ts)
            print(f"   top-{K} by z: {nreal}/{K} real")
        if missed:
            print(f"   missed taps: {[f'{t:.2f}' for t in missed]}")
        # channel split
        for ch in ("pose", "blur"):
            n_ch = sum(1 for e in in_rally if e[2] == ch)
            r_ch = sum(1 for _, e in matched if e[2] == ch)
            print(f"   {ch}: {n_ch} anchors, {r_ch} matched")


if __name__ == "__main__":
    main()
