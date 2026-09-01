"""Sweep track_signals variants (base / torso-relative A / +asym B)
over the 4 local rallies. Blur peaks computed ONCE per rally (they
do not depend on the pose variant until the final min-sep filter)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import hitter_chain as hc                 # noqa: E402
import ts_variants as tv                  # noqa: E402
from make_ball_audit import load_impacts  # noqa: E402
from anchor_diag import RALLIES           # noqa: E402


def match_stats(rally, ev):
    imps, dead = load_impacts(rally=rally)
    in_rally = [e for e in ev if imps[0] - 0.2 <= e[0] <= dead]
    used = set()
    matched, missed = [], []
    for t0 in imps:
        m = [(abs(t0 - e[0]), i) for i, e in enumerate(in_rally)
             if i not in used and abs(t0 - e[0]) <= 0.15]
        if m:
            d, i = min(m)
            used.add(i)
            matched.append(in_rally[i])
        else:
            missed.append(t0)
    fakes = [e for i, e in enumerate(in_rally) if i not in used]
    mz = [e[1] for e in matched]
    fz = [e[1] for e in fakes]
    real_ts = {e[0] for e in matched}
    srt = sorted(in_rally, key=lambda e: -e[1])
    topk = {}
    for K in (10, 20):
        if K <= len(srt):
            topk[K] = sum(1 for e in srt[:K] if e[0] in real_ts)
    # separation: AUC-style P(matched z > fake z)
    auc = float("nan")
    if mz and fz:
        wins = sum((a > b) + 0.5 * (a == b) for a in mz for b in fz)
        auc = wins / (len(mz) * len(fz))
    return dict(n_in=len(in_rally), rec=len(matched), n_imp=len(imps),
                fakes=len(fakes),
                mz=np.median(mz) if mz else float("nan"),
                fz=np.median(fz) if fz else float("nan"),
                auc=auc, topk=topk, missed=missed)


def main():
    blur_cache = {}
    for rally, (npz, clip, offset) in RALLIES.items():
        blur_cache[rally] = hc.blur_gap_fill(str(npz), str(clip),
                                             offset, [])
    for mode in ("base", "A", "B"):
        tv.patch(mode)
        print(f"===== mode {mode}")
        aucs, recs = [], []
        for rally, (npz, clip, offset) in RALLIES.items():
            z = np.load(npz)
            picked = hc.predict_contacts(
                str(npz), float(z["t"].min()), float(z["t"].max()))
            pose_t = [e[0] for e in picked]
            extra = [p for p in blur_cache[rally]
                     if all(abs(p[0] - pt) >= hc.MIN_SEP_S
                            for pt in pose_t)]
            ev = sorted([(e[0], e[1], "pose") for e in picked]
                        + [(p[0], p[1], "blur") for p in extra])
            s = match_stats(rally, ev)
            aucs.append(s["auc"])
            recs.append(s["rec"] / s["n_imp"])
            print(f"  r{rally}: recall {s['rec']}/{s['n_imp']}  "
                  f"in-rally {s['n_in']} (fakes {s['fakes']})  "
                  f"z med real {s['mz']:.2f} vs fake {s['fz']:.2f}  "
                  f"AUC {s['auc']:.3f}  top-z "
                  + " ".join(f"{k}:{v}/{k}" for k, v in s["topk"].items()))
            if s["missed"]:
                print(f"        missed: "
                      f"{[f'{t:.2f}' for t in s['missed']]}")
        print(f"  MEAN: recall {np.mean(recs):.3f}  AUC "
              f"{np.nanmean(aucs):.3f}")
    tv.patch("base")


if __name__ == "__main__":
    main()
