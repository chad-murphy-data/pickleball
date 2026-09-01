"""softdp — tune corridor_dp's W_P_SOFT on the TRAIN rallies only.

Pre-registered protocol (swing_explore_notes 2026-09-01): the learned
emission p enters the DP as a SOFT unary cost W_P_SOFT * (1 - p),
replacing the hard p-filter that killed whole chains on faint fast
drives (r10 297.87: dp 20 -> 0). The weight is chosen HERE, on r6+r7
graded against their own clicks with CROSS-FOLD p (emission.py
cache-cross: r6 scored by an r7-only model and vice versa, so the
tuning never sees in-sample p), then frozen for a one-shot r9/r10
grading via `spaghetti.py <r> --lrn --soft W`. r9/r10 clicks play no
part in the choice.

SELECTION RULE (stated before looking at any number): pick the
smallest W that maximizes total r@12 summed over both rallies and
both arms (prod + oracle), subject to pooled prec@12 >= the W=0
baseline's pooled prec@12. If no W > 0 beats W=0 on total r@12, the
soft term is dead — record that and do NOT run r9/r10.

Reference arms per rally: W=0 (today's dp-cc+body) and the HARD
filter at the fold's own 97%-recall threshold (kp97 from the cross
cache) — the r6/r7 analogue of the graded dp-ccL arm.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import corridor_dp as cdp                              # noqa: E402
import spaghetti as spag                               # noqa: E402
from claim_lab import load, paddle_series              # noqa: E402
from corridor_lab import (load_truth, prod_contacts, corridors,  # noqa
                          decode_recall, R_MAIN)

SP = Path(__file__).parent
WS = (0.0, 3.0, 6.0, 12.0, 25.0, 50.0)


def grade(track, truth, t0, dec):
    """same semantics as corridor_dp.score, returned not printed."""
    h12 = have = added = 0
    for (t, tx, ty, vis), d in zip(truth, dec):
        f = int(round((t - t0) * 60))
        p = track.get(f) or track.get(f - 1) or track.get(f + 1)
        if p is None:
            continue
        have += 1
        dd = float(np.hypot(p[0] - tx, p[1] - ty))
        h12 += dd <= R_MAIN
        added += dd <= R_MAIN and not d
    return h12, have, added


def main():
    panels = []          # (rally, arm, cors, cc, truth, t0, dec, body)
    for rally in (6, 7):
        c = load(rally)
        series = paddle_series(c["npz"])
        truth = load_truth(rally)
        t0 = c["t0"]
        f_lo = int((c["serve"] - 0.4 - t0) * 60)
        f_hi = int((c["end"] + 0.2 - t0) * 60)
        dec = decode_recall(c, truth)
        body = cdp.body_points(c, f_lo, f_hi)
        cc = spag.cands_cached(rally, f_lo, f_hi, 14, "cc",
                               lrn=True, pxs="_x")
        kp97 = float(np.load(SP / f"p_r{rally}_cc_14_x.npz")["kp97"])
        print(f"rally {rally}: {len(truth)} clicks, decode@12 "
              f"{sum(dec)}/{len(dec)}, fold kp97 {kp97:.4f}")
        for arm, times in (("prod", prod_contacts(c, series, 0.5)),
                           ("oracle", list(c["imps"]))):
            cors = corridors(c, series, times)
            panels.append((rally, arm, cors, cc, truth, t0, dec,
                           body, kp97))

    rows = {}
    for W in WS:
        cdp.W_P_SOFT = W
        tot = dict(h12=0, have=0, added=0)
        per = []
        for (rally, arm, cors, cc, truth, t0, dec, body, _kp) in panels:
            tr = cdp.build_track(cc, cors, t0, body=body)
            h12, have, added = grade(tr, truth, t0, dec)
            tot["h12"] += h12
            tot["have"] += have
            tot["added"] += added
            per.append(f"r{rally}-{arm} {h12}/{have}")
        rows[W] = tot
        print(f"W={W:5g}  total r@12 {tot['h12']:4d}  prec@12 "
              f"{tot['h12'] / max(1, tot['have']):.3f}  ADDED "
              f"{tot['added']:3d}  | " + "  ".join(per))
    cdp.W_P_SOFT = 0.0

    # hard-filter reference at each fold's kp97
    tot = dict(h12=0, have=0, added=0)
    per = []
    for (rally, arm, cors, cc, truth, t0, dec, body, kp97) in panels:
        cch = {f: [c_[:4] for c_ in cs if c_[4] >= kp97]
               for f, cs in cc.items()}
        tr = cdp.build_track(cch, cors, t0, body=body)
        h12, have, added = grade(tr, truth, t0, dec)
        tot["h12"] += h12
        tot["have"] += have
        tot["added"] += added
        per.append(f"r{rally}-{arm} {h12}/{have}")
    print(f"HARD@kp97  total r@12 {tot['h12']:4d}  prec@12 "
          f"{tot['h12'] / max(1, tot['have']):.3f}  ADDED "
          f"{tot['added']:3d}  | " + "  ".join(per))

    base = rows[0.0]
    base_prec = base["h12"] / max(1, base["have"])
    best_w, best_h = 0.0, base["h12"]
    for W in WS[1:]:
        t = rows[W]
        if (t["h12"] > best_h and
                t["h12"] / max(1, t["have"]) >= base_prec):
            best_w, best_h = W, t["h12"]
    if best_w == 0.0:
        print("VERDICT: no W beats W=0 under the rule — soft term "
              "dead, do not run r9/r10")
    else:
        print(f"VERDICT: W_P_SOFT = {best_w:g} (total r@12 "
              f"{best_h} vs {base['h12']} at W=0, prec >= "
              f"{base_prec:.3f}) — freeze and one-shot r9/r10")


if __name__ == "__main__":
    main()
