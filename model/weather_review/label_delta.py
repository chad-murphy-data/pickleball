"""B2a follow-up: is the arm-a -> arm-c change in the H4a interaction bigger
than event-resampling noise?  Paired cluster bootstrap: one event resample
drives BOTH label arms, so the difference of estimates is paired.

    python model/weather_review/label_delta.py
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_arms_rerun import build, load_labels, ols  # noqa: E402

NBOOT = 800


def d_of(rows):
    if len(rows) < 50:
        return None
    c = ols(rows, "y", ["skill", "w", "sw"])
    return None if c is None else c[3]


def main():
    geo, ov, arms = load_labels()
    _, game_rows, _ = build()
    clusters = defaultdict(list)
    for r in game_rows:
        clusters[r["ev"]].append(r)
    keys = list(clusters)

    def stats(rows):
        by = {a: {"outdoor": [], "indoor": []} for a in "acd"}
        for r in rows:
            for a in "acd":
                s = arms[a].get(r["ev"])
                if s in ("outdoor", "indoor"):
                    by[a][s].append(r)
        out = {}
        for a in "acd":
            do, di = d_of(by[a]["outdoor"]), d_of(by[a]["indoor"])
            out[a] = (do, di, None if do is None or di is None else do - di)
        return out

    pt = stats(game_rows)
    rng = random.Random(4242)
    draws = defaultdict(list)
    for _ in range(NBOOT):
        s = []
        for _ in keys:
            s.extend(clusters[rng.choice(keys)])
        st = stats(s)
        for lbl, v in (("d_out_a_minus_c", (st["a"][0], st["c"][0])),
                       ("d_out_a_minus_d", (st["a"][0], st["d"][0])),
                       ("diff_a_minus_c", (st["a"][2], st["c"][2]))):
            if v[0] is not None and v[1] is not None:
                draws[lbl].append(v[0] - v[1])

    print("point estimates (d = skill x wind interaction)")
    for a in "acd":
        print(f"  arm {a}: outdoor {pt[a][0]:+.4f}  indoor {pt[a][1]:+.4f}  "
              f"out-in {pt[a][2]:+.4f}")
    print("\npaired change from relabelling (95% cluster-bootstrap CI, "
          f"{NBOOT} event resamples)")
    for lbl in ("d_out_a_minus_c", "d_out_a_minus_d", "diff_a_minus_c"):
        v = sorted(draws[lbl])
        p0 = {"d_out_a_minus_c": pt["a"][0] - pt["c"][0],
              "d_out_a_minus_d": pt["a"][0] - pt["d"][0],
              "diff_a_minus_c": pt["a"][2] - pt["c"][2]}[lbl]
        print(f"  {lbl}: {p0:+.4f} [{v[int(0.025*len(v))]:+.4f}, "
              f"{v[int(0.975*len(v))]:+.4f}]  (n={len(v)})")


if __name__ == "__main__":
    main()
