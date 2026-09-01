"""Plain-language miss map for r10's bounce ledger, on the PRODUCTION
segment path (crossing_demotion, same as score_c3): every credited
tracked bounce and every human bounce, when/where (court feet), which
human bounces go uncredited, and what the owner's clicks show there.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import ball_replicate as br                     # noqa: E402
import court3d as c3                            # noqa: E402
from claim_lab import load, paddle_series       # noqa: E402
from corridor_fit import app_bounds             # noqa: E402

c = load(10)
series = paddle_series(c["npz"])
P = c["P"]
bounds, evs = app_bounds(c, series, 0.5)
bevs = [e for e in c["turns"] if e not in set(bounds)]
obs = [(c["t0"] + f / 60.0, x, y, 1.0) for f, x, y in c["visited"]]
anchors = br.dedupe_anchors(c["anchors"], c["zs"],
                            br.track_sides(c["floors"]), c["turns"])
t_segs, t_cons, t_bounds, t_evs = br.crossing_demotion(
    P, obs, sorted(bounds) + [c["end"]], bevs, c["floors"], anchors)


def bounce_of(seg):
    if seg and seg["ok"] and seg["kind"] == "bounce" \
            and len(seg["arcs"]) >= 2:
        a2 = seg["arcs"][1]
        return a2[0], c3.arc_pos(a2[2], np.array([0.0]))[0]
    return None


print("tracked CREDITED bounces (production path):")
t_credit = []
for k, seg in enumerate(t_segs):
    b = bounce_of(seg)
    if b:
        t_credit.append(b)
        print(f"  @{b[0]:7.2f}  court ({b[1][0]:5.1f},{b[1][1]:5.1f},"
              f"{b[1][2]:5.1f})ft  seg {t_bounds[k]:.2f}-"
              f"{t_bounds[k+1]:.2f}")

h_obs, h_bounds, h_evs = c["hum"]
h_segs = c["h_segs"]
rows = []
for ln in Path("/home/user/pickleball/data/vision/ball_path_r10.csv"
               ).read_text().splitlines()[1:]:
    p = ln.split(",")
    try:
        rows.append((float(p[1]), float(p[2]), float(p[3]),
                     p[4].strip()))
    except ValueError:
        rows.append((float(p[1]), None, None, p[4].strip()))

print("\nhuman bounces vs tracked credit:")
for j, s in enumerate(h_segs):
    b = bounce_of(s)
    if not b:
        continue
    tbn, p = b
    near = [tc for tc, _ in t_credit if abs(tc - tbn) <= 0.45]
    status = "credited" if near else "MISS"
    print(f"  bounce @{tbn:7.2f}  court ({p[0]:5.1f},{p[1]:5.1f},"
          f"{p[2]:5.1f})ft  seg {h_bounds[j]:.2f}-{h_bounds[j+1]:.2f}"
          f"  -> {status}")
    if not near:
        w = [r for r in rows if abs(r[0] - tbn) <= 0.35]
        codes = "".join(r[3] for r in w)
        wc = [r for r in w if r[1] is not None]
        if wc:
            k = int(np.argmin([abs(r[0] - tbn) for r in wc]))
            r = wc[k]
            lowest = max(wc, key=lambda r: r[2])
            print(f"      clicks +/-0.35s: {len(w)} [{codes}]; nearest"
                  f" dt={r[0]-tbn:+.2f}s ({r[1]:.0f},{r[2]:.0f})px "
                  f"code {r[3]}; low point @{lowest[0]:.2f} "
                  f"({lowest[1]:.0f},{lowest[2]:.0f})px")
        else:
            print(f"      clicks +/-0.35s: {len(w)} [{codes}] "
                  f"(no positioned clicks)")
