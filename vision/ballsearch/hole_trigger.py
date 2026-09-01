"""Cost/benefit of the owner's rule: 'if you lose the ball for X
frames, the points you were holding were probably junk — try another
option'. Before building anything, measure what the trigger condemns.

For every hole (gap >= X missing frames) in the visited stream,
condemn the last K stream points before it. Grade condemned points
against the clicks: junk (err > 12 px) = correctly condemned; good
(<= 12 px) = a positive we would lose. Also classify each hole by
the click codes inside it (V = ball visible during the hole ->
detector failure; S/I/N = genuinely hidden/absent -> legit hole),
and a body-qualified variant (condemn only tail points within 16 px
of a pose extremity).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
from claim_lab import load                      # noqa: E402
from corridor_dp import body_points             # noqa: E402

for rally in (10, 9):
    c = load(rally)
    t0 = c["t0"]
    rows = []
    csv = Path(f"/home/user/pickleball/data/vision/ball_path_r{rally}"
               ".csv")
    for ln in csv.read_text().splitlines()[1:]:
        p = ln.split(",")
        f = int(round((float(p[1]) - t0) * 60))
        try:
            rows.append((f, float(p[2]), float(p[3]), p[4].strip()))
        except ValueError:
            rows.append((f, None, None, p[4].strip()))
    tf = {}
    pos = [(f, x, y) for f, x, y, v in rows
           if x is not None and v in ("V", "S")]
    for (f1, x1, y1), (f2, x2, y2) in zip(pos, pos[1:]):
        tf[f1] = (x1, y1)
        if 0 < f2 - f1 <= 3:
            for f in range(f1 + 1, f2):
                a = (f - f1) / (f2 - f1)
                tf[f] = (x1 + a * (x2 - x1), y1 + a * (y2 - y1))
    if pos:
        tf[pos[-1][0]] = pos[-1][1:]
    code_by_f = {f: v for f, x, y, v in rows}
    vis = sorted({(int(f), x, y) for f, x, y in c["visited"]})
    frames = [v[0] for v in vis]
    body = body_points(c, frames[0], frames[-1])

    def near_body(f, x, y):
        arr = body.get(f)
        if arr is None or not len(arr):
            return False
        return float(np.min(np.hypot(arr[:, 0] - x,
                                     arr[:, 1] - y))) <= 16.0

    n_junk_total = sum(
        1 for f, x, y in vis if f in tf
        and np.hypot(x - tf[f][0], y - tf[f][1]) > 12.0)
    print(f"rally {rally}: {len(vis)} stream pts, "
          f"{n_junk_total} junk total")
    for X in (2, 3, 4, 6):
        for K in (1, 2, 3):
            cond = set()
            holes = viz_holes = 0
            good_tail_viz = good_tail_hid = 0
            for i in range(len(vis) - 1):
                gap = vis[i + 1][0] - vis[i][0] - 1
                if gap < X:
                    continue
                holes += 1
                hole_codes = {code_by_f.get(f) for f in
                              range(vis[i][0] + 1, vis[i + 1][0])}
                viz = "V" in hole_codes
                viz_holes += viz
                tail_good = False
                for k in range(K):
                    j = i - k
                    if j < 0:
                        break
                    f, x, y = vis[j]
                    cond.add((f, x, y))
                    if f in tf and np.hypot(x - tf[f][0],
                                            y - tf[f][1]) <= 12.0:
                        tail_good = True
                if tail_good:
                    if viz:
                        good_tail_viz += 1
                    else:
                        good_tail_hid += 1
            nj = ng = nu = nbj = nbg = 0
            for f, x, y in cond:
                nb = near_body(f, x, y)
                if f not in tf:
                    nu += 1
                    continue
                if np.hypot(x - tf[f][0], y - tf[f][1]) > 12.0:
                    nj += 1
                    nbj += nb
                else:
                    ng += 1
                    nbg += nb
            cap = 100 * nj / max(1, n_junk_total)
            print(f"  X>={X} K={K}: holes {holes:3d} "
                  f"({viz_holes} with ball VISIBLE inside) | condemn "
                  f"{len(cond):3d} pts: junk {nj:3d} good {ng:3d} "
                  f"ungraded {nu:2d} | junk capture {cap:3.0f}% | "
                  f"body-only: junk {nbj:3d} good {nbg:3d} | "
                  f"good-tail holes: {good_tail_viz} vis / "
                  f"{good_tail_hid} hidden")
