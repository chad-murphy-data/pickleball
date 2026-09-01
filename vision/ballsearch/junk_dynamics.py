"""What does the decode stream do AFTER it lands on a junk point?

Grade every visited point of r9/r10 against the owner's clicks
(truth interpolated click-to-click, 30->60 fps), classify junk
(err > 12 px), then measure: persistence at +1..+5 visited frames,
junk run lengths, how the junk MOVES while held (static object vs
arm), how runs end (recover vs stream hole), and how much junk sits
within R_BODY=16 px of a pose extremity.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
from claim_lab import load                      # noqa: E402
from corridor_lab import load_truth             # noqa: E402
from corridor_dp import body_points             # noqa: E402

for rally in (10, 9):
    c = load(rally)
    t0 = c["t0"]
    truth = load_truth(rally)
    tf = {}
    pts = [(int(round((t - t0) * 60)), x, y) for t, x, y, v in truth]
    for (f1, x1, y1), (f2, x2, y2) in zip(pts, pts[1:]):
        tf[f1] = (x1, y1)
        if 0 < f2 - f1 <= 3:
            for f in range(f1 + 1, f2):
                a = (f - f1) / (f2 - f1)
                tf[f] = (x1 + a * (x2 - x1), y1 + a * (y2 - y1))
    if pts:
        tf[pts[-1][0]] = pts[-1][1:]
    vis = sorted((int(f), x, y) for f, x, y in c["visited"])
    graded = []                       # (f, x, y, err) truth-covered only
    for f, x, y in vis:
        if f in tf:
            graded.append((f, x, y,
                           float(np.hypot(x - tf[f][0], y - tf[f][1]))))
    errs = np.array([g[3] for g in graded])
    junk = errs > 12.0
    print(f"rally {rally}: visited {len(vis)}, truth-covered "
          f"{len(graded)}, junk(>12px) {junk.sum()} "
          f"({100 * junk.mean():.0f}%)  [>30px {(errs > 30).sum()}]")
    # persistence: at the next 1..5 *frames*, is the stream (a) present
    # and (b) still junk; and how far did the held point move?
    by_f = {g[0]: g for g in graded}
    for h in (1, 2, 3, 5):
        n = pres = still = 0
        for g, isj in zip(graded, junk):
            if not isj:
                continue
            n += 1
            nxt = by_f.get(g[0] + h)
            if nxt is not None:
                pres += 1
                still += nxt[3] > 12.0
        print(f"  junk +{h} frames: stream present {pres}/{n} "
              f"({100 * pres / max(1, n):.0f}%), still junk "
              f"{still}/{max(1, pres)} ({100 * still / max(1, pres):.0f}"
              f"% of present)")
    # junk runs over consecutive graded entries (frame gap <= 2)
    runs, cur = [], None
    for g, isj in zip(graded, junk):
        if isj:
            if cur and g[0] - cur[-1][0] <= 2:
                cur.append(g)
            else:
                if cur:
                    runs.append(cur)
                cur = [g]
        else:
            if cur and g[0] - cur[-1][0] <= 2:
                runs.append(cur)
                cur = None
    if cur:
        runs.append(cur)
    ln = [len(r) for r in runs]
    step = [float(np.hypot(b[1] - a[1], b[2] - a[2]))
            for r in runs for a, b in zip(r, r[1:])]
    ends_rec = ends_hole = 0
    for r in runs:
        nxt = [g for g in graded if r[-1][0] < g[0] <= r[-1][0] + 2]
        if nxt and nxt[0][3] <= 12.0:
            ends_rec += 1
        elif not nxt:
            ends_hole += 1
    print(f"  junk runs: {len(runs)}  len med {np.median(ln):.0f} "
          f"p90 {np.percentile(ln, 90):.0f} max {max(ln)}  | held-junk "
          f"step/frame med {np.median(step):.1f}px  | run ends: "
          f"recover {ends_rec}, stream-hole {ends_hole}, "
          f"other {len(runs) - ends_rec - ends_hole}")
    fs = [g[0] for g in graded]
    body = body_points(c, min(fs), max(fs))
    nb = nb_j = 0
    for g, isj in zip(graded, junk):
        arr = body.get(g[0])
        if arr is None or not len(arr):
            continue
        d = float(np.min(np.hypot(arr[:, 0] - g[1], arr[:, 1] - g[2])))
        if d <= 16.0:
            nb += 1
            nb_j += isj
    print(f"  within 16px of an extremity: {nb} visited pts, "
          f"{nb_j} of them junk ({100 * nb_j / max(1, nb):.0f}%); "
          f"junk pts near-extremity: {nb_j}/{junk.sum()} "
          f"({100 * nb_j / max(1, junk.sum()):.0f}%)")
