"""Owner's rule set, graded before building: classify every stream
hole LABEL-FREE as
  edge       — endpoint within 30 px of the frame border (lobs)
  occluded   — the A->B chord spends >=50% of the hole inside a
               player bbox (pose npz, +8 px pad, height-sane)
  unexplained— neither
then grade each class against the clicks: tail quality (the points
we would trust under 'stick with it'), re-acquisition quality (the
point after the hole), the tail-near-extremity trap (junk tail ON
the arm makes the chord pass through the player -> fake 'occluded'),
and the owner's own V/S/N codes inside the hole as classifier truth.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
from claim_lab import load                      # noqa: E402
from corridor_dp import body_points             # noqa: E402

EDGE = 30.0
PAD = 8.0
W, H = 1280.0, 720.0

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
    z = np.load(c["npz"])
    fr = np.round((np.asarray(z["t"]) - t0) * 60).astype(int)
    boxes = {}
    for i in range(len(fr)):
        b = z["box"][i]
        if b[3] - b[1] > 500:          # close-up shot, not play
            continue
        boxes.setdefault(int(fr[i]), []).append(
            (b[0] - PAD, b[1] - PAD, b[2] + PAD, b[3] + PAD))

    def in_box(f, x, y):
        for df in (0, -1, 1, -2, 2):
            bs = boxes.get(f + df)
            if bs:
                return any(bx1 <= x <= bx2 and by1 <= y <= by2
                           for bx1, by1, bx2, by2 in bs)
        return False

    vis = sorted({(int(f), x, y) for f, x, y in c["visited"]})
    body = body_points(c, vis[0][0], vis[-1][0])

    def near_body(f, x, y):
        arr = body.get(f)
        if arr is None or not len(arr):
            return False
        return float(np.min(np.hypot(arr[:, 0] - x,
                                     arr[:, 1] - y))) <= 16.0

    def grade(f, x, y):
        if f not in tf:
            return "?"
        return ("junk" if np.hypot(x - tf[f][0], y - tf[f][1]) > 12.0
                else "good")

    cls = {}
    for i in range(len(vis) - 1):
        fA, xA, yA = vis[i]
        fB, xB, yB = vis[i + 1]
        gap = fB - fA - 1
        if gap < 2:
            continue
        if (min(xA, xB) <= EDGE or max(xA, xB) >= W - EDGE
                or min(yA, yB) <= EDGE or max(yA, yB) >= H - EDGE):
            k = "edge"
        else:
            inb = tot = 0
            for f in range(fA + 1, fB):
                a = (f - fA) / (fB - fA)
                tot += 1
                inb += in_box(f, xA + a * (xB - xA), yA + a * (yB - yA))
            k = "occluded" if tot and inb / tot >= 0.5 else "unexplained"
        codes = [code_by_f[f] for f in range(fA + 1, fB)
                 if f in code_by_f]
        cls.setdefault(k, []).append(dict(
            tail=grade(fA, xA, yA), reacq=grade(fB, xB, yB),
            tail_body=near_body(fA, xA, yA),
            nV=sum(cd == "V" for cd in codes),
            nH=sum(cd in ("S", "N", "I") for cd in codes)))
    print(f"rally {rally}:")
    for k in ("occluded", "edge", "unexplained"):
        hs = cls.get(k, [])
        if not hs:
            continue
        tj = sum(h["tail"] == "junk" for h in hs)
        tg = sum(h["tail"] == "good" for h in hs)
        tu = sum(h["tail"] == "?" for h in hs)
        rj = sum(h["reacq"] == "junk" for h in hs)
        rg = sum(h["reacq"] == "good" for h in hs)
        nV = sum(h["nV"] for h in hs)
        nH = sum(h["nH"] for h in hs)
        bt = [h for h in hs if h["tail_body"]]
        btj = sum(h["tail"] == "junk" for h in bt)
        btg = sum(h["tail"] == "good" for h in bt)
        print(f"  {k:11s} {len(hs):3d} holes | tail good {tg:3d} junk "
              f"{tj:3d} ?{tu:2d} | reacq good {rg:3d} junk {rj:3d} | "
              f"clicks in hole V {nV:3d} hidden {nH:3d} | "
              f"tail-near-body {len(bt):3d} (good {btg} junk {btj})")
