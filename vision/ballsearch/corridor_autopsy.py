"""Autopsy of decode holes: WHY does a human-visible ball produce no
detector candidate within 12 px?

For every truth click where the decode has nothing within 12 px, measure
at the click location itself:
  - prod-corridor window geometry (was the search box even there?)
  - +/-2-frame min-diff signal near the click (what production sees)
  - +/-6-frame min-diff counterfactual (slow-mode differencing)
  - chroma (max over BGR) +/-2 counterfactual (color differencing)
  - the connected component nearest the click: area + centroid offset
    (merged-into-player diagnosis)
  - local truth speed in px/frame (from neighboring clicks)

Diagnostic only — truth is used to ask WHY, never fed to production.

Usage: python3 corridor_autopsy.py <rally> [--thr 14] [--rth 0.5]
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
from claim_lab import load, paddle_series               # noqa: E402
from corridor_lab import (load_truth, prod_contacts, corridors,  # noqa
                          window_at, decode_recall, CLIPS, SP,
                          AREA_MIN, AREA_MAX)

THR = 14
R = 12.0
PATCH = 120         # half-size of the analysis region around a click


def comp_near(motion, tx, ty, thr):
    """component whose pixels come nearest (tx,ty) after production's
    threshold+dilate; returns (mindist, area, centroid_dist) or None."""
    _, mask = cv2.threshold(motion, thr, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        ys, xs = np.where(lab == i)
        d = float(np.min(np.hypot(xs - tx, ys - ty)))
        if best is None or d < best[0]:
            cd = float(np.hypot(cent[i][0] - tx, cent[i][1] - ty))
            best = (d, int(stats[i, cv2.CC_STAT_AREA]), cd)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rally", type=int)
    ap.add_argument("--thr", type=int, default=THR)
    ap.add_argument("--rth", type=float, default=0.5)
    a = ap.parse_args()
    c = load(a.rally)
    series = paddle_series(c["npz"])
    truth = load_truth(a.rally)
    t0 = c["t0"]
    dec = decode_recall(c, truth)
    cors = corridors(c, series, prod_contacts(c, series, a.rth))
    # local truth speed (px per 60fps frame) from neighbouring clicks
    speeds = []
    for i, (t, x, y, v) in enumerate(truth):
        nb = [truth[j] for j in (i - 1, i + 1) if 0 <= j < len(truth)
              and abs(truth[j][0] - t) < 0.09]
        if nb:
            sp = np.mean([np.hypot(x - q[1], y - q[2])
                          / (abs(t - q[0]) * 60) for q in nb])
        else:
            sp = np.nan
        speeds.append(float(sp))

    sched = {}
    for i, (t, x, y, v) in enumerate(truth):
        if dec[i]:
            continue                      # autopsy the HOLES only
        f = int(round((t - t0) * 60))
        sched.setdefault(f, []).append(i)
    holes = sum(len(v) for v in sched.values())
    print(f"rally {a.rally}: {len(truth)} V/S clicks, "
          f"decode holes {holes}, thr={a.thr}")

    rows = {}
    gbuf, cbuf = deque(maxlen=13), deque(maxlen=13)
    cap = cv2.VideoCapture(str(SP / CLIPS[a.rally]))
    Wf = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hf = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fi = 0
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        cbuf.append(fr)
        gbuf.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        fi += 1
        mid = fi - 7
        if len(gbuf) < 13 or mid not in sched:
            continue
        g = {k: gbuf[6 + k] for k in (-6, -2, 0, 2, 6)}
        cc = {k: cbuf[6 + k].astype(np.int16) for k in (-2, 0, 2)}
        for i in sched[mid]:
            t, tx, ty, v = truth[i]
            x0, x1 = int(max(0, tx - PATCH)), int(min(Wf, tx + PATCH))
            y0, y1 = int(max(0, ty - PATCH)), int(min(Hf, ty + PATCH))
            px, py = tx - x0, ty - y0
            m2 = cv2.min(
                cv2.absdiff(g[0][y0:y1, x0:x1], g[-2][y0:y1, x0:x1]),
                cv2.absdiff(g[0][y0:y1, x0:x1], g[2][y0:y1, x0:x1]))
            m6 = cv2.min(
                cv2.absdiff(g[0][y0:y1, x0:x1], g[-6][y0:y1, x0:x1]),
                cv2.absdiff(g[0][y0:y1, x0:x1], g[6][y0:y1, x0:x1]))
            ch = np.minimum(
                np.abs(cc[0][y0:y1, x0:x1] - cc[-2][y0:y1, x0:x1])
                .max(axis=2),
                np.abs(cc[0][y0:y1, x0:x1] - cc[2][y0:y1, x0:x1])
                .max(axis=2)).astype(np.uint8)
            cor = next((co for co in cors if co[0] <= t <= co[1]), None)
            if cor is None:
                inwin, edge = False, np.nan
            else:
                cx, cy, wx, wy = window_at(cor, t)
                inwin = abs(tx - cx) <= wx and abs(ty - cy) <= wy
                edge = float(min(wx - abs(tx - cx), wy - abs(ty - cy)))
            # motion at the click pixel itself (5x5): the ball's OWN
            # signal, merged blob or not
            iy0, iy1 = int(max(0, py - 2)), int(py + 3)
            ix0, ix1 = int(max(0, px - 2)), int(px + 3)
            click2 = float(m2[iy0:iy1, ix0:ix1].max()) \
                if m2[iy0:iy1, ix0:ix1].size else 0.0
            # ball-scale local maxima of the motion image (the blob-
            # decomposition counterfactual): nearest peak >= thr
            mx = cv2.dilate(m2, np.ones((7, 7), np.uint8))
            pys, pxs = np.where((m2 >= mx) & (m2 >= a.thr))
            if len(pxs):
                nmsd = float(np.min(np.hypot(pxs - px, pys - py)))
            else:
                nmsd = np.inf
            rows[i] = dict(
                inwin=inwin, edge=edge, sp=speeds[i], vis=v,
                click2=click2, nmsd=nmsd, nmsn=len(pxs),
                near2=comp_near(m2, px, py, a.thr),
                near6=comp_near(m6, px, py, a.thr),
                nearC=comp_near(ch, px, py, a.thr))
    cap.release()

    def cand_ok(nr):
        """would production have yielded a candidate within R of truth
        from this motion image? (component touches R, legal area,
        centroid within R)"""
        return (nr is not None and nr[0] <= R
                and AREA_MIN <= nr[1] <= AREA_MAX and nr[2] <= R)

    OUT, NOSIG, MERGED, TINY, OK = [], [], [], [], []
    fix6, fixC, fixEither = 0, 0, 0
    for i, r in rows.items():
        if not r["inwin"]:
            OUT.append(i)
            continue
        n2 = r["near2"]
        if n2 is None or n2[0] > R:
            NOSIG.append(i)
            f6 = cand_ok(r["near6"])
            fC = cand_ok(r["nearC"])
            fix6 += f6
            fixC += fC
            fixEither += (f6 or fC)
            continue
        if n2[1] > AREA_MAX or n2[2] > R:
            MERGED.append(i)
            continue
        if n2[1] < AREA_MIN:
            TINY.append(i)
            continue
        OK.append(i)

    def spmed(idx):
        ss = [rows[i]["sp"] for i in idx
              if not np.isnan(rows[i]["sp"])]
        return f"{np.median(ss):.1f}" if ss else "-"

    n = len(rows)
    print(f"  analysed {n} holes (in prod corridors era)")
    print(f"  OUT-OF-WINDOW      {len(OUT):3d}  "
          f"(median speed {spmed(OUT)} px/fr)")
    print(f"  NO-SIGNAL @+/-2    {len(NOSIG):3d}  "
          f"(median speed {spmed(NOSIG)})")
    print(f"      -> fixed by +/-6 diff: {fix6}, by chroma: {fixC}, "
          f"either: {fixEither}")
    ma = [rows[i]["near2"][1] for i in MERGED]
    mc = [rows[i]["near2"][2] for i in MERGED]
    mo = sum(1 for i in MERGED if rows[i]["click2"] >= a.thr)
    mn = sum(1 for i in MERGED if rows[i]["nmsd"] <= R)
    print(f"  MERGED/DISPLACED   {len(MERGED):3d}  "
          f"(median speed {spmed(MERGED)}; med area "
          f"{np.median(ma) if ma else 0:.0f}, med centroid-off "
          f"{np.median(mc) if mc else 0:.0f}px)")
    print(f"      own signal AT click >= thr: {mo}/{len(MERGED)}; "
          f"NMS peak within {R:.0f}px: {mn}/{len(MERGED)}")
    print(f"  TINY (<{AREA_MIN}px)      {len(TINY):3d}")
    ko = sum(1 for i in OK if rows[i]["click2"] >= a.thr)
    print(f"  SIGNAL-OK (candidate existed; selection failed) "
          f"{len(OK):3d}  (own signal at click: {ko}/{len(OK)})")
    # counterfactual recoverable mass, all holes
    rec_now = sum(1 for i, r in rows.items() if cand_ok(r["near2"]))
    rec_all = sum(1 for i, r in rows.items()
                  if cand_ok(r["near2"]) or cand_ok(r["near6"])
                  or cand_ok(r["nearC"]))
    rec_nms = sum(1 for i, r in rows.items() if r["nmsd"] <= R)
    junk = [r["nmsn"] for r in rows.values()]
    print(f"  candidate-within-{R:.0f}px at click:  now {rec_now}/{n}"
          f"  with slow-mode+chroma {rec_all}/{n}"
          f"  with NMS peaks {rec_nms}/{n}"
          f"  (NMS peaks/patch med {np.median(junk):.0f})")
    slow = [i for i, r in rows.items()
            if not np.isnan(r["sp"]) and r["sp"] < 1.5]
    print(f"  slow holes (<1.5 px/fr): {len(slow)}/{n}"
          f"  of which NO-SIGNAL: "
          f"{sum(1 for i in slow if i in set(NOSIG))}")


if __name__ == "__main__":
    main()
