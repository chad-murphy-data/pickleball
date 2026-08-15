"""One decode pass: ball and players, in court feet, with the camera checked.

This replaces the exploratory extractor.  The difference that matters is that
the court model is present DURING detection rather than bolted on afterwards,
which lets three things happen that could not happen before:

  * OFF-COURT CANDIDATES DIE IMMEDIATELY.  Roughly 44% of raw yellowness
    candidates on the Chicago match back-project outside the court; they are
    signage, crowd and line noise.  Discarding them is free precision.
  * PLAYER BLOBS GET A SIZE PRIOR.  A homography says how tall a 6 ft person
    should appear at any court position.  Blobs far from that are rejected,
    and blobs about twice too WIDE are split — which is the actual failure
    mode near the camera, where two players standing together merge into one
    2,000-px blob and silently become one player.
  * THE CAMERA IS VERIFIED PER FRAME.  Broadcast cuts to replays and crowd
    shots.  Court-colour coverage inside the fitted hull collapses on a cut,
    so every frame carries a view flag and downstream code can refuse to
    measure anything during a replay.

SAMPLING
    Ball is detected on EVERY frame — speed-ups live at 0.15-0.25 s and at
    60 fps that is 9-15 frames, so there is nothing to spare.  Players move
    slowly by comparison and are detected every `--player-every` frames
    (default 3, i.e. 20 Hz), which is where most of the speed comes from.

OUTPUT (prefix_*.csv)
    _ball    frame, t_s, x_img, y_img, area, strength, x_ft, y_ft
    _players frame, t_s, x_img, y_img, x_ft, y_ft, h_px, split
    _view    frame, t_s, court_frac, is_main
Coordinates: x_ft in [0,20] left-to-right as the camera sees it, y_ft in
[0,44] from the FAR baseline, net at 22.

    python vision/track_match.py --video v.mp4 --court court.json --out pfx
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import court as C                                            # noqa: E402

PLAYER_FT = 6.0            # nominal standing height, for the size prior
PLAYER_W_FT = 2.0          # nominal shoulder width, for the split rule


def decode(path, w, h):
    cmd = [C.FF, "-v", "error", "-i", str(path), "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 8)
    n = w * h * 3
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    finally:
        p.kill()


def yellowness(f):
    a = f.astype(np.int16)
    return np.minimum(a[..., 0], a[..., 1]) - a[..., 2]


def px_per_ft_upright(H, x_ft, y_ft):
    """Pixels per foot for an UPRIGHT object standing at a court point.

    Height is out of the homography's plane, so it has to be inferred from
    a ground length — but which one matters.  A foot of DEPTH shrinks like
    1/d^2 (the ground recedes as well as shrinks), while a foot ACROSS the
    court shrinks like 1/d, the same law an upright object follows.  Using
    the depth scale therefore under-predicts far players by roughly the
    depth ratio, which rejected them as impossibly tall: 95% of surviving
    detections were near-side.  The lateral scale reproduces the measured
    near:far player-height ratio (2.04 predicted, 2.0 observed).
    """
    p0 = C.apply_H(H, np.stack([x_ft, y_ft], 1))
    p1 = C.apply_H(H, np.stack([x_ft + 1.0, y_ft], 1))     # 1 ft across
    return np.linalg.norm(p1 - p0, axis=1)


class Tracker:
    def __init__(self, court_json, scale=1.0):
        d = json.load(open(court_json))
        S = np.diag([scale, scale, 1.0])
        self.H = S @ np.array(d["H_court_to_img"])
        self.Hi = np.linalg.inv(self.H)
        self.w = int(round(d["w"] * scale))
        self.h = int(round(d["h"] * scale))
        self.fps = d["fps"]
        # court hull mask + a generous play region for the ball
        gy, gx = np.mgrid[0:self.h, 0:self.w]
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        p = np.hstack([pts.astype(float), np.ones((len(pts), 1))]) @ self.Hi.T
        # Above the horizon the homogeneous coordinate changes sign and the
        # back-projection wraps around, mapping crowd rows onto plausible
        # court values.  Keep only pixels on the camera's side of it.
        wsign = np.sign(p[:, 2])
        ref = np.sign((np.array([self.w / 2, self.h * 0.7, 1.0]) @ self.Hi.T)[2])
        self.front = (wsign == ref).reshape(self.h, self.w)
        cc = p[:, :2] / p[:, 2:3]
        self.court_x = cc[:, 0].reshape(self.h, self.w)
        self.court_y = cc[:, 1].reshape(self.h, self.w)
        self.in_court = ((self.court_x > 0) & (self.court_x < C.W_FT)
                         & (self.court_y > 0) & (self.court_y < C.L_FT)
                         & self.front)
        # Players legitimately stand BEHIND both baselines to serve and
        # return, so the region searched for them is the court plus a
        # margin in court feet — not a pixel dilation of the court hull,
        # which cut deep players in half and lost them to the size gate.
        self.region = ((self.court_x > -8) & (self.court_x < C.W_FT + 8)
                       & (self.court_y > -13) & (self.court_y < C.L_FT + 14)
                       & self.front)
        # the ball flies above the plane, so its ground back-projection runs
        # LONG (away from camera); the play region is deliberately generous
        self.play = ((self.court_x > -12) & (self.court_x < C.W_FT + 12)
                     & (self.court_y > -25) & (self.court_y < C.L_FT + 18)
                     & self.front)
        ys, xs = np.nonzero(self.play)
        self.by0, self.by1 = ys.min(), ys.max() + 1
        self.bx0, self.bx1 = xs.min(), xs.max() + 1

    def to_court(self, pts):
        return C.apply_H(self.Hi, pts)

    # ---- ball --------------------------------------------------------
    def ball(self, frame, ybg, thresh=45, amin=1, amax=60):
        sl = (slice(self.by0, self.by1), slice(self.bx0, self.bx1))
        dy = yellowness(frame[sl]).astype(np.int16) - ybg[sl]
        m = (dy > thresh) & self.play[sl]
        if not m.any():
            return []
        lab, n = ndimage.label(m)
        if not n:
            return []
        areas = ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1))
        keep = [i + 1 for i, a in enumerate(areas) if amin <= a <= amax]
        if not keep:
            return []
        cy, cx = zip(*ndimage.center_of_mass(m, lab, keep))
        out = []
        for j, k in enumerate(keep):
            x = cx[j] + self.bx0
            y = cy[j] + self.by0
            out.append((float(x), float(y), int(areas[k - 1]),
                        float(dy[lab == k].mean())))
        return out

    # ---- players -----------------------------------------------------
    def players(self, frame, bg, half=2, thresh=34):
        """Foreground blobs, filtered and split against the size prior."""
        f = frame[::half, ::half].mean(axis=2)
        b = bg[::half, ::half].mean(axis=2)
        m = (np.abs(f - b) > thresh) & self.region[::half, ::half]
        m = ndimage.binary_closing(m, np.ones((3, 3)))
        lab, n = ndimage.label(m)
        if not n:
            return []
        objs = ndimage.find_objects(lab)
        hh, hw = m.shape
        out = []
        for i, sl in enumerate(objs, 1):
            if sl is None:
                continue
            ys, xs = sl
            bh = (ys.stop - ys.start) * half
            bw = (xs.stop - xs.start) * half
            area = int((lab[sl] == i).sum()) * half * half
            # Players standing BEHIND either baseline are clipped by the
            # frame: the far ones by the top edge, the near ones by the
            # bottom.  That is how this broadcast is framed and it is not
            # recoverable by better segmentation, but it is survivable —
            # a top-clipped player still shows their FEET, so their court
            # position is intact and only the height test needs waiving.
            clip_top = ys.start == 0
            clip_bot = ys.stop >= hh
            if bh < 24 or area < 350:
                if not (clip_top and area >= 200):
                    continue
            fx = (xs.start + xs.stop) / 2 * half
            fy = (ys.stop - 1) * half                    # feet: bottom edge
            cx, cy = self.to_court(np.array([[fx, fy]]))[0]
            if not (-6 < cx < C.W_FT + 6 and -12 < cy < C.L_FT + 12):
                continue
            ppf = float(px_per_ft_upright(
                self.H, np.array([np.clip(cx, 0, C.W_FT)]),
                np.array([np.clip(cy, 0, C.L_FT)]))[0])
            exp_h, exp_w = ppf * PLAYER_FT, ppf * PLAYER_W_FT
            if clip_bot:
                # feet are off-frame: put them one player-height below the
                # visible top, then re-read the court position
                fy = ys.start * half + exp_h
                cx, cy = self.to_court(np.array([[fx, fy]]))[0]
                if not (-6 < cx < C.W_FT + 6 and -12 < cy < C.L_FT + 14):
                    continue
            elif not (clip_top or 0.40 * exp_h < bh < 1.9 * exp_h):
                continue
            # two players standing together merge into one wide blob; the
            # size prior is what makes that detectable at all
            nsplit = 1
            if bw > 1.6 * exp_w:
                nsplit = min(int(round(bw / exp_w)), 3)
            for k in range(nsplit):
                sx = (xs.start * half
                      + (k + 0.5) * bw / nsplit)
                sc = self.to_court(np.array([[sx, fy]]))[0]
                out.append((float(sx), float(fy), float(sc[0]), float(sc[1]),
                            int(bh), int(nsplit > 1)))
        return out

    # ---- view --------------------------------------------------------
    def court_frac(self, frame, seed, tol=62.0):
        s = frame[::4, ::4].astype(np.float32)
        m = self.in_court[::4, ::4]
        if not m.any():
            return 0.0
        d = np.linalg.norm(s - seed, axis=2)
        return float(((d < tol) & m).sum() / m.sum())


def run(video, court_json, out, player_every=3, bg_every=45, bg_keep=13,
        limit=None, progress=60.0):
    w0, h0, fps, dur = C.probe(video)
    d = json.load(open(court_json))
    scale = w0 / d["w"]
    tr = Tracker(court_json, scale=scale)
    if (tr.w, tr.h) != (w0, h0):
        raise SystemExit(f"court fit is {d['w']}x{d['h']}, video is {w0}x{h0}"
                         f" — non-uniform scale {scale}")
    seed = None
    bgbuf, bg, ybg = [], None, None
    fb = open(f"{out}_ball.csv", "w", newline="")
    fp = open(f"{out}_players.csv", "w", newline="")
    fv = open(f"{out}_view.csv", "w", newline="")
    wb, wp, wv = csv.writer(fb), csv.writer(fp), csv.writer(fv)
    wb.writerow(["frame", "t_s", "x_img", "y_img", "area", "strength",
                 "x_ft", "y_ft"])
    wp.writerow(["frame", "t_s", "x_img", "y_img", "x_ft", "y_ft", "h_px",
                 "split"])
    wv.writerow(["frame", "t_s", "court_frac", "is_main"])
    n_ball = n_pl = 0
    t0 = time.time()
    last = 0.0
    for i, fr in enumerate(decode(video, w0, h0)):
        if limit and i >= limit:
            break
        t = i / fps
        if i % bg_every == 0:
            bgbuf.append(fr.copy())
            if len(bgbuf) > bg_keep:
                bgbuf.pop(0)
            bg = np.median(np.stack(bgbuf), axis=0)
            ybg = yellowness(bg).astype(np.int16)
            if seed is None:
                m, seed = C.surface_mask(bg)
        if bg is None:
            continue
        cf = tr.court_frac(fr, seed)
        is_main = cf > 0.45
        wv.writerow([i, f"{t:.3f}", f"{cf:.3f}", int(is_main)])
        if not is_main:
            continue
        for x, y, a, s in tr.ball(fr, ybg):
            cx, cy = tr.to_court(np.array([[x, y]]))[0]
            wb.writerow([i, f"{t:.3f}", f"{x:.1f}", f"{y:.1f}", a,
                         f"{s:.1f}", f"{cx:.2f}", f"{cy:.2f}"])
            n_ball += 1
        if i % player_every == 0:
            for x, y, cx, cy, bh, sp in tr.players(fr, bg):
                wp.writerow([i, f"{t:.3f}", f"{x:.0f}", f"{y:.0f}",
                             f"{cx:.2f}", f"{cy:.2f}", bh, sp])
                n_pl += 1
        if progress and t - last >= progress:
            last = t
            el = time.time() - t0
            print(f"  {t/60:6.1f} min video   {el/60:5.1f} min elapsed   "
                  f"{t/max(el,1e-9):.1f}x realtime   ball {n_ball}  "
                  f"players {n_pl}", flush=True)
    for f in (fb, fp, fv):
        f.close()
    el = time.time() - t0
    print(f"done: {i+1} frames, {n_ball} ball candidates, {n_pl} player "
          f"detections, {el/60:.1f} min ({(i+1)/fps/max(el,1e-9):.1f}x realtime)")
    return {"frames": i + 1, "ball": n_ball, "players": n_pl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--court", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--player-every", type=int, default=3)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    run(a.video, a.court, a.out, player_every=a.player_every, limit=a.limit)


if __name__ == "__main__":
    main()
