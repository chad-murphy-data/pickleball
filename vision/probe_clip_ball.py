"""Video probe: is the ball detectable in a broadcast clip? (RUN + VALIDATED
2026-08-10 on a 60 s, 360p clip of Chicago Slice v Utah Black Diamonds,
game 1, rallies 29-30 — found via the scorebug reading 9-5.)

    python vision/probe_clip_ball.py <clip.mp4>

WHAT IT ESTABLISHED, on real footage, with zero ML:
  * Ball IS detectable at 360p: chroma ("yellowness" = min(R,G)-B, gated by
    change vs a median background) found 796 candidates, median area 8 px;
    nearest-neighbour linking gave 11 fast tracks (>80 px/s, >=10 frames),
    ALL inside rally windows; 92 ball positions in rally 30 alone. At 1080p
    the ball is ~9-10 px across — comfortable for a learned tracker.
  * Static camera confirmed: frame-to-frame motion median 0.64 grey levels.
  * Scorebug flip = FRAME-EXACT sync: the CHICAGO 5->6 digit change hit
    frame 1742 with region delta 185 vs next-biggest 48, anchoring video
    time to the referee log at 18:32:18Z. Scorebug OCR sync validated in
    miniature — no OCR model needed for flips, a pixel-region diff does it.
  * CAUTION: raw candidate DENSITY is not a validator — players walking
    toward the camera between rallies shed yellow-ish candidates (27/s in
    dead time vs 9-15/s in-rally). Fast smooth TRACKS are the validator.
"""
import csv
import subprocess
import sys

import numpy as np
from scipy import ndimage

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
CLIP = sys.argv[1] if len(sys.argv) > 1 else "data/vision/clip.mp4"
import os
S = os.path.dirname(os.path.abspath(CLIP)) or "."
W, H = 640, 360          # set to the clip's actual resolution


def frames():
    cmd = [FF, "-v", "error", "-i", CLIP, "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 3 * 4)
    n = W * H * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)


def yellowness(f):
    f = f.astype(np.int16)
    return np.clip(np.minimum(f[..., 0], f[..., 1]) - f[..., 2], 0, 255)


# ---- pass 1: background + motion curve --------------------------------
bg_s, motion, prev = [], [], None
count = 0
for i, f in enumerate(frames()):
    g = f.mean(axis=2, dtype=np.float32)
    motion.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
    prev = g
    if i % 30 == 0:
        bg_s.append(f.copy())
    count = i + 1
bg = np.median(np.stack(bg_s), axis=0).astype(np.uint8)
ybg = yellowness(bg)
motion = np.array(motion)
print(f"{count} frames at 30fps = {count/30:.1f}s   "
      f"motion curve: median {np.median(motion):.2f}, p95 {np.percentile(motion,95):.2f}")
# cuts would be huge spikes
print(f"   cut-sized spikes (>6x median): {np.sum(motion > 6*np.median(motion))}")

# ---- pass 2: ball candidates ------------------------------------------
cands = []          # (frame, x, y, area, strength)
scorebug = []       # mean of the CHICAGO score digit region per frame
trail = np.zeros((H, W), np.float32)
for i, f in enumerate(frames()):
    dy = yellowness(f).astype(np.int16) - ybg
    mask = dy > 45
    if mask.any():
        lab, n = ndimage.label(mask)
        if n:
            areas = ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1))
            for j, a in enumerate(areas, 1):
                if 1 <= a <= 40:
                    ys, xs = np.nonzero(lab == j)
                    stren = float(dy[ys, xs].mean())
                    cands.append((i, float(xs.mean()), float(ys.mean()),
                                  int(a), stren))
    trail = np.maximum(trail, np.clip(dy, 0, 255) * ((dy > 45)))
    # scorebug digit region (CHICAGO score, from the viewed frames): x 148-166, y 40-58
    scorebug.append(float(f[40:58, 148:166].mean()))

print(f"ball candidates: {len(cands)} over {count} frames "
      f"({len(cands)/count:.2f}/frame)")
sizes = [c[3] for c in cands]
if sizes:
    print(f"   candidate area px: median {np.median(sizes):.0f}  "
          f"p90 {np.percentile(sizes,90):.0f}")

# ---- link into tracks --------------------------------------------------
byf = {}
for c in cands:
    byf.setdefault(c[0], []).append(c)
tracks, open_tr = [], []
for i in range(count):
    cur = byf.get(i, [])
    used = set()
    nxt = []
    for tr in open_tr:
        li, lx, ly = tr[-1][0], tr[-1][1], tr[-1][2]
        if i - li > 3:
            tracks.append(tr)
            continue
        best, bj = 45.0 * (i - li), None
        for j, c in enumerate(cur):
            if j in used:
                continue
            d = ((c[1] - lx) ** 2 + (c[2] - ly) ** 2) ** 0.5
            if d < best:
                best, bj = d, j
        if bj is not None:
            used.add(bj)
            tr.append(cur[bj])
            nxt.append(tr)
        else:
            nxt.append(tr) if i - li <= 3 else tracks.append(tr)
    for j, c in enumerate(cur):
        if j not in used:
            nxt.append([c])
    open_tr = nxt
tracks.extend(open_tr)
long_tr = [t for t in tracks if len(t) >= 8]
print(f"tracks >=8 frames: {len(long_tr)}")
for t in sorted(long_tr, key=len, reverse=True)[:8]:
    dur = (t[-1][0] - t[0][0]) / 30.0
    dx = np.diff([c[1] for c in t])
    dy_ = np.diff([c[2] for c in t])
    sp = np.hypot(dx, dy_).mean() * 30
    print(f"   frames {t[0][0]:4d}-{t[-1][0]:4d} ({dur:4.1f}s) "
          f"len {len(t):3d}  mean speed {sp:5.0f} px/s  "
          f"area~{np.median([c[3] for c in t]):.0f}px")

# ---- scorebug flip -----------------------------------------------------
sb = np.array(scorebug)
d = np.abs(np.diff(sb))
flip = int(np.argmax(d))
print(f"scorebug digit region: biggest change at frame {flip} "
       f"(t={flip/30:.2f}s), delta {d.max():.1f} (next biggest {np.sort(d)[-2]:.1f})")

# ---- save artifacts ----------------------------------------------------
np.save(f"{S}/motion.npy", motion)
with open(f"{S}/ball_candidates.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["frame", "x", "y", "area", "strength"])
    for c in cands:
        w.writerow([c[0], f"{c[1]:.1f}", f"{c[2]:.1f}", c[3], f"{c[4]:.1f}"])

def save_png(arr, path):
    a = np.clip(arr, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    p = subprocess.run([FF, "-v", "error", "-y", "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-i", "-",
                        "-frames:v", "1", path], input=a.tobytes())
    assert p.returncode == 0

# trail alone, and trail burned onto the background in red
save_png(trail * (255.0 / max(trail.max(), 1)), f"{S}/trail.png")
over = bg.astype(np.float32) * 0.45
m = trail > 30
over[..., 0][m] = 255
over[..., 1][m] = 40
over[..., 2][m] = 40
save_png(over, f"{S}/trail_overlay.png")
print("wrote trail.png, trail_overlay.png, ball_candidates.csv, motion.npy")
