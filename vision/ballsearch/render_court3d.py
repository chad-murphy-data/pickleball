"""court3d_r{N}.html -> court3d_r{N}.mp4 (owner ask 2026-09-02: the 3D viewer
as a video, easier to post).

Reads the PATH / IMPACTS / PLAYERS arrays straight out of the committed
viewer HTML and replays them with the viewer's own orthographic projection
and colours, in real time, with a slow orbit.  Differences from the page,
all cosmetic and all in the honest direction: the path is broken at
flight gaps instead of joined by a straight line (the page joins
consecutive samples regardless), the ball is hidden while the track is
lost, and a ring flashes at each attributed hit (the page loads IMPACTS
and never draws them).  Nothing is re-fit; the file is the input.

    python3 render_court3d.py 9            ->  court3d_r9.mp4  (1280x720, 30 fps)
"""
import argparse
import json
import math
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

SP = Path(__file__).parent
W, H, FPS = 1280, 720, 30
SS = 2                              # supersample, then INTER_AREA down
LEAD, TAIL = 1.0, 2.5               # seconds held before / after the rally
GAP_S = 0.03                        # path samples further apart than this are a lost track
BALL_HOLD = 0.12                    # ball shown this long after its last sample
HIT_FLASH = 0.35
AZ0, EL, ZOOM = -2.4, 0.5, 13.0     # the viewer's defaults
AZ_SWEEP = 0.55                     # radians of orbit across the rally


def hexc(s, a=1.0, bg=(0x0c, 0x0c, 0x10)):
    r, g, b = int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
    r, g, b = (round(a * v + (1 - a) * w) for v, w in zip((r, g, b), bg))
    return (b, g, r)


BG = hexc("#0c0c10")
LINE = hexc("#3b6ea5")
NETP = hexc("#777777")
TAPE = hexc("#cccccc")
PATH_FAINT = hexc("#e8c44a", 0.25)
PATH_LIT = hexc("#ffd94a")
NEAR, FAR = hexc("#e05c5c"), hexc("#5ca8e0")
WHITE = (255, 255, 255)
GREY = (170, 170, 170)


def load(html):
    s = Path(html).read_text()
    path = json.loads(re.search(r"const PATH = (\[.*?\]);\n", s).group(1))
    imp = json.loads(re.search(r"const IMPACTS = (\[.*?\]);\n", s).group(1))
    pl = json.loads(re.search(r"const PLAYERS = (\{.*?\});\n", s).group(1))
    return np.array(path, float), np.array(imp, float), {k: np.array(v, float) for k, v in pl.items()}


class Cam:
    def __init__(self, az, el, zoom, w, h):
        self.ca, self.sa = math.cos(az), math.sin(az)
        self.ce, self.se = math.cos(el), math.sin(el)
        self.zoom, self.cx, self.cy = zoom, w / 2, h / 2

    def __call__(self, x, y, z):
        X = (x - 10) * self.ca - (y - 22) * self.sa
        Y = (x - 10) * self.sa + (y - 22) * self.ca
        u, v = X, -z * self.ce - Y * self.se
        return (int(round(self.cx + u * self.zoom)), int(round(self.cy + v * self.zoom)))


def line(img, cam, a, b, col, w):
    cv2.line(img, cam(*a), cam(*b), col, w, cv2.LINE_AA)


def court(img, cam):
    S = [[[0, 0], [20, 0]], [[0, 44], [20, 44]], [[0, 0], [0, 44]], [[20, 0], [20, 44]],
         [[0, 15], [20, 15]], [[0, 29], [20, 29]], [[10, 0], [10, 15]], [[10, 29], [10, 44]]]
    for s in S:
        line(img, cam, (s[0][0], s[0][1], 0), (s[1][0], s[1][1], 0), LINE, int(1.5 * SS))
    for x in range(-1, 22):
        h = 2.833 + 0.167 * abs(x - 10) / 11
        line(img, cam, (x, 22, 0), (x, 22, h), NETP, SS)
    line(img, cam, (-1, 22, 3), (21, 22, 2.84), TAPE, 2 * SS)
    line(img, cam, (21, 22, 2.84), (21, 22, 3), TAPE, 2 * SS)


def interp(tr, t):
    i = int(np.searchsorted(tr[:, 0], t))
    if i <= 0:
        return tr[0, 1], tr[0, 2]
    if i >= len(tr):
        return tr[-1, 1], tr[-1, 2]
    a, b = tr[i - 1], tr[i]
    f = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
    f = min(1.0, max(0.0, f))
    return a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f


def frame(path, imp, players, tcur, az, t_start, hits_so_far):
    w, h = W * SS, H * SS
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BG
    cam = Cam(az, EL, ZOOM * SS, w, h)
    court(img, cam)
    dt = np.diff(path[:, 0])
    ok = dt <= GAP_S                                  # segment i-1 -> i is within one flight
    for i in np.where(ok)[0]:
        line(img, cam, path[i, 1:], path[i + 1, 1:], PATH_FAINT, SS)
    lit = np.where(ok & (path[1:, 0] <= tcur))[0]
    for i in lit:
        recent = tcur - path[i + 1, 0] <= 0.6
        line(img, cam, path[i, 1:], path[i + 1, 1:], PATH_LIT, (3 if recent else 2) * SS)
    for nm, tr in players.items():
        x, y = interp(tr, tcur)
        col = NEAR if nm.startswith("near") else FAR
        line(img, cam, (x, y, 0), (x, y, 5.3), col, 3 * SS)
        hp = cam(x, y, 5.8)
        cv2.circle(img, hp, 4 * SS, col, -1, cv2.LINE_AA)
        cv2.putText(img, nm, (hp[0] + 7 * SS, hp[1] - 4 * SS), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42 * SS, col, SS, cv2.LINE_AA)
    # hit flashes: ring at the ball's position at the hit, growing and fading
    for ti in imp:
        age = tcur - ti
        if 0 <= age <= HIT_FLASH:
            k = int(np.argmin(np.abs(path[:, 0] - ti)))
            if abs(path[k, 0] - ti) <= 0.15:
                p = cam(*path[k, 1:])
                f = age / HIT_FLASH
                col = hexc("#ffffff", 1 - f)
                cv2.circle(img, p, int((6 + 16 * f) * SS), col, 2 * SS, cv2.LINE_AA)
    # ball: last sample at or before tcur, hidden once the track is lost
    j = int(np.searchsorted(path[:, 0], tcur, side="right")) - 1
    if j >= 0 and tcur - path[j, 0] <= BALL_HOLD:
        bx, by, bz = path[j, 1:]
        s = cam(bx, by, 0)
        cv2.ellipse(img, s, (5 * SS, 2 * SS), 0, 0, 360, hexc("#000000", 0.5), -1, cv2.LINE_AA)
        cv2.circle(img, cam(bx, by, bz), 5 * SS, PATH_LIT, -1, cv2.LINE_AA)
    out = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return out


def ffmpeg_bin():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rally", type=int)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    a = ap.parse_args()
    src = SP / f"court3d_r{a.rally}.html"
    out = SP / f"court3d_r{a.rally}.mp4"
    path, imp, players = load(src)
    t0, t1 = path[0, 0], path[-1, 0]
    dur = t1 - t0
    n_lead, n_tail = int(LEAD * FPS), int(TAIL * FPS)
    n_play = int(math.ceil(dur / a.speed * FPS))
    raw = out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    cap1 = (f"rally {a.rally} in 3D  ·  reconstructed from one broadcast camera  ·  "
            f"yellow = tracked ball, breaks = lost track  ·  rings = attributed hits")
    cap2 = ("player marks = pose floor positions, coloured by side  ·  "
            "depth (down the court) is the weak axis of a one-camera fit")
    for k in range(n_lead + n_play + n_tail):
        tp = min(max(0.0, (k - n_lead) / FPS * a.speed), dur)   # rally-relative playhead
        tcur = t0 + tp
        az = AZ0 + AZ_SWEEP * (tp / dur)
        hits = int(np.sum(imp <= tcur))
        img = frame(path, imp, players, tcur, az, t0, hits)
        cv2.putText(img, f"rally {a.rally}  -  3D", (14, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, f"t = {tp:5.2f} s    hits {hits}/{len(imp)}"
                    + (f"    {a.speed:g}x" if a.speed != 1 else ""),
                    (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1, cv2.LINE_AA)
        cv2.putText(img, cap1, (14, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREY, 1, cv2.LINE_AA)
        cv2.putText(img, cap2, (14, 702), cv2.FONT_HERSHEY_SIMPLEX, 0.48, GREY, 1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-i", str(raw),
           "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    raw.unlink()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {n_lead + n_play + n_tail} frames "
          f"@ {FPS} fps, rally {dur:.1f} s at {a.speed:g}x, {len(path)} samples, "
          f"{len(imp)} hits, {len(players)} players")


if __name__ == "__main__":
    main()
