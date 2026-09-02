"""Draw the path-first track on top of a rally clip so a human can watch it.

    python3 render_pathfirst.py <rally> [--speed 0.5] [--out FILE]

Reads the frozen cell from pathfirst_tune.json, runs pathfirst.run on the
rally's caches, and writes an H.264 mp4 (960x540) with:
  - faint grey dots  = the candidate blobs the tracker could choose from
  - coloured trail   = the selected flight (one colour per flight)
  - white ring       = the tracked ball on this frame
  - "hit"/"bounce"   = flight-end labels, shown for a few frames
Viewer only: no truth is read, nothing is tuned, nothing is written back.
The clip must be the same one the caches were built from (r{N}_clip.mp4).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import pathfirst as pf                                     # noqa: E402

SP = Path(__file__).parent
COLORS = [(66, 135, 245), (52, 199, 89), (255, 149, 0), (255, 59, 48),
          (175, 82, 222), (90, 200, 250), (255, 204, 0), (255, 45, 85)]
TRAIL = 14          # frames of trail behind the ball
LABEL_HOLD = 18     # frames a hit/bounce label stays up


def ffmpeg_bin():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rally", type=int)
    ap.add_argument("--speed", type=float, default=0.5, help="playback speed (0.5 = half)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cell = json.loads(pf.TUNE_JSON.read_text())
    assert not cell.get("dead"), "no live path-first verdict"
    ctx = pf.context(a.rally)
    res = pf.run(ctx, cell["p_seed"], cell["s_min"], cell["gap"])
    chosen, track, t0 = res["chosen"], res["track"], ctx["t0"]
    bd = pf.boundaries(ctx, chosen)
    labels = {}     # frame -> (text, xy)
    for t, kind in bd:
        f = int(round((t - t0) * pf.FPS))
        if f in track:
            labels[f] = ("bounce" if "bounce" in kind else "hit", track[f])
    flight_of = {}
    for i, fl in enumerate(chosen):
        for f in range(fl["fa"], fl["fb"] + 1):
            flight_of[f] = i
    clip = SP / f"r{a.rally}_clip.mp4"
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_lo, f_hi = max(0, ctx["f_lo"] - 30), min(n - 1, ctx["f_hi"] + 30)
    raw = SP / f"_render_r{a.rally}_raw.mp4"
    out = Path(a.out) if a.out else SP / f"pathfirst_r{a.rally}.mp4"
    fps_out = 60.0 * a.speed
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (1280, 720))
    active = []     # (frame_shown_until, text, xy)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_lo)
    for f in range(f_lo, f_hi + 1):
        ok, img = cap.read()
        if not ok:
            break
        fr = ctx["frames"].get(f)
        if fr is not None:
            for x, y, p in fr[0]:
                cv2.circle(img, (int(x), int(y)), 2, (160, 160, 160), -1)
        i = flight_of.get(f)
        if i is not None:
            col = COLORS[i % len(COLORS)][::-1]       # BGR
            fa = max(chosen[i]["fa"], f - TRAIL)
            pts = [track[g] for g in range(fa, f + 1) if g in track]
            for k in range(1, len(pts)):
                cv2.line(img, (int(pts[k - 1][0]), int(pts[k - 1][1])),
                         (int(pts[k][0]), int(pts[k][1])), col, 2)
            x, y = track[f]
            cv2.circle(img, (int(x), int(y)), 9, (255, 255, 255), 2)
            cv2.circle(img, (int(x), int(y)), 9, col, 1)
        if f in labels:
            active.append((f + LABEL_HOLD, *labels[f]))
        active = [z for z in active if z[0] >= f]
        for _, text, (x, y) in active:
            cv2.circle(img, (int(x), int(y)), 14, (0, 255, 255) if text == "hit" else (0, 200, 255), 2)
            cv2.putText(img, text, (int(x) + 16, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (int(x) + 16, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
        hud = f"rally {a.rally}  t={t0 + f / pf.FPS:7.2f}s  frame {f}"
        if i is not None:
            hud += f"  flight {i + 1}/{len(chosen)}"
        cv2.putText(img, hud, (14, 612), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, hud, (14, 612), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        if a.speed != 1.0:
            cv2.putText(img, f"{a.speed:g}x speed", (14, 640), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    cap.release()
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-i", str(raw), "-vf", "scale=960:540",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    raw.unlink()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): frames {f_lo}-{f_hi}, "
          f"{len(chosen)} flights, {len(labels)} end labels, {a.speed:g}x")


if __name__ == "__main__":
    main()
