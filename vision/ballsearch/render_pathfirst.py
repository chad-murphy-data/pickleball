"""Draw the path-first track on top of a rally clip so a human can watch it.

    python3 render_pathfirst.py <rally> [--speed 0.5] [--out FILE] [--reddit]

Reads the frozen cell from pathfirst_tune.json, runs pathfirst.run on the
rally's caches, and writes an H.264 mp4 (960x540) with:
  - faint grey dots  = the candidate blobs the tracker could choose from
  - coloured trail   = the selected flight (one colour per flight)
  - white ring       = the tracked ball on this frame
  - labels at flight ends, shown for a few frames: with the adopted
    events layer (events_tune_v3.json) one "event" per change of flight
    (hit or bounce, untyped); "arrive"/"depart" mark the two sides of a
    gap where the tracker lost the ball. Without it, raw hit/bounce ends.
--reddit (owner request 2026-09-02, a share-able cut): full 1280x720, no
labels and no HUD, the trail and ring always red, and the four tracked
players' pose keypoints + skeleton drawn from the rally's pose npz.
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
RED = (0, 0, 255)   # BGR
# COCO-17 skeleton (pose npz keypoint order)
SKEL = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
KP_CONF = 0.3


def ffmpeg_bin():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def pose_by_frame(npz_path, t0, f_lo, f_hi):
    """frame -> list of (kpt(17,2), kpc(17)) for the four longest tracks."""
    z = np.load(npz_path)
    tids = sorted(set(z["track"].tolist()),
                  key=lambda k: -(z["track"] == k).sum())[:4]
    fr = np.round((np.asarray(z["t"]) - t0) * 60).astype(int)
    keep = np.isin(z["track"], tids) & (fr >= f_lo) & (fr <= f_hi)
    out = {}
    for i in np.where(keep)[0]:
        out.setdefault(int(fr[i]), []).append((z["kpt"][i], z["kpc"][i]))
    return out


def draw_pose(img, poses):
    for kpt, kpc in poses:
        ok = kpc >= KP_CONF
        for a, b in SKEL:
            if ok[a] and ok[b]:
                cv2.line(img, (int(kpt[a, 0]), int(kpt[a, 1])),
                         (int(kpt[b, 0]), int(kpt[b, 1])), (40, 40, 40), 3, cv2.LINE_AA)
                cv2.line(img, (int(kpt[a, 0]), int(kpt[a, 1])),
                         (int(kpt[b, 0]), int(kpt[b, 1])), (255, 220, 120), 1, cv2.LINE_AA)
        for j in np.where(ok)[0]:
            cv2.circle(img, (int(kpt[j, 0]), int(kpt[j, 1])), 4, (40, 40, 40), -1, cv2.LINE_AA)
            cv2.circle(img, (int(kpt[j, 0]), int(kpt[j, 1])), 3, (255, 255, 255), -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rally", type=int)
    ap.add_argument("--speed", type=float, default=0.5, help="playback speed (0.5 = half)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reddit", action="store_true",
                    help="share cut: no labels/HUD, red ball, pose skeletons, 1280x720")
    a = ap.parse_args()
    cell = json.loads(pf.TUNE_JSON.read_text())
    assert not cell.get("dead"), "no live path-first verdict"
    ctx = pf.context(a.rally)
    res = pf.run(ctx, cell["p_seed"], cell["s_min"], cell["gap"])
    chosen, track, t0 = res["chosen"], res["track"], ctx["t0"]
    labels = {}     # frame -> (text, xy)
    ev_json = SP / "events_tune_v3.json"
    if a.reddit:
        pass
    elif ev_json.exists() and not json.loads(ev_json.read_text()).get("dead"):
        # adopted events layer (events_gate.md v3): one label per change of flight
        import events as evm
        cell = json.loads(ev_json.read_text())
        evs = evm.events(ctx, chosen, cell["r_seam"], cell["a_seam"], cell["dt_pair"],
                         cell["off"], d_pair=cell["d_pair"])
        for e in evs:
            f = int(round((e["t"] - t0) * pf.FPS))
            near = min(track, key=lambda g: abs(g - f)) if track else None
            if near is not None and abs(near - f) <= 6:
                labels[f] = ("event" if e["how"] == "pair" or e["how"] == "serve"
                             else e["how"], track[near])
    else:
        for t, kind in pf.boundaries(ctx, chosen):
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
    poses = pose_by_frame(ctx["c"]["npz"], t0, f_lo, f_hi) if a.reddit else {}
    raw = SP / f"_render_r{a.rally}_raw.mp4"
    suffix = "_reddit" if a.reddit else ""
    out = Path(a.out) if a.out else SP / f"pathfirst_r{a.rally}{suffix}.mp4"
    fps_out = 60.0 * a.speed
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (1280, 720))
    active = []     # (frame_shown_until, text, xy)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_lo)
    for f in range(f_lo, f_hi + 1):
        ok, img = cap.read()
        if not ok:
            break
        if a.reddit and f in poses:
            draw_pose(img, poses[f])
        fr = ctx["frames"].get(f)
        if fr is not None:
            for x, y, p in fr[0]:
                cv2.circle(img, (int(x), int(y)), 2, (160, 160, 160), -1)
        i = flight_of.get(f)
        if i is not None:
            col = RED if a.reddit else COLORS[i % len(COLORS)][::-1]       # BGR
            fa = max(chosen[i]["fa"], f - TRAIL)
            pts = [track[g] for g in range(fa, f + 1) if g in track]
            for k in range(1, len(pts)):
                p, q = (int(pts[k - 1][0]), int(pts[k - 1][1])), (int(pts[k][0]), int(pts[k][1]))
                if a.reddit:
                    cv2.line(img, p, q, (255, 255, 255), 5, cv2.LINE_AA)
                    cv2.line(img, p, q, col, 3, cv2.LINE_AA)
                else:
                    cv2.line(img, p, q, col, 2)
            x, y = track[f]
            if a.reddit:
                cv2.circle(img, (int(x), int(y)), 11, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.circle(img, (int(x), int(y)), 11, col, 2, cv2.LINE_AA)
                cv2.circle(img, (int(x), int(y)), 3, col, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, (int(x), int(y)), 9, (255, 255, 255), 2)
                cv2.circle(img, (int(x), int(y)), 9, col, 1)
        if f in labels:
            active.append((f + LABEL_HOLD, *labels[f]))
        active = [z for z in active if z[0] >= f]
        for _, text, (x, y) in active:
            cv2.circle(img, (int(x), int(y)), 14, (0, 255, 255) if text in ("hit", "event") else (0, 200, 255), 2)
            cv2.putText(img, text, (int(x) + 16, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (int(x) + 16, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if not a.reddit:
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
    scale = ["-vf", "scale=960:540"] if not a.reddit else []
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-i", str(raw), *scale,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20" if a.reddit else "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    raw.unlink()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): frames {f_lo}-{f_hi}, "
          f"{len(chosen)} flights, {len(labels)} end labels, {a.speed:g}x"
          f"{', reddit cut' if a.reddit else ''}")


if __name__ == "__main__":
    main()
