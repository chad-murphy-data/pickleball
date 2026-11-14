"""Watch the tracker decide what is a contact and what is a bounce.

The over-calling lever (owner-approved 2026-09-04).  ball_replicate
splits every direction change in the tracked path two ways:

    claimed by a hitter-chain anchor  -> CONTACT
    everything else                   -> BOUNCE      <- the bug

so a tracking wobble and a contact the anchors missed both come out as
bounce markers.  turn_audit.py grades each turn against the owner's own
reconstruction; this draws that grading on the video so the failures can
be watched instead of read.

    green   CONTACT   an anchor claimed it, and it is one
    blue    BOUNCE    emitted as a bounce, and it is one
    red     FALSE     emitted as a bounce, is tracking junk
    orange  MISSED    emitted as a bounce, is really a CONTACT the
                      anchor chain failed to claim
    grey    (claimed but the truth disagrees -- shown for completeness)

Each marker holds for HOLD_S so it can be paused on, and carries the
turn angle plus the vertical-velocity signature (falling -> rising is
what a real bounce does and a wobble does not).

    python3 vision/render_turns.py --rally 9 [--speed 0.4] [--out FILE]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import csv
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ballsearch"))

DATA = HERE.parent / "data" / "vision"
FPS = 60.0
TRAIL = 16          # frames of ball trail
HOLD_S = 1.2        # how long a turn marker stays on screen
BGR = {"CONTACT": (80, 220, 90), "BOUNCE": (245, 160, 60),
       "FALSE": (60, 60, 240), "MISSED": (0, 165, 255),
       "OTHER": (150, 150, 150)}


def ffmpeg_bin():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def classify(row):
    """what the pipeline emitted, crossed with what it really is."""
    if row["claimed"]:
        return ("CONTACT", "contact") if row["truth"] == "CONTACT" \
            else ("OTHER", f"contact? really {row['truth'].lower()}")
    return {"BOUNCE": ("BOUNCE", "bounce"),
            "WOBBLE": ("FALSE", "FALSE bounce (junk)"),
            "CONTACT": ("MISSED", "MISSED contact")}[row["truth"]]


def shadowed(img, txt, org, scale, col, th=1):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                th + 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th,
                cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rally", type=int, required=True)
    ap.add_argument("--speed", type=float, default=0.4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from claim_lab import load as c3load
    c = c3load(a.rally)
    t0, serve, end = c["t0"], c["imps"][0], c["dead"]
    path = sorted(c["timing_ref"])

    rows = []
    with open(DATA / f"turn_audit_r{a.rally}.csv") as f:
        for r in csv.DictReader(f):
            r["claimed"] = int(r["claimed"])
            r["t"], r["ang"] = float(r["t"]), float(r["ang"])
            r["dy_pre"], r["dy_post"] = float(r["dy_pre"]), float(r["dy_post"])
            kind, label = classify(r)
            r["kind"], r["label"] = kind, label
            rows.append(r)

    # marker pixel position = the tracked point nearest the turn time
    pt = np.asarray([[p[0], p[1], p[2]] for p in path], float)
    for r in rows:
        j = int(np.argmin(np.abs(pt[:, 0] - r["t"])))
        r["xy"] = (pt[j, 1], pt[j, 2]) if abs(pt[j, 0] - r["t"]) < 0.2 else None

    clip = HERE / "ballsearch" / f"r{a.rally}_clip.mp4"
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"no clip at {clip}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_lo = max(0, int((serve - 0.6 - t0) * FPS))
    f_hi = min(n - 1, int((end + 0.6 - t0) * FPS))

    out = Path(a.out) if a.out else HERE / "ballsearch" / f"turns_r{a.rally}.mp4"
    raw = out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                         FPS * a.speed, (1280, 720))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_lo)

    counts = {}
    for f in range(f_lo, f_hi + 1):
        ok, img = cap.read()
        if not ok:
            break
        t = t0 + f / FPS

        seg = [p for p in path if 0 <= t - p[0] <= TRAIL / FPS]
        for i in range(1, len(seg)):
            p, q = seg[i - 1], seg[i]
            cv2.line(img, (int(p[1]), int(p[2])), (int(q[1]), int(q[2])),
                     (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(img, (int(p[1]), int(p[2])), (int(q[1]), int(q[2])),
                     (0, 220, 255), 1, cv2.LINE_AA)
        if seg:
            cv2.circle(img, (int(seg[-1][1]), int(seg[-1][2])), 10,
                       (255, 255, 255), 2, cv2.LINE_AA)

        for r in rows:
            if r["xy"] is None or not (0 <= t - r["t"] <= HOLD_S):
                continue
            col = BGR[r["kind"]]
            x, y = int(r["xy"][0]), int(r["xy"][1])
            age = (t - r["t"]) / HOLD_S
            cv2.circle(img, (x, y), int(16 + 14 * age), col, 2, cv2.LINE_AA)
            cv2.circle(img, (x, y), 4, col, -1, cv2.LINE_AA)
            fall = r["dy_pre"] > 0 > r["dy_post"]
            shadowed(img, r["label"], (x + 22, y - 10), 0.62, col, 2)
            shadowed(img, f"turn {r['ang']:.0f}deg   "
                          f"{'falling->rising' if fall else 'no bounce signature'}",
                     (x + 22, y + 12), 0.48, col)

        counts = {}
        for r in rows:
            if r["xy"] is not None and r["t"] <= t:
                counts[r["kind"]] = counts.get(r["kind"], 0) + 1
        shadowed(img, f"r{a.rally}   t={t:7.2f}s   "
                      f"the tracker's contact-vs-bounce call, graded",
                 (16, 34), 0.68, (255, 255, 255), 2)
        for i, (k, txt) in enumerate((
                ("CONTACT", "contact (anchor claimed it)"),
                ("BOUNCE", "bounce - correct"),
                ("FALSE", "bounce - tracking junk"),
                ("MISSED", "bounce - really a missed contact"))):
            cv2.circle(img, (26, 664 + 0), 0, (0, 0, 0), 1)
            yy = 596 + i * 26
            cv2.circle(img, (28, yy - 5), 7, BGR[k], -1, cv2.LINE_AA)
            shadowed(img, f"{txt}   [{counts.get(k, 0)}]", (44, yy), 0.5,
                     BGR[k])
        vw.write(img)

    cap.release()
    vw.release()
    cmd = [ffmpeg_bin(), "-y", "-i", str(raw), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "20", str(out)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0:
        raw.unlink()
    else:
        print("ffmpeg failed, keeping raw mp4v:", r.stderr.decode()[-300:])
        out = raw
    print(f"-> {out}")
    print("   " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
