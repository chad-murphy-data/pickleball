"""Cheap sanity check, BEFORE building anything: is a paddle actually
visible in a tight crop around the hitter's wrist at contact time?

THE QUESTION (user, 2026-08-23, after watching the touch_attribute
review viewer catch a missed serve): a person-detector missing a
stationary server is a different failure than "can we find a paddle
near an already-tracked wrist" — the paddle search is bounded (a small
crop around a keypoint we already trust), unlike ball-tracking (search
the whole frame) which is the thing that got killed
(vision/ball_visibility.py, 64% in-play findability, POSTMORTEM.md).
This script answers ONLY "is it there to look at", not "can a model
find it" — same discipline as vision/ball_visibility.py's own
hand-labeled sanity check before any detector got built.

Needs the real video + ffmpeg, so it runs on the user's machine, same
as pose_extract.py and every --video step in touch_attribute.py.

Samples labeled contacts across a SPREAD of shot types (fast smashes
and slow dinks specifically — the pitch was that paddle POSITION might
beat wrist SPEED on soft shots where the wrist barely moves), finds
the labeled hitter's wrist via the pose data already extracted, and
crops a tight frame around it with ffmpeg. Also crops the other three
players at the same instant, for a same-frame baseline: if the paddle
is visible on non-hitters just as often, it says more about lighting/
resolution than about swings.

Output is a single self-contained HTML contact sheet (embedded
base64 JPEGs, nothing external) - easy to eyeball locally, or to send
back for a second pair of eyes without shipping the raw video.

RUN (on the machine with the video + ffmpeg; needs the pose_rtm/ npz
already extracted, same rallies as before):
    python vision/paddle_probe.py --video full_match.mp4.webm \
        --pose-dir pose_rtm --n 24 --out paddle_probe.html
"""
from __future__ import annotations

import argparse
import base64
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import touch_attribute as TA          # noqa: E402
import swing_explore as SE            # noqa: E402
from contact_ceiling import L_WRIST, R_WRIST, load_rosters, load_labels  # noqa: E402

LABELS = Path(__file__).parent.parent / "data/vision/contact_labels_chicago0725.csv"
WINDOWS = Path(__file__).parent.parent / "data/vision/rally_windows_chicago0725_v4.csv"
CROP = 130          # half-size in px of the extraction frame (native res)
BUILD = "2026-08-23  paddle_probe: is it even there to look at"


def ffprobe_dims(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=s=x:p=0", video],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


def nearest_kpt(rd, tid, t, tol=0.4):
    """(x, y, conf) of the more-confident wrist for this track near t,
    in the pose model's OWN working frame (1280x720 per hw) - the
    exact same coordinate space box_at/cx/ynorm already use everywhere
    else in this file, so nothing new is being trusted here."""
    z = rd["z"]
    m = z["track"] == tid
    ts = z["t"][m]
    if not len(ts):
        return None
    i = int(abs(ts - t).argmin())
    if abs(ts[i] - t) > tol:
        return None
    kpt, kpc = z["kpt"][m][i], z["kpc"][m][i]
    lw, rw = kpt[L_WRIST], kpt[R_WRIST]
    lc, rc = float(kpc[L_WRIST]), float(kpc[R_WRIST])
    x, y, c = (lw[0], lw[1], lc) if lc >= rc else (rw[0], rw[1], rc)
    return float(x), float(y), c


def extract_crop(video, t, cx, cy, sx, sy, out_path):
    """ffmpeg: seek to t, crop CROPxCROP centered on the wrist point
    scaled from the pose model's 1280x720 working frame to the video's
    NATIVE resolution (sx, sy are those scale factors) - a mismatch
    here would crop the wrong part of the frame silently, so this is
    the one piece of arithmetic in the whole script worth double-
    checking against ffprobe's own reported dimensions."""
    x = int(cx * sx - CROP / 2)
    y = int(cy * sy - CROP / 2)
    vf = f"crop={CROP}:{CROP}:{max(0, x)}:{max(0, y)}"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", video,
         "-frames:v", "1", "-vf", vf, "-q:v", "2", str(out_path)],
        capture_output=True, check=True)


def load_track_names(review_html):
    """{rally_cum: {tid: name}} straight from an already-generated
    --review-html file (touch_attribute.py) - that mapping was already
    computed by the production labelling path and eyeballed against
    real video once already, so re-deriving the lineup state machine a
    THIRD time here would be pure duplication risk for no new
    information."""
    import json
    import re
    html = Path(review_html).read_text()
    m = re.search(r"const DATA = (\{.*?\});", html, re.S)
    if not m:
        raise SystemExit(f"couldn't find embedded DATA in {review_html} "
                         "- is this a touch_attribute --review-html file?")
    data = json.loads(m.group(1))
    return {int(cum): {tr["tid"]: tr["name"] for tr in d["tracks"]}
            for cum, d in data.items()}


def pick_samples(labels_by_rally, rallies_with_pose, n):
    """A SPREAD across shot types, not just the first N contacts -
    the pitch was specifically about dinks/slow shots, where wrist
    SPEED is weak but paddle POSITION might not be, so a sample that's
    all serves and drives would never test the actual question."""
    by_type = defaultdict(list)
    for cum, rows in labels_by_rally.items():
        if cum not in rallies_with_pose:
            continue
        for r in rows:
            if r.get("contact", "1") != "1":
                continue
            t = float(r["t_refined_s"] or r["t_tap_s"])
            st = (r.get("shot_type") or "other").strip() or "other"
            by_type[st].append((cum, t, r["hitter_name"], st))
    types = sorted(by_type, key=lambda k: -len(by_type[k]))
    out, i = [], 0
    while len(out) < n and any(by_type.values()):
        for st in types:
            if by_type[st]:
                idx = (i * 7) % len(by_type[st])   # spread, not clustered
                out.append(by_type[st].pop(idx))
                if len(out) >= n:
                    break
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--pose-dir", default="pose_rtm")
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--windows", default=str(WINDOWS))
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="paddle_probe.html")
    ap.add_argument("--tmp-dir", default="paddle_probe_tmp")
    ap.add_argument("--review-html", default=None,
                    help="an already-generated touch_attribute.py "
                         "--review-html file, to mark which crop is "
                         "the labeled hitter's own track. Without it, "
                         "crops are shown unmarked rather than "
                         "guessed at")
    a = ap.parse_args()
    print(f"paddle_probe build: {BUILD}")

    for exe in ("ffmpeg", "ffprobe"):
        if subprocess.run(["which", exe], capture_output=True).returncode:
            raise SystemExit(f"{exe} not found - this needs to run on a "
                             "machine with ffmpeg installed")

    rosters = load_rosters(Path(a.windows))
    labels_by_rally = defaultdict(list)
    for r in csv.DictReader(open(a.labels)):
        labels_by_rally[int(r["rally_cum"])].append(r)

    tmp_dir = Path(a.tmp_dir)
    tmp_dir.mkdir(exist_ok=True)
    rallies_with_pose = {int(p.stem[1:]) for p in
                         Path(a.pose_dir).glob("r*.npz")}
    samples = pick_samples(labels_by_rally, rallies_with_pose, a.n)
    if not samples:
        raise SystemExit("no labeled contacts found in a rally with pose "
                         "data - check --pose-dir / --labels")
    print(f"sampled {len(samples)} contacts across "
          f"{len(set(s[3] for s in samples))} shot types")

    vw, vh = ffprobe_dims(a.video)
    sx, sy = vw / 1280.0, vh / 720.0
    print(f"video native {vw}x{vh}  (pose worked at 1280x720, "
          f"scale {sx:.2f}x{sy:.2f})")

    track_names = load_track_names(a.review_html) if a.review_html else {}
    if a.review_html:
        print(f"loaded track names for {len(track_names)} rallies from "
              f"{a.review_html}")
    else:
        print("no --review-html given - crops will be unmarked "
              "(can't tell which one is the labeled hitter)")

    rd_cache = {}
    cards = []
    for k, (cum, t, hitter, shot_type) in enumerate(samples):
        if cum not in rd_cache:
            rd_cache[cum] = SE.load_rally(a.pose_dir, cum)
        rd = rd_cache[cum]
        if rd is None:
            continue
        tids = TA.player_tracks(rd)
        if len(tids) != 4:
            continue
        names = track_names.get(cum, {})
        crops = []
        for tid in tids:
            kp = nearest_kpt(rd, tid, t)
            if kp is None:
                continue
            x, y, conf = kp
            fname = tmp_dir / f"r{cum}_{k}_{tid}.jpg"
            try:
                extract_crop(a.video, t, x, y, sx, sy, fname)
            except subprocess.CalledProcessError:
                continue
            if fname.exists() and fname.stat().st_size:
                is_hitter = names.get(tid) == hitter
                crops.append((tid, conf, fname, is_hitter, names.get(tid)))
        if crops:
            cards.append((cum, t, hitter, shot_type, crops))
        print(f"  [{k + 1}/{len(samples)}] r{cum} t={t:.2f}s "
              f"{hitter} ({shot_type}): {len(crops)} crops")

    if not cards:
        raise SystemExit("no crops extracted - check the video path and "
                         "that ffmpeg can seek it")

    _write_html(cards, a.out)
    print(f"\nwrote {a.out} - {len(cards)} contacts, "
          f"{sum(len(c[4]) for c in cards)} crops total")
    print("  open it and eyeball: is a paddle visible in the crops "
          "labeled with the hitter's name? in the other three?")


def _write_html(cards, out_path):
    def b64(path):
        return base64.b64encode(Path(path).read_bytes()).decode()

    parts = ["""<!doctype html><html><head><meta charset="utf-8">
<title>paddle probe</title>
<style>
  body { font-family: -apple-system, sans-serif; background: #111;
         color: #eee; padding: 16px; }
  .card { display: inline-block; margin: 10px; vertical-align: top;
          background: #1c1c1c; border-radius: 6px; padding: 8px; }
  .row { display: flex; gap: 4px; }
  .crop { text-align: center; }
  .crop img { width: 130px; height: 130px; object-fit: cover;
              border-radius: 4px; border: 2px solid #333; }
  .hitter img { border-color: #4caf50; }
  .label { font-size: 11px; margin-top: 2px; }
  h4 { margin: 4px 0; }
</style></head><body>
<h2>paddle probe</h2>
<p>Green border = the labeled hitter's own wrist crop. The other three
are the same instant, other players, for a same-frame baseline.
Question: is a paddle visible near the wrist, in either group?</p>
"""]
    for cum, t, hitter, shot_type, crops in cards:
        parts.append(f'<div class="card"><h4>r{cum} @ {t:.2f}s — '
                     f'{hitter} ({shot_type})</h4><div class="row">')
        for tid, conf, path, is_hitter, name in crops:
            cls = "crop hitter" if is_hitter else "crop"
            label = name or f"tid {tid}"
            parts.append(
                f'<div class="{cls}"><img src="data:image/jpeg;base64,'
                f'{b64(path)}"><div class="label">{label}<br>'
                f'wrist conf {conf:.2f}</div></div>')
        parts.append("</div></div>")
    parts.append("</body></html>")
    Path(out_path).write_text("".join(parts))


if __name__ == "__main__":
    main()
