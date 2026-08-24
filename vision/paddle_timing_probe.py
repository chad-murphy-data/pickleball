"""Does PADDLE position localize contact time tighter than WRIST speed?

THE ACTUAL BET (user, 2026-08-23, after paddle_probe.py confirmed a
paddle is visible in a wrist-anchored crop almost everywhere). Wrist
speed already localizes contacts to a median 0.27s against the labeled
serve-to-lunge sample, but it is WORST specifically on soft shots -
dink 0.46s, counter 0.45s, lob 0.39s median error - vs 0.16-0.19s on
fast/drive/drop (measured this session, no new labels, straight off
the pose data already extracted). A soft shot barely moves the wrist,
so wrist SPEED is a weak signal there by construction; the paddle
still has to move to make contact, so paddle POSITION might not be.

This is the test BEFORE building anything permanent: a zero-shot
open-vocabulary detector (OWL-ViT, no training, no new labels) run on
a short frame sequence cropped around the LABELED HITTER's own wrist
at the LABELED true contact time - same discipline as paddle_probe.py,
one step further. If paddle-centroid speed beats wrist speed on the
soft-shot sample specifically, a trained detector is worth building.
If it doesn't, the paddle idea is a real, tested null, not a hunch.

FIX (2026-08-24): the first version centered the search window on
t_wrist (wrist-speed's own guess), not t_true (the label) - so on a
shot where wrist speed is a bad estimator (the whole premise for
testing dinks), paddle position never got an independent look at the
real event; it just chased whatever wrist-speed already (possibly
wrongly) pointed at, sometimes on the wrong player entirely. First run
showed paddle getting WORSE than wrist specifically on dink/lob, and
the extracted crops confirmed it: full dynamic swings, not soft
touches - proof the window had drifted off the labeled event. Now
requires --review-html (a touch_attribute.py --review-html file) to
look up the hitter's own track id and anchors both the crop location
and the extraction window on t_true directly.

Needs the real video + ffmpeg (runs on the user's machine, same as
every --video step) AND a zero-shot detector:
    pip install torch transformers pillow

RUN:
    python vision/paddle_timing_probe.py --video full_match.mp4.webm \
        --pose-dir pose_rtm --review-html review_train.html --n 24
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import touch_attribute as TA          # noqa: E402
import swing_explore as SE            # noqa: E402
import paddle_probe as PP             # noqa: E402

LABELS = Path(__file__).parent.parent / "data/vision/contact_labels_chicago0725.csv"
BUILD = "2026-08-24  paddle_timing_probe: anchor search on t_true, not t_wrist"

# the soft shots are the whole bet; a few fast ones ride along as a
# control - if paddle timing made THOSE worse, that would be a warning
# sign the crop/detector setup itself is broken, not evidence for or
# against the actual hypothesis
FOCUS_TYPES = ["dink", "counter", "lob", "return", "lunge"]
CONTROL_TYPES = ["drive", "fast", "drop"]

CROP = 260          # native px, wider than paddle_probe's 130: the
                    # window spans +-0.5s, and the crop has to hold
                    # the hand across whatever the arm does in that time
WINDOW = 0.5        # seconds each side of the wrist-speed peak
FPS_OUT = 10        # frames/sec extracted for the sequence


def wrist_peak(rd, t_true, tids):
    """The SAME method used to establish the baseline this session:
    whichever of the 4 tracks has the highest torso-relative arm-speed
    peak within 0.6s of the true contact is the swinger, and the time
    of that peak is the wrist-speed estimate. Returns (tid, t_peak) or
    None."""
    best_tid, best_t, best_v = None, None, -1
    for tid in tids:
        ser = rd["tracks"][tid]
        ts, arm = ser["t"], ser["arm"]
        win = (ts >= t_true - 0.6) & (ts <= t_true + 0.6)
        if not win.any():
            continue
        idx = win.nonzero()[0]
        i = idx[arm[idx].argmax()]
        if arm[i] > best_v:
            best_v, best_tid, best_t = arm[i], int(tid), float(ts[i])
    return None if best_tid is None else (best_tid, best_t)


def sample_focus(labels_by_rally, rallies_with_pose, n_focus, n_control):
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
    out = []
    for grp, k in ((FOCUS_TYPES, n_focus), (CONTROL_TYPES, n_control)):
        for st in grp:
            pool = by_type.get(st, [])
            step = max(1, len(pool) // max(1, k))
            out += pool[::step][:k]
    return out


def extract_sequence(video, t_center, cx, cy, sx, sy, tmp_dir, tag):
    """A short frame sequence, cropped once (static box - the window
    is short enough that a swinging arm should not leave a generous
    260px native-res crop), at FPS_OUT. Returns [(t, path)]."""
    x = int(cx * sx - CROP / 2)
    y = int(cy * sy - CROP / 2)
    vf = f"crop={CROP}:{CROP}:{max(0, x)}:{max(0, y)},fps={FPS_OUT}"
    t0 = max(0.0, t_center - WINDOW)
    dur = 2 * WINDOW
    pattern = str(tmp_dir / f"{tag}_%03d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{t0:.3f}", "-i", video, "-t", f"{dur:.3f}",
         "-vf", vf, "-q:v", "3", pattern],
        capture_output=True, check=True)
    frames = sorted(tmp_dir.glob(f"{tag}_*.jpg"))
    return [(t0 + i / FPS_OUT, p) for i, p in enumerate(frames)]


def load_detector():
    try:
        import torch
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
    except ImportError:
        raise SystemExit(
            "needs a zero-shot detector: pip install torch transformers")
    proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained(
        "google/owlvit-base-patch32")
    model.eval()
    return proc, model, torch


def detect_paddle(proc, model, torch, frames):
    """[(t, cx, cy, score)] - one best box per frame, or the frame is
    dropped if nothing scores above floor. Queries hedge across the
    couple of phrasings a general-purpose vision-language model is
    likelier to have seen paddles described as."""
    from PIL import Image
    queries = ["a pickleball paddle", "a paddle", "a racket"]
    out = []
    for t, path in frames:
        img = Image.open(path).convert("RGB")
        inputs = proc(text=[queries], images=img, return_tensors="pt")
        with torch.no_grad():
            res = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]])
        det = proc.post_process_object_detection(
            res, threshold=0.03, target_sizes=target_sizes)[0]
        if not len(det["scores"]):
            continue
        i = int(det["scores"].argmax())
        x0, y0, x1, y1 = [float(v) for v in det["boxes"][i]]
        out.append((t, (x0 + x1) / 2, (y0 + y1) / 2, float(det["scores"][i])))
    return out


def paddle_peak(traj):
    """Frame-to-frame paddle-centroid speed, peak instant. Same
    'peak speed = contact' heuristic as the wrist channel, applied to
    a different tracked point, so the comparison is apples-to-apples -
    not a different, more forgiving metric for the new instrument."""
    if len(traj) < 2:
        return None
    best_t, best_v = None, -1
    for (t0, x0, y0, _s0), (t1, x1, y1, _s1) in zip(traj, traj[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        v = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt
        if v > best_v:
            best_v, best_t = v, (t0 + t1) / 2
    return best_t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--pose-dir", default="pose_rtm")
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--review-html", required=True,
                    help="an already-generated touch_attribute.py "
                         "--review-html file. Needed to look up the "
                         "LABELED HITTER'S OWN track at the labeled "
                         "contact time, so the paddle search gets an "
                         "independent shot at the real event instead "
                         "of inheriting whatever track/time wrist-speed "
                         "already (possibly wrongly) picked.")
    ap.add_argument("--n", type=int, default=24,
                    help="total contacts, split across FOCUS_TYPES "
                         "(3/4) and CONTROL_TYPES (1/4)")
    ap.add_argument("--tmp-dir", default="paddle_timing_tmp")
    ap.add_argument("--out", default="paddle_timing_report.txt")
    a = ap.parse_args()
    print(f"paddle_timing_probe build: {BUILD}")

    for exe in ("ffmpeg", "ffprobe"):
        if subprocess.run(["which", exe], capture_output=True).returncode:
            raise SystemExit(f"{exe} not found")
    proc, model, torch = load_detector()

    labels_by_rally = defaultdict(list)
    for r in csv.DictReader(open(a.labels)):
        labels_by_rally[int(r["rally_cum"])].append(r)
    rallies_with_pose = {int(p.stem[1:]) for p in
                         Path(a.pose_dir).glob("r*.npz")}
    n_focus = int(a.n * 0.75)
    samples = sample_focus(labels_by_rally, rallies_with_pose,
                           n_focus // len(FOCUS_TYPES) + 1,
                           (a.n - n_focus) // len(CONTROL_TYPES) + 1)
    print(f"sampled {len(samples)} contacts")

    vw, vh = PP.ffprobe_dims(a.video)
    sx, sy = vw / 1280.0, vh / 720.0
    tmp_dir = Path(a.tmp_dir)
    tmp_dir.mkdir(exist_ok=True)

    track_names = PP.load_track_names(a.review_html)
    print(f"loaded track names for {len(track_names)} rallies from "
          f"{a.review_html}")

    rd_cache, rows = {}, []
    for k, (cum, t_true, hitter, shot_type) in enumerate(samples):
        if cum not in rd_cache:
            rd_cache[cum] = SE.load_rally(a.pose_dir, cum)
        rd = rd_cache[cum]
        if rd is None:
            continue
        tids = TA.player_tracks(rd)
        if len(tids) != 4:
            continue

        # baseline: the ORIGINAL, unconstrained wrist-speed estimate -
        # whichever track has peak arm speed near t_true, same method
        # already used to measure the session's standing wrist numbers.
        # Left untouched: this is what paddle position is being compared
        # AGAINST, not what it should search around.
        wp = wrist_peak(rd, t_true, tids)
        wrist_err = None if wp is None else wp[1] - t_true

        # paddle search: anchored on the LABELED HITTER's own track, at
        # the LABELED true contact time - independent of whatever track/
        # time wrist-speed guessed, so a wrist miss on a soft shot can't
        # drag the paddle search down with it.
        name_to_tid = {nm: tid for tid, nm in
                      track_names.get(cum, {}).items()}
        hit_tid = name_to_tid.get(hitter)
        if hit_tid is None or hit_tid not in tids:
            continue
        kp = PP.nearest_kpt(rd, hit_tid, t_true)
        if kp is None:
            continue
        cx, cy, _conf = kp
        frames = extract_sequence(a.video, t_true, cx, cy, sx, sy,
                                  tmp_dir, f"r{cum}_{k}")
        traj = detect_paddle(proc, model, torch, frames)
        t_paddle = paddle_peak(traj)
        paddle_err = None if t_paddle is None else t_paddle - t_true
        rows.append((cum, t_true, hitter, shot_type, wrist_err, paddle_err,
                    len(traj), len(frames)))
        we = f"{wrist_err:+.3f}s" if wrist_err is not None else "n/a"
        pe = f"{paddle_err:+.3f}s" if paddle_err is not None else "n/a"
        print(f"  [{k + 1}/{len(samples)}] r{cum} {hitter} ({shot_type}): "
              f"wrist {we}  paddle {pe}  "
              f"({len(traj)}/{len(frames)} frames detected)")

    _report(rows, a.out)


def _report(rows, out_path):
    lines = ["paddle vs wrist contact-timing comparison", ""]
    have_both = [r for r in rows if r[4] is not None and r[5] is not None]
    lines.append(f"{len(rows)} contacts attempted, {len(have_both)} had a "
                 "paddle detection to compare")

    def summarize(errs, label):
        if not errs:
            return f"  {label}: n=0"
        absd = sorted(abs(e) for e in errs)
        n = len(absd)
        return (f"  {label}: n={n}  median={absd[n // 2]:.3f}s  "
               f"p90={absd[int(0.9 * n)]:.3f}s  "
               f"within0.35s={sum(1 for x in absd if x <= 0.35)}/{n}")

    lines.append("")
    lines.append("OVERALL (only contacts where both signals exist):")
    lines.append(summarize([r[4] for r in have_both], "wrist"))
    lines.append(summarize([r[5] for r in have_both], "paddle"))
    wins = sum(1 for r in have_both if abs(r[5]) < abs(r[4]))
    lines.append(f"  paddle beats wrist on {wins}/{len(have_both)} "
                 f"individual contacts")

    by_type = defaultdict(list)
    for r in have_both:
        by_type[r[3]].append(r)
    lines.append("\nBY SHOT TYPE:")
    for st, rs in sorted(by_type.items()):
        lines.append(f"  {st} (n={len(rs)}):")
        lines.append("  " + summarize([r[4] for r in rs], "wrist"))
        lines.append("  " + summarize([r[5] for r in rs], "paddle"))

    missing = len(rows) - len(have_both)
    if missing:
        lines.append(f"\n{missing} contacts had NO paddle detection at "
                     "all in the sequence - coverage gap, not a timing "
                     "error; report it, don't fold it into the median")

    text = "\n".join(lines)
    print("\n" + text)
    Path(out_path).write_text(text)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
