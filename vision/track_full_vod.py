"""Full-VOD ball-candidate extractor — streams the whole broadcast with
BOUNDED memory and emits one row per detected yellow blob, every frame.
Track-building, contact extraction and log alignment happen AFTER, off this
CSV — the same split that worked for audio_contacts.py -> poc_report.py.

    python vision/track_full_vod.py <video.mp4> [--out FILE] [--stride N]

MEMORY. The 60s probe (vision/probe_clip_ball.py) collects a background
sample into a Python list every 30 frames — fine for 1800 frames, but at 80
min / 720p that scales to ~13 GB and would not run on a laptop. This keeps a
small ROLLING buffer of the last `--bg-buffer` sampled frames and
recomputes the median background from it periodically, so memory stays
bounded (~100 MB) however long the video is. As a side effect this also
adapts across camera cuts, which the matchup VOD is known to contain (Design
note: right after a cut the buffer briefly holds stale-scene frames, so
expect a few seconds of noisy candidates there — flag, don't hand-tune away).

DETECTOR is unchanged from the validated probe: "yellowness" =
min(R,G)-B, gated by change against the rolling background. Area bounds
auto-scale with resolution (validated at ~8px median area on a 640x360
clip; scales ~4x per doubling of linear resolution).

UNLIKE THE PROBE, this keeps EVERY candidate, not just fast (>80 px/s)
ones. The probe's fast-only filter was for that script's OWN validation
(prove ball motion tracks against rally windows); the actual question here
— does the inter-contact interval separate dinks from speed-ups — needs
slow shots too, since dinks are slow by design.

--stride N processes every Nth frame (default 1). Useful for a fast sanity
run before committing the full pass. Output is flushed continuously, so a
Ctrl-C leaves a valid partial CSV, not a corrupt one.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
FPS = 30.0          # MLP broadcasts are 30fps (confirmed on the probe clip)
REF_W, REF_H = 640, 360      # the resolution the detector was validated at


def probe_video(path):
    p = subprocess.run([FF, "-i", str(path)], capture_output=True, text=True)
    err = p.stderr
    m = re.search(r"(\d{2,5})x(\d{2,5})", err)
    if not m:
        sys.exit(f"could not read video size from ffmpeg output:\n{err[-800:]}")
    w, h = int(m.group(1)), int(m.group(2))
    d = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    dur = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3)) if d else None
    return w, h, dur


def frames(path, w, h, stride=1):
    cmd = [FF, "-v", "error", "-i", str(path)]
    if stride > 1:
        cmd += ["-vf", f"select=not(mod(n\\,{stride}))", "-vsync", "0"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 8)
    n = w * h * 3
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()


def yellowness(f):
    f = f.astype(np.int16)
    return np.clip(np.minimum(f[..., 0], f[..., 1]) - f[..., 2], 0, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="data/vision/full_candidates.csv")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame; 1 = every frame")
    ap.add_argument("--bg-buffer", type=int, default=24,
                    help="rolling background sample count (bounds memory)")
    ap.add_argument("--bg-sample-every-s", type=float, default=1.0)
    ap.add_argument("--bg-recompute-every-s", type=float, default=2.0)
    ap.add_argument("--min-area", type=int, default=1)
    ap.add_argument("--max-area", type=int, default=None,
                    help="default: auto-scaled from the 640x360 validation "
                         "(40px there)")
    ap.add_argument("--thresh", type=int, default=45)
    args = ap.parse_args()

    w, h, dur = probe_video(args.video)
    scale = (w * h) / (REF_W * REF_H)
    max_area = args.max_area if args.max_area is not None else max(40, int(40 * scale))
    print(f"video: {w}x{h} (scale {scale:.2f}x vs validation clip), "
          + (f"duration {dur/60:.1f} min" if dur else "duration unknown"))
    print(f"area window: [{args.min_area}, {max_area}] px  "
          f"(validated median was 8px at 640x360)")

    bg_every = max(1, round(args.bg_sample_every_s * FPS / args.stride))
    bg_recompute = max(1, round(args.bg_recompute_every_s * FPS / args.stride))
    dens_bin_frames = max(1, round(10.0 * FPS / args.stride))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dens_path = out.with_name(out.stem + "_density.csv")

    bg_buf, bg, bg_yellow = [], None, None
    n_written, bin_count, bin_start_t = 0, 0, 0.0
    t0 = time.time()

    with out.open("w", newline="") as fh, dens_path.open("w", newline="") as fh2:
        w_cand = csv.writer(fh)
        w_cand.writerow(["frame", "t_s", "x", "y", "area", "strength"])
        w_dens = csv.writer(fh2)
        w_dens.writerow(["t_start_s", "n_candidates"])

        for i, f in enumerate(frames(args.video, w, h, args.stride)):
            real_frame = i * args.stride
            t_s = real_frame / FPS

            if i % bg_every == 0:
                bg_buf.append(f.copy())
                if len(bg_buf) > args.bg_buffer:
                    bg_buf.pop(0)
            if (bg is None or i % bg_recompute == 0) and bg_buf:
                bg = np.median(np.stack(bg_buf), axis=0)
                bg_yellow = yellowness(bg).astype(np.int16)
            if bg is None:
                continue

            dy = yellowness(f).astype(np.int16) - bg_yellow
            mask = dy > args.thresh
            n_this_frame = 0
            if mask.any():
                lab, n = ndimage.label(mask)
                if n:
                    areas = ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1))
                    for j, a in enumerate(areas, 1):
                        if args.min_area <= a <= max_area:
                            ys, xs = np.nonzero(lab == j)
                            stren = float(dy[ys, xs].mean())
                            w_cand.writerow([real_frame, f"{t_s:.3f}",
                                             f"{xs.mean():.1f}", f"{ys.mean():.1f}",
                                             int(a), f"{stren:.1f}"])
                            n_written += 1
                            n_this_frame += 1

            bin_count += n_this_frame
            if (i + 1) % dens_bin_frames == 0:
                w_dens.writerow([f"{bin_start_t:.1f}", bin_count])
                bin_count = 0
                bin_start_t = t_s

            if (i + 1) % max(1, round(60 * FPS / args.stride)) == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-9)
                remain = ((dur * FPS / args.stride - (i + 1)) / rate
                         if dur and rate > 0 else None)
                print(f"  {t_s/60:6.1f} min of video, {n_written} candidates, "
                      f"{elapsed/60:.1f} min elapsed" +
                      (f", ~{remain/60:.0f} min left" if remain else ""),
                      flush=True)
                fh.flush()
                fh2.flush()

    print(f"\nwrote {n_written} candidates -> {out}")
    print(f"wrote density bins -> {dens_path}")
    print("\nSend both files back. Next: track-building, contact extraction "
          "and alignment to the referee\nlog happen off these — same as the "
          "audio pass, that part is iteration, not another video run.")


if __name__ == "__main__":
    main()
