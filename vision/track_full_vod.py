"""Full-VOD extractor — streams the broadcast ONCE with bounded memory and
harvests everything cheap enough to grab during the decode:

  <out>.csv               ball candidates (frame, t, x, y, area, strength)
  <out>_density.csv       candidates per 10 s (run sanity check)
  <out>_motion.csv        per-frame motion level -> cuts, replays, edit anatomy
  <out>_scorebug.csv      per-frame scorebug-strip change -> every score flip,
                          frame-exact (validated: the 5->6 flip hit one frame)
  <out>_scorebug_crops/   1/s JPEG crops of the scorebug -> read WHAT the
                          score is offline; flips + identities = the full
                          rally->video-timestamp index for the match
  <out>_players.csv       big moving blobs (crude player positions) ->
                          movement heatmaps, distance covered, kitchen
                          arrival, and the seed of hitter attribution
  <out>_loudness.csv      audio RMS per 0.25 s (full band + 2-8 kHz cheer
                          band) -> crowd roar vs the leverage scale

    python vision/track_full_vod.py <video.mp4> [--out FILE] [--stride N]
                                    [--no-players] [--no-scorebug]

Track-building, contact extraction and log alignment happen AFTER, off
these files — the same split that worked for audio_contacts -> poc_report.

MEMORY. The 60 s probe collects background frames into a list — ~13 GB at
80 min/720p. This keeps a small rolling buffer instead (bounded whatever
the video length), which also self-corrects across the camera cuts the
matchup VOD is known to contain; expect a few seconds of noisy candidates
right after each cut.

DETECTOR unchanged from the validated probe (yellowness = min(R,G)-B gated
by change vs the background; area bounds auto-scale with resolution).
Keeps EVERY candidate, not just fast ones — dinks are slow by design.

Output is flushed continuously: Ctrl-C leaves valid partial files.
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

try:
    from scipy import ndimage
except ImportError:
    sys.exit("scipy is required:  python3 -m pip install scipy")

import imageio_ffmpeg

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

FF = imageio_ffmpeg.get_ffmpeg_exe()
DEFAULT_FPS = 30.0  # fallback only — the real rate is PARSED from the file.
                    # (The probe clip was 30fps mp4, but YouTube's 720p
                    # AV1/VP9 streams are often 60fps; assuming 30 would
                    # silently halve every timestamp.)
REF_W, REF_H = 640, 360      # resolution the detector was validated at


def probe_video(path):
    p = subprocess.run([FF, "-i", str(path)], capture_output=True, text=True)
    err = p.stderr
    m = re.search(r"(\d{2,5})x(\d{2,5})", err)
    if not m:
        sys.exit(f"could not read video size from ffmpeg output:\n{err[-800:]}")
    w, h = int(m.group(1)), int(m.group(2))
    d = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    dur = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3)) if d else None
    f = re.search(r"([\d.]+) fps", err)
    fps = float(f.group(1)) if f else None
    return w, h, dur, fps


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


def loudness_pass(path, out_path):
    """Audio RMS per 0.25 s, full band + a 2-8 kHz 'cheer' band. Runs before
    the video loop; ~a minute for a full match."""
    sr, hop = 22050, int(22050 * 0.25)
    cmd = [FF, "-v", "error", "-i", str(path), "-f", "s16le",
           "-ac", "1", "-ar", str(sr), "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=sr * 4)
    # simple 2-8k bandpass via FFT per block
    rows, t, buf = [], 0.0, b""
    blk = hop * 2
    while True:
        need = blk * 2 - len(buf)
        chunk = p.stdout.read(max(need, 0)) if need > 0 else b""
        buf += chunk
        if len(buf) < blk * 2:
            break
        x = np.frombuffer(buf[:blk * 2], "<i2").astype(np.float32) / 32768.0
        buf = buf[hop * 2:]
        seg = x[:blk]
        spec = np.fft.rfft(seg * np.hanning(len(seg)))
        fr = np.fft.rfftfreq(len(seg), 1 / sr)
        band = (fr >= 2000) & (fr <= 8000)
        rows.append((t, float(np.sqrt(np.mean(seg ** 2))),
                     float(np.sqrt(np.mean(np.abs(spec[band]) ** 2)) / len(seg))))
        t += 0.25
    p.stdout.close()
    p.wait()
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "rms", "cheer_2_8k"])
        for r in rows:
            w.writerow([f"{r[0]:.2f}", f"{r[1]:.5f}", f"{r[2]:.7f}"])
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="data/vision/full_candidates.csv")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--bg-buffer", type=int, default=24)
    ap.add_argument("--bg-sample-every-s", type=float, default=1.0)
    ap.add_argument("--bg-recompute-every-s", type=float, default=2.0)
    ap.add_argument("--min-area", type=int, default=1)
    ap.add_argument("--max-area", type=int, default=None)
    ap.add_argument("--thresh", type=int, default=45)
    ap.add_argument("--players", action=argparse.BooleanOptionalAction,
                    default=True, help="record big moving blobs (crude "
                    "player positions); ~+30-40%% runtime")
    ap.add_argument("--scorebug", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--crops-every-s", type=float, default=1.0)
    ap.add_argument("--loudness", action=argparse.BooleanOptionalAction,
                    default=True)
    args = ap.parse_args()

    w, h, dur, fps = probe_video(args.video)
    if fps is None:
        print(f"WARNING: could not parse fps; assuming {DEFAULT_FPS}")
        fps = DEFAULT_FPS
    scale = (w * h) / (REF_W * REF_H)
    max_area = args.max_area if args.max_area is not None else max(40, int(40 * scale))
    print(f"video: {w}x{h} @ {fps:g} fps (scale {scale:.2f}x vs validation "
          f"clip), " + (f"duration {dur/60:.1f} min" if dur else "duration unknown"))
    print(f"ball area window: [{args.min_area}, {max_area}] px")
    if fps > 40 and args.stride == 1:
        print(f"NOTE: {fps:g} fps source — twice the frames of the validated "
              f"30fps regime.\n      --stride 2 samples at {fps/2:g} fps "
              f"(the validated rate) and halves detection cost.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sib = lambda s: out.with_name(out.stem + s)

    if args.loudness:
        print("audio loudness pass...", flush=True)
        n = loudness_pass(args.video, sib("_loudness.csv"))
        print(f"  {n} loudness rows -> {sib('_loudness.csv').name}")

    # player-blob area bounds (full-res px), auto-scaled: a 720p player is
    # roughly 40x90 px; allow kneeling to jumping
    pmin, pmax = int(500 * scale), int(40000 * scale)
    crops_dir = sib("_scorebug_crops")
    if args.scorebug and HAVE_PIL:
        crops_dir.mkdir(exist_ok=True)
    elif args.scorebug and not HAVE_PIL:
        print("NOTE: pillow not installed -> scorebug flip CSV still written, "
              "but no crop images.\n      python3 -m pip install pillow")

    bg_every = max(1, round(args.bg_sample_every_s * fps / args.stride))
    bg_recompute = max(1, round(args.bg_recompute_every_s * fps / args.stride))
    dens_bin = max(1, round(10.0 * fps / args.stride))
    crop_every = max(1, round(args.crops_every_s * fps / args.stride))
    sb_h, sb_w = int(0.17 * h), int(0.42 * w)

    bg_buf, bg_yellow, bg_gray = [], None, None
    prev_gray, prev_strip = None, None
    n_cand, bin_count, bin_t = 0, 0, 0.0
    t0 = time.time()

    fh = out.open("w", newline="")
    fh_d = sib("_density.csv").open("w", newline="")
    fh_m = sib("_motion.csv").open("w", newline="")
    fh_s = sib("_scorebug.csv").open("w", newline="") if args.scorebug else None
    fh_p = sib("_players.csv").open("w", newline="") if args.players else None
    w_c = csv.writer(fh)
    w_c.writerow(["frame", "t_s", "x", "y", "area", "strength"])
    w_d = csv.writer(fh_d)
    w_d.writerow(["t_start_s", "n_candidates"])
    w_m = csv.writer(fh_m)
    w_m.writerow(["frame", "t_s", "motion"])
    if fh_s:
        w_s = csv.writer(fh_s)
        w_s.writerow(["frame", "t_s", "strip_mean", "strip_diff"])
    if fh_p:
        w_p = csv.writer(fh_p)
        w_p.writerow(["frame", "t_s", "x", "y", "bw", "bh", "area"])

    for i, f in enumerate(frames(args.video, w, h, args.stride)):
        rf = i * args.stride
        t_s = rf / fps
        gray = f[::2, ::2].mean(axis=2, dtype=np.float32)

        # motion curve (cuts show as big spikes on a static camera)
        w_m.writerow([rf, f"{t_s:.3f}",
                      f"{0.0 if prev_gray is None else float(np.abs(gray - prev_gray).mean()):.3f}"])
        prev_gray = gray

        # rolling background
        if i % bg_every == 0:
            bg_buf.append(f.copy())
            if len(bg_buf) > args.bg_buffer:
                bg_buf.pop(0)
        if (bg_yellow is None or i % bg_recompute == 0) and bg_buf:
            bg = np.median(np.stack(bg_buf), axis=0)
            bg_yellow = yellowness(bg).astype(np.int16)
            bg_gray = bg[::2, ::2].mean(axis=2)
        if bg_yellow is None:
            continue

        # ---- ball candidates ------------------------------------------
        dy = yellowness(f).astype(np.int16) - bg_yellow
        mask = dy > args.thresh
        nf = 0
        if mask.any():
            lab, n = ndimage.label(mask)
            if n:
                areas = ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1))
                for j, a in enumerate(areas, 1):
                    if args.min_area <= a <= max_area:
                        ys, xs = np.nonzero(lab == j)
                        w_c.writerow([rf, f"{t_s:.3f}", f"{xs.mean():.1f}",
                                      f"{ys.mean():.1f}", int(a),
                                      f"{float(dy[ys, xs].mean()):.1f}"])
                        n_cand += 1
                        nf += 1
        bin_count += nf

        # ---- player blobs (half-res) ----------------------------------
        if fh_p is not None:
            pm = np.abs(gray - bg_gray) > 25
            lab, n = ndimage.label(pm)
            if n:
                areas = ndimage.sum_labels(np.ones_like(lab), lab, range(1, n + 1))
                order = np.argsort(areas)[::-1][:8]
                sl = ndimage.find_objects(lab)
                for j in order:
                    a4 = areas[j] * 4          # back to full-res px
                    if pmin <= a4 <= pmax:
                        sy, sx = sl[j]
                        w_p.writerow([rf, f"{t_s:.3f}",
                                      (sx.start + sx.stop),      # *2/2 center
                                      (sy.start + sy.stop),
                                      2 * (sx.stop - sx.start),
                                      2 * (sy.stop - sy.start), int(a4)])

        # ---- scorebug strip -------------------------------------------
        if fh_s is not None:
            strip = f[:sb_h, :sb_w]
            sm = float(strip.mean())
            sd_ = 0.0 if prev_strip is None else \
                float(np.abs(strip.astype(np.int16) - prev_strip).mean())
            prev_strip = strip.astype(np.int16)
            w_s.writerow([rf, f"{t_s:.3f}", f"{sm:.2f}", f"{sd_:.3f}"])
            if HAVE_PIL and i % crop_every == 0:
                Image.fromarray(strip).save(
                    crops_dir / f"t{t_s:08.2f}.jpg", quality=70)

        if (i + 1) % dens_bin == 0:
            w_d.writerow([f"{bin_t:.1f}", bin_count])
            bin_count, bin_t = 0, t_s

        if (i + 1) % max(1, round(60 * fps / args.stride)) == 0:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1e-9)
            rem = ((dur * fps / args.stride - (i + 1)) / rate
                   if dur and rate > 0 else None)
            print(f"  {t_s/60:6.1f} min of video, {n_cand} ball candidates, "
                  f"{el/60:.1f} min elapsed"
                  + (f", ~{rem/60:.0f} min left" if rem else ""), flush=True)
            for x in (fh, fh_d, fh_m, fh_s, fh_p):
                if x:
                    x.flush()

    for x in (fh, fh_d, fh_m, fh_s, fh_p):
        if x:
            x.close()
    print(f"\nwrote {n_cand} ball candidates -> {out}")
    print("plus: _density, _motion" + (", _scorebug + crops/" if args.scorebug else "")
          + (", _players" if args.players else "")
          + (", _loudness" if args.loudness else ""))
    print(f"\nto send everything back in one file:\n"
          f"  zip -r vod_outputs.zip {out.stem}*")


if __name__ == "__main__":
    main()
