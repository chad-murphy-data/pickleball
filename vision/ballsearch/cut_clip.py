"""cut_clip — cut a rally clip from the full-match source the way the
staged clips were cut, with the offset and frame count taken from the
COMMITTED candidate CSV so the cut cannot drift from what was graded
(2026-09-01, for the r10 re-cut; pair with check_clip.py).

  python3 cut_clip.py <rally> <full_match source> [out.mp4]

offset = t_s(frame k) - k/60 from data/vision/ball_candidates_r{N}.csv.gz,
frames = max(frame) + 2 (the extractor emits nothing at frame 0 and the
last frame). Output: 1280x720, 60 fps, no audio, libx264 veryfast crf 18
(the staged clips are x264 fast-preset encodes at that size). Uses the
system ffmpeg if present, else the static binary from imageio-ffmpeg.
Then run  python3 check_clip.py <rally> <out.mp4>  before any cache.
"""
import csv
import gzip
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

DATA = Path("/home/user/pickleball/data/vision")


def spec(rally):
    with gzip.open(DATA / f"ball_candidates_r{rally}.csv.gz", "rt") as f:
        rows = [(int(r["frame"]), float(r["t_s"])) for r in csv.DictReader(f)]
    off = float(np.median([t - k / 60.0 for k, t in rows]))
    return round(off, 2), max(k for k, _ in rows) + 2


def ffmpeg_bin():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    rally, src = int(sys.argv[1]), sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else f"r{rally}_clip.mp4"
    off, n = spec(rally)
    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-ss", f"{off:.2f}",
           "-i", src, "-frames:v", str(n), "-r", "60",
           "-vf", "scale=1280:720", "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "18", out]
    print(f"rally {rally}: offset {off:.2f} s, {n} frames -> {out}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print("done; now: python3 check_clip.py", rally, out)


if __name__ == "__main__":
    main()
