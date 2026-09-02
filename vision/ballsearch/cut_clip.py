"""cut_clip — cut a rally clip from the full-match source the way the
staged clips were cut, with the offset and frame count taken from the
COMMITTED candidate CSV so the cut cannot drift from what was graded
(2026-09-01, for the r10 re-cut; pair with check_clip.py).

  python3 cut_clip.py <rally> <full_match source> [out.mp4]
  python3 cut_clip.py --stage <rally> <full_match source> [out.mp4]

--stage (2026-09-02, click-package staging for rallies that have NO
committed candidate CSV yet): the window comes from the owner's manual
contact taps instead — offset = first contact - 1.5 s (rounded down to
0.1 s), frames through last contact + 2.5 s — the same span the ball
audit tool frames (PRE_S 1.0 / dead = last + 2.0) with a margin. The
printed offset is what ball_candidates.py --offset takes; once that CSV
is committed the default mode reproduces the cut exactly.

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


def stage_spec(rally):
    """(offset, frames) from the manual contact taps (never prefill)."""
    import sys as _s
    _s.path.insert(0, "/home/user/pickleball/vision")
    from make_ball_audit import load_impacts
    imps, _dead = load_impacts(rally=rally)
    off = int((imps[0] - 1.5) * 10) / 10.0
    n = int(round((imps[-1] + 2.5 - off) * 60))
    return off, n


def ffmpeg_bin():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    args = sys.argv[1:]
    staging = args and args[0] == "--stage"
    if staging:
        args = args[1:]
    rally, src = int(args[0]), args[1]
    out = args[2] if len(args) > 2 else f"r{rally}_clip.mp4"
    off, n = stage_spec(rally) if staging else spec(rally)
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
