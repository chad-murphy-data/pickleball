"""Pick and extract the frames worth hand-labeling for Gate A (the ball).

Gate A asks for ~300-500 frames stratified BY SHOT SPEED, because the
ball is trivial to see on a dink and nearly invisible on a drive, and the
gate is read on the FAST stratum.  Clicking whatever the video happens to
show next spends most clicks on easy frames; this picks them instead.

Three strata, each earning its budget:

  calib   +/- a few frames around each of the 16 hand-marked serve times.
          Dual purpose: they are clean ball-visible frames AND, with the
          'c' key in the flipbook, they measure the labeler's own timing
          jitter against their played-speed serve marks.  Unmeasured label
          precision confounds every later timing result.
  fast    weighted toward the divisions that hit hardest (mens > mixed >
          womens) and the highest-tempo rallies.  This is a PROXY: shot
          types are known only for rallies 1-16 and per-shot times for
          none, so true speed stratification can only be VERIFIED after
          labeling, by measuring inter-frame displacement of the labeled
          positions.  Do that before believing the stratum, and top up in
          a second round if the fast tail is thin.
  random  uniform over all rallies.  Deliberate ballast: without it the
          set only contains frames we already believe are interesting,
          and the detector inherits our belief.

Each sample extracts a TRIPLET (t-2, t-1, t frames) at native resolution:
TrackNet consumes three consecutive frames to exploit motion, and the
flipbook blink-compares t-1 against t so a blurred ball pops out against
a static background.  Only the last frame of each triplet is clicked.

NOTE: using a detector to CHOOSE frames for a human to label is active
learning and is fine.  Using a detector to GENERATE labels is what
poisoned the previous fine-tune (42% kitchen-band vs 14% base).  Nothing
here generates a label.

    python3 make_ball_frames.py --video full_match.mp4.webm \\
        --windows rally_windows_chicago0725_v4.csv \\
        --labels shot_labels_chicago0725.csv --out ball_frames

    python3 make_ball_frames.py ... --dry-run    # print the plan only
"""
from __future__ import annotations

import argparse
import csv
import random
import subprocess
from pathlib import Path

DIV_WEIGHT = {"mens": 3.0, "mixed": 2.0, "womens": 1.0}   # hardest hitting first
CALIB_SPAN_S = 0.15      # +/- around each serve mark; brackets ~5 frames each way
EDGE_PAD_S = 0.6         # keep samples away from window edges (dead time)


def ffmpeg_bin():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def load_rallies(windows_csv, confident_only):
    out = []
    for r in csv.DictReader(open(windows_csv)):
        if confident_only and r["approx"] != "0":
            continue
        t0, t1 = float(r["t0s"]), float(r["t1s"])
        if t1 - t0 < 2 * EDGE_PAD_S + 0.5:
            continue
        out.append({"cum": int(r["rally_cum"]), "div": r["division"],
                    "t0": t0, "t1": t1})
    return out


def load_marks(labels_csv):
    marks, shots = {}, {}
    for r in csv.DictReader(open(labels_csv)):
        cum = int(r["rally_cum"])
        shots[cum] = max(shots.get(cum, 0), int(r["shot_index"]))
        if r.get("serve_time_s"):
            marks[cum] = float(r["serve_time_s"])
    return marks, shots


def plan(rallies, marks, shots, n_fast, n_random, fps, rng):
    """-> list of (stratum, rally_cum, t_video)."""
    picks = []

    # --- calib: every frame within +/- CALIB_SPAN_S of each serve mark
    step = 1.0 / fps
    for cum, sv in sorted(marks.items()):
        k = int(CALIB_SPAN_S / step)
        for i in range(-k, k + 1):
            picks.append(("calib", cum, round(sv + i * step, 3)))

    # --- fast: division weight x rally tempo (shots/sec where known)
    pool = []
    for r in rallies:
        w = DIV_WEIGHT.get(r["div"], 1.0)
        n = shots.get(r["cum"])
        if n:
            tempo = n / max(r["t1"] - r["t0"], 1e-6)
            w *= max(0.4, min(2.5, tempo / 0.5))     # 0.5 shots/s ~ average
        pool.append((w, r))
    tot = sum(w for w, _ in pool)
    for _ in range(n_fast):
        x, acc = rng.random() * tot, 0.0
        for w, r in pool:
            acc += w
            if acc >= x:
                picks.append(("fast", r["cum"],
                              round(rng.uniform(r["t0"] + EDGE_PAD_S,
                                                r["t1"] - EDGE_PAD_S), 3)))
                break

    # --- random: uniform over rallies, uniform within
    for _ in range(n_random):
        r = rng.choice(rallies)
        picks.append(("random", r["cum"],
                      round(rng.uniform(r["t0"] + EDGE_PAD_S,
                                        r["t1"] - EDGE_PAD_S), 3)))

    # de-dupe near-identical times (within one frame) — a duplicate frame
    # costs a click and teaches nothing
    picks.sort(key=lambda p: (p[2], p[0]))
    kept = []
    for p in picks:
        if kept and abs(p[2] - kept[-1][2]) < step * 0.9:
            continue
        kept.append(p)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--windows", default="rally_windows_chicago0725_v4.csv")
    ap.add_argument("--labels", default="shot_labels_chicago0725.csv")
    ap.add_argument("--out", default="ball_frames")
    ap.add_argument("--n-fast", type=int, default=200)
    ap.add_argument("--n-random", type=int, default=120)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--all-rallies", action="store_true",
                    help="include approx-flagged windows too")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rallies = load_rallies(a.windows, not a.all_rallies)
    marks, shots = load_marks(a.labels)
    if not rallies:
        raise SystemExit("no usable rally windows")
    picks = plan(rallies, marks, shots, a.n_fast, a.n_random, a.fps, rng)

    from collections import Counter
    cs, cd = Counter(p[0] for p in picks), Counter()
    div = {r["cum"]: r["div"] for r in rallies}
    for _, cum, _ in picks:
        cd[div.get(cum, "?")] += 1
    print(f"{len(picks)} frames to label "
          f"({dict(cs)}), by division {dict(cd)}")
    print(f"triplets extracted at native resolution -> {a.out}/")
    if a.dry_run:
        for p in picks[:15]:
            print("   ", p)
        print("    ...")
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = open(out / "manifest.csv", "w", newline="")
    mw = csv.writer(man)
    mw.writerow(["idx", "stratum", "rally_cum", "t_video", "file_target",
                 "file_prev", "file_prev2"])
    fb, step = ffmpeg_bin(), 1.0 / a.fps
    made = 0
    for i, (strat, cum, t) in enumerate(picks):
        stem = f"f{i:04d}_r{cum}_{strat}_t{t:.3f}"
        if (out / f"{stem}_2.jpg").exists():
            made += 1
            continue
        cmd = [fb, "-v", "error", "-ss", f"{max(0, t - 2 * step):.3f}",
               "-i", str(a.video), "-frames:v", "3", "-vsync", "0",
               "-q:v", "2", str(out / f"{stem}_%d.jpg")]
        subprocess.run(cmd, capture_output=True)
        # ffmpeg numbers %d from 1; rename to 0,1,2 with 2 = the click target
        ok = True
        for src, dst in ((1, 0), (2, 1), (3, 2)):
            s, d = out / f"{stem}_{src}.jpg", out / f"{stem}_{dst}.jpg"
            if s.exists():
                s.replace(d)
            elif not d.exists():
                ok = False
        if not ok:
            continue
        mw.writerow([i, strat, cum, f"{t:.3f}", f"{stem}_2.jpg",
                     f"{stem}_1.jpg", f"{stem}_0.jpg"])
        made += 1
        if made % 50 == 0:
            print(f"  {made}/{len(picks)}", flush=True)
    man.close()
    print(f"\n{made} triplets in {out}/ (+ manifest.csv)")
    print("open the flipbook, point it at this folder, and click balls")


if __name__ == "__main__":
    main()
