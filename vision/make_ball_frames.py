"""Pick and extract the frames worth hand-labeling for Gate A (the ball).

Gate A asks for ~300-500 frames stratified BY SHOT SPEED, because the
ball is trivial to see on a dink and nearly invisible on a drive, and the
gate is read on the FAST stratum.  Clicking whatever the video happens to
show next spends most clicks on easy frames; this picks them instead.

Two strata, each earning its budget:

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

DROPPED 2026-08-15 — a dense `calib` stratum (every frame within +/-0.15 s
of each hand-marked serve) existed to measure the labeler's click timing
for the SWING channel.  That channel was killed the same day, so the
stratum was vestigial, and the user caught what it was doing to the
labeling: nine consecutive frames of a player dropping the ball to serve
is nine clicks on a near-stationary ball in one court position.  Nine
near-duplicates teach what one teaches, and worse, they were a third of
the click budget, which would have taught a detector that balls are slow
and live near the baseline — precisely inverted from the fast stratum the
gate is read on.  Same shape as the auto-label poisoning (42% kitchen-band
vs 14% base), reached by a different route.  One tool, one purpose: if
labeler timing ever needs measuring again, that is the shot-click tool's
job, not this one's.

Two guards keep near-duplicates out for good: no two samples from the same
rally land within MIN_SEP_S, and sampling never starts before the serve
(exact where a mark exists, PAD_HEAD otherwise) because the head of a
window is the pre-serve ball-drop, where the ball barely moves.

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
PAD_HEAD = 1.2           # window head is pre-serve ball-drop: a static ball
PAD_TAIL = 0.6           # tail is the point already over
SERVE_FLOOR = 0.10       # where a serve mark exists, start just after contact
MIN_SEP_S = 0.5          # no two samples this close within one rally


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
        if t1 - t0 < PAD_HEAD + PAD_TAIL + MIN_SEP_S:
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


def live_span(r, marks):
    """The stretch of a window where the ball is actually in play. The head
    of a window is the server bouncing/holding the ball before contact —
    a stationary ball, and worthless as training signal. Where the rally
    has a hand-marked serve we know that boundary exactly."""
    lo = r["t0"] + PAD_HEAD
    sv = marks.get(r["cum"])
    if sv is not None:
        lo = max(lo, sv + SERVE_FLOOR)
    return lo, r["t1"] - PAD_TAIL


def plan(rallies, marks, shots, n_fast, n_random, seed):
    """-> list of (stratum, rally_cum, t_video).

    Each stratum draws from its OWN rng stream, so changing one stratum's
    count leaves the other's picks byte-identical — a set can be topped up
    later without invalidating labels already collected."""
    spans = {}
    for r in rallies:
        lo, hi = live_span(r, marks)
        if hi - lo > MIN_SEP_S:
            spans[r["cum"]] = (lo, hi, r)
    picks = []

    # --- fast: division weight x rally tempo (shots/sec where known)
    pool = []
    for cum, (lo, hi, r) in spans.items():
        w = DIV_WEIGHT.get(r["div"], 1.0)
        n = shots.get(cum)
        if n:
            tempo = n / max(r["t1"] - r["t0"], 1e-6)
            w *= max(0.4, min(2.5, tempo / 0.5))     # 0.5 shots/s ~ average
        pool.append((w, cum))
    tot = sum(w for w, _ in pool)
    rf = random.Random(seed ^ 0xFA57)
    for _ in range(n_fast):
        x, acc = rf.random() * tot, 0.0
        for w, cum in pool:
            acc += w
            if acc >= x:
                lo, hi, _r = spans[cum]
                picks.append(("fast", cum, round(rf.uniform(lo, hi), 3)))
                break

    # --- random: uniform over rallies, uniform within
    rr = random.Random(seed ^ 0x8A5E)
    keys = sorted(spans)
    for _ in range(n_random):
        cum = rr.choice(keys)
        lo, hi, _r = spans[cum]
        picks.append(("random", cum, round(rr.uniform(lo, hi), 3)))

    # Thin to MIN_SEP_S WITHIN a rally. Two frames half a second apart in
    # the same point are near-duplicates: one click's worth of information
    # for two clicks of effort, and over-representing whatever the ball was
    # doing at that moment.
    picks.sort(key=lambda p: (p[1], p[2]))
    kept, last = [], {}
    for p in picks:
        prev = last.get(p[1])
        if prev is not None and p[2] - prev < MIN_SEP_S:
            continue
        last[p[1]] = p[2]
        kept.append(p)
    kept.sort(key=lambda p: p[2])
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--windows", default="rally_windows_chicago0725_v4.csv")
    ap.add_argument("--labels", default="shot_labels_chicago0725.csv")
    ap.add_argument("--out", default="ball_frames")
    ap.add_argument("--n-fast", type=int, default=300)
    ap.add_argument("--n-random", type=int, default=190)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--all-rallies", action="store_true",
                    help="include approx-flagged windows too")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rallies = load_rallies(a.windows, not a.all_rallies)
    marks, shots = load_marks(a.labels)
    if not rallies:
        raise SystemExit("no usable rally windows")
    picks = plan(rallies, marks, shots, a.n_fast, a.n_random, a.seed)

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
