"""Read the 1/s scorebug CROPS into a per-second score-change stream.

Why this exists (2026-08-14): the whole-strip diff statistic in
full_candidates_scorebug.csv mathematically cannot see a score flip — a
digit change moves a few hundred pixels out of ~65k, a mean diff of ~0.2
against a noise floor of ~0.9.  The ~150 "strong" events it does contain
are scene cuts and scorebug removals (verified by eye: the bug vanishes
into replay wipes right after score changes).  The CROPS folder, however,
is one readable scoreboard photo per second — the real instrument.

This script:
  1. streams the crops in time order,
  2. flags seconds where the scorebug is ABSENT (replay wipes, cuts) by
     correlating the left logo region against a running reference,
  3. auto-locates the two score-digit boxes as the right-side pixel
     regions that change most across the match (no hand-drawn boxes),
  4. emits per-second change energy inside each digit box, diffed against
     the last BUG-PRESENT second — so a flip hidden behind a replay still
     registers the moment the bug returns.

    python3 scorebug_read.py --crops full_candidates_scorebug_crops \\
                             --out scorebug_score_changes.csv

Output columns: t_s, bug_present, utah_change, chicago_change, either.
Send the CSV back for alignment: candidates are then nearly pure score
events WITH team attribution (which side's digit moved = who scored),
cross-checkable against the referee log's per-rally outcomes.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def load_gray(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def crop_times(crops_dir):
    out = []
    for p in Path(crops_dir).iterdir():
        m = re.match(r"t(\d+\.\d+)\.jpe?g$", p.name)
        if m:
            out.append((float(m.group(1)), p))
    out.sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True, type=Path)
    ap.add_argument("--out", default="scorebug_score_changes.csv")
    a = ap.parse_args()

    items = crop_times(a.crops)
    print(f"{len(items)} crops")
    if len(items) < 100:
        raise SystemExit("too few crops — wrong folder?")

    # ---- pass 0: reference frame + presence detection ------------------
    # sample frames spread across the match; the median is "bug present,
    # generic score" — presence = correlation of the LEFT 45% (logo/names,
    # score-independent) against the reference
    idx = np.linspace(0, len(items) - 1, 60).astype(int)
    sample = [load_gray(items[i][1]) for i in idx]
    h = min(s.shape[0] for s in sample)
    w = min(s.shape[1] for s in sample)
    sample = [s[:h, :w] for s in sample]
    ref = np.median(sample, axis=0)
    left = (slice(0, h), slice(0, int(0.45 * w)))

    def present(fr):
        x, y = fr[left].ravel(), ref[left].ravel()
        x = x - x.mean(); y = y - y.mean()
        n = float(np.sqrt((x * x).sum() * (y * y).sum()))
        return (float((x * y).sum() / n) if n > 0 else 0.0) > 0.55

    # ---- pass 1: change-accumulation map over PRESENT frames ----------
    acc = np.zeros((h, w), np.float32)
    prev = None
    pres_flags = []
    frames_cache = {}
    for k, (t, p) in enumerate(items):
        fr = load_gray(p)[:h, :w]
        pr = present(fr)
        pres_flags.append(pr)
        if pr:
            if prev is not None:
                acc += (np.abs(fr - prev) > 25)
            prev = fr
        if k % 600 == 0:
            print(f"  pass1 {k}/{len(items)}", flush=True)

    # digit boxes: the two hottest change clusters in the RIGHT 60% of
    # the strip (names/logo left of them barely change; digits flip
    # hundreds of times)
    right = acc.copy()
    right[:, :int(0.40 * w)] = 0
    thr = np.percentile(right[right > 0], 97) if (right > 0).any() else 1
    hot = right >= max(thr, 20)
    ys, xs = np.nonzero(hot)
    if len(ys) < 10:
        raise SystemExit("no hot digit pixels found — send me the acc stats")
    # split into top/bottom halves = UTAH / CHICAGO rows
    mid = h / 2
    boxes = {}
    for name, m in (("utah", ys < mid), ("chicago", ys >= mid)):
        if m.sum() < 5:
            boxes[name] = None
            continue
        y0, y1 = ys[m].min(), ys[m].max()
        x0, x1 = xs[m].min(), xs[m].max()
        pad = 3
        boxes[name] = (slice(max(0, y0 - pad), min(h, y1 + pad)),
                       slice(max(0, x0 - pad), min(w, x1 + pad)))
        print(f"  {name} digit box: rows {y0}-{y1}, cols {x0}-{x1}")

    # ---- pass 2: per-second digit-box change vs last PRESENT frame ----
    rows_out = []
    last_present = None
    for k, (t, p) in enumerate(items):
        fr = load_gray(p)[:h, :w]
        pr = pres_flags[k]
        u = c = 0.0
        if pr and last_present is not None:
            for name, box in boxes.items():
                if box is None:
                    continue
                d = float((np.abs(fr[box] - last_present[box]) > 30).mean())
                if name == "utah":
                    u = d
                else:
                    c = d
        if pr:
            last_present = fr
        rows_out.append((t, int(pr), u, c))
        if k % 600 == 0:
            print(f"  pass2 {k}/{len(items)}", flush=True)

    with open(a.out, "w", newline="") as fh:
        wcsv = csv.writer(fh)
        wcsv.writerow(["t_s", "bug_present", "utah_change", "chicago_change",
                       "either"])
        for t, pr, u, c in rows_out:
            wcsv.writerow([f"{t:.2f}", pr, f"{u:.4f}", f"{c:.4f}",
                           f"{max(u, c):.4f}"])

    ch = np.array([max(u, c) for _, pr, u, c in rows_out])
    npres = sum(pr for _, pr, _, _ in rows_out)
    print(f"\nwrote {a.out}")
    print(f"bug present {npres}/{len(rows_out)} seconds")
    for th in (0.05, 0.10, 0.20, 0.30):
        print(f"  seconds with digit-change > {th}: {(ch > th).sum()}"
              f"   (191 rally flips expected, plus game transitions)")


if __name__ == "__main__":
    main()
