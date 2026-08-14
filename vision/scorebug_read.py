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
    ap.add_argument("--debug-bundle", default="scorebug_debug",
                    help="folder for ref/variance/change maps + samples")
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

    # digit boxes.  Two discriminators, both needed (v1 grabbed the TV
    # background right of the panel, which flickers with every camera
    # change): (1) PANEL mask = temporally STABLE pixels (the bug panel
    # is a fixed graphic; the background behind it churns) — low variance
    # across the sampled frames; (2) digits = pixels changing RARELY but
    # repeatedly — a middle band of change counts, excluding both static
    # panel paint (near zero) and background bleed (thousands).
    var = np.var(np.stack(sample), axis=0)
    panel = var < np.percentile(var, 55)
    n_present_pairs = max(1, sum(pres_flags) - 1)
    lo_c, hi_c = 0.01 * n_present_pairs, 0.25 * n_present_pairs
    hot = panel & (acc >= lo_c) & (acc <= hi_c)
    hot[:, :int(0.30 * w)] = False           # logo/team-name zone
    print(f"strip {w}x{h}; panel pixels {panel.mean():.0%}; "
          f"hot digit pixels {int(hot.sum())} (band {lo_c:.0f}-{hi_c:.0f} changes)")
    ys, xs = np.nonzero(hot)
    if len(ys) < 10:
        raise SystemExit("no hot digit pixels found — send the debug bundle")
    # connected components of the hot mask = the panel's changing elements
    # (score digits, serve indicator, game-score chip) — each becomes its
    # own tracked state machine, per the user's design: read the state
    # every second and ask "same as t-1?"
    lab = np.zeros((h, w), np.int32)
    nlab = 0
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        nlab += 1
        stack = [(y0, x0)]
        lab[y0, x0] = nlab
        while stack:
            cy, cx = stack.pop()
            for dy in (-2, -1, 0, 1, 2):
                for dx in (-2, -1, 0, 1, 2):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and hot[ny, nx] \
                            and not lab[ny, nx]:
                        lab[ny, nx] = nlab
                        stack.append((ny, nx))
    boxes = {}
    for li in range(1, nlab + 1):
        m = lab == li
        if m.sum() < 6:
            continue
        yy, xx = np.nonzero(m)
        pad = 3
        boxes[f"c{li}"] = (slice(max(0, yy.min() - pad), min(h, yy.max() + pad)),
                           slice(max(0, xx.min() - pad), min(w, xx.max() + pad)))
    print(f"  {len(boxes)} tracked panel elements:")
    for name, b in boxes.items():
        print(f"    {name}: rows {b[0].start}-{b[0].stop}, "
              f"cols {b[1].start}-{b[1].stop}")

    # ---- pass 2: per-element STATE machines (the user's design) --------
    # each element's crop is matched against its dictionary of previously
    # seen appearances (normalized correlation); no match = a new state.
    # A transition counts only after the new state persists 2 seconds
    # (flash animations and jpeg shimmer do not persist).
    def norm(v):
        v = v.ravel().astype(np.float32)
        v = v - v.mean()
        n = float(np.sqrt((v * v).sum()))
        return v / n if n > 1e-6 else v

    states = {name: [] for name in boxes}      # list of state vectors
    cur = {name: -1 for name in boxes}
    pend = {name: (-1, 0) for name in boxes}   # candidate state, run length
    rows_out = []
    for k, (t, p) in enumerate(items):
        fr = load_gray(p)[:h, :w]
        pr = pres_flags[k]
        rec = {"t": t, "present": int(pr)}
        for name, box in boxes.items():
            changed = 0
            if pr:
                v = norm(fr[box])
                sid = None
                for si, sv in enumerate(states[name]):
                    if float(v @ sv) > 0.92:
                        sid = si
                        break
                if sid is None:
                    states[name].append(v)
                    sid = len(states[name]) - 1
                if sid != cur[name]:
                    csid, run = pend[name]
                    run = run + 1 if sid == csid else 1
                    pend[name] = (sid, run)
                    if run >= 2:
                        if cur[name] >= 0:
                            changed = 1
                        cur[name] = sid
                        pend[name] = (-1, 0)
                else:
                    pend[name] = (-1, 0)
            rec[f"{name}_state"] = cur[name]
            rec[f"{name}_chg"] = changed
            rec[f"{name}_x"] = boxes[name][1].start
            rec[f"{name}_y"] = boxes[name][0].start
        rows_out.append(rec)
        if k % 600 == 0:
            print(f"  pass2 {k}/{len(items)}", flush=True)

    # debug bundle: everything needed to verify the boxes by eye
    from PIL import Image
    bd = Path(a.debug_bundle)
    bd.mkdir(exist_ok=True)
    def save_img(arr, name):
        x = arr.astype(np.float32)
        x = (255 * (x - x.min()) / max(x.max() - x.min(), 1e-9)).astype(np.uint8)
        Image.fromarray(x).save(bd / name)
    save_img(ref, "ref.png")
    save_img(var, "variance.png")
    save_img(np.minimum(acc, np.percentile(acc, 99)), "changecount.png")
    save_img(hot.astype(np.float32), "digitmask.png")
    ov = ref.copy()
    for name, box in boxes.items():
        if box is not None:
            ov[box] = 255
    save_img(ov, "boxes_on_ref.png")
    for i in np.linspace(0, len(items) - 1, 6).astype(int):
        Image.open(items[i][1]).save(bd / f"sample_{items[i][0]:.0f}s.jpg")
    print(f"debug bundle -> {bd}/ (zip and attach this folder)")

    names = list(boxes)
    with open(a.out, "w", newline="") as fh:
        wcsv = csv.writer(fh)
        hdr = ["t_s", "bug_present"]
        for n_ in names:
            hdr += [f"{n_}_state", f"{n_}_chg"]
        wcsv.writerow(hdr)
        for rec in rows_out:
            row = [f"{rec['t']:.2f}", rec["present"]]
            for n_ in names:
                row += [rec[f"{n_}_state"], rec[f"{n_}_chg"]]
            wcsv.writerow(row)

    npres = sum(r["present"] for r in rows_out)
    print(f"\nwrote {a.out}")
    print(f"bug present {npres}/{len(rows_out)} seconds")
    print("per-element transitions (score digits ~78 expected; serve "
          "indicator ~190+; game chip ~a handful):")
    for n_ in names:
        tr = sum(r[f"{n_}_chg"] for r in rows_out)
        ns = len(states[n_])
        b = boxes[n_]
        print(f"  {n_} (rows {b[0].start}-{b[0].stop}, cols {b[1].start}-"
              f"{b[1].stop}): {tr} transitions, {ns} distinct states")


if __name__ == "__main__":
    main()
