"""Steps 1 & 3 of the vision POC — sync the audio to the referee log, then
report the four numbers that decide whether Tier 2 is worth building.

    python vision/poc_report.py --timeline data/vision/rally_timeline_809fe252.csv \
                                --contacts data/vision/contacts.csv

SYNC WITHOUT OCR.  The referee log is a square wave: ~20 s inside a rally,
a few seconds out, 73 times, with irregular durations.  That pattern is
distinctive enough to align against on its own — slide the contact list
against the rally windows and take the offset that puts the most contacts
inside them.  No scorebug reading, one free parameter.  (Scorebug OCR is
the more robust upgrade and unlocks clip-linking for the whole archive, but
it is not needed to answer the POC's question.)

THE FOUR NUMBERS.
  1. sync quality     — how sharply the alignment peaks. A flat curve means
                        the contacts are not tracking the rally structure at
                        all, which is itself the answer.
  2. density contrast — contacts per second inside rally windows vs outside.
                        Free validation: no hand labels, straight from the log.
  3. contacts/rally   — should be >=1 with a sane spread (pro rallies run
                        ~4-14 shots). Zeros and 40s are both detector failure.
  4. interval modes   — the actual hypothesis: does the inter-contact
                        interval distribution separate a slow dink mode
                        (~0.5 s) from a fast speed-up mode (~0.2 s)?

Two more checks that cost nothing because the log provides them:
  * TIMEOUT CONTROL — contact density inside logged timeouts should collapse.
  * LATE CLUSTERING — the rally window starts when the referee marks it,
    which is before the serve, so contacts should sit in the LATTER part of
    each window rather than uniformly. Uniform means we are detecting noise.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_timeline(path):
    rows = []
    for r in csv.DictReader(Path(path).open()):
        rows.append({
            "rally": int(r["rally"]),
            "start": dt.datetime.fromisoformat(r["t_start"]),
            "end": dt.datetime.fromisoformat(r["t_end"]),
            "outcome": r["outcome"],
        })
    t0 = rows[0]["start"]
    for r in rows:
        r["a"] = (r["start"] - t0).total_seconds()
        r["b"] = (r["end"] - t0).total_seconds()
    return rows, t0


def inside_count(windows, times, offset):
    """How many contact times land inside a rally window at this offset."""
    t = np.sort(times - offset)
    tot = 0
    for a, b in windows:
        i = np.searchsorted(t, a, "left")
        j = np.searchsorted(t, b, "right")
        tot += j - i
    return tot


def find_offset(windows, times, lo, hi, coarse=1.0, fine=0.05):
    grid = np.arange(lo, hi, coarse)
    counts = np.array([inside_count(windows, times, o) for o in grid])
    best = grid[int(np.argmax(counts))]
    fgrid = np.arange(best - coarse, best + coarse, fine)
    fcounts = np.array([inside_count(windows, times, o) for o in fgrid])
    return float(fgrid[int(np.argmax(fcounts))]), grid, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--contacts", required=True)
    ap.add_argument("--search-lo", type=float, default=-600.0,
                    help="seconds; video may start well before the first rally")
    ap.add_argument("--search-hi", type=float, default=3600.0)
    ap.add_argument("--fast-cut", type=float, default=0.35,
                    help="interval below this counts as a fast/speed-up shot")
    ap.add_argument("--out", default="data/vision/poc_report.json")
    args = ap.parse_args()

    rallies, t0 = load_timeline(args.timeline)
    windows = [(r["a"], r["b"]) for r in rallies]
    times = np.array([float(r["t_audio_s"])
                      for r in csv.DictReader(Path(args.contacts).open())])
    span = windows[-1][1] - windows[0][0]
    live = sum(b - a for a, b in windows)
    print(f"timeline: {len(rallies)} rallies, {span/60:.1f} min, "
          f"{live/span:.0%} live")
    print(f"contacts: {len(times)} onsets over "
          f"{(times.max()-times.min())/60:.1f} min of audio\n")

    # ---- 1. sync -------------------------------------------------------
    off, grid, counts = find_offset(windows, times, args.search_lo, args.search_hi)
    inside = inside_count(windows, times, off)
    base = np.median(counts)
    peak_ratio = inside / base if base > 0 else float("inf")
    print(f"1. SYNC            offset {off:+.2f}s   {inside}/{len(times)} "
          f"contacts inside rally windows ({inside/len(times):.0%})")
    print(f"                   peak/median of the alignment curve = "
          f"{peak_ratio:.2f}x  (flat ~1.0 means no real alignment)")

    t = times - off

    # ---- 2. density contrast ------------------------------------------
    dens_in = inside / live
    dens_out = (len(t) - inside) / max(1e-9, span - live)
    print(f"2. DENSITY         inside {dens_in:.3f}/s   outside {dens_out:.3f}/s"
          f"   ratio {dens_in/max(dens_out,1e-9):.1f}x")

    # ---- 3. contacts per rally ----------------------------------------
    per = []
    for a, b in windows:
        per.append(int(np.sum((t >= a) & (t <= b))))
    per = np.array(per)
    print(f"3. PER RALLY       median {np.median(per):.0f}   "
          f"p10 {np.percentile(per,10):.0f}   p90 {np.percentile(per,90):.0f}   "
          f"zeros {int(np.sum(per==0))}/{len(per)}")

    # ---- late clustering (free check) ---------------------------------
    rel = []
    for a, b in windows:
        sel = t[(t >= a) & (t <= b)]
        if len(sel) and b > a:
            rel.extend((sel - a) / (b - a))
    rel = np.array(rel)
    if len(rel):
        print(f"   late-clustering  mean position in window "
              f"{rel.mean():.2f} (0.5 = uniform; >0.5 expected, the referee "
              f"marks the start before the serve)")

    # ---- 4. interval modes --------------------------------------------
    iv = []
    for a, b in windows:
        sel = np.sort(t[(t >= a) & (t <= b)])
        if len(sel) > 1:
            iv.extend(np.diff(sel))
    iv = np.array([x for x in iv if x < 2.0])
    print(f"4. INTERVALS       n={len(iv)}")
    if len(iv) > 30:
        fast, slow = iv[iv < args.fast_cut], iv[iv >= args.fast_cut]
        print(f"   fast (<{args.fast_cut:.2f}s)  n={len(fast):5d}  "
              f"median {np.median(fast)*1000:.0f} ms")
        print(f"   slow (>={args.fast_cut:.2f}s) n={len(slow):5d}  "
              f"median {np.median(slow)*1000:.0f} ms")
        edges = np.arange(0, 1.25, 0.05)
        h, _ = np.histogram(iv, bins=edges)
        top = h.max()
        print("\n   inter-contact interval histogram")
        for i, c in enumerate(h):
            if edges[i] > 1.2:
                break
            bar = "#" * int(40 * c / max(top, 1))
            print(f"   {edges[i]:4.2f}-{edges[i+1]:4.2f}s |{bar} {c}")
        # crude bimodality: is there a dip between two peaks?
        sm = np.convolve(h, np.ones(3) / 3, mode="same")
        pk = [i for i in range(1, len(sm) - 1) if sm[i] > sm[i-1] and sm[i] >= sm[i+1]]
        print(f"\n   smoothed peaks at: "
              f"{[f'{edges[i]:.2f}s' for i in pk] or 'none — unimodal'}")

    rep = {"offset_s": off, "n_contacts": int(len(times)),
           "inside_fraction": float(inside / len(times)),
           "peak_ratio": float(peak_ratio),
           "density_in": float(dens_in), "density_out": float(dens_out),
           "per_rally_median": float(np.median(per)),
           "per_rally_zeros": int(np.sum(per == 0)),
           "mean_window_position": float(rel.mean()) if len(rel) else None,
           "n_intervals": int(len(iv)),
           "fast_median_ms": float(np.median(iv[iv < args.fast_cut]) * 1000)
           if len(iv) and np.any(iv < args.fast_cut) else None,
           "slow_median_ms": float(np.median(iv[iv >= args.fast_cut]) * 1000)
           if len(iv) and np.any(iv >= args.fast_cut) else None}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
