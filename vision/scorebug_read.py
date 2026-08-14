"""Read the scorebug state per second from the 1/s crops — v6.

The user's design: every second, look at the score elements and ask
"same as a second ago?"  No OCR: each element's crop is matched against
previously seen appearances; a new appearance = a new state; states
self-label at alignment via the referee log (side-out rules per the
user: side switch = one dot by the new serving team's score, second dot
appears at X-Y-2).

Geometry, learned the hard way across six debug rounds: the panel WIDENS
for long player names, so boxes anchor to the strongest bright->dark
column step (name plate -> digit tile), which measured 299/299/308/313
across games 1-4.  Each team row gets ONE COMBINED dots+digits box
(anchor -16..+32, verified visually on all four game layouts, two-digit
scores included) — every rally end changes at least one row: a point
changes the scorer's row, X-Y-2 changes the serving row's dots, a
side-out changes BOTH rows, and the game chip marks game boundaries.
The log's known outcome sequence disambiguates which-was-which at
alignment.

    python3 scorebug_read.py --crops full_candidates_scorebug_crops \
                             --out scorebug_states.csv

Sanity for this matchup: each row ~150-250 transitions, game chip ~a
handful, distinct states ~tens.  Thousands of states = wrong.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

# y-extents are stable; x-extents FLOAT with the panel's right edge — the
# panel widens for long player names (verified on the men's and mixed
# games), so score/dot columns are defined RELATIVE to the detected edge.
ROWS = {
    "utah_row": slice(30, 74),
    "chi_row":  slice(80, 121),
    "game_chip": slice(46, 82),
}
GAME_CHIP_X = slice(4, 92)  # left side, position-stable
ANCHOR_SEARCH = (275, 345)  # where the digit tile's right edge can live
ROW_XREL = (-16, 32)        # combined dots+digits box around the anchor
CORR_SAME = 0.90
PERSIST_S = 2


def panel_anchor(fr):
    """x of the strongest bright->dark step across the score rows: the
    name plate / digit tile boundary. Stable across the per-game panel
    widening (verified 299/299/308/313 on games 1-4 samples); the
    combined row box straddles it so two-digit scores stay inside."""
    band = fr[30:118, :]
    col = band.mean(axis=0)
    lo, hi = ANCHOR_SEARCH
    seg = np.arange(lo, hi)
    step = col[seg + 3] - col[seg]
    return int(seg[np.argmin(step)]) + 1


def load_gray(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def crop_times(crops_dir):
    out = []
    for p in Path(crops_dir).iterdir():
        m = re.match(r"t(\d+\.\d+)\.jpe?g$", p.name)
        if m:
            out.append((float(m.group(1)), p))
    out.sort()
    return out


def norm(v):
    v = v.ravel().astype(np.float64)
    v = v - v.mean()
    n = float(np.sqrt((v * v).sum()))
    return v / n if n > 1e-9 else v * 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True, type=Path)
    ap.add_argument("--out", default="scorebug_states.csv")
    a = ap.parse_args()

    print("scorebug_read v6 — combined row boxes, plate-step anchor")
    items = crop_times(a.crops)
    print(f"{len(items)} crops")

    # presence reference: median of sampled frames; presence = correlation
    # of the LEFT 45% (logo/names, score-independent) with the reference
    idx = np.linspace(0, len(items) - 1, 60).astype(int)
    sample = [load_gray(items[i][1]) for i in idx]
    h = min(s.shape[0] for s in sample)
    w = min(s.shape[1] for s in sample)
    ref = np.median([s[:h, :w] for s in sample], axis=0)
    left = (slice(0, h), slice(0, int(0.45 * w)))
    ref_left = norm(ref[left])

    names_all = list(ROWS)
    states = {name: [] for name in names_all}
    cur = {name: -1 for name in names_all}
    pend = {name: (-1, 0) for name in names_all}
    rows_out = []
    edge_hist = []
    for k, (t, p) in enumerate(items):
        fr = load_gray(p)[:h, :w]
        pr = float(norm(fr[left]) @ ref_left) > 0.55
        rec = {"t": t, "present": int(pr)}
        if pr:
            edge_hist.append(panel_anchor(fr))
            if len(edge_hist) > 15:
                edge_hist.pop(0)
        edge = int(np.median(edge_hist)) if edge_hist else 300
        rec["edge"] = edge
        boxes = {}
        for name in names_all:
            if name == "game_chip":
                boxes[name] = (ROWS[name], GAME_CHIP_X)
            else:
                boxes[name] = (ROWS[name],
                               slice(max(0, edge + ROW_XREL[0]),
                                     min(w, edge + ROW_XREL[1])))
        for name, box in boxes.items():
            changed = 0
            if pr:
                cropv = fr[box]
                if name != "game_chip":
                    want = ROW_XREL[1] - ROW_XREL[0]
                    if cropv.shape[1] != want:
                        pad = want - cropv.shape[1]
                        cropv = np.pad(cropv, ((0, 0), (max(0, pad), 0)))[:, -want:]
                v = norm(cropv)
                sid = None
                for si, sv in enumerate(states[name]):
                    if float(v @ sv) > CORR_SAME:
                        sid = si
                        break
                if sid is None:
                    states[name].append(v)
                    sid = len(states[name]) - 1
                if sid != cur[name]:
                    csid, run = pend[name]
                    run = run + 1 if sid == csid else 1
                    pend[name] = (sid, run)
                    if run >= PERSIST_S:
                        if cur[name] >= 0:
                            changed = 1
                        cur[name] = sid
                        pend[name] = (-1, 0)
                else:
                    pend[name] = (-1, 0)
            rec[f"{name}_state"] = cur[name]
            rec[f"{name}_chg"] = changed
        rows_out.append(rec)
        if k % 600 == 0:
            print(f"  {k}/{len(items)}", flush=True)

    names = names_all
    with open(a.out, "w", newline="") as fh:
        wcsv = csv.writer(fh)
        hdr = ["t_s", "bug_present", "panel_edge"]
        for n_ in names:
            hdr += [f"{n_}_state", f"{n_}_chg"]
        wcsv.writerow(hdr)
        for rec in rows_out:
            row = [f"{rec['t']:.2f}", rec["present"], rec.get("edge", "")]
            for n_ in names:
                row += [rec[f"{n_}_state"], rec[f"{n_}_chg"]]
            wcsv.writerow(row)

    print(f"\nwrote {a.out}")
    print(f"bug present {sum(r['present'] for r in rows_out)}/{len(rows_out)} s")
    print("element        transitions  distinct-states   (sanity)")
    hints = {"utah_row": "~150-250", "chi_row": "~150-250",
             "game_chip": "~4-10"}
    for n_ in names:
        tr = sum(r[f"{n_}_chg"] for r in rows_out)
        print(f"  {n_:<12} {tr:>6}       {len(states[n_]):>5}"
              f"           expect {hints[n_]}")


if __name__ == "__main__":
    main()
