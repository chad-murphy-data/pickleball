"""Final rally windows (v4): user serve marks pin the 16 labeled rallies.

The windows saga's last mile.  Machine alignment (cheer join, then
scorebug grammar) kept failing exactly where it mattered — early game 1,
where the 203 hand-labeled shots live — so the user hand-marked the serve
instant of rallies 1-16 in the audit tool (serve_time_s column of the
labels CSV).  Two facts make marks-only windows airtight there:

  * labels and marks come from the SAME video, so every labeled shot of
    rally k is visible between serve_k and serve_{k+1} by construction;
  * this condensed VOD cuts between-rally dead time to ~2-5 s (the log's
    wall clock shows 15-22 s pre-serve leads that simply aren't in the
    video), so consecutive serve marks are a tight bracket.

Window rule for the marked block (rallies 1-16):

    t0 = serve - 1.5
    t1 = min(next_serve - 2.0, serve + 1.5 * n_shots + 5.0)

The cadence cap trims replay/dead tails on short rallies (a replay of a
prior rally inside a window would feed the probe duplicated swings); the
next-serve bound keeps windows disjoint.  Every other rally keeps its
grammar-aligned v3 row verbatim (approx flags included) — the scorebug
stream is clean mid-match and junk-stormy only around game 1's start,
which is precisely the block the marks replace.

Marks are accepted ONLY for rally_cum 1-16 (the frozen label set): the
tool's early jumpy version left a stale duplicate mark on rally 18
(91.11 s = rally 3's serve), which this rule drops.

    python3 vision/serve_pin_windows.py \
        --labels data/vision/shot_labels_chicago0725.csv \
        --v3 data/vision/rally_windows_chicago0725_v3.csv \
        --out data/vision/rally_windows_chicago0725_v4.csv

Prints a feasibility table (serve-to-serve gap vs labeled shot count) —
rally 11 (29 shots in a 26.8 s gap, ~0.86 s/shot) is flagged as tight
but stands: the user watched all 29 shots inside that span.
"""
from __future__ import annotations

import argparse
import csv

MARKED = set(range(1, 17))   # the frozen 16-rally label set
PAD_PRE = 1.5                # window opens this early before the serve
GAP_POST = 2.0               # window closes this early before next serve
CADENCE = 1.5                # generous s/shot upper bound
TAIL = 5.0                   # + slack for the final ball landing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/vision/shot_labels_chicago0725.csv")
    ap.add_argument("--v3", default="data/vision/rally_windows_chicago0725_v3.csv")
    ap.add_argument("--out", default="data/vision/rally_windows_chicago0725_v4.csv")
    a = ap.parse_args()

    marks, shots = {}, {}
    for r in csv.DictReader(open(a.labels)):
        cum = int(r["rally_cum"])
        shots[cum] = max(shots.get(cum, 0), int(r["shot_index"]))
        if r.get("serve_time_s"):
            if cum in MARKED:
                marks[cum] = float(r["serve_time_s"])
            else:
                print(f"  dropping mark on rally {cum} "
                      f"({float(r['serve_time_s']):.2f}s — outside label set)")
    missing = MARKED - set(marks)
    if missing:
        raise SystemExit(f"unmarked labeled rallies: {sorted(missing)}")

    print("rally  serve_v   gap-to-next  shots  s/shot   t0      t1     bound")
    win = {}
    for cum in sorted(marks):
        sv = marks[cum]
        nxt = marks.get(cum + 1)
        cap = sv + CADENCE * shots[cum] + TAIL
        t1 = min(nxt - GAP_POST, cap) if nxt else cap
        win[cum] = (sv - PAD_PRE, t1)
        gap = (nxt - sv) if nxt else float("nan")
        rate = (gap - 3.0) / shots[cum] if nxt else float("nan")
        note = "next-serve" if (nxt and nxt - GAP_POST < cap) else "cadence"
        warn = "  <-- tight" if rate == rate and rate < 1.0 else ""
        print(f"{cum:>5}  {sv:7.2f}   {gap:8.2f}   {shots[cum]:>4}  "
              f"{rate:6.2f}  {sv-PAD_PRE:7.2f} {t1:7.2f}  {note}{warn}")

    rows = list(csv.DictReader(open(a.v3)))
    n_rep = n_clamp = 0
    t1_last = max(t1 for _, t1 in win.values())
    for r in rows:
        cum = int(r["rally_cum"])
        if cum in win:
            t0, t1 = win[cum]
            r["t0s"], r["t1s"] = f"{t0:.1f}", f"{t1:.1f}"
            r["approx"] = "0"        # human-pinned: the most confident tier
            n_rep += 1
        elif int(r["game"]) == 1 and float(r["t0s"]) < t1_last + 0.5:
            r["t0s"] = f"{t1_last + 0.5:.1f}"   # no bleed into the marked block
            n_clamp += 1

    for i in range(1, len(rows)):
        a0, b1 = float(rows[i]["t0s"]), float(rows[i - 1]["t1s"])
        if int(rows[i]["rally_cum"]) in win and int(rows[i - 1]["rally_cum"]) in win \
                and a0 < b1:
            raise SystemExit(f"overlap between marked rallies "
                             f"{rows[i-1]['rally_cum']} and {rows[i]['rally_cum']}")

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_conf = sum(1 for r in rows if r["approx"] == "0")
    print(f"\nwrote {a.out}: {len(rows)} rallies, {n_rep} mark-pinned, "
          f"{n_clamp} clamped after the block, {n_conf} total approx=0")


if __name__ == "__main__":
    main()
