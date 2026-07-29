"""Rebuild the Design-B / Design-C half splits from the RAW rally source
(Supabase pb_rally, downloaded to the scratchpad) with strict correction
handling, for EVERY game whose switch-at-6 boundary is recoverable.

    python model/weather_review/b2b_rebuild_splits.py <rally_dir> <out.csv>

Output columns (one row per game):
  match_id, game_number, n_rallies,
  pa_pre, pb_pre, pa_post, pb_post,
  ra_pre, wa_pre, rb_pre, wb_pre, ra_post, wa_post, rb_post, wb_post,
  seq_ok        1 if the running score advances by exactly the outcome of
                every rally with no rewind/correction and starts at 0-0
  boundary_ok   1 if max(pa_pre, pb_pre) == 6 (the switch lands exactly
                where the rules put it)
  fa, fb        derived final point totals for side 0 / side 1

Side 'a' == server_side 0, matching data/decider_splits.csv.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


COLS = ["match_id", "game_number", "rally_number", "server_side",
        "outcome", "won", "server_score", "receiver_score"]


def games(rally_dir):
    """Headerless per-prefix CSVs, globally ordered by match_id."""
    cur = None
    buf = []
    for p in sorted(Path(rally_dir).glob("*.csv")):
        with open(p) as f:
            for rec in csv.reader(f):
                if not rec or rec[0] == "match_id":
                    continue
                row = dict(zip(COLS, rec))
                key = (row["match_id"], int(row["game_number"]))
                if key != cur:
                    if cur is not None:
                        yield cur, buf
                    cur, buf = key, []
                buf.append(row)
    if cur is not None:
        yield cur, buf


def summarise(rows):
    rows.sort(key=lambda r: int(r["rally_number"]))
    pts = [[0, 0], [0, 0]]      # [half][side]
    srv = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]   # [half][side][rallies,wins]
    seq_ok = 1
    exp = [0, 0]
    for i, r in enumerate(rows):
        if r["server_side"] == "" or r["server_score"] == "" or \
                r["receiver_score"] == "":
            seq_ok = 0            # NULL referee fields: game is not clean
            continue
        side = int(r["server_side"])
        ss, rs = int(r["server_score"]), int(r["receiver_score"])
        obs = [0, 0]
        obs[side], obs[1 - side] = ss, rs
        if obs != exp:
            seq_ok = 0
            exp = list(obs)          # resync so one glitch doesn't cascade
        half = 0 if max(obs) < 6 else 1
        srv[half][side][0] += 1
        won = int(r["won"])
        srv[half][side][1] += won
        if r["outcome"] == "point":
            pts[half][side] += 1
            exp[side] += 1
    return pts, srv, seq_ok


def main():
    rally_dir, out_path = sys.argv[1], sys.argv[2]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["match_id", "game_number", "n_rallies",
                    "pa_pre", "pb_pre", "pa_post", "pb_post",
                    "ra_pre", "wa_pre", "rb_pre", "wb_pre",
                    "ra_post", "wa_post", "rb_post", "wb_post",
                    "seq_ok", "boundary_ok", "fa", "fb"])
        n = 0
        for (mid, gn), rows in games(rally_dir):
            pts, srv, seq_ok = summarise(rows)
            pa_pre, pb_pre = pts[0]
            pa_post, pb_post = pts[1]
            boundary_ok = 1 if max(pa_pre, pb_pre) == 6 else 0
            w.writerow([mid, gn, len(rows),
                        pa_pre, pb_pre, pa_post, pb_post,
                        srv[0][0][0], srv[0][0][1], srv[0][1][0], srv[0][1][1],
                        srv[1][0][0], srv[1][0][1], srv[1][1][0], srv[1][1][1],
                        seq_ok, boundary_ok,
                        pa_pre + pa_post, pb_pre + pb_post])
            n += 1
    print(f"{n} games -> {out_path}")


if __name__ == "__main__":
    main()
