"""Rally windows for a FRESH VOD: scorebug flips x referee timeline.

scorebug_windows.py proved the method (frame-exact flip train, monotone
DP alignment to the log's rally ends) but its build() needs the Chicago
v1 windows file for durations and schema, so it cannot window a new
video.  This is the same alignment with the timeline as the ONLY wall
source — the coverage spec's prerequisite ("rally windows at scale come
from the scorebug reader") made runnable on any VOD that has a referee
log.

    python vision/scorebug_windows.py --scan vod.mp4 --diff diff.csv
    python vision/coverage_windows.py --diff diff.csv \
        --timeline data/vision/rally_timeline_<id8>.csv \
        --out data/vision/coverage_windows_<stem>.csv
    python vision/coverage_windows.py --selftest

Window rule: t1 = matched flip - FLIP_LAG_S, t0 = t1 - duration_s.  The
log's duration includes the pre-serve lead (~6-20 s of the referee's
start press), so t0 is EARLY by design — the anchor-frame finder in
coverage.py pins the actual serve inside the window; coverage counts
nothing before it.

approx=1 (dropped by coverage, still extracted) when any of:
  * the rally's flip was unmatched (window is interpolated), or sits in
    the +-2 neighbourhood of an unmatched one;
  * the DP re-run with perturbed skip prices moves the match (the
    instability probe from scorebug_windows);
  * the matched pair claims a replay INSERT beyond jitter (video gap
    longer than wall gap by > LONGER_TOL_S): inserted footage inside the
    span means the window may contain a re-aired rally — the coverage
    spec's replay trap, flagged mechanically.

Replays BETWEEN rallies need no flag at all: coverage counts only
frames inside a window and after its serve anchor, so between-window
footage (where broadcasts air replays) is excluded by construction.

Output schema is windows_from_v4-compatible (rally_cum,t0s,t1s,...) so
pose_extract.py consumes it via --windows with no changes.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import numpy as np

from scorebug_windows import (FLIP_LAG_S, LONGER_TOL_S, align, detect_flips)

ROOT = Path(__file__).resolve().parent.parent

FUZZY_NEIGHBOURHOOD = 5      # rallies to each side of an unmatched flip —
                             # scorebug_windows MEASURED that gap structure
                             # cannot pin which end of a similar-rally run
                             # absorbed a miss, and flags +-5; same rule


def parse_wall(iso):
    """ISO timestamp -> epoch seconds (align() only uses differences)."""
    return dt.datetime.fromisoformat(iso).timestamp()


def load_timeline(path):
    """Accepts BOTH timeline schemas: single-match (rally, game, ...)
    and matchup (rally, slot, match_id, game, ...).  slot/match_id pass
    through so a matchup VOD's rallies keep their per-match identity —
    (game, rally) alone collides across an MLP matchup's slots."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"empty timeline {path}")
    for i, r in enumerate(rows):
        r["_cum"] = i + 1
        r["_wall_end"] = parse_wall(r["t_end"])
        r["_dur"] = float(r["duration_s"])
    return rows


def build_windows(rows, flips):
    """Timeline rows + detected flips -> per-rally video windows.

    Returns (records, diag).  Records carry t0s/t1s/approx plus the
    matched flip's slack for the ledger; diag summarises for the events
    CSV."""
    wall_ends = [r["_wall_end"] for r in rows]
    durs = [r["_dur"] for r in rows]
    match, cost = align(flips, wall_ends, durs)
    match_b, _ = align(flips, wall_ends, durs, early=0.06, insert=0.05)
    unstable = {k for k, (a, b) in enumerate(zip(match, match_b)) if a != b}

    t1 = [flips[m][0] - FLIP_LAG_S if m is not None else None for m in match]
    # interpolate unmatched between matched neighbours, wall-scaled
    for k in range(len(t1)):
        if t1[k] is not None:
            continue
        lo = next((x for x in range(k - 1, -1, -1) if t1[x] is not None), None)
        hi = next((x for x in range(k + 1, len(t1)) if t1[x] is not None), None)
        if lo is not None and hi is not None:
            f = ((wall_ends[k] - wall_ends[lo])
                 / max(wall_ends[hi] - wall_ends[lo], 1e-9))
            t1[k] = t1[lo] + f * (t1[hi] - t1[lo])
        elif lo is not None:
            t1[k] = t1[lo] + (wall_ends[k] - wall_ends[lo])
        elif hi is not None:
            t1[k] = t1[hi] - (wall_ends[hi] - wall_ends[k])
        else:
            raise SystemExit("no flip matched any rally — wrong video?")

    fuzzy = set(unstable)
    for k, mk in enumerate(match):
        if mk is None:
            for d in range(-FUZZY_NEIGHBOURHOOD, FUZZY_NEIGHBOURHOOD + 1):
                if 0 <= k + d < len(match):
                    fuzzy.add(k + d)

    recs, n_insert = [], 0
    for k, r in enumerate(rows):
        slack = ""
        insert_flag = False
        if match[k] is not None and k > 0 and match[k - 1] is not None:
            dv = flips[match[k]][0] - flips[match[k - 1]][0]
            dw = wall_ends[k] - wall_ends[k - 1]
            slack = f"{dw - dv:.1f}"
            if dw - dv < -LONGER_TOL_S:      # video longer than wall: insert
                insert_flag = True
                n_insert += 1
        approx = int(match[k] is None or k in fuzzy or insert_flag)
        # t0 clamped at 0: decode_window clamps its seek to 0 anyway,
        # but stamps frame times from the UNCLAMPED t0 — a negative t0
        # here would silently shift every timestamp of that rally's npz
        recs.append({
            "rally_cum": r["_cum"], "game": r["game"],
            "rally_in_game": r["rally"],
            "slot": r.get("slot", ""), "match_id": r.get("match_id", ""),
            "t0s": f"{max(t1[k] - r['_dur'], 0.0):.1f}",
            "t1s": f"{t1[k]:.1f}",
            "dur_s": f"{r['_dur']:.1f}", "lead_s": r.get("lead_s", ""),
            "approx": approx, "slack_s": slack,
            "start_score": r["start_score"], "outcome": r["outcome"],
            "server_uuid": r["server_uuid"],
            "receiver_uuid": r["receiver_uuid"],
        })
    n_ok = sum(1 for m in match if m is not None)
    diag = {
        "n_rallies": len(rows), "n_flips": len(flips),
        "n_matched": n_ok, "n_confident": sum(1 for r in recs
                                              if r["approx"] == 0),
        "n_unstable": len(unstable), "n_insert_flagged": n_insert,
        "dp_cost": round(cost, 1),
    }
    return recs, diag


def write_windows(recs, out):
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)


def run(diff_csv, timeline_csv, out_csv):
    rows = load_timeline(timeline_csv)
    flips = detect_flips(diff_csv)
    print(f"{len(flips)} flips for {len(rows)} rallies")
    recs, diag = build_windows(rows, flips)
    write_windows(recs, out_csv)
    print(f"matched {diag['n_matched']}/{diag['n_rallies']} "
          f"(DP cost {diag['dp_cost']}), confident {diag['n_confident']}, "
          f"{diag['n_unstable']} unstable, "
          f"{diag['n_insert_flagged']} insert-flagged")
    print(f"wrote {out_csv}")
    return recs, diag


# ------------------------------------------------------------ selftest


def selftest():
    rng = np.random.default_rng(7)
    n = 40
    durs = rng.uniform(8, 30, n)
    dead = rng.uniform(4, 25, n)
    cuts = np.clip((dead - 3.0) * rng.uniform(0, 0.8, n), 0, None) \
        * (rng.random(n) < 0.5)
    base = dt.datetime(2026, 1, 25, 18, 0, tzinfo=dt.timezone.utc)
    wall, video, rows = [], [], []
    tw, tv = 0.0, 30.0
    game = 1
    for k in range(n):
        if k in (14, 29):
            game += 1
        tw += durs[k] + 1.5
        wall.append(tw)
        tv += durs[k] + 1.5
        video.append(tv)
        tw += dead[k] - 1.5
        tv += dead[k] - 1.5 - cuts[k]
        rows.append({
            "rally": str(k + 1), "game": str(game),
            "t_end": (base + dt.timedelta(seconds=wall[k])).isoformat(),
            "duration_s": f"{durs[k]:.1f}", "lead_s": "6.0",
            "start_score": "0-0-2", "outcome": "point",
            "server_uuid": "s", "receiver_uuid": "r",
            "_cum": k + 1, "_wall_end": None, "_dur": durs[k],
        })
    for r, w_ in zip(rows, wall):
        r["_wall_end"] = parse_wall(r["t_end"])
    # replay insert before rally 20: +25 s of video
    for k in range(20, n):
        video[k] += 25.0
    flips = [(float(v), float(rng.uniform(10, 25))) for v in video]
    dropped = {8, 33}
    flips = [f for k, f in enumerate(flips) if k not in dropped]
    junk = [(float(rng.uniform(1, 28)), float(rng.uniform(5, 8)))
            for _ in range(12)]
    flips = sorted(flips + junk)

    recs, diag = build_windows(rows, flips)
    assert diag["n_matched"] >= n - 4, f"too few matched: {diag}"
    err = wrong_unflagged = 0
    for k, r in enumerate(recs):
        want_t1 = video[k] - FLIP_LAG_S
        off = abs(float(r["t1s"]) - want_t1)
        if off < 0.6:
            err += 1
        elif r["approx"] == 0:
            wrong_unflagged += 1
    assert err >= n - 6, f"only {err}/{n} windows within 0.6s"
    assert wrong_unflagged == 0, \
        f"{wrong_unflagged} wrong windows escaped the approx flag"
    for k in dropped:
        assert recs[k]["approx"] == 1, f"dropped-flip rally {k} not flagged"
    # the insert-flag must catch the replay boundary (rally 20)
    assert recs[20]["approx"] == 1 and recs[20]["slack_s"] != "" \
        and float(recs[20]["slack_s"]) < -LONGER_TOL_S, \
        "replay insert not flagged"
    n_conf = diag["n_confident"]
    print(f"selftest: {err}/{n} windows within 0.6s, {n_conf} confident, "
          f"insert + dropped-flip rallies all flagged")
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", type=Path, help="scorebug diff CSV "
                    "(from scorebug_windows.py --scan)")
    ap.add_argument("--timeline", type=Path,
                    help="rally_timeline_<id8>.csv for the match/matchup")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.diff and a.timeline and a.out):
        ap.error("need --diff, --timeline and --out (or --selftest)")
    run(a.diff, a.timeline, a.out)


if __name__ == "__main__":
    main()
