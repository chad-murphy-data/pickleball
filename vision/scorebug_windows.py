"""Rebuild rally->video windows from SCOREBUG FLIPS — the user's fix for
the cheer-join failure (2026-08-14: windows ran up to 40 s off; the ±1 s
"validation" was circular because the audit sheet's times came from the
same join).

WHY THIS WORKS WHERE CHEERS FAILED
    A cheer is a fuzzy, optional event loosely near a rally end.  A score
    flip is a PIXEL-LEVEL, FRAME-EXACT event (validated in the POC: the
    5->6 digit flip hit one frame, diff 185 vs 48 background) that occurs
    exactly once per rally, in a fixed screen region, and the referee log
    already knows the full SEQUENCE of score changes with wall-clock
    times.  So no OCR is needed: detect WHEN the scorebug changes, then
    monotonically align that flip train to the log's rally ends, allowing
    extra flips (replays, bug animations) and missed flips.  Inter-event
    spacing does the identification: within a game, video gap = wall gap
    MINUS whatever dead time the broadcast trimmed, so video gaps can be
    SHORTER than wall gaps but (almost) never longer.

    python vision/scorebug_windows.py --scan full_match.mp4.webm
    python vision/scorebug_windows.py --align
    python vision/scorebug_windows.py --selftest

--scan decodes ONLY the scorebug corner (crop happens inside ffmpeg) at
10 fps -> scorebug_diff.csv (~15-25 min for an 80 min VOD).  If the
ORIGINAL full-VOD run's chicago0725*_scorebug.csv still exists, skip the
scan and pass it via --diff.  --align writes
data/vision/rally_windows_chicago0725_v2.csv (same schema as v1) plus a
shift report against the old windows.  The probe must then be RE-RUN on
the v2 windows: all v1 probe output watched the wrong spans.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/vision"
OLD_WINDOWS = D / "rally_windows_chicago0725.csv"

SB_H, SB_W = 0.17, 0.42          # scorebug corner (top-left), from the
SCAN_FPS = 10.0                  # validated full-VOD pass
FLIP_LAG_S = 1.0                 # flip trails the rally's final ball

# alignment costs
CUT_MAX_S = 240.0                # video gap may be shorter than wall gap
LONGER_TOL_S = 4.0               # ...but longer only by jitter
COST_SKIP_RALLY = 5.0            # missed flip


def skip_flip_cost(z):
    """Real digit flips slam the scorebug diff far harder than replay
    wipes and animations, so skipping a strong flip costs more."""
    return min(0.25 * z, 4.0)
COST_LONGER = 60.0


def ffmpeg_bin():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


# ------------------------------------------------------------------ scan


def scan(video, out_csv):
    """One cheap pass: mean |frame-to-frame diff| of the scorebug corner."""
    W, H = 192, 64
    cmd = [ffmpeg_bin(), "-v", "error", "-i", str(video),
           "-vf", f"crop=iw*{SB_W}:ih*{SB_H}:0:0,scale={W}:{H},fps={SCAN_FPS}",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 64)
    n = W * H
    prev = None
    i = 0
    import time
    t0 = time.time()
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "diff"])
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            fr = np.frombuffer(b, np.uint8).astype(np.int16)
            if prev is not None:
                w.writerow([f"{i / SCAN_FPS:.2f}",
                            f"{float(np.abs(fr - prev).mean()):.3f}"])
            prev = fr
            i += 1
            if i % (600 * int(SCAN_FPS)) == 0:
                el = time.time() - t0
                print(f"  {i / SCAN_FPS / 60:5.1f} min of video scanned "
                      f"({i / SCAN_FPS / el:.1f}x realtime)", flush=True)
    p.wait()
    print(f"wrote {out_csv} ({i} frames)")


# ------------------------------------------------------- flip detection


def detect_flips(diff_csv, floor_z=5.0, refractory=3.0):
    t, d = [], []
    for r in csv.DictReader(open(diff_csv)):
        t.append(float(r["t_s"]))
        d.append(float(r["diff"]))
    t, d = np.array(t), np.array(d)
    w = int(60 * SCAN_FPS)
    z = np.zeros_like(d)
    for i in range(len(d)):
        lo, hi = max(0, i - w // 2), min(len(d), i + w // 2)
        med = np.median(d[lo:hi])
        mad = np.median(np.abs(d[lo:hi] - med)) + 1e-9
        z[i] = (d[i] - med) / (1.4826 * mad)
    cands = [(float(t[i]), float(z[i])) for i in range(1, len(d) - 1)
             if z[i] >= floor_z and z[i] >= z[i - 1] and z[i] >= z[i + 1]]
    keep = []
    for c in sorted(cands, key=lambda x: -x[1]):
        if all(abs(c[0] - k[0]) >= refractory for k in keep):
            keep.append(c)
    keep.sort()
    return keep


# ------------------------------------------------------------ alignment


def align(flips, wall_ends, durs):
    """Exact monotone DP over MATCHED PAIRS.

    A state is "flip i is matched to rally j" — not "i flips and j rallies
    consumed", because the cost of the next match depends on WHICH pair
    was matched last, and collapsing that into an (i, j) counter table
    violates Bellman optimality (the first version did exactly that and
    produced paths costlier than the true chain; caught by the selftest).

    Physics per matched pair (i,j) -> (i',j'): the broadcast trims dead
    time, never play, so the video gap must cover the summed durations of
    rallies j+1..j' (junk flips inside play are infeasible) and cannot
    exceed the wall gap plus jitter.  Among feasible flips the EARLIEST
    is preferred (the score updates ~1-2 s after the final ball; replay
    wipes and animations trail it), charged as 0.1 * (dv - min_play).
    Skipped flips cost skip_flip_cost(z) (strong flips cost more), skipped
    rallies COST_SKIP_RALLY (missed flips)."""
    m, n = len(flips), len(wall_ends)
    INF = 1e18
    MAXJ, MAXI = 4, 14              # consecutive missed flips / junk flips
    F = [f[0] for f in flips]
    zpre = [0.0]
    for f in flips:
        zpre.append(zpre[-1] + skip_flip_cost(f[1]))

    best = {}
    par = {}
    # source: first matched pair (i, j) — i junk flips and j missed
    # rallies precede it
    for j in range(min(n, MAXJ + 1)):
        for i in range(min(m, MAXI + 1)):
            c = zpre[i] + j * COST_SKIP_RALLY
            if c < best.get((i, j), INF):
                best[(i, j)] = c
                par[(i, j)] = None

    order = sorted(best)            # will grow; process by (j, i)
    # iterate states in increasing rally order (monotone => DAG)
    all_states = [(i, j) for j in range(n) for i in range(m)]
    for i, j in sorted(all_states, key=lambda s: (s[1], s[0])):
        c = best.get((i, j), INF)
        if c >= INF:
            continue
        for j2 in range(j + 1, min(n, j + 1 + MAXJ + 1)):
            min_play = sum(durs[k] for k in range(j + 1, j2 + 1))
            dw = wall_ends[j2] - wall_ends[j]
            for i2 in range(i + 1, min(m, i + 1 + MAXI + 1)):
                dv = F[i2] - F[i]
                if dv < min_play - 1.0:
                    continue                     # junk inside play
                slack = dw - dv
                if slack < -LONGER_TOL_S:
                    break                        # dv only grows with i2
                cost = (0.1 * max(0.0, dv - min_play)
                        + (zpre[i2] - zpre[i + 1])
                        + (j2 - j - 1) * COST_SKIP_RALLY
                        + (10.0 if slack > CUT_MAX_S else 0.0))
                if c + cost < best.get((i2, j2), INF):
                    best[(i2, j2)] = c + cost
                    par[(i2, j2)] = (i, j)

    # sink: charge leftover flips and rallies
    end_best, end_state = INF, None
    for (i, j), c in best.items():
        tot = c + (zpre[m] - zpre[i + 1]) + (n - 1 - j) * COST_SKIP_RALLY
        if tot < end_best:
            end_best, end_state = tot, (i, j)

    match = [None] * n
    s = end_state
    while s is not None:
        i, j = s
        match[j] = i
        s = par.get(s)
    return match, end_best


def load_old(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["rally_cum"] = int(r["rally_cum"])
    rows.sort(key=lambda r: r["rally_cum"])
    return rows


def build(diff_csv, old_windows, timeline_csv, out_csv):
    old = load_old(old_windows)
    wall = {}
    for r in csv.DictReader(open(timeline_csv)):
        hh, mm, ss = r["t_end"].split("T")[1].split("+")[0].split(":")
        wall[int(r["rally"])] = int(hh) * 3600 + int(mm) * 60 + float(ss)
    cums = [r["rally_cum"] for r in old]
    wall_ends = [wall[c] for c in cums]
    durs = [float(r["dur_s"]) for r in old]

    flips = detect_flips(diff_csv)
    print(f"{len(flips)} scorebug flips detected for {len(cums)} rallies")
    match, cost = align(flips, wall_ends, durs)
    n_ok = sum(1 for m in match if m is not None)
    print(f"aligned {n_ok}/{len(cums)} rallies (DP cost {cost:.1f})")

    # interpolate the missed ones between matched neighbours (wall-scaled)
    t1_video = [flips[m][0] - FLIP_LAG_S if m is not None else None
                for m in match]
    for k in range(len(t1_video)):
        if t1_video[k] is None:
            lo = next((x for x in range(k - 1, -1, -1)
                       if t1_video[x] is not None), None)
            hi = next((x for x in range(k + 1, len(t1_video))
                       if t1_video[x] is not None), None)
            if lo is not None and hi is not None:
                f = ((wall_ends[k] - wall_ends[lo])
                     / max(wall_ends[hi] - wall_ends[lo], 1e-9))
                t1_video[k] = t1_video[lo] + f * (t1_video[hi] - t1_video[lo])
            elif lo is not None:
                t1_video[k] = t1_video[lo] + (wall_ends[k] - wall_ends[lo])
            elif hi is not None:
                t1_video[k] = t1_video[hi] - (wall_ends[hi] - wall_ends[k])

    shifts = []
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        hdr = list(old[0].keys())
        w.writerow(hdr)
        for r, t1, mk in zip(old, t1_video, match):
            r = dict(r)
            old_t1 = float(r["t1s"])
            dur = float(r["dur_s"])
            r["t1s"] = f"{t1:.1f}"
            r["t0s"] = f"{t1 - dur:.1f}"
            r["approx"] = "0" if mk is not None else "1"
            shifts.append(t1 - old_t1)
            w.writerow([r[h] for h in hdr])
    s = np.array(shifts)
    print(f"wrote {out_csv}")
    print(f"\nshift vs OLD windows (v2 minus v1, seconds):")
    print(f"  median {np.median(s):+.1f}  IQR [{np.percentile(s,25):+.1f}, "
          f"{np.percentile(s,75):+.1f}]  max |shift| {np.abs(s).max():.1f}")
    print(f"  rallies moved >5s: {(np.abs(s)>5).sum()}/{len(s)}   "
          f">20s: {(np.abs(s)>20).sum()}")
    big = sorted(zip(np.abs(s), [r['rally_cum'] for r in old]), reverse=True)[:8]
    print("  biggest: " + ", ".join(f"#{c}({sh:+.0f}s)" for sh, c in
                                    [(s[[r['rally_cum'] for r in old].index(c)], c)
                                     for _, c in big]))


# ------------------------------------------------------------ selftest


def selftest():
    rng = np.random.default_rng(3)
    # synthetic match: 60 rallies. Wall timeline: rally k plays dur_k, the
    # ref logs (wall end + scorebug flip) ~1.5 s later, then dead time.
    # The broadcast cuts only from the dead time AFTER each flip.
    gaps = rng.uniform(15, 60, 60)
    durs = list(gaps * rng.uniform(0.35, 0.70, 60))
    dead = gaps - np.array(durs)               # dead time AFTER each rally
    cuts = np.clip((dead - 4.0) * rng.uniform(0, 0.9, 60), 0, None) \
        * (rng.random(60) < 0.6)
    # WALL anatomy must match reality: the ref's press (the log's t_end)
    # comes right after the rally; dead time FOLLOWS it. Building wall
    # ends at the end of the dead period (the previous selftest) made
    # true pairs infeasible and framed the DP for its own test's bug.
    wall, video = [], []
    tw, tv = 0.0, 40.0
    for k in range(60):
        tw += durs[k] + 1.5
        wall.append(tw)
        tw += dead[k] - 1.5
        tv += durs[k] + 1.5                    # play, then the flip
        video.append(tv)
        tv += dead[k] - 1.5 - cuts[k]          # shown dead time
    wall, video = np.array(wall), np.array(video)
    flips = [(float(v), 10.0) for v in video]
    drop = rng.choice(60, 3, replace=False)             # missed flips
    flips = [f for k, f in enumerate(flips) if k not in drop]
    junk = [(float(rng.uniform(50, video[-1])), 6.0) for _ in range(9)]
    flips = sorted(flips + junk)
    match, cost = align(flips, list(wall), durs)
    ok = bad = miss = 0
    for k in range(60):
        if match[k] is None:
            miss += 1
            continue
        got = flips[match[k]][0]
        if k in drop:
            bad += 1                                     # matched a dropped one
        elif abs(got - video[k]) < 0.5:
            ok += 1
        else:
            bad += 1
    print(f"selftest: {ok} correct, {bad} wrong, {miss} unmatched "
          f"(3 flips were deleted, 9 junk added; DP cost {cost:.1f})")
    assert ok >= 54 and bad <= 2, "alignment selftest failed"
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, help="video file -> scorebug diff CSV")
    ap.add_argument("--diff", type=Path, default=Path("scorebug_diff.csv"))
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--old-windows", type=Path, default=OLD_WINDOWS)
    ap.add_argument("--timeline", type=Path,
                    default=D / "rally_timeline_matchup_20260725_c4e686d1.csv")
    ap.add_argument("--out", type=Path,
                    default=Path("rally_windows_chicago0725_v2.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.scan:
        scan(a.scan, a.diff)
        print("next: python3 scorebug_windows.py --align")
    elif a.align:
        build(a.diff, a.old_windows, a.timeline, a.out)
    else:
        ap.error("pick --scan VIDEO, --align, or --selftest")


if __name__ == "__main__":
    main()
