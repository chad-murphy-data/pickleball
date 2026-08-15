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
REPLAY_MAX_S = 90.0              # ...or LONGER: replays insert video time
LONGER_TOL_S = 4.0               # ...but longer only by jitter
COST_SKIP_RALLY = 5.0            # missed flip
Z_MATCH = 9.0                    # only flips this strong can BE a rally end
                                 # (weaker ones are skippable junk: storms,
                                 # animations; real digit flips are loud)


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
        # accept both this script's scan output ('diff') and the original
        # full-VOD pass's frame-exact stream ('strip_diff')
        d.append(float(r.get("diff") or r.get("strip_diff")))
    t, d = np.array(t), np.array(d)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / SCAN_FPS
    w = max(10, int(60.0 / max(dt, 1e-6)))     # 60 s rolling window at any fps
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


def align(flips, wall_ends, durs, skip_rally=COST_SKIP_RALLY,
          early=0.005, insert=0.2):
    """Exact monotone DP over MATCHED PAIRS, transitions by TIME WINDOW.

    Two failure modes of earlier versions, both caught on real data:
    (1) an index cap on consecutive skipped flips — the pre-match intro
    and between-game breaks are flip STORMS (449 detections for 191
    rallies on the Chicago VOD), so a cap forces the chain to anchor on
    junk and whole segments derail; skips are now unlimited, priced by
    flip strength.  (2) a hard "video gap cannot exceed wall gap" bound —
    REPLAYS INSERT video time, so dv > dw by up to REPLAY_MAX_S is
    allowed at a linear penalty.  Candidate next-flips come from a bisect
    over the feasible TIME band [min_play, dw + REPLAY_MAX_S], so cost is
    O(states x band), no caps anywhere."""
    import bisect
    m, n = len(flips), len(wall_ends)
    INF = 1e18
    MAXJ = 4                        # consecutive missed flips
    F = [f[0] for f in flips]
    zpre = [0.0]
    for f in flips:
        zpre.append(zpre[-1] + skip_flip_cost(f[1]))

    best = [dict() for _ in range(n)]   # best[j][i] = cost, flip i -> rally j
    par = {}
    for j in range(min(n, MAXJ + 1)):   # source: leading junk + missed rallies
        for i in range(m):
            if flips[i][1] < Z_MATCH:
                continue
            c = zpre[i] + j * skip_rally
            if c < best[j].get(i, INF):
                best[j][i] = c
                par[(i, j)] = None

    for j in range(n):
        for i, c in sorted(best[j].items()):
            for j2 in range(j + 1, min(n, j + 1 + MAXJ + 1)):
                min_play = sum(durs[k] for k in range(j + 1, j2 + 1))
                dw = wall_ends[j2] - wall_ends[j]
                lo = bisect.bisect_left(F, F[i] + min_play - 1.0, i + 1)
                hi = bisect.bisect_right(F, F[i] + dw + REPLAY_MAX_S, i + 1)
                for i2 in range(lo, hi):
                    if flips[i2][1] < Z_MATCH:
                        continue
                    dv = F[i2] - F[i]
                    slack = dw - dv
                    # slack > 0 = broadcast cut (common, free); slack < 0
                    # = claimed replay INSERT (rare, priced) — this is
                    # what makes lag-chains at missed flips expensive,
                    # since each one must claim a phantom insert.  The
                    # earliness term is only a tie-break: matchability
                    # (Z_MATCH) already excludes storm junk.
                    cost = (early * max(0.0, dv - min_play)
                            + insert * max(0.0, -slack)
                            + (zpre[i2] - zpre[i + 1])
                            + (j2 - j - 1) * skip_rally
                            + (10.0 if slack > CUT_MAX_S else 0.0))
                    if c + cost < best[j2].get(i2, INF):
                        best[j2][i2] = c + cost
                        par[(i2, j2)] = (i, j)

    end_best, end_state = INF, None
    for j in range(n):
        for i, c in best[j].items():
            tot = c + (zpre[m] - zpre[i + 1]) + (n - 1 - j) * skip_rally
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
    match_b, _ = align(flips, wall_ends, durs, early=0.06, insert=0.05)
    stable = [a is not None and a == b for a, b in zip(match, match_b)]
    n_ok = sum(1 for m in match if m is not None)
    print(f"aligned {n_ok}/{len(cums)} rallies (DP cost {cost:.1f}); "
          f"{sum(stable)} stable under skip-price perturbation")

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

    # Near a missed flip, gap structure CANNOT pin which end of a run of
    # similar-length rallies absorbed the miss — measured on synthetics:
    # the alternative readings differ by draw-noise under any pure-gap
    # cost model, so no coefficient perturbation reliably separates them.
    # Be conservative instead: flag a +-5 neighbourhood of every
    # unmatched rally (union with the instability probe), and let the
    # scorer exclude flagged rallies / the human audit the handful.
    fuzzy = {k for k, s in enumerate(stable) if not s}
    for k, mk in enumerate(match):
        if mk is None:
            for d in range(-5, 6):
                if 0 <= k + d < len(match):
                    fuzzy.add(k + d)
    shifts = []
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        hdr = list(old[0].keys())
        w.writerow(hdr)
        for k, (r, t1, mk) in enumerate(zip(old, t1_video, match)):
            r = dict(r)
            old_t1 = float(r["t1s"])
            dur = float(r["dur_s"])
            r["t1s"] = f"{t1:.1f}"
            r["t0s"] = f"{t1 - dur:.1f}"
            r["approx"] = "0" if (mk is not None and k not in fuzzy) else "1"
            shifts.append(t1 - old_t1)
            w.writerow([r[h] for h in hdr])
    print(f"confident windows: {sum(1 for k, mk in enumerate(match) if mk is not None and k not in fuzzy)}"
          f"/{len(match)} (rest flagged approx: missed-flip neighbourhoods)")
    # validation gates on real data:
    # (1) the POC hand-verified the CHICAGO 5->6 flip (rally #30, wall
    #     18:32:18Z) at video 778.07 s — a photo-grade absolute anchor
    for k, c in enumerate(cums):
        if abs(wall_ends[k] - 66738.0) < 1.0:
            got = t1_video[k] + FLIP_LAG_S
            print(f"ANCHOR rally #{c} (wall 18:32:18Z): flip at video "
                  f"{got:.1f}s vs hand-verified 778.07s  "
                  f"({'PASS' if abs(got - 778.07) < 3 else 'FAIL'})")
    # (2) confident windows must not overlap within a game
    ov = 0
    for a, b in zip(range(len(old)), range(1, len(old))):
        if old[a]['game'] != old[b]['game']:
            continue
        if match[a] is not None and match[b] is not None:
            if (t1_video[b] - float(old[b]['dur_s'])) - t1_video[a] < -1.0:
                ov += 1
    print(f"overlapping consecutive matched windows: {ov} (must be ~0)")
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
    # replay inserts: some gaps carry +15-40 s of inserted video time
    ins = rng.choice(59, 8, replace=False)
    add = np.zeros(60)
    for k in ins:
        add[k + 1:] += rng.uniform(15, 40)
    video = video + add
    flips = [(float(v), rng.uniform(9, 25)) for v in video]
    drop = rng.choice(60, 3, replace=False)             # missed flips
    flips = [f for k, f in enumerate(flips) if k not in drop]
    junk = [(float(rng.uniform(2, 38)), rng.uniform(4.5, 8.5))   # pre-roll
            for _ in range(30)]
    storm_at = [video[19] + 5, video[39] + 5]                    # breaks
    for s in storm_at:
        junk += [(float(s + rng.uniform(0, 25)), rng.uniform(4.5, 8.5))
                 for _ in range(20)]
    junk += [(float(rng.uniform(50, video[-1])), rng.uniform(4.5, 8.5))
             for _ in range(9)]                                  # scattered
    flips = sorted(flips + junk)
    match, cost = align(flips, [float(w) for w in wall], durs)
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
    match_b, _ = align(flips, [float(w) for w in wall], durs, early=0.06, insert=0.05)
    fuzzy = {k for k in range(60)
             if match[k] is None or match[k] != match_b[k]}
    for k in range(60):
        if match[k] is None:
            for d in range(-5, 6):
                if 0 <= k + d < 60:
                    fuzzy.add(k + d)
    unflagged_bad = [k for k in range(60)
                     if match[k] is not None and k not in drop
                     and abs(flips[match[k]][0] - video[k]) >= 1.0
                     and k not in fuzzy]
    print(f"selftest: {ok} correct, {bad} wrong ({len(unflagged_bad)} of them "
          f"UNFLAGGED), {miss} unmatched (3 deleted, 79 junk incl. storms, "
          f"8 replay inserts; DP cost {cost:.1f})")
    assert ok >= 50, "alignment selftest failed: too few correct"
    assert not unflagged_bad, f"wrong matches escaped the approx flag: {unflagged_bad}"
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
