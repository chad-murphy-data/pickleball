"""Rally windows from the scorebug's STATE TIMELINE — the definitive cut.

Flip-spacing alignment fails rally-level on replay-heavy broadcasts
(read off the bug itself: 'confident' windows r83/r84 of the PPA mixed
final showed other rallies' scores).  This reads the bug's actual game
state — the score column plus the serving-team dots — at 2 Hz across
the WHOLE VOD and segments the video into constant-state runs.  Within
a game, score states are unique and monotone, and the bug keeps the
CURRENT state through replays and cutaways, so runs map 1:1, in order,
onto the referee log's known score sequence.  Alignment anchors on the
(serving team, server number) sequence, which is readable WITHOUT digit
OCR (dot blobs on one of two name rows) and known exactly from the log.

    python vision/bug_state_windows.py --video vod.mp4 \
        --timeline data/vision/rally_timeline_<id8>.csv \
        --windows data/vision/coverage_windows_<vod>.csv \
        --out data/vision/coverage_windows_<vod>_v3.csv
    python vision/bug_state_windows.py --selftest

Regions are auto-located from green-cell detection on sampled frames;
run boundaries are the state flips (t1 = boundary - FLIP_LAG_S), and a
rally's window is [t1 - duration, t1] exactly as before.  Output keeps
the input schema + state_check column (bugstate / unmatched)."""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scorebug_windows import FLIP_LAG_S, ffmpeg_bin

FPS = 2.0
W, H = 538, 218              # bug strip decode size (0.42w x 0.17h)


def stream_bug(video, fps=FPS, w=W, h=H):
    cmd = [ffmpeg_bin(), "-v", "error", "-i", str(video),
           "-vf", f"crop=iw*0.42:ih*0.17:0:0,scale={w}:{h},fps={fps}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * 3 * 8)
    n = w * h * 3
    i = 0
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            yield i / fps, np.frombuffer(b, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        p.kill()


def green_mask(fr):
    r = fr[:, :, 0].astype(np.int16)
    g = fr[:, :, 1].astype(np.int16)
    b = fr[:, :, 2].astype(np.int16)
    return (g > r + 25) & (g > b + 25) & (g > 70)


def locate_cells(frames):
    """The two green score cells + the dots bands left of them.

    Component-based: median green mask over sampled frames, then the
    PAIR of similar-width, vertically stacked components (venue turf
    and greenery also read green — a column-profile locator locked
    onto the white games-won column next door on the mixed final)."""
    from scipy import ndimage
    # mean, low threshold, then closing: the WHITE digits carve moving
    # holes in the green cells across frames, so a strict median erodes
    # the cells below any size floor
    m = np.mean(np.stack([green_mask(f) for f in frames]), axis=0) > 0.3
    m = ndimage.binary_closing(m, structure=np.ones((3, 3)), iterations=2)
    lab, n = ndimage.label(m)
    comps = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        if 12 <= h <= 90 and 8 <= w <= 70 and len(ys) > 0.4 * h * w:
            comps.append((ys.min(), ys.max() + 1, xs.min(), xs.max() + 1))
    best = None
    for a in comps:
        for b in comps:
            if a is b or not (a[1] <= b[0] + 6 and b[0] - a[1] < 20):
                continue
            ov = min(a[3], b[3]) - max(a[2], b[2])
            if ov < 0.6 * max(a[3] - a[2], b[3] - b[2]):
                continue
            area = (a[1] - a[0]) * (a[3] - a[2])
            if best is None or area > best[0]:
                best = (area, a, b)
    if best is not None:
        _, a, b = best
    else:
        # the two cells usually TOUCH (one tall component): take the
        # best cell-proportioned tall blob and split at its midpoint
        tall = [c for c in comps if (c[1] - c[0]) >= 24
                and (c[1] - c[0]) >= 0.9 * (c[3] - c[2])]
        if not tall:
            raise SystemExit("no stacked green score cells — different bug")
        c = max(tall, key=lambda c: (c[1] - c[0]) * (c[3] - c[2]))
        mid = (c[0] + c[1]) // 2
        a = (c[0], mid, c[2], c[3])
        b = (mid, c[1], c[2], c[3])
    x0 = min(a[2], b[2])
    x1 = max(a[3], b[3])
    cells = ((a[0], a[1], x0, x1), (b[0], b[1], x0, x1))
    dots = ((a[0], a[1], max(0, x0 - 130), x0 - 34),
            (b[0], b[1], max(0, x0 - 130), x0 - 34))
    return cells, dots


def block_pool(img, rows, cols):
    """Average-pool an arbitrary 2D array to a fixed rows x cols grid."""
    h, w = img.shape
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    out = np.empty((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            blk = img[ys[i]:max(ys[i + 1], ys[i] + 1),
                      xs[j]:max(xs[j + 1], xs[j] + 1)]
            out[i, j] = float(blk.mean())
    return out


def locate_cells_frame(gm):
    """Per-frame green-cell location: the bug SLIDES between layouts
    (a logo panel animates in/out and the score column moves ~130 px),
    so fixed boxes read digits in one layout and background in the
    other — every layout toggle faked a both-cells change.  Find the
    tall stacked green blob in THIS frame instead."""
    from scipy import ndimage
    m = ndimage.binary_closing(gm, structure=np.ones((3, 3)), iterations=2)
    m[:, :m.shape[1] // 3] = False        # names/turf live left
    lab, n = ndimage.label(m)
    best = None
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        if 24 <= h <= 90 and 8 <= w <= 70 and h >= 0.9 * w \
                and len(ys) > 0.4 * h * w:
            if best is None or len(ys) > best[0]:
                best = (len(ys), ys.min(), ys.max() + 1, xs.min(),
                        xs.max() + 1)
    if best is None:
        return None
    _, y0, y1, x0, x1 = best
    mid = (y0 + y1) // 2
    return ((y0, mid, x0, x1), (mid, y1, x0, x1))


def box_of(gm):
    """Union box (y0, y1, x0, x1) of the stacked score cells, or None."""
    loc = locate_cells_frame(gm)
    if loc is None:
        return None
    return (loc[0][0], loc[1][1], loc[0][2], loc[0][3])


def state_at(fr, gm, box):
    """Read the state vector + dot counts at a GIVEN box (the caller
    supplies a temporally smoothed box, so plateau crops are pixel-
    identical and locator wobble can't fake state changes)."""
    y0, y1, x0, x1 = box
    mid = (y0 + y1) // 2
    vecs, nd = [], []
    for ya, yb in ((y0, mid), (mid, y1)):
        cell = fr[ya:yb, x0:x1].mean(axis=2)
        vecs.append(block_pool(cell, 8, 6).ravel())
        dm = gm[ya:yb, max(0, x0 - 130):max(1, x0 - 34)]
        col = dm.sum(axis=0) >= max(2, (yb - ya) * 0.12)
        nd.append(int(np.diff(np.pad(col.astype(int), 1)).clip(0).sum()))
    return np.concatenate(vecs), tuple(nd)


def frame_state(fr, cells, dots):
    """(state vector, present flag, dots per row).  Single-frame path
    (selftest); the streaming pass uses per-layout canonical boxes."""
    gm = green_mask(fr)
    box = box_of(gm)
    if box is None:
        return np.zeros(96, np.float32), False, (0, 0)
    vec, nd = state_at(fr, gm, box)
    return vec, True, nd


# Cached crop region: every observed layout's cells + dots band live in
# here (measured on the PPA Indoor bug: score column x0 in {370, 400,
# 428}, cells y in [93, 175], dots band down to x0-130 = 240).
REG_Y0, REG_Y1, REG_X0, REG_X1 = 85, 180, 230, 536

# The serving dots hug the END OF THE TEAM NAME and are STATIC — they
# do not slide with the score column (measured: top-row dots x 80-99,
# bottom-row 50-69 in region coords across all three layouts; nothing
# else green in [40, 180]).  An x0-relative band read the wrong place
# in two of the three layouts and gave 47/73 runs unstable dot reads.
DOT_X = (45, 105)


def canonical_layouts(boxes, min_frac=0.01):
    """Cluster per-frame boxes into the bug's few discrete LAYOUTS and
    return one canonical (median) box per layout.  The overlay is
    pixel-static within a layout — the +-2 px the locator wobbles is
    read noise, and cropping at the wobbling (or windowed-median) box
    fakes state changes on a constant score.  Cropping every frame at
    its layout's ONE canonical box makes plateau crops pixel-identical
    across the whole video."""
    clusters = []
    for b in boxes:
        for c in clusters:
            if abs(int(b[2]) - c[0]) <= 6:
                c[1].append(b)
                break
        else:
            clusters.append((int(b[2]), [b]))
    keep = [c for c in clusters
            if len(c[1]) >= max(20, min_frac * len(boxes))]
    outs = []
    for _, mem in sorted(keep, key=lambda c: -len(c[1])):
        arr = np.array(mem)
        outs.append(tuple(int(v) for v in np.median(arr, axis=0)))
    return outs


def state_at_gray(gray, gmr, box, dot_x=None):
    """state_at on a cached grayscale region crop (2D) + its mask.
    dot_x: absolute (x_lo, x_hi) dots band; default falls back to the
    legacy x0-relative band."""
    y0, y1, x0, x1 = box
    dx = dot_x or (max(0, x0 - 130), max(1, x0 - 34))
    mid = (y0 + y1) // 2
    vecs, nd = [], []
    for ya, yb in ((y0, mid), (mid, y1)):
        vecs.append(block_pool(gray[ya:yb, x0:x1].astype(np.float32),
                               8, 6).ravel())
        dm = gmr[ya:yb, dx[0]:dx[1]]
        col = dm.sum(axis=0) >= max(2, (yb - ya) * 0.12)
        n_runs = int(np.diff(np.pad(col.astype(int), 1)).clip(0).sum())
        width = int(col.sum())
        # width fallback: two touching dots read as one run (measured
        # single dot 4-6 cols, double 8-13)
        nd.append(max(n_runs, 2 if width >= 9 else (1 if width >= 3
                                                    else 0)))
    return np.concatenate(vecs), tuple(nd)


def states_from_crops(ts, boxes, gray, gmask):
    """Per-frame states from the cached crop stream: assign each frame
    to a canonical layout (or absent), read at the canonical box.

    Layout doubles as the GAME id: the PPA bug appends a completed-game
    score column after each game, sliding the live score column right —
    so canonical layouts sorted by x0 ARE the games in order (measured:
    x0 370 = game 1, 400 = game 2, 428 = game 3, with the extra white
    columns visible in the crops).  Emits (t, vec, present, dots, lay)
    with lay the game-ordered layout index."""
    ok = boxes[:, 0] >= 0
    lays = sorted(canonical_layouts(boxes[ok]), key=lambda L: L[2])
    print(f"{len(lays)} bug layouts (game-ordered): {lays}")
    parsed = []
    for i in range(len(ts)):
        b = boxes[i]
        lay = None
        if b[0] >= 0:
            for li, L in enumerate(lays):
                if abs(int(b[2]) - L[2]) <= 5 and abs(int(b[0]) - L[0]) <= 5 \
                        and abs(int(b[3] - b[2]) - (L[3] - L[2])) <= 5 \
                        and abs(int(b[1] - b[0]) - (L[1] - L[0])) <= 6:
                    lay = (li, L)
                    break
        if lay is None:
            parsed.append((float(ts[i]), np.zeros(96, np.float32),
                           False, (0, 0), -1))
            continue
        li, L = lay
        rel = (L[0] - REG_Y0, L[1] - REG_Y0,
               L[2] - REG_X0, L[3] - REG_X0)
        vec, nd = state_at_gray(gray[i], gmask[i], rel, dot_x=DOT_X)
        parsed.append((float(ts[i]), vec, True, nd, li))
    print(f"present {sum(p[2] for p in parsed)}/{len(parsed)} frames")
    return parsed


def segment(states, thr):
    """Constant-state runs: [(t_start, t_end, med_vec, dots)]."""
    runs = []
    cur = None
    for row in states:
        t, vec, present, nd = row[:4]
        lay = row[4] if len(row) > 4 else 0
        if not present:
            continue
        # a >8 s absence closes the run even if the state resumes
        # unchanged: sparse gap-spanning runs poison window times and
        # dot-change splitting (the value clusters reunite them anyway)
        if cur and float(np.abs(vec - cur["ref"]).mean()) < thr \
                and lay == cur["lays"][-1] and t - cur["t1"] <= 8.0:
            cur["t1"] = t
            cur["vecs"].append(vec)
            cur["dots"].append(nd)
            cur["lays"].append(lay)
            cur["ts"].append(t)
        else:
            if cur and len(cur["vecs"]) >= 2:
                runs.append(cur)
            cur = {"t0": t, "t1": t, "ref": vec, "vecs": [vec],
                   "dots": [nd], "lays": [lay], "ts": [t]}
    if cur and len(cur["vecs"]) >= 2:
        runs.append(cur)
    for r in runs:
        r["vec"] = np.median(np.stack(r["vecs"]), axis=0)
        half = len(r["vec"]) // 2
        r["vtop"], r["vbot"] = r["vec"][:half], r["vec"][half:]
        r["lay"] = r["lays"][0]
        dc = defaultdict(int)
        for nd in r["dots"]:
            dc[nd] += 1
        r["dot"] = max(dc, key=dc.get)
    # merge consecutive runs whose MEDIAN states match — on the CELLS
    # alone (dot-count reads are noisy; requiring dot equality here
    # fragmented same-score runs and flooded the boundary symbols with
    # '?', which let the aligner slide onto junk)
    merged = [runs[0]] if runs else []
    for r in runs[1:]:
        cells_same = float(np.abs(r["vec"] - merged[-1]["vec"]).mean()) < thr \
            and r["lay"] == merged[-1]["lay"]
        # a PERSISTENT dot difference (>=4 samples, ~2 s) is a real
        # side-out/second boundary even with identical cells; a short
        # run's dot disagreement is read noise and merges away
        dot_boundary = (r["dot"] != merged[-1]["dot"]
                        and len(r["vecs"]) >= 4
                        and len(merged[-1]["vecs"]) >= 4)
        gap = r["t0"] - merged[-1]["t1"] > 8.0
        if cells_same and not dot_boundary and not gap:
            m = merged[-1]
            m["t1"] = r["t1"]
            m["dots"] = m["dots"] + r["dots"]
            m["ts"] = m["ts"] + r["ts"]
            m["lays"] = m["lays"] + r["lays"]
            dc = defaultdict(int)
            for nd in m["dots"]:
                dc[nd] += 1
            m["dot"] = max(dc, key=dc.get)
        else:
            merged.append(r)
    return merged


def serve_sig(row):
    """(serving team index, server number) from a timeline row's
    start_score X-Y-N and server side is not in the score — the TEAM is
    resolvable because the bug shows dots on the SERVING TEAM'S row;
    the log's start_score is serving-team-first, so the mapping to the
    bug's fixed team rows needs the team identity, supplied by the
    caller via team_of_server."""
    n = row["start_score"].split("-")
    return int(n[2]) if len(n) == 3 and n[2].isdigit() else 0


def split_on_dots(runs, min_side=4):
    """Split runs at PERSISTENT dot changes.  segment() only splits on
    the score cells, so a side-out or second-server rally — whose only
    on-bug effect is the serving dots moving — stays merged with its
    neighbor and can never be matched (measured: ~half the rallies are
    dots-only transitions).  A change is real when the new dot value
    holds for >= min_side consecutive samples on both sides."""
    out = []
    for r in runs:
        dots = r["dots"]
        # stable segments: value persisting >= min_side consecutively
        segs = []          # (start_idx, end_idx_exclusive, value)
        i = 0
        while i < len(dots):
            j = i
            while j < len(dots) and dots[j] == dots[i]:
                j += 1
            if j - i >= min_side:
                if segs and segs[-1][2] == dots[i]:
                    segs[-1] = (segs[-1][0], j, dots[i])
                else:
                    segs.append((i, j, dots[i]))
            elif segs:
                segs[-1] = (segs[-1][0], j, segs[-1][2])
            i = j
        if len(segs) <= 1:
            out.append(r)
            continue
        segs[0] = (0, segs[0][1], segs[0][2])
        segs[-1] = (segs[-1][0], len(dots), segs[-1][2])
        ts = r["ts"]
        for (a, b, val) in segs:
            if b - a < 2:
                continue
            piece = dict(r)
            piece["t0"] = ts[a]
            piece["t1"] = ts[b - 1]
            piece["ts"] = ts[a:b]
            piece["vecs"] = r["vecs"][a:b]
            piece["dots"] = dots[a:b]
            piece["lays"] = r["lays"][a:b]
            piece["dot"] = val
            out.append(piece)
    return out


def drop_interruptions(runs, thr, max_dur=10.0):
    """A -> B -> A: a short run whose NEIGHBORS match each other is an
    interruption (the bug cycling display modes in place, a replay
    sting, a presence flicker) — the surrounding state RESUMES, so B is
    deleted and A bridged.  Iterates to fixpoint; kills the rhythmic
    both-cells 'X' flood measured on the PPA Indoor bug (52/95 junk
    boundaries were in-place mode cycles)."""
    runs = list(runs)
    changed = True
    while changed:
        changed = False
        for i in range(1, len(runs) - 1):
            short = runs[i]["t1"] - runs[i]["t0"] <= max_dur
            if short and runs[i - 1].get("lay") == runs[i + 1].get("lay") \
                    and float(np.abs(runs[i - 1]["vec"]
                                     - runs[i + 1]["vec"]).mean()) < thr:
                a, c = runs[i - 1], runs[i + 1]
                a["t1"] = c["t1"]
                a["dots"] = a["dots"] + c["dots"]
                if "ts" in a and "ts" in c:
                    a["ts"] = a["ts"] + c["ts"]
                    a["lays"] = a["lays"] + c["lays"]
                dc = defaultdict(int)
                for nd in a["dots"]:
                    dc[nd] += 1
                a["dot"] = max(dc, key=dc.get)
                del runs[i:i + 2]
                changed = True
                break
    return runs


def change_symbols_video(runs, thr):
    """Symbol per consecutive-run boundary: which bug region changed.
    'T' = top score cell, 'B' = bottom, 'D' = dots only, '?' = none of
    them clearly (segmentation glitch).  Needs no digit recognition."""
    out = []
    for a, b in zip(runs, runs[1:]):
        dt = float(np.abs(a["vtop"] - b["vtop"]).mean()) > thr
        db = float(np.abs(a["vbot"] - b["vbot"]).mean()) > thr
        dd = a["dot"] != b["dot"]
        if dt and not db:
            out.append("T")
        elif db and not dt:
            out.append("B")
        elif dd and not dt and not db:
            out.append("D")
        else:
            out.append("X")     # both cells changed: game break / junk
    return out


def change_symbols_log(rallies, team_row):
    """The log's predicted boundary symbol per rally end: a POINT bumps
    the serving row's score cell; a side-out or second-server changes
    only the dots."""
    out = []
    for r in rallies:
        if r["outcome"] == "point":
            row = team_row[r["server_uuid"].lower()]
            out.append("T" if row == 0 else "B")
        else:
            out.append("D")
    return out


def align_symbols(sym_vid, sym_log):
    """Needleman on the two symbol strings; '?' matches anything at a
    small cost.  Returns log index -> video boundary index."""
    L, V = len(sym_log), len(sym_vid)
    GAP = 1.0
    D = np.full((L + 1, V + 1), 1e9)
    D[0, :] = np.arange(V + 1) * GAP
    D[:, 0] = np.arange(L + 1) * GAP
    P = np.zeros((L + 1, V + 1), np.int8)
    for i in range(1, L + 1):
        for j in range(1, V + 1):
            if sym_log[i - 1] == sym_vid[j - 1]:
                m = 0.0
            elif sym_vid[j - 1] == "X":
                m = 6.0        # never match a rally end to a game break
            else:
                m = 3.0
            best = (D[i - 1, j - 1] + m, 0)
            if D[i - 1, j] + GAP < best[0]:
                best = (D[i - 1, j] + GAP, 1)
            if D[i, j - 1] + GAP < best[0]:
                best = (D[i, j - 1] + GAP, 2)
            D[i, j], P[i, j] = best
    out = {}
    i, j = L, V
    while i > 0 and j > 0:
        if P[i, j] == 0:
            if sym_vid[j - 1] == sym_log[i - 1]:
                out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif P[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return out


def value_match(runs, rallies, team_row, thr):
    """ABSOLUTE matching via monotone digit identity — no global
    alignment to drift.  Each game's top/bottom score cells take a
    monotone non-decreasing sequence of values; leader-clustering the
    run cell-crops and ordering clusters by FIRST APPEARANCE yields
    each run's (top rank, bottom rank).  The log's per-row score
    sequences are monotone with known values, so rank k = the k-th
    distinct logged value, giving every run an absolute
    (top score, bottom score, dots) key that is UNIQUE within a game —
    rallies match by dict lookup, and a drifted assignment is
    impossible by construction.  Rallies whose state never appears
    (bug hidden all through) stay unmatched, honestly."""

    def ranks(vs, spans, n_expect):
        cents, out = [], []
        for v in vs:
            best, bd = -1, 1e18
            for i, c in enumerate(cents):
                d = float(np.abs(v - c[0] / c[1]).mean())
                if d < bd:
                    best, bd = i, d
            if best >= 0 and bd < thr:
                cents[best][0] += v
                cents[best][1] += 1
                out.append(best)
            else:
                cents.append([v.copy(), 1])
                out.append(len(cents) - 1)
        # TRUE score clusters occupy tight, pairwise-DISJOINT time
        # blocks (scores are monotone); junk clusters (score-change
        # flash animations, replay stings) recur across the game, so
        # their time range OVERLAPS the real blocks — measured: one
        # junk cluster spanned 2086-2162 s across four score states and
        # stole a first-appearance rank, shifting every later value.
        # Weighted interval scheduling by dwell keeps the disjoint
        # chain and drops anything that straddles it.
        info = {}
        for c, (t0, t1) in zip(out, spans):
            info.setdefault(c, []).append((t0, t1, t1 - t0))
        items = []
        for c, mem in info.items():
            mx = max(d for _, _, d in mem)
            core = [(a, b) for a, b, d in mem if d >= 0.15 * mx]
            lo = min(a for a, b in core)
            hi = max(b for a, b in core)
            items.append((lo, hi, sum(d for _, _, d in mem), c))
        items.sort(key=lambda it: (it[1], it[0]))
        n = len(items)
        dp = [0.0] * n
        par = [-1] * n
        took = [False] * n
        for i in range(n):
            lo, hi, w, c = items[i]
            bj, bw = -1, 0.0
            for j in range(i):
                if items[j][1] <= lo and dp[j] > bw:
                    bj, bw = j, dp[j]
            with_i = bw + w
            without = dp[i - 1] if i else 0.0
            dp[i] = max(with_i, without)
            took[i] = with_i >= without
            par[i] = bj
        sel = []
        i = n - 1
        while i >= 0:
            if took[i]:
                sel.append(i)
                i = par[i]
            else:
                i -= 1
        # one-off junk with a tight range can still slip in; real score
        # states dwell for whole rallies, so trim the shortest extras
        sel = sorted(sel, key=lambda i: -items[i][2])[:n_expect]
        chain = sorted(sel, key=lambda i: items[i][0])
        rank_of = {items[i][3]: k for k, i in enumerate(chain)}
        return [rank_of.get(c, -1) for c in out]

    spans = [(u["t0"], u["t1"]) for u in runs]
    # logged per-row score values, in rally order
    top_seq, bot_seq, keys = [], [], []
    for r in rallies:
        n = r["start_score"].split("-")
        x, y = int(n[0]), int(n[1])
        srv_row = team_row[r["server_uuid"].lower()]
        top = x if srv_row == 0 else y
        bot = y if srv_row == 0 else x
        top_seq.append(top)
        bot_seq.append(bot)
        keys.append((top, bot, srv_row, serve_sig(r)))
    dt = sorted(set(top_seq))         # distinct values in monotone order
    db = sorted(set(bot_seq))
    rt = ranks([u["vtop"] for u in runs], spans, len(dt))
    rb = ranks([u["vbot"] for u in runs], spans, len(db))
    out = {}
    for j, u in enumerate(runs):
        if not (0 <= rt[j] < len(dt) and 0 <= rb[j] < len(db)):
            continue
        row = 0 if u["dot"][0] > 0 else (1 if u["dot"][1] > 0 else -1)
        key = (dt[rt[j]], db[rb[j]], row, max(u["dot"]))
        for i, k in enumerate(keys):
            if k == key and rallies[i]["_cum"] not in out:
                out[rallies[i]["_cum"]] = u
                break
    return out


def align_game(runs, rallies, team_row):
    """Monotone Needleman alignment of video runs to log rallies on the
    (dots row, server number) signature.  Returns rally->run map."""
    L, V = len(rallies), len(runs)
    sig_log = []
    for r in rallies:
        row = team_row[r["server_uuid"].lower()]
        sig_log.append((row, serve_sig(r)))
    sig_vid = []
    for u in runs:
        d = u["dot"]
        row = 0 if d[0] > 0 else (1 if d[1] > 0 else -1)
        sig_vid.append((row, max(d)))
    GAP = 1.2
    D = np.full((L + 1, V + 1), 1e9)
    D[0, :] = np.arange(V + 1) * GAP
    D[:, 0] = np.arange(L + 1) * GAP
    P = np.zeros((L + 1, V + 1), np.int8)
    for i in range(1, L + 1):
        for j in range(1, V + 1):
            m = 0.0 if sig_log[i - 1] == sig_vid[j - 1] else 2.5
            best = (D[i - 1, j - 1] + m, 0)
            if D[i - 1, j] + GAP < best[0]:
                best = (D[i - 1, j] + GAP, 1)
            if D[i, j - 1] + GAP < best[0]:
                best = (D[i, j - 1] + GAP, 2)
            D[i, j], P[i, j] = best
    out = {}
    i, j = L, V
    while i > 0 and j > 0:
        if P[i, j] == 0:
            if sig_log[i - 1] == sig_vid[j - 1]:
                out[rallies[i - 1]["_cum"]] = runs[j - 1]
            i, j = i - 1, j - 1
        elif P[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return out


def finish(a, tl, wins, runs, thr):
    # PER-GAME ABSOLUTE VALUE MATCHING.  The layout index IS the game
    # (completed-game columns slide the live score column right), so
    # runs partition by lay; within a game every run gets an absolute
    # (top score, bottom score, serving row, server number) key via
    # monotone digit identity (value_match) — no global alignment to
    # drift.  Team partition by propagation: server and receiver are
    # ALWAYS opponents, so seeding one player two-colors everyone; the
    # bug row orientation is picked by match count over both options.
    players = {r["server_uuid"].lower() for r in tl} | \
              {r["receiver_uuid"].lower() for r in tl}
    side = {tl[0]["server_uuid"].lower(): 0}
    changed = True
    while changed:
        changed = False
        for r in tl:
            s = r["server_uuid"].lower()
            v = r["receiver_uuid"].lower()
            if s in side and v not in side:
                side[v] = 1 - side[s]
                changed = True
            elif v in side and s not in side:
                side[s] = 1 - side[v]
                changed = True
    runs = drop_interruptions(runs, thr)
    print(f"{len(runs)} runs after interruption bridging")
    runs = split_on_dots(runs)
    print(f"{len(runs)} runs after dot-change splitting")
    games = sorted({int(r["game"]) for r in tl})
    matched = {}
    for a_row in (0, 1):
        team_row = {p: (side.get(p, 0) + a_row) % 2 for p in players}
        mp = {}
        for gi, g in enumerate(games):
            rallies_g = [r for r in tl if int(r["game"]) == g]
            runs_g = [u for u in runs if u.get("lay", 0) == gi]
            mp.update(value_match(runs_g, rallies_g, team_row, thr))
        print(f"orientation {a_row}: {len(mp)}/{len(tl)} rallies matched")
        if len(mp) > len(matched):
            matched = mp
    # window per matched rally: the run SHOWS the rally's start score,
    # so the rally plays inside it; t1 = the boundary to the next run
    # in time.  A run that swallowed following states (missed dots-only
    # flip) would overrun — cap by the log duration.
    t_next = {}
    for u, v in zip(runs, runs[1:]):
        t_next[id(u)] = (u["t1"] + v["t0"]) / 2
    n_set = 0
    out_rows = []
    for w in wins:
        w = dict(w)
        cum = int(w["rally_cum"])
        u = matched.get(cum)
        if u is not None:
            t_end = t_next.get(id(u), u["t1"])
            dur = float(w.get("dur_s") or 0.0)
            if dur > 0:
                t_end = min(t_end, u["t0"] + dur + 12.0)
            w["t0s"] = f"{max(u['t0'] - 1.0, 0.0):.1f}"
            w["t1s"] = f"{t_end + 0.5:.1f}"
            w["approx"] = "0"
            w["state_check"] = "bugstate"
            n_set += 1
        else:
            w["approx"] = "1"
            w["state_check"] = "unmatched"
        out_rows.append(w)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        wr.writeheader()
        wr.writerows(out_rows)
    print(f"windows set from bug state: {n_set}/{len(out_rows)} -> {a.out}")


def run_main(a):
    tl = list(csv.DictReader(open(a.timeline)))
    wins = list(csv.DictReader(open(a.windows)))
    for i, r in enumerate(tl):
        r["_cum"] = i + 1
    # team row on the bug: row 0 = team listed first.  The lineup teams:
    # bug row is resolved by majority vote later; start with server ->
    # candidate rows from BOTH assignments and pick the consistent one.
    servers = sorted({r["server_uuid"].lower() for r in tl})
    cache = Path(str(a.out) + ".crops3.npz")
    if cache.exists():
        z = np.load(cache)
        ts, boxes = z["t"], z["box"]
        gray, gmask = z["gray"], z["gmask"]
        print(f"crops from cache ({len(ts)})")
    else:
        print("streaming bug crops at 2 Hz (one pass)...")
        ts_l, box_l, gray_l, gm_l = [], [], [], []
        for t, fr in stream_bug(a.video):
            gm = green_mask(fr)
            b = box_of(gm)
            if b is not None and not (REG_Y0 <= b[0] and b[1] <= REG_Y1
                                      and REG_X0 <= b[2] and b[3] <= REG_X1):
                b = None
            ts_l.append(t)
            box_l.append(b if b is not None else (-1, -1, -1, -1))
            gray_l.append(fr[REG_Y0:REG_Y1, REG_X0:REG_X1]
                          .mean(axis=2).astype(np.uint8))
            gm_l.append(gm[REG_Y0:REG_Y1, REG_X0:REG_X1])
        ts = np.array(ts_l)
        boxes = np.array(box_l, np.int16)
        gray = np.stack(gray_l)
        gmask = np.stack(gm_l)
        del ts_l, box_l, gray_l, gm_l
        np.savez_compressed(cache, t=ts, box=boxes, gray=gray, gmask=gmask)
        print(f"cached {len(ts)} crops -> {cache}")
    parsed = states_from_crops(ts, boxes, gray, gmask)
    ds = [float(np.abs(parsed[i][1] - parsed[i + 1][1]).mean())
          for i in range(200, min(1200, len(parsed) - 1))
          if parsed[i][2] and parsed[i + 1][2]]
    # floor 3.2: within-run scatter tops out ~2.5 (canonical boxes) and
    # subtle digit changes (8->9) sit at 3.6-5 — the old 5.0 missed them
    thr = max(4.0 * float(np.median(ds)), 3.2)
    print(f"state threshold {thr:.1f}")
    runs = segment(parsed, thr)
    print(f"{len(runs)} constant-state runs")
    return finish(a, tl, wins, runs, thr)




def selftest():
    fr = np.zeros((40, 100, 3), np.uint8)
    fr[:, :, :] = 30
    fr[5:18, 70:90, 1] = 200      # green cell row 0
    fr[22:35, 70:90, 1] = 200     # green cell row 1
    fr[8:14, 22:25, 1] = 220      # one dot row 0
    fr[8:14, 28:31, 1] = 220      # second dot row 0
    cells, dots = locate_cells([fr] * 8)
    vec, present, nd = frame_state(fr, cells, dots)
    assert present and nd[0] == 2 and nd[1] == 0, (present, nd)
    fr2 = fr.copy()
    fr2[5:18, 70:90, 1] = 120     # score changed (dimmer digits region)
    v2, _, _ = frame_state(fr2, cells, dots)
    assert float(np.abs(vec - v2).mean()) > 5
    states = [(i * 0.5, vec if i < 10 else v2, True, nd) for i in range(20)]
    runs = segment(states, thr=5.0)
    assert len(runs) == 2 and abs(runs[0]["t1"] - 4.5) < 0.6, \
        [(r["t0"], r["t1"]) for r in runs]
    print("SELFTEST OK (cells, dots, presence, segmentation)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path)
    ap.add_argument("--timeline", type=Path)
    ap.add_argument("--windows", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    for req in ("video", "timeline", "windows", "out"):
        if not getattr(a, req):
            ap.error(f"--{req} required")
    run_main(a)


if __name__ == "__main__":
    main()
