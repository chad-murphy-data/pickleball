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


def frame_state(fr, cells, dots):
    """(cell pixels vector, present flag, dots per row)."""
    gm = green_mask(fr)
    present = gm[cells[0][0]:cells[1][1], cells[0][2]:cells[0][3]].mean() > 0.35
    vecs = []
    for (ya, yb, xa, xb) in cells:
        cell = fr[ya:yb, xa:xb].mean(axis=2)
        hh, ww = cell.shape
        sub = cell[:hh - hh % 4, :ww - ww % 4]
        sub = sub.reshape(hh // 4, 4, ww // 4, 4).mean(axis=(1, 3))
        vecs.append(sub.ravel())
    nd = []
    for (ya, yb, xa, xb) in dots:
        dm = gm[ya:yb, xa:xb]
        col = dm.sum(axis=0) >= max(2, (yb - ya) * 0.12)
        nd.append(int(np.diff(np.pad(col.astype(int), 1)).clip(0).sum()))
    return np.concatenate(vecs), bool(present), tuple(nd)


def segment(states, thr):
    """Constant-state runs: [(t_start, t_end, med_vec, dots)]."""
    runs = []
    cur = None
    for t, vec, present, nd in states:
        if not present:
            continue
        if cur and float(np.abs(vec - cur["ref"]).mean()) < thr:
            cur["t1"] = t
            cur["vecs"].append(vec)
            cur["dots"].append(nd)
        else:
            if cur and len(cur["vecs"]) >= 2:
                runs.append(cur)
            cur = {"t0": t, "t1": t, "ref": vec, "vecs": [vec], "dots": [nd]}
    if cur and len(cur["vecs"]) >= 2:
        runs.append(cur)
    for r in runs:
        r["vec"] = np.median(np.stack(r["vecs"]), axis=0)
        dc = defaultdict(int)
        for nd in r["dots"]:
            dc[nd] += 1
        r["dot"] = max(dc, key=dc.get)
    # merge consecutive runs whose MEDIAN states match (brief occlusions)
    merged = [runs[0]] if runs else []
    for r in runs[1:]:
        if float(np.abs(r["vec"] - merged[-1]["vec"]).mean()) < thr \
                and r["dot"] == merged[-1]["dot"]:
            merged[-1]["t1"] = r["t1"]
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


def run_main(a):
    tl = list(csv.DictReader(open(a.timeline)))
    wins = list(csv.DictReader(open(a.windows)))
    for i, r in enumerate(tl):
        r["_cum"] = i + 1
    # team row on the bug: row 0 = team listed first.  The lineup teams:
    # bug row is resolved by majority vote later; start with server ->
    # candidate rows from BOTH assignments and pick the consistent one.
    servers = sorted({r["server_uuid"].lower() for r in tl})
    print("sampling bug states at 2 Hz (one pass)...")
    # locator reference frames from MID-RALLY times (the bug is provably
    # up there; sampling early video medians the cells away while venue
    # turf survives — measured failure)
    mids = sorted(float(w["t1s"]) - 3.0 for w in wins)
    wanted = set()
    for k in np.linspace(0, len(mids) - 1, 40).astype(int):
        wanted.add(round(mids[k] * FPS) / FPS)
    sample, states = [], []
    for t, fr in stream_bug(a.video):
        if round(t * FPS) / FPS in wanted:
            sample.append(fr)
        states.append((t, fr))
    cells, dots = locate_cells(sample if len(sample) >= 8 else
                               [f for _, f in states[::311]][:40])
    print(f"cells {cells} dots {dots}")
    parsed = []
    for t, fr in states:
        vec, present, nd = frame_state(fr, cells, dots)
        parsed.append((t, vec, present, nd))
    del states
    # noise threshold from consecutive same-state samples
    ds = [float(np.abs(parsed[i][1] - parsed[i + 1][1]).mean())
          for i in range(200, min(1200, len(parsed) - 1))
          if parsed[i][2] and parsed[i + 1][2]]
    thr = max(4.0 * float(np.median(ds)), 5.0)
    print(f"state threshold {thr:.1f}")
    runs = segment(parsed, thr)
    print(f"{len(runs)} constant-state runs")

    # games from the timeline; try both team->row mappings per game and
    # keep the one that matches more rallies
    by_game = defaultdict(list)
    for r in tl:
        by_game[r["game"]].append(r)
    out_map = {}
    for g, rallies in by_game.items():
        t_lo = min(float(w["t0s"]) for w in wins if w["game"] == g) - 120
        t_hi = max(float(w["t1s"]) for w in wins if w["game"] == g) + 120
        gruns = [u for u in runs if u["t0"] >= t_lo and u["t1"] <= t_hi]
        # team partition by propagation: server and receiver are ALWAYS
        # opponents, so seeding one player and sweeping the rally list
        # (both directions, until fixpoint) two-colors everyone
        players = {r["server_uuid"].lower() for r in rallies} | \
                  {r["receiver_uuid"].lower() for r in rallies}
        side = {rallies[0]["server_uuid"].lower(): 0}
        changed = True
        while changed:
            changed = False
            for r in rallies:
                s = r["server_uuid"].lower()
                v = r["receiver_uuid"].lower()
                if s in side and v not in side:
                    side[v] = 1 - side[s]
                    changed = True
                elif v in side and s not in side:
                    side[s] = 1 - side[v]
                    changed = True
        best = {}
        for a_row in (0, 1):
            tr = {p: (side.get(p, 0) + a_row) % 2 for p in players}
            m = align_game(gruns, rallies, tr)
            if len(m) > len(best):
                best = m
        out_map.update(best)
        print(f"game {g}: matched {len(best)}/{len(rallies)} rallies "
              f"({len(gruns)} runs in span)")

    n_set = 0
    out_rows = []
    for w in wins:
        w = dict(w)
        cum = int(w["rally_cum"])
        u = out_map.get(cum)
        if u is not None:
            t1 = u["t1"] + 0.5 / FPS - FLIP_LAG_S + 1.0
            dur = float(w["dur_s"])
            w["t1s"] = f"{u['t1'] + 0.75:.1f}"
            w["t0s"] = f"{max(u['t0'] - 1.5, 0.0):.1f}"
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
