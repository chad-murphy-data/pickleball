"""Extract full-skeleton pose streams for labeled rally windows (Gate C).

vision/swing_probe.py read ONE number off the pose model (absolute wrist
speed) and threw the skeleton away; the Gate B autopsy blamed exactly
that (locomotion contamination, frame-local side shredding). This script
saves the WHOLE stream — all 17 COCO keypoints + confidences + boxes,
on identity-continuous tracks with per-rally near/far side assignment —
so features can be recomputed forever without re-decoding video, and the
contact ceiling (vision/contact_ceiling.py) and any eventual trainer eat
the same substrate.

Two of the three v2-instrument pillars live here:
  * identity-continuous tracking — greedy IoU association across frames,
    a track survives detection gaps up to GAP_S; no frame-local slots;
  * per-rally near/far by track mean BOX HEIGHT clustering — the repaired
    side method from the autopsy (frame-local bottom-y splitting is what
    corrupted 42% of far labels).
Torso-relative wrist velocity (the third pillar) is computed downstream
by contact_ceiling.py from these arrays.

Windows come from the CONTACT LABELS themselves (the stamped serve of
each labeled rally + the v4 window rule), so extraction covers exactly
the cases being scored — the measurement-frame lesson, kept by
construction. Unlabeled rallies can still be extracted via --windows.

RUN (needs the VOD; GPU ~2-4 s/rally, CPU with --fast ~1-2 min/rally)
    pip install ultralytics imageio-ffmpeg
    python vision/pose_extract.py --video full_match.mp4 --rallies labeled
    # CPU: add --fast   (yolov8n-pose, 640 px, 20 fps)

Outputs (data/vision/pose/ is gitignored — regenerable from the VOD):
    data/vision/pose/r0001.npz ...   one per rally
    data/vision/pose/meta.json      params + per-rally counts

SELF-TEST (no torch, no video)
    python vision/pose_extract.py --selftest
    Synthetic moving boxes with a within-side crossing and a dropout gap:
    asserts identity survives both, sides come out right, windows follow
    the v4 rule, and the npz round-trips exactly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from swing_probe import ffmpeg_bin, decode_window, court_halfwidth
from serve_pin_windows import PAD_PRE, GAP_POST, CADENCE, TAIL

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data/vision/contact_labels_chicago0725.csv"
WINDOWS_V4 = ROOT / "data/vision/rally_windows_chicago0725_v4.csv"
OUT_DIR = ROOT / "data/vision/pose"

IOU_MIN = 0.15        # association floor
GAP_S = 0.6           # a track survives a detection gap up to this
MIN_TRACK_DET = 5     # shorter tracks get side=-1 (junk fragments)
MAX_PERSONS = 6       # per frame, by detection confidence


# ------------------------------------------------------------- windows


def windows_from_labels(path: Path):
    """Rally windows from stamped serves, v4 rule: t0 = serve - PAD_PRE,
    t1 = min(next_serve - GAP_POST, serve + CADENCE*n_shots + TAIL); the
    next-serve bound applies only when the NEXT LABELED rally is cum+1
    (a gap in labeling means unlabeled rallies sit between)."""
    serves, n_shots = {}, {}
    for r in csv.DictReader(open(path)):
        if r.get("contact", "1") == "0":
            continue
        cum = int(r["rally_cum"])
        t = float(r["t_refined_s"] or r["t_tap_s"])
        serves[cum] = min(serves.get(cum, math.inf), t)
        n_shots[cum] = n_shots.get(cum, 0) + 1
    win = {}
    for cum in sorted(serves):
        sv = serves[cum]
        cap = sv + CADENCE * n_shots[cum] + TAIL
        nxt = serves.get(cum + 1)
        t1 = min(nxt - GAP_POST, cap) if nxt is not None else cap
        win[cum] = (sv - PAD_PRE, max(t1, sv + 2.0))
    return win


def windows_from_v4(path: Path):
    win = {}
    for r in csv.DictReader(open(path)):
        win[int(r["rally_cum"])] = (float(r["t0s"]), float(r["t1s"]))
    return win


# ------------------------------------------------------------- tracker


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ar_a = (a[2] - a[0]) * (a[3] - a[1])
    ar_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ar_a + ar_b - inter)


class IoUTracker:
    """Greedy IoU association on CONSTANT-VELOCITY-predicted boxes (the
    SORT insight, no Kalman needed at 4 players). Plain last-box IoU
    swaps identities when two same-size boxes pass through each other —
    the selftest plants exactly that crossing and plain IoU fails it at
    a 55/45 split; predicted boxes separate by direction of motion and
    keep identity. Prediction horizon is clamped (PRED_CLAMP_S) so a
    stale velocity cannot fling a box across the court during a
    detection gap."""

    PRED_CLAMP_S = 0.2
    VEL_EMA = 0.5

    def __init__(self, gap_s=GAP_S):
        self.gap_s = gap_s
        self.next_id = 0
        self.live = {}        # id -> (t, box, vel)

    def feed(self, t, boxes):
        """boxes: (N,4) array. Returns a track id per box."""
        self.live = {i: v for i, v in self.live.items()
                     if t - v[0] <= self.gap_s}
        preds = {i: b + v * min(t - tl, self.PRED_CLAMP_S)
                 for i, (tl, b, v) in self.live.items()}
        ids = [-1] * len(boxes)
        pairs = [(iou(p, b), i, k)
                 for i, p in preds.items()
                 for k, b in enumerate(boxes)]
        pairs.sort(key=lambda x: -x[0])
        used_t, used_d = set(), set()
        for ov, i, k in pairs:
            if ov < IOU_MIN or i in used_t or k in used_d:
                continue
            ids[k] = i
            used_t.add(i)
            used_d.add(k)
        for k in range(len(boxes)):
            if ids[k] == -1:
                ids[k] = self.next_id
                self.next_id += 1
        for k, b in enumerate(boxes):
            b = np.asarray(b, float)
            prev = self.live.get(ids[k])
            vel = np.zeros(4)
            if prev is not None and t > prev[0]:
                inst = (b - prev[1]) / (t - prev[0])
                vel = self.VEL_EMA * prev[2] + (1 - self.VEL_EMA) * inst
            self.live[ids[k]] = (t, b, vel)
        return ids


def assign_sides(track_ids, boxes):
    """track_id -> side (0 near / 1 far / -1 junk) by mean box-height
    max-gap clustering over tracks — the repaired method from the Gate B
    autopsy; the near pair images ~2x taller than the far pair."""
    hs = {}
    for tid, b in zip(track_ids, boxes):
        hs.setdefault(tid, []).append(b[3] - b[1])
    means = {tid: float(np.mean(v)) for tid, v in hs.items()
             if len(v) >= MIN_TRACK_DET}
    side = {tid: -1 for tid in hs}
    if len(means) >= 2:
        vals = np.sort(np.array(list(means.values())))
        gaps = vals[1:] - vals[:-1]
        i = int(np.argmax(gaps))
        thr = (vals[i] + vals[i + 1]) / 2
        if gaps[i] > 0.12 * np.median(vals):      # a real near/far gap
            for tid, m in means.items():
                side[tid] = 0 if m >= thr else 1
    return side


# ------------------------------------------------------------- filter


def keep_boxes(res, H, W):
    """Court players only — feet band + height + perspective court gate
    (reused from swing_probe) — but NO wrist-confidence gate: keypoint
    confidences are features now, not filters."""
    out = []
    if res.keypoints is None or not len(res.boxes):
        return out
    kxy = res.keypoints.xy.cpu().numpy()
    kcf = (res.keypoints.conf.cpu().numpy()
           if res.keypoints.conf is not None
           else np.ones(kxy.shape[:2], np.float32))
    confs = res.boxes.conf.cpu().numpy()
    for box, kps, kpc, cf in zip(res.boxes.xyxy.cpu().numpy(),
                                 kxy, kcf, confs):
        x0, y0, x1, y1 = box[:4]
        bh, cx = y1 - y0, (x0 + x1) / 2
        if y1 < 0.30 * H or bh < 0.06 * H:
            continue
        if abs(cx / W - 0.5) > court_halfwidth(y1 / H):
            continue
        out.append((float(cf), box[:4].astype(np.float32),
                    kps.astype(np.float32), kpc.astype(np.float32)))
    out.sort(key=lambda x: -x[0])
    return out[:MAX_PERSONS]


# ------------------------------------------------------------- extract


def extract_rally(model, video, cum, t0, t1, fps, width, imgsz, device):
    trk = IoUTracker()
    rows = []          # (t, tid, conf, box, kpt, kpc)
    H = W = None
    for i, frame in enumerate(decode_window(video, t0, t1 - t0, fps, width)):
        t = t0 + i / fps
        H, W = frame.shape[:2]
        res = model(frame, imgsz=imgsz, verbose=False,
                    device=(device or None), conf=0.30)[0]
        kept = keep_boxes(res, H, W)
        ids = trk.feed(t, [k[1] for k in kept])
        for tid, (cf, box, kpt, kpc) in zip(ids, kept):
            rows.append((t, tid, cf, box, kpt, kpc))
    side = assign_sides([r[1] for r in rows], [r[3] for r in rows]) \
        if rows else {}
    return rows, side, (H, W)


def save_rally(out_dir: Path, cum, rows, side, hw, fps):
    n = len(rows)
    d = {
        "t": np.array([r[0] for r in rows], np.float64),
        "track": np.array([r[1] for r in rows], np.int32),
        "side": np.array([side.get(r[1], -1) for r in rows], np.int8),
        "conf": np.array([r[2] for r in rows], np.float32),
        "box": (np.stack([r[3] for r in rows])
                if n else np.zeros((0, 4), np.float32)),
        "kpt": (np.stack([r[4] for r in rows])
                if n else np.zeros((0, 17, 2), np.float32)),
        "kpc": (np.stack([r[5] for r in rows])
                if n else np.zeros((0, 17), np.float32)),
        "hw": np.array(hw if hw[0] else (0, 0), np.int32),
        "fps": np.array([fps], np.float64),
    }
    np.savez_compressed(out_dir / f"r{cum:04d}.npz", **d)
    return n


def run(a):
    from ultralytics import YOLO          # lazy: --selftest needs no torch

    labels_win = windows_from_labels(a.labels) if Path(a.labels).exists() else {}
    v4_win = windows_from_v4(a.windows) if Path(a.windows).exists() else {}
    if a.rallies == "labeled":
        todo = sorted(labels_win)
    elif a.rallies:
        todo = sorted(int(x) for x in a.rallies.split(","))
    else:
        todo = sorted(labels_win) or sorted(v4_win)
    if not todo:
        raise SystemExit("nothing to extract: no labels found and no "
                         "--rallies given")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {int(p.stem[1:]) for p in out_dir.glob("r*.npz")}
    todo = [c for c in todo if c not in done or a.force]
    print(f"{len(todo)} rallies to extract "
          f"({len(labels_win)} label-windowed, {len(done)} already done)")

    model = YOLO(a.model)
    t_start = time.time()
    counts = {}
    for k, cum in enumerate(todo):
        if cum in labels_win:
            t0, t1 = labels_win[cum]
            src = "labels"
        elif cum in v4_win:
            t0, t1 = v4_win[cum]
            src = "v4"
        else:
            print(f"rally #{cum}: no window anywhere — skipped")
            continue
        rows, side, hw = extract_rally(model, a.video, cum, t0, t1,
                                       a.fps, a.width, a.imgsz, a.device)
        n = save_rally(out_dir, cum, rows, side, hw, a.fps)
        n_tracks = len({r[1] for r in rows})
        n_sided = len({tid for tid, s in side.items() if s >= 0})
        counts[cum] = {"detections": n, "tracks": n_tracks,
                       "sided_tracks": n_sided, "window": src,
                       "t0": round(t0, 2), "t1": round(t1, 2)}
        el = time.time() - t_start
        eta = (len(todo) - k - 1) * el / (k + 1) / 60
        print(f"rally #{cum:>3} ({k + 1}/{len(todo)})  [{src}] "
              f"{t1 - t0:5.1f}s  {n:5d} det  {n_tracks:2d} trk "
              f"({n_sided} sided)  eta {eta:.0f} min", flush=True)

    meta = {"video": str(a.video), "model": a.model, "imgsz": a.imgsz,
            "fps": a.fps, "width": a.width, "device": a.device or "auto",
            "iou_min": IOU_MIN, "gap_s": GAP_S,
            "runtime_s": round(time.time() - t_start, 1), "rallies": counts}
    old = {}
    mp = out_dir / "meta.json"
    if mp.exists():
        old = json.loads(mp.read_text()).get("rallies", {})
    old.update({str(k): v for k, v in counts.items()})
    meta["rallies"] = old
    mp.write_text(json.dumps(meta, indent=1))
    print(f"done in {(time.time() - t_start) / 60:.1f} min. "
          f"Next: python vision/contact_ceiling.py")


# ------------------------------------------------------------ selftest


def selftest():
    import tempfile

    rng = np.random.default_rng(11)
    fps = 30.0
    T = int(20 * fps)

    # 4 synthetic players: near pair (tall) crossing at 8-10 s, far pair
    # (short) crossing at 12-14 s, far-left dropped out 5.0-5.4 s.
    def path(cx0, cx1, tc, dur):
        def f(t):
            if t < tc:
                return cx0
            if t > tc + dur:
                return cx1
            return cx0 + (cx1 - cx0) * (t - tc) / dur
        return f

    players = [  # (cx(t), bottom, box_h, dropout)
        (path(420, 860, 8.0, 2.0), 600.0, 240.0, None),
        (path(860, 420, 8.0, 2.0), 600.0, 240.0, None),
        (path(500, 760, 12.0, 2.0), 380.0, 120.0, (5.0, 5.4)),
        (path(760, 500, 12.0, 2.0), 380.0, 120.0, None),
    ]
    trk = IoUTracker()
    ids_by_player = [dict() for _ in players]
    seq_by_player = [[] for _ in players]
    all_ids, all_boxes = [], []
    for i in range(T):
        t = i / fps
        boxes, who = [], []
        for pi, (cxf, bot, bh, drop) in enumerate(players):
            if drop and drop[0] <= t <= drop[1]:
                continue
            cx = cxf(t) + rng.normal(0, 1.5)
            bw = bh * 0.45
            boxes.append([cx - bw / 2, bot - bh + rng.normal(0, 1.5),
                          cx + bw / 2, bot + rng.normal(0, 1.5)])
            who.append(pi)
        ids = trk.feed(t, np.array(boxes))
        for pi, tid in zip(who, ids):
            ids_by_player[pi][tid] = ids_by_player[pi].get(tid, 0) + 1
            seq_by_player[pi].append((i, tid))
        all_ids.extend(ids)
        all_boxes.extend(boxes)

    # At the EXACT coincidence frame of a symmetric crossing the two boxes
    # are identical and assignment is unresolvable from geometry — the
    # tracker blips for one frame and self-recovers (measured: exactly one
    # stray frame per crossing player, swap-back on the next frame). The
    # claims that matter downstream: identity is PERSISTENT through the
    # crossing (same track before and after), blips are bounded, and a
    # sub-GAP_S detection gap does not split or re-badge a track.
    seqs = {}
    for pi, cnt in enumerate(ids_by_player):
        seq = seq_by_player[pi]
        dom = max(cnt.values()) / sum(cnt.values())
        first = max(set(s for _, s in seq[:60]),
                    key=[s for _, s in seq[:60]].count)
        last = max(set(s for _, s in seq[-60:]),
                   key=[s for _, s in seq[-60:]].count)
        stray = sum(cnt.values()) - max(cnt.values())
        print(f"  player {pi}: {len(cnt)} track(s), dominant {dom:.1%}, "
              f"{stray} blip frame(s)")
        assert dom >= 0.98, f"identity fragmented for player {pi}"
        assert first == last, f"player {pi}: crossing swapped identity"
        assert stray <= 3, f"player {pi}: too many blip frames"
        seqs[pi] = seq
    # the dropout gap (0.4 s < GAP_S) must reacquire the SAME track
    seq2 = seqs[2]
    before = [s for i, s in seq2 if i / fps < 5.0][-1]
    after = [s for i, s in seq2 if i / fps > 5.4][0]
    assert before == after, "0.4s dropout re-badged the track"

    side = assign_sides(all_ids, [np.array(b) for b in all_boxes])
    for pi, cnt in enumerate(ids_by_player):
        dom_tid = max(cnt, key=cnt.get)
        want = 0 if pi < 2 else 1
        assert side[dom_tid] == want, \
            f"player {pi}: side {side[dom_tid]} != {want}"
    print("  sides: near/far by height clustering all correct")

    # ---- windows_from_labels: v4 rule incl. the gap case --------------
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "labels.csv"
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rally_cum", "shot_index", "contact",
                        "t_tap_s", "t_refined_s"])
            for i in range(1, 13):                # rally 1: 12 shots, so the
                w.writerow([1, i, 1, 100 + i * 1.5, ""])   # next-serve bound
            w.writerow([1, 13, 0, 103.2, ""])     # binds; whiff: no effect
            for i in range(1, 3):                 # rally 2: consecutive
                w.writerow([2, i, 1, 120 + i, ""])
            w.writerow([7, 1, 1, 300.0, 299.8])   # gapped; refined wins
        win = windows_from_labels(p)
        assert abs(win[1][0] - (101.5 - PAD_PRE)) < 1e-9
        assert abs(win[1][1] - (121.0 - GAP_POST)) < 1e-9, \
            "consecutive rally must bound by next serve"
        assert abs(win[2][1] - (121.0 + CADENCE * 2 + TAIL)) < 1e-9, \
            "gapped rally must use the cadence cap, not rally 7's serve"
        assert abs(win[7][0] - (299.8 - PAD_PRE)) < 1e-9, "refined time wins"
        print("  windows: v4 rule + gap handling + refined-time priority OK")

        # ---- npz round trip ------------------------------------------
        rows = [(i / fps, all_ids[i], 0.9,
                 np.array(all_boxes[i], np.float32),
                 rng.normal(size=(17, 2)).astype(np.float32),
                 rng.random(17).astype(np.float32)) for i in range(50)]
        n = save_rally(Path(td), 3, rows, side, (720, 1280), fps)
        z = np.load(Path(td) / "r0003.npz")
        assert n == 50 and z["kpt"].shape == (50, 17, 2)
        assert np.allclose(z["box"][7], all_boxes[7])
        assert z["side"][0] == side[all_ids[0]]
        print("  npz round trip OK")
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path)
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--windows", default=str(WINDOWS_V4),
                    help="fallback windows for rallies without labels")
    ap.add_argument("--rallies", default="labeled",
                    help="'labeled' (default) or comma-separated rally_cum")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--model", default="yolov8s-pose.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--device", default="")
    ap.add_argument("--force", action="store_true",
                    help="re-extract rallies that already have an npz")
    ap.add_argument("--fast", action="store_true",
                    help="CPU preset: yolov8n-pose, imgsz 640, 20 fps")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.fast:
        a.model, a.imgsz, a.fps = "yolov8n-pose.pt", 640, 20.0
    if not a.video:
        raise SystemExit("--video required (or --selftest)")
    run(a)


if __name__ == "__main__":
    main()
