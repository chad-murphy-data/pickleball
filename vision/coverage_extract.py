"""Coverage-owned pose extraction: pose_extract's machinery + a court
pre-filter, WITHOUT touching the frozen Gate C instrument.

Why this exists (found on the first real VOD, PPA Indoor Nationals
2026-01): that broadcast's main angle shows front-row CROWD at the
bottom edge of frame.  Crowd bodies image large and confident, so
pose_extract's keep-top-6-persons cap hands them detection slots and
the far-court players — the smallest, faintest persons on screen —
lose theirs: rally 84 kept ~1 far-court detection per frame.  The cap
must be applied AFTER crowd is filtered, and the filter needs the court
homography, which the contact-thread extractor has no business knowing.
pose_extract.py is mid-measurement (contact_gate.md) and is imported,
not edited.

Filters, applied per person BEFORE the cap:
  * frame-edge truncation: box bottom within 6 px of the frame's bottom
    edge AND both ankles unconfident — a body cut off by the frame is
    front-row crowd; a real server behind the baseline images fully,
    feet visible;
  * court bounds: projected foot outside x in [-4, 24] or y in [-7, 51]
    (court + serve-stance slack) — spectators along the sides, staff.
Cap raised to 8 (fragments cost little; lost players cost everything).

    python vision/coverage_extract.py --video vod.mp4 \
        --windows data/vision/coverage_windows_<vod>.csv \
        --court data/vision/court_<vod>.json \
        --out-dir data/vision/pose_<vod> [--backend rtmpose --fps 10]
    python vision/coverage_extract.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coverage import foot_point, load_court, project
from pose_extract import IoUTracker, assign_sides, make_infer, save_rally
from swing_probe import decode_window

MAX_KEEP = 8
EDGE_PX = 6
X_LO, X_HI = -4.0, 24.0
Y_LO, Y_HI = -7.0, 51.0


def court_filter(persons, court, H, W):
    """Drop frame-edge-truncated bodies and off-court projections
    BEFORE any confidence cap.  persons: (conf, box, kpt, kpc) tuples
    from pose_extract's infer.  Returns (kept, n_edge, n_off)."""
    kept, n_edge, n_off = [], 0, 0
    for cf, box, kpt, kpc in persons:
        if box[3] >= H - EDGE_PX and kpc[15] < 0.3 and kpc[16] < 0.3:
            n_edge += 1
            continue
        fx, fy, _src = foot_point(box, kpt, kpc)
        cx, cy = project(court, [(fx, fy)], W, H)[0]
        if not (X_LO <= cx <= X_HI and Y_LO <= cy <= Y_HI):
            n_off += 1
            continue
        kept.append((cf, box, kpt, kpc))
    kept.sort(key=lambda p: -p[0])
    return kept[:MAX_KEEP], n_edge, n_off


def extract(a):
    court = load_court(a.court)
    import csv
    wins = {int(r["rally_cum"]): (float(r["t0s"]), float(r["t1s"]))
            for r in csv.DictReader(open(a.windows))}
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {int(p.stem[1:]) for p in out_dir.glob("r*.npz")}
    todo = [c for c in sorted(wins) if c not in done or a.force]
    infer, backend = make_infer(a)
    print(f"pose backend: {backend} + court pre-filter; "
          f"{len(todo)} rallies to extract")
    t_start = time.time()
    counts = {}
    for k, cum in enumerate(todo):
        t0, t1 = wins[cum]
        trk = IoUTracker()
        rows = []
        Hf = Wf = None
        edge = off = 0
        for i, frame in enumerate(decode_window(a.video, t0, t1 - t0,
                                                a.fps, a.width)):
            t = t0 + i / a.fps
            Hf, Wf = frame.shape[:2]
            persons = infer(frame, Hf, Wf)
            persons, ne, no = court_filter(persons, court, Hf, Wf)
            edge += ne
            off += no
            ids = trk.feed(t, [p[1] for p in persons])
            for tid, (cf, box, kpt, kpc) in zip(ids, persons):
                rows.append((t, tid, cf, box, kpt, kpc))
        side = assign_sides([r[1] for r in rows], [r[3] for r in rows]) \
            if rows else {}
        n = save_rally(out_dir, cum, rows, side, (Hf, Wf) if Hf else (0, 0),
                       a.fps)
        counts[cum] = {"detections": n, "edge_dropped": edge,
                       "offcourt_dropped": off,
                       "t0": round(t0, 2), "t1": round(t1, 2)}
        el = time.time() - t_start
        eta = (len(todo) - k - 1) * el / (k + 1) / 60
        print(f"rally #{cum:>3} ({k + 1}/{len(todo)}) {t1 - t0:5.1f}s "
              f"{n:5d} det (edge-drop {edge}, off {off})  "
              f"eta {eta:.0f} min", flush=True)
    meta = {"video": str(a.video), "backend": backend, "fps": a.fps,
            "width": a.width, "court": str(a.court),
            "filter": {"max_keep": MAX_KEEP, "edge_px": EDGE_PX,
                       "x": [X_LO, X_HI], "y": [Y_LO, Y_HI]},
            "runtime_s": round(time.time() - t_start, 1),
            "rallies": {str(c): v for c, v in counts.items()}}
    mp = out_dir / "meta.json"
    old = json.loads(mp.read_text()).get("rallies", {}) if mp.exists() else {}
    old.update(meta["rallies"])
    meta["rallies"] = old
    mp.write_text(json.dumps(meta, indent=1))
    print(f"done in {(time.time() - t_start) / 60:.1f} min")


def selftest():
    from coverage import synth_court
    court = synth_court()          # 30 px/ft, origin (100, 50), 1280x1440
    H, W = 1440, 1280

    def person(cx_ft, cy_ft, box_h=200.0, ankles=True, bottom=None):
        x = 100 + cx_ft * 30
        y = 50 + cy_ft * 30
        box = np.array([x - 40, (bottom or y) - box_h, x + 40,
                        bottom or y], np.float32)
        kpt = np.zeros((17, 2), np.float32)
        kpc = np.zeros(17, np.float32)
        if ankles:
            kpt[15] = (x - 8, y)
            kpt[16] = (x + 8, y)
            kpc[15] = kpc[16] = 0.9
        return (0.9, box, kpt, kpc)

    persons = [person(10, 30),                     # mid court: keep
               person(5, 46.0),                    # server behind line: keep
               person(10, 60),                     # deep crowd zone: off
               (0.95, np.array([600, H - 180, 700, H - 2], np.float32),
                np.zeros((17, 2), np.float32),
                np.zeros(17, np.float32)),         # frame-edge, no ankles
               ]
    kept, n_edge, n_off = court_filter(persons, court, H, W)
    assert len(kept) == 2 and n_edge == 1 and n_off == 1, \
        (len(kept), n_edge, n_off)
    # a server at the frame edge WITH visible ankles is kept
    srv = person(5, 46.0)
    srv[1][3] = H - 2
    kept2, ne2, _ = court_filter([srv], court, H, W)
    assert len(kept2) == 1 and ne2 == 0
    # cap: 10 on-court persons -> best MAX_KEEP survive
    many = [person(2 + i * 1.5, 30) for i in range(10)]
    kept3, _, _ = court_filter(many, court, H, W)
    assert len(kept3) == MAX_KEEP
    print("SELFTEST OK (edge-truncation, court bounds, cap)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path)
    ap.add_argument("--windows")
    ap.add_argument("--court")
    ap.add_argument("--out-dir")
    ap.add_argument("--backend", choices=["vitpose", "rtmpose", "yolo"],
                    default="rtmpose")
    ap.add_argument("--rtm-mode", default="balanced",
                    choices=["performance", "balanced", "lightweight"])
    ap.add_argument("--pose-model",
                    default="usyd-community/vitpose-plus-huge")
    ap.add_argument("--det-thresh", type=float, default=0.3)
    ap.add_argument("--model", default="yolov8s-pose.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--device", default="")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    for req in ("video", "windows", "court", "out_dir"):
        if not getattr(a, req):
            ap.error(f"--{req.replace('_', '-')} required")
    extract(a)


if __name__ == "__main__":
    main()
