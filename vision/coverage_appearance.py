"""Match-specific player-appearance model — identity repair, stage 1.

The overlay spot-check (user, 2026-08-18) caught Tyra/Alshon label
FLIPS mid-game-2 plus long grey (unnamed) stretches: the geometry
chain resolves names at the serve and rides tracks, so a mid-rally
tracker swap between two same-side players silently swaps names for
the rest of the rally.  The four players of a match look nothing
alike, so a per-match appearance model can catch and repair this.

NOT generalizable across matches BY DESIGN: it self-trains from THIS
match's serve-anchor moments — the identity chain resolves all four
names there with high confidence, so anchor-time person crops are free
labeled data (no hand labels anywhere).

Embedding: full-body crop resized 32x96, split into 3 vertical bands,
8x8 H/S histogram per band (kit + skin + silhouette; deliberately no
face anything).  Classifier: nearest centroid, cosine.

Stage 1 (this module): harvest -> train -> leave-one-rally-out
validation -> full-match DISAGREEMENT SCAN vs the geometry chain
(per track x rally: appearance majority vote vs carried name).
Output: data/vision/appearance_scan_<vod>.csv + printed summary.
Stage 2 (separate, only if stage 1 validates): wire as a correction
into coverage.run's carry step and re-run metrics + overlay.

    python vision/coverage_appearance.py --video ... --pose-dir ... \
        --court ... --windows ... --lineup ... [--cam ...]
    python vision/coverage_appearance.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coverage as C
from swing_probe import decode_window

ROOT = Path(__file__).resolve().parent.parent
EMB_DIM = 192
ANCHOR_SPAN = (-0.2, 0.8)      # decode window around the serve anchor
SCAN_FPS = 5.0
WIDTH = 1280                   # must match the extraction width


def embed(frame, box):
    import cv2
    x0, y0, x1, y1 = [max(0, int(v)) for v in box[:4]]
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 6:
        return None
    crop = cv2.resize(crop, (32, 96), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    parts = []
    for band in np.array_split(hsv, 3, axis=0):
        h = cv2.calcHist([band], [0, 1], None, [8, 8],
                         [0, 180, 0, 256]).ravel()
        parts.append(h / (h.sum() + 1e-9))
    return np.concatenate(parts).astype(np.float32)


def cos(a, b):
    return float(a @ b / ((np.linalg.norm(a) + 1e-9)
                          * (np.linalg.norm(b) + 1e-9)))


class Model:
    def __init__(self):
        self.cent = {}

    def fit(self, X_by_u):
        self.cent = {u: np.mean(np.stack(X), 0)
                     for u, X in X_by_u.items() if len(X) >= 8}

    def predict(self, x):
        if not self.cent:
            return None, 0.0
        scored = sorted(((cos(x, c), u) for u, c in self.cent.items()),
                        reverse=True)
        best, second = scored[0], scored[1] if len(scored) > 1 else (0, "")
        return best[1], best[0] - second[0]


def resolve_rallies(a):
    """Per covered rally: dets, serve anchor, carried assignment —
    the same call sequence coverage.run uses (same public functions,
    same gates: approx windows, qual 0, CONF_MIN)."""
    court = C.load_court(a.court)
    cam = C.load_camera(a.cam)
    windows = C.load_windows(a.windows)
    _rows, lineup_by, lineup_ids = C.load_lineup(a.lineup)
    genders, _names = C.player_meta()
    out = []
    for cum in sorted(windows):
        win = windows[cum]
        npz = Path(a.pose_dir) / f"r{cum:04d}.npz"
        if not npz.exists() or win.get("approx") == "1" \
                or win["outcome"] not in ("point", "sideout", "second"):
            continue
        dets, _drops = C.load_rally(npz, court, cam)
        if not dets:
            continue
        t0, t1 = float(win["t0s"]), float(win["t1s"])
        lead = float(win["lead_s"]) if win.get("lead_s") else 0.0
        t_serve, qual, _end = C.find_serve(dets, t0, t1, lead)
        if qual <= 0:
            continue
        lin, _id8 = C.lineup_for(win, lineup_by, lineup_ids)
        if lin is None:
            continue
        heights = defaultdict(list)
        for d in dets:
            heights[d.track].append(d.h_ft or d.h_px)
        heights = {tr: float(np.median(v)) for tr, v in heights.items()}
        names_map, conf, _checks = C.anchor_identity(
            dets, t_serve, win, lin, genders, heights)
        if names_map is None or conf < C.CONF_MIN:
            continue
        dets_sorted = sorted(dets, key=lambda d: d.t)
        assign = C.carry_names(dets_sorted, names_map, conf)
        out.append((cum, win, dets_sorted, assign, t_serve, conf))
    return out


def harvest(a, rallies):
    """Anchor-time labeled crops: (uuid, emb, cum) triples."""
    samples = []
    for cum, win, dets, assign, t_serve, conf in rallies:
        by_t = C.by_frame(dets)
        det_times = np.array(sorted(by_t))
        t0 = t_serve + ANCHOR_SPAN[0]
        frames = decode_window(a.video, t0, ANCHOR_SPAN[1] - ANCHOR_SPAN[0],
                               SCAN_FPS, WIDTH)
        a_by_id = {id(d): u for d, (u, c, _h) in zip(dets, assign)
                   if u is not None}
        for i, frame in enumerate(frames):
            t = t0 + i / SCAN_FPS
            j = int(np.argmin(np.abs(det_times - t)))
            if abs(det_times[j] - t) > 0.6 / SCAN_FPS:
                continue
            for d in by_t[det_times[j]]:
                u = a_by_id.get(id(d))
                if u is None:
                    continue
                e = embed(frame, d.box)
                if e is not None:
                    samples.append((u, e, cum))
    return samples


def validate(samples):
    """Leave-one-rally-out nearest-centroid accuracy + confusion."""
    by_rally = defaultdict(list)
    for u, e, cum in samples:
        by_rally[cum].append((u, e))
    hits = tot = 0
    confusion = Counter()
    for cum, test in by_rally.items():
        X = defaultdict(list)
        for u2, e2, c2 in samples:
            if c2 != cum:
                X[u2].append(e2)
        m = Model()
        m.fit(X)
        for u, e in test:
            p, _m = m.predict(e)
            tot += 1
            hits += p == u
            if p != u:
                confusion[(u, p)] += 1
    return hits / max(tot, 1), tot, confusion


def scan(a, rallies, model, names_full):
    """Full-rally appearance vote per (rally, track) vs carried name."""
    rows = []
    for cum, win, dets, assign, t_serve, conf in rallies:
        by_t = C.by_frame(dets)
        det_times = np.array(sorted(by_t))
        a_by_id = {id(d): u for d, (u, c, _h) in zip(dets, assign)}
        t0, t1 = float(win["t0s"]), float(win["t1s"])
        votes = defaultdict(Counter)     # track -> Counter(uuid)
        carried = defaultdict(Counter)   # track -> Counter(assigned uuid)
        nfr = 0
        for i, frame in enumerate(decode_window(a.video, t0, t1 - t0,
                                                SCAN_FPS, WIDTH)):
            t = t0 + i / SCAN_FPS
            if t < t_serve or not len(det_times):
                continue
            j = int(np.argmin(np.abs(det_times - t)))
            if abs(det_times[j] - t) > 0.6 / SCAN_FPS:
                continue
            nfr += 1
            for d in by_t[det_times[j]]:
                e = embed(frame, d.box)
                if e is None:
                    continue
                p, margin = model.predict(e)
                if p is not None and margin >= 0.02:
                    votes[d.track][p] += 1
                carried[d.track][a_by_id.get(id(d))] += 1
        for tr, v in votes.items():
            app_u, app_n = v.most_common(1)[0]
            car = carried[tr].most_common(1)[0][0]
            n = sum(v.values())
            rows.append(dict(
                rally_cum=cum, game=win["game"], track=tr,
                frames=n, appearance=names_full.get(app_u, app_u[:8]),
                appearance_frac=f"{app_n / n:.2f}",
                carried=(names_full.get(car, (car or 'UNNAMED')[:8])
                         if car else "UNNAMED"),
                agree=int(car == app_u)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path)
    ap.add_argument("--pose-dir")
    ap.add_argument("--court")
    ap.add_argument("--windows")
    ap.add_argument("--lineup")
    ap.add_argument("--cam", default="")
    ap.add_argument("--vod", default="match")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    for req in ("video", "pose_dir", "court", "windows", "lineup"):
        if not getattr(a, req):
            ap.error(f"--{req} required")
    names_full = {r["player_uuid"]: r["player"].split()[-1]
                  for r in csv.DictReader(
                      open(ROOT / "data/coverage_players.csv"))}
    print("resolving rallies (geometry chain)...")
    rallies = resolve_rallies(a)
    print(f"{len(rallies)} rallies resolved")
    print("harvesting anchor-time labeled crops...")
    samples = harvest(a, rallies)
    per = Counter(u for u, _e, _c in samples)
    print(f"{len(samples)} samples: "
          + ", ".join(f"{names_full.get(u, u[:8])} {n}"
                      for u, n in per.most_common()))
    acc, tot, confusion = validate(samples)
    print(f"leave-one-rally-out accuracy {acc:.1%} on {tot}")
    for (u, p), n in confusion.most_common(6):
        print(f"  confused {names_full.get(u, u[:8])} -> "
              f"{names_full.get(p, p[:8])}: {n}")
    model = Model()
    X = defaultdict(list)
    for u, e, _cum in samples:
        X[u].append(e)
    model.fit(X)
    print("scanning all rallies for geometry/appearance disagreement...")
    rows = scan(a, rallies, model, names_full)
    out = ROOT / f"data/vision/appearance_scan_{a.vod}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    dis = [r for r in rows if not r["agree"]]
    n_grey = sum(1 for r in rows if r["carried"] == "UNNAMED")
    print(f"{len(rows)} rally-tracks: {len(dis)} disagree, "
          f"{n_grey} grey (unnamed) with an appearance vote")
    per_game = Counter((r["game"]) for r in dis)
    print(f"disagreements by game: {dict(per_game)}")
    for r in sorted(dis, key=lambda r: -int(r["frames"]))[:12]:
        print(f"  r{r['rally_cum']} g{r['game']} track {r['track']} "
              f"({r['frames']} fr): carried {r['carried']} vs "
              f"appearance {r['appearance']} ({r['appearance_frac']})")
    print(f"-> {out}")


def selftest():
    # two synthetic 'players': red-shirt vs green-shirt over dark floor
    import numpy as np
    rng = np.random.default_rng(7)
    def person(top):
        fr = np.zeros((200, 100, 3), np.uint8)
        fr[:, :] = (40, 40, 40)
        fr[20:90, 30:70] = top          # torso band
        fr[90:160, 35:65] = (30, 30, 90)
        fr += rng.integers(0, 12, fr.shape, np.uint8)
        return fr
    X = defaultdict(list)
    for _ in range(10):
        e = embed(person((200, 30, 30)), (25, 10, 75, 170))
        X["red"].append(e)
        e = embed(person((30, 200, 30)), (25, 10, 75, 170))
        X["green"].append(e)
    m = Model()
    m.fit(X)
    p, margin = m.predict(embed(person((210, 25, 25)), (25, 10, 75, 170)))
    assert p == "red" and margin > 0.02, (p, margin)
    p, _ = m.predict(embed(person((25, 210, 25)), (25, 10, 75, 170)))
    assert p == "green"
    print("SELFTEST OK (embedding separates kit colors, margins sane)")


if __name__ == "__main__":
    main()
