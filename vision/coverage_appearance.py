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


CROP_W, CROP_H = 48, 128
# COCO segments per body region — colors are sampled ALONG THE
# PLAYER'S OWN SKELETON.  Box-crop histograms validated at 64% LORO
# with within-team confusion; the crop sheet showed why: serve-time
# boxes are half floor and often contain the PARTNER (Bright's crops
# carry Patriquin's black tee and vice versa).  Skeleton sampling
# excludes background and partner pixels by construction.
TORSO = [(5, 11), (6, 12), (5, 6)]
LEGS = [(11, 13), (13, 15), (12, 14), (14, 16)]
ARMS = [(5, 7), (6, 8)]


def crop_of(frame, box, kpt=None, kpc=None):
    import cv2
    x0, y0, x1, y1 = [max(0, int(v)) for v in box[:4]]
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 6:
        return None
    out = cv2.resize(crop, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
    if kpt is None:
        return out
    k = np.zeros((17, 2), np.float32)
    k[:, 0] = (kpt[:, 0] - x0) * CROP_W / max(x1 - x0, 1)
    k[:, 1] = (kpt[:, 1] - y0) * CROP_H / max(y1 - y0, 1)
    return out, k


def embed_crop(crop, kpt, kpc):
    """Per-region (torso/legs/arms) HSV histogram of colors sampled in
    3x3 patches along the skeleton segments."""
    import cv2
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    Hh, Ww = crop.shape[:2]

    def region(segs):
        cols = []
        for a, b in segs:
            if kpc[a] < 0.3 or kpc[b] < 0.3:
                continue
            pa, pb = kpt[a], kpt[b]
            for f in np.linspace(0.15, 0.85, 8):
                x = int(pa[0] + f * (pb[0] - pa[0]))
                y = int(pa[1] + f * (pb[1] - pa[1]))
                if 1 <= x < Ww - 1 and 1 <= y < Hh - 1:
                    cols.append(hsv[y - 1:y + 2, x - 1:x + 2]
                                .reshape(-1, 3).mean(0))
        return cols

    parts = []
    for segs in (TORSO, LEGS, ARMS):
        cols = region(segs)
        if len(cols) < 4:
            parts.append(np.zeros(72, np.float32))
            continue
        hist, _ = np.histogramdd(
            np.array(cols, np.float64), bins=(8, 3, 3),
            range=((0, 181), (0, 256), (0, 256)))
        h = hist.ravel()
        parts.append((h / (h.sum() + 1e-9)).astype(np.float32))
    return np.concatenate(parts)


def embed(frame, d):
    """Embedding for a detection (needs its keypoints)."""
    if d.kpt is None:
        return None
    got = crop_of(frame, d.box, np.asarray(d.kpt, np.float32),
                  np.asarray(d.kpc, np.float32))
    if got is None:
        return None
    crop, k = got
    return embed_crop(crop, k, np.asarray(d.kpc, np.float32))


def cos(a, b):
    return float(a @ b / ((np.linalg.norm(a) + 1e-9)
                          * (np.linalg.norm(b) + 1e-9)))


def box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    i = (x1 - x0) * (y1 - y0)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return i / (aa + bb - i)


def max_iou(d, ds):
    return max((box_iou(d.box, e.box) for e in ds if e is not d),
               default=0.0)


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
    """Anchor-time labeled crops: (uuid, emb, cum) triples.  Crops are
    cached next to the scan CSV so embedding iteration is offline —
    the decode pass runs once."""
    cache = ROOT / f"data/vision/appearance_crops_{a.vod}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if "iso" in z:
            crops, uuids, cums = z["crop"], z["uuid"], z["cum"]
            sides, hs, isos = z["side"], z["h"], z["iso"]
            kpts, kpcs = z["kpt"], z["kpc"]
            print(f"harvest crops from cache ({len(crops)})")
            return [(str(uuids[i]),
                     embed_crop(crops[i], kpts[i], kpcs[i]),
                     int(cums[i]), int(sides[i]), float(hs[i]),
                     float(isos[i]))
                    for i in range(len(crops))]
        cache.unlink()          # pre-iso cache: rebuild
    crops, uuids, cums, sides, hs, kpts, kpcs, isos = ([] for _ in range(8))
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
                if u is None or d.kpt is None:
                    continue
                got = crop_of(frame, d.box,
                              np.asarray(d.kpt, np.float32),
                              np.asarray(d.kpc, np.float32))
                if got is None:
                    continue
                cr, k = got
                isos.append(max_iou(d, by_t[det_times[j]]))
                crops.append(cr)
                kpts.append(k)
                kpcs.append(np.asarray(d.kpc, np.float32))
                uuids.append(u)
                cums.append(cum)
                sides.append(int(d.side))
                hs.append(float(d.box[3] - d.box[1]))
    np.savez_compressed(cache, crop=np.stack(crops),
                        uuid=np.array(uuids), cum=np.array(cums),
                        side=np.array(sides), h=np.array(hs),
                        kpt=np.stack(kpts), kpc=np.stack(kpcs),
                        iso=np.array(isos))
    print(f"cached {len(crops)} crops -> {cache}")
    return [(uuids[i], embed_crop(crops[i], kpts[i], kpcs[i]),
             cums[i], sides[i], hs[i], isos[i])
            for i in range(len(crops))]


def validate(samples, partner):
    """Leave-one-rally-out diagnostics: overall 4-way, per-side 4-way
    with per-side centroids (near and far court are different lighting
    and scale regimes), and the binary within-team call the swap check
    actually needs."""
    by_rally = defaultdict(list)
    for u, e, cum, side, h, _iso in samples:
        by_rally[cum].append((u, e, side, h))
    st = Counter()
    confusion = Counter()
    for cum, test in by_rally.items():
        Xs = {0: defaultdict(list), 1: defaultdict(list)}
        Xall = defaultdict(list)
        for u2, e2, c2, s2, _h2, _i2 in samples:
            if c2 != cum:
                Xs[s2][u2].append(e2)
                Xall[u2].append(e2)
        m_all = Model()
        m_all.fit(Xall)
        m_side = {}
        for s in (0, 1):
            m_side[s] = Model()
            m_side[s].fit(Xs[s])
        for u, e, s, _h in test:
            p, _ = m_all.predict(e)
            st["tot"] += 1
            st["hit"] += p == u
            if p != u:
                confusion[(u, p)] += 1
            ps, _ = m_side[s].predict(e)
            st[f"stot{s}"] += 1
            st[f"shit{s}"] += ps == u
            q = partner.get(u)
            if q and u in m_side[s].cent and q in m_side[s].cent:
                bu = cos(e, m_side[s].cent[u])
                bq = cos(e, m_side[s].cent[q])
                st[f"btot{s}"] += 1
                st[f"bhit{s}"] += bu > bq
    return st, confusion


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
                e = embed(frame, d)
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



def moments_crop(crop, kpt, kpc):
    """Per-region Lab color moments (mean+std of L,a,b for torso/legs/
    arms = 18 dims).  Histogram embeddings could not separate the
    desaturated kits (black tee, dark shorts) — compact moments in a
    perceptual space do: 97.9%/98.3% within-team sample accuracy
    against EM-consistent labels on the mixed final."""
    import cv2
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    Hh, Ww = crop.shape[:2]
    out = []
    for segs in (TORSO, LEGS, ARMS):
        cols = []
        for a, b in segs:
            if kpc[a] < 0.3 or kpc[b] < 0.3:
                continue
            pa, pb = kpt[a], kpt[b]
            for f in np.linspace(0.15, 0.85, 8):
                x = int(pa[0] + f * (pb[0] - pa[0]))
                y = int(pa[1] + f * (pb[1] - pa[1]))
                if 1 <= x < Ww - 1 and 1 <= y < Hh - 1:
                    cols.append(lab[y - 1:y + 2, x - 1:x + 2]
                                .reshape(-1, 3).mean(0))
        if len(cols) < 4:
            out.extend([np.nan] * 6)
        else:
            arr = np.array(cols)
            out.extend(arr.mean(0).tolist() + arr.std(0).tolist())
    return np.array(out, np.float32)


def team_lda_em(X, y, cs, iters=4):
    """2-class LDA direction with RALLY-level label-flip EM: anchor
    identity errors swap BOTH partners for a whole rally, so labels are
    noisy per-rally, not per-sample.  The direction is global; a rally
    flips when its samples project majority-wrong; polarity is anchored
    by the unswapped majority."""
    yy = y.copy()
    w = thr = None
    for it in range(iters):
        mu0, mu1 = X[yy == 0].mean(0), X[yy == 1].mean(0)
        Sw = np.cov(X[yy == 0].T) + np.cov(X[yy == 1].T)
        w = np.linalg.solve(Sw + 1e-3 * np.eye(X.shape[1]), mu1 - mu0)
        proj = X @ w
        thr = (proj[yy == 0].mean() + proj[yy == 1].mean()) / 2
        flips = 0
        for c in set(cs.tolist()):
            mm = cs == c
            if np.mean((proj[mm] > thr) == (yy[mm] == 1)) < 0.5:
                yy[mm] = 1 - yy[mm]
                flips += 1
        if flips == 0 and it > 0:
            break
    return w, thr, yy


def audit(a, samples, partner, names_full):
    """Anchor-identity audit: per (team x GAME) EM over Lab moments ->
    a swap ledger (rally x team -> swap 0/1 + unanimity).  Consumed by
    coverage.run --swaps.  Per-game because kits change between games
    (the measured Alshon game-3 shirt change)."""
    game_of = {int(r["rally_cum"]): int(r["game"]) for r in
               csv.DictReader(open(a.windows))}
    z = np.load(ROOT / f"data/vision/appearance_crops_{a.vod}.npz")
    crops, uuids, cums = z["crop"], z["uuid"], z["cum"]
    kpts, kpcs, isos = z["kpt"], z["kpc"], z["iso"]
    keep = [i for i in range(len(crops)) if isos[i] <= 0.05]
    F = np.stack([moments_crop(crops[i], kpts[i], kpcs[i]) for i in keep])
    lab_u = np.array([str(uuids[i]) for i in keep])
    cum_k = np.array([int(cums[i]) for i in keep])
    gm = np.array([game_of.get(int(c), -1) for c in cum_k])
    ok = ~np.isnan(F).any(1)
    teams = sorted({tuple(sorted((u, partner[u]))) for u in set(lab_u)
                    if partner.get(u)})
    rows = []
    for ua, ub in teams:
        for g in sorted(set(gm.tolist())):
            m = ok & (gm == g) & np.isin(lab_u, (ua, ub))
            if m.sum() < 16:
                continue
            X, y = F[m], (lab_u[m] == ub).astype(int)
            cs = cum_k[m]
            w, thr, yy = team_lda_em(X, y, cs)
            proj = X @ w
            pred = (proj > thr).astype(int)
            for c in sorted(set(cs.tolist())):
                mm = cs == c
                if mm.sum() < 3:
                    continue
                swap = int(not np.array_equal(yy[mm], y[mm]))
                unan = float(np.mean(pred[mm] == yy[mm]))
                rows.append(dict(rally_cum=int(c), team=f"{ua}|{ub}",
                                 swap=swap, unanimity=f"{unan:.2f}",
                                 n=int(mm.sum())))
        n_sw = sum(r["swap"] for r in rows if r["team"] == f"{ua}|{ub}")
        n_all = sum(1 for r in rows if r["team"] == f"{ua}|{ub}")
        print(f"{names_full[ua]}/{names_full[ub]}: chain swapped "
              f"{n_sw}/{n_all} audited rallies")
    out = ROOT / f"data/vision/identity_swaps_{a.vod}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"-> {out}")



class FourWay:
    """4-way nearest centroid on z-scored Lab moments, trained on
    EM-cleaned labels.  predict -> (uuid, margin)."""

    def fit(self, F, labels):
        self.mu = F.mean(0)
        self.sd = F.std(0) + 1e-6
        Z = (F - self.mu) / self.sd
        self.cent = {u: Z[np.array(labels) == u].mean(0)
                     for u in set(labels)}

    def predict(self, f, candidates=None):
        """candidates = restrict the call to these uuids.  Partners
        stand on the SAME side, so a track's near/far assignment plus
        the per-game team->end map narrows a 4-way call to the two
        partners it could possibly be.  Measured on this match: only
        32 of 340 4-way errors were cross-team, so this removes a
        small but structurally impossible error class (the other 308
        are partner confusions, which side knowledge cannot touch)."""
        z = (f - self.mu) / self.sd
        pool = (self.cent.items() if candidates is None
                else [(u, self.cent[u]) for u in candidates
                      if u in self.cent])
        d = sorted((float(np.linalg.norm(z - c)), u) for u, c in pool)
        if not d:
            return None, 0.0
        if len(d) == 1:
            return d[0][1], np.inf
        return d[0][1], d[1][0] - d[0][0]


def clean_labels(a, partner, game_of):
    """EM-corrected anchor labels, PER (team x GAME): kits can change
    between games (measured: Alshon swapped his red tee for a gray one
    in game 3 and every appearance model trained across games called
    him Bright), so no direction is ever fit across a game boundary."""
    z = np.load(ROOT / f"data/vision/appearance_crops_{a.vod}.npz")
    crops, uuids, cums = z["crop"], z["uuid"], z["cum"]
    kpts, kpcs, isos = z["kpt"], z["kpc"], z["iso"]
    keep = [i for i in range(len(crops)) if isos[i] <= 0.05]
    F = np.stack([moments_crop(crops[i], kpts[i], kpcs[i]) for i in keep])
    lab = np.array([str(uuids[i]) for i in keep])
    cum_k = np.array([int(cums[i]) for i in keep])
    gm = np.array([game_of.get(int(c), -1) for c in cum_k])
    ok = ~np.isnan(F).any(1)
    teams = sorted({tuple(sorted((u, partner[u]))) for u in set(lab)
                    if partner.get(u)})
    lab2 = lab.copy()
    for g in sorted(set(gm.tolist())):
        for ua, ub in teams:
            m = ok & (gm == g) & np.isin(lab, (ua, ub))
            if m.sum() < 16:
                continue
            X, y = F[m], (lab[m] == ub).astype(int)
            _w, _t, yy = team_lda_em(X, y, cum_k[m])
            lab2[m] = np.where(yy == 1, ub, ua)
    return F[ok], lab2[ok], cum_k[ok], gm[ok]


def stage2(a, rallies, partner, names_full):
    """Full mid-rally identity repair: 4-way appearance votes over
    every detection -> per-track rebind / grey rescue / changepoint
    split -> identity_track_map_<vod>.csv (consumed by coverage.run
    --track-map on top of the anchor swap ledger)."""
    game_of = {int(r["rally_cum"]): int(r["game"]) for r in
               csv.DictReader(open(a.windows))}
    F, lab, cum_k, gm = clean_labels(a, partner, game_of)
    fw = {}
    for g in sorted(set(gm.tolist())):
        fw[g] = FourWay()
        fw[g].fit(F[gm == g], lab[gm == g])
    # leave-one-rally-out check per game on the cleaned labels;
    # games below the adjudication bar contribute NO corrections
    # (measured: game 3 at 54.8% — Alshon's gray game-3 shirt vs
    # Bright's green is under-separable at this crop resolution)
    LORO_BAR = 0.85
    for g in sorted(fw):
        hits = tot = 0
        for c in sorted(set(cum_k[gm == g].tolist())):
            m = (gm == g) & (cum_k != c)
            f2 = FourWay()
            f2.fit(F[m], lab[m])
            for i in np.nonzero((gm == g) & (cum_k == c))[0]:
                p, _ = f2.predict(F[i])
                tot += 1
                hits += p == lab[i]
        acc = hits / max(tot, 1)
        print(f"game {g} 4-way LORO on EM-cleaned labels: "
              f"{acc:.1%} on {tot}")
        if acc < LORO_BAR:
            print(f"  game {g} BELOW the {LORO_BAR:.0%} bar — "
                  f"no corrections will be emitted for it")
            del fw[g]

    # team -> end per game, from the anchor labels themselves: which
    # two players hold side 0 (near) and side 1 (far) in each game
    # (game, end-segment) -> side -> pair.  Segments are FITTED, not
    # assumed: teams change ends mid-game under league-specific rules
    # (MLP at 6 every game, PPA only in a decider), and a per-game map
    # would be wrong for half of every MLP game.
    per_game = defaultdict(list)
    for cum, _w, dets, assign, _ts, _cf in rallies:
        cnt = defaultdict(Counter)
        for d, (u, _c, _h) in zip(dets, assign):
            if u and d.side in (0, 1):
                cnt[u][d.side] += 1
        near = tuple(sorted(u for u, c in cnt.items() if c[0] > c[1]))
        per_game[game_of.get(cum, -1)].append((cum, near))
    seg_of = {}
    for g, obs in per_game.items():
        for cum, seg in C.fit_end_segments(obs).items():
            seg_of[cum] = (g, seg)
    end_by_seg = defaultdict(lambda: defaultdict(Counter))
    for cum, _w, dets, assign, _ts, _cf in rallies:
        gs = seg_of.get(cum)
        if gs is None:
            continue
        for d, (u, _c, _h) in zip(dets, assign):
            if u and d.side in (0, 1):
                end_by_seg[gs][d.side][u] += 1
    endmap = {gs: {s: [u for u, _ in c.most_common(2)]
                   for s, c in by_s.items()}
              for gs, by_s in end_by_seg.items()}

    swaps = defaultdict(list)
    led = ROOT / f"data/vision/identity_swaps_{a.vod}.csv"
    if led.exists():
        for r in csv.DictReader(open(led)):
            if r["swap"] == "1" and float(r["unanimity"]) >= 0.8:
                swaps[int(r["rally_cum"])].append(
                    tuple(r["team"].split("|")))

    rows = []
    n_rebind = n_rescue = n_split = 0
    for cum, win, dets, assign, t_serve, conf in rallies:
        g = game_of.get(cum, -1)
        if g not in fw:
            continue
        fwg = fw[g]
        by_t = C.by_frame(dets)
        det_times = np.array(sorted(by_t))
        a_by_id = {id(d): u for d, (u, c, _h) in zip(dets, assign)}
        # post-swap carried names (anchor ledger applied)
        for (ua, ub) in swaps.get(cum, ()):
            for k in a_by_id:
                if a_by_id[k] == ua:
                    a_by_id[k] = "__tmp__"
            for k in a_by_id:
                if a_by_id[k] == ub:
                    a_by_id[k] = ua
            for k in a_by_id:
                if a_by_id[k] == "__tmp__":
                    a_by_id[k] = ub
        t0, t1 = float(win["t0s"]), float(win["t1s"])
        votes = defaultdict(list)        # track -> [(t, uuid)]
        carried = defaultdict(Counter)
        sides = defaultdict(Counter)
        for i, frame in enumerate(decode_window(a.video, t0, t1 - t0,
                                                SCAN_FPS, WIDTH)):
            t = t0 + i / SCAN_FPS
            if t < t_serve or not len(det_times):
                continue
            j = int(np.argmin(np.abs(det_times - t)))
            if abs(det_times[j] - t) > 0.6 / SCAN_FPS:
                continue
            ds = by_t[det_times[j]]
            for d in ds:
                if d.kpt is None or max_iou(d, ds) > 0.05:
                    continue
                got = crop_of(frame, d.box,
                              np.asarray(d.kpt, np.float32),
                              np.asarray(d.kpc, np.float32))
                if got is None:
                    continue
                cr, k = got
                f = moments_crop(cr, k, np.asarray(d.kpc, np.float32))
                if np.isnan(f).any():
                    continue
                cands = endmap.get(seg_of.get(cum, (g, 0)),
                                   {}).get(d.side)
                p, margin = fwg.predict(f, cands)
                if p is not None and margin >= 0.5:
                    votes[d.track].append((float(d.t), p))
                carried[d.track][a_by_id.get(id(d))] += 1
                sides[d.track][d.side] += 1

        for tr, vt in votes.items():
            if len(vt) < 5:
                continue
            vt.sort()
            us = [u for _t, u in vt]
            car = carried[tr].most_common(1)[0][0]
            cnt = Counter(us)
            app, n_app = cnt.most_common(1)[0]
            frac = n_app / len(us)
            # changepoint: best split with both halves pure+different
            split_at = None
            if frac < 0.9 and len(us) >= 12:
                best = 0.0
                for s in range(6, len(us) - 6):
                    c1, c2 = Counter(us[:s]), Counter(us[s:])
                    (u1, n1), = c1.most_common(1)
                    (u2, n2), = c2.most_common(1)
                    if u1 != u2 and n1 / s >= 0.75 \
                            and n2 / (len(us) - s) >= 0.75:
                        pur = (n1 + n2) / len(us)
                        if pur > best:
                            best, split_at = pur, (s, u1, u2)
            if split_at:
                s, u1, u2 = split_at
                t_mid = (vt[s - 1][0] + vt[s][0]) / 2
                for (uu, ta, tb) in ((u1, t0, t_mid), (u2, t_mid, t1)):
                    rows.append(dict(
                        rally_cum=cum, track=tr, t0=f"{ta:.2f}",
                        t1=f"{tb:.2f}", uuid=uu, action="split",
                        frac=f"{best:.2f}", n=len(us)))
                n_split += 1
                continue
            if frac < 0.7:
                continue
            if car is None:
                # grey rescue: side must match the player's end majority
                rows.append(dict(rally_cum=cum, track=tr, t0=f"{t0:.2f}",
                                 t1=f"{t1:.2f}", uuid=app,
                                 action="rescue", frac=f"{frac:.2f}",
                                 n=len(us)))
                n_rescue += 1
            elif car != app:
                rows.append(dict(rally_cum=cum, track=tr, t0=f"{t0:.2f}",
                                 t1=f"{t1:.2f}", uuid=app,
                                 action="rebind", frac=f"{frac:.2f}",
                                 n=len(us)))
                n_rebind += 1
    out = ROOT / f"data/vision/identity_track_map_{a.vod}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["rally_cum", "track", "t0",
                                            "t1", "uuid", "action",
                                            "frac", "n"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"stage 2: {n_rebind} rebinds, {n_rescue} grey rescues, "
          f"{n_split} track splits -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path)
    ap.add_argument("--pose-dir")
    ap.add_argument("--court")
    ap.add_argument("--windows")
    ap.add_argument("--lineup")
    ap.add_argument("--cam", default="")
    ap.add_argument("--vod", default="match")
    ap.add_argument("--validate-only", action="store_true",
                    help="harvest + LORO validation, skip the full scan")
    ap.add_argument("--audit", action="store_true",
                    help="anchor-identity audit -> swap ledger")
    ap.add_argument("--stage2", action="store_true",
                    help="full mid-rally repair -> track map")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    for req in ("video", "pose_dir", "court", "windows", "lineup"):
        if not getattr(a, req):
            ap.error(f"--{req} required")
    names_full, partner = {}, {}
    for r in csv.DictReader(open(ROOT / "data/coverage_players.csv")):
        names_full[r["player_uuid"]] = r["player"].split()[-1]
        partner[r["player_uuid"]] = r["partner_uuid"]
    print("resolving rallies (geometry chain)...")
    rallies = resolve_rallies(a)
    print(f"{len(rallies)} rallies resolved")
    print("harvesting anchor-time labeled crops...")
    samples = harvest(a, rallies)
    n0 = len(samples)
    samples = [s for s in samples if s[5] <= 0.05]
    print(f"isolation filter: {len(samples)}/{n0} crops kept "
          f"(IoU<=0.05 with every other box)")
    per = Counter(s[0] for s in samples)
    print(f"{len(samples)} samples: "
          + ", ".join(f"{names_full.get(u, u[:8])} {n}"
                      for u, n in per.most_common()))
    if a.audit:
        audit(a, samples, partner, names_full)
        return
    if a.stage2:
        stage2(a, rallies, partner, names_full)
        return
    st, confusion = validate(samples, partner)
    print(f"leave-one-rally-out 4-way accuracy "
          f"{st['hit'] / max(st['tot'], 1):.1%} on {st['tot']}")
    for s, nm in ((0, "NEAR"), (1, "FAR")):
        print(f"  {nm}: 4-way(per-side) "
              f"{st[f'shit{s}'] / max(st[f'stot{s}'], 1):.1%} "
              f"on {st[f'stot{s}']}   within-team binary "
              f"{st[f'bhit{s}'] / max(st[f'btot{s}'], 1):.1%} "
              f"on {st[f'btot{s}']}")
    for (u, p), n in confusion.most_common(6):
        print(f"  confused {names_full.get(u, u[:8])} -> "
              f"{names_full.get(p, p[:8])}: {n}")
    if a.validate_only:
        return
    model = Model()
    X = defaultdict(list)
    for s in samples:
        X[s[0]].append(s[1])
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
    # two synthetic 'players': red-shirt vs green-shirt over dark floor,
    # PLUS a partner-colored slab inside the box that skeleton sampling
    # must ignore (the crop-histogram failure mode, reproduced)
    rng = np.random.default_rng(7)

    class D:
        pass

    def person(top, bleed):
        fr = np.zeros((200, 100, 3), np.uint8)
        fr[:, :] = (170, 110, 60)              # bright floor
        fr[20:90, 30:70] = top                 # torso
        fr[90:160, 35:65] = (30, 30, 90)       # shorts
        fr[10:190, 75:98] = bleed              # partner bleeding into box
        fr += rng.integers(0, 12, fr.shape, np.uint8)
        d = D()
        d.box = (25, 10, 99, 190)
        k = np.zeros((17, 2), np.float32)
        k[5] = (35, 25)
        k[6] = (65, 25)                        # shoulders
        k[11] = (38, 88)
        k[12] = (62, 88)                       # hips
        k[13] = (40, 120)
        k[14] = (60, 120)                      # knees
        k[15] = (42, 155)
        k[16] = (58, 155)                      # ankles
        k[7] = (32, 55)
        k[8] = (68, 55)                        # elbows
        d.kpt = k
        d.kpc = np.full(17, 0.9, np.float32)
        return fr, d

    X = defaultdict(list)
    for _ in range(10):
        fr, d = person((200, 30, 30), (30, 200, 30))
        X["red"].append(embed(fr, d))
        fr, d = person((30, 200, 30), (200, 30, 30))
        X["green"].append(embed(fr, d))
    m = Model()
    m.fit(X)
    fr, d = person((210, 25, 25), (25, 210, 25))
    p, margin = m.predict(embed(fr, d))
    assert p == "red" and margin > 0.02, (p, margin)
    fr, d = person((25, 210, 25), (210, 25, 25))
    p, _ = m.predict(embed(fr, d))
    assert p == "green"
    # side-restricted predict: the far pair can never win a near call
    fw = FourWay()
    fw.fit(np.array([[0.0, 0], [0.2, 0], [9.0, 0], [9.2, 0]], np.float32),
           ["nearA", "nearB", "farA", "farB"])
    p4, _ = fw.predict(np.array([8.9, 0], np.float32))
    assert p4 == "farA", p4                      # unconstrained
    p2, m2 = fw.predict(np.array([8.9, 0], np.float32), ["nearA", "nearB"])
    assert p2 in ("nearA", "nearB") and m2 >= 0, (p2, m2)
    p1, m1 = fw.predict(np.array([0.0, 0], np.float32), ["nearA"])
    assert p1 == "nearA" and np.isinf(m1)        # lone candidate
    pn, mn = fw.predict(np.array([0.0, 0], np.float32), ["ghost"])
    assert pn is None and mn == 0.0              # unknown candidate
    print("  side-restricted predict + lone/unknown candidates OK")
    print("SELFTEST OK (skeleton sampling ignores floor + partner bleed)")


if __name__ == "__main__":
    main()
