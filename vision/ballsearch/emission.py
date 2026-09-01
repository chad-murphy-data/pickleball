"""Learned per-candidate emission scorer (TRAINING UNLOCKED by owner
2026-09-01 — "I'm here for training in the ball thread").

Discipline that still binds (the auto-label poisoning lesson):
  - positives come ONLY from the owner's hand clicks — never tracker
    output, never model self-labels;
  - TRAIN rallies = r6 + r7 (never used for grading in this thread);
    r9/r10 clicks stay EVALUATION-ONLY so every graded number remains
    comparable to the v1/v2/v3 baselines;
  - V clicks are exact -> positives within R_POS; S clicks are "close
    to where the ball is" (owner caution) -> ignore-zone only, never
    positives; anything within R_IGN of any click is excluded from
    negatives.

Model: standardized logistic regression (hand-rolled, no sklearn) on
14 appearance/motion/temporal features per candidate. The key
temporal one: PERSISTENCE = min(motion at this pixel one frame
earlier, one frame later) — a moving ball vacates its pixel, static
shimmer (net tape, crowd, scoreboard) does not.

Usage:
  python3 emission.py train        # harvest r6+r7, cross-val, save
  python3 emission.py cache 9      # p-cache for rally 9 (cc + peak)
Outputs: emission_model.json, p_r{r}_{mode}_{thr}.npz (row-aligned
with cands_r{r}_{mode}_{thr}.npz).
"""
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import corridor_dp as cdp                       # noqa: E402
from claim_lab import load                      # noqa: E402
from corridor_lab import CLIPS, SP, load_truth  # noqa: E402

THR = 14
R_POS = 6.0     # candidate within this of a V click -> positive
R_IGN = 22.0    # within this of ANY click (V or S) -> excluded
K3 = np.ones((3, 3), np.uint8)
K9 = np.ones((9, 9), np.uint8)
FEATS = ["mot", "mot_prev", "mot_next", "persist", "tophat", "yellow",
         "gray", "std9", "area", "crowd", "dbody", "xn", "yn", "pk"]


def cands_rows(rally, mode, thr=THR):
    """raw rows (f,x,y,ar,pk) from the decode cache, original order;
    builds (and saves) the cache if missing (r6/r7)."""
    p = SP / f"cands_r{rally}_{mode}_{thr}.npz"
    if not p.exists():
        from corridor_chain import frame_candidates
        c = load(rally)
        f_lo = int((c["serve"] - 0.4 - c["t0"]) * 60)
        f_hi = int((c["end"] + 0.2 - c["t0"]) * 60)
        cands = frame_candidates(rally, f_lo, f_hi, thr, mode=mode)
        rows = [(f, x, y, ar, pk) for f, cs in cands.items()
                for (x, y, ar, pk) in cs]
        np.savez_compressed(p, a=np.asarray(rows, np.float32))
    return np.load(p)["a"]


def frame_maps_pass(rally, fmin, fmax):
    """sequential video pass; yields (f, maps) for f in [fmin, fmax].
    maps: mot, mot_prev, mot_next (uint8, 3x3-dilated), tophat,
    yellow (3x3-dilated), gray, std9, areamap (float32).
    mot_prev/mot_next fall back to each other at range edges."""
    cap = cv2.VideoCapture(str(SP / CLIPS[rally]))
    buf, cbuf = deque(maxlen=5), deque(maxlen=5)
    hist = {}       # f -> dict(mot3, mot_raw, th3, yel3, gray, std9, areamap)
    done = {}       # f -> plain motion3 (for prev/next lookups)
    fi = 0
    pend = []       # frames waiting for their next-frame motion
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cbuf.append(frame)
        buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        fi += 1
        mid = fi - 3
        if len(buf) == 5 and fmin - 1 <= mid <= fmax + 1:
            a = cv2.absdiff(buf[2], buf[0])
            b = cv2.absdiff(buf[2], buf[4])
            motion = cv2.min(a, b)
            mot3 = cv2.dilate(motion, K3)
            done[mid] = mot3
            if fmin <= mid <= fmax:
                g = buf[2]
                th3 = cv2.dilate(
                    cv2.morphologyEx(g, cv2.MORPH_TOPHAT, K9), K3)
                col = cbuf[2].astype(np.int16)
                yel = np.clip((col[..., 2] + col[..., 1]) // 2
                              - col[..., 0], 0, 255).astype(np.uint8)
                yel3 = cv2.dilate(yel, K3)
                gf = g.astype(np.float32)
                m1 = cv2.boxFilter(gf, -1, (9, 9))
                m2 = cv2.boxFilter(gf * gf, -1, (9, 9))
                std9 = cv2.sqrt(cv2.max(m2 - m1 * m1, 0))
                _, mask = cv2.threshold(motion, THR, 255,
                                        cv2.THRESH_BINARY)
                mask = cv2.dilate(mask, K3)
                n, lab, stats, _ = \
                    cv2.connectedComponentsWithStats(mask)
                areas = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
                areas[0] = 0.0
                areamap = areas[lab]
                hist[mid] = dict(mot3=mot3, th3=th3, yel3=yel3,
                                 gray=g.copy(), std9=std9,
                                 areamap=areamap)
                pend.append(mid)
            # emit any pending frame whose next motion now exists
            while pend and pend[0] + 1 in done:
                f = pend.pop(0)
                m = hist.pop(f)
                m["mot_prev"] = done.get(f - 1, done[f + 1])
                m["mot_next"] = done[f + 1]
                yield f, m
                done.pop(f - 1, None)
        if fi > fmax + 4:
            break
    cap.release()
    # flush tail (no next frame available)
    for f in pend:
        m = hist.pop(f)
        m["mot_prev"] = done.get(f - 1, done[f])
        m["mot_next"] = m["mot_prev"]
        yield f, m


def featurize(rally, rows_by_mode):
    """rows_by_mode: mode -> (rows array, out feature array to fill).
    One video pass serving every mode's rows."""
    allf = np.concatenate([r[:, 0] for r, _ in rows_by_mode.values()])
    fmin, fmax = int(allf.min()), int(allf.max())
    c = load(rally)
    body = cdp.body_points(c, fmin, fmax)
    idx = {m: {} for m in rows_by_mode}     # mode -> f -> row indices
    for m, (rows, _) in rows_by_mode.items():
        fs = rows[:, 0].astype(int)
        for i, f in enumerate(fs):
            idx[m].setdefault(f, []).append(i)
    # peak-candidate positions per frame for the crowding feature
    pk_rows = (rows_by_mode.get("peak") or rows_by_mode["cc"])[0]
    pk_by_f = {}
    for i in range(len(pk_rows)):
        pk_by_f.setdefault(int(pk_rows[i, 0]), []).append(
            (pk_rows[i, 1], pk_rows[i, 2]))
    H, W = 720, 1280
    for f, m in frame_maps_pass(rally, fmin, fmax):
        pkpts = np.asarray(pk_by_f.get(f, []), float).reshape(-1, 2)
        barr = body.get(f)
        for mode, (rows, out) in rows_by_mode.items():
            ii = idx[mode].get(f)
            if not ii:
                continue
            ii = np.asarray(ii)
            x = rows[ii, 1].astype(float)
            y = rows[ii, 2].astype(float)
            xi = np.clip(np.round(x).astype(int), 0, W - 1)
            yi = np.clip(np.round(y).astype(int), 0, H - 1)
            mot = m["mot3"][yi, xi].astype(float)
            mp = m["mot_prev"][yi, xi].astype(float)
            mn = m["mot_next"][yi, xi].astype(float)
            if len(pkpts):
                d2 = (np.hypot(x[:, None] - pkpts[None, :, 0],
                               y[:, None] - pkpts[None, :, 1]))
                crowd = (d2 <= 16.0).sum(axis=1) - 1.0
            else:
                crowd = np.zeros(len(ii))
            if barr is not None and len(barr):
                db = np.hypot(x[:, None] - barr[None, :, 0],
                              y[:, None] - barr[None, :, 1]).min(axis=1)
            else:
                db = np.full(len(ii), 120.0)
            out[ii, 0] = np.log1p(mot)
            out[ii, 1] = np.log1p(mp)
            out[ii, 2] = np.log1p(mn)
            out[ii, 3] = np.log1p(np.minimum(mp, mn))
            out[ii, 4] = np.log1p(m["th3"][yi, xi].astype(float))
            out[ii, 5] = np.log1p(m["yel3"][yi, xi].astype(float))
            out[ii, 6] = m["gray"][yi, xi].astype(float) / 255.0
            out[ii, 7] = np.log1p(m["std9"][yi, xi].astype(float))
            out[ii, 8] = np.log1p(m["areamap"][yi, xi].astype(float))
            out[ii, 9] = np.log1p(np.maximum(crowd, 0.0))
            out[ii, 10] = np.minimum(db, 120.0) / 120.0
            out[ii, 11] = x / W
            out[ii, 12] = y / H
            out[ii, 13] = np.log1p(rows[ii, 4].astype(float))


def labels_for(rally, rows):
    """+1 pos / 0 neg / -1 ignore per row, from the owner clicks."""
    c = load(rally)
    t0 = c["t0"]
    clicks = {}
    for (t, x, y, vis) in load_truth(rally):
        clicks.setdefault(int(round((t - t0) * 60)), []).append(
            (x, y, vis))
    lab = np.full(len(rows), -1, np.int8)
    for i in range(len(rows)):
        f = int(rows[i, 0])
        cl = clicks.get(f)
        if not cl:
            continue            # unlabeled frame -> ignore
        x, y = rows[i, 1], rows[i, 2]
        dmin, dv = 1e9, 1e9
        for (cx, cy, vis) in cl:
            d = float(np.hypot(x - cx, y - cy))
            dmin = min(dmin, d)
            if vis == "V":
                dv = min(dv, d)
        if dv <= R_POS:
            lab[i] = 1
        elif dmin > R_IGN:
            lab[i] = 0
    return lab


def fit_logistic(X, y, w, l2=1e-3, iters=4000, lr=0.05, seed=7):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    th = rng.normal(0, 0.01, d + 1)
    Xb = np.hstack([X, np.ones((n, 1))])
    m = np.zeros(d + 1)
    v = np.zeros(d + 1)
    for it in range(1, iters + 1):
        z = Xb @ th
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = Xb.T @ (w * (p - y)) / w.sum() + l2 * np.r_[th[:-1], 0.0]
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        th -= lr * m / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8) \
            / (1 - 0.9 ** it)
    return th


def auc(scores, y):
    o = np.argsort(scores)
    r = np.empty(len(scores))
    r[o] = np.arange(1, len(scores) + 1)
    # midranks for ties
    s_sorted = scores[o]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    np_, nn = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - np_ * (np_ + 1) / 2) / (np_ * nn)


def harvest_train(rally):
    rows = cands_rows(rally, "peak")
    F = np.zeros((len(rows), len(FEATS)), np.float32)
    featurize(rally, {"peak": (rows, F)})
    lab = labels_for(rally, rows)
    keep = lab >= 0
    return F[keep], lab[keep].astype(float)


def train():
    packs = {r: harvest_train(r) for r in (6, 7)}
    for r, (F, y) in packs.items():
        print(f"r{r}: {len(y)} labeled cands, {int(y.sum())} pos")
    # cross-rally validation both directions
    stats = {}
    for tr_r, te_r in ((6, 7), (7, 6)):
        Xtr, ytr = packs[tr_r]
        Xte, yte = packs[te_r]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        w = np.where(ytr == 1, (ytr == 0).sum() / max(ytr.sum(), 1), 1.0)
        th = fit_logistic((Xtr - mu) / sd, ytr, w)
        sc = ((Xte - mu) / sd) @ th[:-1] + th[-1]
        a = auc(sc, yte)
        # threshold keeping 97% of held-out positives
        pos_sc = np.sort(sc[yte == 1])
        thr97 = pos_sc[max(0, int(0.03 * len(pos_sc)) - 1)]
        keep_neg = float((sc[yte == 0] >= thr97).mean())
        stats[f"{tr_r}->{te_r}"] = dict(auc=round(float(a), 4),
                                        neg_kept=round(keep_neg, 4))
        print(f"train r{tr_r} -> test r{te_r}: AUC {a:.4f}; at 97% "
              f"pos recall, negatives kept {keep_neg:.1%}")
    # pooled final model
    X = np.vstack([packs[6][0], packs[7][0]])
    y = np.concatenate([packs[6][1], packs[7][1]])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    w = np.where(y == 1, (y == 0).sum() / max(y.sum(), 1), 1.0)
    th = fit_logistic((X - mu) / sd, y, w)
    sc = ((X - mu) / sd) @ th[:-1] + th[-1]
    p = 1 / (1 + np.exp(-sc))
    pos_p = np.sort(p[y == 1])
    dp_keep = float(pos_p[max(0, int(0.03 * len(pos_p)) - 1)])
    print(f"pooled: in-sample AUC {auc(sc, y):.4f}; dp_keep_thr "
          f"(97% pos recall) p >= {dp_keep:.4f}")
    print("weights:", {f: round(float(v), 3)
                       for f, v in zip(FEATS, th[:-1])})
    out = dict(feats=FEATS, mu=mu.tolist(), sd=sd.tolist(),
               w=th[:-1].tolist(), b=float(th[-1]),
               dp_keep_thr=dp_keep, thr=THR,
               train_rallies=[6, 7], xval=stats,
               n=int(len(y)), n_pos=int(y.sum()))
    (SP / "emission_model.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {SP / 'emission_model.json'}")


def write_cache(rally):
    mdl = json.loads((SP / "emission_model.json").read_text())
    mu = np.asarray(mdl["mu"])
    sd = np.asarray(mdl["sd"])
    wv = np.asarray(mdl["w"])
    rows_by = {}
    for mode in ("cc", "peak"):
        rows = cands_rows(rally, mode)
        rows_by[mode] = (rows, np.zeros((len(rows), len(FEATS)),
                                        np.float32))
    featurize(rally, rows_by)
    for mode, (rows, F) in rows_by.items():
        sc = ((F - mu) / sd) @ wv + mdl["b"]
        p = 1 / (1 + np.exp(-np.clip(sc, -30, 30)))
        out = SP / f"p_r{rally}_{mode}_{THR}.npz"
        np.savez_compressed(out, p=p.astype(np.float32),
                            fxy=rows[:, :3].astype(np.float32))
        print(f"{out.name}: {len(p)} rows, p median {np.median(p):.3f}"
              f", p>= dp_keep {float((p >= mdl['dp_keep_thr']).mean()):.1%}")


def write_cache_cross():
    """Cross-fold p-caches for the TRAIN rallies themselves: r6 scored
    by a model fit on r7 only, r7 by r6 only — so any DP weight tuned
    on r6/r7 graded numbers sees OUT-OF-SAMPLE p, not the pooled
    model's in-sample optimism. r9/r10 caches (pooled model) are
    untouched. Each fold's own 97%-recall threshold (p units, from its
    train positives — mirrors dp_keep_thr) is stored as kp97 for the
    hard-filter reference arm."""
    folds = {}
    for r in (6, 7):
        F, y = harvest_train(r)
        mu, sd = F.mean(0), F.std(0) + 1e-6
        w = np.where(y == 1, (y == 0).sum() / max(y.sum(), 1), 1.0)
        th = fit_logistic((F - mu) / sd, y, w)
        sc = ((F - mu) / sd) @ th[:-1] + th[-1]
        p = 1 / (1 + np.exp(-np.clip(sc, -30, 30)))
        pos_p = np.sort(p[y == 1])
        kp97 = float(pos_p[max(0, int(0.03 * len(pos_p)) - 1)])
        folds[r] = (mu, sd, th, kp97)
        print(f"fold r{r}: {len(y)} labeled cands, {int(y.sum())} pos,"
              f" kp97 {kp97:.4f}")
    for target, src in ((6, 7), (7, 6)):
        mu, sd, th, kp97 = folds[src]
        rows_by = {}
        for mode in ("cc", "peak"):
            rows = cands_rows(target, mode)
            rows_by[mode] = (rows, np.zeros((len(rows), len(FEATS)),
                                            np.float32))
        featurize(target, rows_by)
        for mode, (rows, F) in rows_by.items():
            sc = ((F - mu) / sd) @ th[:-1] + th[-1]
            p = 1 / (1 + np.exp(-np.clip(sc, -30, 30)))
            out = SP / f"p_r{target}_{mode}_{THR}_x.npz"
            np.savez_compressed(out, p=p.astype(np.float32),
                                fxy=rows[:, :3].astype(np.float32),
                                kp97=np.float32(kp97))
            print(f"{out.name}: {len(p)} rows (model=r{src}), p median"
                  f" {np.median(p):.3f}, p>=kp97 "
                  f"{float((p >= kp97).mean()):.1%}")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "cache":
        write_cache(int(sys.argv[2]))
    elif sys.argv[1] == "cache-cross":
        write_cache_cross()
    else:
        raise SystemExit(
            "usage: emission.py train | cache <rally> | cache-cross")
