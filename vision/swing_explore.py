"""EXPLORATION (no gate, no bars): can a LEARNED scorer find swings
where the raw-peak ceiling could not?

The Gate C diagnostic (contact_ceiling_report_rtm.json) killed a DUMB
selector: top-2x candidates ranked by raw peak height covered 45.1% of
contacts — near-perfect on smashes/speed-ups, blind to dinks/counters,
with @4x=64% hinting the stream holds more than the ranking finds.
This script trains the thing the ceiling could only proxy: a small
classifier over WINDOWED whole-skeleton motion around the user's
timestamped contacts, evaluated leave-one-rally-out so every reported
number is out-of-sample. It answers, empirically:

  1. Can a model taught "this is what a swing looks like" find swings
     in rallies it never saw?  (headline: learned coverage@2x vs 45.1%)
  2. Is the PREPARATION ARC real signal? (ablation: full window
     [-0.75s..+0.35s] vs strike-only [-0.1s..+0.35s] — the user's
     "I can tell who is swinging many frames early" hypothesis; blur
     kills the wrist AT contact, but preparation frames are sharp,
     which is exactly where the dink/counter failure should yield)
  3. Do position + cadence features carry type information for free?
     (per-type recall table at the operating point)

RUN (the user's flat folder; needs pose_rtm/ from pose_extract
--backend rtmpose and the labels export; numpy only, CPU, minutes):

    python3 swing_explore.py
    python3 swing_explore.py --report swing_explore_report.json

STATUS: exploration on the spent dev rallies. Nothing here is a
verdict; if a result wants to be believed, it graduates to a fresh
pre-registration gated on rallies never used here (contact_gate.md).

SELF-TEST (no files):  python3 swing_explore.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from swing_probe import strongest_first
from contact_ceiling import (load_rosters, load_labels, rally_candidates,
                             rally_coverage, ARM_JOINTS, L_SHO, R_SHO,
                             L_ELB, R_ELB, L_WRIST, R_WRIST, L_HIP, R_HIP)

# COCO-17 indices not needed by Gate C, so not in contact_ceiling.py —
# footwork and gaze channels only (feature_check.py)
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16

LABELS = "contact_labels_chicago0725.csv"
WINDOWS_V4 = "rally_windows_chicago0725_v4.csv"
POSE_DIR = "pose_rtm"

MAX_ROT_RAD = 1.2   # hard physical cap on dsho/dgaze: ~69 deg between
# two valid consecutive frames (ok_dt already bounds that gap at <=2.5
# frames), well above any real human shoulder/head turn even at elite
# rotational speed, comfortably below the ~pi (180 deg) an L/R swap
# produces. Confidence gating alone does NOT catch swaps: a model can
# be fully confident it found "a shoulder" while wrong about which side
# -- measured on the user's real run, 2026-08-18, where the confidence
# gate made ZERO difference to the reported dsho numbers (identical
# max/median/histogram to the pre-gate run). Belt and suspenders.

PRE_S, POST_S = 0.75, 0.35     # the window around a click (question 2)
STRIKE_PRE_S = 0.10            # "strike-only" ablation window start
GUARD_S = 0.5                  # negatives keep this far from any contact
WHIFF_GUARD_S = 0.6            # ...and further from whiffs (real swings)
TOL_S = 0.30                   # match tolerance (same as the ceiling)
REFRACTORY_S = 0.25
STRIDE = 3                     # frames between scored windows at inference
SEED = 20260817


# ---------------------------------------------------------- per-track


def track_series(t, box, kpt, kpc, fps):
    """Per-frame engineered channels for ONE track (sorted by t).
    Everything torso-relative and torso-scaled so locomotion and
    near/far image scale cancel where they should."""
    n = len(t)
    bh = box[:, 3] - box[:, 1]
    cx = (box[:, 0] + box[:, 2]) / 2

    def joint(a, b, fy):
        ok = (kpc[:, a] >= 0.2) & (kpc[:, b] >= 0.2)
        pt = (kpt[:, a] + kpt[:, b]) / 2
        fb = np.stack([cx, box[:, 1] + fy * bh], 1)
        return np.where(ok[:, None], pt, fb)

    hip = joint(L_HIP, R_HIP, 0.62)
    sho = joint(L_SHO, R_SHO, 0.25)
    scale = np.maximum(np.linalg.norm(sho - hip, axis=1), 0.35 * bh)
    scale = np.maximum(scale, 3.0)

    dt = np.diff(t)
    ok_dt = dt <= 2.5 / fps
    chans = {}
    for name, j in (("lw", L_WRIST), ("rw", R_WRIST),
                    ("le", L_ELB), ("re", R_ELB),
                    ("la", L_ANK), ("ra", R_ANK),
                    ("lk", L_KNEE), ("rk", R_KNEE)):
        rel = (kpt[:, j] - hip) / scale[:, None]
        v = np.zeros(n)
        d = np.linalg.norm(np.diff(rel, axis=0), axis=1)
        # torso-scales per frame, gap-safe; dt FLOOR at a third of a
        # frame (v1 divided by near-zero dt on duplicate-adjacent rows,
        # spraying inf through the feature matrix — the matmul overflow
        # warnings in the first real run), and a physical cap: no human
        # wrist moves 3 torso-lengths per frame
        vv = np.where(ok_dt, d / np.maximum(dt * fps, 0.33), 0.0)
        conf = kpc[:, j] >= 0.15
        v[1:] = np.where(conf[1:] & conf[:-1], vv, 0.0)
        chans[name] = np.minimum(v, 3.0)
    arm = np.maximum.reduce([chans[k] for k in ("lw", "rw", "le", "re")])
    # hip-relative, same as arm: cancels overall court travel (hipv,
    # below, already covers that), isolates STEPPING/shuffle motion —
    # a player planted with feet moving under them reads differently
    # here than one just jogging to a spot with a steady stride
    leg = np.maximum.reduce([chans[k] for k in ("la", "ra", "lk", "rk")])

    shovec = kpt[:, R_SHO] - kpt[:, L_SHO]
    shoang = np.arctan2(shovec[:, 1], shovec[:, 0] + 1e-9)
    dsho_step = np.abs(np.angle(np.exp(1j * np.diff(shoang))))
    dsho = np.zeros(n)
    # confidence gate alone measured to do NOTHING here (see MAX_ROT_RAD
    # above) -- an L/R swap is often confidently reported, not flagged
    # low. Keep the confidence gate anyway (catches genuinely-uncertain
    # frames the plausibility cap wouldn't) and add the cap as a second,
    # independent check.
    sho_conf = (kpc[:, L_SHO] >= 0.2) & (kpc[:, R_SHO] >= 0.2)
    dsho_ok = ok_dt & sho_conf[1:] & sho_conf[:-1] & (dsho_step <= MAX_ROT_RAD)
    dsho[1:] = np.where(dsho_ok, dsho_step, 0.0)

    # gaze proxy: same angular-velocity math as dsho, on the ear line
    # instead of the shoulder line. Ears over eyes — same head-yaw
    # geometry, but a pose model localizes the ear landmark from head
    # shape/context and doesn't need to resolve anything as fine as an
    # eye at broadcast distance. Raw eye/ear/nose confidence is returned
    # separately below (feature_check.py reports it directly) so how
    # much "eyes" specifically buys here is an honest, checkable number
    # — but note confidence measures EXISTENCE ("a keypoint is probably
    # here"), not PRECISION ("this pixel is right"), so high confidence
    # on its own doesn't vindicate the signal either (same lesson as
    # dsho, right above).
    earvec = kpt[:, R_EAR] - kpt[:, L_EAR]
    earang = np.arctan2(earvec[:, 1], earvec[:, 0] + 1e-9)
    dgaze_step = np.abs(np.angle(np.exp(1j * np.diff(earang))))
    dgaze = np.zeros(n)
    ear_conf = (kpc[:, L_EAR] >= 0.2) & (kpc[:, R_EAR] >= 0.2)
    dgaze_ok = (ok_dt & ear_conf[1:] & ear_conf[:-1] &
               (dgaze_step <= MAX_ROT_RAD))
    dgaze[1:] = np.where(dgaze_ok, dgaze_step, 0.0)

    hipv = np.zeros(n)
    hipv[1:] = np.where(ok_dt,
                        np.linalg.norm(np.diff(hip, axis=0), axis=1)
                        / np.maximum(dt * fps, 0.33) / scale[1:], 0.0)
    hipv = np.minimum(hipv, 3.0)
    wrist_hi = np.clip(np.maximum(
        (sho[:, 1] - kpt[:, L_WRIST, 1]) / scale,
        (sho[:, 1] - kpt[:, R_WRIST, 1]) / scale), -3.0, 3.0)
    kconf = (kpc[:, L_WRIST] + kpc[:, R_WRIST]) / 2

    return {"t": t, "arm": arm, "lw": chans["lw"], "rw": chans["rw"],
            "le": chans["le"], "re": chans["re"], "dsho": dsho,
            "leg": leg, "dgaze": dgaze,
            "nose_c": kpc[:, NOSE], "leye_c": kpc[:, L_EYE],
            "reye_c": kpc[:, R_EYE], "lear_c": kpc[:, L_EAR],
            "rear_c": kpc[:, R_EAR],
            "hipv": hipv, "whi": wrist_hi, "kconf": kconf,
            "ynorm": box[:, 3], "cx": cx, "bh": bh}


def load_rally(pose_dir, cum):
    p = Path(pose_dir) / f"r{cum:04d}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    t, trk = np.asarray(z["t"]), np.asarray(z["track"])
    side, box = np.asarray(z["side"]), np.asarray(z["box"])
    kpt, kpc = np.asarray(z["kpt"]), np.asarray(z["kpc"])
    fps = float(np.asarray(z["fps"]).ravel()[0])
    H = int(np.asarray(z["hw"]).ravel()[0]) or 720
    tracks = {}
    for tid in np.unique(trk):
        m = trk == tid
        s = int(side[m][0])
        if s < 0 or m.sum() < 8:
            continue
        o = np.argsort(t[m])
        ser = track_series(t[m][o], box[m][o], kpt[m][o], kpc[m][o], fps)
        ser["side"] = s
        ser["H"] = H
        tracks[int(tid)] = ser
    return {"tracks": tracks, "fps": fps, "z": z,
            "bounds": (float(t.min()), float(t.max())) if len(t) else (0, 0)}


# ----------------------------------------------------------- features


def window_feats(ser, tc, pre=PRE_S, post=POST_S):
    """Feature vector for one (track, time) instance, or None if the
    window lacks coverage. Prep-arc features are computed on
    [tc-pre, tc-0.1]; strike on [tc-0.1, tc+post]."""
    t = ser["t"]
    m_all = (t >= tc - pre) & (t <= tc + post)
    if m_all.sum() < max(6, 0.5 * (pre + post) * 20):
        return None
    # JITTER-AWARE split (v2's crisp prep/strike boundary at tc-0.1
    # fought the user's tap rhythm: a slightly-late tap put the real
    # strike into "prep", voiding the ablation). EARLY = beyond any
    # plausible tap jitter; CORE = wide enough to contain the true
    # contact wherever the tap rhythm put it.
    m_early = (t >= tc - pre) & (t < tc - 0.35)
    m_core = (t >= tc - 0.35) & (t <= tc + post)
    f = []
    # per channel: [early_max, early_mean, core_max, core_mean,
    #               core_std, rise, core_tpeak]
    for ch in ("arm", "lw", "rw", "le", "re", "dsho"):
        v = ser[ch]
        em = v[m_early].max() if m_early.any() else 0.0
        eu = v[m_early].mean() if m_early.any() else 0.0
        cm = v[m_core].max() if m_core.any() else 0.0
        cu = v[m_core].mean() if m_core.any() else 0.0
        cs = v[m_core].std() if m_core.any() else 0.0
        tp = (t[m_core][np.argmax(v[m_core])] - tc) if m_core.any() else 0.0
        f += [em, eu, cm, cu, cs, cm - eu, tp]
    f += [ser["hipv"][m_all].max(), ser["hipv"][m_all].mean()]
    f += [ser["whi"][m_all].max(), ser["kconf"][m_all].mean()]
    f += [float(ser["side"]), ser["ynorm"][m_all].mean() / ser["H"],
          ser["cx"][m_all].mean() / max(ser["H"] * 16 / 9, 1.0)]
    # cadence proxies, label-free: time since this track's last big arm
    # peak, and since the other side's (alternation prior stand-in)
    pk = ser.get("_peaks")
    if pk is None:
        arm = ser["arm"]
        cands = [(t[i], arm[i]) for i in range(1, len(arm) - 1)
                 if arm[i] >= 0.05 and arm[i] >= arm[i - 1]
                 and arm[i] >= arm[i + 1]]
        pk = [x[0] for x in strongest_first(cands, REFRACTORY_S)]
        ser["_peaks"] = pk
    prev = [pt for pt in pk if pt < tc - 0.05]
    f += [min(tc - prev[-1], 3.0) if prev else 3.0]
    out = np.array(f, dtype=np.float64)
    return out if np.isfinite(out).all() else None


def strike_only(x):
    """Ablation (question 2, jitter-safe): remove the ANTICIPATION
    window — everything more than 0.35 s before the tap, which no
    plausible tap jitter can contaminate with the strike itself.
    Blocks of 7: [early_max, early_mean, core_max, core_mean, core_std,
    rise, core_tpeak] — zero early_max, early_mean, rise."""
    x = x.copy()
    for b in range(6):
        x[:, b * 7 + 0] = 0.0
        x[:, b * 7 + 1] = 0.0
        x[:, b * 7 + 5] = 0.0
    return x


# ------------------------------------------------------------- model


def fit_logreg(X, y, l2=0.02, iters=800, lr=0.15, seed=SEED):
    """v3: the v1/v2 'matmul overflow' warnings were NOT input inf —
    they were THIS function diverging (lr 0.5 with a ~l2/n penalty is
    effectively unregularized on near-separable data, so w grew without
    bound and every downstream number was confident garbage). Now:
    absolute l2, decaying lr, and a HARD abort if weights go nonfinite
    — this script prints numbers or fails loudly, never both."""
    rng = np.random.default_rng(seed)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = np.clip((X - mu) / sd, -8.0, 8.0)
    n, d = Xs.shape
    w = rng.normal(0, 0.01, d)
    b = 0.0
    # NO class weighting: it inflates the probability scale (debug on
    # synth: median dense score 0.37 — half of all peaks "confident"),
    # and everything downstream needs honest RANKING, not recall-tilted
    # probabilities. Imbalance only shifts the operating threshold,
    # which the decoder's self-calibrating reference absorbs anyway.
    sw = np.ones_like(y)
    # np.errstate: Apple's Accelerate BLAS (numpy matmul on macOS ARM)
    # sets FP error flags as a SIMD side effect even on perfectly
    # finite data — the "overflow in matmul" warnings the first three
    # runs printed were that, not real divergence (the hard finiteness
    # checks below are the actual guarantee, and they pass).
    with np.errstate(all="ignore"):
        for it in range(iters):
            z = Xs @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = (p - y) * sw
            step = lr / (1.0 + it / 200.0)
            w -= step * (Xs.T @ g / n + l2 * w)
            b -= step * g.mean()
    if not (np.isfinite(w).all() and np.isfinite(b)):
        raise RuntimeError("logistic training diverged — do not trust "
                           "any output; report this")
    return {"w": w, "b": b, "mu": mu, "sd": sd}


def predict(model, X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xs = np.clip((X - model["mu"]) / model["sd"], -8.0, 8.0)
    with np.errstate(all="ignore"):    # Accelerate flag noise; see fit
        z = Xs @ model["w"] + model["b"]
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    assert np.isfinite(p).all(), "non-finite predictions — report this"
    return p


# ----------------------------------------------------- set construction


def rally_instances(rd, contacts, whiffs, mapping):
    """Training instances for one rally. Positives: at each contact, the
    max-arm-energy track on the hitter's image side (weak assignment —
    the actor within a side is not separately labeled). Negatives:
    same-time tracks on the other side + guard-banded motion peaks and
    uniform times, all >=GUARD_S from that side's contacts and
    >=WHIFF_GUARD_S from whiffs (whiffs are real swings, never
    negatives)."""
    rng = np.random.default_rng(SEED + len(contacts))
    by_side = {0: [], 1: []}
    for tid, ser in rd["tracks"].items():
        by_side[ser["side"]].append(ser)
    side_ct = {0: [], 1: []}
    for tc, team, *_ in contacts:
        side_ct[team ^ mapping].append(tc)
    wh_t = [t for t, _ in whiffs]

    X, y = [], []
    for tc, team, *_ in contacts:
        s = team ^ mapping
        best, bf = None, None
        for ser in by_side[s]:
            f = window_feats(ser, tc)
            if f is None:
                continue
            if best is None or f[0] > best[0]:
                best, bf = f, f
        if bf is not None:
            X.append(bf)
            y.append(1)
        # matched same-instant negatives: the other side's tracks,
        # provided THEY are not near one of their own contacts
        o = 1 - s
        if all(abs(tc - c) > GUARD_S for c in side_ct[o]):
            for ser in by_side[o]:
                f = window_feats(ser, tc)
                if f is not None:
                    X.append(f)
                    y.append(0)

    t0, t1 = rd["bounds"]
    for s in (0, 1):
        cts = side_ct[s]
        for ser in by_side[s]:
            # hard negatives: this track's own motion peaks away from
            # contacts (the locomotion spikes that fooled the ceiling)
            window_feats(ser, (t0 + t1) / 2)   # ensure _peaks cached
            for pt in ser.get("_peaks", []):
                if all(abs(pt - c) > GUARD_S for c in cts) and \
                        all(abs(pt - w) > WHIFF_GUARD_S for w in wh_t):
                    f = window_feats(ser, pt)
                    if f is not None:
                        X.append(f)
                        y.append(0)
            for _ in range(max(2, len(cts) // 2)):   # easy negatives
                rt = rng.uniform(t0 + PRE_S, t1 - POST_S)
                if all(abs(rt - c) > GUARD_S for c in cts) and \
                        all(abs(rt - w) > WHIFF_GUARD_S for w in wh_t):
                    f = window_feats(ser, rt)
                    if f is not None:
                        X.append(f)
                        y.append(0)
    return X, y


def score_rally(model, rd, ablate=False):
    """Dense out-of-sample scoring: per-track score series -> peaks.
    Returns [(t, side, score)] detections."""
    dets = []
    for tid, ser in rd["tracks"].items():
        t = ser["t"]
        idx = range(0, len(t), STRIDE)
        ts, feats = [], []
        for i in idx:
            f = window_feats(ser, float(t[i]))
            if f is not None:
                ts.append(float(t[i]))
                feats.append(f)
        if not feats:
            continue
        Xd = np.stack(feats)
        if ablate:
            Xd = strike_only(Xd)
        p = predict(model, Xd)
        cands = [(ts[i], float(p[i])) for i in range(1, len(p) - 1)
                 if p[i] >= p[i - 1] and p[i] >= p[i + 1] and p[i] > 0.02]
        for tt, sc in strongest_first(cands, REFRACTORY_S):
            dets.append((tt, ser["side"], sc))
    dets.sort()
    return dets


def decode_rally(dets, s0, floor=0.02, min_gap=0.25, max_gap=3.0,
                 ghost_pen=-3.2, max_ghost=2):
    """The user's 'logic', formalized (and the spec's missing pillar):
    teams STRICTLY alternate contacts and the serve side is known from
    the log, so decode the best time-increasing, side-alternating path
    through the scored candidates instead of thresholding them
    independently. Weak dink peaks get SELECTED when it's their side's
    turn; same-side echoes get pruned by construction; occlusion gaps
    are bridged by penalized GHOSTS (parity kept, no timestamp claimed
    — still correct for counts and attribution). Returns
    [(t, side, score, n_ghosts_before)]."""
    # PRE-MERGE per side: dense scoring sprouts clusters of peaks
    # around each real swing; without this the path zigzags through
    # parity-legal cluster pairs (selftest: 47 events on 12 contacts)
    merged = []
    for s in (0, 1):
        side_c = [(t, sc) for t, sd, sc in dets if sd == s and sc >= floor]
        merged += [(t, s, sc) for t, sc in strongest_first(side_c, 0.55)]
    cands = sorted(merged)
    if not cands:
        return []
    # per-event GAIN relative to a SELF-CALIBRATING reference: the 70th
    # percentile of this rally's own merged candidate scores. Absolute
    # probability references break the moment the model's scale drifts
    # (the selftest walked through all three failure modes: log p alone
    # collapsed paths to one event, floor-relative chained every noise
    # peak, and a fixed 0.35 met a model whose median score WAS 0.37).
    # Top-tail candidates add value; the bulk cost a little — less than
    # a ghost, so a weak dink candidate still beats a blind ghost when
    # it's that side's turn.
    ref = max(float(np.quantile([sc for _, _, sc in cands], 0.70)), 0.05)
    logp = [math.log(max(sc, 1e-6)) - math.log(ref)
            for _, _, sc in cands]
    n = len(cands)
    best = [-1e18] * n
    prev = [-1] * n
    ghosts = [0] * n
    t0 = cands[0][0]
    for i, (t, s, sc) in enumerate(cands):
        # start: the serve side, near the window head (1 leading ghost ok)
        if t - t0 < 8.0:
            if s == s0:
                best[i] = max(best[i], logp[i])
            elif best[i] < logp[i] + ghost_pen:
                best[i] = logp[i] + ghost_pen
                ghosts[i] = 1
    for j in range(n):
        tj, sj, _ = cands[j]
        for i in range(j):
            if best[i] <= -1e17:
                continue
            ti, si, _ = cands[i]
            dt = tj - ti
            if dt < min_gap:
                continue
            if dt > max_gap * (max_ghost + 1):
                continue
            for g in range(0, max_ghost + 1):
                if sj != (si ^ ((1 + g) % 2)):
                    continue
                if dt > max_gap * (g + 1) or dt < min_gap * (g + 1):
                    continue
                per = dt / (g + 1)
                gap_bonus = (0.0 if 0.45 <= per <= 2.2 else
                             -1.2 if per < 0.45 and per >= 0.3 else
                             -3.0)
                cand = best[i] + logp[j] + g * ghost_pen + gap_bonus
                if cand > best[j]:
                    best[j] = cand
                    prev[j] = i
                    ghosts[j] = g
    # SPAN CONSTRAINT (the second half of the user's sentence: alternate
    # "until the rally is over"). v4 let the path STOP whenever extending
    # went net-negative, and it did exactly that in the weak dink
    # stretches — r9 decoded 1 of 29 events, ghosts never fired, because
    # quitting was free. The path must now reach the neighborhood of the
    # LAST confident candidate (self-calibrating: >= ref), so the weak
    # middle gets explained by weak candidates or paid ghosts — "nothing"
    # is no longer on the menu.
    conf_ts = [t for (t, _, sc) in cands if sc >= ref]
    t_late = max(conf_ts) if conf_ts else cands[-1][0]
    finishers = [j for j in range(n)
                 if best[j] > -1e17 and cands[j][0] >= t_late - 1.0]
    end = max(finishers, key=lambda j: best[j]) if finishers \
        else int(np.argmax(best))
    if best[end] <= -1e17:
        return []
    path = []
    i = end
    while i >= 0:
        path.append((cands[i][0], cands[i][1], cands[i][2], ghosts[i]))
        i = prev[i]
    return path[::-1]


def serve_mapping(rd, contacts):
    """Per-rally team->image-side orientation from the SERVE anchor:
    the first labeled contact is the serve, by a known team, and a
    serve is a big clean motion at a known instant — so the image side
    whose tracks carry the most arm energy within ±0.35 s of the serve
    tap is the server's side. Independent of raw-peak luck, which is
    what v1's inherited mapping rode on (the near-null rallies in the
    first real run had the flip signature). Returns (m, margin);
    side = team ^ m; margin near 1 = ambiguous, caller should fall
    back."""
    t_serve, team, *_ = sorted(contacts)[0]
    e = {0: 1e-9, 1: 1e-9}
    for ser in rd["tracks"].values():
        m = (ser["t"] >= t_serve - 0.35) & (ser["t"] <= t_serve + 0.35)
        if m.any():
            e[ser["side"]] = max(e[ser["side"]],
                                 float(ser["arm"][m].max()))
    srv_side = 0 if e[0] >= e[1] else 1
    margin = max(e.values()) / min(e.values())
    return team ^ srv_side, margin


# ---------------------------------------------------------- evaluation


def coverage_at_budget(dets, contacts, mapping, mult=2):
    flags = []
    for s in (0, 1):
        cts = [(c[0], c[2] if len(c) > 2 else "other")
               for c in contacts if (c[1] ^ mapping) == s]
        if not cts:
            continue
        ds = sorted([d for d in dets if d[1] == s], key=lambda x: -x[2])
        sel = sorted(d[0] for d in ds[:int(math.ceil(mult * len(cts)))])
        for tc, ty in cts:
            hit = any(abs(tc - dt) <= TOL_S for dt in sel)
            flags.append((ty, hit))
    return flags


def pr_curve(dets, contacts, mapping, thresholds):
    out = []
    ct_side = {}
    for tc, team, *rest in contacts:
        ct_side.setdefault(team ^ mapping, []).append(tc)
    n_ct = len(contacts)
    for th in thresholds:
        sel = [d for d in dets if d[2] >= th]
        matched_c = 0
        for s, cts in ct_side.items():
            dts = [d[0] for d in sel if d[1] == s]
            matched_c += sum(any(abs(tc - dt) <= TOL_S for dt in dts)
                             for tc in cts)
        matched_d = sum(
            any(abs(d[0] - tc) <= TOL_S for tc in ct_side.get(d[1], []))
            for d in sel)
        prec = matched_d / len(sel) if sel else float("nan")
        rec = matched_c / n_ct if n_ct else float("nan")
        out.append((th, prec, rec, len(sel)))
    return out


def run(a):
    rosters = load_rosters(Path(a.windows))
    labels = load_labels(Path(a.labels), rosters)
    rallies = {}
    for cum, d in labels.items():
        rd = load_rally(a.pose_dir, cum)
        if rd is None or not d["contacts"]:
            continue
        cands, _b = rally_candidates(rd["z"])
        _fl, m_raw = rally_coverage(d["contacts"], cands, 2, TOL_S)
        m_srv, margin = serve_mapping(rd, d["contacts"])
        m = m_srv if margin >= 1.25 else m_raw
        rallies[cum] = {"rd": rd, "contacts": d["contacts"],
                        "whiffs": d["whiffs"], "m": m,
                        "flip": m != m_raw, "margin": margin}
    if len(rallies) < 3:
        raise SystemExit("need >=3 rallies with labels + pose "
                         f"(found {len(rallies)}) — check --pose-dir")
    print(f"swing_explore: {len(rallies)} rallies, "
          f"{sum(len(r['contacts']) for r in rallies.values())} contacts "
          f"(leave-one-rally-out; EXPLORATION, not a gate)\n")

    all_flags, all_flags_ab, pr_all = [], [], {}
    per_rally, decode_stats = {}, {}
    for held in sorted(rallies):
        Xtr, ytr = [], []
        for cum, r in rallies.items():
            if cum == held:
                continue
            X, y = rally_instances(r["rd"], r["contacts"], r["whiffs"],
                                   r["m"])
            Xtr += X
            ytr += y
        Xtr = np.stack(Xtr)
        ytr = np.array(ytr, float)
        model = fit_logreg(Xtr, ytr)
        model_ab = fit_logreg(strike_only(Xtr), ytr)

        r = rallies[held]
        dets = score_rally(model, r["rd"])
        dets_ab = score_rally(model_ab, r["rd"], ablate=True)
        fl = coverage_at_budget(dets, r["contacts"], r["m"])
        fl_alt = coverage_at_budget(dets, r["contacts"], r["m"] ^ 1)
        fl_ab = coverage_at_budget(dets_ab, r["contacts"], r["m"])
        all_flags += fl
        all_flags_ab += fl_ab
        h, ha = sum(x for _, x in fl), sum(x for _, x in fl_alt)
        # label-drift sweep: slide this rally's label times and re-score.
        # A coverage peak away from tau=0 = the taps drifted (rhythm slip
        # on long rallies); a flat low curve = the stream is blind here.
        best_tau, best_cov = 0.0, h / len(fl)
        for tau in np.arange(-0.8, 0.81, 0.1):
            sc = [(tc + tau, tm, *rest) for tc, tm, *rest in r["contacts"]]
            fh = sum(x for _, x in
                     coverage_at_budget(dets, sc, r["m"]))
            if fh / len(fl) > best_cov + 1e-9:
                best_tau, best_cov = float(tau), fh / len(fl)
        # the user's alternation decode: serve side from the log, best
        # alternating path through the same scored candidates
        s0 = r["contacts"][0][1] ^ r["m"]
        path = decode_rally(dets, s0)
        real_ev = [(t, s) for t, s, _, _ in path]
        n_gh = sum(g for _, _, _, g in path)
        ct_by_side = {}
        for tc, tm, *_ in r["contacts"]:
            ct_by_side.setdefault(tm ^ r["m"], []).append(tc)
        dec_hit = sum(
            any(abs(tc - t) <= TOL_S for t, s in real_ev
                if s == (tm ^ r["m"]))
            for tc, tm, *_ in r["contacts"])
        dec_matched_ev = sum(
            any(abs(t - tc) <= TOL_S for tc in ct_by_side.get(s, []))
            for t, s in real_ev)
        decode_stats[held] = (dec_hit, len(r["contacts"]),
                              dec_matched_ev, len(real_ev), n_gh)
        per_rally[held] = (h, len(fl), max(h, ha), r["flip"],
                           ha > h, best_tau, best_cov)
        for th, p, rc, nd in pr_curve(dets, r["contacts"], r["m"],
                                      [0.3, 0.5, 0.7, 0.85]):
            agg = pr_all.setdefault(th, [0, 0, 0, 0])
            agg[0] += sum(1 for _ in r["contacts"])
            if not math.isnan(rc):
                agg[1] += rc * len(r["contacts"])
            if not math.isnan(p) and nd:
                agg[2] += p * nd
                agg[3] += nd

    n = len(all_flags)
    cov = sum(h for _, h in all_flags) / n
    cov_ab = sum(h for _, h in all_flags_ab) / len(all_flags_ab)
    print(f"LEARNED coverage@2x (out-of-sample): {cov:6.1%}   "
          f"(raw-height ceiling was 45.1%)")
    print(f"  strike-only ablation:              {cov_ab:6.1%}   "
          f"(prep-arc contribution: {cov - cov_ab:+.1%})\n")
    print("  threshold  precision  recall  (pooled)")
    for th in sorted(pr_all):
        c_tot, r_sum, p_sum, d_tot = pr_all[th]
        rec = r_sum / c_tot if c_tot else float("nan")
        prec = p_sum / d_tot if d_tot else float("nan")
        print(f"    {th:.2f}     {prec:8.1%} {rec:8.1%}")
    print("\n  per type (coverage@2x):")
    by_ty = {}
    for ty, h in all_flags:
        aa, bb = by_ty.get(ty, (0, 0))
        by_ty[ty] = (aa + h, bb + 1)
    for ty, (aa, bb) in sorted(by_ty.items(), key=lambda kv: -kv[1][1]):
        print(f"    {ty:<10} {aa:>3}/{bb:<3} {aa / bb:6.1%}")
    d_hit = sum(v[0] for v in decode_stats.values())
    d_ct = sum(v[1] for v in decode_stats.values())
    d_mev = sum(v[2] for v in decode_stats.values())
    d_ev = sum(v[3] for v in decode_stats.values())
    d_gh = sum(v[4] for v in decode_stats.values())
    print(f"\n  ALTERNATION-DECODED (the 'logic': serve side from the "
          f"log, teams alternate,\n  best side-alternating path through "
          f"the same scored candidates):")
    print(f"    decoded coverage:   {d_hit / d_ct:6.1%}   "
          f"sequence precision: {d_mev / max(d_ev, 1):6.1%}   "
          f"ghosts: {d_gh}")
    print(f"    decoded count vs labeled: {d_ev + d_gh} vs {d_ct}  "
          f"(per rally: " + ", ".join(
              f"r{c}:{v[3] + v[4] - v[1]:+d}"
              for c, v in sorted(decode_stats.items())) + ")")
    oracle = sum(v[2] for v in per_rally.values()) / n
    print(f"\n  oracle-orientation coverage@2x: {oracle:6.1%}  "
          f"(per-rally best of both team->side mappings — a DIAGNOSTIC "
          f"upper line;\n   a big gap vs the headline means orientation, "
          f"not detection, is what's failing)")
    print("\n  per rally:  (drift: coverage if this rally's labels are "
          "slid by tau —\n               a peak away from 0 means the "
          "taps drifted, not that the stream is blind)")
    for cum, (aa, bb, oo, flip, alt, bt, bc) in sorted(per_rally.items()):
        tags = ("  serve-flip" if flip else "") + \
               (f"  alt-better({oo}/{bb})" if alt else "")
        if abs(bt) > 1e-9 and bc > aa / bb + 0.08:
            tags += f"  drift tau={bt:+.1f}s -> {bc:.0%}"
        print(f"    r{cum:<3} {aa:>3}/{bb:<3} {aa / bb:6.1%}{tags}")
    print("\nEXPLORATION ONLY: dev rallies, same match, same day. If a "
          "number here\nwants to be believed, it graduates to a fresh "
          "pre-registration on\nrallies never touched by this script "
          "(contact_gate.md).")
    if a.report:
        Path(a.report).write_text(json.dumps({
            "coverage_2x": cov, "coverage_2x_strike_only": cov_ab,
            "coverage_2x_oracle": oracle,
            "decoded": {"coverage": d_hit / d_ct if d_ct else None,
                        "precision": d_mev / d_ev if d_ev else None,
                        "ghosts": d_gh, "events": d_ev},
            "per_type": {k: list(v) for k, v in by_ty.items()},
            "per_rally": {str(k): [x if not isinstance(x, np.floating)
                                   else float(x) for x in v]
                          for k, v in per_rally.items()},
            "n_contacts": n, "rallies": sorted(rallies)}, indent=1))
        print(f"\nreport -> {a.report}")


# ------------------------------------------------------------ selftest


def selftest():
    from contact_ceiling import synth_rally
    rng = np.random.default_rng(3)
    types = ["serve", "return", "dink", "dink", "speed-up", "counter",
             "smash", "dink", "counter", "dink", "drive", "dink"]
    rallies = {}
    for cum in (1, 2, 3, 4):
        # 1.1 s between contacts = realistic side-alternation cadence
        # (ball flight time); the original 2.0 s spacing was slower than
        # real rallies and left phantom-pair insertions sitting inside
        # the plausible-cadence band, testing the wrong regime
        contacts = [(103.0 + k * 1.1 + rng.normal(0, 0.05), k % 2,
                     types[k]) for k in range(12)]
        z = synth_rally(rng, [(t, team) for t, team, _ in contacts],
                        planted=True, t1=117.5)
        rd = {"tracks": {}, "fps": 30.0, "z": z,
              "bounds": (100.0, 117.5)}
        t, trk = np.asarray(z["t"]), np.asarray(z["track"])
        for tid in np.unique(trk):
            m = trk == tid
            o = np.argsort(t[m])
            ser = track_series(t[m][o], np.asarray(z["box"])[m][o],
                               np.asarray(z["kpt"])[m][o],
                               np.asarray(z["kpc"])[m][o], 30.0)
            ser["side"] = int(np.asarray(z["side"])[m][0])
            ser["H"] = 720
            rd["tracks"][int(tid)] = ser
        rallies[cum] = {"rd": rd, "contacts": sorted(contacts),
                        "whiffs": [], "m": 0}

    held = 4
    Xtr, ytr = [], []
    for cum, r in rallies.items():
        if cum == held:
            continue
        X, y = rally_instances(r["rd"], r["contacts"], r["whiffs"], r["m"])
        Xtr += X
        ytr += y
    Xtr = np.stack(Xtr)
    ytr = np.array(ytr, float)
    assert (ytr == 1).sum() >= 30 and (ytr == 0).sum() >= 30, \
        f"thin instance sets: {(ytr == 1).sum()}/{(ytr == 0).sum()}"
    m0, marg = serve_mapping(rallies[1]["rd"], rallies[1]["contacts"])
    assert m0 == 0 and marg > 1.2, f"serve mapping failed: {m0}/{marg:.2f}"
    print(f"  serve-anchored mapping recovered (margin {marg:.1f}x)")
    model = fit_logreg(Xtr, ytr)
    tr_auc = _auc(predict(model, Xtr), ytr)
    r = rallies[held]
    dets = score_rally(model, r["rd"])
    fl = coverage_at_budget(dets, r["contacts"], r["m"])
    cov = sum(h for _, h in fl) / len(fl)
    print(f"  synth: train AUC {tr_auc:.3f}, held-out coverage@2x "
          f"{cov:.1%} on {len(fl)} contacts")
    assert tr_auc > 0.9, "model failed to separate planted swings"
    assert cov >= 0.85, "held-out coverage too low on planted data"

    # null control: no planted swings -> low coverage
    z0 = synth_rally(rng, [(t, tm) for t, tm, _ in rallies[1]["contacts"]],
                     planted=False, t1=117.5)
    rd0 = {"tracks": {}, "fps": 30.0, "z": z0, "bounds": (100.0, 117.5)}
    t0a, trk0 = np.asarray(z0["t"]), np.asarray(z0["track"])
    for tid in np.unique(trk0):
        m = trk0 == tid
        o = np.argsort(t0a[m])
        ser = track_series(t0a[m][o], np.asarray(z0["box"])[m][o],
                          np.asarray(z0["kpt"])[m][o],
                          np.asarray(z0["kpc"])[m][o], 30.0)
        ser["side"] = int(np.asarray(z0["side"])[m][0])
        ser["H"] = 720
        rd0["tracks"][int(tid)] = ser
    dets0 = score_rally(model, rd0)
    fl0 = coverage_at_budget(dets0, rallies[1]["contacts"], 0)
    cov0 = sum(h for _, h in fl0) / len(fl0)
    print(f"  null control coverage@2x: {cov0:.1%} (must be << planted)")
    assert cov0 <= 0.7, "null control suspiciously covered"

    # alternation decode on the held-out planted rally
    s0 = r["contacts"][0][1] ^ 0
    path = decode_rally(dets, s0)
    ev = [(t, s) for t, s, _, _ in path]
    hit = sum(any(abs(tc - t) <= TOL_S for t, s in ev if s == tm)
              for tc, tm, *_ in r["contacts"])
    n_gh = sum(g for *_, g in path)
    cnt = len(ev) + n_gh
    print(f"  decode: {hit}/{len(r['contacts'])} covered, "
          f"count {cnt} vs {len(r['contacts'])}, ghosts {n_gh}")
    assert hit / len(r["contacts"]) >= 0.85, "decoder lost planted swings"
    assert abs(cnt - len(r["contacts"])) <= 2, "decoded count way off"

    # dsho/dgaze/leg gates, one synthetic 6-frame rig. Shoulders glitch
    # at frame 3, ears glitch at frame 4 (DIFFERENT frames) so each
    # gate's independence from the other is checked, not just that both
    # fire together; ankle gets a real small drift plus one huge,
    # low-confidence jump at frame 3 (same shape of bug as the dsho
    # ceiling, for a speed channel instead of a wrapped angle).
    n6 = 6
    tg = np.arange(n6) / 30.0
    boxg = np.tile([100., 100., 150., 300.], (n6, 1)).astype(np.float32)
    kptg = np.zeros((n6, 17, 2), np.float32)
    kpcg = np.zeros((n6, 17), np.float32)
    kptg[:, L_HIP] = kptg[:, R_HIP] = [125., 250.]
    kpcg[:, L_HIP] = kpcg[:, R_HIP] = 0.9
    sho_ang = [0.0, 0.02, 0.04, None, 0.06, 0.08]     # frame 3 glitches
    ear_ang = [0.0, 0.015, 0.03, 0.045, None, 0.075]  # frame 4 glitches
    ank_x = [110., 112., 114., None, 116., 118.]      # frame 3 glitches
    for i in range(n6):
        if sho_ang[i] is not None:
            a = sho_ang[i]
            kptg[i, L_SHO] = [125 - 20 * np.cos(a), 180 - 20 * np.sin(a)]
            kptg[i, R_SHO] = [125 + 20 * np.cos(a), 180 + 20 * np.sin(a)]
            kpcg[i, L_SHO] = kpcg[i, R_SHO] = 0.9
        else:
            kptg[i, L_SHO] = [146., 181.]    # swapped-looking position
            kptg[i, R_SHO] = [104., 179.]
            kpcg[i, L_SHO] = 0.05            # low confidence -> gated
            kpcg[i, R_SHO] = 0.9
        if ear_ang[i] is not None:
            a = ear_ang[i]
            kptg[i, L_EAR] = [125 - 15 * np.cos(a), 165 - 15 * np.sin(a)]
            kptg[i, R_EAR] = [125 + 15 * np.cos(a), 165 + 15 * np.sin(a)]
            kpcg[i, L_EAR] = kpcg[i, R_EAR] = 0.9
        else:
            kptg[i, L_EAR] = [140., 166.]
            kptg[i, R_EAR] = [110., 164.]
            kpcg[i, L_EAR] = 0.9
            kpcg[i, R_EAR] = 0.05             # low confidence -> gated
        if ank_x[i] is not None:
            for j in (L_ANK, R_ANK, L_KNEE, R_KNEE):
                kptg[i, j] = [ank_x[i], 295.]
                kpcg[i, j] = 0.9
        else:
            for j in (L_ANK, R_ANK, L_KNEE, R_KNEE):
                kptg[i, j] = [174., 295.]     # a 60px jump — huge if
                kpcg[i, j] = 0.05             # counted; low conf instead
    serg = track_series(tg, boxg, kptg, kpcg, 30.0)

    assert serg["dsho"][3] < 1e-6 and serg["dsho"][4] < 1e-6, \
        "dsho glitch frame (and the diff touching it) should gate to 0"
    assert 0.01 < serg["dsho"][1] < 0.05 and 0.01 < serg["dsho"][2] < 0.05, \
        "dsho: real gradual rotation should still register"
    assert 0.01 < serg["dsho"][5] < 0.05, \
        "dsho should recover once past the glitch frame"

    assert serg["dgaze"][4] < 1e-6 and serg["dgaze"][5] < 1e-6, \
        "dgaze glitch frame (and the diff touching it) should gate to 0"
    assert 0.005 < serg["dgaze"][1] < 0.03 and 0.005 < serg["dgaze"][2] < 0.03, \
        "dgaze: real gradual rotation should still register"
    assert serg["dgaze"][3] > 0.005, \
        "dgaze at the SHOULDER glitch frame must not be gated by it " \
        "(ears were fine there) -- the two gates must be independent"

    assert serg["leg"][3] < 1e-6 and serg["leg"][4] < 1e-6, \
        "leg glitch frame (and the diff touching it) should gate to 0, " \
        "not read the 60px jump as real speed"
    assert serg["leg"][1] > 1e-4 and serg["leg"][2] > 1e-4, \
        "leg: real small drift should still register"
    print("  dsho/dgaze/leg gates: glitches suppressed, real motion "
          "preserved, shoulder/ear gates independent OK")

    # plausibility cap: this is the case that actually happened on the
    # user's real data (2026-08-18) -- a CONFIDENTLY reported L/R swap.
    # The confidence gate above only proves low-confidence glitches are
    # caught; it says nothing about this failure mode, which is exactly
    # what got missed the first time this shipped.
    tp = np.arange(3) / 30.0
    boxp = np.tile([100., 100., 150., 300.], (3, 1)).astype(np.float32)
    kptp = np.zeros((3, 17, 2), np.float32)
    kpcp = np.zeros((3, 17), np.float32)
    kptp[:, L_HIP] = kptp[:, R_HIP] = [125., 250.]
    kpcp[:, L_HIP] = kpcp[:, R_HIP] = 0.9
    for i, a in enumerate([0.0, 0.03]):     # frames 0-1: real slow turn
        kptp[i, L_SHO] = [125 - 20 * np.cos(a), 180 - 20 * np.sin(a)]
        kptp[i, R_SHO] = [125 + 20 * np.cos(a), 180 + 20 * np.sin(a)]
        kpcp[i, L_SHO] = kpcp[i, R_SHO] = 0.9
    kptp[2, L_SHO] = [146., 181.]            # frame 2: swapped-looking
    kptp[2, R_SHO] = [104., 179.]            # position, but...
    kpcp[2, L_SHO] = kpcp[2, R_SHO] = 0.9    # ...CONFIDENTLY reported
    serp = track_series(tp, boxp, kptp, kpcp, 30.0)
    assert serp["dsho"][2] < 1e-6, \
        "a confidently-wrong swap must still be capped by plausibility " \
        "-- confidence alone passed this exact case on real data"
    assert serp["dsho"][1] > 0.01, "real slow rotation must still register"
    print("  plausibility cap: confident-but-wrong swap suppressed even "
          "though confidence alone would pass it OK")
    print("SELFTEST OK")


def _auc(p, y):
    o = np.argsort(p)
    r = np.empty_like(o, dtype=float)
    r[o] = np.arange(1, len(p) + 1)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--windows", default=WINDOWS_V4)
    ap.add_argument("--pose-dir", default=POSE_DIR)
    ap.add_argument("--report", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    run(a)


if __name__ == "__main__":
    main()
