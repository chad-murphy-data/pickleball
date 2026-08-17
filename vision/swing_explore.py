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

LABELS = "contact_labels_chicago0725.csv"
WINDOWS_V4 = "rally_windows_chicago0725_v4.csv"
POSE_DIR = "pose_rtm"

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
                    ("le", L_ELB), ("re", R_ELB)):
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

    shovec = kpt[:, R_SHO] - kpt[:, L_SHO]
    shoang = np.arctan2(shovec[:, 1], shovec[:, 0] + 1e-9)
    dsho = np.zeros(n)
    dsho[1:] = np.abs(np.angle(np.exp(1j * np.diff(shoang))))
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
    pos_w = min((y == 0).sum() / max((y == 1).sum(), 1), 5.0)
    sw = np.where(y == 1, pos_w, 1.0)
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
    z = Xs @ model["w"] + model["b"]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


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
    per_rally = {}
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
        contacts = [(103.0 + k * 2.0 + rng.normal(0, 0.05), k % 2,
                     types[k]) for k in range(12)]
        z = synth_rally(rng, [(t, team) for t, team, _ in contacts],
                        planted=True)
        rd = {"tracks": {}, "fps": 30.0, "z": z,
              "bounds": (100.0, 130.0)}
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
                     planted=False)
    rd0 = {"tracks": {}, "fps": 30.0, "z": z0, "bounds": (100.0, 130.0)}
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
