"""Learned claimer — is this moment a contact?  (2026-09-05)

The shipped claim step (`ball_replicate.claim_bounds`) is a rule: a pose
anchor claims its largest-angle ball turn within MATCH_S, and the anchor
generator behind it (`hitter_chain.predict_contacts`) is a peak picker
with three hand-set constants.  Nothing in that chain is fitted to a
contact label.  The bound oracle (`bound_oracle.py`) showed the bounce
counter loses its bounces in exactly that step: 17 of 79 contacts
unclaimed AND 34 junk bounds, and the first rule-level fix (anchor-only
bounds) traded recall for junk and lost intact flights.

This is the first claimer that LEARNS the call.  Candidates = every ball
turn and every pose anchor; each candidate gets features from BOTH
channels at that instant (turn geometry, path coverage, ball speed in
and out, pose excitement of the most and second-most excited tracks,
ball-to-paddle distance, which side is swinging vs which way the ball
goes next, anchor density); the label is "within MATCH_S of a human
contact tap".  A gradient-boosted classifier scores candidates, a 0.30 s
non-max suppression turns scores into bounds, and the threshold is
chosen on the TRAINING rallies by the intact-flight count -- the number
the oracle says the bounce counter actually depends on.

Rules kept: labels are the owner's contact taps only (never tracker
output); r9/r10 are READ once with a model fitted on the other rallies
and a threshold fixed there (no knob tuned on them); r20 untouched; the
temporal-gate holdout (rallies 22+) never loaded.

    python3 vision/ballsearch/claimer.py --loro          # train rallies, leave-one-rally-out
    python3 vision/ballsearch/claimer.py --read-eval     # ONE read of r9/r10
    python3 vision/ballsearch/claimer.py --save          # write claimer_bounds_r{N}.json
                                                         #   for bound_oracle --bounds claimer

Grades printed per rally: contacts matched (within MATCH_S), junk bounds,
intact bounce flights (bound_oracle.intact_flights), for the shipped
claim and for the learned one.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import ball_replicate as br                    # noqa: E402
import hitter_chain as hc                      # noqa: E402
import bound_oracle as bo                      # noqa: E402
from claim_lab import load as c3load           # noqa: E402

TRAIN = [2, 3, 4, 5, 6, 7, 17]     # contact taps: r6/r7/r17 manual, r2-r5 prefill
EVAL = [9, 10]                     # read once; never tuned on
NMS_S = 0.30                       # one bound per contact
MERGE_S = 0.08                     # turn and anchor this close = one candidate
TAUS = [round(0.2 + 0.05 * i, 2) for i in range(13)]   # 0.20 .. 0.80
FEATS = ["has_turn", "turn_ang", "dt_turn", "cov15", "cov50", "v_in",
         "v_out", "logv", "dy_pre", "dy_post", "bshape", "exc1", "exc2",
         "has_anchor", "dt_anchor", "anchor_z", "n_anchor25",
         "n_anchor60", "d_anchor_paddle", "d_exc_paddle", "side_dy_post",
         "side_dy_pre", "t_from_serve", "t_to_end", "dt_prev", "dt_next"]


# ------------------------------------------------------------ inputs

def load_rally(r):
    c = c3load(r)
    obs, sh_bounds, sh_evs, anchors = bo.predem(c, "raw")
    import pickle
    with open(HERE / f"bound_oracle_predem_r{r}.pkl", "rb") as f:
        d = pickle.load(f)
    timing = sorted(d["timing_ref"])
    angs = d.get("angs") or br.turn_angles(timing, list(d["turns"]))
    R = dict(r=r, obs=obs, timing=timing, turns=list(d["turns"]),
             angs=dict(angs), anchors=list(anchors), zs=list(c["zs"]),
             floors=c["floors"], sides=br.track_sides(c["floors"]),
             npz=c["npz"], imps=list(c["imps"]), end=float(c["dead"]),
             h_bnc=bo.human_bounces(c), sh_bounds=list(sh_bounds),
             sh_evs=list(sh_evs), P=c["P"])
    R["serve"] = R["imps"][0]
    R["pose"] = pose_signals(R["npz"])
    return R


def pose_signals(npz_path):
    """Per-track smoothed excitement (predict_contacts' signal) plus the
    paddle point and box height at every sample."""
    z = np.load(npz_path)
    tids = sorted(set(z["track"].tolist()),
                  key=lambda k: -(z["track"] == k).sum())[:4]
    out = {}
    for tid in tids:
        t, sp, re, wx, wy, pxa, pya = hc.track_signals(z, tid)
        m = np.where(z["track"] == tid)[0]
        box = z["box"][m]
        h = np.maximum(box[:, 3] - box[:, 1], 20.0)
        exc = np.nanmax(np.vstack([hc.zn(sp), hc.zn(re)]), axis=0)
        o = np.argsort(t)
        t, exc, pxa, pya, h = t[o], exc[o], pxa[o], pya[o], h[o]
        sm = np.copy(exc)
        for i in range(len(t)):
            v = exc[np.abs(t - t[i]) <= hc.SMOOTH_S]
            v = v[~np.isnan(v)]
            sm[i] = v.mean() if len(v) else np.nan
        out[int(tid)] = (t, sm, pxa, pya, h)
    return out


# ---------------------------------------------------------- features

def _nearest(ts, t, tol):
    if not len(ts):
        return None
    i = int(np.argmin(np.abs(ts - t)))
    return i if abs(ts[i] - t) <= tol else None


def candidates(R):
    """Turn times, plus anchor times not within MERGE_S of a turn."""
    lo, hi = R["serve"] - 0.3, R["end"] - 0.05
    ts = [e for e in R["turns"] if lo <= e < hi]
    for a in R["anchors"]:
        if lo <= a[0] < hi and not any(abs(a[0] - e) <= MERGE_S for e in ts):
            ts.append(a[0])
    return sorted(ts)


def features(R, t, cands):
    pts = R["timing"]
    tt = np.array([p[0] for p in pts])
    f = {}
    # ---- ball channel
    near_turns = [(abs(e - t), e) for e in R["turns"] if abs(e - t) <= br.MATCH_S]
    f["has_turn"] = float(any(d <= 0.12 for d, _ in near_turns))
    f["turn_ang"] = max([R["angs"].get(e, 0.0) for _, e in near_turns] or [0.0])
    f["dt_turn"] = min([d for d, _ in near_turns] or [0.5])
    f["cov15"] = float(np.sum(np.abs(tt - t) <= 0.15))
    f["cov50"] = float(np.sum(np.abs(tt - t) <= 0.50))

    def seg_vel(a, b):
        w = [p for p in pts if a <= p[0] <= b]
        if len(w) < 2 or w[-1][0] - w[0][0] < 1e-3:
            return np.nan, np.nan
        dt = w[-1][0] - w[0][0]
        return (math.hypot(w[-1][1] - w[0][1], w[-1][2] - w[0][2]) / dt,
                (w[-1][2] - w[0][2]) / dt)
    f["v_in"], f["dy_pre"] = seg_vel(t - 0.20, t - 0.02)
    f["v_out"], f["dy_post"] = seg_vel(t + 0.02, t + 0.20)
    f["logv"] = (math.log((f["v_out"] + 1) / (f["v_in"] + 1))
                 if not (np.isnan(f["v_in"]) or np.isnan(f["v_out"])) else np.nan)
    f["bshape"] = float(br.bounce_shaped(pts, t))
    ib = _nearest(tt, t, 0.10)
    ball = (pts[ib][1], pts[ib][2]) if ib is not None else None

    # ---- pose channel: excitement of every main track at t
    exc = []
    for tid, (pt, sm, pxa, pya, h) in R["pose"].items():
        i = _nearest(pt, t, 0.05)
        if i is None or np.isnan(sm[i]):
            continue
        exc.append((float(sm[i]), tid, pxa[i], pya[i], h[i]))
    exc.sort(reverse=True)
    f["exc1"] = exc[0][0] if exc else np.nan
    f["exc2"] = exc[1][0] if len(exc) > 1 else np.nan
    f["d_exc_paddle"] = np.nan
    side = 0
    if exc and ball is not None and not np.isnan(exc[0][2]):
        f["d_exc_paddle"] = math.hypot(ball[0] - exc[0][2], ball[1] - exc[0][3]) / exc[0][4]
    if exc:
        side = R["sides"].get(int(exc[0][1]), 0)
    f["side_dy_post"] = side * np.sign(f["dy_post"]) if not np.isnan(f["dy_post"]) else 0.0
    f["side_dy_pre"] = side * np.sign(f["dy_pre"]) if not np.isnan(f["dy_pre"]) else 0.0

    # ---- anchors
    near_a = [(abs(a[0] - t), a, z) for a, z in zip(R["anchors"], R["zs"])
              if abs(a[0] - t) <= br.MATCH_S]
    f["has_anchor"] = float(bool(near_a))
    f["n_anchor25"] = float(len(near_a))
    f["n_anchor60"] = float(sum(1 for a in R["anchors"] if abs(a[0] - t) <= 0.60))
    f["dt_anchor"], f["anchor_z"], f["d_anchor_paddle"] = 0.5, np.nan, np.nan
    if near_a:
        d, a, z = min(near_a, key=lambda x: x[0])
        f["dt_anchor"], f["anchor_z"] = d, z
        if ball is not None:
            f["d_anchor_paddle"] = math.hypot(ball[0] - a[4], ball[1] - a[5])

    # ---- context
    f["t_from_serve"] = min(t - R["serve"], 2.0)
    f["t_to_end"] = min(R["end"] - t, 2.0)
    i = cands.index(t)
    f["dt_prev"] = min(t - cands[i - 1], 1.0) if i > 0 else 1.0
    f["dt_next"] = min(cands[i + 1] - t, 1.0) if i + 1 < len(cands) else 1.0
    return [f[k] for k in FEATS]


END_FEATS = ("t_from_serve", "t_to_end")   # the rally window's two ends
DROP = set()                                 # --no-end-feats fills this


def panel(R):
    cands = candidates(R)
    X = np.array([features(R, t, cands) for t in cands], float)
    if DROP:
        keep = [i for i, k in enumerate(FEATS) if k not in DROP]
        X = X[:, keep]
    y = np.array([bo.is_real(t, R["imps"]) for t in cands], int)
    return cands, X, y


# ------------------------------------------------------------- model

def fit(X, y, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                       learning_rate=0.05,
                                       min_samples_leaf=8, l2_regularization=1.0,
                                       random_state=seed)
    m.fit(X, y)
    return m


def to_bounds(cands, p, tau, R):
    keep = []
    for t, pr in sorted(zip(cands, p), key=lambda x: -x[1]):
        if pr >= tau and all(abs(t - k) >= NMS_S for k in keep):
            keep.append(t)
    keep = sorted(k for k in keep if k < R["end"] - 0.05)
    return keep + [R["end"]]


def grade(bounds, R):
    imps = R["imps"]
    cm = sum(1 for hc_ in imps if bo.is_real(hc_, bounds))
    junk = sum(1 for b in bounds[:-1] if not bo.is_real(b, imps))
    intact = bo.intact_flights(bounds, imps, R["h_bnc"])
    return cm, len(imps), junk, intact, len(R["h_bnc"])


def pick_tau(models_cands, rallies, taus=TAUS):
    """Threshold maximising the summed intact-flight count over the given
    rallies (tie: contacts minus junk), using out-of-sample probabilities
    already computed per rally."""
    best = None
    for tau in taus:
        tot_i = tot_cj = 0
        for R in rallies:
            cands, p = models_cands[R["r"]]
            cm, n, junk, intact, nb = grade(to_bounds(cands, p, tau, R), R)
            tot_i += intact
            tot_cj += cm - junk
        key = (tot_i, tot_cj)
        if best is None or key > best[0]:
            best = (key, tau)
    return best[1], best[0]


def fmt(g):
    cm, n, junk, intact, nb = g
    return f"{cm:>3}/{n:<3} {junk:>4} {intact:>3}/{nb:<3}"


def evs_for(bounds, R):
    return [e for e in R["turns"] if not any(abs(e - b) <= br.MATCH_S for b in bounds)
            and br.bounce_shaped(R["timing"], e)]


# ---------------------------------------------------------------- runs

def loro(train=TRAIN, save=False, seed=0):
    from sklearn.metrics import roc_auc_score
    Rs = {r: load_rally(r) for r in train}
    P = {r: panel(Rs[r]) for r in train}
    print(f"panel: " + ", ".join(f"r{r} {len(P[r][0])}c/{int(P[r][2].sum())}+"
                                 for r in train))
    # out-of-fold probabilities for every train rally
    oof = {}
    for r in train:
        X = np.vstack([P[q][1] for q in train if q != r])
        y = np.concatenate([P[q][2] for q in train if q != r])
        m = fit(X, y, seed)
        oof[r] = (P[r][0], m.predict_proba(P[r][1])[:, 1])
    allp = np.concatenate([oof[r][1] for r in train])
    ally = np.concatenate([P[r][2] for r in train])
    print(f"out-of-fold AUC {roc_auc_score(ally, allp):.3f} "
          f"(n={len(ally)}, positives {int(ally.sum())})")
    # nested threshold: for held-out r, tau from the OTHER rallies' oof
    print(f"\n{'rally':>6} {'tau':>5} | {'shipped: cont junk intact':^28} | "
          f"{'claimer: cont junk intact':^28}")
    tot = np.zeros((2, 3), int)
    out = {}
    for r in train:
        others = [Rs[q] for q in train if q != r]
        tau, _ = pick_tau(oof, others)
        b = to_bounds(oof[r][0], oof[r][1], tau, Rs[r])
        gs, gc = grade(Rs[r]["sh_bounds"], Rs[r]), grade(b, Rs[r])
        for k, g in enumerate((gs, gc)):
            tot[k] += [g[0], g[2], g[3]]
        print(f"r{r:<5} {tau:>5.2f} | {fmt(gs):^28} | {fmt(gc):^28}")
        out[r] = dict(bounds=b, evs=evs_for(b, Rs[r]), tau=tau)
    nb = sum(len(Rs[r]["h_bnc"]) for r in train)
    nc = sum(len(Rs[r]["imps"]) for r in train)
    print(f"{'total':>6} {'':>5} | {tot[0][0]:>3}/{nc:<3} {tot[0][1]:>4} "
          f"{tot[0][2]:>3}/{nb:<3}" + " " * 9 + f"| {tot[1][0]:>3}/{nc:<3} "
          f"{tot[1][1]:>4} {tot[1][2]:>3}/{nb:<3}")
    if save:
        for r, d in out.items():
            with open(HERE / f"claimer_bounds_r{r}.json", "w") as f:
                json.dump(d, f)
        print("saved claimer_bounds_r*.json for the train rallies (LORO bounds)")
    return Rs, P, oof


def read_eval(train=TRAIN, ev=EVAL, save=False, seed=0):
    """ONE read of r9/r10: model on all train rallies, tau fixed on the
    train rallies' out-of-fold bounds."""
    Rs, P, oof = loro(train, save=save, seed=seed)   # --save: train LORO bounds too
    tau, key = pick_tau(oof, [Rs[r] for r in train])
    X = np.vstack([P[q][1] for q in train])
    y = np.concatenate([P[q][2] for q in train])
    m = fit(X, y, seed)
    print(f"\nREAD on r{ev} — tau {tau:.2f} fixed on train (intact {key[0]}, "
          f"cont-junk {key[1]})")
    print(f"{'rally':>6} {'tau':>5} | {'shipped: cont junk intact':^28} | "
          f"{'claimer: cont junk intact':^28}")
    for r in ev:
        R = load_rally(r)
        cands, X_, _ = panel(R)
        p = m.predict_proba(X_)[:, 1]
        b = to_bounds(cands, p, tau, R)
        print(f"r{r:<5} {tau:>5.2f} | {fmt(grade(R['sh_bounds'], R)):^28} | "
              f"{fmt(grade(b, R)):^28}")
        if save:
            with open(HERE / f"claimer_bounds_r{r}.json", "w") as f:
                json.dump(dict(bounds=b, evs=evs_for(b, R), tau=tau), f)
    if save:
        print("saved claimer_bounds_r*.json for the eval rallies (train-only model)")
    # importances: permutation on the train panel (in-sample, indicative)
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(m, X, y, n_repeats=5, random_state=0,
                                scoring="roc_auc")
    order = np.argsort(-pi.importances_mean)
    print("\nfeature importance (permutation, AUC drop, in-sample):")
    names = [k for k in FEATS if k not in DROP]
    for i in order[:12]:
        print(f"  {names[i]:16s} {pi.importances_mean[i]:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loro", action="store_true")
    ap.add_argument("--read-eval", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--manual-only", action="store_true",
                    help="train on r6/r7/r17 only (no prefill contacts)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", help="comma-separated train rallies "
                    "(default 2,3,4,5,6,7,17)")
    ap.add_argument("--no-end-feats", action="store_true",
                    help="drop t_from_serve/t_to_end: the rally window's "
                         "end is last-contact+2 s on most rallies, so "
                         "t_to_end leaks the last contact's position")
    a = ap.parse_args()
    if a.no_end_feats:
        DROP.update(END_FEATS)
    train = [6, 7, 17] if a.manual_only else TRAIN
    if a.train:
        train = [int(x) for x in a.train.split(",")]
    if a.read_eval:
        read_eval(train, save=a.save, seed=a.seed)
    else:
        loro(train, save=a.save, seed=a.seed)


if __name__ == "__main__":
    main()
