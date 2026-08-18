"""Phase-structure grader — the decision instrument for the
2000-rally question. (2026-08-18) — EXPLORATION, not a gate; the
VERDICT BANDS are pre-registered in swing_explore_notes.md BEFORE the
first real run.

WHAT IT MEASURES. The product analytics worth shipping are rally
STRUCTURE, not per-contact tags: did the rally break open (has_fast),
WHEN (first-fast time = the speed-up moment), WHICH TEAM initiated,
and how firefight-y it was (fast share). fastslow_check established
that pace lives in TIMING + context, not per-contact pose magnitude
(arm_cmax AUC 0.445, inverted). This script grades those structure
stats end-to-end, in two arms whose DIFFERENCE is the whole verdict:

  LEVEL C — ceiling arm: pace classifiers applied at TRUE labeled
    times. Placement error removed; measures whether the structure
    stats are recoverable at all on this stream.
  LEVEL B — deployment arm: the actual pipeline. LORO-trained swing
    scorer -> dense scoring -> alternation decoder (teams strictly
    alternate; serve side anchors the parity; ghosts bridge occlusion
    with ordinals preserved) -> pace classification of decoded events
    -> structure stats vs truth.

  If B ~= C and both clear the bands  -> ship on the current decoder;
                                         2000 rallies NOT needed for
                                         training (validation only).
  If C clears and B misses            -> placement is the binding
                                         constraint; the temporal
                                         model is justified and 2000
                                         rallies are its fuel.
  If C misses                         -> more labels cannot fix this
                                         footage; counts-only product.

PACE CLASSIFIERS (both trained at TRUE times on train-fold paced
contacts, applied to either arm — training at truth, deploying on
decode, is the honest deployment condition):
  GAP  — one threshold on min(gap_prev, gap_next), fit per fold.
         Label-free at inference; the floor.
  FULL — logistic on POSE-N + gaps + KITCHEN context (new here):
         per-rally net-line estimate from the two sides' track y
         distributions; hitter kitchen-proximity + all-players
         proximity at the contact instant. "Time between shots when
         players are at the kitchen" is exactly the fast/slow tell
         the user proposed, and location is a channel this stream
         measures WELL (unlike contact-instant arm magnitude).

TRUTH has holes: 'other'/untyped contacts carry no pace. first-fast
truth is flagged UNCERTAIN when any unpaced contact precedes it (an
untagged contact might have been fast earlier); headline structure
stats are reported on certain rallies, with the uncertain count
printed. Tagging the 'other' backlog fast/slow shrinks the holes.

Decoded events with ordinal 0/1 (serve/return, ghost-aware ordinals)
are openings — excluded from pace, mirroring the truth convention.
Ghost events (no timestamp) count toward ordinals, never toward pace.

V2 (2026-08-18, after run 1): adds SEQ — the model run 1's pivot
pointed at and should have shipped with. Run 1 graded PER-CONTACT
classifiers; the actual phase model is a segmentation of the whole
GAP SEQUENCE: a firefight is a RUN of short gaps, not one short gap.
SEQ is that intuition made principled — a supervised 2-state HMM:
  - one hidden state per CONTACT = the state of the gap it PRODUCES
    (a fast shot forces a fast reply, so the gap AFTER a shot is the
    shot's signature; fastslow v2 measured gap_next as the strongest
    single feature, 0.758 in the coded direction). This lands the
    phase boundary exactly ON the speed-up shot — the gap before it
    is dink-paced, the gap after it is fast — structurally fixing
    min-gap's initiator misattribution. The rally-ending shot
    produces no gap and takes its state from the transition prior
    (usually: the firefight it ended).
  - emissions: log-normal over the gap per state, fitted on the
    train fold's LABELED gaps. GAP-ONLY BY DESIGN: run 1 measured
    FULL (68 features) LOSING to a single threshold at n=84 — at
    this label count extra channels subtract, and pose/kitchen were
    each independently null. The HMM's edge is the temporal prior,
    not more features.
  - transitions fitted from adjacent labeled pairs within rallies
    (the stickiness = the run-length prior, learned not hand-set);
    decoded over the full sequence, unlabeled and opening gaps
    included as evidence. V3 readout: forward-backward POSTERIOR
    marginals at 0.5, not Viterbi — run 2 measured the MAP path
    suppressing the rare state (B has_fast 1/6, first fast 12.9 s
    late); MAP optimizes whole-path probability, which nothing here
    grades.
Level C SEQ reads out at paced contacts (comparable to gap/full
columns); Level B SEQ runs on decoded events with ghost-adjusted
gaps. Falls back to the GAP threshold if a train fold lacks
MIN_PER_CLASS labeled gaps per state.

RUN (flat folder, after fastslow_check.py works; trains the swing
scorer once per held rally — a few minutes, like channel_ablation):
    python3 phase_grader.py

SELF-TEST (no files): python3 phase_grader.py --selftest
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from contact_ceiling import load_rosters, load_labels
from swing_explore import (window_feats, fit_logreg, predict,
                           rally_instances, score_rally, decode_rally,
                           CHANNELS_BASE, CHANNELS_EXT, track_series)
from feature_check import load_split, pick_hitter
from channel_ablation import (assemble_rallies, labels_fingerprint,
                              pose_fingerprint)
from fastslow_check import (classify_type, normalize_pose, GAP_CAP_S,
                            MIN_PER_CLASS)

LABELS = "contact_labels_chicago0725.csv"
WINDOWS_V4 = "rally_windows_chicago0725_v4.csv"
POSE_DIR = "pose_rtm"
SPLIT = "label_split.csv"

MATCH_TOL_S = 0.5      # decoded-event <-> labeled-contact matching
FF_TOL_S = 1.0         # first-fast time counted as a hit within this


# --------------------------------------------------------------- truth


def truth_structure(contacts):
    """One rally's ground-truth structure from its labels.
    contacts: [(t, team, ty)]. Returns dict:
      seq: [(t, team, pace)] non-opening contacts, pace in
           {'fast','slow',None}; t_serve; n_fast/n_slow/n_unpaced/
           n_nonswing; has_fast; first_fast: (t, team) | None;
      uncertain: True if an UNJUDGED contact could change first_fast
                 (one precedes it, or has_fast is False with holes).
    Deliberate non-swings ('lunge', user rule 2026-08-18) carry pace
    None like untyped rows, but they are a JUDGMENT: a lunge cannot
    be the rally's first attack, so unlike an unjudged hole it never
    makes the boundary uncertain."""
    evs = sorted(contacts)
    t_serve = evs[0][0] if evs else None
    seq, unjudged_t = [], []
    n_nonswing = 0
    for t, team, ty in evs:
        cls = classify_type(ty)
        if cls == "position":
            continue
        pace = cls if cls in ("fast", "slow") else None
        if cls == "nonswing":
            n_nonswing += 1
        elif pace is None:
            unjudged_t.append(t)
        seq.append((t, team, pace))
    n_fast = sum(1 for *_, p in seq if p == "fast")
    n_slow = sum(1 for *_, p in seq if p == "slow")
    first_fast = next(((t, team) for t, team, p in seq if p == "fast"),
                      None)
    if first_fast is not None:
        uncertain = any(u < first_fast[0] for u in unjudged_t)
    else:
        uncertain = len(unjudged_t) > 0
    return {"seq": seq, "t_serve": t_serve, "n_fast": n_fast,
            "n_slow": n_slow, "n_unpaced": len(unjudged_t),
            "n_nonswing": n_nonswing,
            "has_fast": first_fast is not None,
            "first_fast": first_fast, "uncertain": uncertain}


# ----------------------------------------------------- kitchen context


def rally_net_y(rd):
    """(y_net, span): net-line y estimated as the midpoint between the
    two image sides' mean track y, span = half their separation. Uses
    only track data — label-free, deployment-available."""
    ys = {0: [], 1: []}
    for ser in rd["tracks"].values():
        if ser["side"] in (0, 1):
            ys[ser["side"]].append(float(np.mean(ser["ynorm"])))
    if not ys[0] or not ys[1]:
        return None, None
    y0, y1 = np.mean(ys[0]), np.mean(ys[1])
    return (y0 + y1) / 2.0, max(abs(y1 - y0) / 2.0, 1e-6)


def kitchen_prox(ser, tc, y_net, span):
    """1 at the net line, 0 a full side-separation away. The kitchen
    is the band nearest the net, so proximity ~ 'is this player up'."""
    t = ser["t"]
    i = int(np.argmin(np.abs(t - tc)))
    if abs(float(t[i]) - tc) > 0.5:
        return None
    y = float(ser["ynorm"][i])
    return float(np.clip(1.0 - abs(y - y_net) / (2.0 * span), 0.0, 1.0))


def kitchen_feats(rd, ser_hit, tc, y_net, span):
    """[hitter_prox, all_players_prox] with neutral 0.5 fallbacks."""
    if y_net is None:
        return [0.5, 0.5]
    hp = kitchen_prox(ser_hit, tc, y_net, span) if ser_hit is not None \
        else None
    pr = [p for s in rd["tracks"].values()
          if (p := kitchen_prox(s, tc, y_net, span)) is not None]
    ap = float(np.mean(pr)) if pr else None
    return [hp if hp is not None else 0.5,
            ap if ap is not None else 0.5]


# ------------------------------------------------- pace feature rows


def pace_row(rd, m, tc, team, gap_prev, gap_next, y_net, span):
    """Feature bundle for one contact instant (true or decoded).
    Returns {'x_full', 'g'} or None when the FULL features lack pose
    coverage ('g', the min-gap, is still usable by GAP)."""
    g = min(gap_prev if gap_prev is not None else GAP_CAP_S,
            gap_next if gap_next is not None else GAP_CAP_S,
            GAP_CAP_S)
    ser = pick_hitter(list(rd["tracks"].values()), team ^ m, tc)
    xp = window_feats(ser, tc, channels=CHANNELS_EXT) \
        if ser is not None else None
    if xp is None:
        return {"x_full": None, "g": g}
    cad = [min(gap_prev, GAP_CAP_S) if gap_prev is not None
           else GAP_CAP_S,
           min(gap_next, GAP_CAP_S) if gap_next is not None
           else GAP_CAP_S]
    kf = kitchen_feats(rd, ser, tc, y_net, span)
    x = np.concatenate([normalize_pose(xp, ser), cad, kf])
    return {"x_full": x, "g": g}


def true_pace_rows(r):
    """Training/ceiling rows for one assembled rally: paced contacts
    at TRUE times with any-team true gaps. [(row, pace, t, team)]."""
    ts = truth_structure(r["contacts"])
    all_t = sorted(t for t, *_ in r["contacts"])
    y_net, span = rally_net_y(r["rd"])
    out = []
    for t, team, pace in ts["seq"]:
        if pace is None:
            continue
        prev = [c for c in all_t if c < t - 1e-9]
        nxt = [c for c in all_t if c > t + 1e-9]
        row = pace_row(r["rd"], r["m"], t, team,
                       (t - prev[-1]) if prev else None,
                       (nxt[0] - t) if nxt else None, y_net, span)
        out.append((row, pace, t, team))
    return out


# ------------------------------------------------------- classifiers


def fit_gap_threshold(rows_pace):
    """Best 'fast iff min-gap <= theta' on training rows
    [(g, pace)]. Ties resolve to the smallest theta (least
    trigger-happy)."""
    if not rows_pace:
        return 0.85
    gs = sorted({g for g, _ in rows_pace})
    cands = [gs[0] - 0.01] + \
        [(a + b) / 2 for a, b in zip(gs, gs[1:])] + [gs[-1] + 0.01]
    best_th, best_acc = cands[0], -1.0
    for th in cands:
        acc = sum((g <= th) == (p == "fast")
                  for g, p in rows_pace) / len(rows_pace)
        if acc > best_acc:
            best_th, best_acc = th, acc
    return best_th


def fit_full(rows_pace_x):
    """Logistic on FULL features; rows [(x, pace)] with x not None."""
    X = np.stack([x for x, _ in rows_pace_x])
    y = np.array([p == "fast" for _, p in rows_pace_x], float)
    return fit_logreg(X, y)


def classify(row, th, model):
    """(gap_call, full_call) for one pace_row. FULL falls back to the
    GAP call when pose coverage is missing (deployment-realistic)."""
    gap_call = "fast" if row["g"] <= th else "slow"
    if model is None or row["x_full"] is None:
        return gap_call, gap_call
    p = float(predict(model, row["x_full"][None, :])[0])
    return gap_call, ("fast" if p >= 0.5 else "slow")


# ------------------------------------------------- SEQ: 2-state HMM


G_LO, G_HI = 0.15, 6.0    # gap clip for log-normal emissions
SD_FLOOR = 0.15           # log-space sd floor (degenerate-fit guard)


def gap_truth_rows(contacts):
    """One rally's gap observations with supervision: [(g, lab)] where
    gap i sits between contact i and i+1 and its label is the pace of
    the PRODUCING contact i (fast/slow, or None for opening/unpaced —
    present at inference, excluded from fitting)."""
    evs = sorted(contacts)
    rows = []
    for i in range(len(evs) - 1):
        cls = classify_type(evs[i][2])
        rows.append((evs[i + 1][0] - evs[i][0],
                     cls if cls in ("fast", "slow") else None))
    return rows


def fit_hmm(per_rally_rows):
    """Supervised 2-state HMM from labeled gaps. per_rally_rows keeps
    rallies separate so transitions never pair gaps across rallies.
    Returns None when either state has < MIN_PER_CLASS labeled gaps
    (caller falls back to the GAP threshold)."""
    lg = {"slow": [], "fast": []}
    tc = {("slow", "slow"): 1.0, ("slow", "fast"): 1.0,
          ("fast", "slow"): 1.0, ("fast", "fast"): 1.0}
    for rows in per_rally_rows:
        for i, (g, lab) in enumerate(rows):
            if lab is not None:
                lg[lab].append(math.log(min(max(g, G_LO), G_HI)))
            if i + 1 < len(rows) and lab is not None \
                    and rows[i + 1][1] is not None:
                tc[(lab, rows[i + 1][1])] += 1.0
    if min(len(lg["slow"]), len(lg["fast"])) < MIN_PER_CLASS:
        return None
    hmm = {"mu": {}, "sd": {}, "logT": {}, "logpi": {}}
    for s in ("slow", "fast"):
        v = np.array(lg[s])
        hmm["mu"][s] = float(v.mean())
        hmm["sd"][s] = max(float(v.std()), SD_FLOOR)
    for a in ("slow", "fast"):
        tot = tc[(a, "slow")] + tc[(a, "fast")]
        for b in ("slow", "fast"):
            hmm["logT"][(a, b)] = math.log(tc[(a, b)] / tot)
    n_tot = len(lg["slow"]) + len(lg["fast"])
    for s in ("slow", "fast"):
        hmm["logpi"][s] = math.log(max(len(lg[s]), 1) / n_tot)
    return hmm


def _emit(g, s, hmm):
    if g is None:
        return 0.0
    x = math.log(min(max(g, G_LO), G_HI))
    mu, sd = hmm["mu"][s], hmm["sd"][s]
    return -math.log(sd) - 0.5 * ((x - mu) / sd) ** 2


def _lse(vals):
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def posterior_fast(gaps, hmm):
    """Forward-backward P(fast) per step. V3 readout: Viterbi's MAP
    path SUPPRESSES the rare state — with overlapping emissions a
    single short gap pays back less log-likelihood than the fitted
    slow->fast transition costs, so the MAP path enters firefights
    late or never (measured on run 2: Level B has_fast 1/6, first
    fast declared 12.9 s late; Level C first-fast 1/5 at med
    +3.25 s). Marginals are the right readout for per-contact calls
    and short-run detection; MAP optimizes a loss nothing here
    grades."""
    S = ("slow", "fast")
    n = len(gaps)
    fa = [{s: hmm["logpi"][s] + _emit(gaps[0], s, hmm) for s in S}]
    for g in gaps[1:]:
        prev = fa[-1]
        fa.append({b: _lse([prev[a] + hmm["logT"][(a, b)] for a in S])
                   + _emit(g, b, hmm) for b in S})
    bw = [{s: 0.0 for s in S}]
    for g in reversed(gaps[1:]):
        nxt = bw[0]
        bw.insert(0, {a: _lse([hmm["logT"][(a, b)]
                               + _emit(g, b, hmm) + nxt[b]
                               for b in S]) for a in S})
    out = []
    for i in range(n):
        d = (fa[i]["slow"] + bw[i]["slow"]) - \
            (fa[i]["fast"] + bw[i]["fast"])
        out.append(1.0 / (1.0 + math.exp(min(max(d, -500.0), 500.0))))
    return out


def viterbi(gaps, hmm):
    """MAP state per step over gaps (None = unobserved step, scored by
    transitions alone). States are per PRODUCING contact; the caller
    appends a final None so the rally-ending shot gets a state too."""
    if not gaps:
        return []
    S = ("slow", "fast")
    score = {s: hmm["logpi"][s] + _emit(gaps[0], s, hmm) for s in S}
    back = []
    for g in gaps[1:]:
        nb, ns = {}, {}
        for b in S:
            a_best = max(S, key=lambda a: score[a] + hmm["logT"][(a, b)])
            nb[b] = a_best
            ns[b] = score[a_best] + hmm["logT"][(a_best, b)] + \
                _emit(g, b, hmm)
        back.append(nb)
        score = ns
    end = max(S, key=lambda s: score[s])
    path = [end]
    for nb in reversed(back):
        path.append(nb[path[-1]])
    return path[::-1]


def seq_states(times, hmm, th, gaps=None):
    """Pace state per event for a time-ordered sequence: the gap
    PRODUCED by event i observes step i; the last event's step is
    unobserved (transition prior decides). `gaps` overrides the raw
    time diffs — Level B passes the decoder's ghost-adjusted per-gap
    estimates. With no HMM (thin training fold), falls back to the
    GAP threshold on the produced gap, the last event inheriting its
    predecessor."""
    n = len(times)
    if n == 0:
        return []
    if gaps is None:
        gaps = [times[i + 1] - times[i] for i in range(n - 1)] + [None]
    assert len(gaps) == n
    if hmm is not None:
        return ["fast" if p >= 0.5 else "slow"
                for p in posterior_fast(gaps, hmm)]
    calls = []
    for g in gaps:
        if g is not None:
            calls.append("fast" if min(g, GAP_CAP_S) <= th else "slow")
        else:
            calls.append(calls[-1] if calls else "slow")
    return calls


# ------------------------------------------------------ decoded events


def decoded_events(path, m):
    """decode_rally path -> [{'t','team','ord','gp','gn'}]. Ordinals
    count ghosts (parity bookkeeping the decoder already did); gaps
    are per-contact, ghost-adjusted: dt spanning g ghosts is g+1
    contact gaps."""
    out = []
    ordn = 0
    for j, (t, side, _sc, g) in enumerate(path):
        ordn += g
        gp = (t - path[j - 1][0]) / (g + 1) if j > 0 else None
        if j + 1 < len(path):
            t2, _s2, _c2, g2 = path[j + 1]
            gn = (t2 - t) / (g2 + 1)
        else:
            gn = None
        out.append({"t": t, "team": side ^ m, "ord": ordn,
                    "gp": gp, "gn": gn})
        ordn += 1
    return out


def predict_structure(evs, rd, m, th, model):
    """Structure stats from decoded events under one classifier pair.
    Returns {'gap': stats, 'full': stats}; each stats = {has_fast,
    first_fast (t, team)|None, n_fast, n_slow}."""
    y_net, span = rally_net_y(rd)
    calls = {"gap": [], "full": []}
    for e in evs:
        if e["ord"] in (0, 1):
            continue
        row = pace_row(rd, m, e["t"], e["team"], e["gp"], e["gn"],
                       y_net, span)
        gc, fc = classify(row, th, model)
        calls["gap"].append((e["t"], e["team"], gc))
        calls["full"].append((e["t"], e["team"], fc))
    out = {}
    for k, seq in calls.items():
        ff = next(((t, team) for t, team, c in seq if c == "fast"),
                  None)
        out[k] = {"has_fast": ff is not None, "first_fast": ff,
                  "n_fast": sum(1 for *_, c in seq if c == "fast"),
                  "n_slow": sum(1 for *_, c in seq if c == "slow")}
    return out


# ------------------------------------------------------------ grading


def match_events(dec_evs, lab_seq, tol=MATCH_TOL_S):
    """Greedy one-to-one nearest matching, same team, |dt| <= tol.
    dec_evs: [{'t','team',...}] non-opening; lab_seq: [(t, team,
    pace)]. Returns [(i_dec, i_lab)]."""
    cands = []
    for i, e in enumerate(dec_evs):
        for j, (t, team, _p) in enumerate(lab_seq):
            if e["team"] == team and abs(e["t"] - t) <= tol:
                cands.append((abs(e["t"] - t), i, j))
    cands.sort()
    used_i, used_j, pairs = set(), set(), []
    for _d, i, j in cands:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j))
    return pairs


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def grade(truths, preds):
    """Aggregate structure grading over rallies. truths/preds:
    {cum: truth_structure dict} / {cum: stats dict (one classifier)}.
    Uncertain-boundary rallies are excluded from first-fast/team but
    kept for has_fast (uncertainty there only when has_fast False)."""
    hf_ok = hf_n = 0
    ff_err, team_ok, ff_n = [], 0, 0
    sh_t, sh_p = [], []
    for cum, tr in truths.items():
        pr = preds[cum]
        if not (tr["has_fast"] is False and tr["uncertain"]):
            hf_n += 1
            hf_ok += (pr["has_fast"] == tr["has_fast"])
        if tr["has_fast"] and not tr["uncertain"]:
            ff_n += 1
            if pr["first_fast"] is not None:
                dt = abs(pr["first_fast"][0] - tr["first_fast"][0])
                ff_err.append(dt)
                team_ok += (pr["first_fast"][1] == tr["first_fast"][1])
        den_t = tr["n_fast"] + tr["n_slow"]
        den_p = pr["n_fast"] + pr["n_slow"]
        if den_t and den_p:
            sh_t.append(tr["n_fast"] / den_t)
            sh_p.append(pr["n_fast"] / den_p)
    return {"hf_ok": hf_ok, "hf_n": hf_n,
            "ff_hit": sum(1 for d in ff_err if d <= FF_TOL_S),
            "ff_found": len(ff_err), "ff_n": ff_n,
            "ff_med": float(np.median(ff_err)) if ff_err else None,
            "team_ok": team_ok, "share_corr": pearson(sh_t, sh_p)}


def print_grade(tag, g):
    med = f"{g['ff_med']:.2f}s" if g["ff_med"] is not None else "n/a"
    corr = f"{g['share_corr']:+.2f}" if g["share_corr"] is not None \
        else "n/a"
    print(f"    {tag:<6} has_fast {g['hf_ok']}/{g['hf_n']}   "
          f"first-fast<={FF_TOL_S:.0f}s {g['ff_hit']}/{g['ff_n']} "
          f"(med {med}; found {g['ff_found']})   "
          f"init-team {g['team_ok']}/{g['ff_n']}   "
          f"fast-share corr {corr}")


# --------------------------------------------------------------- main


def run_all(rallies):
    """LORO over rallies: per held rally, train scorer + pace
    classifiers on the others, produce Level C and Level B structure
    predictions + per-contact tallies. Returns everything report()
    needs."""
    truths = {cum: truth_structure(r["contacts"])
              for cum, r in rallies.items()}
    KS = ("gap", "full", "seq")
    out = {"C": {k: {} for k in KS}, "B": {k: {} for k in KS},
           "acc": {"C": {k: [0, 0] for k in KS},
                   "A": {k: [0, 0] for k in KS}},
           "match": [0, 0], "decoded_n": {}, "no_decode": []}
    for held in sorted(rallies):
        # pace classifiers from the OTHER rallies' true-time rows
        tr_rows = []
        for cum, r in rallies.items():
            if cum != held:
                tr_rows += true_pace_rows(r)
        th = fit_gap_threshold([(row["g"], p)
                                for row, p, *_ in tr_rows])
        full_rows = [(row["x_full"], p) for row, p, *_ in tr_rows
                     if row["x_full"] is not None]
        n_f = sum(1 for _, p in full_rows if p == "fast")
        model = fit_full(full_rows) \
            if min(n_f, len(full_rows) - n_f) >= MIN_PER_CLASS else None
        hmm = fit_hmm([gap_truth_rows(r["contacts"])
                       for cum, r in rallies.items() if cum != held])

        r = rallies[held]
        # ---- LEVEL C: true times
        all_evs = sorted(r["contacts"])
        states_c = seq_states([t for t, *_ in all_evs], hmm, th)
        idx_of = {t: i for i, (t, *_r) in enumerate(all_evs)}
        held_rows = true_pace_rows(r)
        c_calls = {k: [] for k in KS}
        for row, pace, t, team in held_rows:
            gc, fc = classify(row, th, model)
            sc = states_c[idx_of[t]]
            for k, call in (("gap", gc), ("full", fc), ("seq", sc)):
                out["acc"]["C"][k][0] += (call == pace)
                out["acc"]["C"][k][1] += 1
                c_calls[k].append((t, team, call))
        for k, calls in c_calls.items():
            ff = next(((t, tm) for t, tm, c in calls if c == "fast"),
                      None)
            out["C"][k][held] = {
                "has_fast": ff is not None, "first_fast": ff,
                "n_fast": sum(1 for *_, c in calls if c == "fast"),
                "n_slow": sum(1 for *_, c in calls if c == "slow")}

        # ---- LEVEL B: scorer -> decoder -> pace
        Xtr, ytr = [], []
        for cum, rr in rallies.items():
            if cum == held:
                continue
            X, y = rally_instances(rr["rd"], rr["contacts"],
                                   rr["whiffs"], rr["m"])
            Xtr += X
            ytr += y
        scorer = fit_logreg(np.stack(Xtr), np.array(ytr, float))
        dets = score_rally(scorer, r["rd"])
        s0 = sorted(r["contacts"])[0][1] ^ r["m"]
        path = decode_rally(dets, s0)
        evs = decoded_events(path, r["m"])
        out["decoded_n"][held] = len(evs)
        if not evs:
            out["no_decode"].append(held)
            for k in KS:
                out["B"][k][held] = {"has_fast": False,
                                     "first_fast": None,
                                     "n_fast": 0, "n_slow": 0}
            continue
        stats = predict_structure(evs, r["rd"], r["m"], th, model)
        states_b = seq_states([e["t"] for e in evs], hmm, th,
                              gaps=[e["gn"] for e in evs])
        b_seq = [(e["t"], e["team"], states_b[i])
                 for i, e in enumerate(evs) if e["ord"] not in (0, 1)]
        ffb = next(((t, tm) for t, tm, c in b_seq if c == "fast"),
                   None)
        stats["seq"] = {
            "has_fast": ffb is not None, "first_fast": ffb,
            "n_fast": sum(1 for *_, c in b_seq if c == "fast"),
            "n_slow": sum(1 for *_, c in b_seq if c == "slow")}
        for k in KS:
            out["B"][k][held] = stats[k]
        # ---- LEVEL A: matched-contact pace accuracy at decoded times
        idxs = [i for i, e in enumerate(evs) if e["ord"] not in (0, 1)]
        non_open = [evs[i] for i in idxs]
        lab_seq = truths[held]["seq"]
        pairs = match_events(non_open, lab_seq)
        paced_idx = {j for j, (*_, p) in enumerate(lab_seq)
                     if p is not None}
        out["match"][0] += len([j for _i, j in pairs
                                if j in paced_idx])
        out["match"][1] += len(paced_idx)
        y_net, span = rally_net_y(r["rd"])
        for i, j in pairs:
            if j not in paced_idx:
                continue
            e = non_open[i]
            row = pace_row(r["rd"], r["m"], e["t"], e["team"],
                           e["gp"], e["gn"], y_net, span)
            gc, fc = classify(row, th, model)
            sc = states_b[idxs[i]]
            pace = lab_seq[j][2]
            for k, call in (("gap", gc), ("full", fc), ("seq", sc)):
                out["acc"]["A"][k][0] += (call == pace)
                out["acc"]["A"][k][1] += 1
    return truths, out


def report(rallies, truths, out):
    n_unc = sum(1 for t in truths.values() if t["uncertain"])
    n_hf = sum(1 for t in truths.values() if t["has_fast"])
    n_unp = sum(t["n_unpaced"] for t in truths.values())
    print(f"truth: {len(truths)} rallies, {n_hf} with a paced fast "
          f"contact; {n_unp} unpaced contacts leave {n_unc} rallies "
          f"boundary-UNCERTAIN (excluded from first-fast/team "
          f"grading; tag the 'other' backlog to shrink this)\n")
    for lvl, name in (("C", "LEVEL C — TRUE times (placement removed; "
                            "the ceiling)"),
                      ("B", "LEVEL B — DECODED pipeline (deployment "
                            "condition)")):
        print(name)
        if lvl == "C":
            for k in ("gap", "full", "seq"):
                ok, n = out["acc"]["C"][k]
                print(f"    {k:<6} per-contact pace acc "
                      f"{ok}/{n} = {ok / n:.1%}" if n else
                      f"    {k:<6} per-contact pace acc n/a")
        else:
            mt, mn = out["match"]
            print(f"    decoded events per rally: "
                  f"{[out['decoded_n'][c] for c in sorted(rallies)]}"
                  + (f"   (no decode: {out['no_decode']})"
                     if out["no_decode"] else ""))
            print(f"    paced labels matched by a decoded event "
                  f"(±{MATCH_TOL_S}s, same team): {mt}/{mn} = "
                  f"{mt / mn:.1%}" if mn else "    (no paced labels)")
            for k in ("gap", "full", "seq"):
                ok, n = out["acc"]["A"][k]
                if n:
                    print(f"    {k:<6} pace acc on matched contacts "
                          f"{ok}/{n} = {ok / n:.1%}")
        for k in ("gap", "full", "seq"):
            print_grade(k, grade(truths, out[lvl][k]))
        print()
    print("VERDICT GUIDE (bands pre-registered in "
          "swing_explore_notes.md; read in ~10pp grains at this n):\n"
          "  bands: has_fast >=80%, first-fast<=1s >=70% of certain "
          "fast rallies,\n"
          "         init-team >=75%, share corr >=+0.6 (FULL "
          "classifier)\n"
          "  B clears        -> ship on current decoder; 2000 rallies "
          "not needed for training\n"
          "  C clears, B not -> placement binds; temporal model "
          "justified, 2000 rallies are its fuel\n"
          "  C misses        -> structure not recoverable on this "
          "footage; counts-only product")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--windows", default=WINDOWS_V4)
    ap.add_argument("--pose-dir", default=POSE_DIR)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    rosters = load_rosters(Path(a.windows))
    labels = load_labels(Path(a.labels), rosters)
    split = load_split(a.split)
    if split is not None:
        labels = {c: d for c, d in labels.items()
                  if split.get(c, "train") == "train"}
        note = f"train split only ({a.split})"
    else:
        note = f"!! no {a.split} — ALL labeled rallies"
    rallies = assemble_rallies(labels, a.pose_dir)
    if len(rallies) < 3:
        raise SystemExit(f"need >=3 rallies with labels + pose "
                         f"(found {len(rallies)})")
    print(f"phase_grader: {len(rallies)} rallies (LORO; trains the "
          f"swing scorer per fold — a few minutes)")
    print(f"labels fingerprint: {labels_fingerprint(rallies)}   "
          f"pose fingerprint: {pose_fingerprint(rallies)}   ({note})\n")
    truths, out = run_all(rallies)
    report(rallies, truths, out)


# ------------------------------------------------------------ selftest


FPS = 30.0


def _mk_track(t0, t1, side, bursts=(), y=None):
    """Boxes in the REAL corner format [x1, y1, x2, y2] (track_series
    reads bh = y[:,3]-y[:,1], ynorm = y[:,3] = the FEET line — the
    court-position channel this grader leans on). Earlier synth
    builders used a flat [x,y,w,h] layout, inert for arm-only tests
    but fatal for anything positional. `y` = the feet line (box
    bottom); default puts side 0 at 380, side 1 at 520."""
    n = int((t1 - t0) * FPS)
    t = t0 + np.arange(n) / FPS
    y2 = (380.0 + side * 140.0) if y is None else float(y)
    box = np.zeros((n, 4))
    box[:, 0] = 100 + side * 400
    box[:, 1] = y2 - 160.0
    box[:, 2] = box[:, 0] + 60.0
    box[:, 3] = y2
    kpt = np.zeros((n, 17, 2))
    kpt[:, :, 0] = box[:, 0:1] + 30
    kpt[:, :, 1] = box[:, 1:2] + np.linspace(0, 150, 17)[None, :]
    kpc = np.full((n, 17), 0.9)
    for bt, amp in bursts:
        for center, a in ((bt, amp), (bt - 0.5, 0.4 * amp)):
            m = np.abs(t - center) < 0.1
            kpt[m, 9, 0] += a * np.sin(np.arange(m.sum()))
    ser = track_series(t, box, kpt, kpc, FPS)
    ser["side"] = side
    ser["H"] = 720
    return ser


def _mk_rally(contacts, amp=25.0, y_by_side=None):
    t0 = min(t for t, *_ in contacts) - 3.0
    t1 = max(t for t, *_ in contacts) + 3.0
    tracks = {}
    for side in (0, 1):
        bursts = [(t, amp) for t, team, _ in contacts if team == side]
        y = None if y_by_side is None else y_by_side[side]
        tracks[side + 1] = _mk_track(t0, t1, side, bursts, y=y)
    return {"rd": {"tracks": tracks, "z": None, "bounds": (t0, t1)},
            "contacts": sorted(contacts), "whiffs": [], "m": 0}


def _phase_contacts(base, n_slow=6, n_fast=6):
    """Dink phase -> firefight -> reset + cool-down, with PRODUCED-GAP
    physics: a shot's pace shows in the gap AFTER it (a fast shot
    forces a fast reply), so the speed-up ARRIVES on a dink-paced gap
    and PRODUCES the first 0.45 s gap (real data agrees: speed-up
    gap_prev med 0.98 s ~ dink's 0.93). v1 of this synth had the
    speed-up arriving 0.45 s after the last dink — unphysical, and it
    manufactured a fake "min-gap misattributes the initiator" lesson.
    The real min-gap failure mode is the RESET: a slow shot ARRIVING
    on a fast gap (here the dink that ends the firefight) — min-gap
    calls it fast; produced-gap semantics don't. Times JITTERED off
    the exact frame grid (seeded by base) — unjittered grid-aligned
    synths let high-dim models ride float aliasing (trap recorded
    twice on 2026-08-18)."""
    rng = np.random.default_rng(int(base))

    def j():
        return float(rng.normal(0, 0.05))

    c = [(base, 0, "serve"), (base + 1.4 + j(), 1, "return")]
    t = c[-1][0]
    for k in range(n_slow):
        t += 2.2 + j()
        c.append((t, k % 2, "dink"))
    for k in range(n_fast):
        t += (2.2 if k == 0 else 0.45) + j()
        c.append((t, (n_slow + k) % 2, "counter"))
    for k in range(3):
        t += (0.45 if k == 0 else 2.2) + j()
        c.append((t, (n_slow + n_fast + k) % 2, "dink"))
    return c


def selftest():
    # ---- truth builder: holes, uncertainty, opening exclusion
    ct = [(10.0, 0, "serve"), (11.5, 1, "return"), (13.0, 0, "dink"),
          (14.0, 1, "other"), (15.0, 0, "smash"), (15.5, 1, "dink")]
    ts = truth_structure(ct)
    assert len(ts["seq"]) == 4 and ts["t_serve"] == 10.0
    assert ts["has_fast"] and ts["first_fast"] == (15.0, 0)
    assert ts["uncertain"], "unpaced before first fast must flag"
    assert (ts["n_fast"], ts["n_slow"], ts["n_unpaced"]) == (1, 2, 1)
    ts2 = truth_structure([c for c in ct if c[2] != "other"])
    assert not ts2["uncertain"]
    ts3 = truth_structure([(10.0, 0, "serve"), (12.0, 1, "dink"),
                           (14.0, 0, "other")])
    assert not ts3["has_fast"] and ts3["uncertain"]
    # lunge = judged non-swing: excluded from pace, counted apart, and
    # NEVER a boundary hole (a lunge cannot be the first attack) —
    # unlike 'other' in the identical position
    ts4 = truth_structure([(10.0, 0, "serve"), (11.5, 1, "return"),
                           (13.0, 0, "dink"), (14.0, 1, "lunge"),
                           (15.0, 0, "smash")])
    assert ts4["first_fast"] == (15.0, 0) and not ts4["uncertain"]
    assert (ts4["n_fast"], ts4["n_slow"], ts4["n_unpaced"],
            ts4["n_nonswing"]) == (1, 1, 0, 1)
    ts5 = truth_structure([(10.0, 0, "serve"), (12.0, 1, "dink"),
                           (14.0, 0, "lunge")])
    assert not ts5["has_fast"] and not ts5["uncertain"]
    print("selftest: truth builder OK (incl. lunge rule)")

    # ---- ghost-aware ordinals and gaps
    path = [(10.0, 0, 0.9, 0), (10.9, 1, 0.8, 0), (12.7, 0, 0.7, 1)]
    evs = decoded_events(path, 0)
    assert [e["ord"] for e in evs] == [0, 1, 3]
    assert abs(evs[2]["gp"] - 0.9) < 1e-9 and evs[2]["gn"] is None
    assert evs[0]["gp"] is None and abs(evs[0]["gn"] - 0.9) < 1e-9
    assert [e["team"] for e in decoded_events(path, 1)] == [1, 0, 1]
    print("selftest: decoded events (ghost ordinals, gaps) OK")

    # ---- kitchen proximity: a track at the measured net line reads
    # ~1, one a full 4-span deep reads ~0. Self-consistent against
    # whatever y convention track_series' ynorm uses: the box-y ->
    # ynorm offset is measured from the rally's own tracks first.
    rd = _mk_rally([(103.0, 0, "dink"), (105.2, 1, "dink")],
                   y_by_side={0: 330.0, 1: 470.0})["rd"]
    y_net, span = rally_net_y(rd)
    assert y_net is not None and span > 10.0
    side0 = next(s for s in rd["tracks"].values() if s["side"] == 0)
    off = float(np.mean(side0["ynorm"])) - 330.0
    net_trk = _mk_track(100.0, 108.0, 0, y=float(y_net) - off)
    deep_trk = _mk_track(100.0, 108.0, 0,
                         y=float(y_net) - off - 4.1 * span)
    assert kitchen_prox(net_trk, 104.0, y_net, span) > 0.9
    assert kitchen_prox(deep_trk, 104.0, y_net, span) < 0.1
    assert kitchen_prox(net_trk, 300.0, y_net, span) is None
    print("selftest: kitchen proximity OK")

    # ---- gap threshold fit recovers a separating cut
    rows = [(0.4, "fast"), (0.5, "fast"), (0.6, "fast"),
            (1.4, "slow"), (1.8, "slow"), (2.2, "slow")]
    th = fit_gap_threshold(rows)
    assert 0.6 < th < 1.4, th
    assert sum((g <= th) == (p == "fast") for g, p in rows) == 6
    print("selftest: gap threshold fit OK")

    # ---- matching: greedy, one-to-one, team-respecting
    dec = [{"t": 10.0, "team": 0}, {"t": 10.4, "team": 1},
           {"t": 12.0, "team": 0}]
    lab = [(10.1, 0, "slow"), (10.5, 1, "fast"), (14.0, 0, "slow")]
    pairs = match_events(dec, lab)
    assert sorted(pairs) == [(0, 0), (1, 1)], pairs
    assert match_events([{"t": 10.1, "team": 1}],
                        [(10.1, 0, "slow")]) == []
    print("selftest: matching OK")

    # ---- grading teeth: perfect predictions score perfectly; a 2 s
    # first-fast shift and a team flip are caught; uncertain rallies
    # stay out of first-fast/team but negatives with holes stay out
    # of has_fast
    tr = {1: truth_structure(_phase_contacts(100.0)),
          2: truth_structure(_phase_contacts(200.0)),
          3: truth_structure([(300.0, 0, "serve"), (301.4, 1, "return"),
                              (303.0, 0, "dink"), (305.2, 1, "dink")])}
    ff1, ff2 = tr[1]["first_fast"], tr[2]["first_fast"]
    perfect = {1: {"has_fast": True, "first_fast": ff1,
                   "n_fast": 6, "n_slow": 9},
               2: {"has_fast": True, "first_fast": ff2,
                   "n_fast": 6, "n_slow": 9},
               3: {"has_fast": False, "first_fast": None,
                   "n_fast": 0, "n_slow": 2}}
    g = grade(tr, perfect)
    assert g["hf_ok"] == 3 and g["ff_hit"] == 2 and g["team_ok"] == 2
    broken = dict(perfect)
    broken[1] = {"has_fast": True,
                 "first_fast": (ff1[0] + 2.0, 1 - ff1[1]),
                 "n_fast": 6, "n_slow": 9}
    g2 = grade(tr, broken)
    assert g2["ff_hit"] == 1 and g2["team_ok"] == 1 and \
        g2["ff_med"] > 0.5
    print("selftest: grading teeth OK")

    # ---- viterbi against brute-force enumeration (real DP teeth):
    # every 2^n state path scored by the same emissions/transitions,
    # argmax must equal the DP's path — including an unobserved step
    hmm_t = {"mu": {"slow": math.log(1.05), "fast": math.log(0.7)},
             "sd": {"slow": 0.4, "fast": 0.4},
             "logT": {("slow", "slow"): math.log(0.8),
                      ("slow", "fast"): math.log(0.2),
                      ("fast", "slow"): math.log(0.25),
                      ("fast", "fast"): math.log(0.75)},
             "logpi": {"slow": math.log(0.6), "fast": math.log(0.4)}}
    gs = [1.3, 0.85, 0.6, None, 0.7, 1.6]
    from itertools import product as _prod
    best_sc, best_path = -1e18, None
    for pth in _prod(("slow", "fast"), repeat=len(gs)):
        sc = hmm_t["logpi"][pth[0]] + _emit(gs[0], pth[0], hmm_t)
        for i in range(1, len(gs)):
            sc += hmm_t["logT"][(pth[i - 1], pth[i])] + \
                _emit(gs[i], pth[i], hmm_t)
        if sc > best_sc:
            best_sc, best_path = sc, list(pth)
    assert viterbi(gs, hmm_t) == best_path, \
        (viterbi(gs, hmm_t), best_path)
    print("selftest: viterbi = brute force OK")

    # ---- context flips the marginal call (the HMM's whole point on
    # OVERLAPPING real-data emissions): the same 0.85 s gap reads
    # slow inside a 1.3 s dink run and fast inside a 0.6 s firefight
    st1 = viterbi([1.3, 1.3, 0.85, 1.3, 1.3], hmm_t)
    st2 = viterbi([0.6, 0.6, 0.85, 0.6, 0.6], hmm_t)
    assert st1[2] == "slow" and st2[2] == "fast", (st1, st2)
    p1 = posterior_fast([1.3, 1.3, 0.85, 1.3, 1.3], hmm_t)
    p2 = posterior_fast([0.6, 0.6, 0.85, 0.6, 0.6], hmm_t)
    assert p1[2] < 0.5 < p2[2], (p1[2], p2[2])
    print("selftest: context flips the marginal gap OK "
          "(viterbi + posterior)")

    # ---- posterior against brute-force path enumeration: marginal
    # P(fast) at step i = sum of exp(path score) over paths fast at i
    # over the total — forward-backward must reproduce it to 1e-9
    tot = [0.0] * len(gs)
    Z = 0.0
    for pth in _prod(("slow", "fast"), repeat=len(gs)):
        sc = hmm_t["logpi"][pth[0]] + _emit(gs[0], pth[0], hmm_t)
        for i in range(1, len(gs)):
            sc += hmm_t["logT"][(pth[i - 1], pth[i])] + \
                _emit(gs[i], pth[i], hmm_t)
        w = math.exp(sc)
        Z += w
        for i, s in enumerate(pth):
            if s == "fast":
                tot[i] += w
    pf = posterior_fast(gs, hmm_t)
    assert all(abs(pf[i] - tot[i] / Z) < 1e-9 for i in range(len(gs)))
    print("selftest: posterior = brute force OK")

    # ---- fit_hmm: supervised estimates are sane; unlabeled gaps are
    # excluded; transitions never pair across rallies (two rallies
    # ending fast / starting slow must not manufacture fast->slow)
    ra_ = [(2.2, "slow"), (2.2, "slow"), (2.2, "slow"),
           (0.45, "fast"), (0.45, "fast"), (0.45, "fast")]
    rb_ = [(2.1, "slow"), (1.9, None), (2.3, "slow"), (2.2, "slow"),
           (0.5, "fast"), (0.4, "fast")]
    hf_ = fit_hmm([ra_ * 3, rb_ * 3])
    assert hf_ is not None
    assert hf_["mu"]["fast"] < hf_["mu"]["slow"]
    assert hf_["logT"][("slow", "slow")] > hf_["logT"][("slow", "fast")]
    assert fit_hmm([[(2.2, "slow")] * 20]) is None   # one class only
    two = fit_hmm([[(0.45, "fast")] * 9, [(2.2, "slow")] * 9])
    # cross-rally isolation: no labeled fast->slow pair exists, so the
    # smoothed count stays at the +1 prior exactly
    assert two is not None and \
        abs(math.exp(two["logT"][("fast", "slow")]) - 1.0 / 10.0) < 1e-9
    print("selftest: fit_hmm OK (holes excluded, rally-isolated "
          "transitions)")

    # ---- seq_states: produced-gap semantics, last-event rule,
    # threshold fallback, explicit-gaps override
    times_ = [10.0, 12.2, 14.4, 14.85, 15.3]
    st = seq_states(times_, hmm_t, 1.0)
    assert st[:2] == ["slow", "slow"] and st[2] == "fast", st
    assert st[4] == "fast", "rally-ending shot inherits the firefight"
    stf = seq_states(times_, None, 1.0)
    assert stf == ["slow", "slow", "fast", "fast", "fast"], stf
    stg = seq_states([10.0, 11.0], None, 1.0, gaps=[0.5, None])
    assert stg == ["fast", "fast"], stg
    print("selftest: seq_states OK")

    # ---- end-to-end LEVEL C on synthetic phase rallies: 4 with
    # dinks -> firefight -> reset + cool-down, 1 all-dink. Derived
    # expectations under PRODUCED-GAP physics: the min-gap heuristic
    # errs on exactly the RESET each rally (arrives on a 0.45 s gap,
    # truth slow -> 64/68) while getting the initiator RIGHT (the
    # speed-up is the first short min-gap); SEQ is exact everywhere
    # including the reset (it produces a 2.2 s gap) and the
    # rally-ending dink (transition prior from a slow run) -> 68/68.
    # v1 of this test encoded a "min-gap misattributes the initiator"
    # lesson — an artifact of the old unphysical synth timing,
    # RETRACTED (real-data run 1 agreed: GAP teams 4/5 at C).
    def _synth_set():
        rallies = {c: _mk_rally(_phase_contacts(100.0 + 400.0 * c))
                   for c in (1, 2, 3, 4)}
        rng5 = np.random.default_rng(5)
        ts, t = [], 2101.4
        for k in range(8):
            t += 2.2 + float(rng5.normal(0, 0.05))
            ts.append(t)
        rallies[5] = _mk_rally(
            [(2100.0, 0, "serve"), (2101.4, 1, "return")] +
            [(ts[k], k % 2, "dink") for k in range(8)])
        return rallies

    rallies = _synth_set()
    truths, out = run_all(rallies)
    for k, want_acc in (("gap", 64), ("seq", 68)):
        gk = grade(truths, out["C"][k])
        assert gk["hf_ok"] == gk["hf_n"] == 5, (k, gk)
        assert gk["ff_hit"] == gk["ff_n"] == 4, (k, gk)
        assert gk["team_ok"] == 4, (k, gk)
        ok, n = out["acc"]["C"][k]
        assert n == 68 and ok == want_acc, (k, ok, n)
    gF = grade(truths, out["C"]["full"])
    assert gF["hf_ok"] == 5 and gF["ff_hit"] == 4 and \
        gF["team_ok"] == 4, gF
    okF, nF = out["acc"]["C"]["full"]
    assert nF == 68 and okF >= 66, (okF, nF)
    print(f"selftest: end-to-end Level C OK (gap 64/68 with the 4 "
          f"reset errors as derived; full {okF}/68; seq 68/68 — "
          f"the reset and the rally-ending shot both land right)")

    # ---- Level B ran end-to-end on the same synth (scorer -> decoder
    # -> pace incl. SEQ over ghost-adjusted gaps); structure output
    # exists for every rally and the no-decode list is empty
    # (assertions are existence/shape, not accuracy — synthetic dets
    # aren't the claim under test)
    assert not out["no_decode"]
    assert set(out["B"]["gap"]) == set(out["B"]["seq"]) == set(rallies)
    assert all(out["decoded_n"][c] > 0 for c in rallies)
    print(f"selftest: end-to-end Level B smoke OK (decoded "
          f"{[out['decoded_n'][c] for c in sorted(rallies)]} events)")

    # ---- determinism: full pipeline twice from fresh objects,
    # bit-identical aggregates (order-bug regression class)
    _truths2, out2 = run_all(_synth_set())
    assert out2["acc"] == out["acc"] and out2["match"] == out["match"]
    assert out2["decoded_n"] == out["decoded_n"]
    print("selftest: determinism OK")

    print("\nselftest: ALL OK")


if __name__ == "__main__":
    main()
