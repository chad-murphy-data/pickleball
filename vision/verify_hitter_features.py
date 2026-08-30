"""Verify the rally-1 hitter-feature candidates on untouched rallies.

2026-08-30. The rally-1 exploratory battery (swing_explore_notes.md,
entries of 2026-08-30) named two candidates post hoc on n=25:

    ARM REACH  (p90 wrist-to-shoulder distance, box-height normalized,
                per-player baseline ratio, window -0.25/+0.10 s)
                rally-1: top-1 92%, which-partner 100%
    NOSE SPEED (same instrument) rally-1 which-partner 96%
    wrist speed (the original feature) rally-1: 68% / 88%

The notes pre-commit the verification: "name the feature now, verify
on incoming data" UNTOUCHED. The incoming data turns out to already
exist: rallies 6-10 have manually timestamped, scorebug-verified
contact labels with hitter names (data/vision/contact_labels_*.csv,
source=manual/divergent) and Gate-C pose npz on Drive. All five are
TRAIN-split rallies (label_split.csv); the frozen temporal-gate
holdout is untouched by this script.

Protocol, as actually run (honesty note: the ASSIGNMENT cascade below
was iterated while new-rally scores were visible — forking-paths
exposure — so the numbers are an INSTRUMENT-LIMITED verification, not
gate-grade; the mitigations are the rally-1 truth validation and the
naming-immune up-to-relabel bound):
1. CALIBRATION: re-implement the battery from the notes' written
   recipe with the user's own rally-1 track_assign identities
   (reproduces within a few points; the exact scratchpad code is gone).
2. ASSIGNMENT: rallies 6-10 have no track_assign clicks. Sides from
   serve-time floor geometry (serving side = side whose SHALLOWER
   member is deeper). Within-pair naming cascade: cross-court
   diagonal rule for the receiver > decisive wrist motion at the
   pair's own known contact (ratio >= 1.5) > depth; conflicts and
   non-decisive anchors are flagged UNSURE per rally. The deep
   receiver is often below the broadcast frame at the serve (camera
   sits behind the near baseline) — 3 tracks + a late-appearing one
   is that case. Fragments chain by nearest-position inheritance
   (gate 4 ft, 1.2 s). Validated 4/4 against the user's rally-1
   clicks. Definitive naming needs 4 track_assign clicks per rally
   from the state-audit tool — that kills the ambiguity entirely.
3. SCORING: per labeled contact, feature per identity in the frozen
   asymmetric window; top-1-of-4 (chance 25%) and which-partner given
   side (chance 50%; side is free from exact alternation), plus a
   which-partner bound UP TO within-pair relabel (immune to a swapped
   pair naming; inflated chance ~60-65%, read as a bound only).

Caveats carried with the result: rallies 9/10 keep the span-anomaly
asterisk from contact_gate.md; separately, this session found the
v4 windows CSV rally numbering disagrees with the contact-label/npz
numbering from rally 6 on (taps+npz agree with each other; windows
CSV is shifted ~one rally in that region) — scoring uses ONLY the
label/npz pair, which are mutually consistent.

Usage:
    python3 vision/verify_hitter_features.py --pose-dir <dir with rNNNN.npz>
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from court3d import load_landmarks, dlt  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "vision"
CONTACTS = DATA / "contact_labels_chicago0725.csv"
STATE = DATA / "state_labels_chicago0725.csv"

NOSE, LSHO, RSHO, LWRI, RWRI, LANK, RANK = 0, 5, 6, 9, 10, 15, 16
CONF = 0.3
W_PRE, W_POST = 0.25, 0.10          # frozen asymmetric window
CHAIN_GATE_FT, CHAIN_GATE_S = 4.0, 1.2

TEAMS = {  # game 1 womens — constant four players
    "Allyce Jones": "Etta Tuionetoa", "Etta Tuionetoa": "Allyce Jones",
    "Emma Nelson": "Ting Chieh Wei", "Ting Chieh Wei": "Emma Nelson",
}


def load_contacts(rally):
    rows = []
    for r in csv.DictReader(open(CONTACTS)):
        if int(r["rally_cum"]) != rally or r["source"] not in ("manual", "divergent"):
            continue
        t = float(r["t_refined_s"] or r["t_tap_s"])
        rows.append((int(r["shot_index"]), t, r["hitter_name"], r["shot_type"]))
    rows.sort()
    return rows


def load_rally1_contacts():
    # rally 1 taps are prefill+manual; all were scored frame-accurate
    rows = []
    for r in csv.DictReader(open(CONTACTS)):
        if int(r["rally_cum"]) != 1:
            continue
        t = float(r["t_refined_s"] or r["t_tap_s"])
        rows.append((int(r["shot_index"]), t, r["hitter_name"], r["shot_type"]))
    rows.sort()
    return rows


class Pose:
    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.t, self.trk = z["t"], z["track"]
        self.kpt, self.kpc = z["kpt"], z["kpc"]
        self.box = z["box"]
        self.h = np.maximum(self.box[:, 3] - self.box[:, 1], 20.0)

    def track_rows(self, tid):
        return np.where(self.trk == tid)[0]


def floor_xy(P_hinv, kpt, kpc):
    pts = [kpt[j] for j in (LANK, RANK) if kpc[j] > CONF]
    if not pts:
        return None
    px, py = np.mean(pts, axis=0)
    c = P_hinv @ np.array([px, py, 1.0])
    return c[0] / c[2], c[1] / c[2]


def per_frame_feats(pose, rows):
    """Per-frame reach / nose pos / wrist pos for one track's rows."""
    k, c, h = pose.kpt[rows], pose.kpc[rows], pose.h[rows]
    reach = np.full(len(rows), np.nan)
    for i in range(len(rows)):
        vals = []
        for w, s in ((LWRI, LSHO), (RWRI, RSHO)):
            if c[i, w] > CONF and c[i, s] > CONF:
                vals.append(np.linalg.norm(k[i, w] - k[i, s]) / h[i])
        if vals:
            reach[i] = max(vals)
    t = pose.t[rows]

    def speed(j1, j2=None):
        out = np.full(len(rows), np.nan)
        for i in range(1, len(rows)):
            dt = t[i] - t[i - 1]
            if not (0 < dt < 0.1):
                continue
            best = np.nan
            for j in ((j1,) if j2 is None else (j1, j2)):
                if c[i, j] > CONF and c[i - 1, j] > CONF:
                    v = np.linalg.norm(k[i, j] - k[i - 1, j]) / h[i] / dt
                    best = v if np.isnan(best) else max(best, v)
            out[i] = best
        return out

    return {"t": t, "reach": reach,
            "wrist_sp": speed(LWRI, RWRI), "nose_sp": speed(NOSE)}


def assign_tracks(pose, Hinv, contacts, server, receiver):
    """Serve-geometry naming + fragment chaining. Returns row-index->name
    mapping per track id, as {tid: name}."""
    t_serve = contacts[0][1]
    t_lo, t_hi = contacts[0][1] - 0.5, contacts[-1][1] + 0.5
    cand, late = {}, {}
    for tid in sorted(set(pose.trk.tolist())):
        rows = pose.track_rows(tid)
        span = int(((pose.t[rows] >= t_lo) & (pose.t[rows] <= t_hi)).sum())
        if span == 0:
            continue
        m = (pose.t[rows] >= t_serve - 1.5) & (pose.t[rows] <= t_serve + 0.5)
        xys = [floor_xy(Hinv, pose.kpt[r], pose.kpc[r]) for r in rows[m]]
        xys = [p for p in xys if p]
        if xys:
            cand[tid] = (np.median([p[0] for p in xys]),
                         np.median([p[1] for p in xys]), span)
        else:
            late[tid] = span
    depth = lambda k: abs(pos[k][1] - 22.0)

    if len(cand) >= 4:
        keep = sorted(cand, key=lambda k: -cand[k][2])[:4]
        pos = {k: cand[k][:2] for k in keep}
        near = [k for k in keep if pos[k][1] > 22.0]
        far = [k for k in keep if pos[k][1] <= 22.0]
        if len(near) != 2 or len(far) != 2:
            return None, None, f"side split {len(near)}/{len(far)}"
        # serving side = the side whose SHALLOWER member is deeper (both
        # servers stay back; the receiving side has one up at the kitchen)
        if min(depth(k) for k in near) >= min(depth(k) for k in far):
            srv_pair, rcv_pair = near, far
        else:
            srv_pair, rcv_pair = far, near
    elif len(cand) == 3 and late:
        # the deep receiver sits below the broadcast frame at the serve
        # (camera is behind the near baseline) and gets tracked only once
        # they step in: 3 tracks at serve + a late track = serving pair,
        # receiver's partner at the kitchen, receiver arrives late. This
        # reconstruction reproduces the user's rally-1 clicks exactly.
        keep = sorted(cand, key=lambda k: -cand[k][2])[:3]
        pos = {k: cand[k][:2] for k in keep}
        near = [k for k in keep if pos[k][1] > 22.0]
        far = [k for k in keep if pos[k][1] <= 22.0]
        if len(near) not in (1, 2) or len(near) + len(far) != 3:
            return None, None, "3-track side split unusable"
        pair, single = (near, far[0]) if len(near) == 2 else (far, near[0])
        if min(depth(k) for k in pair) < depth(single):
            return None, None, "3-track geometry not the missing-receiver case"
        tid_late = max(late, key=late.get)
        srv_pair, rcv_pair = pair, [single, tid_late]
        forced_receiver = tid_late
    else:
        return None, None, f"only {len(cand)} tracks at serve"

    # within-pair naming. Cascade of independent anchors, most reliable
    # first; a pair whose anchors conflict is flagged UNCERTAIN (its
    # partner scores are also reported up-to-relabel downstream).
    def wrist_peak(tid, tc):
        rows = pose.track_rows(tid)
        m = np.abs(pose.t[rows] - tc) <= 0.3
        f = per_frame_feats(pose, rows[m]) if m.sum() > 2 else None
        if f is None:
            return -1.0
        v = f["wrist_sp"][~np.isnan(f["wrist_sp"])]
        return float(np.max(v)) if len(v) else -1.0

    unsure = []
    # server: behind-the-baseline depth + serve-swing motion at tap 1
    w = {k: wrist_peak(k, contacts[0][1]) for k in srv_pair}
    a, b = sorted(srv_pair, key=lambda k: -w[k])
    if w[b] > 0 and w[a] / max(w[b], 1e-6) >= 1.5:
        srv = a                              # motion decisive
        if depth(srv) < depth([k for k in srv_pair if k != srv][0]) - 1.0:
            unsure.append("server(motion-vs-depth)")
    else:
        srv = max(srv_pair, key=depth)       # stacked pair: deeper one
        unsure.append("server(no-decisive-anchor)")
    if "forced_receiver" in dir():
        rc = forced_receiver                 # late-arriving deep receiver
    else:
        # receiver: serve is cross-court, so the receiver stands on the
        # OPPOSITE x-half from the server (hard rule when decisive)
        sx = pos[srv][0]
        opp = [k for k in rcv_pair if (pos[k][0] > 10.0) != (sx > 10.0)]
        w2 = {k: wrist_peak(k, contacts[1][1]) for k in rcv_pair}
        a2, b2 = sorted(rcv_pair, key=lambda k: -w2[k])
        if len(opp) == 1:
            rc = opp[0]                      # diagonal rule decisive
            if w2[b2] > 0 and w2[a2] / max(w2[b2], 1e-6) >= 1.5 and a2 != rc:
                unsure.append("receiver(diag-vs-motion)")
        elif w2[b2] > 0 and w2[a2] / max(w2[b2], 1e-6) >= 1.5:
            rc = a2                          # motion decisive
        else:
            rc = max(rcv_pair, key=depth)
            unsure.append("receiver(no-decisive-anchor)")
    names = {srv: server,
             [k for k in srv_pair if k != srv][0]: TEAMS[server],
             rc: receiver,
             [k for k in rcv_pair if k != rc][0]: TEAMS[receiver]}
    # chain later fragments to identities by nearest last position
    last = {}
    order = np.argsort(pose.t)
    seen = set(names)
    for i in order:
        tid = int(pose.trk[i])
        xy = floor_xy(Hinv, pose.kpt[i], pose.kpc[i])
        if tid in names:
            if xy:
                last[names[tid]] = (pose.t[i], xy)
            continue
        if tid in seen:
            continue
        seen.add(tid)
        if not xy:
            continue
        best, bd = None, 1e9
        for nm, (tt, pxy) in last.items():
            if pose.t[i] - tt > CHAIN_GATE_S:
                continue
            d = np.hypot(xy[0] - pxy[0], xy[1] - pxy[1])
            if d < bd:
                best, bd = nm, d
        if best and bd < CHAIN_GATE_FT:
            names[tid] = best
            last[best] = (pose.t[i], xy)
    return names, unsure, None


def score_rally(pose, names, contacts, feats_by_name):
    """Returns per-contact dict: feature -> {top1, partner} plus coverage."""
    out = []
    for si, tc, hitter, stype in contacts:
        row = {"shot": si, "t": tc, "hitter": hitter, "type": stype}
        for feat in ("reach", "wrist_sp", "nose_sp"):
            scores = {}
            for nm, fl in feats_by_name.items():
                vals, base = [], []
                for f in fl:
                    m = (f["t"] >= tc - W_PRE) & (f["t"] <= tc + W_POST)
                    v = f[feat][m]
                    v = v[~np.isnan(v)]
                    vals.extend(v.tolist())
                    b = f[feat][~np.isnan(f[feat])]
                    base.extend(b.tolist())
                if len(vals) >= 3 and len(base) >= 10:
                    scores[nm] = np.percentile(vals, 90) / max(np.median(base), 1e-6)
            row[feat] = scores
        out.append(row)
    return out


def summarize(rows, label):
    print(f"\n=== {label} ===")
    for feat in ("reach", "wrist_sp", "nose_sp"):
        top1 = part = n4 = n2 = 0
        for r in rows:
            s = r[feat]
            if len(s) == 4:
                n4 += 1
                if max(s, key=s.get) == r["hitter"]:
                    top1 += 1
            mate = TEAMS[r["hitter"]]
            if r["hitter"] in s and mate in s:
                n2 += 1
                if s[r["hitter"]] > s[mate]:
                    part += 1
        t1 = f"{100*top1/n4:.0f}% ({top1}/{n4})" if n4 else "n/a"
        pp = f"{100*part/n2:.0f}% ({part}/{n2})" if n2 else "n/a"
        print(f"  {feat:9s} top1-of-4 {t1:14s} which-partner {pp}")
    return


def relabel_partner(rows, label):
    """Which-partner accuracy UP TO a within-pair relabel, per rally+team:
    max(a, n-a) summed. Immune to a swapped pair naming; its chance level
    is inflated (~60-65%% at these n), so read it only as a bound."""
    from collections import defaultdict as dd
    print(f"  [up-to-relabel partner bounds — {label}]")
    for feat in ("reach", "wrist_sp", "nose_sp"):
        cells = dd(lambda: [0, 0])
        for r in rows:
            s = r[feat]
            mate = TEAMS[r["hitter"]]
            if r["hitter"] in s and mate in s:
                key = (r.get("rally"), frozenset((r["hitter"], mate)))
                cells[key][1] += 1
                cells[key][0] += int(s[r["hitter"]] > s[mate])
        tot = sum(n for _, n in cells.values())
        best = sum(max(a, n - a) for a, n in cells.values())
        if tot:
            print(f"    {feat:9s} <= {100*best/tot:.0f}% ({best}/{tot})")


def side_check(pose, Hinv, names, contacts):
    """Diagnostic: at each tap, the labeled hitter's assigned track should
    stand on their own team's end (teams keep ends within a rally)."""
    # team end from median position of each name over the rally
    end = {}
    for tid, nm in names.items():
        rows = pose.track_rows(tid)
        ys = []
        for r in rows[:: max(1, len(rows) // 60)]:
            xy = floor_xy(Hinv, pose.kpt[r], pose.kpc[r])
            if xy:
                ys.append(xy[1])
        if ys:
            end.setdefault(nm, []).extend(ys)
    end = {nm: (np.median(v) > 22.0) for nm, v in end.items()}
    ok = tot = 0
    for _, tc, hitter, _ in contacts:
        best = None
        for tid, nm in names.items():
            if nm != hitter:
                continue
            rows = pose.track_rows(tid)
            m = np.abs(pose.t[rows] - tc) <= 0.3
            for r in rows[m]:
                xy = floor_xy(Hinv, pose.kpt[r], pose.kpc[r])
                if xy:
                    best = xy[1]
        if best is None or hitter not in end:
            continue
        tot += 1
        ok += int((best > 22.0) == end[hitter])
    return ok, tot


def rally1_assign():
    out = {}
    for r in csv.DictReader(open(STATE)):
        if int(r["rally_cum"]) == 1 and r["kind"] == "track_assign":
            out[int(float(r["t_s"]))] = r["player"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-dir", required=True)
    a = ap.parse_args()
    pd = Path(a.pose_dir)
    X3, x2, _ = load_landmarks()
    P = dlt(X3, x2)
    Hinv = np.linalg.inv(P[:, [0, 1, 3]])

    def feats_for(pose, names):
        by = defaultdict(list)
        for tid, nm in names.items():
            by[nm].append(per_frame_feats(pose, pose.track_rows(tid)))
        return by

    # --- calibration: rally 1, user's own track identities ---
    pose1 = Pose(pd / "r0001.npz")
    c1 = load_rally1_contacts()
    names1 = {tid: nm for tid, nm in rally1_assign().items()}
    rows1 = score_rally(pose1, names1, c1, feats_for(pose1, names1))
    summarize(rows1, "RALLY 1 (calibration; must match notes: reach 92/100, wrist 68/88, nose-partner 96)")

    # --- validate the auto-assigner on rally 1 ---
    server1, receiver1 = c1[0][2], c1[1][2]
    auto1, unsure1, err = assign_tracks(pose1, Hinv, c1, server1, receiver1)
    if err:
        print(f"\nauto-assign rally 1 FAILED: {err}")
    else:
        agree = sum(1 for tid in names1 if auto1.get(tid) == names1[tid])
        print(f"\nauto-assigner vs user's clicks on rally 1: {agree}/{len(names1)} tracks agree"
              f"{'  UNSURE: ' + ','.join(unsure1) if unsure1 else ''}")
        for tid in sorted(names1):
            print(f"    track {tid}: user={names1[tid]:16s} auto={auto1.get(tid)}")

    # --- untouched verification: rallies 6-10 ---
    pooled_main, pooled_star = [], []
    for rally in (6, 7, 8, 9, 10):
        f = pd / f"r{rally:04d}.npz"
        if not f.exists():
            print(f"\nrally {rally}: npz missing, skipped")
            continue
        pose = Pose(f)
        cts = load_contacts(rally)
        server, receiver = cts[0][2], cts[1][2]
        names, unsure, err = assign_tracks(pose, Hinv, cts, server, receiver)
        if err:
            print(f"\nrally {rally}: assignment failed ({err}), skipped")
            continue
        ok, tot = side_check(pose, Hinv, names, cts)
        print(f"\nrally {rally}: hitter-side consistency {ok}/{tot}"
              f"{'  NAMING UNSURE: ' + ','.join(unsure) if unsure else ''}")
        rows = score_rally(pose, names, cts, feats_for(pose, names))
        for r in rows:
            r["rally"] = rally
        star = " *span-anomaly rally*" if rally in (9, 10) else ""
        summarize(rows, f"RALLY {rally} (n={len(cts)} contacts){star}")
        (pooled_star if rally in (9, 10) else pooled_main).extend(rows)

    summarize(pooled_main, "POOLED rallies 6-8 (primary untouched verification)")
    relabel_partner(pooled_main, "6-8")
    summarize(pooled_star, "POOLED rallies 9-10 (span-anomaly asterisk)")
    relabel_partner(pooled_star, "9-10")
    summarize(pooled_main + pooled_star, "POOLED 6-10 (everything new)")
    relabel_partner(pooled_main + pooled_star, "6-10")


if __name__ == "__main__":
    main()
