"""Shot vs NON-shot pose-feature check (2026-08-18) — EXPLORATION, not
a gate. Third life of this file: v1 checked shoulder rotation only, v2
added footwork + gaze but framed everything as shot type A vs shot type
B (soft dink/counter vs committed drive/speed-up/smash medians). The
user call that ended that framing (2026-08-18): the pipeline is a
TWO-STEP process — (1) swing / no-swing, (2) shot type — and step 2 is
premature while step 1 sits at ~45-57% coverage. So this file now
measures what step 1 actually needs from a candidate channel: does it
read differently at a real contact than at an in-play moment where the
same side is NOT hitting? All shot types are pooled; "type" survives
only as a passenger column in the CSV dump for later step-2 work.

Non-shot moments are sampled with the SAME guard constants the trained
detector's negatives already use (GUARD_S / WHIFF_GUARD_S, imported
from swing_explore rather than re-invented), on a deterministic 0.5 s
grid between each rally's first and last contact — during play only,
no pre-serve standing around. Opponent-contact instants are allowed on
purpose: the other side hitting IS a non-shot moment for this side,
and rally_instances builds its matched negatives at exactly those
instants. The hitter-proxy track selection (max prep-arc arm energy)
is applied to BOTH classes, so a channel merely correlated with the
selection rule can't manufacture a fake gap.

Separation is summarized as AUC = P(random shot reading > random
non-shot reading), computed with midrank tie handling because the
gated channels emit exact zeros (a naive rank AUC would break ties by
array order and bias the number). 0.5 = the channel tells the two
classes apart not at all; 1.0 = perfectly.

None of the three channels are new extraction — all ride on keypoints
`pose_extract.py` already saves (COCO-17). `dsho` (shoulder-line
angular velocity) already feeds the learned scorer as one of six
channels; `leg` (hip-relative ankle/knee speed, same math as the arm
channel) and `dgaze` (ear-line angular velocity, same math as dsho)
live in swing_explore.track_series but are not wired into the trained
scorer — this script reads them directly. Nothing here retrains that
model; the frozen v1-v5 numbers in swing_explore_notes.md stand.

Eyes specifically: the user flagged skepticism up front given the
video (a 720p condensed broadcast VOD where `ball_visibility.py`
already found the ball findable in only 64% of in-play frames — faces
are smaller and blurrier than the ball for most of the court). Rather
than assume the answer, this script reports raw eye/ear/nose keypoint
CONFIDENCE/coverage directly — a cheap, decisive first check. The gaze
channel itself uses EARS, not eyes (same head-yaw geometry, but a pose
model localizes an ear from head shape/context and doesn't need to
resolve anything as fine as an eye at distance) — the eye-specific
numbers are reported so that substitution is visible and checkable.

RUN (same flat folder as swing_explore.py; numpy only, seconds):
    python3 feature_check.py
    python3 feature_check.py --sweep-leg       # footwork window sweep
    python3 feature_check.py --csv feature_check.csv   # per-row dump

SELF-TEST (no files):  python3 feature_check.py --selftest
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from contact_ceiling import load_rosters, load_labels, rally_candidates, rally_coverage
from swing_explore import (load_rally, serve_mapping, TOL_S, PRE_S, POST_S,
                          MAX_ROT_RAD, GUARD_S, WHIFF_GUARD_S)

LABELS = "contact_labels_chicago0725.csv"
WINDOWS_V4 = "rally_windows_chicago0725_v4.csv"
POSE_DIR = "pose_rtm"
SPLIT = "label_split.csv"

EARLY = (-PRE_S, -0.35)   # prep arc — matches window_feats' m_early
CORE = (-0.35, POST_S)    # strike — matches window_feats' m_core

NONSHOT_STEP_S = 0.5   # grid step for non-shot anchors. 0.5 s makes
# adjacent anchors' 0.40-0.55 s-wide windows ~non-overlapping, so the
# non-shot rows aren't just the same frames counted five times. Rows
# within a rally still share tracks — never read the n's as independent.

SWEEP_WIDTH = 0.40   # sliding window for leg_sweep (--sweep-leg): the
SWEEP_STARTS = [1.55, 1.35, 1.15, 0.95, 0.75]  # existing EARLY window
# was PRE_S=0.75s, chosen for an unrelated question (arm/prep-arc
# timing, weeks ago) — never re-derived for footwork specifically.
# Window = [tc-start, tc-start+SWEEP_WIDTH]; the LAST point (0.75)
# exactly reproduces EARLY as a checkpoint. Doesn't sweep past -0.35s
# (into CORE territory) -- that's the separate, already-covered
# strike-window question.

# channel key -> (display label, which window is the headline stat).
# shoulder/gaze are about the STRIKE itself (core); footwork is about
# GETTING READY to strike (early, prep arc) — both windows are always
# collected either way, this just picks which one gets summarized first.
CHANNELS = {
    "dsho": ("shoulder rotation", "core"),
    "leg": ("footwork: ankle/knee speed", "early"),
    "dgaze": ("head/gaze rotation (ear line)", "core"),
}

FACE_KEYS = ("nose_c", "leye_c", "reye_c", "lear_c", "rear_c")
FACE_LABELS = {"nose_c": "nose", "leye_c": "L eye", "reye_c": "R eye",
               "lear_c": "L ear", "rear_c": "R ear"}
FACE_THRESH = 0.2   # matches the confidence gate track_series uses on
# these same keypoints for dgaze itself


def load_split(path):
    if not Path(path).exists():
        return None
    out = {}
    for r in csv.DictReader(open(path)):
        out[int(r["rally_cum"])] = r["split"]
    return out


def window_stat(ser, tc, lo, hi, ch):
    """(max, mean) of channel `ch` inside [tc+lo, tc+hi], or None."""
    t, v = ser["t"], ser[ch]
    m = (t >= tc + lo) & (t <= tc + hi)
    if not m.any():
        return None
    return float(v[m].max()), float(v[m].mean())


def sweep_contaminated(tc, lo, prev_tc):
    """Does a window starting at tc+lo reach back past the most recent
    PREVIOUS contact (any player)? For a shot row that means measuring
    recovery from the last shot as much as prep for this one; for a
    non-shot row it means real shot movement leaking into the "quiet"
    reading. Any-player is the conservative outer bound (the actor
    within a side isn't labeled, so own-swing pollution can't be
    isolated more tightly than this)."""
    return prev_tc is not None and (tc + lo) < prev_tc


def prev_contact_before(all_ct, anchor):
    """Most recent contact time (any player) strictly before `anchor`,
    or None. For a shot anchor this returns the PREVIOUS contact, not
    itself."""
    return max((c for c in all_ct if c < anchor - 1e-9), default=None)


def auc(pos, neg):
    """P(random pos reading > random neg reading), ties counting 1/2
    (Mann-Whitney with midranks). Midranks matter here: the gated
    channels emit exact zeros, and a rank AUC that breaks ties by
    array order would systematically shade the number by however the
    two groups happened to be concatenated."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def face_coverage(ser, tc):
    """Fraction of frames in the full [tc-PRE_S, tc+POST_S] window with
    usable (>=FACE_THRESH) confidence, per face keypoint. Answers "how
    much do we even have here" independent of any behavioral question."""
    t = ser["t"]
    m = (t >= tc - PRE_S) & (t <= tc + POST_S)
    if not m.any():
        return None
    return {k: float((ser[k][m] >= FACE_THRESH).mean()) for k in FACE_KEYS}


def pick_hitter(tracks, side, tc):
    """Same convention as swing_explore.rally_instances: the track on
    the given image side with the most PREP-ARC arm energy — the actor
    within a side isn't separately labeled, so this is the model's own
    hitter proxy, reused here rather than re-invented. Applied to
    non-shot anchors too (where it just picks the most-active track at
    a quiet moment) so both classes go through the identical selection
    rule."""
    best, best_e = None, -1.0
    for ser in tracks:
        if ser["side"] != side:
            continue
        t, arm = ser["t"], ser["arm"]
        m = (t >= tc - PRE_S) & (t < tc - 0.35)
        e = float(arm[m].max()) if m.any() else -1.0
        if e > best_e:
            best, best_e = ser, e
    return best


def nonshot_anchors(side_ct, wh_t, t_first, t_last, step=NONSHOT_STEP_S):
    """Deterministic grid of in-play (t, side) anchors where that side
    is genuinely NOT hitting: >=GUARD_S from the side's own contacts,
    >=WHIFF_GUARD_S from any whiff (whiffs are real swings, never
    non-shots) — the exact guard constants rally_instances trains the
    real detector's negatives with. Opponent-contact instants pass on
    purpose; see module docstring."""
    out = []
    n = int((t_last - t_first) / step) + 1 if t_last >= t_first else 0
    for s in (0, 1):
        for k in range(n):
            t = t_first + k * step
            if all(abs(t - c) > GUARD_S for c in side_ct[s]) and \
                    all(abs(t - w) > WHIFF_GUARD_S for w in wh_t):
                out.append((t, s))
    return out


def measure_row(ser, anchor, prev_tc, kind, ty, cum):
    """One measured row (shot or non-shot) with the full stat set, or
    None if the track can't cover the windows. Identical schema for
    both kinds so every downstream consumer (report, sweep, CSV) sees
    one shape."""
    row = {"rally": cum, "kind": kind, "type": ty}
    for ch in CHANNELS:
        early = window_stat(ser, anchor, *EARLY, ch)
        core = window_stat(ser, anchor, *CORE, ch)
        if early is None or core is None:
            return None
        row[f"{ch}_early_max"], row[f"{ch}_early_mean"] = early
        row[f"{ch}_core_max"], row[f"{ch}_core_mean"] = core
    for ch in ("dsho", "dgaze"):
        nc = window_stat(ser, anchor, *CORE, f"{ch}_nocap")
        row[f"{ch}_nocap_core_max"] = nc[0] if nc else 0.0
    for start in SWEEP_STARTS:
        lo, hi = -start, -start + SWEEP_WIDTH
        st = window_stat(ser, anchor, lo, hi, "leg")
        row[f"leg_sweep_{start:.2f}_max"] = st[0] if st else 0.0
        row[f"leg_sweep_{start:.2f}_contam"] = \
            sweep_contaminated(anchor, lo, prev_tc)
    face = face_coverage(ser, anchor)
    if face is None:
        return None
    for k in FACE_KEYS:
        row[f"face_{k}"] = face[k]
    return row


def collect(labels, split, pose_dir):
    rows, n_holdout = [], 0
    for cum, d in sorted(labels.items()):
        if split is not None and split.get(cum, "train") != "train":
            n_holdout += len(d["contacts"])
            continue
        rd = load_rally(pose_dir, cum)
        if rd is None or not d["contacts"]:
            continue
        cands, _b = rally_candidates(rd["z"])
        _fl, m_raw = rally_coverage(d["contacts"], cands, 2, TOL_S)
        m_srv, margin = serve_mapping(rd, d["contacts"])
        m = m_srv if margin >= 1.25 else m_raw
        tracks = list(rd["tracks"].values())
        all_ct = [tc for tc, *_ in d["contacts"]]
        side_ct = {0: [], 1: []}
        for tc, team, *_ in d["contacts"]:
            side_ct[team ^ m].append(tc)
        wh_t = [t for t, *_ in d["whiffs"]]

        for tc, team, ty in d["contacts"]:
            ser = pick_hitter(tracks, team ^ m, tc)
            if ser is None:
                continue
            row = measure_row(ser, tc, prev_contact_before(all_ct, tc),
                              "shot", ty or "other", cum)
            if row is not None:
                rows.append(row)

        for t, s in nonshot_anchors(side_ct, wh_t, min(all_ct), max(all_ct)):
            ser = pick_hitter(tracks, s, t)
            if ser is None:
                continue
            row = measure_row(ser, t, prev_contact_before(all_ct, t),
                              "nonshot", "", cum)
            if row is not None:
                rows.append(row)
    return rows, n_holdout


def report(rows, n_holdout, split_path):
    if not rows:
        raise SystemExit("no scored rows — check --pose-dir/--labels")
    if n_holdout:
        print(f"(skipped {n_holdout} holdout contacts per {split_path} — "
              f"train-only, per labeling_protocol.md)\n")
    shot = [r for r in rows if r["kind"] == "shot"]
    non = [r for r in rows if r["kind"] == "nonshot"]

    print(f"=== face-keypoint EXISTENCE confidence ({len(shot)} shot + "
          f"{len(non)} non-shot windows, >= {FACE_THRESH:.1f}) ===")
    print(f"  this measures 'did the model think a keypoint is probably "
          f"here', not 'is the position right' — pose models are known "
          f"to stay confident on precision for small/ambiguous landmarks "
          f"like eyes at distance (the same gap that let a confidently- "
          f"WRONG shoulder swap pass the dsho confidence gate untouched "
          f"on the first real run). A high number here does not by "
          f"itself vindicate dgaze; the plausibility-capped distribution "
          f"below is the more direct test.")
    means = {}
    for k in FACE_KEYS:
        vals = np.array([r[f"face_{k}"] for r in rows])
        means[k] = vals.mean()
        print(f"  {FACE_LABELS[k]:<6} mean {vals.mean():5.1%}   "
              f"median {np.median(vals):5.1%}   "
              f"zero-coverage windows {int((vals == 0).sum())}/{len(vals)}")
    eye_mean = np.mean([r["face_leye_c"] for r in rows] +
                       [r["face_reye_c"] for r in rows])
    ear_mean = np.mean([r["face_lear_c"] for r in rows] +
                       [r["face_rear_c"] for r in rows])
    print(f"  -> eyes average {eye_mean:.1%}, ears average {ear_mean:.1%}. "
          f"dgaze below rides on ears; treat it as noisy-to-unusable if "
          f"either number is low.")
    if min(means.values()) > 0.97:
        print(f"  NOTE: every keypoint above 97% with essentially no "
              f"variation is itself a flag, not a clean result — real "
              f"tracking on this footage (see ball_visibility.py: the "
              f"ball itself was findable in only 64% of in-play frames) "
              f"essentially never reads this saturated. Existence "
              f"confidence and positional precision are different "
              f"things; this number is the former.")
    print()

    def summarize(vals, name):
        if not vals:
            print(f"{name}: no rows")
            return vals
        v = np.array(vals)
        print(f"{name:<22} n={len(v):<4} median={np.median(v):.3f}  "
              f"p25={np.quantile(v, .25):.3f}  p75={np.quantile(v, .75):.3f}"
              f"  max={v.max():.3f}")
        return vals

    for ch, (label, window) in CHANNELS.items():
        field = f"{ch}_{window}_max"
        print(f"=== {ch} ({label}) — {window}-window peak, "
              f"shot vs non-shot ===")
        shot_vals = summarize(sorted(r[field] for r in shot),
                              "shot (all types)")
        non_vals = summarize(sorted(r[field] for r in non),
                             "non-shot (in play)")

        if shot_vals and non_vals:
            a = auc(shot_vals, non_vals)
            print(f"AUC = {a:.3f}   (P(random shot reading > random "
                  f"non-shot reading); 0.5 = channel can't tell a swing "
                  f"moment from a quiet one at all)")
            allv = np.array(shot_vals + non_vals)
            lo, hi = allv.min(), allv.max()
            if hi > lo:
                edges = np.linspace(lo, hi, 9)
                print(f"histograms ({lo:.2f} to {hi:.2f}, 8 bins, low to "
                      f"high):")
                for name, vals in (("shot", shot_vals),
                                   ("non-shot", non_vals)):
                    counts, _ = np.histogram(vals, bins=edges)
                    bar = "  ".join(f"{c:>2}" for c in counts)
                    print(f"  {name:<9} {bar}")

        if ch in ("dsho", "dgaze") and shot_vals:
            # is MAX_ROT_RAD cutting off real signal or artifact? _nocap
            # keeps the confidence/gap-gated-but-uncapped reading, so
            # this splits what the cap suppressed into "ambiguous" (could
            # be a genuinely fast turn) vs "unambiguous artifact" (only
            # an L/R swap reads this close to pi) without re-running
            # pose extraction at a different threshold.
            nocap = [r[f"{ch}_nocap_core_max"] for r in shot]
            kept = sum(1 for x in nocap if x <= MAX_ROT_RAD)
            ambig = sum(1 for x in nocap if MAX_ROT_RAD < x <= 2.0)
            artifact = sum(1 for x in nocap if x > 2.0)
            near_cap = sum(1 for x in shot_vals if x >= 0.9 * MAX_ROT_RAD)
            print(f"cap sensitivity (shot rows): {kept} kept "
                  f"(<={MAX_ROT_RAD:.1f} rad), {ambig} suppressed in the "
                  f"ambiguous {MAX_ROT_RAD:.1f}-2.0 band, {artifact} "
                  f"suppressed above 2.0 (unambiguous — real pi is 3.14, "
                  f"nothing legitimate reads there); {near_cap} of the "
                  f"KEPT values sit within 10% of the cap itself — worth "
                  f"a second look if that number is more than a couple.")
        print()
    print(f"(rows within a rally share the same few tracks — the n's "
          f"are not independent samples; read AUCs as descriptive, "
          f"not as tested. Anything worth believing graduates to a "
          f"fresh pre-registration on untouched holdout.)")


def report_leg_sweep(rows):
    shot = [r for r in rows if r["kind"] == "shot"]
    non = [r for r in rows if r["kind"] == "nonshot"]
    print(f"\n=== leg (footwork) sweep: sliding {SWEEP_WIDTH:.2f}s window, "
          f"shot vs non-shot, by seconds-before-anchor ===")
    print(f"{'window':<16}{'shot n':>7}{'shot md':>9}{'non n':>7}"
          f"{'non md':>9}{'gap':>8}{'AUC':>7}{'ctm s':>7}{'ctm n':>7}")
    for start in SWEEP_STARTS:
        mf, cf = f"leg_sweep_{start:.2f}_max", f"leg_sweep_{start:.2f}_contam"
        sv = [r[mf] for r in shot]
        nv = [r[mf] for r in non]
        if not sv or not nv:
            continue
        sm, nm = float(np.median(sv)), float(np.median(nv))
        cs = float(np.mean([r[cf] for r in shot]))
        cn = float(np.mean([r[cf] for r in non]))
        lo, hi = -start, -start + SWEEP_WIDTH
        label = f"[{lo:.2f},{hi:.2f}]"
        print(f"{label:<16}{len(sv):>7}{sm:>9.3f}{len(nv):>7}"
              f"{nm:>9.3f}{sm - nm:>+8.3f}{auc(sv, nv):>7.3f}"
              f"{cs:>7.0%}{cn:>7.0%}")
    print(f"(checkpoint: the [{-SWEEP_STARTS[-1]:.2f},-0.35] row should "
          f"match the leg early-window medians above exactly — same "
          f"window, computed two ways)")
    print(f"ctm = share of rows whose window at that offset reaches back "
          f"past the most recent prior contact (any player). For shot "
          f"rows that means recovery from the last shot pollutes the "
          f"prep reading; for non-shot rows it means real shot movement "
          f"leaks into the 'quiet' reading and pushes the AUC toward "
          f"0.5. Either way: discount far-back rows where these run "
          f"high.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--windows", default=WINDOWS_V4)
    ap.add_argument("--pose-dir", default=POSE_DIR)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--sweep-leg", action="store_true",
                    help="sweep the footwork window further back before "
                         "the anchor instead of trusting the single fixed "
                         "EARLY window")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    rosters = load_rosters(Path(a.windows))
    labels = load_labels(Path(a.labels), rosters)
    split = load_split(a.split)
    rows, n_holdout = collect(labels, split, a.pose_dir)
    report(rows, n_holdout, a.split)
    if a.sweep_leg:
        report_leg_sweep(rows)

    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["kind"], r["rally"])))
        print(f"per-row dump -> {a.csv}")


# ------------------------------------------------------------ selftest


def _mk_ser(side, t, **chans):
    ser = {"side": side, "t": np.array(t, float)}
    for k, v in chans.items():
        ser[k] = np.array(v, float)
    return ser


def _full_ser(side, t, **overrides):
    """A ser with EVERY key measure_row touches, zeros unless overridden
    — glue-level tests need the full schema, not just the channel under
    test (the exact class of bug a stale-file KeyError once produced in
    the field)."""
    z = np.zeros_like(np.asarray(t, float))
    chans = {k: z for k in ("arm", "dsho", "leg", "dgaze",
                            "dsho_nocap", "dgaze_nocap")}
    chans.update({k: np.ones_like(z) for k in FACE_KEYS})
    chans.update(overrides)
    return _mk_ser(side, t, **chans)


def selftest():
    # pick_hitter: two same-side tracks, only one has prep-arc arm energy
    t = np.arange(0, 3, 0.05)
    tc = 1.5
    z = np.zeros_like(t)
    quiet = _mk_ser(0, t, arm=np.full_like(t, 0.02), dsho=z, leg=z, dgaze=z)
    prep_mask = (t >= tc - 0.75) & (t < tc - 0.35)
    active_arm = np.where(prep_mask, 0.9, 0.02)
    active = _mk_ser(0, t, arm=active_arm, dsho=z, leg=z, dgaze=z)
    other_side = _mk_ser(1, t, arm=np.full_like(t, 0.9), dsho=z, leg=z,
                         dgaze=z)
    picked = pick_hitter([quiet, active, other_side], 0, tc)
    assert picked is active, "pick_hitter did not select the prep-active track"
    assert pick_hitter([quiet, active, other_side], 1, tc) is other_side
    print("  pick_hitter: selects prep-active same-side track OK")

    # window_stat: a planted spike inside CORE must not leak into EARLY,
    # and must be recovered exactly by the core window (channel-generic
    # — test it via "leg" to confirm the parameter isn't hardcoded)
    leg = np.zeros_like(t)
    core_mask = (t >= tc - 0.35) & (t <= tc + POST_S)
    spike_idx = np.where(core_mask)[0][3]
    leg[spike_idx] = 1.23
    ser = _mk_ser(0, t, arm=z, dsho=z, leg=leg, dgaze=z)
    core = window_stat(ser, tc, *CORE, "leg")
    early = window_stat(ser, tc, *EARLY, "leg")
    assert abs(core[0] - 1.23) < 1e-9, "core window missed the planted spike"
    assert early[0] < 1e-9, "early window leaked the core spike"
    print("  window_stat: core/early boundary is clean OK (channel-generic)")

    # sweep checkpoint: the sweep's last (closest-to-contact) window is
    # DESIGNED to exactly reproduce EARLY -- same bounds, computed two
    # different ways. If this doesn't match, the sweep's window math has
    # a bug (off-by-one, wrong sign) that the rest of the sweep inherits.
    leg2 = np.zeros_like(t)
    early_mask = (t >= tc - PRE_S) & (t < tc - 0.35)
    leg2[np.where(early_mask)[0][2]] = 2.46
    ser2 = _mk_ser(0, t, arm=z, dsho=z, leg=leg2, dgaze=z)
    sweep_lo = -SWEEP_STARTS[-1]
    sweep_hi = -SWEEP_STARTS[-1] + SWEEP_WIDTH
    assert abs(sweep_lo - EARLY[0]) < 1e-9 and abs(sweep_hi - EARLY[1]) < 1e-9, \
        "SWEEP_STARTS[-1]/SWEEP_WIDTH no longer reproduce EARLY's bounds"
    sweep_last = window_stat(ser2, tc, sweep_lo, sweep_hi, "leg")
    early2 = window_stat(ser2, tc, *EARLY, "leg")
    assert sweep_last == early2, \
        f"sweep checkpoint should exactly match EARLY, got " \
        f"{sweep_last} vs {early2}"
    print(f"  sweep checkpoint: [{sweep_lo:.2f},{sweep_hi:.2f}] "
          f"matches EARLY exactly OK")

    # contamination: a window that would reach past the previous
    # contact must be flagged; one that stays clear must not
    assert sweep_contaminated(tc=10.0, lo=-1.5, prev_tc=9.0), \
        "window starting at t=8.5 reaches past a prev contact at t=9.0"
    assert not sweep_contaminated(tc=10.0, lo=-0.75, prev_tc=9.0), \
        "window starting at t=9.25 does NOT reach past a prev contact at t=9.0"
    assert not sweep_contaminated(tc=10.0, lo=-1.5, prev_tc=None), \
        "no previous contact (first in rally) should never flag contaminated"
    assert prev_contact_before([9.0, 10.0, 11.0], 10.0) == 9.0, \
        "prev_contact_before must skip the anchor itself"
    assert prev_contact_before([9.0, 10.0], 8.0) is None
    assert prev_contact_before([9.0, 10.0], 10.3) == 10.0
    print("  sweep_contaminated + prev_contact_before: flags reaching "
          "past the previous contact, not windows that stay clear OK")

    # nonshot_anchors: own-side contacts and whiffs must repel anchors
    # (GUARD_S / WHIFF_GUARD_S, strict), opponent contacts must NOT —
    # that instant is a legitimate non-shot moment for this side and the
    # detector's own matched negatives use it
    side_ct = {0: [10.0], 1: [11.0]}
    anchors = nonshot_anchors(side_ct, [13.0], 10.0, 14.0, step=0.5)
    assert (10.0, 0) not in anchors, "own contact instant must be excluded"
    assert (10.5, 0) not in anchors, \
        f"0.5s from an own contact is not > GUARD_S={GUARD_S}"
    assert (11.0, 0) in anchors, \
        "OPPONENT's contact instant is a valid non-shot moment for side 0"
    assert (13.0, 0) not in anchors and (12.5, 0) not in anchors, \
        f"whiff at 13.0 must repel by WHIFF_GUARD_S={WHIFF_GUARD_S}"
    assert (10.0, 1) in anchors, "side 1 is clear at side 0's contact"
    assert (11.0, 1) not in anchors, "own contact instant (side 1)"
    assert all(t <= 14.0 for t, _ in anchors), "grid must stay in play"
    print("  nonshot_anchors: own-contact + whiff guards repel, opponent "
          "instants pass OK")

    # auc: exact values on hand-built cases, including the zero-tie case
    # the midrank handling exists for
    assert auc([3, 4, 5], [1, 2]) == 1.0
    assert auc([1, 2], [3, 4, 5]) == 0.0
    assert auc([1, 2], [1, 2]) == 0.5, "pure ties must read exactly 0.5"
    assert abs(auc([0, 0, 1], [0, 0, 0]) - 2 / 3) < 1e-12, \
        "3 wins + 6 half-ties out of 9 pairs = 2/3 exactly"
    print("  auc: perfect separation 1.0/0.0, ties midranked OK")

    # measure_row glue: identical full schema for both kinds (the CSV
    # writer and every report reader depend on one shape)
    fs = _full_ser(0, t)
    r_shot = measure_row(fs, tc, None, "shot", "dink", 1)
    r_non = measure_row(fs, tc, 0.5, "nonshot", "", 1)
    assert r_shot is not None and r_non is not None
    assert set(r_shot) == set(r_non), "shot/non-shot rows must share schema"
    n_expect = 3 + 4 * len(CHANNELS) + 2 + 2 * len(SWEEP_STARTS) + len(FACE_KEYS)
    assert len(r_shot) == n_expect, \
        f"row schema drifted: {len(r_shot)} keys vs expected {n_expect}"
    assert r_shot["kind"] == "shot" and r_non["kind"] == "nonshot"
    assert not any(r_shot[f"leg_sweep_{s:.2f}_contam"] for s in SWEEP_STARTS), \
        "prev_tc=None must never flag contamination"
    assert r_non["leg_sweep_1.55_contam"], \
        "prev contact at 0.5 with anchor 1.5: the -1.55 window reaches past it"
    print(f"  measure_row: uniform {n_expect}-key schema for both kinds, "
          f"contam flags wired OK")
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
