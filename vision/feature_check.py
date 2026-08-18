"""Multi-channel pose-feature check (2026-08-18) — EXPLORATION, not a
gate. Successor to the first version of this file, which only looked at
shoulder rotation; extended to the three candidate signals from the
"what else could a swing-detector look at" discussion: shoulder
rotation, footwork, and gaze/head orientation ("a player looking at
their partner wouldn't be hitting").

None of the three channels are new extraction — all three ride on
keypoints `pose_extract.py` already saves (COCO-17: everything below the
shoulders and everything on the face was extracted from day one and sat
unused). `dsho` (shoulder-line angular velocity) already feeds the
learned scorer in swing_explore.py as one of six channels; `leg`
(hip-relative ankle/knee speed, same math as the existing arm channel)
and `dgaze` (ear-line angular velocity, same math as dsho) are new
additions to swing_explore.track_series, but not wired into the trained
scorer — this script reads them directly. Nothing here retrains or
re-tunes that model; the frozen v1-v5 numbers in swing_explore_notes.md
stand as they are.

No forehand/backhand or cross-court/down-the-line label exists (checked
make_shot_audit.py's SHOT_TYPES — not there), so none of this can
attribute WHICH dinks/counters show a signal, only whether the
distribution's SHAPE is consistent with a mixed population at all —
same caveat as the original shoulder-only version, now applied to all
three.

Eyes specifically: the user flagged skepticism up front given the
video (a 720p condensed broadcast VOD where `ball_visibility.py`
already found the ball findable in only 64% of in-play frames — faces
are smaller and blurrier than the ball for most of the court). Rather
than assume the answer, this script reports raw eye/ear/nose keypoint
CONFIDENCE/coverage directly, before ever looking at whether gaze
predicts anything — a cheap, decisive first check. The gaze channel
itself uses EARS, not eyes (same head-yaw geometry, but a pose model
localizes an ear from head shape/context and doesn't need to resolve
anything as fine as an eye at distance) — the eye-specific numbers are
reported so that substitution is visible and checkable, not assumed.

RUN (same flat folder as swing_explore.py; numpy only, seconds):
    python3 feature_check.py
    python3 feature_check.py --csv feature_check.csv   # per-contact dump

SELF-TEST (no files):  python3 feature_check.py --selftest
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from contact_ceiling import load_rosters, load_labels, rally_candidates, rally_coverage
from swing_explore import (load_rally, serve_mapping, TOL_S, PRE_S, POST_S,
                          MAX_ROT_RAD)

LABELS = "contact_labels_chicago0725.csv"
WINDOWS_V4 = "rally_windows_chicago0725_v4.csv"
POSE_DIR = "pose_rtm"
SPLIT = "label_split.csv"

SOFT = {"dink", "counter"}                    # close-to-net, compact
COMMITTED = {"drive", "speed-up", "smash"}     # power shots — the
# comparison group for "maybe these are square/still too, not X"

EARLY = (-PRE_S, -0.35)   # prep arc — matches window_feats' m_early
CORE = (-0.35, POST_S)    # strike — matches window_feats' m_core

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
    the hitter's side with the most PREP-ARC arm energy (window_feats'
    f[0] = arm channel's early_max) — the actor within a side isn't
    separately labeled, so this is the model's own hitter proxy,
    reused here rather than re-invented."""
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


def bimodality_coefficient(v):
    """Sarle's bimodality coefficient: (skew^2 + 1) / kurtosis (kurtosis
    on the raw, not excess, scale — normal = 3). 0.555 (a uniform
    distribution's value) is the standard rule-of-thumb cutoff; higher
    reads more bimodal. Crude but real dynamic range: normal -> 0.33,
    a well-separated symmetric 2-point mixture -> 1.0 (kurtosis bottoms
    out at 1 for any two-point distribution, which is what a clean
    bimodal split degenerates toward).

    NOT what this first tried: a 1D 2-means gap normalized by the
    POOLED std. That statistic turned out to be broken by construction
    — total variance already contains the between-cluster gap, so it
    caps near 2.0 for even infinitely-separated clusters, and a plain
    unimodal Gaussian split down the middle already scores ~1.6. The
    selftest below caught it (a "tight" input failed the discrimination
    assertion) before this ever ran on real data."""
    v = np.asarray(v, float)
    mu, sd = v.mean(), v.std() + 1e-12
    skew = np.mean(((v - mu) / sd) ** 3)
    kurt = np.mean(((v - mu) / sd) ** 4)
    return (skew ** 2 + 1) / max(kurt, 1e-9)


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
        for tc, team, ty in d["contacts"]:
            ty = ty or "other"
            if ty not in SOFT and ty not in COMMITTED:
                continue
            ser = pick_hitter(tracks, team ^ m, tc)
            if ser is None:
                continue
            row = {"rally": cum, "type": ty}
            ok = True
            for ch in CHANNELS:
                early = window_stat(ser, tc, *EARLY, ch)
                core = window_stat(ser, tc, *CORE, ch)
                if early is None or core is None:
                    ok = False
                    break
                row[f"{ch}_early_max"], row[f"{ch}_early_mean"] = early
                row[f"{ch}_core_max"], row[f"{ch}_core_mean"] = core
            if not ok:
                continue
            for ch in ("dsho", "dgaze"):
                nc = window_stat(ser, tc, *CORE, f"{ch}_nocap")
                row[f"{ch}_nocap_core_max"] = nc[0] if nc else 0.0
            face = face_coverage(ser, tc)
            if face is None:
                continue
            for k in FACE_KEYS:
                row[f"face_{k}"] = face[k]
            rows.append(row)
    return rows, n_holdout


def report(rows, n_holdout, split_path):
    if not rows:
        raise SystemExit("no scored contacts — check --pose-dir/--labels")
    if n_holdout:
        print(f"(skipped {n_holdout} holdout contacts per {split_path} — "
              f"train-only, per labeling_protocol.md)\n")

    print(f"=== face-keypoint EXISTENCE confidence ({len(rows)} contacts, "
          f"full prep+strike window, >= {FACE_THRESH:.1f}) ===")
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
              f"zero-coverage contacts {int((vals == 0).sum())}/{len(vals)}")
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

    def summarize(ty_set, name, field):
        vals = sorted(r[field] for r in rows if r["type"] in ty_set)
        if not vals:
            print(f"{name}: no contacts")
            return vals
        v = np.array(vals)
        print(f"{name:<28} n={len(v):<3} median={np.median(v):.3f}  "
              f"p25={np.quantile(v, .25):.3f}  p75={np.quantile(v, .75):.3f}"
              f"  max={v.max():.3f}")
        return vals

    for ch, (label, window) in CHANNELS.items():
        field = f"{ch}_{window}_max"
        print(f"=== {ch} ({label}) — {window}-window peak, "
              f"by shot type ===")
        soft_vals = summarize(SOFT, "soft (dink+counter)", field)
        committed_vals = summarize(COMMITTED,
                                   "committed (drive/speedup/smash)", field)

        if soft_vals and committed_vals:
            allv = np.array(soft_vals + committed_vals)
            lo, hi = allv.min(), allv.max()
            if hi > lo:
                edges = np.linspace(lo, hi, 9)
                print(f"histograms ({lo:.2f} to {hi:.2f}, 8 bins, low to "
                      f"high):")
                for name, vals in (("soft", soft_vals),
                                   ("committed", committed_vals)):
                    counts, _ = np.histogram(vals, bins=edges)
                    bar = "  ".join(f"{c:>2}" for c in counts)
                    print(f"  {name:<9} {bar}")

            v = np.array(soft_vals)
            if len(v) >= 6:
                bc = bimodality_coefficient(v)
                print(f"soft-group bimodality coefficient: {bc:.3f}  "
                      f"(rule-of-thumb bimodal if > 0.555; normal-shaped "
                      f"scores ~0.33). Rough heuristic, not a calibrated "
                      f"test — read it next to the histogram, not "
                      f"instead of it.")

        if ch in ("dsho", "dgaze") and soft_vals:
            # is MAX_ROT_RAD cutting off real signal or artifact? _nocap
            # keeps the confidence/gap-gated-but-uncapped reading, so
            # this splits what the cap suppressed into "ambiguous" (could
            # be a genuinely fast turn) vs "unambiguous artifact" (only
            # an L/R swap reads this close to pi) without re-running
            # pose extraction at a different threshold.
            soft_nocap = [r[f"{ch}_nocap_core_max"] for r in rows
                         if r["type"] in SOFT]
            kept = sum(1 for x in soft_nocap if x <= MAX_ROT_RAD)
            ambig = sum(1 for x in soft_nocap if MAX_ROT_RAD < x <= 2.0)
            artifact = sum(1 for x in soft_nocap if x > 2.0)
            near_cap = sum(1 for x in soft_vals if x >= 0.9 * MAX_ROT_RAD)
            print(f"cap sensitivity (soft group): {kept} kept "
                  f"(<={MAX_ROT_RAD:.1f} rad), {ambig} suppressed in the "
                  f"ambiguous {MAX_ROT_RAD:.1f}-2.0 band, {artifact} "
                  f"suppressed above 2.0 (unambiguous — real pi is 3.14, "
                  f"nothing legitimate reads there); {near_cap} of the "
                  f"KEPT values sit within 10% of the cap itself — worth "
                  f"a second look if that number is more than a couple.")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--windows", default=WINDOWS_V4)
    ap.add_argument("--pose-dir", default=POSE_DIR)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--csv", default=None)
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

    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["type"], -r["dsho_core_max"])))
        print(f"per-contact rows -> {a.csv}")


# ------------------------------------------------------------ selftest


def _mk_ser(side, t, **chans):
    ser = {"side": side, "t": np.array(t, float)}
    for k, v in chans.items():
        ser[k] = np.array(v, float)
    return ser


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
    # now — test it via "leg" to confirm the parameter isn't hardcoded)
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

    # face_coverage: frames below FACE_THRESH must not count, frames at
    # or above it must
    fser = {"t": t,
           "nose_c": np.where(t < 1.0, 0.9, 0.05),   # 1/3 of frames ok
           "leye_c": np.zeros_like(t),                # never ok
           "reye_c": np.ones_like(t),                 # always ok
           "lear_c": np.ones_like(t), "rear_c": np.ones_like(t)}
    fc = face_coverage(fser, tc)
    assert fc["leye_c"] == 0.0, "all-low confidence should read 0 coverage"
    assert fc["reye_c"] == 1.0, "all-high confidence should read full coverage"
    assert 0.0 < fc["nose_c"] < 1.0, "partial coverage should read partial"
    print("  face_coverage: thresholding and partial coverage OK")

    # bimodality_coefficient must clear the 0.555 rule-of-thumb on a
    # real separated mixture, and sit comfortably under it on plain
    # unimodal data — including data an over-eager clustering statistic
    # would mistake for structure (this is the check that would have
    # caught the broken gap/spread version before it shipped).
    rng = np.random.default_rng(0)
    bimodal = np.concatenate([rng.normal(0.1, 0.02, 15),
                              rng.normal(0.9, 0.02, 15)])
    bc_bi = bimodality_coefficient(bimodal)
    tight = rng.normal(0.3, 0.05, 30)
    bc_t = bimodality_coefficient(tight)
    normal_ish = rng.normal(0.5, 0.1, 200)
    bc_n = bimodality_coefficient(normal_ish)
    assert bc_bi > 0.555, f"separated mixture should clear 0.555 (got {bc_bi:.3f})"
    assert bc_t < 0.555, f"plain unimodal should sit under 0.555 (got {bc_t:.3f})"
    assert bc_n < 0.4, f"large-n normal should read near 1/3 (got {bc_n:.3f})"
    print(f"  bimodality_coefficient: separated mixture {bc_bi:.3f} (>0.555), "
          f"unimodal {bc_t:.3f} and {bc_n:.3f} (<0.555) OK")
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
