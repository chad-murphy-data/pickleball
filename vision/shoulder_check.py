"""Shoulder-rotation bimodality check (2026-08-18) — EXPLORATION, not a
gate. Prompted by a direct pushback on an earlier claim: shoulder
rotation was ranked as the weakest of three candidate swing-detection
features, reasoning "dinks are compact, deliberately minimal-rotation
shots." The counter-argument: "dink" isn't one mechanic. A down-the-line
dink can stay square, but a cross-court backhand roll dink (the shot the
user watched a lot of this past weekend) should show real rotation —
from body mechanics alone (a backhand contact point sits across the
body's midline) plus the active pronation a roll needs to generate
topspin. And committed shots (drives/smashes) may be MORE square than
assumed: pickleball's short paddle lever doesn't need tennis-style trunk
rotation to generate pace.

Nothing new is being built here. `dsho` already exists in
swing_explore.track_series — the frame-to-frame angular velocity of the
2D shoulder-line orientation (image-space; foreshortening/rotation shows
up as the line's apparent angle changing, no camera calibration needed)
— and already feeds the learned scorer as one of six channels. This
script just looks at what that channel already shows on the 10 labeled
rallies, split by shot type, to see which mental model the data fits:
bimodal (some dinks/counters rotate a lot, some don't — consistent with
a mixed population of shot mechanics) or uniformly low (the original,
now-doubted story).

No direction/side label exists yet — checked make_shot_audit.py's
SHOT_TYPES, there is no forehand/backhand or cross-court/down-the-line
field — so this cannot attribute WHICH dinks rotate, only whether the
distribution's SHAPE is consistent with a mixed population at all.
Wide/bimodal with a real high tail = worth a schema addition to test
directly. Tight and low, or statistically the same as committed shots =
the original flat story holds.

RUN (same flat folder as swing_explore.py; numpy only, seconds):
    python3 shoulder_check.py
    python3 shoulder_check.py --csv shoulder_check.csv   # per-contact dump

SELF-TEST (no files):  python3 shoulder_check.py --selftest
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from contact_ceiling import load_rosters, load_labels, rally_candidates, rally_coverage
from swing_explore import load_rally, serve_mapping, TOL_S, PRE_S, POST_S

LABELS = "contact_labels_chicago0725.csv"
WINDOWS_V4 = "rally_windows_chicago0725_v4.csv"
POSE_DIR = "pose_rtm"
SPLIT = "label_split.csv"

SOFT = {"dink", "counter"}                    # close-to-net, compact
COMMITTED = {"drive", "speed-up", "smash"}     # power shots — the
# comparison group for "maybe these are square too, not rotated"

EARLY = (-PRE_S, -0.35)   # prep arc — matches window_feats' m_early
CORE = (-0.35, POST_S)    # strike — matches window_feats' m_core


def load_split(path):
    if not Path(path).exists():
        return None
    out = {}
    for r in csv.DictReader(open(path)):
        out[int(r["rally_cum"])] = r["split"]
    return out


def window_stat(ser, tc, lo, hi, ch="dsho"):
    """(max, mean) of channel `ch` inside [tc+lo, tc+hi], or None."""
    t, v = ser["t"], ser[ch]
    m = (t >= tc + lo) & (t <= tc + hi)
    if not m.any():
        return None
    return float(v[m].max()), float(v[m].mean())


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


def collect(labels, split, pose_dir, windows_path):
    rosters = load_rosters(Path(windows_path))
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
            early = window_stat(ser, tc, *EARLY)
            core = window_stat(ser, tc, *CORE)
            if early is None or core is None:
                continue
            rows.append({"rally": cum, "type": ty,
                         "early_max": early[0], "early_mean": early[1],
                         "core_max": core[0], "core_mean": core[1]})
    _ = rosters  # loaded for load_labels' caller upstream; kept for parity
    return rows, n_holdout


def report(rows, n_holdout, split_path):
    if not rows:
        raise SystemExit("no scored contacts — check --pose-dir/--labels")
    if n_holdout:
        print(f"(skipped {n_holdout} holdout contacts per {split_path} — "
              f"train-only, per labeling_protocol.md)\n")

    def summarize(ty_set, name):
        vals = sorted(r["core_max"] for r in rows if r["type"] in ty_set)
        if not vals:
            print(f"{name}: no contacts")
            return vals
        v = np.array(vals)
        print(f"{name:<28} n={len(v):<3} median={np.median(v):.3f}  "
              f"p25={np.quantile(v, .25):.3f}  p75={np.quantile(v, .75):.3f}"
              f"  max={v.max():.3f}")
        return vals

    print("=== dsho (shoulder angular-velocity) core-window peak, "
          "by shot type ===")
    soft_vals = summarize(SOFT, "soft (dink+counter)")
    committed_vals = summarize(COMMITTED, "committed (drive/speedup/smash)")

    if soft_vals and committed_vals:
        allv = np.array(soft_vals + committed_vals)
        lo, hi = allv.min(), allv.max()
        edges = np.linspace(lo, hi, 9)
        print(f"\nhistograms ({lo:.2f} to {hi:.2f}, 8 bins, low to high):")
        for name, vals in (("soft", soft_vals), ("committed", committed_vals)):
            counts, _ = np.histogram(vals, bins=edges)
            bar = "  ".join(f"{c:>2}" for c in counts)
            print(f"  {name:<9} {bar}")

        v = np.array(soft_vals)
        if len(v) >= 6:
            bc = bimodality_coefficient(v)
            print(f"\nsoft-group bimodality coefficient: {bc:.3f}  "
                  f"(rule-of-thumb bimodal if > 0.555; normal-shaped "
                  f"scores ~0.33). Rough heuristic, not a calibrated "
                  f"test — read it next to the histogram above, not "
                  f"instead of it.")


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
    rows, n_holdout = collect(labels, split, a.pose_dir, a.windows)
    report(rows, n_holdout, a.split)

    if a.csv and rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["type"], -r["core_max"])))
        print(f"\nper-contact rows -> {a.csv}")


# ------------------------------------------------------------ selftest


def _mk_ser(side, t, arm, dsho):
    return {"side": side, "t": np.array(t, float),
            "arm": np.array(arm, float), "dsho": np.array(dsho, float)}


def selftest():
    # pick_hitter: two same-side tracks, only one has prep-arc arm energy
    t = np.arange(0, 3, 0.05)
    tc = 1.5
    quiet = _mk_ser(0, t, np.full_like(t, 0.02), np.zeros_like(t))
    prep_mask = (t >= tc - 0.75) & (t < tc - 0.35)
    active_arm = np.where(prep_mask, 0.9, 0.02)
    active = _mk_ser(0, t, active_arm, np.zeros_like(t))
    other_side = _mk_ser(1, t, np.full_like(t, 0.9), np.zeros_like(t))
    picked = pick_hitter([quiet, active, other_side], 0, tc)
    assert picked is active, "pick_hitter did not select the prep-active track"
    assert pick_hitter([quiet, active, other_side], 1, tc) is other_side
    print("  pick_hitter: selects prep-active same-side track OK")

    # window_stat: a planted dsho spike inside CORE must not leak into
    # EARLY, and must be recovered exactly by the core window
    dsho = np.zeros_like(t)
    core_mask = (t >= tc - 0.35) & (t <= tc + POST_S)
    spike_idx = np.where(core_mask)[0][3]
    dsho[spike_idx] = 1.23
    ser = _mk_ser(0, t, np.zeros_like(t), dsho)
    core = window_stat(ser, tc, *CORE)
    early = window_stat(ser, tc, *EARLY)
    assert abs(core[0] - 1.23) < 1e-9, "core window missed the planted spike"
    assert early[0] < 1e-9, "early window leaked the core spike"
    print("  window_stat: core/early boundary is clean OK")

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
