"""Does adding leg/dgaze to the trained scorer beat what's already
there? (2026-08-18) — EXPLORATION, not a verdict.

Different question from feature_check.py. That script asked "among
KNOWN shots, does movement look different by shot type" — useful for
deciding whether a feature is worth trying, but it says nothing about
whether the feature actually helps the thing swing_explore.py is
FOR: telling a real swing moment apart from a non-swing moment. This
script answers that directly, same methodology already validated
there — leave-one-rally-out, coverage@2x, TOL_S=0.30s — just run twice
under two channel sets and compared side by side:

    BASE = arm/lw/rw/le/re/dsho   (the v1-v5 pipeline's channels)
    EXT  = BASE + leg + dgaze

IMPORTANT: BASE here is NOT the historical "56.2%" from
swing_explore_notes.md. That number was measured before dsho's
confidence-vs-precision bug was found and fixed (2026-08-18, see
MAX_ROT_RAD in swing_explore.py) — the old dsho values it was trained
on don't exist anymore. This script re-measures BASE fresh, on the
SAME rallies, so the BASE-vs-EXT comparison is apples to apples
regardless of what the old number was.

Default output is pure swing/no-swing (user call, 2026-08-18: the
pipeline is a two-step process — swing detection first, shot type
later — and the per-type slices at n=3-10 per type were pure noise:
on the first real run every large-looking per-type delta was exactly
one contact flipping). --per-type restores the sliced tables for
debugging.

RUN (same flat folder as swing_explore.py; numpy only, minutes):
    python3 channel_ablation.py

SELF-TEST (no files): python3 channel_ablation.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from contact_ceiling import load_rosters, load_labels, rally_candidates, rally_coverage
from swing_explore import (load_rally, serve_mapping, rally_instances,
                           fit_logreg, score_rally, coverage_at_budget,
                           TOL_S, CHANNELS_BASE, CHANNELS_EXT)

LABELS = "contact_labels_chicago0725.csv"
WINDOWS_V4 = "rally_windows_chicago0725_v4.csv"
POSE_DIR = "pose_rtm"


def loro_eval(rallies, channels):
    """Leave-one-rally-out coverage@2x for one channel set. Returns
    (overall, per_type {ty: (hit, n)}, n_contacts)."""
    all_flags = []
    for held in sorted(rallies):
        Xtr, ytr = [], []
        for cum, r in rallies.items():
            if cum == held:
                continue
            X, y = rally_instances(r["rd"], r["contacts"], r["whiffs"],
                                   r["m"], channels=channels)
            Xtr += X
            ytr += y
        Xtr = np.stack(Xtr)
        ytr = np.array(ytr, float)
        model = fit_logreg(Xtr, ytr)
        r = rallies[held]
        dets = score_rally(model, r["rd"], channels=channels)
        all_flags += coverage_at_budget(dets, r["contacts"], r["m"])
    n = len(all_flags)
    overall = sum(h for _, h in all_flags) / n if n else float("nan")
    per_type = {}
    for ty, h in all_flags:
        a, b = per_type.get(ty, (0, 0))
        per_type[ty] = (a + h, b + 1)
    return overall, per_type, n


def assemble_rallies(labels, pose_dir):
    rallies = {}
    for cum, d in labels.items():
        rd = load_rally(pose_dir, cum)
        if rd is None or not d["contacts"]:
            continue
        cands, _b = rally_candidates(rd["z"])
        _fl, m_raw = rally_coverage(d["contacts"], cands, 2, TOL_S)
        m_srv, margin = serve_mapping(rd, d["contacts"])
        m = m_srv if margin >= 1.25 else m_raw
        rallies[cum] = {"rd": rd, "contacts": d["contacts"],
                        "whiffs": d["whiffs"], "m": m}
    return rallies


def labels_fingerprint(rallies):
    """Deterministic content hash of the label data actually evaluated
    (contacts + whiffs of the pose-covered rallies). Printed in the run
    header so cross-run comparability is VISIBLE: the 2026-08-18 real
    runs printed BASE 57.4% and then 54.3% on 'the same' 162 contacts —
    the code was cleared (diff + order/hash-seed probes all clean), so
    the label rows themselves had changed between runs (ongoing labeling
    edits/re-exports). A different fingerprint means any delta vs an
    older run is void; only same-fingerprint runs compare."""
    parts = []
    for cum in sorted(rallies):
        r = rallies[cum]
        parts.append((cum,
                      tuple((round(tc, 3), team, ty)
                            for tc, team, ty in r["contacts"]),
                      tuple(round(t, 3) for t, *_ in r["whiffs"])))
    return hashlib.md5(repr(parts).encode()).hexdigest()[:10]


def parse_drop(spec):
    """--drop 'dsho' (or comma list) -> (dropped, reduced-BASE). Answers
    the incremental question feature_check.py's per-channel AUCs cannot:
    an AUC measures a channel ALONE vs nothing, not what it adds on top
    of the channels already in the model — and dsho's detector slot has
    never been re-measured since its confidence-vs-precision bug fix
    (the v3-era lift was earned partly on artifact values)."""
    dropped = tuple(s.strip() for s in spec.split(",") if s.strip())
    if not dropped:
        raise SystemExit("--drop given but empty")
    bad = [c for c in dropped if c not in CHANNELS_BASE]
    if bad:
        raise SystemExit(f"--drop {','.join(bad)}: not in BASE "
                         f"{'/'.join(CHANNELS_BASE)}")
    reduced = tuple(c for c in CHANNELS_BASE if c not in dropped)
    if not reduced:
        raise SystemExit("--drop removed every channel; nothing to fit")
    return dropped, reduced


def print_result(name, overall, per_type, n, show_types=False):
    print(f"=== {name} ===")
    print(f"  coverage@2x overall: {overall:.1%}  (n={n})")
    if show_types:
        for ty, (h, b) in sorted(per_type.items(), key=lambda kv: -kv[1][1]):
            print(f"    {ty:<10} {h:>3}/{b:<3} {h / b:6.1%}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--windows", default=WINDOWS_V4)
    ap.add_argument("--pose-dir", default=POSE_DIR)
    ap.add_argument("--per-type", action="store_true",
                    help="also print the per-shot-type slices (noise at "
                         "this label count — debugging only)")
    ap.add_argument("--drop", default=None, metavar="CH[,CH]",
                    help="compare BASE-minus-these vs full BASE instead "
                         "of BASE vs EXT (e.g. --drop dsho: does the "
                         "channel still earn its detector slot?)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    rosters = load_rosters(Path(a.windows))
    labels = load_labels(Path(a.labels), rosters)
    rallies = assemble_rallies(labels, a.pose_dir)
    if len(rallies) < 3:
        raise SystemExit(f"need >=3 rallies with labels + pose "
                         f"(found {len(rallies)}) — check --pose-dir")
    n_contacts = sum(len(r["contacts"]) for r in rallies.values())
    print(f"channel_ablation: {len(rallies)} rallies, {n_contacts} "
          f"contacts (leave-one-rally-out; EXPLORATION, not a gate)")
    print(f"labels fingerprint: {labels_fingerprint(rallies)} — numbers "
          f"only compare across runs that print the SAME fingerprint; a "
          f"changed value means the label rows changed (relabeling, "
          f"re-export) and any delta vs an older run is void\n")

    if a.drop:
        dropped, reduced = parse_drop(a.drop)
        set_a, name_a = reduced, (f"BASE minus {'+'.join(dropped)}  "
                                  f"({'/'.join(reduced)})")
        set_b, name_b = CHANNELS_BASE, "BASE  (full)"
        delta_note = f"what {'+'.join(dropped)} currently adds"
    else:
        set_a, name_a = CHANNELS_BASE, ("BASE  (arm/lw/rw/le/re/dsho — "
                                        "re-measured post-fix, NOT the "
                                        "historical 56.2%)")
        set_b, name_b = CHANNELS_EXT, "EXT   (BASE + leg + dgaze)"
        delta_note = "what leg+dgaze add"

    ov_a, pt_a, n_a = loro_eval(rallies, set_a)
    print_result(name_a, ov_a, pt_a, n_a, a.per_type)

    ov_b, pt_b, n_b = loro_eval(rallies, set_b)
    print_result(name_b, ov_b, pt_b, n_b, a.per_type)

    d = ov_b - ov_a
    print(f"delta: {d:+.1%} overall = {delta_note} "
          f"(positive = the extra channel(s) help)")
    if a.per_type:
        for ty in sorted(set(pt_a) | set(pt_b)):
            ha, ba = pt_a.get(ty, (0, 0))
            hb, bb = pt_b.get(ty, (0, 0))
            if ba and bb:
                print(f"  {ty:<10} A {ha / ba:6.1%}  B {hb / bb:6.1%}  "
                      f"({hb / bb - ha / ba:+.1%})")
    print(f"\nn={n_contacts} contacts, 10 rallies — read any single-digit "
          f"delta as noise, not a verdict. A result worth believing "
          f"graduates to a fresh pre-registration on untouched holdout "
          f"(data/vision/label_split.csv), same rule as everything else "
          f"in this file.")


# ------------------------------------------------------------ selftest


def selftest():
    from contact_ceiling import synth_rally
    from swing_explore import track_series
    rng = np.random.default_rng(7)
    types = ["serve", "return", "dink", "dink", "speed-up", "counter",
             "smash", "dink", "counter", "dink", "drive", "dink"]
    rallies = {}
    for cum in (1, 2, 3, 4):
        contacts = [(103.0 + k * 1.1 + rng.normal(0, 0.05), k % 2, types[k])
                    for k in range(12)]
        z = synth_rally(rng, [(t, team) for t, team, _ in contacts],
                        planted=True, t1=117.5)
        rd = {"tracks": {}, "z": z, "bounds": (100.0, 117.5)}
        t, trk = np.asarray(z["t"]), np.asarray(z["track"])
        for tid in np.unique(trk):
            m = trk == tid
            o = np.argsort(t[m])
            ser = track_series(t[m][o], np.asarray(z["box"])[m][o],
                               np.asarray(z["kpt"])[m][o],
                               np.asarray(z["kpc"])[m][o], 30.0)
            ser["side"] = int(np.asarray(z["side"])[m][0])
            ser["H"] = 720
            # plant a genuinely informative dgaze burst at THIS side's
            # own contacts, mirroring how synth_rally plants arm bursts
            # -- proves the LORO pipeline can exploit a real extra
            # channel end to end, not just that the vector is longer
            tser = ser["t"]
            planted = np.zeros_like(tser)
            for tc, team, *_ in contacts:
                if team != ser["side"]:
                    continue
                planted[np.abs(tser - tc) < 0.1] = 0.9
            ser["dgaze"] = planted
            if cum in (3, 4):
                # BASE's real signal, wiped -- a "does EXT ever ACTUALLY
                # help" test needs BASE to genuinely struggle somewhere,
                # not just tie at the ceiling everywhere (which is all
                # the first version of this test proved). Zeroing "arm"
                # alone wasn't enough (BASE only dropped to 93.8%) --
                # lw/rw/le/re are the SEPARATE per-joint channels arm is
                # max-reduced FROM, each independently fed to
                # window_feats, and synth_rally's burst still shows up
                # in them untouched. dsho is left alone: synth_rally
                # never perturbs shoulder position, so it's already
                # near-uninformative here regardless. dgaze (EXT-only)
                # stays informative throughout.
                for ch in ("arm", "lw", "rw", "le", "re"):
                    ser[ch] = np.zeros_like(ser[ch])
            rd["tracks"][int(tid)] = ser
        rallies[cum] = {"rd": rd, "contacts": sorted(contacts),
                        "whiffs": [], "m": 0}

    ov_base, _, _ = loro_eval(rallies, CHANNELS_BASE)
    ov_ext, _, _ = loro_eval(rallies, CHANNELS_EXT)
    print(f"  informative-dgaze synth (rallies 3-4 have arm zeroed, "
          f"dgaze intact): BASE {ov_base:.1%}, EXT {ov_ext:.1%}")
    assert ov_base <= 0.85, \
        "BASE should genuinely struggle once its only real signal is " \
        "gone on half the rallies -- if it doesn't, this test proves " \
        "nothing about whether EXT's dgaze channel is doing any work"
    assert ov_ext >= ov_base + 0.10, \
        "EXT should meaningfully recover coverage on the arm-blind " \
        "rallies via dgaze, not just tie BASE"

    # plumbing: EXT vectors must be exactly 14 columns longer (2 extra
    # channels x 7 sub-features) and those columns must actually vary
    r1 = rallies[1]
    Xb, _ = rally_instances(r1["rd"], r1["contacts"], r1["whiffs"], 0,
                            channels=CHANNELS_BASE)
    Xe, _ = rally_instances(r1["rd"], r1["contacts"], r1["whiffs"], 0,
                            channels=CHANNELS_EXT)
    Xb, Xe = np.stack(Xb), np.stack(Xe)
    assert Xe.shape[1] == Xb.shape[1] + 14, \
        f"EXT should add exactly 14 columns, got {Xe.shape[1] - Xb.shape[1]}"
    assert Xe[:, -14:].std() > 0, \
        "the new leg/dgaze columns should not be constant/degenerate"
    print(f"  plumbing: EXT adds exactly 14 columns "
          f"({Xb.shape[1]}->{Xe.shape[1]}), non-degenerate OK")

    # --drop: parsing must validate against BASE, and a reduced set must
    # genuinely fit a reduced feature space — dropping the entire arm
    # family (all the synth's real signal) has to hurt vs full BASE, or
    # loro_eval is silently ignoring the channel set it was handed
    dropped, reduced = parse_drop("dsho")
    assert dropped == ("dsho",) and "dsho" not in reduced
    assert len(reduced) == len(CHANNELS_BASE) - 1
    for bad in ("nope", "dsho,nope", ",".join(CHANNELS_BASE)):
        try:
            parse_drop(bad)
            raise AssertionError(f"parse_drop({bad!r}) should have exited")
        except SystemExit:
            pass
    Xd, _ = rally_instances(r1["rd"], r1["contacts"], r1["whiffs"], 0,
                            channels=reduced)
    assert np.stack(Xd).shape[1] == Xb.shape[1] - 7, \
        "dropping one channel should remove exactly 7 columns"
    ov_dsho_only, _, _ = loro_eval(rallies, ("dsho",))
    print(f"  --drop: dsho-only synth coverage {ov_dsho_only:.1%} vs "
          f"full BASE {ov_base:.1%}")
    # the failure mode guarded here (loro_eval silently ignoring the
    # channel set) would make the two EQUAL, so any deterministic gap is
    # teeth; 5 points leaves margin (measured gap on this seeded synth:
    # 8.3 points, 66.7% vs 75.0% — coverage@2x has a luck floor, garbage
    # ranking still hits a fair fraction at a 2x budget)
    assert ov_dsho_only <= ov_base - 0.05, \
        "a channel set stripped of the synth's only real signal should " \
        "do clearly worse — if it doesn't, loro_eval isn't honoring the " \
        "channel set"

    # labels_fingerprint: stable on identical rows, sensitive to a
    # single retimed contact (the failure mode it exists to expose:
    # label edits between runs silently voiding cross-run deltas)
    fp1 = labels_fingerprint(rallies)
    assert fp1 == labels_fingerprint(rallies), "must be deterministic"
    import copy
    r_edit = copy.deepcopy(rallies)
    tc0, team0, ty0 = r_edit[1]["contacts"][0]
    r_edit[1]["contacts"][0] = (tc0 + 0.01, team0, ty0)
    assert labels_fingerprint(r_edit) != fp1, \
        "a 10ms retime of one contact must change the fingerprint"
    print(f"  labels_fingerprint: stable {fp1}, flips on a 10ms edit OK")
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
