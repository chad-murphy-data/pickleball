"""Fast vs slow on TRUE contact windows — the step-2 ceiling
instrument. (2026-08-18) — EXPLORATION, not a verdict.

The pipeline is a two-step process (user call, 2026-08-18): step 1 =
swing/no-swing (swing_explore.py detects, channel_ablation.py measures),
step 2 = shot type. This script measures the COARSEST useful step-2
split — fast vs slow ("if not fast, then slow", user proposal same
day) — with placement error removed: features are read at the LABELED
contact time on the picked hitter track. Deployed accuracy will be
lower (the detector places ~75% of contacts upstream); this isolates
whether the type signal EXISTS, same role contact_ceiling.py played
for detection.

Why fast/slow before any finer taxonomy: detection misses concentrate
in soft shots while per-rally shot COUNTS are near-exact (decoder
161/162), so "slow = total - fast" makes fast the only class needing
direct recognition — and the binary is the smallest taxonomy that
supports the firefight/speed-up analytics (who breaks the dink rally,
firefight length, hands-battle win rates).

TYPE MAPPING (frozen before any real-data result; changing it voids
comparisons across runs):
    fast     = smash, speed-up, drive, counter
    slow     = dink, drop, lob, reset
    position = serve, return   (excluded: rally position identifies
               them for free downstream — first two shots)
    untyped  = "", "other"     (excluded, counted; tag them 'fast' or
               'slow' in the type column — coarse tags accepted here —
               to add them)
Unknown type words are excluded and printed LOUDLY so a vocabulary
drift never silently vanishes rows.

Three feature sets, because the answer directs the build:
    CADENCE   gaps to the previous/next labeled contact (any team,
              capped at GAP_CAP_S). Downstream these come from decoder
              timestamps — if CADENCE alone matches the rest, fast/slow
              ships on timing and needs no pose at the contact at all.
    POSE      window_feats at the true time, CHANNELS_EXT (leg/dgaze
              added nothing to DETECTION, but type is a different
              question — locomotion may mark firefights). Note POSE is
              not cadence-blind: window_feats carries its own
              label-free time-since-own-last-peak proxy.
    POSE+CAD  the deployable combination; per-type table and confusion
              are printed for this one.

Train split only (label_split.csv), same rule as feature_check.py:
holdout rallies are never loaded and burn on use.

RUN (same flat folder as swing_explore.py; numpy only, ~1 min):
    python3 fastslow_check.py

SELF-TEST (no files): python3 fastslow_check.py --selftest
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from contact_ceiling import load_rosters, load_labels
from swing_explore import (window_feats, fit_logreg, predict,
                           CHANNELS_EXT, PRE_S, track_series)
from feature_check import load_split, pick_hitter, auc
from channel_ablation import (assemble_rallies, labels_fingerprint,
                              pose_fingerprint)

LABELS = "contact_labels_chicago0725.csv"
WINDOWS_V4 = "rally_windows_chicago0725_v4.csv"
POSE_DIR = "pose_rtm"
SPLIT = "label_split.csv"

FAST = frozenset({"smash", "speed-up", "drive", "counter", "fast"})
SLOW = frozenset({"dink", "drop", "lob", "reset", "slow"})
POSITION = frozenset({"serve", "return"})
GAP_CAP_S = 3.0        # matches window_feats' own cadence-proxy cap
MIN_PER_CLASS = 8

FEATSETS = (("CADENCE  (label gaps only — no pose)", "cad"),
            ("POSE     (window_feats EXT at true time)", "pose"),
            ("POSE+CAD (the deployable combination)", "both"))


def classify_type(ty):
    """'fast' | 'slow' | 'position' | 'untyped' | 'unknown'."""
    ty = (ty or "").strip().lower()
    if ty in FAST:
        return "fast"
    if ty in SLOW:
        return "slow"
    if ty in POSITION:
        return "position"
    if ty in ("", "other"):
        return "untyped"
    return "unknown"


def feat_vec(row, key):
    if key == "cad":
        return row["cad"]
    if key == "pose":
        return row["pose"]
    return np.concatenate([row["pose"], row["cad"]])


def contact_rows(rallies):
    """One row per classifiable contact that has pose coverage. ALL
    three feature sets are evaluated on this same instance set —
    comparability beats n (a cadence row needs no pose, but scoring it
    would compare feature sets on different contacts). Gaps use every
    labeled contact (serves included): they are real rally events and
    the tempo measure must reflect them."""
    rows = []
    excl = {"position": 0, "untyped": 0, "unknown": 0, "no_pose": 0,
            "unknown_words": set()}
    for cum in sorted(rallies):
        r = rallies[cum]
        tracks = list(r["rd"]["tracks"].values())
        all_ct = sorted(t for t, *_ in r["contacts"])
        for tc, team, ty in r["contacts"]:
            cls = classify_type(ty)
            if cls in ("position", "untyped", "unknown"):
                excl[cls] += 1
                if cls == "unknown":
                    excl["unknown_words"].add((ty or "").strip().lower())
                continue
            ser = pick_hitter(tracks, team ^ r["m"], tc)
            xp = window_feats(ser, tc, channels=CHANNELS_EXT) \
                if ser is not None else None
            if xp is None:
                excl["no_pose"] += 1
                continue
            prev = [c for c in all_ct if c < tc - 1e-9]
            nxt = [c for c in all_ct if c > tc + 1e-9]
            xc = np.array([min(tc - prev[-1], GAP_CAP_S) if prev
                           else GAP_CAP_S,
                           min(nxt[0] - tc, GAP_CAP_S) if nxt
                           else GAP_CAP_S])
            rows.append({"cum": cum, "tc": tc,
                         "ty": (ty or "").strip().lower(),
                         "cls": cls, "pose": xp, "cad": xc})
    return rows, excl


def class_guard(rows):
    nf = sum(1 for r in rows if r["cls"] == "fast")
    ns = len(rows) - nf
    if min(nf, ns) < MIN_PER_CLASS:
        raise SystemExit(
            f"only {nf} fast / {ns} slow classifiable contacts — need "
            f">={MIN_PER_CLASS} per class. Tag some 'other' contacts "
            f"fast/slow to proceed.")
    return nf, ns


def loro(rows, key):
    """Leave-one-rally-out predictions P(fast). Returns (acc, auc_val,
    preds) over the full row list, in row order."""
    cums = sorted({r["cum"] for r in rows})
    preds = np.full(len(rows), np.nan)
    for held in cums:
        tr = [i for i, r in enumerate(rows) if r["cum"] != held]
        te = [i for i, r in enumerate(rows) if r["cum"] == held]
        assert held not in {rows[i]["cum"] for i in tr}
        X = np.stack([feat_vec(rows[i], key) for i in tr])
        y = np.array([rows[i]["cls"] == "fast" for i in tr], float)
        model = fit_logreg(X, y)
        preds[te] = predict(model,
                            np.stack([feat_vec(rows[i], key)
                                      for i in te]))
    assert np.isfinite(preds).all()
    truth = np.array([r["cls"] == "fast" for r in rows])
    acc = float(((preds >= 0.5) == truth).mean())
    auc_val = auc(preds[truth], preds[~truth])
    return acc, auc_val, preds


def per_type_table(rows, preds):
    """{ty: (n_correct, n)} under the 0.5 threshold."""
    out = {}
    for r, p in zip(rows, preds):
        ok = (p >= 0.5) == (r["cls"] == "fast")
        a, b = out.get(r["ty"], (0, 0))
        out[r["ty"]] = (a + ok, b + 1)
    return out


def confusion(rows, preds):
    """((fast->fast, fast->slow), (slow->fast, slow->slow))."""
    ff = fs = sf = ss = 0
    for r, p in zip(rows, preds):
        pf = p >= 0.5
        if r["cls"] == "fast":
            ff, fs = ff + pf, fs + (not pf)
        else:
            sf, ss = sf + pf, ss + (not pf)
    return (ff, fs), (sf, ss)


def report(rows, excl, n_holdout):
    nf, ns = class_guard(rows)
    n = len(rows)
    print(f"classified contacts: {n}  (fast {nf} / slow {ns}; majority "
          f"baseline {max(nf, ns) / n:.1%})")
    print(f"excluded: {excl['position']} serve/return (position-"
          f"identified downstream), {excl['untyped']} untyped/'other' "
          f"(tag them fast/slow to add), {excl['no_pose']} without pose "
          f"coverage, {n_holdout} holdout (untouched)")
    if excl["unknown"]:
        print(f"!! {excl['unknown']} contacts with UNKNOWN type words "
              f"excluded: {sorted(excl['unknown_words'])} — extend the "
              f"frozen mapping deliberately, don't let vocabulary "
              f"drift eat rows")
    print()
    preds_both = None
    for name, key in FEATSETS:
        acc, auc_val, preds = loro(rows, key)
        print(f"  {name:<44} acc {acc:6.1%}   AUC {auc_val:.3f}")
        if key == "both":
            preds_both = preds
    (ff, fs), (sf, ss) = confusion(rows, preds_both)
    print(f"\nPOSE+CAD confusion:  true fast -> {ff} fast / {fs} slow"
          f"     true slow -> {sf} fast / {ss} slow")
    print("POSE+CAD per true type:")
    for ty, (ok, b) in sorted(per_type_table(rows, preds_both).items(),
                              key=lambda kv: -kv[1][1]):
        print(f"    {ty:<10} {ok:>3}/{b:<3} {ok / b:6.1%}")
    print(f"\nn={n} — one contact is {1 / n:.1%}; read differences in "
          f"~5pp grains. True-time features = a CEILING for step 2, "
          f"not deployed accuracy (placement error comes on top). A "
          f"result worth believing graduates to a fresh "
          f"pre-registration on untouched holdout "
          f"(data/vision/label_split.csv).")


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
    n_holdout = 0
    if split is not None:
        kept = {}
        for cum, d in labels.items():
            if split.get(cum, "train") == "train":
                kept[cum] = d
            else:
                n_holdout += len(d["contacts"])
        labels = kept
        split_note = f"train split only ({a.split}); holdout untouched"
    else:
        split_note = (f"!! no {a.split} found — running on ALL labeled "
                      f"rallies")
    rallies = assemble_rallies(labels, a.pose_dir)
    if len(rallies) < 3:
        raise SystemExit(f"need >=3 rallies with labels + pose (found "
                         f"{len(rallies)}) — check --pose-dir")
    print(f"fastslow_check: {len(rallies)} rallies "
          f"(leave-one-rally-out; EXPLORATION, not a gate)")
    print(f"labels fingerprint: {labels_fingerprint(rallies)}   "
          f"pose fingerprint: {pose_fingerprint(rallies)}")
    print(f"({split_note}; numbers only compare across runs printing "
          f"the SAME two fingerprints)")
    print(f"mapping: fast={'/'.join(sorted(FAST))}  "
          f"slow={'/'.join(sorted(SLOW))}\n")
    rows, excl = contact_rows(rallies)
    report(rows, excl, n_holdout)


# ------------------------------------------------------------ selftest


FPS = 30.0


def _mk_track(t0, t1, side, bursts=()):
    """Full-span synthetic track; bursts = [(time, amplitude)]. Each
    burst gets a 0.4x lead ripple 0.5 s early so pick_hitter's
    prep-arc energy criterion selects the hitter track."""
    n = int((t1 - t0) * FPS)
    t = t0 + np.arange(n) / FPS
    box = np.zeros((n, 4))
    box[:, 0], box[:, 1] = 100 + side * 400, 300
    box[:, 2], box[:, 3] = 60, 160
    kpt = np.zeros((n, 17, 2))
    kpt[:, :, 0] = box[:, 0:1] + 30
    kpt[:, :, 1] = 300 + np.linspace(0, 150, 17)[None, :]
    kpc = np.full((n, 17), 0.9)
    for bt, amp in bursts:
        for center, a in ((bt, amp), (bt - 0.5, 0.4 * amp)):
            m = np.abs(t - center) < 0.1
            kpt[m, 9, 0] += a * np.sin(np.arange(m.sum()))
    ser = track_series(t, box, kpt, kpc, FPS)
    ser["side"] = side
    ser["H"] = 720
    return ser


def _mk_rally(cum, contacts, amp_of):
    """contacts = [(t, team, ty)]; amp_of(ty) -> burst amplitude.
    One track per side, m=0 (team == image side)."""
    t0 = min(t for t, *_ in contacts) - 3.0
    t1 = max(t for t, *_ in contacts) + 3.0
    tracks = {}
    for side in (0, 1):
        bursts = [(t, amp_of(ty)) for t, team, ty in contacts
                  if team == side]
        tracks[side + 1] = _mk_track(t0, t1, side, bursts)
    return {"rd": {"tracks": tracks, "z": None, "bounds": (t0, t1)},
            "contacts": sorted(contacts), "whiffs": [], "m": 0}


def _cycle_types(k):
    """fast/slow so that BOTH teams (k%2) hit BOTH classes — otherwise
    the side feature alone would fake separation."""
    return "smash" if k % 4 in (0, 3) else "dink"


def selftest():
    # ---- mapping frozen, normalization, vocabulary drift is loud
    assert classify_type("smash") == "fast"
    assert classify_type("counter") == "fast"
    assert classify_type(" Speed-Up ") == "fast"
    assert classify_type("dink") == "slow"
    assert classify_type("lob") == "slow"
    assert classify_type("serve") == "position"
    assert classify_type("return") == "position"
    assert classify_type("") == "untyped"
    assert classify_type(None) == "untyped"
    assert classify_type("other") == "untyped"
    assert classify_type("banana") == "unknown"
    assert classify_type("fast") == "fast" and classify_type("slow") == "slow"
    print("selftest: mapping OK")

    # ---- separable synth: fast bursts 4x slow bursts, uniform gaps.
    # Each rally is BRACKETED by a serve and an 'other' (both excluded
    # from classification, like real rallies) so the first/last
    # classified rows don't carry the capped-3.0s edge gaps — without
    # the bracket, edge rows' distinctive cadence features correlate
    # with whatever class the cycle puts at the edges and cadence
    # 'separates' for a fake reason (first draft of this test did
    # exactly that, 66.7%). Interior cadence is then CONSTANT: logreg
    # can only emit the base rate, so cadence accuracy must equal the
    # majority share exactly. POSE must separate on amplitude.
    def sep_rallies():
        rallies = {}
        for cum in (1, 2, 3, 4):
            contacts = [(103.0 + 2.2 * k, k % 2,
                         "serve" if k == 0 else
                         "other" if k == 11 else _cycle_types(k))
                        for k in range(12)]
            rallies[cum] = _mk_rally(
                cum, contacts,
                lambda ty: 40.0 if classify_type(ty) == "fast" else 10.0)
        return rallies

    rows, excl = contact_rows(sep_rallies())
    assert len(rows) == 40 and excl["no_pose"] == 0, \
        (len(rows), excl)
    nf, ns = class_guard(rows)
    assert nf == 16 and ns == 24, (nf, ns)
    acc_p, auc_p, preds_p = loro(rows, "pose")
    acc_c, _auc_c, _ = loro(rows, "cad")
    acc_b, _auc_b, _ = loro(rows, "both")
    assert acc_p >= 0.9 and auc_p >= 0.95, (acc_p, auc_p)
    assert acc_b >= 0.9, acc_b
    assert abs(acc_c - ns / (nf + ns)) < 0.05, acc_c
    print(f"selftest: separable synth OK (pose {acc_p:.0%}, "
          f"cadence pinned at base rate {acc_c:.0%})")

    # ---- determinism: same assembled objects re-run + fresh rebuild
    # must reproduce bit-identical predictions (regression on the
    # 2026-08-18 order-dependence incident class: window_feats mutates
    # ser via the _peaks cache, so a second pass over warm objects is
    # exactly the hazard case).
    acc_p2, _a2, preds_p2 = loro(rows, "pose")
    assert acc_p2 == acc_p and np.array_equal(preds_p, preds_p2)
    rows_f, _ = contact_rows(sep_rallies())
    _accf, _aucf, preds_f = loro(rows_f, "pose")
    assert np.array_equal(preds_p, preds_f), "fresh rebuild differs"
    print("selftest: determinism OK (warm re-run + fresh rebuild "
          "bit-identical)")

    # ---- null synth: identical bursts both classes — nothing real to
    # find; LORO accuracy must hover near the majority share. Two
    # de-aliasing measures, both load-bearing: (a) classes are
    # SHUFFLED per rally (seeded), not assigned by slot index — with a
    # fixed k->class pattern repeated across rallies, frame-grid float
    # aliasing (window masks gaining/losing one frame as a
    # deterministic function of k) separated this null at 85%
    # (arm_cmax d=1.09 on sd=0.012 wobble); (b) contact times are
    # jittered off the exact-frame grid. Real data has neither
    # determinism. Bracketed like sep_rallies for the edge-gap channel.
    def null_rallies():
        rallies = {}
        for cum in range(1, 7):
            rng = np.random.default_rng(1000 + cum)
            types = list(rng.permutation(["smash"] * 4 + ["dink"] * 6))
            contacts = [(103.0 + 2.2 * k + rng.normal(0, 0.05), k % 2,
                         "serve" if k == 0 else
                         "other" if k == 11 else types[k - 1])
                        for k in range(12)]
            rallies[cum] = _mk_rally(cum, contacts, lambda ty: 25.0)
        return rallies

    rows_n, _ = contact_rows(null_rallies())
    assert len(rows_n) == 60
    acc_n, auc_n, _ = loro(rows_n, "both")
    assert acc_n <= 0.72, f"null synth separated at {acc_n:.0%}"
    assert 0.28 <= auc_n <= 0.72, auc_n
    print(f"selftest: null synth OK (acc {acc_n:.0%}, AUC {auc_n:.2f} "
          f"— chance-level as required)")

    # ---- cadence synth: identical bursts, but fast contacts arrive in
    # a tight exchange (0.45 s gaps) after a 2.2 s dink phase. CADENCE
    # alone must separate. No pose-only assertion: window_feats' own
    # cadence proxy legitimately sees tempo too — that's real physics,
    # not a rig bug.
    def cad_rallies():
        rallies = {}
        for cum in (1, 2, 3, 4):
            slow = [(103.0 + 2.2 * k, k % 2, "dink") for k in range(6)]
            t_ff = slow[-1][0] + 2.2
            fast = [(t_ff + 0.45 * k, k % 2, "counter")
                    for k in range(6)]
            rallies[cum] = _mk_rally(cum, slow + fast,
                                     lambda ty: 25.0)
        return rallies

    rows_c, _ = contact_rows(cad_rallies())
    assert len(rows_c) == 48
    acc_cad, auc_cad, _ = loro(rows_c, "cad")
    assert acc_cad >= 0.9 and auc_cad >= 0.95, (acc_cad, auc_cad)
    print(f"selftest: cadence synth OK (cadence-only {acc_cad:.0%})")

    # ---- exclusion accounting: serves/returns/other/unknown counted,
    # never scored
    mixed = {1: _mk_rally(1, [(103.0, 0, "serve"), (104.5, 1, "return"),
                              (106.7, 0, "other"), (108.9, 1, "banana"),
                              (111.1, 0, "dink"), (113.3, 1, "smash")],
                          lambda ty: 25.0)}
    rows_m, excl_m = contact_rows(mixed)
    assert [r["ty"] for r in rows_m] == ["dink", "smash"]
    assert excl_m["position"] == 2 and excl_m["untyped"] == 1
    assert excl_m["unknown"] == 1 and \
        excl_m["unknown_words"] == {"banana"}
    # gaps computed over ALL labeled contacts, serves included
    assert abs(rows_m[0]["cad"][0] - 2.2) < 1e-6
    print("selftest: exclusion accounting OK")

    # ---- class guard fires on thin classes
    thin = [r for r in rows if r["cls"] == "slow"][:20] + \
        [r for r in rows if r["cls"] == "fast"][:5]
    try:
        class_guard(thin)
        raise AssertionError("class_guard failed to fire")
    except SystemExit:
        pass
    print("selftest: class guard OK")

    # ---- per-type table + confusion plumbing
    fake = [{"ty": "smash", "cls": "fast"}, {"ty": "smash", "cls": "fast"},
            {"ty": "dink", "cls": "slow"}]
    tab = per_type_table(fake, np.array([0.9, 0.2, 0.1]))
    assert tab == {"smash": (1, 2), "dink": (1, 1)}, tab
    (ff, fs), (sf, ss) = confusion(fake, np.array([0.9, 0.2, 0.1]))
    assert (ff, fs, sf, ss) == (1, 1, 0, 1)
    print("selftest: table/confusion plumbing OK")

    print("\nselftest: ALL OK")


if __name__ == "__main__":
    main()
