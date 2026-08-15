"""Gate C: the pose-stream contact ceiling. PRE-REGISTERED — the bars
live in vision/contact_gate.md and in BARS below, frozen 2026-08-15
before any timestamped label existed. Do not tune anything here after
looking at real output; the one pre-specified fallback (tolerance
0.30 -> 0.20 s if the null is fat) is already encoded.

THE QUESTION. A trained classifier can only fire where a candidate
event exists. Gate B's 0.442 recall was measured through an audio gate
later shown uncorrelated with shots (a near-random thinning), so the
pose channel's own ceiling was never measured. This script measures it:
what fraction of hand-stamped contacts have a torso-relative wrist-speed
peak, on an identity-continuous track on the hitter's side, within
tolerance — under a candidate BUDGET that keeps "a peak exists"
falsifiable (unbudgeted, dense noise peaks cover any label set, which
is why the shifted-label null is computed by the IDENTICAL procedure
and reported next to the headline).

INPUTS
    data/vision/contact_labels_chicago0725.csv   the new tool's export
    data/vision/pose/r####.npz                   vision/pose_extract.py
    data/vision/rally_windows_chicago0725_v4.csv team rosters per rally

RUN
    python vision/contact_ceiling.py             # prints the verdict
    python vision/contact_ceiling.py --selftest  # no files needed

Edge note: a real serve sits PAD_PRE=1.5 s inside its window while a
shifted label can land within tolerance of a window edge, deflating the
null by ~2% absolute at most (0.6 s of edge per ~30 s window) — i.e.
the null is slightly conservative in the direction that makes the
null<=0.55 bar HARDER to clear, never easier.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from swing_probe import strongest_first

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data/vision/contact_labels_chicago0725.csv"
POSE_DIR = ROOT / "data/vision/pose"
WINDOWS_V4 = ROOT / "data/vision/rally_windows_chicago0725_v4.csv"
REPORT = ROOT / "data/vision/contact_ceiling_report.json"

# ---- the pre-registered spec (vision/contact_gate.md) ----------------
TOL_S = 0.30              # primary tolerance
TOL_FALLBACK_S = 0.20     # one shot, only if null mean > NULL_MAX
MULT_PRIMARY = 2          # candidate budget = mult x side's contacts
MULT_SECONDARY = 4
BARS = {"overall": 0.85, "fast": 0.75, "kill": 0.75}
NULL_MAX = 0.55
N_NULL = 200
NULL_SEED = 20260815
REFRACTORY_S = 0.25
PEAK_FLOOR = 0.02         # permissive; budgets do the real selection
# frozen from vision/swing_score.py FAST_TYPES (not imported: that module
# is the Gate B scorer and stays untouched)
FAST_TYPES = {"speed-up", "counter", "smash"}

# COCO-17 keypoints
L_SHO, R_SHO, L_WRIST, R_WRIST, L_HIP, R_HIP = 5, 6, 9, 10, 11, 12


# ------------------------------------------------------------- loading


def load_rosters(path: Path):
    """rally_cum -> (frozenset teamA, frozenset teamB)."""
    out = {}
    for r in csv.DictReader(open(path)):
        out[int(r["rally_cum"])] = (
            frozenset(u.lower() for u in r["teamA_uuids"].split("|")),
            frozenset(u.lower() for u in r["teamB_uuids"].split("|")))
    return out


def load_labels(path: Path, rosters):
    """rally_cum -> {"contacts": [(t, team, type)], "whiffs": [(t, team)],
    "jitter": (t_tap_shot1, pin_ref) | None}. t = refined || tap."""
    out = {}
    for r in csv.DictReader(open(path)):
        cum = int(r["rally_cum"])
        d = out.setdefault(cum, {"contacts": [], "whiffs": [], "jitter": None})
        t = float(r["t_refined_s"] or r["t_tap_s"])
        uuid = r["hitter_uuid"].lower()
        ta, tb = rosters.get(cum, (frozenset(), frozenset()))
        team = 0 if uuid in ta else 1 if uuid in tb else None
        if team is None:
            raise SystemExit(f"rally {cum}: hitter {uuid[:8]} in neither "
                             f"roster — labels/windows mismatch")
        if r.get("contact", "1") == "0":
            d["whiffs"].append((t, team))
        else:
            d["contacts"].append((t, team, r["shot_type"] or "other"))
        if int(r["shot_index"]) == 1 and r.get("pin_ref_s"):
            d["jitter"] = (float(r["t_tap_s"]), float(r["pin_ref_s"]))
    for d in out.values():
        d["contacts"].sort()
        d["whiffs"].sort()
    return out


# ---------------------------------------------- torso-relative candidates


def track_peaks(t, box, kpt, kpc, fps):
    """Candidate peaks for ONE track: torso-relative wrist speed (the
    Gate B autopsy's v2 feature — absolute wrist speed registers
    locomotion; subtracting the hip and scaling by the torso cancels
    both locomotion and near/far image scale). Returns [(t, v)]."""
    n = len(t)
    if n < 3:
        return []
    bh = box[:, 3] - box[:, 1]
    cx = (box[:, 0] + box[:, 2]) / 2

    def joint(a, b, fallback_y):
        ok = (kpc[:, a] >= 0.2) & (kpc[:, b] >= 0.2)
        pt = (kpt[:, a] + kpt[:, b]) / 2
        fb = np.stack([cx, box[:, 1] + fallback_y * bh], 1)
        return np.where(ok[:, None], pt, fb), ok

    hip, hip_ok = joint(L_HIP, R_HIP, 0.62)
    sho, sho_ok = joint(L_SHO, R_SHO, 0.25)
    scale = np.where(hip_ok & sho_ok,
                     np.linalg.norm(sho - hip, axis=1), 0.35 * bh)
    scale = np.maximum(scale, 3.0)

    dt = np.diff(t)
    ok_dt = dt <= 2.5 / fps            # no speeds across detection gaps
    v = np.full(n, np.nan)
    for w in (L_WRIST, R_WRIST):
        rel = kpt[:, w] - hip
        conf = kpc[:, w] >= 0.15
        d = np.linalg.norm(np.diff(rel, axis=0), axis=1)
        vw = d / np.maximum(dt * fps, 1e-9) / scale[1:]
        vw = np.where(ok_dt & conf[1:] & conf[:-1], vw, np.nan)
        v[1:] = np.fmax(v[1:], vw)     # fmax: nan-aware max over wrists
    s = np.nan_to_num(v)
    if n > 2:
        s = np.convolve(s, [0.25, 0.5, 0.25], mode="same")
    cands = [(float(t[i]), float(s[i])) for i in range(1, n - 1)
             if s[i] >= PEAK_FLOOR and s[i] >= s[i - 1] and s[i] >= s[i + 1]]
    return strongest_first(cands, REFRACTORY_S)


def rally_candidates(z):
    """npz (or dict of arrays) -> ({side: [(t,v)]}, (t_lo, t_hi)).
    side -1 (junk fragments) pools into key 2: usable for ceiling-any,
    never for ceiling-side."""
    t, trk = np.asarray(z["t"]), np.asarray(z["track"])
    side, box = np.asarray(z["side"]), np.asarray(z["box"])
    kpt, kpc = np.asarray(z["kpt"]), np.asarray(z["kpc"])
    fps = float(np.asarray(z["fps"]).ravel()[0])
    by_side = {0: [], 1: [], 2: []}
    for tid in np.unique(trk):
        m = trk == tid
        order = np.argsort(t[m])
        pk = track_peaks(t[m][order], box[m][order],
                         kpt[m][order], kpc[m][order], fps)
        s = int(side[m][0])
        by_side[s if s >= 0 else 2].extend(pk)
    for s in by_side:
        by_side[s].sort()
    return by_side, (float(t.min()), float(t.max())) if len(t) else (0, 0)


# --------------------------------------------------------- the ceiling


def budget_top(cands, k):
    return sorted(sorted(cands, key=lambda x: -x[1])[:k])


def covered_flags(times, cands, tol):
    ts = np.array([c[0] for c in cands])
    out = []
    for t in times:
        out.append(bool(len(ts)) and bool(np.min(np.abs(ts - t)) <= tol))
    return out


def rally_coverage(contacts, by_side, mult, tol):
    """Per-rally flags under the better of the two team->side mappings;
    the identical maximization runs inside every null draw, so its
    selection optimism is absorbed by the null (pre-registered)."""
    best = None
    for m in (0, 1):
        flags = [False] * len(contacts)
        for s in (0, 1):
            idx = [i for i, (_, team, *_ ) in enumerate(contacts)
                   if team ^ m == s]
            if not idx:
                continue
            sel = budget_top(by_side.get(s, []),
                             math.ceil(mult * len(idx)))
            fl = covered_flags([contacts[i][0] for i in idx], sel, tol)
            for i, f in zip(idx, fl):
                flags[i] = f
        score = sum(flags)
        if best is None or score > best[0]:
            best = (score, m, flags)
    return best[2], best[1]


def rally_coverage_any(contacts, by_side, mult, tol):
    pool = by_side.get(0, []) + by_side.get(1, []) + by_side.get(2, [])
    sel = budget_top(pool, math.ceil(mult * len(contacts)))
    return covered_flags([c[0] for c in contacts], sel, tol)


def shift_contacts(contacts, lo_hi, off):
    t0, t1 = lo_hi
    L = t1 - t0
    return [((t - t0 + off) % L + t0, team, *rest)
            for t, team, *rest in contacts]


def score_all(rallies, mult, tol, shift_offsets=None):
    """rallies: {cum: {"contacts":..., "cands":..., "bounds":...}}.
    Returns per-contact records [(cum, type, covered)] under side
    mapping-maximization; shift_offsets ({cum: off}) applies the null."""
    recs = []
    for cum, r in rallies.items():
        contacts = r["contacts"]
        if not contacts:
            continue
        if shift_offsets is not None:
            contacts = shift_contacts(contacts, r["bounds"],
                                      shift_offsets[cum])
        flags, _m = rally_coverage(contacts, r["cands"], mult, tol)
        recs.extend((cum, c[2] if len(c) > 2 else "other", f)
                    for c, f in zip(contacts, flags))
    return recs


def summarize(recs):
    n = len(recs)
    ok = sum(1 for *_, f in recs if f)
    fast = [(c, ty, f) for c, ty, f in recs if ty in FAST_TYPES]
    per_type = {}
    for _, ty, f in recs:
        a, b = per_type.get(ty, (0, 0))
        per_type[ty] = (a + f, b + 1)
    return {"n": n, "overall": ok / n if n else float("nan"),
            "fast": (sum(f for *_, f in fast) / len(fast)
                     if fast else float("nan")),
            "n_fast": len(fast), "per_type": per_type}


def wilson(p, n, z=1.96):
    if n == 0 or p != p:
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def null_offsets(rallies, rng):
    out = {}
    for cum, r in rallies.items():
        L = r["bounds"][1] - r["bounds"][0]
        lo = 3.0 if L >= 8 else 2.0
        lo = min(lo, max(L / 3, 0.5))
        out[cum] = rng.uniform(lo, max(L - lo, lo + 1e-6))
    return out


def run_gate(rallies, n_null=N_NULL, seed=NULL_SEED, verbose=True):
    """The pre-registered battery. Returns the report dict."""
    rep = {"spec": {"tol_s": TOL_S, "fallback_tol_s": TOL_FALLBACK_S,
                    "mult_primary": MULT_PRIMARY, "bars": BARS,
                    "null_max": NULL_MAX, "n_null": n_null}}
    rng = np.random.default_rng(seed)

    def battery(tol):
        real = summarize(score_all(rallies, MULT_PRIMARY, tol))
        nulls = []
        for _ in range(n_null):
            offs = null_offsets(rallies, rng)
            nulls.append(summarize(
                score_all(rallies, MULT_PRIMARY, tol, offs))["overall"])
        nulls = np.array(nulls)
        if not len(nulls):
            return real, float("nan"), float("nan")
        return real, float(nulls.mean()), float(np.quantile(nulls, 0.95))

    real, null_mean, null_p95 = battery(TOL_S)
    tol_used = TOL_S
    fell_back = False
    if null_mean > NULL_MAX:
        fell_back = True
        real, null_mean, null_p95 = battery(TOL_FALLBACK_S)
        tol_used = TOL_FALLBACK_S

    sec = summarize(score_all(rallies, MULT_SECONDARY, tol_used))
    any_recs = []
    for cum, r in rallies.items():
        if r["contacts"]:
            any_recs.extend(
                (cum, c[2], f) for c, f in
                zip(r["contacts"],
                    rally_coverage_any(r["contacts"], r["cands"],
                                       MULT_PRIMARY, tol_used)))
    any_sum = summarize(any_recs)

    ov, fa = real["overall"], real["fast"]
    if ov >= BARS["overall"] and (fa != fa or fa >= BARS["fast"]) \
            and null_mean <= NULL_MAX:
        verdict = "PROCEED"
    elif ov < BARS["kill"]:
        verdict = "KILL"
    elif null_mean > NULL_MAX:
        verdict = "UNINTERPRETABLE — null fat even at fallback tolerance"
    else:
        verdict = "MIDDLE"
    ci = wilson(ov, real["n"])

    rep.update({
        "tol_used": tol_used, "fell_back": fell_back,
        "n_contacts": real["n"], "n_fast": real["n_fast"],
        "ceiling_side": ov, "ceiling_side_ci": ci,
        "ceiling_fast": fa,
        "per_type": {k: list(v) for k, v in real["per_type"].items()},
        "null_mean": null_mean, "null_p95": null_p95,
        "ceiling_side_4x": sec["overall"], "ceiling_any": any_sum["overall"],
        "verdict": verdict,
    })

    if verbose:
        print(f"\n=== Gate C: pose-stream contact ceiling "
              f"(tol ±{tol_used:.2f}s, budget {MULT_PRIMARY}x"
              f"{', FELL BACK from ±0.30' if fell_back else ''}) ===")
        print(f"  contacts scored     {real['n']}  "
              f"({len(rallies)} rallies)")
        print(f"  ceiling-side        {ov:6.1%}  "
              f"[{ci[0]:.1%}, {ci[1]:.1%}]   (bar {BARS['overall']:.0%})")
        print(f"  fast stratum        {fa:6.1%}  on n={real['n_fast']}"
              f"   (bar {BARS['fast']:.0%})")
        print(f"  shifted-label null  {null_mean:6.1%}  "
              f"p95 {null_p95:.1%}   (max {NULL_MAX:.0%})")
        print(f"  ceiling-side @4x    {sec['overall']:6.1%}   "
              f"ceiling-any @2x {any_sum['overall']:6.1%}")
        print("  per type:")
        for ty, (a, b) in sorted(real["per_type"].items(),
                                 key=lambda kv: -kv[1][1]):
            star = "  <- fast" if ty in FAST_TYPES else ""
            print(f"    {ty:<10} {a:>3}/{b:<3} {a / b:6.1%}{star}")
        print(f"  VERDICT: {verdict}")
    return rep


# ------------------------------------------------------------- assembly


def assemble(labels, pose_dir: Path):
    rallies, missing = {}, []
    for cum, d in labels.items():
        p = pose_dir / f"r{cum:04d}.npz"
        if not p.exists():
            missing.append(cum)
            continue
        z = np.load(p)
        cands, bounds = rally_candidates(z)
        rallies[cum] = {"contacts": d["contacts"], "cands": cands,
                        "bounds": bounds}
    if missing:
        print(f"WARNING: no pose npz for rallies {missing} — "
              f"run pose_extract first; scoring the rest")
    return rallies


def jitter_report(labels):
    ds = [tap - pin for d in labels.values()
          if d["jitter"] for tap, pin in [d["jitter"]]]
    if not ds:
        return None
    a = np.abs(ds)
    guard = max(0.4, 3 * float(np.quantile(a, 0.95)))
    print(f"  tap jitter vs {len(ds)} pinned serves: "
          f"median |d| {np.median(a):.2f}s, p95 {np.quantile(a, 0.95):.2f}s "
          f"-> training guard band {guard:.2f}s")
    return {"n": len(ds), "median_abs": float(np.median(a)),
            "p95_abs": float(np.quantile(a, 0.95)), "guard_band_s": guard}


# ------------------------------------------------------------ selftest


def synth_rally(rng, contacts, planted=True, fps=30.0, t0=100.0, t1=130.0):
    """Two tracks per side; hip+wrist share a locomotion sinusoid (which
    torso-relative speed must cancel); bursts planted at contact times on
    the mapped side's first track. Returns arrays shaped like an npz."""
    n = int((t1 - t0) * fps)
    ts = t0 + np.arange(n) / fps
    rows = {"t": [], "track": [], "side": [], "box": [], "kpt": [],
            "kpc": [], "conf": []}
    for tid in range(4):
        s = 0 if tid < 2 else 1
        bh = 240.0 if s == 0 else 120.0
        base = np.array([400 + 400 * (tid % 2), 600 if s == 0 else 380],
                        float)
        loco = 60 * np.sin(2 * np.pi * ts / 7 + tid)      # walking drift
        for i, t in enumerate(ts):
            cx = base[0] + loco[i]
            bot = base[1]
            box = np.array([cx - bh * 0.22, bot - bh, cx + bh * 0.22, bot],
                           np.float32)
            kpt = np.zeros((17, 2), np.float32)
            kpc = np.full(17, 0.9, np.float32)
            hip = np.array([cx, bot - 0.38 * bh])
            sho = np.array([cx, bot - 0.75 * bh])
            kpt[L_HIP] = kpt[R_HIP] = hip
            kpt[L_SHO] = kpt[R_SHO] = sho
            wr = hip + np.array([0.3 * bh, -0.2 * bh])
            wr = wr + rng.normal(0, 0.004 * bh, 2)        # idle jitter
            if planted:
                for (tc, team, *_ ) in contacts:
                    if team == s and tid in (0, 2) and abs(t - tc) < 0.08:
                        wr = wr + np.array([0.5 * bh *
                                            np.sin((t - tc) * 40), 0])
            kpt[L_WRIST] = wr
            kpt[R_WRIST] = wr + np.array([8, 4])
            rows["t"].append(t)
            rows["track"].append(tid)
            rows["side"].append(s)
            rows["box"].append(box)
            rows["kpt"].append(kpt)
            rows["kpc"].append(kpc)
            rows["conf"].append(0.9)
    return {"t": np.array(rows["t"]), "track": np.array(rows["track"]),
            "side": np.array(rows["side"], np.int8),
            "box": np.stack(rows["box"]), "kpt": np.stack(rows["kpt"]),
            "kpc": np.stack(rows["kpc"]),
            "conf": np.array(rows["conf"], np.float32),
            "fps": np.array([fps])}


def selftest():
    rng = np.random.default_rng(5)
    types = ["serve", "return", "drop", "dink", "dink", "speed-up",
             "counter", "counter", "smash", "dink", "counter", "smash"]

    def make(planted, flip_teams=False):
        rallies = {}
        for cum in (1, 2, 3):
            contacts = [(103.0 + k * 2.0 + rng.normal(0, 0.05),
                         (k % 2) ^ (1 if flip_teams else 0), types[k])
                        for k in range(12)]
            z = synth_rally(rng, [(t, team ^ (1 if flip_teams else 0))
                                  for t, team, _ in contacts],
                            planted=planted)
            cands, bounds = rally_candidates(z)
            rallies[cum] = {"contacts": sorted(contacts),
                            "cands": cands, "bounds": bounds}
        return rallies

    # planted: teams map straight onto sides (team k%2 -> side k%2)
    rallies = make(True)
    rep = run_gate(rallies, n_null=60, verbose=False)
    print(f"  planted:  ceiling {rep['ceiling_side']:.1%} "
          f"fast {rep['ceiling_fast']:.1%} null {rep['null_mean']:.1%} "
          f"-> {rep['verdict']}")
    assert rep["ceiling_side"] >= 0.90, "planted bursts not recovered"
    assert rep["verdict"] == "PROCEED", rep["verdict"]
    assert rep["null_mean"] <= 0.60, "null implausibly fat"

    # flipped team labels: the mapping maximization must recover it
    rep2 = run_gate(make(True, flip_teams=True), n_null=0, verbose=False)
    print(f"  flipped:  ceiling {rep2['ceiling_side']:.1%} "
          f"(mapping recovered)")
    assert rep2["ceiling_side"] >= 0.90, "team->side maximization failed"

    # control: no bursts -> ceiling must sit at the null, verdict KILL
    rep3 = run_gate(make(False), n_null=60, verbose=False)
    print(f"  control:  ceiling {rep3['ceiling_side']:.1%} "
          f"null {rep3['null_mean']:.1%} -> {rep3['verdict']}")
    assert rep3["verdict"] == "KILL", "no-signal control failed to KILL"
    assert abs(rep3["ceiling_side"] - rep3["null_mean"]) < 0.25, \
        "control ceiling should sit near its null"

    # loader: whiffs excluded, refined-time priority, roster mapping
    import io
    rosters = {7: (frozenset({"aaa"}), frozenset({"bbb"}))}
    rows = ('game,division,rally_in_game,rally_cum,shot_index,hitter_name,'
            'hitter_uuid,shot_type,contact,t_tap_s,t_refined_s,tap_rate,'
            'source,rally_note,pin_ref_s\n'
            '1,w,7,7,1,A,aaa,serve,1,10.50,10.44,0.5,prefill,,10.33\n'
            '1,w,7,7,2,B,bbb,whiff,0,12.00,,0.5,manual,,\n'
            '1,w,7,7,3,B,bbb,dink,1,13.00,,0.5,prefill,,\n')
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(rows)
        p = f.name
    lab = load_labels(Path(p), rosters)
    assert len(lab[7]["contacts"]) == 2 and len(lab[7]["whiffs"]) == 1
    assert abs(lab[7]["contacts"][0][0] - 10.44) < 1e-9, "refined wins"
    assert lab[7]["contacts"][0][1] == 0 and lab[7]["contacts"][1][1] == 1
    assert lab[7]["jitter"] == (10.50, 10.33)
    print("  loader: whiff exclusion, refined priority, rosters, jitter OK")
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--pose-dir", default=str(POSE_DIR), type=Path)
    ap.add_argument("--windows", default=str(WINDOWS_V4))
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not Path(a.labels).exists():
        raise SystemExit(f"{a.labels} not found — label the core 16 in "
                         f"data/vision/contact_audit_chicago0725.html "
                         f"and export first")
    mp = Path(a.pose_dir) / "meta.json"
    if mp.exists():
        be = json.loads(mp.read_text()).get("backend", "?")
        if be != "rtmpose-balanced":
            print(f"WARNING: pose backend '{be}' != the pre-registered "
                  f"'rtmpose-balanced' (contact_gate.md amendment) — "
                  f"this run is DIAGNOSTIC, not the gate")
    rosters = load_rosters(Path(a.windows))
    labels = load_labels(Path(a.labels), rosters)
    print(f"labels: {len(labels)} rallies, "
          f"{sum(len(d['contacts']) for d in labels.values())} contacts, "
          f"{sum(len(d['whiffs']) for d in labels.values())} whiffs")
    jit = jitter_report(labels)
    rallies = assemble(labels, a.pose_dir)
    if not rallies:
        raise SystemExit("no rallies with both labels and pose — "
                         "run pose_extract first")
    rep = run_gate(rallies, n_null=a.n_null)
    rep["jitter"] = jit
    rep["rallies_scored"] = sorted(rallies)
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print(f"report -> {a.report}")


if __name__ == "__main__":
    main()
