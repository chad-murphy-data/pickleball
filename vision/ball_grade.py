"""The graded run of ball_gate.md — all three checks, one verdict.

Runs the merged pipeline (candidates -> hitter chain with blur
gap-fill -> decoder -> arc refit -> replication) on ONE rally using
only the licensed grade-time inputs (serve pin, rally-window end,
clip, pose npz, court calibration), then scores it against that
rally's ball pass and taps — which are read ONLY here, as the answer
key. Anchor generation calls predict_contacts/blur_gap_fill directly
so hitter_chain's train score() never touches the sealed pass.

Configuration and licensing interpretation: the dated graded-run
note at the end of ball_gate.md, recorded before the run.

Dry-run on a TRAIN rally first (readiness discipline); then the one
sealed run:
    python3 vision/ball_grade.py --rally 7 --serve 164.7 --end 175.2 \
        --npz r0007.npz --clip r7_clip.mp4 --offset 164.50
    python3 vision/ball_grade.py --rally 8 ... --graded-run
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_ball_audit import detect_events, score_events, load_impacts  # noqa: E402
import ball_decoder as bdec  # noqa: E402
import ball_replicate as br  # noqa: E402
import hitter_chain as hc  # noqa: E402
import court3d as c3  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "vision"
SEALED = set()       # r10 spent 2026-09-01 (graded MIDDLE, now train); next seal = r20 when labeled


def make_anchors(npz, clip, offset, out_csv):
    z = np.load(npz)
    picked = hc.predict_contacts(npz, float(z["t"].min()),
                                 float(z["t"].max()))
    extra = hc.blur_gap_fill(npz, clip, offset, picked)
    picked = sorted(picked + extra)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "excitement_z", "track", "wrist_x",
                    "wrist_y", "paddle_x", "paddle_y"])
        w.writerows([(round(e[0], 3), round(e[1], 2), e[2],
                      round(e[3], 1), round(e[4], 1),
                      round(e[5], 1), round(e[6], 1)) for e in picked])
    print(f"anchors: {len(picked)} ({len(extra)} blur gap-fill) "
          f"-> {out_csv}")
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rally", type=int, required=True)
    ap.add_argument("--serve", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--offset", type=float, required=True)
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--graded-run", action="store_true")
    a = ap.parse_args()
    if a.rally in SEALED and not a.graded_run:
        raise SystemExit(f"rally {a.rally} is SEALED — this harness "
                         "runs it only as THE graded run (--graded-run)")

    anchors_csv = make_anchors(
        a.npz, a.clip, a.offset,
        str(Path(a.workdir) / f"anchors_grade_r{a.rally}.csv"))
    anchors = br.load_anchors(anchors_csv)
    zs = [float(r["excitement_z"] or 0)
          for r in csv.DictReader(open(anchors_csv))]

    # ---- decode with licensed window only
    byf, t0 = bdec.load_candidates(a.rally)
    f_min = round((a.serve - 0.3 - t0) * bdec.FPS)
    f_max = round((a.end + 0.3 - t0) * bdec.FPS)
    byf = {f: c for f, c in byf.items() if f_min <= f <= f_max}
    oflags = bdec.out_of_court_flags(byf, bdec.court_hull())
    # ANCHORS NEVER TOUCH A DECODE (final architecture, 2026-09-01,
    # measured twice): in the timing stream they eat the null pct
    # (77.8@86 vs @98.4 on r7); in the position stream their turn
    # waivers subsidize junk near wrists and block the lob excursion
    # (r10 V 66.2 anchored vs 73.6 anchor-free; r7 82.3 vs 84.8).
    # Their job is bound CLAIMING in check 3, nothing else.
    visited = bdec.decode(byf, None, oflags, None)
    refined = bdec.refine_arcs(visited, t0)
    # two-regime split (decoder-fix addendum): the TIMING stream feeds
    # check 2, on SELF-FEEDBACK anchors ONLY. Unioning the hitter-chain
    # anchors in was MEASURED HARMFUL on the r7 readiness run
    # (2026-08-31: timing 77.8@86 with the union vs 77.8@98.4 without —
    # hitter anchors add cheap-turn zones at fake swings, inflating the
    # event set and eating the null pct). Hitter anchors stay in the
    # POSITION stream, where check 3 shows they work (bound claiming).
    _, timing_ref = bdec.timing_decode(byf, None, oflags, t0, [])
    per_frame = {}
    for t, x, y in refined:
        per_frame[round((t - t0) * bdec.FPS)] = (t, x, y)
    print(f"decoded {len(visited)} visited points (position stream), "
          f"window {a.serve:.1f}-{a.end:.1f}s")

    # ================= answer key opens here =================
    # prefill_ok: r2-r5 carry only PREFILL contact rows (approximate
    # times). They bound the gate panel and the check-1 window, not
    # any tuned quantity, so an approximate bracket is sufficient;
    # load_impacts prints a WARNING when it falls back.
    imps, dead = load_impacts(rally=a.rally, prefill_ok=True)
    labels = list(csv.DictReader(open(DATA / f"ball_path_r{a.rally}.csv")))

    # ---- CHECK 1: V hit rate on the gate panel
    p_lo, p_hi = imps[0], imps[-1] + 0.5
    rates = {}
    for cls, tol in (("V", 25.0), ("S", 40.0)):
        hit = tot = 0
        for r in labels:
            if not r["x"] or r["vis"] != cls:
                continue
            tt = float(r["t_s"])
            if not (p_lo <= tt <= p_hi):
                continue
            tot += 1
            f0 = round((tt - t0) * bdec.FPS)
            best = min((math.hypot(per_frame[g][1] - float(r["x"]),
                                   per_frame[g][2] - float(r["y"]))
                        for g in (f0 - 1, f0, f0 + 1) if g in per_frame),
                       default=1e9)
            hit += int(best <= tol)
        rates[cls] = 100 * hit / max(tot, 1)
        print(f"CHECK 1 {cls}: {rates[cls]:.1f}% ({hit}/{tot})"
              + (" [bars: PASS>=70, FAIL<40]" if cls == "V" else ""))
    c1_pass = rates["V"] >= 70.0
    c1_fail = rates["V"] < 40.0

    # ---- CHECK 2: frozen battery, human-matched (Amendment 1)
    span = (imps[0] - 1.0, dead)
    hum_pts = [(float(r["t_s"]), float(r["x"]), float(r["y"]))
               for r in labels if r["x"] and r["vis"] == "V"]
    res = {}
    for name, pts in (("tracker", timing_ref), ("human", hum_pts)):
        evs = detect_events(pts)
        obs, p95, pct, med = score_events(evs, imps, span)
        res[name] = (obs, pct, p95)
        print(f"CHECK 2 turns[{name:7s}]: recall {100*obs:.1f}% at "
              f"null pct {100*pct:.0f} (median {100*med:.1f}, "
              f"95th {100*p95:.1f})")
    c2_pass = (res["tracker"][0] >= res["human"][0]
               and res["tracker"][1] >= res["human"][1])
    c2_nullfail = res["tracker"][0] <= res["tracker"][2]
    print(f"CHECK 2 human-matched: {'PASS' if c2_pass else 'no'} "
          f"(recall {100*res['tracker'][0]:.0f} vs "
          f"{100*res['human'][0]:.0f}, pct {100*res['tracker'][1]:.0f} "
          f"vs {100*res['human'][1]:.0f}); "
          f"beats own null 95th: {'no' if c2_nullfail else 'yes'}")

    # ---- CHECK 3: replication (ball_replicate machinery)
    # bounds come from the TIMING stream's turns — the instrument that
    # passes check 2 (two-regime completion: "when" from timing,
    # position evidence from the position stream). Measured 2026-09-01:
    # position-stream turns under the anchor-free decode broke r7's
    # bound structure (3/8 matched vs readiness 7/8).
    pts = [(t0 + f / bdec.FPS, x, y) for f, x, y in visited]
    turns = [e for e in detect_events(timing_ref)
             if a.serve - 0.3 <= e < a.end - 0.05]
    angs = br.turn_angles(timing_ref, turns)
    X3, x2, _ = c3.load_landmarks()
    P = c3.dlt(X3, x2)
    floors = br.track_floor(a.npz, P)
    anchors = br.dedupe_anchors(anchors, zs, br.track_sides(floors),
                                turns)
    print(f"anchors after dedupe: {len(anchors)}")
    matched = br.claim_bounds(turns, angs, timing_ref, anchors)  # LOOSE
    bounds = matched + [a.end]
    bounce_evs = [e for e in turns if e not in set(matched)]
    obs = [(t, x, y, 1.0) for t, x, y in pts]
    c3_pass = br.compare(a.rally, (obs, bounds, bounce_evs),
                         br.human_side(a.rally, a.end), P, floors,
                         anchors)

    # ---- verdict per the frozen bars
    if c1_pass and c2_pass and c3_pass:
        verdict = "PASS"
    elif c1_fail or c2_nullfail:
        verdict = "FAIL (autopsy required)"
    else:
        verdict = "MIDDLE (one train-only iteration, then one " \
                  "re-grade on a NEWLY labeled sealed rally)"
    print(f"\n=== GATE VERDICT, rally {a.rally}: {verdict} ===")


if __name__ == "__main__":
    main()
