"""Check-3 lab: cache the expensive pipeline stages once per rally,
then iterate claiming logic cheaply.

Stage A (this script): per rally — windowed candidates, position
decode + refined arcs, timing stream, turns/angles, anchors (grade
CSVs, pose+blur pre-dedupe), floors, human side + its full fit.
Pickled to c3_cache_r{r}.pkl.

Windows = the exact battery runs.
"""
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
import ball_decoder as bdec           # noqa: E402
import ball_replicate as br           # noqa: E402
import court3d as c3                  # noqa: E402
from make_ball_audit import detect_events, load_impacts  # noqa: E402

SP = Path(__file__).resolve().parent

WINDOWS = {                            # rally -> (serve, end, npz)
    6:  (146.30, 151.90, SP / "r0006.npz"),
    7:  (164.70, 175.20, SP / "r0007.npz"),
    9:  (252.60, 282.53, SP / "r0009.npz"),
    10: (294.30, 318.45, SP / "r0010.npz"),
    17: (427.31, 442.25, SP / "r0017.npz"),   # click package 2026-09-02: first tap - 1.0 / last tap + 2.0 (audit-tool span); npz = CPU rtmpose-balanced here (gitignored; re-extract or take the Colab npz)
    18: (455.69, 461.91, SP / "r0018.npz"),   # click package 2026-09-02: first tap - 1.0 / last tap + 2.0 (audit-tool span); npz pending
    19: (477.29, 481.75, SP / "r0019.npz"),   # click package 2026-09-02: first tap - 1.0 / last tap + 2.0 (audit-tool span); npz pending
    20: (484.28, 506.64, SP / "r0020.npz"),   # click package 2026-09-02: first tap - 1.0 / last tap + 2.0 (audit-tool span); npz pending
    21: (515.99, 523.62, SP / "r0021.npz"),   # click package 2026-09-02: first tap - 1.0 / last tap + 2.0 (audit-tool span); npz pending
}


def build(rally):
    serve, end, npz = WINDOWS[rally]
    byf, t0 = bdec.load_candidates(rally)
    f_min = round((serve - 0.3 - t0) * bdec.FPS)
    f_max = round((end + 0.3 - t0) * bdec.FPS)
    byf = {f: c for f, c in byf.items() if f_min <= f <= f_max}
    oflags = bdec.out_of_court_flags(byf, bdec.court_hull())
    visited = bdec.decode(byf, None, oflags, None)
    refined = bdec.refine_arcs(visited, t0)
    _, timing_ref = bdec.timing_decode(byf, None, oflags, t0, [])
    turns = [e for e in detect_events(timing_ref)
             if serve - 0.3 <= e < end - 0.05]
    angs = br.turn_angles(timing_ref, turns)

    anchors_csv = SP / f"anchors_grade_r{rally}.csv"
    anchors = br.load_anchors(str(anchors_csv))
    zs = [float(r["excitement_z"] or 0)
          for r in csv.DictReader(open(anchors_csv))]

    X3, x2, _ = c3.load_landmarks()
    P = c3.dlt(X3, x2)
    floors = br.track_floor(str(npz), P)

    hum = br.human_side(rally, end)
    h_obs, h_bounds, h_evs = hum
    h_pa = br.bound_anchor_positions(h_bounds, anchors, floors)
    h_segs, h_cons = br.reconstruct(P, h_obs, h_bounds, h_evs, h_pa)

    imps, dead = load_impacts(rally=rally)
    cache = dict(rally=rally, serve=serve, end=end, t0=t0,
                 visited=visited, refined=refined, timing_ref=timing_ref,
                 turns=turns, angs=angs, anchors=anchors, zs=zs,
                 P=P, floors=floors, hum=hum, h_segs=h_segs,
                 h_cons=h_cons, imps=imps, dead=dead,
                 npz=str(npz))
    with open(SP / f"c3_cache_r{rally}.pkl", "wb") as f:
        pickle.dump(cache, f)
    print(f"r{rally}: cached — {len(visited)} visited, "
          f"{len(turns)} turns, {len(anchors)} anchors, "
          f"human {len(h_bounds)-1} segs "
          f"({sum(1 for s in h_segs if s and s['ok'])} ok)")


if __name__ == "__main__":
    # `python3 c3_lab.py 17 18` builds those; no args = the original four
    rallies = [int(a) for a in sys.argv[1:]] or [7, 6, 9, 10]
    for r in rallies:
        build(r)
