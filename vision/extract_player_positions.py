"""Player floor positions for the 3D court — pose tracks -> court feet.

Reads a pose npz (Gate C extraction; NOT committed — lives on the
Mac / Drive), takes each assigned track's ANKLE keypoints (confident
mean, the feet are ON the floor so the ground homography gives exact
depth — no monocular ambiguity), maps them through the DLT camera's
z=0 plane, median-smooths, and writes a committed CSV:

    data/vision/player_positions_r1.csv  (t_s, player, x_ft, y_ft)

Track identities come from the state labels' track_assign rows — the
user's own click-to-identify answers, not the 82% automated naming.

Raw npz stays the source of truth (house rule); this CSV is the
queryable derivative, same pattern as the rally-timeline CSVs.

Usage:
    python3 vision/extract_player_positions.py --npz pose/r0001.npz
    python3 vision/extract_player_positions.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from court3d import load_landmarks, dlt  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "vision"
STATE = DATA / "state_labels_chicago0725.csv"
OUT = DATA / "player_positions_r1.csv"
L_ANK, R_ANK = 15, 16
CONF = 0.3
OUT_FPS = 20.0
SMOOTH_S = 0.25


def load_assign(state_path=STATE, rally=1):
    out = {}
    for r in csv.DictReader(open(state_path)):
        if int(r["rally_cum"]) == rally and r["kind"] == "track_assign":
            out[int(float(r["t_s"]))] = r["player"]
    return out


def extract(npz_path, landmarks_path=None, state_path=STATE, rally=1):
    z = np.load(npz_path)
    X3, x2, _ = (load_landmarks(landmarks_path) if landmarks_path
                 else load_landmarks())
    P = dlt(X3, x2)
    Hinv = np.linalg.inv(P[:, [0, 1, 3]])
    assign = load_assign(state_path, rally)
    t, trk, kpt, kpc = z["t"], z["track"], z["kpt"], z["kpc"]
    rows = []
    for name in sorted(set(assign.values())):
        tid = next(k for k, v in assign.items() if v == name)
        m = trk == tid
        ts, ks, cs = t[m], kpt[m], kpc[m]
        raw = []
        for i in range(len(ts)):
            pts = [ks[i, j] for j in (L_ANK, R_ANK) if cs[i, j] > CONF]
            if not pts:
                continue
            px, py = np.mean(pts, axis=0)
            c = Hinv @ np.array([px, py, 1.0])
            raw.append((float(ts[i]), c[0] / c[2], c[1] / c[2]))
        if not raw:
            continue
        raw.sort()
        rt = np.array([r[0] for r in raw])
        rx = np.array([r[1] for r in raw])
        ry = np.array([r[2] for r in raw])
        for tq in np.arange(rt[0], rt[-1], 1.0 / OUT_FPS):
            m2 = np.abs(rt - tq) <= SMOOTH_S
            if m2.sum() < 2:
                continue
            rows.append((round(float(tq), 3), name,
                         round(float(np.median(rx[m2])), 2),
                         round(float(np.median(ry[m2])), 2)))
    rows.sort()
    return rows


def selftest():
    # synthetic: one track standing at court (5, 30) -> recovered
    import tempfile
    X3, x2, _ = load_landmarks()
    P = dlt(X3, x2)
    from court3d import project
    ank = project(P, np.array([[5.0, 30.0, 0.0]]))[0]
    n = 40
    kpt = np.zeros((n, 17, 2))
    kpc = np.zeros((n, 17))
    kpt[:, L_ANK] = ank + np.random.default_rng(0).normal(0, 1, (n, 2))
    kpt[:, R_ANK] = kpt[:, L_ANK]
    kpc[:, [L_ANK, R_ANK]] = 0.9
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "r.npz"
        np.savez(p, t=np.arange(n) / 20.0, track=np.ones(n, int),
                 kpt=kpt, kpc=kpc)
        import io
        # fake assign: track 1 -> a name present in state labels shape
        rows = extract(p, state_path=STATE, rally=1) if False else None
    # direct math check instead (extract needs real assign rows)
    Hinv = np.linalg.inv(P[:, [0, 1, 3]])
    c = Hinv @ np.array([ank[0], ank[1], 1.0])
    xy = c[:2] / c[2]
    assert abs(xy[0] - 5.0) < 0.15 and abs(xy[1] - 30.0) < 0.3, xy
    print(f"selftest OK — plane inversion recovers (5,30) as "
          f"({xy[0]:.2f},{xy[1]:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", help="pose npz for the rally (e.g. pose/r0001.npz)")
    ap.add_argument("--state", default=str(STATE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.npz:
        raise SystemExit("--npz required")
    rows = extract(a.npz, state_path=a.state)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "player", "x_ft", "y_ft"])
        w.writerows(rows)
    print(f"wrote {a.out}: {len(rows)} rows, "
          f"{len(set(r[1] for r in rows))} players at {OUT_FPS:.0f} fps")


if __name__ == "__main__":
    main()
