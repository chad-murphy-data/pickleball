"""Ball speed from PLAYER GEOMETRY + CONTACT TIMING (owner idea, 2026-09-03):

    "If we know where player X is and player Y is, and we know timing,
     then we'd have the answer for speed (more or less)."

The 3D launch fit reads speed off the ball's own arc and is depth-dominated
(speed_lab_train.txt: a labeled SLOW shot at 84.9 mph, a labeled FAST one at
18.0).  This estimator never touches the ball's depth.  It uses the two
channels that ARE solved:

  * player court position -- feet on the z=0 plane, where the homography is
    exact (0.06 ft), so no depth degeneracy exists;
  * contact timing -- owner-labeled here (LEVEL C: the ceiling arm, what the
    measure could do given perfect contact times).

    avg speed over the flight = || xy(hitter_{i+1}) - xy(hitter_i) || / (t_{i+1} - t_i)

It is an AVERAGE over the flight, not a launch speed, and it charges the ball
a straight line, so a lob reads slow twice over.  That is stated, not hidden.

DECOMPOSITION IS THE POINT.  Three columns are printed side by side --
distance alone, 1/dt alone, and dist/dt -- because if 1/dt separates fast from
slow just as well, the geometry is adding UNITS, not information, and the
honest headline is "we can measure pace from timing".

Scope / discipline:
  * TRAIN read = r6 + r7 + r17 (owner ball clicks + manual contact taps).
    r9 / r10 print separately, flagged, as the evaluation/autopsy read.
  * Nothing is tuned.  No threshold is fit, no gate is run, no seal is
    touched.  The only free choice (which shot types count as fast) is the
    class split already written down in swing_explore_notes.md.
  * Hitter attribution uses the OWNER's clicked ball pixel at the contact
    frame -> nearest pose track in PIXEL space.  Pixel space is
    depth-independent, and this is ground truth on both sides, so the read
    grades the SPEED measure and not the tracker.

    python3 geom_speed.py
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import pathfirst as pf                                       # noqa: E402
from rally_stats import ground_point, players, foot_xy, FPS  # noqa: E402

LABELS = Path("/home/user/pickleball/data/vision/contact_labels_chicago0725.csv")
PATHS = Path("/home/user/pickleball/data/vision")
TRAIN = [6, 7, 17]
EVAL = [9, 10]
FAST = {"fast", "speed-up", "smash", "drive", "counter"}
SLOW = {"slow", "dink", "drop"}
FT_S_TO_MPH = 0.681818


def contacts(rally):
    """owner-labeled contacts for one rally, manual only, time-ordered."""
    out = []
    with open(LABELS) as f:
        for r in csv.DictReader(f):
            if r["division"] != "womens" or int(r["rally_in_game"]) != rally:
                continue
            if r["source"] == "prefill":
                continue
            t = r["t_refined_s"] or r["t_tap_s"]
            if not t:
                continue
            out.append(dict(t=float(t), type=r["shot_type"], name=r["hitter_name"],
                            i=int(r["shot_index"])))
    return sorted(out, key=lambda s: s["t"])


def clicks(rally):
    """owner ball clicks: t -> (x, y) pixel, any placed kind (V / S / I)."""
    p = PATHS / f"ball_path_r{rally}.csv"
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            if r["vis"] in ("N", "") or not r["x"]:
                continue
            out[float(r["t_s"])] = (float(r["x"]), float(r["y"]))
    return out



def hitter_track(ctx, pls, t, cl):
    """track id whose paddle/wrist is nearest the OWNER's clicked ball at t."""
    ts = np.array(sorted(cl))
    if not len(ts):
        return None
    j = int(np.argmin(np.abs(ts - t)))
    if abs(ts[j] - t) > 3.0 / FPS:
        return None
    bx, by = cl[ts[j]]
    best = None
    for tid, p in pls.items():
        pts = []
        m = np.abs(p["t"] - t) <= 0.05
        if m.any():
            pts.append(np.c_[p["px"][m], p["py"][m]])
        if len(p["wr"]):
            m = np.abs(p["wr"][:, 0] - t) <= 0.05
            if m.any():
                pts.append(p["wr"][m, 1:3])
        if not pts:
            continue
        q = np.vstack(pts)
        d = float(np.hypot(q[:, 0] - bx, q[:, 1] - by).min())
        if best is None or d < best[1]:
            best = (tid, d)
    return None if best is None else best[0]


def check_attribution(rally, pls, cs):
    """Free internal validators (ROADMAP Phase 2): side alternation is EXACT in
    this footage (0 violations / 229 contacts), and each labeled NAME must map
    to one track.  A rally that fails these has bad hitter attribution and its
    speeds are not trustworthy."""
    viol, prev = 0, None
    for s in cs:
        if s["type"] == "whiff" or s["tid"] is None:
            continue
        side = pls[s["tid"]]["label"].split("-")[0]
        if prev is not None and side == prev:
            viol += 1
        prev = side
    mp = defaultdict(Counter)
    for s in cs:
        if s["tid"] is not None:
            mp[s["name"]][pls[s["tid"]]["label"]] += 1
    pure = sum(v.most_common(1)[0][1] for v in mp.values())
    tot = sum(sum(v.values()) for v in mp.values())
    return viol, pure, tot


def rally_rows(rally, verbose=True):
    ctx = pf.context(rally)
    P = ctx["P"]
    z = np.load(ctx["c"]["npz"])
    pls = players(ctx)
    cs = contacts(rally)
    cl = clicks(rally)
    # resolve each contact to a track + a court position
    for s in cs:
        tid = hitter_track(ctx, pls, s["t"], cl)
        s["tid"] = tid
        s["xy"] = foot_xy(z, P, tid, s["t"]) if tid is not None else None
    viol, pure, tot = check_attribution(rally, pls, cs)
    if verbose:
        print(f"  r{rally}: {len(cs)} contacts, {len(pls)} pose tracks, "
              f"side-alternation violations {viol} (exact footage: expect 0), "
              f"name->track purity {pure}/{tot}")
    rows = []
    for a, b in zip(cs, cs[1:]):
        if a["type"] == "whiff" or b["type"] == "whiff":
            continue
        if a["xy"] is None or b["xy"] is None:
            continue
        dt = b["t"] - a["t"]
        if not (0.10 <= dt <= 3.0):
            continue
        dist = float(np.linalg.norm(b["xy"] - a["xy"]))
        rows.append(dict(rally=rally, i=a["i"], t=a["t"], type=a["type"],
                         name=a["name"], same=a["tid"] == b["tid"],
                         dist=dist, dt=dt, v=dist / dt))
    return rows


def auc(score, lab):
    """P(score of a fast > score of a slow), ties at 0.5."""
    x = np.asarray(score, float)
    y = np.asarray(lab, bool)
    if not y.any() or y.all():
        return float("nan")
    r = np.argsort(np.argsort(x)) + 1.0
    # average ranks for ties
    o = np.argsort(x)
    xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = np.mean(r[o[i:j + 1]])
        i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def report(name, rows, detail=False):
    keep = [r for r in rows if r["type"] in FAST or r["type"] in SLOW]
    lab = [r["type"] in FAST for r in keep]
    print(f"\n=== {name}: {len(rows)} flights with both endpoints placed, "
          f"{len(keep)} pace-labeled ({sum(lab)} fast / {len(lab) - sum(lab)} slow)")
    if len(keep) < 6:
        print("  too few to read")
        return
    for col, sign, unit in (("dist", +1, "ft"), ("dt", -1, "s"), ("v", +1, "ft/s")):
        sc = [sign * r[col] for r in keep]
        a = auc(sc, lab)
        f = [r[col] for r, L in zip(keep, lab) if L]
        s = [r[col] for r, L in zip(keep, lab) if not L]
        # permutation null on the same labels
        rng = np.random.default_rng(0)
        nul = np.array([auc(sc, rng.permutation(lab)) for _ in range(2000)])
        extra = f"  ({np.median(f) * FT_S_TO_MPH:5.1f} vs {np.median(s) * FT_S_TO_MPH:5.1f} mph)" if col == "v" else ""
        print(f"  {col:5s} AUC {a:.3f}   null {np.median(nul):.3f} "
              f"[{np.percentile(nul, 2.5):.3f}, {np.percentile(nul, 97.5):.3f}]   "
              f"median fast {np.median(f):6.2f} {unit} / slow {np.median(s):6.2f} {unit}{extra}")
    if not detail:
        return
    print("  -- flights, time-ordered")
    print(f"  {'r':>3} {'#':>3} {'t':>8} {'type':9s} {'dist ft':>8} {'dt s':>6} "
          f"{'ft/s':>7} {'mph':>6}  same-hitter")
    for r in rows:
        m = "F" if r["type"] in FAST else ("S" if r["type"] in SLOW else " ")
        print(f"  {r['rally']:3d} {r['i']:3d} {r['t']:8.2f} {r['type']:9s} {r['dist']:8.1f} "
              f"{r['dt']:6.2f} {r['v']:7.1f} {r['v'] * FT_S_TO_MPH:6.1f} {m}"
              f"{'  SAME' if r['same'] else ''}")


def main():
    print("attribution validators (a rally that fails these has untrustworthy speeds):")
    tr, ev = [], []
    for rl in TRAIN:
        tr += rally_rows(rl)
    for rl in EVAL:
        ev += rally_rows(rl)
    report("TRAIN r6+r7+r17", tr, detail=True)
    report("EVAL r9+r10 (owner clicks are spent evaluation; read only, tunes nothing)",
           ev, detail=True)
    report("POOLED all five rallies", tr + ev)


if __name__ == "__main__":
    main()
