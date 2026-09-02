"""Rally stats off the path-first track + events layer (owner ask 2026-09-02:
"how many times each player hit the ball, who sped up first, where the
last shot was").  PROTOTYPE — the numbers below are read off the adopted
track (pathfirst_tune.json) and adopted events (events_tune_v3.json), plus
three rules written down here before any rally was looked at:

  hit attribution  an event is a HIT by player p if the ball at the event
                   is within D_HIT ft (local px/ft scale) of p's paddle
                   proxy (wrist + 0.5*(wrist-elbow), claim_lab.paddle_series)
                   or wrist within +-T_HIT s; nearest player wins; two hits
                   by the same player within DT_DUP s count once (the
                   double-hit the owner saw on the overlay). Otherwise the
                   event is a bounce / unassigned.
  speed-up         first flight after the 3rd shot whose launch speed is
                   >= V_FAST ft/s (a dink leaves the paddle at ~15-25 ft/s,
                   a drive at 40+); credited to that flight's hitter.
  last shot        the last flight: who started it, and where the ball was
                   last tracked, in court feet (x from the left sideline as
                   the camera sees it, y from the FAR baseline; net y=22).
                   DEPTH IS THE WEAK AXIS of a single-camera 3D fit.

Player identity is POSITION ONLY (near/far x left/right at the serve) from
the pose tracks; no name is read from anywhere.  `--grade` (r9/r10 only,
EVALUATION: never used to pick any of the three rules) compares against
the owner's shot labels, mapping names to tracks by majority vote over the
labeled contacts, and prints agreement.  Nothing is written back.

    python3 rally_stats.py <rally> [--grade]
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import court3d as c3                                        # noqa: E402
import events as evm                                        # noqa: E402
import pathfirst as pf                                      # noqa: E402
from claim_lab import paddle_series                         # noqa: E402

SP = Path(__file__).parent
LABELS = Path("/home/user/pickleball/data/vision/contact_labels_chicago0725.csv")
FPS = 60.0
D_HIT = 3.0        # ft: ball-to-paddle-proxy distance that makes an event a hit
T_HIT = 0.08       # s: pose sample tolerance around the event time
DT_DUP = 0.6       # s: same player twice inside this = one hit
V_FAST = 38.0      # ft/s launch speed (~26 mph) = a speed-up / drive
LANK, RANK, LWRI, RWRI = 15, 16, 9, 10
KP_CONF = 0.3


# ------------------------------------------------------------ players

def ground_point(P, uv):
    """image (u,v) -> court (x,y) on the z=0 plane."""
    H = P[:, [0, 1, 3]]
    q = np.linalg.solve(H, np.array([uv[0], uv[1], 1.0]))
    return q[:2] / q[2]


def players(ctx):
    """top-4 pose tracks: per-track wrists/paddle over time + a position label."""
    c, P = ctx["c"], ctx["P"]
    z = np.load(c["npz"])
    series = ctx["series"]
    out = {}
    for tid, (t, px, py, h) in series.items():
        m = np.where(z["track"] == tid)[0]
        tt = z["t"][m]
        kpt, kpc = z["kpt"][m], z["kpc"][m]
        wr = []
        for i in range(len(m)):
            for j in (LWRI, RWRI):
                if kpc[i, j] >= KP_CONF:
                    wr.append((tt[i], kpt[i, j, 0], kpt[i, j, 1]))
        # court position around the serve: ankle midpoint when both ankles
        # are seen, else the bottom-centre of the box; median over 4 s
        sel = (tt >= c["serve"] - 1.0) & (tt <= c["serve"] + 3.0)
        box = z["box"][m]
        feet = []
        for i in np.where(sel)[0]:
            if kpc[i, LANK] >= KP_CONF and kpc[i, RANK] >= KP_CONF:
                uv = (kpt[i, [LANK, RANK], 0].mean(), kpt[i, [LANK, RANK], 1].mean())
            else:
                uv = ((box[i, 0] + box[i, 2]) / 2, box[i, 3])
            feet.append(ground_point(P, uv))
        xy = np.median(feet, axis=0) if feet else np.array([np.nan, np.nan])
        out[tid] = dict(tid=tid, t=t, px=px, py=py, wr=np.asarray(wr), xy=xy)
    # labels: near = y > 22 (NEAR baseline is y=44); left/right by image x
    for side, ids in (("near", [k for k, v in out.items() if v["xy"][1] > 22]),
                      ("far", [k for k, v in out.items() if not v["xy"][1] > 22])):
        ids = sorted(ids, key=lambda k: out[k]["xy"][0])
        for pos, k in zip(("left", "right") if len(ids) == 2 else ["?"] * len(ids), ids):
            out[k]["label"] = f"{side}-{pos}"
    return out


def nearest_player(ctx, pls, t, uv, scale):
    """(label, distance ft) of the player whose paddle/wrist is closest to uv at t."""
    best = None
    for p in pls.values():
        cands = []
        m = np.abs(p["t"] - t) <= T_HIT
        if m.any():
            cands.append(np.c_[p["px"][m], p["py"][m]])
        if len(p["wr"]):
            m = np.abs(p["wr"][:, 0] - t) <= T_HIT
            if m.any():
                cands.append(p["wr"][m, 1:3])
        if not cands:
            continue
        pts = np.vstack(cands)
        d = float(np.hypot(pts[:, 0] - uv[0], pts[:, 1] - uv[1]).min()) / scale
        if best is None or d < best[1]:
            best = (p["label"], d)
    return best


# ------------------------------------------------------------- stats

def flight_launch(fl):
    v = c3.arc_vel(fl["theta"], 0.0)
    return float(np.linalg.norm(v)), float(np.hypot(v[0], v[1]))


def flight_for_time(chosen, t0, t):
    """index of the flight whose start is nearest AFTER-or-at t (the flight an event launches)."""
    best = None
    for i, fl in enumerate(chosen):
        ta = t0 + fl["fa"] / FPS
        if ta >= t - 0.02 and (best is None or ta < best[0]):
            best = (ta, i)
    return None if best is None else best[1]


def rally_stats(ctx, chosen, evs, pls):
    P, t0 = ctx["P"], ctx["t0"]
    rows = []
    for e in evs:
        if e["how"] == "arrive":
            fl = min(chosen, key=lambda g: abs(t0 + g["fb"] / FPS - e["t"]))
        else:
            i = flight_for_time(chosen, t0, e["t"])
            fl = chosen[i] if i is not None else chosen[-1]
        tau = e["t"] - fl["t_ref"]
        X = c3.arc_pos(fl["theta"], [tau])[0]
        uv = c3.project(P, X[None, :])[0]
        # local px/ft at that 3D point
        a = c3.project(P, (X + np.array([1.0, 0, 0]))[None, :])[0]
        scale = max(1e-6, float(np.linalg.norm(a - uv)))
        near = nearest_player(ctx, pls, e["t"], uv, scale)
        rows.append(dict(t=e["t"], how=e["how"], uv=uv, X=X, near=near,
                         hit=near is not None and near[1] <= D_HIT))
    # dedupe: same player twice within DT_DUP = one hit
    hits = []
    for r in rows:
        if not r["hit"]:
            continue
        if hits and hits[-1]["near"][0] == r["near"][0] and r["t"] - hits[-1]["t"] <= DT_DUP:
            continue
        hits.append(r)
    counts = Counter(h["near"][0] for h in hits)
    # speed-up: first flight after the 3rd hit with launch >= V_FAST
    speedup = None
    t3 = hits[2]["t"] if len(hits) >= 3 else t0
    for i, fl in enumerate(chosen):
        ta = t0 + fl["fa"] / FPS
        sp, _ = flight_launch(fl)
        who = [h for h in hits if abs(h["t"] - ta) <= 0.15]
        # a speed-up is a SHOT: the flight must start at an attributed hit
        # (a fragment that starts mid-air or after a bounce is not one)
        if ta > t3 and sp >= V_FAST and who:
            speedup = dict(t=ta, speed=sp, i=i, who=who[0]["near"][0])
            break
    last = chosen[-1]
    tb = t0 + last["fb"] / FPS
    Xe = c3.arc_pos(last["theta"], [tb - last["t_ref"]])[0]
    # the last SHOT is the last attributed hit; the ball's last position is
    # the end of the final flight (which may be a post-bounce fragment)
    hl = hits[-1] if hits else None
    last_shot = dict(t_start=hl["t"] if hl else t0 + last["fa"] / FPS, t_end=tb, X_end=Xe,
                     who=hl["near"][0] if hl else "?", speed=flight_launch(last)[0])
    return dict(rows=rows, hits=hits, counts=counts, speedup=speedup, last=last_shot)


def describe_xy(X):
    x, y, z = X
    side = "near" if y > 22 else "far"
    lr = "left" if x < 10 else "right"
    d_net = abs(y - 22)
    dx = x if x < 10 else 20 - x                   # distance inside the sideline
    dy = y if y < 22 else 44 - y                   # distance inside the baseline
    where = (f"{abs(dx):.1f} ft {'in from' if dx >= 0 else 'OUTSIDE'} the sideline"
             + (f", {abs(dy):.1f} ft beyond the baseline" if dy < 0 else ""))
    return (f"{side} side, {lr} half as the camera sees it, {d_net:.1f} ft from the net, "
            f"{where}, height {z:.1f} ft")


# ------------------------------------------------------------- grade

def truth_shots(rally):
    out = []
    with open(LABELS) as f:
        for r in csv.DictReader(f):
            if r["division"] == "womens" and int(r["rally_in_game"]) == rally:
                t = float(r["t_refined_s"] or r["t_tap_s"])
                out.append(dict(name=r["hitter_name"], t=t, type=r["shot_type"], i=int(r["shot_index"])))
    return out


def grade(ctx, pls, st, rally):
    tr = truth_shots(rally)
    real = [s for s in tr if s["type"] != "whiff"]
    print(f"\n== grade vs owner labels (EVALUATION only): {len(real)} contacts + {len(tr) - len(real)} whiff")
    # name -> track label by majority vote over labeled contacts
    P = ctx["P"]
    votes = defaultdict(Counter)
    for s in real:
        best = None
        for p in pls.values():
            m = np.abs(p["t"] - s["t"]) <= 0.1
            if not m.any():
                continue
            # distance from the player's paddle to the tracked ball at that time, else skip
            f = int(round((s["t"] - ctx["t0"]) * FPS))
            tk = ctx["_track"]
            g = min(tk, key=lambda q: abs(q - f)) if tk else None
            if g is None or abs(g - f) > 6:
                continue
            d = float(np.hypot(p["px"][m] - tk[g][0], p["py"][m] - tk[g][1]).min())
            if best is None or d < best[1]:
                best = (p["label"], d)
        if best:
            votes[s["name"]][best[0]] += 1
    name_of = {}
    for name, v in votes.items():
        name_of[v.most_common(1)[0][0]] = name
    print("  name map (majority vote of labeled contacts -> nearest track):")
    for name, v in votes.items():
        print(f"    {name:16s} {dict(v)}")
    truth_counts = Counter(s["name"] for s in real)
    print(f"  {'player':16s} {'label':11s} truth  ours")
    for lab in sorted(pls[k]["label"] for k in pls):
        name = name_of.get(lab, "?")
        print(f"  {name:16s} {lab:11s} {truth_counts.get(name, 0):5d} {st['counts'].get(lab, 0):5d}")
    fast = [s for s in tr if s["type"] in ("fast", "speed-up", "smash", "drive")]
    if fast:
        s = fast[0]
        print(f"  first fast shot (truth): #{s['i']} {s['name']} at {s['t']:.2f}  type {s['type']}")
    if st["speedup"]:
        su = st["speedup"]
        print(f"  first speed-up (ours):   {name_of.get(su['who'], su['who'])} at {su['t']:.2f}  "
              f"{su['speed']:.0f} ft/s")
    s = real[-1]
    print(f"  last contact (truth): #{s['i']} {s['name']} at {s['t']:.2f} ({s['type']}); "
          f"ours: {name_of.get(st['last']['who'], st['last']['who'])} at {st['last']['t_start']:.2f}")
    su = st["speedup"]
    if su:
        near_truth = [s for s in tr if abs(s["t"] - su["t"]) <= 0.2]
        print(f"  our speed-up flight starts at a labeled shot: "
              f"{near_truth[0]['type'] + ' by ' + near_truth[0]['name'] if near_truth else 'NO'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rally", type=int)
    ap.add_argument("--grade", action="store_true")
    a = ap.parse_args()
    cell = json.loads(pf.TUNE_JSON.read_text())
    assert not cell.get("dead")
    ev_cell = json.loads((SP / "events_tune_v3.json").read_text())
    assert not ev_cell.get("dead")
    ctx = pf.context(a.rally)
    import gapfill
    res = gapfill.product(ctx)                  # tracked flights + tagged gap fill (gapfill_gate.md v2)
    chosen, t0 = res["chosen"], ctx["t0"]
    ctx["_track"] = res["track"]
    evs = evm.events(ctx, chosen, ev_cell["r_seam"], ev_cell["a_seam"], ev_cell["dt_pair"],
                     ev_cell["off"], d_pair=ev_cell["d_pair"])
    pls = players(ctx)
    st = rally_stats(ctx, chosen, evs, pls)
    print(f"rally {a.rally}: {len(chosen)} flights, {len(evs)} events, "
          f"{len(st['hits'])} attributed hits ({sum(r['hit'] for r in st['rows'])} before de-dup)")
    print("players (position at the serve, court ft):")
    for k in sorted(pls, key=lambda k: pls[k]["label"]):
        p = pls[k]
        print(f"  {p['label']:11s} track {k}  x {p['xy'][0]:5.1f}  y {p['xy'][1]:5.1f}")
    print("hits per player:")
    for lab, n in sorted(st["counts"].items()):
        print(f"  {lab:11s} {n}")
    print("-- events: t | how | nearest player d(ft) | call")
    for r in st["rows"]:
        nr = f"{r['near'][0]:11s} {r['near'][1]:4.1f}" if r["near"] else f"{'-':11s}  -- "
        print(f"   {r['t']:8.3f} | {r['how']:7s} | {nr} | {'HIT' if r['hit'] else 'bounce/?'}")
    print("-- flights: start | launch ft/s | hitter")
    hit_at = {}
    for h in st["hits"]:
        hit_at[round(h["t"], 1)] = h["near"][0]
    for i, fl in enumerate(chosen):
        ta = t0 + fl["fa"] / FPS
        sp, _ = flight_launch(fl)
        who = [h["near"][0] for h in st["hits"] if abs(h["t"] - ta) <= 0.15]
        print(f"   {ta:8.3f} | {sp:5.1f}{' FAST' if sp >= V_FAST else '     '} | {who[0] if who else '?'}")
    su = st["speedup"]
    if su:
        print(f"first speed-up: {su['who']} at {su['t']:.2f} s, flight {su['i'] + 1}, launch {su['speed']:.0f} ft/s "
              f"({su['speed'] * 0.6818:.0f} mph)")
    else:
        print("first speed-up: none reached the threshold")
    L = st["last"]
    print(f"last shot: by {L['who']} at {L['t_start']:.2f} s; ball last tracked at "
          f"{L['t_end']:.2f} s: {describe_xy(L['X_end'])}")
    if a.grade:
        assert a.rally in (9, 10), "labels are for r9/r10 evaluation only"
        grade(ctx, pls, st, a.rally)


if __name__ == "__main__":
    main()
