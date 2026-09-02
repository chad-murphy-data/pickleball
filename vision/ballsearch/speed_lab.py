"""Which ball-speed measure separates fast from slow shots? (owner's
framing 2026-09-02: "turn it into 3D and then count pixels per second")

Per flight that STARTS at an attributed hit (rally_stats rules), four
candidate measures, joined to the owner's shot labels within 0.2 s:
  launch   |v0| of the fitted arc at its start (what rally_stats used; the
           extrapolated launch of a one-camera drag fit)
  path3d   median 3D speed along the sampled arc over the flight's
           interior (feet moved per second in court coordinates)
  image    median image speed of the projected track, px/s converted with
           the local px/ft scale at each sample (the owner's measure: no
           depth component at all)
  gap      seconds from this hit to the next attributed hit (or the end of
           the flight when there is none)
TRAIN = r6/r7 only (`python3 speed_lab.py 6 7`).  The rule is written down
in the docstring of rally_stats.speed_measure before r9/r10 are run once.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import court3d as c3                                        # noqa: E402
import events as evm                                        # noqa: E402
import pathfirst as pf                                      # noqa: E402
import rally_stats as rs                                    # noqa: E402

SP = Path(__file__).parent
FPS = 60.0


def measures(ctx, chosen, hits):
    P, t0 = ctx["P"], ctx["t0"]
    hit_ts = sorted(h["t"] for h in hits)
    rows = []
    for i, fl in enumerate(chosen):
        ta, tb = t0 + fl["fa"] / FPS, t0 + fl["fb"] / FPS
        who = [h for h in hits if abs(h["t"] - ta) <= 0.15]
        if not who:
            continue
        ts = np.arange(ta, tb + 1e-9, 1 / FPS)
        X = c3.arc_pos(fl["theta"], ts - fl["t_ref"])
        if len(ts) < 4:
            continue
        v3 = np.linalg.norm(np.diff(X, axis=0), axis=1) * FPS
        uv = c3.project(P, X)
        # local px/ft at each sample (mean of a 1-ft x and a 1-ft y offset)
        ux = c3.project(P, X + np.array([1.0, 0, 0]))
        uy = c3.project(P, X + np.array([0, 1.0, 0]))
        scale = (np.linalg.norm(ux - uv, axis=1) + np.linalg.norm(uy - uv, axis=1)) / 2
        vpx = np.linalg.norm(np.diff(uv, axis=0), axis=1) * FPS
        vimg = vpx / scale[:-1]
        k = slice(1, max(2, len(v3) - 1))         # interior
        nxt = [t for t in hit_ts if t > who[0]["t"] + 0.05]
        gap = (nxt[0] - who[0]["t"]) if nxt else (tb - who[0]["t"])
        rows.append(dict(i=i, t=ta, who=who[0]["near"][0],
                         launch=rs.flight_launch(fl)[0],
                         path3d=float(np.median(v3[k])),
                         image=float(np.median(vimg[k])),
                         gap=float(gap), dur=tb - ta))
    return rows


def labels(rally):
    out = []
    with open(rs.LABELS) as f:
        for r in csv.DictReader(f):
            if r["division"] == "womens" and int(r["rally_in_game"]) == rally:
                out.append((float(r["t_refined_s"] or r["t_tap_s"]), r["shot_type"], r["hitter_name"]))
    return out


def main():
    cell = json.loads(pf.TUNE_JSON.read_text())
    ev_cell = json.loads((SP / "events_tune_v3.json").read_text())
    for rally in [int(a) for a in sys.argv[1:]]:
        ctx = pf.context(rally)
        res = pf.run(ctx, cell["p_seed"], cell["s_min"], cell["gap"])
        chosen, t0 = res["chosen"], ctx["t0"]
        evs = evm.events(ctx, chosen, ev_cell["r_seam"], ev_cell["a_seam"], ev_cell["dt_pair"],
                         ev_cell["off"], d_pair=ev_cell["d_pair"])
        pls = rs.players(ctx)
        st = rs.rally_stats(ctx, chosen, evs, pls)
        rows = measures(ctx, chosen, st["hits"])
        lab = labels(rally)
        print(f"rally {rally}: {len(chosen)} flights, {len(st['hits'])} hits, {len(rows)} flights start at a hit")
        print(f"   {'t':8s} {'who':11s} {'label':8s} {'launch':>7s} {'path3d':>7s} {'image':>7s} {'gap':>5s} {'dur':>5s}")
        for r in rows:
            near = [l for l in lab if abs(l[0] - r["t"]) <= 0.2]
            typ = near[0][1] if near else "-"
            print(f"   {r['t']:8.2f} {r['who']:11s} {typ:8s} {r['launch']:7.1f} {r['path3d']:7.1f} "
                  f"{r['image']:7.1f} {r['gap']:5.2f} {r['dur']:5.2f}")


if __name__ == "__main__":
    main()
