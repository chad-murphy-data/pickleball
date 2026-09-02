"""Orbitable 3D court view of a rally from the path-first track (owner ask
2026-09-02: "can we create the 3D fit from this?").  Every path-first
flight IS a 3D arc (court3d.arc_pos: position + launch velocity + drag,
fit through the camera matrix), so this only samples them, adds the four
players' floor positions from the pose npz (ankle midpoint, else box
bottom, through the z=0 homography) and the attributed hits from
rally_stats, and writes court3d.write_viewer's self-contained HTML.

    python3 rally_3d.py <rally>        ->  court3d_r{N}.html  (space = play,
                                          drag = orbit, wheel = zoom)

Caveats carried over: depth (along the camera axis) is the weak axis of a
one-camera fit, so positions are trustworthy across the court and in
height, less so in how far down the court; gaps in the track are gaps in
the path (nothing is interpolated across a lost flight).  Viewer only;
nothing is tuned or written back.
"""
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
LANK, RANK = 15, 16


def ball_path(chosen, t0):
    path = []
    for fl in chosen:
        for f in range(fl["fa"], fl["fb"] + 1):
            t = t0 + f / FPS
            X = c3.arc_pos(fl["theta"], [t - fl["t_ref"]])[0]
            path.append([t, float(X[0]), float(X[1]), float(X[2])])
    return path


def player_tracks(ctx, pls):
    """{label: (t[], x[], y[])} at every pose sample (viewer thins to 10 fps)."""
    z = np.load(ctx["c"]["npz"])
    P = ctx["P"]
    out = {}
    for tid, p in pls.items():
        m = np.where(z["track"] == tid)[0]
        tt, kpt, kpc, box = z["t"][m], z["kpt"][m], z["kpc"][m], z["box"][m]
        rows = []
        for i in range(len(m)):
            if kpc[i, LANK] >= rs.KP_CONF and kpc[i, RANK] >= rs.KP_CONF:
                uv = (kpt[i, [LANK, RANK], 0].mean(), kpt[i, [LANK, RANK], 1].mean())
            else:
                uv = ((box[i, 0] + box[i, 2]) / 2, box[i, 3])
            xy = rs.ground_point(P, uv)
            rows.append((float(tt[i]), float(xy[0]), float(xy[1])))
        rows.sort()
        # light smoothing: 5-sample running median kills single-frame pose jumps
        arr = np.array(rows)
        if len(arr) >= 5:
            for k in (1, 2):
                arr[:, k] = np.array([np.median(arr[max(0, i - 2):i + 3, k]) for i in range(len(arr))])
        out[p["label"]] = (arr[:, 0], arr[:, 1], arr[:, 2])
    return out


def main():
    rally = int(sys.argv[1])
    cell = json.loads(pf.TUNE_JSON.read_text())
    assert not cell.get("dead")
    ev_cell = json.loads((SP / "events_tune_v3.json").read_text())
    assert not ev_cell.get("dead")
    ctx = pf.context(rally)
    res = pf.run(ctx, cell["p_seed"], cell["s_min"], cell["gap"])
    chosen, t0 = res["chosen"], ctx["t0"]
    ctx["_track"] = res["track"]
    evs = evm.events(ctx, chosen, ev_cell["r_seam"], ev_cell["a_seam"], ev_cell["dt_pair"],
                     ev_cell["off"], d_pair=ev_cell["d_pair"])
    pls = rs.players(ctx)
    st = rs.rally_stats(ctx, chosen, evs, pls)
    path = ball_path(chosen, t0)
    players = player_tracks(ctx, pls)
    impacts = [h["t"] for h in st["hits"]]
    out = SP / f"court3d_r{rally}.html"
    c3.write_viewer(path, impacts, out, players)
    # the viewer colours players by surname; ours are positions -> colour by side
    html = out.read_text().replace(
        'const TEAM = {', 'const TEAM = {"near-left":"#e05c5c","near-right":"#e05c5c",'
        '"far-left":"#5ca8e0","far-right":"#5ca8e0",', 1)
    out.write_text(html)
    # net-crossing check, free ground truth: the ball must clear the tape
    n_x, low = 0, 0
    for fl in chosen:
        ts = np.arange(t0 + fl["fa"] / FPS, t0 + fl["fb"] / FPS + 1e-9, 1 / 120.0)
        X = c3.arc_pos(fl["theta"], ts - fl["t_ref"])
        y = X[:, 1]
        for i in range(1, len(y)):
            if (y[i - 1] - c3.NET_Y) * (y[i] - c3.NET_Y) < 0:
                n_x += 1
                if X[i, 2] < c3.TAPE_FT:
                    low += 1
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB): {len(path)} path samples over "
          f"{len(chosen)} flights, {len(impacts)} hits, {len(players)} players; "
          f"net crossings {n_x}, of which under the 34-in tape {low}")


if __name__ == "__main__":
    main()
