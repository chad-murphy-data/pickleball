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
height, less so in how far down the court.  Since gapfill_gate.md v2
(2026-09-02) the path carries the adopted TAGGED gap fill: frames that
exist only by extending the two arcs through an occlusion are flagged
and drawn dashed; lost flights stay gaps.  Since 2026-09-02 (owner ask,
this file's relift()) every flight is RE-LIFTED for display: refit to its
own graded pixels with the four players' floor positions as a soft
boundary (+4 ft reach) and a floor hinge, which resolves the depth
degeneracy that put arcs 30 ft behind the baseline.  The 2D track is
untouched (pixel rms of the re-lift is printed).  Viewer only; nothing is
tuned or written back.
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
import gapfill                                              # noqa: E402

SP = Path(__file__).parent
FPS = 60.0
SEG = """function seg(i,col,w){
  if(PATH[i][0]-PATH[i-1][0]>0.03) return;            // lost track: draw nothing
  const inf = PATH[i][4]||PATH[i-1][4];
  g.setLineDash(inf?[3,4]:[]);
  line(PATH[i-1].slice(1,4),PATH[i].slice(1,4),col,w);
  g.setLineDash([]);
}
function draw(){"""
LANK, RANK = 15, 16
# player-box re-lift (owner ask 2026-09-02, restoring the pseudo-boundary the
# court3d.py pass-2 fits carried): depth is the weak axis of a one-camera
# arc, so a flight can match its pixels while racing 30 ft behind the
# baseline.  The four players' floor positions bound where the ball can be:
# each flight is REFIT to its own graded pixels with a hinge that keeps the
# arc inside the players' bounding box (+PAD_FT reach) and above the floor.
# Viewer-only: the 2D track (the graded product) is the fit's data and is
# not written back; the pixel rms of the re-lift vs the graded track is
# printed so the cost in 2D is on record.
PAD_FT = 4.0            # reach beyond the four players' box, ft
FLOOR_FT = -0.3         # ball radius below z = 0 is the lowest physical centre
W_BOX = 1.0             # px per ft of excursion (the court-box prior uses 0.5)
BOX_TOL = 0.3           # s: a player's position counts if a pose sample is this close
BOX_STEP = 3            # sample the hinge every 3 frames over the flight's span


def player_box(players, t):
    """(xlo, xhi, ylo, yhi) of the players seen within BOX_TOL of t, or None."""
    xs, ys = [], []
    for tt, xx, yy in players.values():
        i = int(np.abs(tt - t).argmin())
        if abs(tt[i] - t) <= BOX_TOL:
            xs.append(xx[i])
            ys.append(yy[i])
    if len(xs) < 2:
        return None
    return min(xs) - PAD_FT, max(xs) + PAD_FT, min(ys) - PAD_FT, max(ys) + PAD_FT


def relift(ctx, chosen, inferred, players):
    """Refit every flight to its own projected pixels on its TRACKED frames,
    with the player-box + floor hinge over its full span (inferred extension
    included).  Returns (flights, rows); rows = (fa, fb, rms_px, exc_before,
    exc_after) where exc = worst excursion outside the box, ft."""
    P, t0 = ctx["P"], ctx["t0"]
    out, rows = [], []
    for fl in sorted(chosen, key=lambda f: f["fa"]):
        t_ref = fl["t_ref"]
        obs = [(t0 + f / FPS, *map(float, pf.arc_px(P, fl["theta"], t_ref, f, t0)), 1.0)
               for f in range(fl["fa"], fl["fb"] + 1) if f not in inferred]
        fs = list(range(fl["fa"], fl["fb"] + 1, BOX_STEP))
        taus = np.array([t0 + f / FPS - t_ref for f in fs])
        boxes = [player_box(players, t0 + f / FPS) for f in fs]
        lo = np.array([[b[0], b[2], FLOOR_FT] if b else [-np.inf, -np.inf, FLOOR_FT] for b in boxes])
        hi = np.array([[b[1], b[3], np.inf] if b else [np.inf, np.inf, np.inf] for b in boxes])

        def excursion(th):
            X = c3.arc_pos(th, taus)
            return np.maximum(lo - X, 0) + np.maximum(X - hi, 0)

        def extra(th):
            return W_BOX * excursion(th).ravel()

        if len(obs) < 4:
            out.append(dict(fl))
            rows.append((fl["fa"], fl["fb"], 0.0, float(excursion(fl["theta"]).max()), None))
            continue
        th, rms = c3.fit_arc(P, obs, t_ref, theta0=np.array(fl["theta"], float), extra=extra)
        nf = dict(fl)
        nf["theta"] = th
        out.append(nf)
        rows.append((fl["fa"], fl["fb"], rms, float(excursion(fl["theta"]).max()),
                     float(excursion(th).max())))
    return out, rows


def ball_path(chosen, t0, inferred=frozenset()):
    """[[t, x, y, z, inf]...]; inf = 1 on frames that exist only through the
    gap fill (gapfill_gate.md v2) — the viewer draws those dashed."""
    path = []
    for fl in chosen:
        for f in range(fl["fa"], fl["fb"] + 1):
            t = t0 + f / FPS
            X = c3.arc_pos(fl["theta"], [t - fl["t_ref"]])[0]
            path.append([t, float(X[0]), float(X[1]), float(X[2]), int(f in inferred)])
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
    res = gapfill.product(ctx)                  # tracked flights + tagged gap fill
    chosen, t0 = res["chosen"], ctx["t0"]
    ctx["_track"] = res["track"]
    evs = evm.events(ctx, chosen, ev_cell["r_seam"], ev_cell["a_seam"], ev_cell["dt_pair"],
                     ev_cell["off"], d_pair=ev_cell["d_pair"])
    pls = rs.players(ctx)
    st = rs.rally_stats(ctx, chosen, evs, pls)
    players = player_tracks(ctx, pls)
    lifted, lrows = relift(ctx, chosen, res["inferred"], players)
    path = ball_path(lifted, t0, res["inferred"])
    impacts = [h["t"] for h in st["hits"]]
    out = SP / f"court3d_r{rally}.html"
    c3.write_viewer(path, impacts, out, players)
    # the viewer colours players by surname; ours are positions -> colour by side
    html = out.read_text().replace(
        'const TEAM = {', 'const TEAM = {"near-left":"#e05c5c","near-right":"#e05c5c",'
        '"far-left":"#5ca8e0","far-right":"#5ca8e0",', 1)
    # path drawing: break at lost-track gaps (the stock viewer joins every
    # consecutive sample), dash the inferred frames, real-time playback
    html = html.replace(
        'for(let i=1;i<PATH.length;i++)\n    line(PATH[i-1].slice(1),PATH[i].slice(1),"#e8c44a",1);',
        'for(let i=1;i<PATH.length;i++) seg(i,"#e8c44a",1);')
    html = html.replace(
        'for(let i=1;i<PATH.length && PATH[i][0]<=tcur;i++){\n'
        '    line(PATH[i-1].slice(1),PATH[i].slice(1),"#ffd94a",2); last=PATH[i];}',
        'for(let i=1;i<PATH.length && PATH[i][0]<=tcur;i++){ seg(i,"#ffd94a",2); last=PATH[i];}\n'
        '  if(last && tcur-last[0]>0.12) last=null;')
    html = html.replace('function draw(){', SEG)
    html = html.replace('tcur += (T1-T0)/600;', 'tcur += 1/60;')
    html = html.replace('rally 1 in 3D', f'rally {rally} in 3D').replace(
        '<b>rally 1 — 3D</b>', f'<b>rally {rally} — 3D</b>')
    html = html.replace('<span id="tl"></span></div>',
        '<span id="tl"></span><br><span style="color:#999">solid = tracked ball · '
        'dashed = inferred through an occlusion by extending the two arcs (gapfill v2, '
        'right at 12 px about 2 in 3) · breaks = lost track · depth bounded by the four '
        'players\' positions (+4 ft)</span></div>')
    assert 'function seg' in html and 'seg(i,"#ffd94a",2)' in html
    out.write_text(html)
    # net-crossing check, free ground truth: the ball must clear the tape
    n_x, low = 0, 0
    for fl in lifted:
        ts = np.arange(t0 + fl["fa"] / FPS, t0 + fl["fb"] / FPS + 1e-9, 1 / 120.0)
        X = c3.arc_pos(fl["theta"], ts - fl["t_ref"])
        y = X[:, 1]
        for i in range(1, len(y)):
            if (y[i - 1] - c3.NET_Y) * (y[i] - c3.NET_Y) < 0:
                n_x += 1
                if X[i, 2] < c3.TAPE_FT:
                    low += 1
    rms = [r[2] for r in lrows]
    print(f"player-box re-lift: {len(lrows)} flights, pixel rms vs graded track "
          f"median {np.median(rms):.2f} max {max(rms):.2f} px; worst excursion outside the "
          f"players' box+{PAD_FT:g} ft: before {max(r[3] for r in lrows):.1f} ft, "
          f"after {max(r[4] for r in lrows if r[4] is not None):.1f} ft")
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB): {len(path)} path samples "
          f"({sum(p[4] for p in path)} inferred) over {len(chosen)} flights, {len(impacts)} hits, "
          f"{len(players)} players; "
          f"net crossings {n_x}, of which under the 34-in tape {low}")


if __name__ == "__main__":
    main()
