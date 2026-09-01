"""Harvest the launch-state prior — the 'measured constants' of the
spaghetti model (owner-approved 2026-09-01: hurricane trails).

Sources: ONLY the click-driven human segment fits (h_segs) of the four
cached rallies — arcs fit on the owner's own ball clicks, ok=True and
rms-gated. Never tracker output (the fake 10-ft bounce had ok=True at
rms 6.05; click-driven + rms<3 is the vouching bar).

Per validated segment, harvest:
  - LAUNCH state at the segment start (a contact): speed |v0| ft/s,
    loft angle (deg above horizontal), contact height z0 ft, drag k.
    Canonical frame: launcher at the near side (mirror y when the
    launch point is on the far half) so vy>0 means 'toward opponent'.
  - BOUNCE physics from kind=='bounce' segs: e_z = -vz_out/vz_in,
    horizontal retention mu = |vxy_out|/|vxy_in|.

Writes SP/launch_prior.json: raw samples + percentile summaries.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
sys.path.insert(0, str(Path(__file__).parent))
import court3d as c3                    # noqa: E402
from claim_lab import load              # noqa: E402

SP = Path(__file__).parent
RMS_BAR = 3.0
NET_Y = 22.0

launches = []          # dicts: rally, t, speed, loft, z0, k, dy_sign
bounces = []           # dicts: rally, ts, e_z, mu, sp_in, sp_out
for rally in (6, 7, 9, 10):
    c = load(rally)
    h_segs = c["h_segs"]
    _, h_bounds, _ = c["hum"]
    for j, seg in enumerate(h_segs):
        if not seg or not seg.get("ok") or seg["rms"] >= RMS_BAR:
            continue
        t0, t1, th = seg["arcs"][0]
        p0 = c3.arc_pos(th, [0.0])[0]
        v0 = c3.arc_vel(th, 0.0)
        sp = float(np.linalg.norm(v0))
        hor = float(np.hypot(v0[0], v0[1]))
        loft = float(np.degrees(np.arctan2(v0[2], hor)))
        k = float(abs(th[6])) if len(th) >= 7 else None
        near = p0[1] > NET_Y
        vy_c = -v0[1] if near else v0[1]     # canonical: + = toward opp
        launches.append(dict(
            rally=rally, t=round(float(t0), 2), speed=round(sp, 1),
            loft=round(loft, 1), z0=round(float(p0[2]), 2),
            k=round(k, 3) if k is not None else None,
            fwd=round(float(vy_c), 1)))
        if seg["kind"] == "bounce":
            (a0, a1, thA), (b0, b1, thB) = seg["arcs"][0], seg["arcs"][1]
            v_in = c3.arc_vel(thA, b0 - a0)
            v_out = c3.arc_vel(thB, 0.0)
            if v_in[2] < -0.5 and v_out[2] > 0.05:
                ez = float(-v_out[2] / v_in[2])
                hin = float(np.hypot(v_in[0], v_in[1]))
                hout = float(np.hypot(v_out[0], v_out[1]))
                bounces.append(dict(
                    rally=rally, ts=round(float(b0), 2),
                    e_z=round(ez, 3),
                    mu=round(hout / max(hin, 1e-6), 3),
                    sp_in=round(float(np.linalg.norm(v_in)), 1),
                    sp_out=round(float(np.linalg.norm(v_out)), 1)))

sp_arr = np.array([r["speed"] for r in launches])
lo_arr = np.array([r["loft"] for r in launches])
z_arr = np.array([r["z0"] for r in launches])
k_arr = np.array([r["k"] for r in launches if r["k"] is not None])
fw_arr = np.array([r["fwd"] for r in launches])


def pct(a):
    return {p: round(float(np.percentile(a, p)), 2)
            for p in (2, 5, 10, 25, 50, 75, 90, 95, 98)}


print(f"{len(launches)} validated launches, {len(bounces)} bounces "
      f"(rms<{RMS_BAR}, click-driven fits only)")
print(f"speed ft/s : {pct(sp_arr)}")
print(f"loft deg   : {pct(lo_arr)}")
print(f"z0 ft      : {pct(z_arr)}")
print(f"drag k     : {pct(k_arr)}")
print(f"fwd vy     : {pct(fw_arr)}  (negative = launched AWAY from "
      f"opponent — should be rare)")
neg = [r for r in launches if r["fwd"] < 0]
print(f"  away-launches: {len(neg)}/{len(launches)}")
if bounces:
    ez = np.array([b["e_z"] for b in bounces])
    mu = np.array([b["mu"] for b in bounces])
    print(f"bounce e_z : {pct(ez)}")
    print(f"bounce mu  : {pct(mu)}")

out = dict(
    rms_bar=RMS_BAR, n_launch=len(launches), n_bounce=len(bounces),
    launches=launches, bounces=bounces,
    speed=pct(sp_arr), loft=pct(lo_arr), z0=pct(z_arr),
    k=pct(k_arr) if len(k_arr) else None, fwd=pct(fw_arr),
    e_z=pct(np.array([b["e_z"] for b in bounces])) if bounces else None,
    mu=pct(np.array([b["mu"] for b in bounces])) if bounces else None)
(SP / "launch_prior.json").write_text(json.dumps(out, indent=1))
print(f"wrote {SP / 'launch_prior.json'}")
