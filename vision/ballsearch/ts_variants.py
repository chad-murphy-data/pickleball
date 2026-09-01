"""track_signals variants for the anchor-quality session.

Variant A: TORSO-RELATIVE wrist speed — subtract the shoulder-center
displacement before differencing, so whole-body translation (running,
repositioning) stops reading as arm speed. Falls back to raw speed
when shoulders are unmeasurable in either frame.

Variant B: A + ASYMMETRY discount — when both wrists are measurable,
scale the signal by how much the fast wrist beats the slow one
(running pumps both arms; a swing is one-armed). factor =
clip((fast-slow)/fast, ASYM_FLOOR, 1).
"""
import math
import sys

import numpy as np

sys.path.insert(0, "/home/user/pickleball/vision")
import hitter_chain as hc

LSHO, RSHO, LWRI, RWRI = hc.LSHO, hc.RSHO, hc.LWRI, hc.RWRI
LELB, RELB = hc.LELB, hc.RELB
CONF = hc.CONF
ASYM_FLOOR = 0.3


def track_signals_variant(z, tid, mode="A"):
    m = np.where(z["track"] == tid)[0]
    t, k, c = z["t"][m], z["kpt"][m], z["kpc"][m]
    box = z["box"][m]
    h = np.maximum(box[:, 3] - box[:, 1], 20.0)
    big = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]) > hc.JUNK_AREA
    n = len(m)
    speed = np.full(n, np.nan)
    wx = np.full(n, np.nan)
    wy = np.full(n, np.nan)
    pxa = np.full(n, np.nan)
    pya = np.full(n, np.nan)

    def shoulder_center(i):
        if c[i, LSHO] > CONF and c[i, RSHO] > CONF:
            return 0.5 * (k[i, LSHO] + k[i, RSHO])
        return None

    for i in range(n):
        if big[i]:
            continue
        best = None
        for w, e in ((LWRI, LELB), (RWRI, RELB)):
            if c[i, w] > CONF:
                if best is None or c[i, w] > best[0]:
                    px, py = k[i, w, 0], k[i, w, 1]
                    ex, ey = px, py
                    if c[i, e] > CONF:
                        vx, vy = px - k[i, e, 0], py - k[i, e, 1]
                        if math.hypot(vx, vy) > 5:
                            ex, ey = px + hc.EXT_LAM * vx, py + hc.EXT_LAM * vy
                    best = (c[i, w], px, py, ex, ey)
        if best:
            wx[i], wy[i] = best[1], best[2]
            pxa[i], pya[i] = best[3], best[4]
        if i and not big[i - 1]:
            dt = t[i] - t[i - 1]
            if 0 < dt < 0.1:
                sc0, sc1 = shoulder_center(i - 1), shoulder_center(i)
                per_wrist = {}
                for w in (LWRI, RWRI):
                    if c[i, w] > CONF and c[i - 1, w] > CONF:
                        if sc0 is not None and sc1 is not None:
                            d = (k[i, w] - sc1) - (k[i - 1, w] - sc0)
                        else:
                            d = k[i, w] - k[i - 1, w]
                        per_wrist[w] = np.linalg.norm(d) / h[i] / dt
                if per_wrist:
                    vals = sorted(per_wrist.values())
                    s = vals[-1]
                    if mode == "B" and len(vals) == 2 and vals[-1] > 1e-9:
                        fac = (vals[-1] - vals[0]) / vals[-1]
                        s *= max(ASYM_FLOOR, min(1.0, fac))
                    speed[i] = s
    reach = np.full(n, np.nan)
    for i in range(n):
        if big[i]:
            continue
        vals = []
        for w, s in ((LWRI, LSHO), (RWRI, RSHO)):
            if c[i, w] > CONF and c[i, s] > CONF:
                vals.append(np.linalg.norm(k[i, w] - k[i, s]) / h[i])
        if vals:
            reach[i] = max(vals)
    return t, speed, reach, wx, wy, pxa, pya


def patch(mode):
    """Monkey-patch hitter_chain.track_signals to the variant."""
    if mode == "base":
        hc.track_signals = _orig
    else:
        hc.track_signals = lambda z, tid: track_signals_variant(z, tid, mode)


_orig = hc.track_signals
