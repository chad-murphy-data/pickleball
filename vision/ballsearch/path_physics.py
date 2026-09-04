"""Track-level physics filter: throw away path that no ball could have made.

The audit (four labelled videos, owner-graded) found the largest single
error class was never the detector -- it was path that is physically
impossible: teleports onto a shoe, zigzags behind a player, hairpins,
apex double-turns, and a ball tracked before the serve exists.

Three rules, in order, all from the ball's own physics:

  BOUNDS    the rally's ball does not exist before the serve contact or
            after the ball dies.
  TELEPORT  a step implying a speed no pickleball reaches is not the ball
            moving, it is the tracker changing its mind about what the
            ball is.  Cut the path there.
  SPUR      a run left over after cutting that is too short to be flight
            is a momentary latch onto something else.  Drop it.

V_MAX is measured, not chosen: pooled over the nine owner-clicked human
paths (3,800 steps) the ball's apparent speed runs p50 355, p99 1395,
p99.9 2047 px/s.  2200 keeps essentially all real ball motion.
"""
import numpy as np

V_MAX = 2200.0      # px/s -- above this is not a ball (human p99.9 = 2047)
MIN_RUN = 4         # frames; a shorter surviving run is a latch, not flight
SERVE_LEAD = 0.10   # s of grace before the serve contact

# STALL: owner's rule -- "any path not toward the court needs
# justification", "no one ever hits the ball straight to the left".  The
# instantaneous direction version is null (lift 1.0x: junk rides players,
# and players move up-court too), but in COURT FEET over half a second it
# is decisive, because the ball always goes somewhere and a shoe does not.
# Measured on the nine human paths: a +/-0.25 s window whose court extent
# is under 4 ft holds 39 junk points and 4 good ones.
STALL_WIN = 0.25    # s each side
STALL_FT = 4.0      # court feet of extent below which nothing is flying

# RETRACE: the owner, watching the tracker shuttle in and out from behind
# a player -- "went behind the player, toward the player, behind her
# again, toward her again".  A ball reverses only when something reverses
# it: a paddle, the floor, the net.  Four reversals in half a second is
# four causes that do not exist.  Measured as the fraction of a window's
# points that come back within 2 ft of where the path already was at
# least 0.15 s earlier: >0.70 holds 20.9% of junk and 3.7% of good.
RETRACE_WIN = 0.30  # s each side
RETRACE_LAG = 0.15  # s -- how much earlier counts as "already been there"
RETRACE_FT = 2.0    # court feet
RETRACE_MAX = 0.70  # above this the path is retracing, not flying


def _speeds(pts):
    dt = np.diff(pts[:, 0])
    d = np.hypot(np.diff(pts[:, 1]), np.diff(pts[:, 2]))
    with np.errstate(divide="ignore", invalid="ignore"):
        sp = np.where(dt > 1e-6, d / np.maximum(dt, 1e-9), np.inf)
    return sp


def _court(P):
    """image -> court(z=0) homography from the 3x4 camera matrix."""
    return np.linalg.inv(np.asarray(P)[:, [0, 1, 3]])


def stalled(pts, P, win_s=STALL_WIN, dmin_ft=STALL_FT):
    """Points whose local window covers too little court to be a flight."""
    Hi = _court(P)
    pts = np.asarray(pts, float)
    cp = np.empty((len(pts), 2))
    for i, (_, x, y) in enumerate(pts[:, :3]):
        v = Hi @ np.array([x, y, 1.0])
        cp[i] = v[:2] / v[2]
    t = pts[:, 0]
    out = np.zeros(len(pts), bool)
    for i in range(len(pts)):
        a = np.searchsorted(t, t[i] - win_s)
        b = np.searchsorted(t, t[i] + win_s)
        if b - a < 4:
            continue
        w = cp[a:b]
        out[i] = float(np.hypot(*(w.max(0) - w.min(0)))) < dmin_ft
    return out


def retracing(pts, P, win_s=RETRACE_WIN, lag_s=RETRACE_LAG,
              near_ft=RETRACE_FT):
    """Fraction of each window that revisits where the path already was."""
    Hi = _court(P)
    pts = np.asarray(pts, float)
    cp = np.empty((len(pts), 2))
    for i, (_, x, y) in enumerate(pts[:, :3]):
        v = Hi @ np.array([x, y, 1.0])
        cp[i] = v[:2] / v[2]
    t = pts[:, 0]
    out = np.zeros(len(pts))
    for i in range(len(pts)):
        a = np.searchsorted(t, t[i] - win_s)
        b = np.searchsorted(t, t[i] + win_s)
        if b - a < 6:
            continue
        idx = np.arange(a, b)
        hit = tot = 0
        for k in idx:
            far = np.abs(t[idx] - t[k]) >= lag_s
            if not far.any():
                continue
            tot += 1
            d = np.hypot(*(cp[idx[far]] - cp[k]).T)
            if (d < near_ft).any():
                hit += 1
        out[i] = hit / max(tot, 1)
    return out


def clean(pts, serve=None, dead=None, P=None,
          v_max=V_MAX, min_run=MIN_RUN):
    """Return (kept_pts, mask, reason) for a (t,x,y) path.

    reason[i] is '' for kept points, else which rule dropped it.
    """
    pts = np.asarray(pts, float)
    n = len(pts)
    reason = np.array([""] * n, dtype=object)

    # BOUNDS
    if serve is not None:
        reason[pts[:, 0] < serve - SERVE_LEAD] = "pre-serve"
    if dead is not None:
        reason[pts[:, 0] > dead] = "post-dead"

    # STALL -- needs the camera matrix; cheap and near-free when present
    if P is not None:
        st = stalled(pts, P)
        reason[(reason == "") & st] = "stalled"

    if P is not None:
        rt = retracing(pts, P)
        reason[(reason == "") & (rt > RETRACE_MAX)] = "retrace"

    live = np.where(reason == "")[0]
    if len(live) < 2:
        return pts[live], reason == "", reason

    # TELEPORT: cut the live path wherever a step is impossible
    sub = pts[live]
    sp = _speeds(sub)
    cuts = np.where(sp > v_max)[0]          # step i joins live[i], live[i+1]
    edges = np.concatenate(([0], cuts + 1, [len(sub)]))
    runs = [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)
            if edges[k + 1] > edges[k]]

    # SPUR: a short surviving run is a latch onto something that is not
    # the ball.  Which side of a cut is the ball is decided by length.
    for a, b in runs:
        if b - a < min_run:
            reason[live[a:b]] = "spur"

    keep = reason == ""
    return pts[keep], keep, reason


def summarize(reason):
    out = {}
    for r in reason:
        out[r or "kept"] = out.get(r or "kept", 0) + 1
    return out


# ---------------------------------------------------------------------------
# OCCLUSION BRIDGING
#
# Owner's observation while auditing the labelled videos, which turned out
# to be the strongest feature in the whole path: the junk "goes to the
# paddle where it's hidden, goes behind, turns left, turns right, spins,
# and ends up back on path pretty quickly".
#
# Measured against the nine clicked human paths: 69.6% of junk points sit
# inside a player's box, vs 23.7% of good ones.  Nothing else came close
# (instantaneous speed 1.0x, parabola residual 1.4x, straightness 2.0x).
#
# So a player box is not a place the ball IS, it is a place the ball is
# INVISIBLE.  Inside one, don't believe the tracker -- bridge across it
# from the path either side.  Because the excursion leaves and rejoins the
# true path, the bridge's endpoints are both on the ball.
#
# A contact happens at a paddle, i.e. inside a box, so a bridge must never
# span one: known impact times split the span and the ball is carried to
# the impact from each side separately.
# ---------------------------------------------------------------------------

BOX_PAD = 0.0       # fraction of box size to grow by
BOX_TOL = 0.02      # s: pose sample must be this close in time


def in_player_box(pts, z, pad=BOX_PAD, tol=BOX_TOL):
    """Boolean mask: is this path point inside any tracked player's box?"""
    pts = np.asarray(pts, float)
    t, box = z["t"], z["box"]
    order = np.argsort(t)
    t, box = t[order], box[order]
    out = np.zeros(len(pts), bool)
    for i, (tt, x, y) in enumerate(pts[:, :3]):
        lo, hi = np.searchsorted(t, tt - tol), np.searchsorted(t, tt + tol)
        if hi <= lo:
            continue
        b = box[lo:hi]
        w = (b[:, 2] - b[:, 0]) * pad
        h = (b[:, 3] - b[:, 1]) * pad
        out[i] = bool(((x >= b[:, 0] - w) & (x <= b[:, 2] + w) &
                       (y >= b[:, 1] - h) & (y <= b[:, 3] + h)).any())
    return out


def bridge(pts, occluded, imps=(), max_span_s=0.60):
    """Replace occluded runs with straight-line interpolation in time.

    Returns (pts_out, bridged_mask).  Runs that touch either end of the
    path, that are longer than max_span_s, or that contain a known impact
    are left alone -- there is nothing trustworthy to bridge between, or
    the ball genuinely changed direction in there.
    """
    pts = np.asarray(pts, float).copy()
    occ = np.asarray(occluded, bool)
    n = len(pts)
    done = np.zeros(n, bool)
    i = 0
    while i < n:
        if not occ[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and occ[j + 1]:
            j += 1
        a, b = i - 1, j + 1
        i = j + 1
        if a < 0 or b >= n:
            continue
        t0, t1 = pts[a, 0], pts[b, 0]
        if t1 - t0 > max_span_s:
            continue
        if any(t0 < u < t1 for u in imps):
            continue
        f = (pts[a:b + 1, 0] - t0) / max(t1 - t0, 1e-9)
        pts[a:b + 1, 1] = pts[a, 1] + f * (pts[b, 1] - pts[a, 1])
        pts[a:b + 1, 2] = pts[a, 2] + f * (pts[b, 2] - pts[a, 2])
        done[a + 1:b] = True
    return pts, done
