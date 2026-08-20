"""Classical contact detection: track the ball, find the kinks.

The end-to-end test the candidate probe pointed at (notes 2026-08-20).
No language model, no learned weights, no labels at inference:

    stream frames -> 3-frame motion-difference candidates
      -> velocity-gated beam tracker (coasts through occlusions)
      -> direction change in the track = CONTACT
      -> score against the 323 hand-labeled contacts

The metric is the SAME one the rest of the thread is quoted on: a
labeled contact counts as found if a detected contact lands within
+/-0.5 s. Reference points on that scale: decoded pose pipeline 45.7%,
VLM on frame grids 93%.

Why a beam tracker rather than nearest-neighbour linking: the probe
measured the ball present in only ~76% of sampled frames, and the
misses are STRUCTURED (w24 lost three consecutive samples), so any
linker that requires a detection every frame dies immediately. The beam
coasts across gaps on constant velocity and pays a score penalty.

Why kinks are only trusted between SEEN points: a coasted stretch is
linear BY CONSTRUCTION, so it cannot exhibit a direction change; a
contact hidden inside one is reported at the gap's midpoint and flagged
`inferred`, never quoted as a precise time. This is the discipline the
w01 cell-7 error bought (a trajectory-inferred position reported as an
observation).

    python3 vision/ball_track.py --selftest
    python3 vision/ball_track.py --video full_match.mp4.webm [--rallies 1,2]
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

from ball_candidates import candidates
from swing_probe import ffmpeg_bin, decode_window

LABELS = "contact_labels_chicago0725.csv"
SPLIT = "label_split.csv"
MATCH_TOL_S = 0.5          # same tolerance phase_grader uses

# PARAMETER SELECTION IS CLOSED (2026-08-20). Three configurations were
# measured on a 2-rally clip: 24/36, 23/36, 22/36 recall. At n=36 the
# standard error is 8 pp and the 95% CIs run 45-82% — those are ONE
# number, and picking between them on that sample is fitting noise. The
# values below are chosen on physical grounds (occlusion durations, ball
# speed, and one documented regularisation), not on the 36-contact
# score. The real measurement is 19 train rallies = 229 contacts, where
# se ~3 pp can actually separate hypotheses.

# ---- tracker
# PHYSICAL CONSTANTS, in seconds and px/second. Storing these in FRAMES
# and px/FRAME is what made the 60 fps source misbehave in three places
# at once: the segment cap halved (0.8 s instead of 1.6), the coast
# window halved, and the minimum ball speed DOUBLED in real terms, so
# genuine slow flights were being rejected. Anything that means a
# physical thing is stored physically and converted with fps.
COAST_S = 0.12             # how long a track may fly with no detection.
                           #   Sized as an OCCLUSION duration (a few
                           #   frames while the ball passes a body), not
                           #   as a free parameter.
MAX_SEG_S = 0.85           # a flight between contacts is short. This
                           #   cap is NOT purely physical: 1.6 s is the
                           #   honest upper bound on a flight, and
                           #   setting it there MEASURABLY HURT (rally 1
                           #   fell 60% -> 44% recall, segments 33 -> 11).
                           #   The cap also does REGULARISATION work — a
                           #   short cap forces the beam to commit to
                           #   short clean chains instead of growing long
                           #   wandering ones. 0.85 s still spans the
                           #   fitted gap distribution (fast 0.65 s,
                           #   slow 1.00 s) for all but the slowest.
JOIN_MAX_S = 0.35          # gap that still counts as one contact.
                           #   Segments END at contacts by design, so
                           #   this is again an occlusion duration, not
                           #   the length of a flight.
MIN_SPEED_PXS = 300.0      # px/second: a ball outruns a limb. At 60 fps
                           #   on this 1280-wide frame the court runs
                           #   ~40 px/ft, so a dink at ~14 ft/s is
                           #   ~560 px/s near court and roughly half
                           #   that when compressed at the far end.
BEAM = 60                  # hypotheses kept per frame
SEED_TOP = 12              # candidates per frame that may start a track
GATE_BASE = 10.0           # px, gate radius at zero speed. Was 26, which
                           #   is huge next to one frame of ball motion:
                           #   a SLOW hypothesis then had a gate big
                           #   enough that any nearby clutter looked
                           #   consistent, and the beam duly grew chains
                           #   crawling at 1.3-2.2 px/frame. A gate should
                           #   be sized by detection noise plus one
                           #   frame of gravity, not by the search
GATE_SLOPE = 0.55          # smooth-flight gate: px per px/frame of speed
# NOTE, arrived at by two failed designs: do NOT let a track cross a
# contact. v1 had a tight gate and died at every kink. v2 added a wide
# "kink gate" so tracks could cross reversals — but a track spanning
# several legs is by construction not straight, so it collided head-on
# with the tortuosity gate that keeps clutter out. The physics settles
# it: a segment IS one flight between contacts, and the contact is the
# JOIN between consecutive segments. Tracks now end at contacts by
# design, which is also what makes the straightness test meaningful.
ACC_SCALE = 6.0            # px of departure from the constant-velocity
                           #   prediction that costs a full point. Gain
                           #   must be scored in ABSOLUTE px, not as a
                           #   fraction of the gate: gate-relative gain
                           #   let a 35-frame wandering clutter chain
                           #   (straightness 0.16) outscore the real ball
                           #   and then CONSUME its candidates. Measured.
SEED_GATE = 62.0           # a 1-point track has no velocity yet
MISS_COST = 1.4            # score penalty per coasted frame
MIN_TRACK = 8              # a kept track needs this many SEEN points
STRAIGHT_MIN = 0.78        # |displacement| / path length. A flight
                           #   segment between contacts is near-straight
                           #   in image space (a lob still runs ~0.9);
                           #   a chain of clutter wanders. Without this,
                           #   multi-segment extraction manufactured 7
                           #   contacts out of pure noise (measured).
TRAVEL_MIN = 55.0          # px a real segment must actually cover
MAX_TRACKS = 45            # segments per rally: one per contact, plus
                           #   the clutter chains rejected on the way
STRAIGHT_AFTER = 6         # seen points before straightness is enforced
STRAIGHT_BEAM = 0.75       # in-beam kill. A HARD FLOOR GETS SATURATED:
                           #   at 0.60 the beam produced chains sitting
                           #   at exactly 0.60-0.74 — maximally wandering
                           #   while still legal — which then all failed
                           #   the 0.78 final gate. Keep the two close so
                           #   the search cannot farm the gap.

JOIN_SPEED_MULT = 1.6      # ...and the separation must be CONSISTENT WITH
JOIN_BASE_PX = 70.0        #   FLIGHT, not under a fixed px cap: over a
                           #   12-frame gap a ball at 30 px/frame really
                           #   does travel ~360 px, and a 220 px cap threw
                           #   away exactly those joins (measured).

# ---- kink detector
K_FIT = 3                  # points each side used for the direction fit
THETA_MIN = 32.0           # degrees of direction change to call a contact
NMS_S = 0.18               # suppress weaker kinks within this window


def _predict(pts):
    """(x, y, vx, vy) predicted for the next frame from a track's tail."""
    (f1, x1, y1) = pts[-1]
    if len(pts) == 1:
        return x1, y1, 0.0, 0.0
    (f0, x0, y0) = pts[-2]
    df = max(f1 - f0, 1)
    vx, vy = (x1 - x0) / df, (y1 - y0) / df
    return x1 + vx, y1 + vy, vx, vy


def track_ball(cand_by_frame, fps=30.0):
    """Beam-search one ball flight SEGMENT. Returns [(frame,x,y,seen)].

    Three things keep the beam honest, all of them learned from
    failures rather than designed up front:
      * gain is scored in ABSOLUTE px of departure from the
        constant-velocity prediction, not as a fraction of the gate;
      * a hypothesis that WANDERS is killed inside the beam once it has
        enough points, rather than being filtered afterwards;
      * segments are LENGTH-CAPPED, because score grows with length and
        on a 700-frame rally a smooth clutter chain (a swinging arm)
        otherwise outscores every real 20-frame flight and consumes its
        candidates first. That is precisely what produced 0/229 on real
        video while every synthetic test passed.
    """
    max_coast = max(1, int(round(COAST_S * fps)))
    max_seg = max(8, int(round(MAX_SEG_S * fps)))
    min_step = MIN_SPEED_PXS / fps
    hyps, best = [], None
    for f, cands in enumerate(cand_by_frame):
        arr = (np.array([[c[1], c[2], c[0]] for c in cands], dtype=float)
               if cands else np.zeros((0, 3)))
        nxt = []
        for h in hyps:
            if len(h["pts"]) >= max_seg:
                continue                       # finished; `best` saw it
            px, py, vx, vy = _predict(h["pts"])
            sp = math.hypot(vx, vy)
            gate = (SEED_GATE if len(h["pts"]) == 1
                    else GATE_BASE + GATE_SLOPE * sp)
            if len(arr):
                d = np.hypot(arr[:, 0] - px, arr[:, 1] - py)
                for i in np.argsort(d)[:4]:
                    if d[i] > gate:
                        continue
                    x, y = float(arr[i, 0]), float(arr[i, 1])
                    path = h["path"] + math.hypot(x - h["lsx"], y - h["lsy"])
                    nseen = h["nseen"] + 1
                    if nseen >= STRAIGHT_AFTER and path > 0:
                        disp = math.hypot(x - h["fsx"], y - h["fsy"])
                        if disp / path < STRAIGHT_BEAM:
                            continue           # wanders: kill it here
                        if path / (nseen - 1) < min_step:
                            continue           # crawls: not a ball
                    nxt.append({**h,
                                "score": h["score"] + 1.0
                                         - min(1.0, d[i] / ACC_SCALE),
                                "pts": h["pts"] + [(f, x, y)],
                                "seen": h["seen"] + [True], "coast": 0,
                                "path": path, "nseen": nseen,
                                "lsx": x, "lsy": y})
            if h["coast"] < max_coast:
                nxt.append({**h, "score": h["score"] - MISS_COST,
                            "pts": h["pts"] + [(f, px, py)],
                            "seen": h["seen"] + [False],
                            "coast": h["coast"] + 1})
        for c in cands[:SEED_TOP]:
            nxt.append({"score": 0.0, "pts": [(f, c[1], c[2])],
                        "seen": [True], "coast": 0, "path": 0.0,
                        "nseen": 1, "fsx": c[1], "fsy": c[2],
                        "lsx": c[1], "lsy": c[2]})
        nxt.sort(key=lambda h: -h["score"])
        hyps = nxt[:BEAM]
        for h in hyps:
            pts, seen = h["pts"], h["seen"]
            while seen and not seen[-1]:
                pts, seen = pts[:-1], seen[:-1]
            if sum(seen) >= MIN_TRACK and (best is None
                                           or h["score"] > best[0]):
                best = (h["score"], pts, seen)
    if best is None:
        return []
    _sc, pts, seen = best
    return [(f, x, y, s) for (f, x, y), s in zip(pts, seen)]


def _dir(pts):
    """Unit direction of a short run of points, by least squares."""
    a = np.array(pts, dtype=float)
    a = a - a.mean(axis=0)
    if len(a) < 2 or not a.any():
        return None
    u, _s, _v = np.linalg.svd(a.T @ a)
    d = u[:, 0]
    # orient along travel
    if np.dot(a[-1] - a[0], d) < 0:
        d = -d
    n = np.linalg.norm(d)
    return d / n if n else None


def contacts_from_track(track, fps):
    """Direction changes in the track. Returns [(t, degrees, inferred)].

    A kink is only measured across points that exist; when the change
    straddles a coasted gap its time is the gap midpoint and it is
    flagged inferred, because a coasted stretch is linear by
    construction and cannot show where inside it the ball turned."""
    if len(track) < 2 * K_FIT + 1:
        return []
    xy = [(x, y) for _f, x, y, _s in track]
    out = []
    for i in range(K_FIT, len(track) - K_FIT):
        d0 = _dir(xy[i - K_FIT:i + 1])
        d1 = _dir(xy[i:i + K_FIT + 1])
        if d0 is None or d1 is None:
            continue
        ang = math.degrees(math.acos(
            max(-1.0, min(1.0, float(np.dot(d0, d1))))))
        if ang < THETA_MIN:
            continue
        seen_win = [s for _f, _x, _y, s in track[i - K_FIT:i + K_FIT + 1]]
        inferred = not all(seen_win)
        out.append((track[i][0] / fps, ang, inferred))
    out.sort(key=lambda r: -r[1])          # non-max suppression by angle
    kept = []
    for t, ang, inf in out:
        if all(abs(t - k[0]) > NMS_S for k in kept):
            kept.append((t, ang, inf))
    return sorted(kept)


def _seg_vel(track, tail=True):
    """(f, x, y, vx, vy) at a segment's end (tail) or start (head),
    from its last/first few SEEN points."""
    pts = [(f, x, y) for f, x, y, s in track if s]
    if len(pts) < 2:
        return None
    run = pts[-K_FIT - 1:] if tail else pts[:K_FIT + 1]
    (f0, x0, y0), (f1, x1, y1) = run[0], run[-1]
    df = max(f1 - f0, 1)
    anchor = run[-1] if tail else run[0]
    return (anchor[0], anchor[1], anchor[2],
            (x1 - x0) / df, (y1 - y0) / df)


def _join_time(a, b):
    """Frame at which two flight segments' lines come closest — i.e.
    where the ball actually turned. The gap MIDPOINT is a poor stand-in
    when segments end early (measured 0.17 s off on a 12-frame gap);
    the intersection is what the geometry says."""
    va, vb = _seg_vel(a, True), _seg_vel(b, False)
    if va is None or vb is None:
        return None
    fa, xa, ya, ax, ay = va
    fb, xb, yb, bx, by = vb
    cx = (xa - ax * fa) - (xb - bx * fb)
    cy = (ya - ay * fa) - (yb - by * fb)
    dx, dy = ax - bx, ay - by
    den = dx * dx + dy * dy
    if den < 1e-9:
        return (fa + fb) / 2.0
    t = -(cx * dx + cy * dy) / den
    return min(max(t, fa), fb)          # never outside the gap


def _seg_speed(track):
    """Mean px/frame between a segment's SEEN points."""
    pts = [(x, y) for _f, x, y, s in track if s]
    if len(pts) < 2:
        return 0.0
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(pts, pts[1:])) / (len(pts) - 1)


def _ballistic_ok(track, fps=30.0):
    """Reject wandering chains: a ball in flight goes somewhere, in a
    line. Tortuosity is the cheap discriminator and needs no physics."""
    pts = [(x, y) for _f, x, y, s in track if s]
    if len(pts) < 3:
        return False
    path = sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(pts, pts[1:]))
    disp = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
    speed = path / max(len(pts) - 1, 1)
    return (disp >= TRAVEL_MIN and path > 0
            and disp / path >= STRAIGHT_MIN
            and speed >= MIN_SPEED_PXS / fps)


def track_all(cand_by_frame, fps=30.0, max_tracks=MAX_TRACKS):
    """Extract several non-overlapping flight SEGMENTS, best first.

    Why not one long track: a ball is most likely to be occluded
    exactly AT a contact, because a player is swinging at it there. A
    coast predicts the old direction, so a blackout spanning a reversal
    leaves the re-acquired ball ~2x speed x gap away — unbridgeable,
    and measured breaking the single-track version. Segments sidestep
    it: the contact becomes the JOIN between consecutive segments,
    which is what the physics actually shows (ball vanishes into the
    player, reappears going the other way)."""
    cands = [list(c) for c in cand_by_frame]
    out = []
    for _ in range(max_tracks):
        tr = track_ball(cands, fps)
        if not tr:
            break
        if _ballistic_ok(tr, fps):
            out.append(tr)
        for f, x, y, seen in tr:          # consume what this track used
            if seen and f < len(cands):
                cands[f] = [c for c in cands[f]
                            if math.hypot(c[1] - x, c[2] - y) > 6.0]
    return sorted(out, key=lambda t: t[0][0])


def _seg_vel(track, tail=True):
    """(f, x, y, vx, vy) at a segment's end (tail) or start (head),
    from its last/first few SEEN points."""
    pts = [(f, x, y) for f, x, y, s in track if s]
    if len(pts) < 2:
        return None
    run = pts[-K_FIT - 1:] if tail else pts[:K_FIT + 1]
    (f0, x0, y0), (f1, x1, y1) = run[0], run[-1]
    df = max(f1 - f0, 1)
    anchor = run[-1] if tail else run[0]
    return (anchor[0], anchor[1], anchor[2],
            (x1 - x0) / df, (y1 - y0) / df)


def _join_time(a, b):
    """Frame at which two flight segments' lines come closest — i.e.
    where the ball actually turned. The gap MIDPOINT is a poor stand-in
    when segments end early (measured 0.17 s off on a 12-frame gap);
    the intersection is what the geometry says."""
    va, vb = _seg_vel(a, True), _seg_vel(b, False)
    if va is None or vb is None:
        return None
    fa, xa, ya, ax, ay = va
    fb, xb, yb, bx, by = vb
    cx = (xa - ax * fa) - (xb - bx * fb)
    cy = (ya - ay * fa) - (yb - by * fb)
    dx, dy = ax - bx, ay - by
    den = dx * dx + dy * dy
    if den < 1e-9:
        return (fa + fb) / 2.0
    t = -(cx * dx + cy * dy) / den
    return min(max(t, fa), fb)          # never outside the gap


def _seg_speed(track):
    """Mean px/frame between a segment's SEEN points."""
    pts = [(x, y) for _f, x, y, s in track if s]
    if len(pts) < 2:
        return 0.0
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(pts, pts[1:])) / (len(pts) - 1)


def _ballistic_ok(track, fps=30.0):
    """Reject wandering chains: a ball in flight goes somewhere, in a
    line. Tortuosity is the cheap discriminator and needs no physics."""
    pts = [(x, y) for _f, x, y, s in track if s]
    if len(pts) < 3:
        return False
    path = sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(pts, pts[1:]))
    disp = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
    speed = path / max(len(pts) - 1, 1)
    return (disp >= TRAVEL_MIN and path > 0
            and disp / path >= STRAIGHT_MIN
            and speed >= MIN_SPEED_PXS / fps)


def track_all(cand_by_frame, fps=30.0, max_tracks=MAX_TRACKS):
    """Extract several non-overlapping flight SEGMENTS, best first.

    Why not one long track: a ball is most likely to be occluded
    exactly AT a contact, because a player is swinging at it there. A
    coast predicts the old direction, so a blackout spanning a reversal
    leaves the re-acquired ball ~2x speed x gap away — unbridgeable,
    and measured breaking the single-track version. Segments sidestep
    it: the contact becomes the JOIN between consecutive segments,
    which is what the physics actually shows (ball vanishes into the
    player, reappears going the other way)."""
    cands = [list(c) for c in cand_by_frame]
    out = []
    for _ in range(max_tracks):
        tr = track_ball(cands, fps)
        if not tr:
            break
        if _ballistic_ok(tr, fps):
            out.append(tr)
        for f, x, y, seen in tr:          # consume what this track used
            if seen and f < len(cands):
                cands[f] = [c for c in cands[f]
                            if math.hypot(c[1] - x, c[2] - y) > 6.0]
    return sorted(out, key=lambda t: t[0][0])


def contacts_from_tracks(tracks, fps):
    """Kinks WITHIN segments plus JOINS BETWEEN them."""
    out = []
    for tr in tracks:
        out += contacts_from_track(tr, fps)
    # Pair each segment with its best PHYSICAL successor, not merely
    # the next one in sorted order: a rally yields dozens of segments
    # (real flights plus surviving clutter), so "adjacent in the list"
    # pairs unrelated things. NOTE: an earlier attempt at this patch
    # silently no-opped, and the unchanged result was misread as
    # "pairing is not the bottleneck". Verify your edits.
    for ai, a in enumerate(tracks):
        fa, xa, ya, _s = a[-1]
        best = None
        for bi, b in enumerate(tracks):
            if bi == ai:
                continue
            fb, xb, yb, _s2 = b[0]
            gap = fb - fa
            if not (0 < gap <= JOIN_MAX_S * fps):
                continue
            sp = max(_seg_speed(a), _seg_speed(b))
            reach = JOIN_BASE_PX + JOIN_SPEED_MULT * sp * gap
            dist = math.hypot(xb - xa, yb - ya)
            if dist > reach:
                continue
            da = _dir([(x, y) for _f, x, y, _s in a[-K_FIT - 1:]])
            db = _dir([(x, y) for _f, x, y, _s in b[:K_FIT + 1]])
            if da is None or db is None:
                continue
            ang = math.degrees(math.acos(
                max(-1.0, min(1.0, float(np.dot(da, db))))))
            if ang < THETA_MIN:
                continue
            cost = dist / max(reach, 1.0) + gap / (JOIN_MAX_S * fps)
            if best is None or cost < best[0]:
                best = (cost, b, ang)
        if best is None:
            continue
        _c, b2, ang = best
        tj = _join_time(a, b2)
        if tj is None:
            tj = (fa + b2[0][0]) / 2.0
        out.append((tj / fps, ang, True))       # inferred: inside a gap
    out.sort(key=lambda r: -r[1])
    kept = []
    for t, ang, inf in out:
        if all(abs(t - k[0]) > NMS_S for k in kept):
            kept.append((t, ang, inf))
    return sorted(kept)


def rally_contacts(video, t0, t1, fps, width=1280):
    """Decode one rally window and return detected contact times (s)."""
    frames, cand = [], []
    for fr in decode_window(video, t0, t1 - t0, fps, width):
        frames.append(fr)
        if len(frames) == 3:
            cand.append(candidates(frames[0], frames[1], frames[2]))
            frames.pop(0)
    if not cand:
        return [], []
    tracks = track_all(cand, fps)
    ks = contacts_from_tracks(tracks, fps)
    track = max(tracks, key=len) if tracks else []
    # candidate index i was built from the middle of frames i..i+2
    return [(t0 + (t + 1.0 / fps), a, inf) for t, a, inf in ks], track


# ------------------------------------------------------------ scoring


def match(det, truth, tol=MATCH_TOL_S):
    cand = sorted((abs(d - t), i, j) for i, d in enumerate(det)
                  for j, t in enumerate(truth) if abs(d - t) <= tol)
    ui, uj, out = set(), set(), []
    for _x, i, j in cand:
        if i not in ui and j not in uj:
            ui.add(i); uj.add(j); out.append((det[i], truth[j]))
    return out


def load_truth(labels_path, split_path):
    train = {int(r["rally_cum"]) for r in csv.DictReader(open(split_path))
             if r["split"] == "train"}
    out = {}
    for r in csv.DictReader(open(labels_path)):
        c = int(r["rally_cum"])
        if c in train and r.get("contact", "1") == "1":
            out.setdefault(c, []).append(
                float(r["t_refined_s"] or r["t_tap_s"]))
    for v in out.values():
        v.sort()
    return out


# ----------------------------------------------------------- selftest


def _synth(fps=30, drop=0.24, gap=(None, 0), clutter=14, seed=3):
    """A piecewise-linear ball path with KNOWN kinks, then degraded to
    match what the probe actually measured: 24% random dropout, an
    optional STRUCTURED blackout, and clutter candidates every frame."""
    rng = np.random.default_rng(seed)
    legs = [((120.0, 500.0), (620.0, 190.0), 22),      # up the court
            ((620.0, 190.0), (240.0, 520.0), 18),      # returned
            ((240.0, 520.0), (700.0, 240.0), 20)]      # returned again
    pts, kinks, f = [], [], 0
    for (x0, y0), (x1, y1), n in legs:
        if pts:
            kinks.append(f - 1)
        for k in range(n):
            u = k / n
            pts.append((f, x0 + u * (x1 - x0), y0 + u * (y1 - y0)))
            f += 1
    cand = []
    g0, glen = gap
    for i, (fi, x, y) in enumerate(pts):
        cs = [(40.0 + rng.random() * 5, float(rng.integers(0, 800)),
               float(rng.integers(0, 600)), 6) for _ in range(clutter)]
        blacked = g0 is not None and g0 <= i < g0 + glen
        if not blacked and rng.random() > drop:
            cs.append((90.0, x + rng.normal(0, 1.1),
                       y + rng.normal(0, 1.1), 8))
        cs.sort(reverse=True)
        cand.append(cs)
    return cand, [k / fps for k in kinks], pts


def selftest():
    fps = 30
    # ---- clean-ish: recovers the path and both kinks
    cand, kinks, pts = _synth(drop=0.0, clutter=10)
    tr = track_ball(cand, fps)
    assert tr, "no track on a clean synthetic"
    seen = [(f, x, y) for f, x, y, s in tr if s]
    truth = {f: (x, y) for f, x, y in pts}
    err = [math.hypot(x - truth[f][0], y - truth[f][1])
           for f, x, y in seen if f in truth]
    assert np.median(err) < 3.0, f"track drifted: median {np.median(err):.1f}px"
    ks = [t for t, _a, _i in contacts_from_tracks(track_all(cand, fps), fps)]
    for kt in kinks:
        assert any(abs(kt - k) <= 0.1 for k in ks), (kt, ks)
    assert len(ks) <= len(kinks) + 1, f"spurious kinks: {ks}"
    print(f"  clean: track median err {np.median(err):.1f}px, "
          f"kinks {len(ks)}/{len(kinks)} found, no spurious")

    # ---- measured conditions: 24% random dropout still works
    cand, kinks, _ = _synth(drop=0.24)
    ks = [t for t, _a, _i in contacts_from_tracks(track_all(cand, fps), fps)]
    hit = sum(1 for kt in kinks if any(abs(kt - k) <= 0.15 for k in ks))
    assert hit == len(kinks), f"dropout broke kinks: {ks} vs {kinks}"
    assert len(ks) - hit <= 1, f"dropout spurious: {ks}"
    print(f"  24% dropout (the measured rate): kinks {hit}/{len(kinks)}, "
          f"spurious {len(ks) - hit}")

    # ---- STRUCTURED blackout across a kink: the tracker must survive,
    # and the kink it reports there must be FLAGGED inferred
    # A blackout across a kink BREAKS a single track (measured), so the
    # single-track path must fail here and the SEGMENT-JOIN path must
    # rescue it. Asserting both, so this cannot pass by coincidence the
    # way the v1 test did.
    cand, kinks, _ = _synth(drop=0.24, gap=(20, 5))
    one = track_ball(cand, fps)
    single = [r for r in contacts_from_track(one, fps)
              if abs(r[0] - kinks[0]) <= 0.15]
    trs = track_all(cand, fps)
    res = contacts_from_tracks(trs, fps)
    near = [r for r in res if abs(r[0] - kinks[0]) <= 0.15]
    assert len(trs) >= 2, f"segments not split: {len(trs)}"
    assert near, f"segment join lost the blacked-out kink: {res}"
    assert all(r[2] for r in near), "join not flagged inferred"
    hit = sum(1 for kt in kinks
              if any(abs(kt - r[0]) <= 0.15 for r in res))
    assert hit == len(kinks), f"blackout: {hit}/{len(kinks)}"
    print(f"  structured blackout: single track got "
          f"{len(single)}/1 there, segment joins got {hit}/{len(kinks)} "
          f"— flagged inferred")

    # ---- clutter alone must NOT manufacture contacts
    rng = np.random.default_rng(11)
    noise = [[(40.0, float(rng.integers(0, 800)), float(rng.integers(0, 600)),
               6) for _ in range(14)] for _ in range(60)]
    ks = contacts_from_tracks(track_all(noise, fps), fps)
    assert len(ks) <= 1, f"pure clutter produced {len(ks)} contacts"
    print(f"  pure clutter: {len(ks)} contacts (<=1)")

    # ---- matcher + determinism
    assert len(match([1.0, 2.0], [1.2, 5.0])) == 1
    assert len(match([1.0], [1.6])) == 0
    a = track_ball(_synth(drop=0.24)[0], fps)
    b = track_ball(_synth(drop=0.24)[0], fps)
    assert a == b, "not deterministic"
    print("  matcher + determinism OK")
    print("selftest: ALL OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--fps", type=float, default=0.0,
                    help="0 = use the video's NATIVE rate. Resampling a "
                         "60 fps source to 30 measurably costs recall "
                         "(ball found 5/9 vs 8/9 at verified positions) "
                         "AND adds clutter (101 vs 72 candidates/frame): "
                         "bigger inter-frame motion means more of the "
                         "scene differences, and the ball streaks.")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--pad", type=float, default=1.0,
                    help="seconds of context each side of a rally")
    ap.add_argument("--rallies", default="",
                    help="comma-separated rally_cum; default = all train")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.video:
        raise SystemExit("--video required (or --selftest)")

    if a.fps <= 0:
        import subprocess as _sp
        err = _sp.run([ffmpeg_bin(), "-i", str(a.video)],
                      capture_output=True).stderr.decode(errors="replace")
        import re as _re
        m = _re.search(r"(\d+(?:\.\d+)?)\s*fps", err)
        a.fps = float(m.group(1)) if m else 30.0
        print(f"native frame rate: {a.fps:g} fps")

    truth = load_truth(a.labels, a.split)
    want = ([int(x) for x in a.rallies.split(",")] if a.rallies
            else sorted(truth))
    tp = fp = fn = 0
    errs, inf_n = [], 0
    for cum in want:
        ts = truth.get(cum)
        if not ts:
            continue
        det, track = rally_contacts(a.video, ts[0] - a.pad, ts[-1] + a.pad,
                                    a.fps, a.width)
        dt = [d for d, _a, _i in det]
        inf_n += sum(1 for _d, _a, i in det if i)
        m = match(dt, ts)
        tp += len(m); fp += len(dt) - len(m); fn += len(ts) - len(m)
        errs += [abs(d - t) for d, t in m]
        nseen = sum(1 for *_x, s in track if s) if track else 0
        print(f"  rally {cum:>3}: truth {len(ts):>2}  detected "
              f"{len(dt):>2}  matched {len(m):>2}   "
              f"(track {nseen} seen pts)")

    n = tp + fn
    print(f"\nCONTACT RECALL  (+/-{MATCH_TOL_S}s)  {tp}/{n} = "
          f"{tp/max(n,1):.0%}")
    print(f"precision                     {tp}/{tp+fp} = "
          f"{tp/max(tp+fp,1):.0%}")
    if errs:
        errs.sort()
        print(f"median timing error           "
              f"{errs[len(errs)//2]:.2f}s")
    print(f"contacts reported inside a coasted gap (inferred): {inf_n}")
    print("\nreference on the same metric: decoded pose pipeline 45.7%, "
          "VLM 93%")


if __name__ == "__main__":
    main()
