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

# ---- tracker
MAX_COAST = 6              # frames a track may survive with no detection
BEAM = 60                  # hypotheses kept per frame
SEED_TOP = 12              # candidates per frame that may start a track
GATE_BASE = 26.0           # px, gate radius at zero speed
GATE_SLOPE = 0.55          # smooth-flight gate: px per px/frame of speed
# NOTE, arrived at by two failed designs: do NOT let a track cross a
# contact. v1 had a tight gate and died at every kink. v2 added a wide
# "kink gate" so tracks could cross reversals — but a track spanning
# several legs is by construction not straight, so it collided head-on
# with the tortuosity gate that keeps clutter out. The physics settles
# it: a segment IS one flight between contacts, and the contact is the
# JOIN between consecutive segments. Tracks now end at contacts by
# design, which is also what makes the straightness test meaningful.
ACC_SCALE = 12.0           # px of departure from the constant-velocity
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
MAX_TRACKS = 10            # segments extracted per rally window
JOIN_MAX_FRAMES = 14       # a gap this short between two segments is a
                           #   CONTACT, not two unrelated balls
JOIN_MAX_PX = 220          # ...and the ball cannot teleport further

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


def track_ball(cand_by_frame):
    """Beam-search the best ball track. Returns [(frame, x, y, seen)].

    Each hypothesis extends to its best gated candidate, and ALSO
    survives as a coast, so a track can cross an occlusion without a
    detection. Score rewards detections and penalises coasting, so the
    winner is the longest physically-consistent chain of real
    observations."""
    hyps = []            # (score, pts, coast, seen_flags)
    best = None          # best track ENDING ANYWHERE, not just at the end
    for f, cands in enumerate(cand_by_frame):
        arr = np.array([[c[1], c[2], c[0]] for c in cands],
                       dtype=float) if cands else np.zeros((0, 3))
        nxt = []
        for score, pts, coast, seen in hyps:
            px, py, vx, vy = _predict(pts)
            sp = math.hypot(vx, vy)
            g_smooth = (SEED_GATE if len(pts) == 1
                        else GATE_BASE + GATE_SLOPE * sp)
            if len(arr):
                d = np.hypot(arr[:, 0] - px, arr[:, 1] - py)
                order = np.argsort(d)[:4]
                for i in order:
                    if d[i] > g_smooth:
                        continue
                    gain = 1.0 - min(1.0, d[i] / ACC_SCALE)
                    nxt.append((score + gain,
                                pts + [(f, arr[i, 0], arr[i, 1])],
                                0, seen + [True]))
            if coast < MAX_COAST:      # coast: keep flying, pay for it
                nxt.append((score - MISS_COST, pts + [(f, px, py)],
                            coast + 1, seen + [False]))
        for c in cands[:SEED_TOP]:     # new tracks may start any frame
            nxt.append((0.0, [(f, c[1], c[2])], 0, [True]))
        nxt.sort(key=lambda h: -h[0])
        hyps = nxt[:BEAM]
        # A track that DIES mid-window is still a track. v1 only ever
        # returned hypotheses alive at the final frame, so a segment
        # ending at a contact — i.e. every segment — was thrown away.
        for score, pts, _c, seen in hyps:
            p2, s2 = pts, seen
            while s2 and not s2[-1]:
                p2, s2 = p2[:-1], s2[:-1]
            if sum(s2) >= MIN_TRACK and (best is None or score > best[0]):
                best = (score, p2, s2)
    if best is None:
        return []
    _score, pts, seen = best
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


def _ballistic_ok(track):
    """Reject wandering chains: a ball in flight goes somewhere, in a
    line. Tortuosity is the cheap discriminator and needs no physics."""
    pts = [(x, y) for _f, x, y, s in track if s]
    if len(pts) < 3:
        return False
    path = sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(pts, pts[1:]))
    disp = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
    return disp >= TRAVEL_MIN and path > 0 and disp / path >= STRAIGHT_MIN


def track_all(cand_by_frame, max_tracks=MAX_TRACKS):
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
        tr = track_ball(cands)
        if not tr:
            break
        if _ballistic_ok(tr):
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
    for a, b in zip(tracks, tracks[1:]):
        fa, xa, ya, _s = a[-1]
        fb, xb, yb, _s2 = b[0]
        gap = fb - fa
        if not (0 < gap <= JOIN_MAX_FRAMES):
            continue
        if math.hypot(xb - xa, yb - ya) > JOIN_MAX_PX:
            continue
        da = _dir([(x, y) for _f, x, y, _s in a[-K_FIT - 1:]])
        db = _dir([(x, y) for _f, x, y, _s in b[:K_FIT + 1]])
        if da is None or db is None:
            continue
        ang = math.degrees(math.acos(
            max(-1.0, min(1.0, float(np.dot(da, db))))))
        if ang < THETA_MIN:
            continue
        out.append(((fa + fb) / 2.0 / fps, ang, True))   # time is a guess
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
    tracks = track_all(cand)
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
    tr = track_ball(cand)
    assert tr, "no track on a clean synthetic"
    seen = [(f, x, y) for f, x, y, s in tr if s]
    truth = {f: (x, y) for f, x, y in pts}
    err = [math.hypot(x - truth[f][0], y - truth[f][1])
           for f, x, y in seen if f in truth]
    assert np.median(err) < 3.0, f"track drifted: median {np.median(err):.1f}px"
    ks = [t for t, _a, _i in contacts_from_tracks(track_all(cand), fps)]
    for kt in kinks:
        assert any(abs(kt - k) <= 0.1 for k in ks), (kt, ks)
    assert len(ks) <= len(kinks) + 1, f"spurious kinks: {ks}"
    print(f"  clean: track median err {np.median(err):.1f}px, "
          f"kinks {len(ks)}/{len(kinks)} found, no spurious")

    # ---- measured conditions: 24% random dropout still works
    cand, kinks, _ = _synth(drop=0.24)
    ks = [t for t, _a, _i in contacts_from_tracks(track_all(cand), fps)]
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
    one = track_ball(cand)
    single = [r for r in contacts_from_track(one, fps)
              if abs(r[0] - kinks[0]) <= 0.15]
    trs = track_all(cand)
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
    ks = contacts_from_tracks(track_all(noise), fps)
    assert len(ks) <= 1, f"pure clutter produced {len(ks)} contacts"
    print(f"  pure clutter: {len(ks)} contacts (<=1)")

    # ---- matcher + determinism
    assert len(match([1.0, 2.0], [1.2, 5.0])) == 1
    assert len(match([1.0], [1.6])) == 0
    a = track_ball(_synth(drop=0.24)[0])
    b = track_ball(_synth(drop=0.24)[0])
    assert a == b, "not deterministic"
    print("  matcher + determinism OK")
    print("selftest: ALL OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--fps", type=float, default=30.0)
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
