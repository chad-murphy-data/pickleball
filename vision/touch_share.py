"""Touch share and average hit position, per player, per match.

2026-08-20. The user asked directly: what % of balls did each player
hit, and where. NEITHER EXISTS YET. coverage_dominance.py says so in
its own docstring: "these measure SPACE TAKEN, not balls taken — true
poaching (hitting the partner's ball) needs ball data this stack does
not have." Every coverage metric shipped so far is about where a
player STANDS over time; nothing in that stack ever looks at a swing.

WHAT THIS DOES. Composes three ALREADY-BUILT, ALREADY-VALIDATED pieces
that were never plugged into each other:

  1. contact_ceiling.track_peaks() — per-TRACK candidate swing peaks
     from torso-relative wrist speed. This is Gate C's own instrument;
     Gate C killed it for EXACT sub-second contact-timestamp matching
     against a hand-labeled contact (40.7% vs an 85% bar). That is a
     harder bar than anything used here — nothing below tries to name
     a candidate as THE SAME EVENT as a specific labeled contact.
  2. swing_explore.decode_rally() — the alternation-prior sequence
     decoder that selects which candidates are real swings. Validated
     for shot COUNT: 161/162 on the Chicago holdout. It runs on
     candidates already tagged with a SIDE (0/1) and returns
     [(t, side, score, n_ghosts)] — side only, no player identity.
  3. coverage.py's identity chain (carry_names / anchor_identity) —
     resolves TRACK to PLAYER UUID per detection, already measured on
     THIS match (Gate A 96.1% / 100.0% / 46.7% per game). Crucially
     this resolves at the TRACK level, not just the SIDE level, which
     is what makes per-player attribution possible without a new
     model: at a decoded event's time, whichever of the (at most two)
     tracks on that side has ITS OWN candidate peak nearest that time
     is the attributed player.

So: track_peaks (per track) -> tag with side -> decode_rally (per
side, validated) -> at each decoded time, disambiguate the track via
each candidate's own peak proximity -> carry_names gives the uuid ->
coverage's homography gives court xy at that instant. No new pose
extraction (same npz coverage already reads — it just never looks past
the ankle keypoints). No new VLM cost.

WHAT IS GENUINELY NEW AND UNTESTED, stated plainly:
  (a) decode_rally is validated on CHICAGO ONLY (a single condensed
      MLP broadcast). coverage.py's own comment about THIS match says
      it "cuts between TWO elevated court angles plus close-ups
      mid-window" — a harder camera situation than Chicago's. Whether
      shot-count accuracy holds here is unmeasured.
  (b) The track-disambiguation step (which same-side track's peak is
      closer to a decoded event) has never been built or measured
      anywhere in this project. It is a much easier bar than Gate C
      (no specific labeled contact to hit within 0.5s, just "which of
      two known tracks swung"), but "easier" is not "measured."
  (c) A candidate's TIME determining a court POSITION assumes the
      player has not moved much in the timing-error window. True for
      most dinking exchanges; unverified for drives and transitions.

None of this is trustworthy as a published number yet. The house rule
applies: pre-register, run, spot-check a real sample before quoting a
percentage. AMBIGUOUS_FRAC below exists so that gate has a number to
look at, not just a vibe.

USAGE (needs a pose-dir/court/windows/lineup for the match, i.e. the
SAME coverage_pipeline.sh prerequisites — not producible from this
repo alone; see coverage_spec.md's ledger-fingerprint caveat):

    python vision/touch_share.py --pose-dir ... --court ... \\
        --windows ... --lineup ... [--cam ...] [--match-id ...]
    python vision/touch_share.py --selftest

Output: data/touch_share.csv (per player: n_touches, touch_share,
mean court x/y, n_ambiguous) + data/touch_events.csv (per event: the
raw ledger, so any threshold or ambiguity rule can be re-scored
without re-running the pipeline — same pattern as ball_track.py --dump).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# --- both are cross-branch dependencies, not yet on the same branch:
# coverage.py (identity + homography) lives on claude/court-coverage-
# model-8rg94l; contact_ceiling.py and swing_explore.py are on main via
# PR #63. A real run needs all three present locally regardless of git
# branch, since the user runs scripts from a flat folder, not a
# checkout — this only matters for which branch a fresh clone needs.
try:
    import coverage as C
except ImportError:
    C = None

from contact_ceiling import track_peaks
from swing_explore import decode_rally

ROOT = Path(__file__).resolve().parent.parent
DISAMBIG_TOL_S = 0.25   # a decoded event's time vs a track's own peak
FPS_DEFAULT = 10.0


def per_track_candidates(dets_by_track, fps):
    """track -> [(t, score)] via contact_ceiling.track_peaks, using
    each track's OWN box/kpt/kpc — exactly what Gate C's instrument
    consumes, just applied to every track instead of matched against
    a specific labeled contact."""
    out = {}
    for tr, ds in dets_by_track.items():
        ds = sorted(ds, key=lambda d: d.t)
        t = np.array([d.t for d in ds])
        box = np.stack([d.box for d in ds])
        kpt = np.stack([d.kpt for d in ds])
        kpc = np.stack([d.kpc for d in ds])
        out[tr] = track_peaks(t, box, kpt, kpc, fps)
    return out


def disambiguate(decoded, by_side):
    """Pure function, tested in isolation: GIVEN an already-decoded
    event list (whatever decode_rally selected -- this makes no
    assumption about how), attribute each event to a specific track by
    proximity of that track's own candidate peak. Returns
    [(t, side, score, track_or_None, gap_s)]. track is None when a
    second same-side track has a peak within DISAMBIG_TOL_S and is not
    clearly farther than the first -- reported, not guessed.

    Split out from decode_with_tracks so this step -- the one genuinely
    new, unvalidated piece in this file -- can be tested without also
    depending on decode_rally's cross-side alternation logic, which
    has nothing to chain against in the single-side scenario this
    ambiguity check is actually about."""
    out = []
    for t, side, score, n_gh in decoded:
        cands_here = by_side.get(side, [])
        near = sorted(((abs(ct - t), tr) for ct, tr, _sc in cands_here
                      if abs(ct - t) <= DISAMBIG_TOL_S))
        track = None
        gap = float("nan")
        if near:
            # ambiguous iff a second track's peak is within the SAME
            # tolerance and not clearly farther than the first
            if len(near) == 1 or near[1][0] - near[0][0] > 0.05:
                track = near[0][1]
            gap = near[0][0]
        out.append((t, side, score, track, gap))
    return out


def decode_with_tracks(per_track, track_side, s0):
    """Run the VALIDATED side-level decoder, then disambiguate (see
    disambiguate() above). Returns [(t, side, score, track_or_None,
    gap_s)]."""
    tagged = []
    for tr, cands in per_track.items():
        side = track_side.get(tr)
        if side is None:
            continue
        for t, sc in cands:
            tagged.append((t, side, sc))
    tagged.sort()
    decoded = decode_rally(tagged, s0)

    by_side = defaultdict(list)
    for tr, cands in per_track.items():
        side = track_side.get(tr)
        if side is not None:
            for t, sc in cands:
                by_side[side].append((t, tr, sc))
    return disambiguate(decoded, by_side)


def xy_at(dets_by_track, track, t, tol=0.5):
    """Nearest detection's court xy for (track, t); None if nothing
    within tol seconds (should not happen for an event the track's own
    candidate peak produced, but a rally boundary can clip it)."""
    ds = dets_by_track.get(track, [])
    if not ds:
        return None
    best = min(ds, key=lambda d: abs(d.t - t))
    return best.xy if abs(best.t - t) <= tol else None


def run_rally(dets, track_uuid, track_side, s0, fps=FPS_DEFAULT):
    """dets: this rally's coverage.Det list (all tracks).
    track_uuid: {track: player_uuid} for this rally (from carry_names).
    track_side: {track: 0|1}, from coverage's court-geometry side (more
    robust on multi-angle broadcasts than the raw npz height cluster --
    see coverage.load_rally's own comment on why it stopped trusting
    that field).
    Returns a list of dicts, one per attributed touch."""
    by_track = defaultdict(list)
    for d in dets:
        by_track[d.track].append(d)
    peaks = per_track_candidates(by_track, fps)
    events = decode_with_tracks(peaks, track_side, s0)
    rows = []
    n_ambig = 0
    for t, side, score, track, gap in events:
        if track is None:
            n_ambig += 1
            continue
        uuid = track_uuid.get(track)
        if uuid is None:
            continue
        xy = xy_at(by_track, track, t)
        if xy is None:
            continue
        rows.append(dict(t=t, side=side, score=score, track=track,
                         player_uuid=uuid, x=xy[0], y=xy[1],
                         disambig_gap_s=gap))
    return rows, n_ambig, len(events)


def summarize(all_rows, names=None):
    per = defaultdict(lambda: {"n": 0, "x": [], "y": []})
    for r in all_rows:
        p = per[r["player_uuid"]]
        p["n"] += 1
        p["x"].append(r["x"])
        p["y"].append(r["y"])
    total = sum(p["n"] for p in per.values())
    out = []
    for u, p in sorted(per.items(), key=lambda kv: -kv[1]["n"]):
        out.append(dict(
            player_uuid=u, player=(names or {}).get(u, ""),
            n_touches=p["n"],
            touch_share=p["n"] / total if total else 0.0,
            mean_x=float(np.mean(p["x"])) if p["x"] else float("nan"),
            mean_y=float(np.mean(p["y"])) if p["y"] else float("nan")))
    return out, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-dir")
    ap.add_argument("--court")
    ap.add_argument("--windows")
    ap.add_argument("--lineup")
    ap.add_argument("--cam", default="")
    ap.add_argument("--no-cam-gate", action="store_true")
    ap.add_argument("--match-id", default="")
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    ap.add_argument("--out", default=str(ROOT / "data/touch_share.csv"))
    ap.add_argument("--events-out",
                    default=str(ROOT / "data/touch_events.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if C is None:
        raise SystemExit(
            "coverage.py not importable -- it lives on "
            "claude/court-coverage-model-8rg94l and has not merged yet. "
            "Put it (and lineup.py, court.py) next to this file.")
    for req in ("pose_dir", "court", "windows", "lineup"):
        if not getattr(a, req):
            ap.error(f"--{req} required")

    got = {}
    C.run(a, collect=lambda rt: got.update(rt), write=False)

    all_rows, n_ambig_total, n_events_total = [], 0, 0
    # got: {(match_id, game): [(cum, rally_data, lin), ...]}
    # rally_data is player_uuid -> (ts, xy, end); it has no track ids,
    # so this walks pose_dir directly for the raw per-track Dets and
    # re-derives track->uuid the same way coverage.run() does, via
    # carry_names -- duplicating a few lines of coverage.py rather
    # than depending on it exposing an internal.
    print("touch_share needs coverage.py's per-detection track->uuid "
          "assignment, which coverage.run() does not currently expose "
          "outside its own loop. Wire-up point, not a research gap: "
          "either coverage.run() grows a second collect hook for "
          "(dets, assign) per rally, or this script re-runs "
          "carry_names() itself against the same pose-dir/lineup. "
          "Left as the next concrete step; everything below this line "
          "is the composition, verified against real function "
          "signatures and selftested on synthetic data.")
    raise SystemExit(1)


def selftest():
    """Two attempts preceded this one and both were WRONG in a way
    worth keeping on record, because the mistake is about test design,
    not about touch_share.py's actual code.

    Attempt 1 used a single side (no opponent tracks at all) -- decode_
    rally had nothing to alternate against and collapsed several real
    swings into one event. Attempt 2 fixed that but used a 6s rally
    against decode_rally's 8.0s "could be a leading ghost" head-start
    window, so EVERY event in the rally qualified as a plausible fresh
    start and out-competed the real chain; widening the rally then hit
    a SECOND tuned constant, the [0.45, 2.2]s per-hop pacing band,
    because the wider spacing chosen to dodge the first constant
    happened to sit just outside the second one.

    Both failures are about decode_rally's OWN calibration (tuned
    elsewhere on real Chicago dink cadence, validated there at
    161/162) -- not about anything in this file. Re-deriving that
    calibration well enough to fabricate a "nice" synthetic rally is
    not this test's job, and chasing it burned real time proving
    nothing about the code actually being tested here.

    So: this test does NOT assert what decode_rally selects. It hands
    decode_rally realistic-cadence input (real swings with a plausible
    noise floor under them, like actual footage has, not the isolated
    clean peaks the first two attempts used) and checks ONLY the thing
    this file is actually responsible for -- that EVERY event decode_
    rally does return gets attributed to the correct track, the
    correct player, and that player's own position, never their
    partner's, and never a non-swinging track."""
    from contact_ceiling import L_WRIST, R_WRIST, L_HIP, R_HIP, L_SHO, R_SHO

    class D:
        def __init__(self, t, track, xy, kpt, kpc, box):
            self.t, self.track, self.xy = t, track, xy
            self.kpt, self.kpc, self.box = kpt, kpc, box

    def make_track(track, xy0, swing_ts, jitter_ts=(), fps=10.0, dur=16.0):
        """swing_ts get a real, unambiguous wrist-speed spike.
        jitter_ts get a MUCH smaller one -- incidental motion, the
        noise floor decode_rally's ref percentile is calibrated
        against. Without any jitter, ref gets computed FROM the real
        swings themselves and collapses the margin that is supposed to
        separate them from noise -- attempt 2's data had this bug too,
        just not disclosed at the time."""
        ts = np.arange(0, dur, 1 / fps)
        dets = []
        for t in ts:
            kpt = np.zeros((17, 2))
            kpc = np.full(17, 0.9)
            kpt[L_HIP] = kpt[R_HIP] = [0, 30]
            kpt[L_SHO] = kpt[R_SHO] = [0, 10]
            swing = any(abs(t - st) < 0.1 for st in swing_ts)
            jitter = any(abs(t - st) < 0.1 for st in jitter_ts)
            wrist_y = -20 if swing else (2 if jitter else 5)
            kpt[L_WRIST] = kpt[R_WRIST] = [10 if swing else 0, wrist_y]
            box = np.array([xy0[0] - 5, 0, xy0[0] + 5, 40])
            dets.append(D(float(t), track, xy0, kpt, kpc, box))
        return dets

    # side 0: track 5 real swings, evenly paced within decode_rally's
    # documented [0.45, 2.2]s band; track 9 stands still except for a
    # little incidental motion (the noise floor). Side 1 mirrors it.
    tracks = {
        5:  make_track(5,  (3.0, 33.0),  [1.5, 3.0, 4.5, 6.0, 7.5],
                       jitter_ts=[2.3, 5.2]),
        9:  make_track(9,  (17.0, 33.0), [], jitter_ts=[2.8, 6.6]),
        21: make_track(21, (8.0, 5.0),   [2.25, 3.75, 5.25, 6.75],
                       jitter_ts=[1.1, 4.4]),
        22: make_track(22, (12.0, 5.0),  [], jitter_ts=[3.3, 7.0]),
    }
    track_side = {5: 0, 9: 0, 21: 1, 22: 1}
    track_uuid = {5: "uuid-A", 9: "uuid-B", 21: "uuid-C", 22: "uuid-D"}
    dets = [d for ds in tracks.values() for d in ds]
    by_track = defaultdict(list)
    for d in dets:
        by_track[d.track].append(d)

    peaks = per_track_candidates(by_track, fps=10.0)
    for tr in (5, 21):
        assert len(peaks[tr]) >= 1, f"track {tr} should show real peaks"
    print(f"  per-track candidates: {', '.join(f'{tr}->{len(peaks[tr])}' for tr in (5, 9, 21, 22))}")

    rows, n_ambig, n_ev = run_rally(dets, track_uuid, track_side, s0=0)
    assert n_ev >= 1, "decode_rally returned nothing for a real rally"

    uuids_seen = {r["player_uuid"] for r in rows}
    assert uuids_seen <= {"uuid-A", "uuid-C"}, (
        f"only the real swingers should ever get a touch, got "
        f"{uuids_seen} -- a non-swinging track's noise was attributed "
        f"as if it were a real shot")
    for r in rows:
        want_x = 3.0 if r["player_uuid"] == "uuid-A" else 8.0
        assert abs(r["x"] - want_x) < 1e-6, (
            f"{r['player_uuid']} attributed at x={r['x']}, "
            f"should be their own position {want_x}, not a partner's")
    print(f"  decode_rally returned {n_ev} events ({n_ambig} ambiguous); "
          f"every attributed one names a real swinger at THEIR OWN "
          f"position, never a non-swinger and never a partner's spot  OK")

    summ, total = summarize(rows)
    assert total == len(rows)
    print(f"  touch share on this draw: " +
          ", ".join(f"{s['player']or s['player_uuid']} {s['touch_share']:.0%}"
                    for s in summ))

    # ambiguity case, tested DIRECTLY against disambiguate() rather
    # than through the full pipeline: decode_rally is built for
    # cross-side alternation and has nothing to chain against in a
    # single-side scenario (the same trap attempt 1 hit), so routing
    # this specific check through it would test decode_rally's
    # behavior on an unrealistic input, not this file's own logic.
    # Hand the disambiguator a decoded event exactly as decode_rally
    # would shape one, and two same-side tracks with peaks 20ms apart.
    hand_decoded = [(2.00, 0, 0.70, 0)]
    by_side_amb = {0: [(2.00, 5, 0.70), (2.02, 9, 0.68)]}
    out_amb = disambiguate(hand_decoded, by_side_amb)
    assert len(out_amb) == 1 and out_amb[0][3] is None, (
        f"two same-side peaks 20ms apart should be left AMBIGUOUS, not "
        f"guessed: {out_amb}")
    print(f"  two same-side peaks 20ms apart -> correctly left "
          f"AMBIGUOUS (track=None), not guessed  OK")

    # and the companion case: a CLEARLY closer peak should NOT be
    # flagged ambiguous just because a second track has one nearby
    by_side_clear = {0: [(2.00, 5, 0.70), (2.20, 9, 0.68)]}
    out_clear = disambiguate(hand_decoded, by_side_clear)
    assert out_clear[0][3] == 5, (
        f"track 5's peak is 0.20s closer than track 9's -- should "
        f"resolve, not go ambiguous: {out_clear}")
    print(f"  a clearly-closer peak (0.20s vs 0.0s gap) resolves to "
          f"the right track instead of going ambiguous  OK")

    print("selftest: ALL OK. Synthetic only, and proves the "
          "ATTRIBUTION step -- not decode_rally's own selection, "
          "which is validated elsewhere on real footage. Real-footage "
          "validation of THIS composition is still pending a pose-dir "
          "for this match; do not trust a percentage from a real run "
          "without a spot-check first.")


if __name__ == "__main__":
    main()
