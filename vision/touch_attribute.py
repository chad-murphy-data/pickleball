"""Touch attribution WITHOUT an appearance model — and without the API.

WHY THIS REPLACES THE VLM PILOT'S CORE (2026-08-21). vlm_join_pilot
asked the model "which named player hit this?" and measured identity
44%, side 70%. That is the wall lineup.py's docstring already named:
"The first attribution attempt tried to name players from appearance
and calibrated at 57% against a ~60% ceiling, i.e. nothing. It did not
need to: side-out doubles is a state machine, and the log hands us its
inputs." We rebuilt the anti-pattern. This module is the intended
architecture, assembled from parts that are each already measured:

  DECODER      which track swung, and when        (counts 161/162)
  TRACKER      that track's image position        (pose, solved)
  ALTERNATION  sides strictly alternate           (0 violations/229)
  REFEREE LOG  server + receiver, exactly         (free, per rally)
  LINEUP       all four players' court halves     (99.25%/45,689)

The only camera question left is SPATIAL — near/far end, and which of
the two players on a side is further left — and neither needs the
players' faces, kit or names. Identity comes from the log.

THE THREE PIECES THE USER ASKED FOR, and where they live:
  1 position-not-identity  quadrant_of_event + assign_names
  2 alternation            alternation_fix (hard constraint, phase from 3)
  3 serve anchor           serve_anchor + vote_orientation

THE RELATIVE TRICK. Absolute court geometry is not needed to answer
"which of the two teammates is on the left": each side holds exactly
two players, so comparing their box centres at the contact instant
settles it. That is robust to zoom, pan and homography drift in a way
an absolute x-threshold is not.

THE TWO FREE BITS. lineup.py: "Two bits per game, and only two: which
team is at the near end, and whether a team's 'right' is image-right.
Everything else above is determined. Both bits are massively
over-identified by ~30 rallies of agreement." vote_orientation solves
them by voting SERVES — where the hitter's identity is known from the
log — so orientation is measured, never assumed.

Costs nothing to run: no API, no video decode, only the pose npz files
and the committed logs.

    python3 touch_attribute.py --selftest          (no deps, no files)
    python3 touch_attribute.py                     (pose_rtm/ + labels)
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

LABELS = "contact_labels_chicago0725.csv"
WINDOWS_V4 = "rally_windows_chicago0725_v4.csv"
SPLIT = "label_split.csv"
POSE_DIR = "pose_rtm"
EXCLUDE = {9, 10}          # contact_gate.md span anomaly
RIGHT, LEFT = "R", "L"


# ------------------------------------------------------- orientation
# Pure geometry+convention, kept in one place and SOLVED FROM DATA.
#
# A player's half (R/L) is from their own perspective facing the net,
# so the two ends mirror each other in the image: the near team faces
# away from camera and their R falls image-right, the far team faces
# the camera and their R falls image-left. Rather than trust that
# reasoning (broadcasts mirror, cameras move), `flip` carries it as a
# free bit and the serve votes decide.

def predict_image_quadrant(team, half, near_team, flip):
    """(team, half) -> (is_near, is_image_right) under a hypothesis."""
    is_near = (team == near_team)
    img_right = (half == RIGHT) if is_near else (half == LEFT)
    if flip:
        img_right = not img_right
    return is_near, img_right


def invert_image_quadrant(is_near, is_image_right, near_team, flip):
    """(is_near, is_image_right) -> (team, half). Exact inverse of
    predict_image_quadrant, which the selftest pins as a round trip."""
    team = near_team if is_near else other_team(near_team)
    if flip:
        is_image_right = not is_image_right
    half = (RIGHT if is_image_right else LEFT) if is_near else \
           (LEFT if is_image_right else RIGHT)
    return team, half


def other_team(t):
    return "B" if t == "A" else "A"


def vote_orientation(samples):
    """samples = [(team, half, is_near, is_image_right)] from SERVES,
    where the hitter is known from the log. Returns
    (near_team, flip, agreement, n). Four hypotheses, majority wins;
    ~30 serves over-identify two bits, so a healthy match should come
    back near 1.0 and a low score means the observations are noise."""
    best = None
    for near_team in ("A", "B"):
        for flip in (0, 1):
            ok = sum(
                predict_image_quadrant(tm, hf, near_team, flip)
                == (near, right)
                for tm, hf, near, right in samples)
            cand = (ok, near_team, flip)
            if best is None or cand > best:
                best = cand
    ok, near_team, flip = best
    n = len(samples)
    return near_team, flip, (ok / n if n else 0.0), n


# ------------------------------------------------------- alternation
def alternation_fix(sides, first_side):
    """Strict alternation is EXACT in this archive (0 violations over
    229 contacts), so the side sequence carries no free choices at all
    once its phase is known: contact k is on first_side ^ (k & 1).

    The serve anchor supplies the phase, so this does not vote or
    smooth — it overwrites. Returns (fixed, n_changed); n_changed is
    the honest count of how often the tracker's side disagreed, and is
    reported rather than hidden."""
    fixed = [(first_side ^ (k & 1)) for k in range(len(sides))]
    changed = sum(1 for a, b in zip(sides, fixed) if a != b)
    return fixed, changed


def serve_anchor(rec, team_of_name):
    """The log's server is contact 1, exactly. Returns (name, team)."""
    return rec["server_name"], team_of_name[rec["server_name"]]


# ------------------------------------------------------- attribution
def label_tracks_at_serve(rd, t_serve, rec, near_team, flip, name_of):
    """{track_id: name} for all four players, fixed AT THE SERVE.

    THIS IS WHAT THE TRACKER IS FOR (lineup.py: "four blobs, four known
    labels, no appearance model, no hand labelling"). The lineup halves
    are only defined at the serve — mid-rally the players poach, switch
    and cross — so left/right may be read ONCE, at the serve instant,
    and thereafter identity rides the track id. Re-deriving left/right
    per contact (the first version of this module) silently swaps two
    partners for the rest of any rally containing a switch, which is
    common and is exactly the case touch share cares about.

    Returns ({tid: name}, ok) where ok is False if a side did not show
    exactly two tracks — the geometry was unreadable and the rally
    should be skipped rather than guessed at."""
    labels = {}
    for side in (0, 1):
        present = []
        for tid, ser in rd["tracks"].items():
            if ser["side"] != side:
                continue
            c = box_at(ser, t_serve)
            if c is None or abs(_nearest_dt(ser, t_serve)) > 0.5:
                continue
            present.append((c[0], tid))
        if len(present) != 2:
            return labels, False
        present.sort()                      # by cx: left first
        for (_cx, tid), img_right in zip(present, (False, True)):
            team, half = invert_image_quadrant(side == 0, img_right,
                                               near_team, flip)
            uuid = rec.get(f"team_{team}_{half}", "")
            nm = name_of.get(uuid.lower())
            if nm is None:
                return labels, False
            labels[tid] = nm
    return labels, True


def assign_names(events, rec, near_team, flip):
    """events = [(t, is_near, is_image_right)] -> [name].

    rec is the rally's lineup record, giving every (team, half) a name.
    No appearance model touches this: the camera answers only WHERE,
    the referee log answers WHO."""
    out = []
    for _t, is_near, img_right in events:
        team, half = invert_image_quadrant(is_near, img_right,
                                           near_team, flip)
        out.append(rec[f"team_{team}_{half}"])
    return out


# --------------------------------------------------- numpy-side glue
def score_rally_tracked(model, rd):
    """swing_explore.score_rally, but KEEPING the track id.

    score_rally drops tid on the floor — it only ever needed side. We
    need the track itself, because the track is what carries the image
    position that becomes the quadrant. Same features, same peaks,
    same refractory pruning; only the return shape differs."""
    import numpy as np
    import swing_explore as SE
    dets = []
    for tid, ser in rd["tracks"].items():
        t = ser["t"]
        ts, feats = [], []
        for i in range(0, len(t), SE.STRIDE):
            f = SE.window_feats(ser, float(t[i]))
            if f is not None:
                ts.append(float(t[i]))
                feats.append(f)
        if not feats:
            continue
        p = SE.predict(model, np.stack(feats))
        cands = [(ts[i], float(p[i])) for i in range(1, len(p) - 1)
                 if p[i] >= p[i - 1] and p[i] >= p[i + 1] and p[i] > 0.02]
        for tt, sc in SE.strongest_first(cands, SE.REFRACTORY_S):
            dets.append((tt, ser["side"], sc, tid))
    dets.sort()
    return dets


def _nearest_i(ser, t):
    """Index of this track's sample nearest time t, or None.

    Deliberately pure python (it indexes numpy arrays just as happily):
    the geometry helpers are the part worth unit-testing, and a numpy
    import here would make the selftest need the whole pose stack."""
    tt = ser["t"]
    n = len(tt)
    if not n:
        return None
    return min(range(n), key=lambda i: abs(float(tt[i]) - t))


def box_at(ser, t):
    """(cx, y_bottom) of this track nearest time t, or None.

    track_series exposes 'cx' (box centre x) and 'ynorm' (box bottom)
    rather than raw boxes — cx is precisely the quantity the left/right
    comparison needs, so nothing is reconstructed here."""
    i = _nearest_i(ser, t)
    if i is None:
        return None
    return float(ser["cx"][i]), float(ser["ynorm"][i])


def is_image_right_of_pair(rd, side, tid, t):
    """Is track `tid` the RIGHT-hand one of the two players on `side`
    at time t? Relative comparison — see THE RELATIVE TRICK above.
    Returns None when the partner is not visible, so the caller can
    count how often the geometry was unavailable instead of guessing."""
    mine = box_at(rd["tracks"][tid], t)
    if mine is None:
        return None
    others = []
    for other_tid, ser in rd["tracks"].items():
        if other_tid == tid or ser["side"] != side:
            continue
        c = box_at(ser, t)
        if c is not None and abs(_nearest_dt(ser, t)) <= 0.5:
            others.append(c[0])
    if not others:
        return None
    return mine[0] > max(others) if len(others) == 1 else \
        mine[0] > sorted(others)[len(others) // 2]


def _nearest_dt(ser, t):
    i = _nearest_i(ser, t)
    return 9e9 if i is None else float(ser["t"][i]) - t


# ------------------------------------------------------------ report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--windows", default=WINDOWS_V4)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--pose-dir", default=POSE_DIR)
    ap.add_argument("--timeline-dir",
                    help="folder holding rally_timeline_<mid8>.csv, if "
                         "it is somewhere the search does not cover")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    run(a)


def run(a):
    import numpy as np
    import swing_explore as SE
    from contact_ceiling import (load_rosters, load_labels,
                                 rally_candidates, rally_coverage)
    try:
        import lineup as LU
    except ModuleNotFoundError:
        raise SystemExit(
            "lineup.py not found next to this script.\n"
            "It is the state machine that turns court positions into "
            "names (vision/lineup.py in the repo); only its walk_match "
            "is used here, and it needs no paths of its own.")

    rosters = load_rosters(Path(a.windows))
    labels = load_labels(Path(a.labels), rosters)
    train = {int(r["rally_cum"]) for r in csv.DictReader(open(a.split))
             if r["split"] == "train"}
    wrows = {int(r["rally_cum"]): r
             for r in csv.DictReader(open(a.windows))}
    names_by_uuid = {}
    truth = defaultdict(list)
    for r in csv.DictReader(open(a.labels)):
        names_by_uuid[r["hitter_uuid"].lower()] = r["hitter_name"]
        if r.get("contact", "1") == "1":
            truth[int(r["rally_cum"])].append(
                (float(r["t_refined_s"] or r["t_tap_s"]), r["hitter_name"]))
    for v in truth.values():
        v.sort()

    # ---- lineup state machine, per match, from the referee log alone
    match_ids = {w["match_id"] for c, w in wrows.items()
                 if c in train and c not in EXCLUDE}
    lineup_by_rally = {}
    for mid in sorted(match_ids):
        rows, path = load_timeline(mid, a.timeline_dir)
        if rows is None:
            print(f"no rally_timeline_{mid[:8]}.csv found — searched "
                  f"{', '.join(TIMELINE_DIRS)} (pass --timeline-dir)")
            continue
        try:
            rec_rows, diag = LU.walk_match(rows)
        except Exception as e:                     # noqa: BLE001
            print(f"lineup failed for {mid[:8]}: {e}")
            continue
        acc = diag.get("acc") if isinstance(diag, dict) else None
        print(f"lineup {mid[:8]} ({path}): receiver-prediction "
              f"{acc if acc is None else format(acc, '.1%')} "
              f"(its own free self-check)")
        for rr in rec_rows:
            lineup_by_rally[(mid, int(rr["rally"]))] = rr

    # ---- rallies with pose + labels (same loader as swing_explore)
    rallies = {}
    for cum, d in labels.items():
        if cum not in train or cum in EXCLUDE or not d["contacts"]:
            continue
        rd = SE.load_rally(a.pose_dir, cum)
        if rd is None:
            continue
        cands, _b = rally_candidates(rd["z"])
        _fl, m_raw = rally_coverage(d["contacts"], cands, 2, SE.TOL_S)
        m_srv, margin = SE.serve_mapping(rd, d["contacts"])
        rallies[cum] = {"rd": rd, "contacts": d["contacts"],
                        "whiffs": d["whiffs"],
                        "m": m_srv if margin >= 1.25 else m_raw}
    if not rallies:
        raise SystemExit("no train rallies with pose — check --pose-dir")

    # ---- decode every rally (leave-one-rally-out, as swing_explore)
    decoded = {}
    for held in sorted(rallies):
        Xtr, ytr = [], []
        for cum, r in rallies.items():
            if cum == held:
                continue
            X, y = SE.rally_instances(r["rd"], r["contacts"], r["whiffs"],
                                      r["m"])
            Xtr += X
            ytr += y
        model = SE.fit_logreg(np.stack(Xtr), np.array(ytr, float))
        r = rallies[held]
        dets = score_rally_tracked(model, r["rd"])
        tid_of = {(round(t, 3), s): tid for t, s, _sc, tid in dets}
        path = SE.decode_rally([(t, s, sc) for t, s, sc, _ in dets],
                               r["contacts"][0][1] ^ r["m"])
        evs = []
        for t, s, _sc, _g in path:
            evs.append((t, s, tid_of.get((round(t, 3), s))))
        decoded[held] = evs

    # ---- orientation: vote on SERVES, where the log knows the hitter
    samples, name_team = [], {}
    for cum, evs in decoded.items():
        w = wrows[cum]
        rec = lineup_by_rally.get((w["match_id"], int(w["rally_in_game"])))
        if rec is None or not evs:
            continue
        for tm in ("A", "B"):
            for hf in (RIGHT, LEFT):
                nm = names_by_uuid.get(rec[f"team_{tm}_{hf}"].lower())
                if nm:
                    name_team[nm] = tm
        srv = names_by_uuid.get(rec["server_uuid"].lower())
        t0, s0, tid0 = evs[0]
        if srv is None or tid0 is None:
            continue
        right = is_image_right_of_pair(rd_of(rallies, cum), s0, tid0, t0)
        if right is None:
            continue
        half = rec["server_half"]
        team = rec["server_team"]
        samples.append((team, half, bool(s0 == 0), bool(right)))
    near_team, flip, agree, n_s = vote_orientation(samples)
    print(f"\norientation from {n_s} serves: near team {near_team}, "
          f"flip {flip}, agreement {agree:.0%} "
          f"(two bits, over-identified — low agreement means the "
          f"geometry is noise, not that the bits are wrong)")

    # ---- attribute, with alternation + serve anchor
    tot = ok = 0
    alt_changed = no_geom = unreadable = 0
    serve_checked = serve_agree = 0
    per_player = defaultdict(lambda: [0, 0])   # name -> [pipeline, truth]
    for cum in sorted(decoded):
        w = wrows[cum]
        rec = lineup_by_rally.get((w["match_id"], int(w["rally_in_game"])))
        evs = decoded[cum]
        if rec is None or not evs:
            continue
        rd = rd_of(rallies, cum)
        sides = [s for _t, s, _tid in evs]
        fixed, ch = alternation_fix(sides, sides[0])
        alt_changed += ch
        # LABEL ONCE AT THE SERVE, then ride the track ids: halves are
        # only defined at the serve, so a mid-rally switch must not be
        # allowed to rename anyone.
        tnames, geom_ok = label_tracks_at_serve(
            rd, evs[0][0], rec, near_team, flip, names_by_uuid)
        if not geom_ok:
            unreadable += 1
            continue
        called = []
        for (t, _s, tid), _s_fix in zip(evs, fixed):
            nm = tnames.get(tid)
            if nm is None:
                no_geom += 1
            called.append(nm)
        # FREE PER-RALLY CHECK: the serve's track should already carry
        # the logged server's name. Disagreement means the orientation
        # or the serve geometry is wrong for this rally, and it is
        # counted before the anchor overwrites the evidence.
        srv = names_by_uuid.get(rec["server_uuid"].lower())
        if srv and called:
            serve_checked += 1
            serve_agree += (called[0] == srv)
        # serve anchor: contact 1 is the logged server, full stop
        if called and srv:
            called[0] = srv
        for nm in called:
            if nm:
                per_player[nm][0] += 1
        for _t, nm in truth[cum]:
            per_player[nm][1] += 1
        # grade: k-th call vs k-th truth (same order-join as the pilot)
        for k, nm in enumerate(called):
            if nm is None or k >= len(truth[cum]):
                continue
            tot += 1
            ok += nm == truth[cum][k][1]

    print(f"\nALTERNATION overwrote {alt_changed} decoded sides "
          f"(tracker/decoder disagreements with the exact constraint)")
    print(f"rallies skipped, serve geometry unreadable: {unreadable} "
          f"(a side did not show exactly two tracks)")
    print(f"events on an unlabelled track: {no_geom}")
    if serve_checked:
        print(f"SERVE CHECK: {serve_agree}/{serve_checked} = "
              f"{serve_agree / serve_checked:.0%} of rallies had the "
              f"serve track already carrying the logged server's name "
              f"BEFORE the anchor was applied\n  (this is the honest "
              f"read on the whole geometry chain — orientation, side "
              f"and left/right — measured for free on every rally)")
    if tot:
        print(f"\nATTRIBUTION (no API, no appearance model): "
              f"{ok}/{tot} = {ok / tot:.0%}")
        print(f"  VLM comparison on the same rallies: identity 44%, "
              f"side 70% (2026-08-21, $2.59)")
    print("\nTOUCH COUNTS (pipeline vs truth)")
    for nm in sorted(per_player):
        p, t_ = per_player[nm]
        print(f"  {nm:<22} pipeline {p:>3}  true {t_:>3}  "
              f"delta {p - t_:+d}")


def rd_of(rallies, cum):
    return rallies[cum]["rd"]


TIMELINE_DIRS = (".", "data/vision", "../data/vision", "vision",
                 "../data", "data")


def find_timeline(match_id, extra=None):
    """Path to rally_timeline_<mid8>.csv, searched across layouts.

    lineup.py resolves its own paths as <module>/../data/vision, which
    is right in the repo and wrong in the flat working folder this
    project is actually driven from. Only walk_match is needed from
    that module and it is a pure function over rally rows, so the file
    is located here instead and lineup.py's constants never come into
    it."""
    name = f"rally_timeline_{match_id[:8]}.csv"
    here = Path(__file__).resolve().parent
    roots = [Path(d) for d in ([extra] if extra else [])]
    roots += [Path(d) for d in TIMELINE_DIRS]
    roots += [here / d for d in TIMELINE_DIRS]
    for r in roots:
        p = r / name
        if p.exists():
            return p
    return None


def load_timeline(match_id, extra=None):
    p = find_timeline(match_id, extra)
    if p is None:
        return None, None
    return list(csv.DictReader(open(p))), p


def selftest():
    # quadrant round trip under every hypothesis — the inverse must be
    # exact or names silently land on the wrong player
    for near_team in ("A", "B"):
        for flip in (0, 1):
            for tm in ("A", "B"):
                for hf in (RIGHT, LEFT):
                    q = predict_image_quadrant(tm, hf, near_team, flip)
                    assert invert_image_quadrant(*q, near_team, flip) == \
                        (tm, hf), (tm, hf, near_team, flip)

    # the ends mirror: same half, opposite ends -> opposite image side
    a_near = predict_image_quadrant("A", RIGHT, "A", 0)
    b_far = predict_image_quadrant("B", RIGHT, "A", 0)
    assert a_near == (True, True) and b_far == (False, False)

    # orientation voting recovers planted bits from serve observations
    truth_bits = ("B", 1)
    obs = [(tm, hf) + predict_image_quadrant(tm, hf, *truth_bits)
           for tm in ("A", "B") for hf in (RIGHT, LEFT)] * 5
    nt, fl, agree, n = vote_orientation(obs)
    assert (nt, fl) == truth_bits and agree == 1.0 and n == 20
    # and degrades honestly rather than reporting false confidence
    noisy = obs[:-4] + [("A", RIGHT, False, False)] * 4
    _nt, _fl, agree_n, _n = vote_orientation(noisy)
    assert agree_n < 1.0

    # alternation: phase from the anchor, exact overwrite, honest count
    fixed, ch = alternation_fix([0, 1, 0, 1], 0)
    assert fixed == [0, 1, 0, 1] and ch == 0
    fixed, ch = alternation_fix([0, 0, 0, 1], 0)      # tracker slipped
    assert fixed == [0, 1, 0, 1] and ch == 1
    fixed, ch = alternation_fix([1, 1, 1], 1)
    assert fixed == [1, 0, 1] and ch == 1, (fixed, ch)
    # phase comes from the anchor, so a wrong first side moves EVERY
    # contact — the case that makes the serve anchor load-bearing
    fixed, ch = alternation_fix([0, 1, 0, 1], 1)
    assert fixed == [1, 0, 1, 0] and ch == 4, (fixed, ch)

    # assign_names reads the lineup table, never appearance
    rec = {"team_A_R": "Ann", "team_A_L": "Bea",
           "team_B_R": "Cal", "team_B_L": "Dee"}
    got = assign_names([(0.0, True, True), (1.0, False, True)],
                       rec, "A", 0)
    assert got == ["Ann", "Dee"], got

    # LABEL-AT-SERVE vs the mid-rally switch it exists to survive.
    # Fake rd: two tracks per side; the near pair SWAPS image x
    # partway through the rally, as poaching/stacking really does.
    class _S(dict):
        pass

    def mk(side, xs, ts):
        s = _S(t=list(ts), cx=list(xs), ynorm=[0.0] * len(ts))
        s["side"] = side
        return s

    ts = [0.0, 1.0, 2.0]
    rd = {"tracks": {
        1: mk(0, [100.0, 100.0, 900.0], ts),   # near, starts LEFT, crosses
        2: mk(0, [900.0, 900.0, 100.0], ts),   # near, starts RIGHT
        3: mk(1, [100.0, 100.0, 100.0], ts),   # far, steady
        4: mk(1, [900.0, 900.0, 900.0], ts)}}
    rec = {"team_A_R": "ua", "team_A_L": "ub",
           "team_B_R": "uc", "team_B_L": "ud"}
    nm = {"ua": "Ann", "ub": "Bea", "uc": "Cal", "ud": "Dee"}
    lab, ok = label_tracks_at_serve(rd, 0.0, rec, "A", 0, nm)
    assert ok and lab[1] == "Bea" and lab[2] == "Ann", lab
    # after the crossing, track 1 is on the RIGHT — re-deriving
    # left/right at t=2 would now call it Ann, renaming both partners.
    # Track identity must not move.
    assert is_image_right_of_pair(rd, 0, 1, 2.0) is True
    assert lab[1] == "Bea", "identity must ride the track, not the side"
    # a side missing a player is reported, never guessed
    rd_bad = {"tracks": {1: rd["tracks"][1], 3: rd["tracks"][3],
                         4: rd["tracks"][4]}}
    _lab2, ok2 = label_tracks_at_serve(rd_bad, 0.0, rec, "A", 0, nm)
    assert ok2 is False

    print("selftest OK: quadrant round trip, end mirroring, orientation "
          "voting (clean + noisy), alternation overwrite, name "
          "assignment, label-at-serve survives a mid-rally switch")


if __name__ == "__main__":
    main()
