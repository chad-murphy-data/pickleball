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


SWITCH_AT = 6      # MLP switches ends at 6 in ALL games (house rule)


def epoch_of_score(start_score, switch_at=SWITCH_AT):
    """0 before the end change, 1 after. start_score is the log's
    'serving-receiving-server#' triple, so the ends have swapped once
    either side has reached switch_at.

    THE 2026-08-21 BUG THIS EXISTS FOR: orientation was solved once per
    match, so every rally after the switch was predicted backwards and
    agreement pinned near 50% — the exact signature of a missing state
    transition rather than of noisy geometry."""
    parts = str(start_score).split("-")
    vals = [int(p) for p in parts[:2] if p.isdigit()]
    return 1 if vals and max(vals) >= switch_at else 0


def vote_orientation(samples, ends_switch=True):
    """samples = [(team, half, is_near, is_image_right)] or the same
    with a trailing epoch. Returns (near_team, flip, agreement, n).

    With ends_switch, the near team INVERTS in epoch 1, so all serves
    still inform the same two bits rather than splitting the sample —
    which matters when a match yields only ~15 serves. Four hypotheses,
    majority wins."""
    norm = [(s if len(s) == 5 else tuple(s) + (0,)) for s in samples]
    best = None
    for near_team in ("A", "B"):
        for flip in (0, 1):
            ok = 0
            for tm, hf, near, right, ep in norm:
                nt = (other_team(near_team)
                      if (ends_switch and ep) else near_team)
                ok += predict_image_quadrant(tm, hf, nt, flip) == \
                    (near, right)
            cand = (ok, near_team, flip)
            if best is None or cand > best:
                best = cand
    ok, near_team, flip = best
    n = len(norm)
    return near_team, flip, (ok / n if n else 0.0), n


def best_orientation_model(samples):
    """Fit orientation WITH and WITHOUT an end change and report both.

    The switch is a house rule, not something to take on faith for a
    given clip, so it is tested rather than assumed: if the switching
    model wins clearly the ends really do change here, and if the two
    tie the clip never crossed the switch."""
    a = vote_orientation(samples, ends_switch=True)
    b = vote_orientation(samples, ends_switch=False)
    use_switch = a[2] >= b[2]
    return (a if use_switch else b), use_switch, a[2], b[2]


def effective_near_team(near_team, epoch, ends_switch):
    return (other_team(near_team)
            if (ends_switch and epoch) else near_team)


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
    # ONLY THE PLAYER TRACKS. Rallies carry 5-25 tracks (referees,
    # ball crew, crowd); r17 had 25. Accepting whoever happened to be
    # on screen let a non-player displace a real one and silently
    # rename people, and it also made three rallies "unreadable" for
    # having a fifth body present. anchor_time already selects the
    # four longest tracks as the players — labelling now uses that
    # same set, so the two agree by construction.
    players = set(player_tracks(rd))
    seen = []
    for tid in players:
        ser = rd["tracks"][tid]
        c = box_at(ser, t_serve)
        if c is None or abs(_nearest_dt(ser, t_serve)) > 0.5:
            continue
        seen.append((c[1], c[0], tid))       # (y_bottom, cx, tid)
    if len(seen) != 4:
        return {}, False
    # SIDE FROM GEOMETRY, NOT FROM ser["side"]. The tracker's own side
    # field is a frame-local split and the vision postmortem measured
    # it corrupting 42% of FAR labels — trusting it put per-track
    # errors straight into the names. Image y is unambiguous instead:
    # the two players lower in frame are at the near end. Only the
    # RANKING is used, so zoom and camera height never enter.
    seen.sort()                              # by y: far (higher) first
    labels = {}
    for pair, is_near in ((seen[:2], False), (seen[2:], True)):
        for (_y, _cx, tid), img_right in zip(sorted(pair, key=lambda r: r[1]),
                                             (False, True)):
            team, half = invert_image_quadrant(is_near, img_right,
                                               near_team, flip)
            nm = name_of.get(rec.get(f"team_{team}_{half}", "").lower())
            if nm is None:
                return {}, False
            labels[tid] = nm
    return labels, True


def observe_quadrant(rd, t, tid):
    """(is_near, is_image_right) for track `tid` at time t, or None.

    Same geometry as label_tracks_at_serve and deliberately so: the
    orientation vote must observe the world exactly the way the
    labeller will, or it solves bits for a mapping nobody uses."""
    seen = []
    for other_tid in player_tracks(rd):
        ser = rd["tracks"][other_tid]
        c = box_at(ser, t)
        if c is None or abs(_nearest_dt(ser, t)) > 0.5:
            continue
        seen.append((c[1], c[0], other_tid))
    if len(seen) != 4 or tid not in [s[2] for s in seen]:
        return None
    seen.sort()
    near_ids = [s[2] for s in seen[2:]]
    is_near = tid in near_ids
    pair = seen[2:] if is_near else seen[:2]
    pair = sorted(pair, key=lambda r: r[1])
    return is_near, pair[1][2] == tid


def player_tracks(rd, k=4):
    """The k longest tracks — the four players, as against fragments."""
    order = sorted(rd["tracks"].items(),
                   key=lambda kv: -len(kv[1]["t"]))
    return [tid for tid, _ser in order[:k]]


def anchor_time(rd, fallback):
    """When all four players are first simultaneously on screen.

    The decoder's FIRST EVENT is a poor stand-in for the serve — the
    dry run put it 1.4-1.7s from the true first contact on the long
    rallies — and labelling "at the serve" from the wrong instant
    scrambles the lineup mapping. The moment the four tracks first
    coexist is serve formation by construction and needs no decode at
    all."""
    firsts = []
    for tid in player_tracks(rd):
        t = rd["tracks"][tid]["t"]
        if len(t):
            firsts.append(float(t[0]))
    return max(firsts) if len(firsts) == 4 else fallback


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
        try:
            rec_rows, note = lineup_records(mid, a.timeline_dir,
                                            LU.walk_match)
        except Exception as e:                     # noqa: BLE001
            print(f"lineup failed for {mid[:8]}: {e}")
            continue
        if rec_rows is None:
            print(f"{mid[:8]}: no lineup_{mid[:8]}.csv and no "
                  f"rally_timeline_{mid[:8]}.csv — searched "
                  f"{', '.join(TIMELINE_DIRS)} (pass --timeline-dir)")
            continue
        print(f"lineup {mid[:8]}: {note}")
        for rr in rec_rows:
            lineup_by_rally[(mid, int(rr["game"]), int(rr["rally"]))] = rr

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
    decoded, dets_by_rally = {}, {}
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
        dets_by_rally[held] = dets

    # ---- orientation: vote on SERVES, where the log knows the hitter
    samples, name_team = [], {}
    for cum, evs in decoded.items():
        w = wrows[cum]
        rec = lineup_by_rally.get(
            (w["match_id"], int(w["game"]), int(w["rally_in_game"])))
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
        rd_o = rd_of(rallies, cum)
        t_anc = anchor_time(rd_o, t0)
        obs = observe_quadrant(rd_o, t_anc, tid0)
        if obs is None:
            continue
        is_near, right = obs
        half = rec["server_half"]
        team = rec["server_team"]
        samples.append((team, half, bool(is_near), bool(right),
                        epoch_of_score(rec.get("start_score", ""))))
    (near_team, flip, agree, n_s), ends_switch, ag_sw, ag_no = \
        best_orientation_model(samples)
    print(f"\norientation from {n_s} serves: near team {near_team} "
          f"(epoch 0), flip {flip}, agreement {agree:.0%}")
    print(f"  end-change model: switching {ag_sw:.0%} vs fixed "
          f"{ag_no:.0%} -> using {'SWITCHING' if ends_switch else 'FIXED'}"
          f" ends")
    n_ep1 = sum(1 for s in samples if len(s) == 5 and s[4])
    print(f"  {n_ep1} of {n_s} serves are past the switch (score >= "
          f"{SWITCH_AT}); two bits, over-identified, so agreement well "
          f"under 100% means the geometry reads are noisy")

    # ---- attribute, with alternation + serve anchor
    tot = ok = 0
    alt_changed = no_geom = unreadable = 0
    serve_checked = serve_agree = 0
    per_player = defaultdict(lambda: [0, 0])   # name -> [pipeline, truth]
    for cum in sorted(decoded):
        w = wrows[cum]
        rec = lineup_by_rally.get(
            (w["match_id"], int(w["game"]), int(w["rally_in_game"])))
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
        nt = effective_near_team(
            near_team, epoch_of_score(rec.get("start_score", "")),
            ends_switch)
        tnames, geom_ok = label_tracks_at_serve(
            rd, anchor_time(rd, evs[0][0]), rec, nt, flip, names_by_uuid)
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

    # ---- TRUTH-ANCHORED GEOMETRY TEST (free, and the one that
    # separates the two failures the serve check conflates).
    #
    # SERVE CHECK asks "does the FIRST DECODED EVENT's track carry the
    # server's name", which fails both when the geometry is wrong and
    # when the decoder's first event is not the serve — and the dry run
    # showed that second case is common (median |dt| 0.49s, and 1.4-1.7s
    # on the long rallies). So it cannot say which link is broken.
    #
    # This test hands the decoder's PLACEMENT job to the labels: for
    # each TRUE contact, take the best-scoring detection near that true
    # time, and ask only whether the geometric labelling names its track
    # correctly. Placement error is removed by construction, so what is
    # left is the geometry chain alone. Reported beside placement
    # recall, which is the other half of the same picture.
    print("\nTRUTH-ANCHORED GEOMETRY TEST (decoder placement removed: "
          "true contact times, geometric names)")
    g_ok = g_n = miss = g_team = g_partner = 0
    census = []
    for cum in sorted(decoded):
        w = wrows[cum]
        rec = lineup_by_rally.get(
            (w["match_id"], int(w["game"]), int(w["rally_in_game"])))
        if rec is None or cum not in dets_by_rally:
            continue
        rd = rd_of(rallies, cum)
        nt = effective_near_team(
            near_team, epoch_of_score(rec.get("start_score", "")),
            ends_switch)
        tnames, ok_lab = label_tracks_at_serve(
            rd, anchor_time(rd, dets_by_rally[cum][0][0]
                            if dets_by_rally[cum] else 0.0),
            rec, nt, flip, names_by_uuid)
        durs = sorted((len(s["t"]) for s in rd["tracks"].values()),
                      reverse=True)
        census.append((cum, len(rd["tracks"]), durs[:4], ok_lab))
        if not ok_lab:
            continue
        for t_true, nm_true in truth[cum]:
            near = [d for d in dets_by_rally[cum]
                    if abs(d[0] - t_true) <= 0.35]
            if not near:
                miss += 1
                continue
            best = max(near, key=lambda d: d[2])
            nm = tnames.get(best[3])
            if nm is None:
                miss += 1
                continue
            g_n += 1
            g_ok += nm == nm_true
            # DECOMPOSE: team wrong = the near/far read failed;
            # team right but name wrong = left/right failed, which is
            # the half that stacking would break (a stacked team lines
            # up away from its nominal rules-half, so the state
            # machine's R/L stops describing where anyone stands).
            if name_team.get(nm) == name_team.get(nm_true):
                g_team += 1
                g_partner += nm == nm_true
    if g_n:
        print(f"  geometry names the right player on "
              f"{g_ok}/{g_n} = {g_ok / g_n:.0%} of truth-anchored "
              f"contacts   (chance 25%)")
        print(f"    TEAM (near/far end) right   {g_team}/{g_n} = "
              f"{g_team / g_n:.0%}   (chance 50%)")
        if g_team:
            print(f"    partner GIVEN team right    {g_partner}/{g_team}"
                  f" = {g_partner / g_team:.0%}   (chance 50%)")
        print("    team high + partner ~50% => left/right is the "
              "failure, and STACKING is the\n      prime suspect: a "
              "stacked team stands away from its nominal rules-half, "
              "so\n      the state machine's R/L stops describing "
              "where anybody actually is.")
    print(f"  placement recall: {miss} true contacts had no scored "
          f"detection within 0.35s (or landed on an unlabelled track)")
    print("\n  Read: geometry high here but attribution low => the "
          "decoder's PLACEMENT is the binding\n  constraint. Geometry "
          "low here too => the tracks themselves are not cleanly\n  "
          "four players at the anchor, which the census below shows.")
    print("\nTRACK CENSUS (rally: n_tracks, 4 longest sample counts, "
          "labelled?)")
    for cum, n_tr, durs, ok_lab in census:
        print(f"  r{cum:<4} tracks {n_tr:<3} longest {durs}  "
              f"{'ok' if ok_lab else 'UNREADABLE'}")

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


def find_lineup(match_id, extra=None):
    """Path to a PRECOMPUTED lineup_<mid8>.csv, if one is committed.

    lineup.py writes its walk_match output to data/vision/lineup_*.csv,
    and those files carry exactly the columns needed here (team_A_R/L,
    team_B_R/L, server_half, server_team). Preferring them matters in
    practice: the committed rally_timeline_*.csv files cover DIFFERENT
    matches than the Chicago windows reference, so re-deriving from a
    timeline fails for the very rallies this module runs on, while the
    lineup CSV for that match is right there."""
    name = f"lineup_{match_id[:8]}.csv"
    here = Path(__file__).resolve().parent
    roots = [Path(d) for d in ([extra] if extra else [])]
    roots += [Path(d) for d in TIMELINE_DIRS]
    roots += [here / d for d in TIMELINE_DIRS]
    for r in roots:
        p = r / name
        if p.exists():
            return p
    return None


def lineup_records(match_id, extra, walk_match):
    """Per-rally lineup records: the committed CSV when present, else
    the state machine re-walked over a timeline. Returns (recs, note)."""
    p = find_lineup(match_id, extra)
    if p is not None:
        recs = list(csv.DictReader(open(p)))
        ok = sum(int(r.get("receiver_ok") or 0) for r in recs)
        n = sum(1 for r in recs if r.get("receiver_ok") not in (None, ""))
        note = (f"{p} (precomputed; receiver-prediction "
                f"{ok}/{n} = {ok / n:.1%})" if n else f"{p} (precomputed)")
        return recs, note
    rows, tp = load_timeline(match_id, extra)
    if rows is None:
        return None, None
    recs, diag = walk_match(rows)
    acc = diag.get("acc")
    return recs, (f"{tp} (re-walked; receiver-prediction "
                  f"{acc:.1%})" if acc == acc else f"{tp} (re-walked)")


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

    # END CHANGE: the 2026-08-21 failure. Build serves either side of
    # the switch with the near team genuinely inverted after it; the
    # fixed-ends model must land near 50% while the switching model
    # recovers the bits exactly.
    ep0 = [(tm, hf) + predict_image_quadrant(tm, hf, "B", 0) + (0,)
           for tm in ("A", "B") for hf in (RIGHT, LEFT)] * 3
    ep1 = [(tm, hf) + predict_image_quadrant(tm, hf, "A", 0) + (1,)
           for tm in ("A", "B") for hf in (RIGHT, LEFT)] * 3
    mixed = ep0 + ep1
    (nt2, fl2, ag2, n2), used, ag_sw, ag_no = \
        best_orientation_model(mixed)
    assert used is True and (nt2, fl2) == ("B", 0), (nt2, fl2, used)
    assert ag2 == 1.0 and ag_no <= 0.5, (ag2, ag_no)
    assert n2 == 24
    # epoch parsing off the log's score triple
    assert epoch_of_score("0-0-2") == 0 and epoch_of_score("5-3-1") == 0
    assert epoch_of_score("6-2-1") == 1 and epoch_of_score("3-9-2") == 1
    assert effective_near_team("A", 1, True) == "B"
    assert effective_near_team("A", 1, False) == "A"
    # a clip that never crosses the switch must not be penalised
    (_n3, _f3, ag3, _c3), used3, _s3, no3 = best_orientation_model(ep0)
    assert ag3 == 1.0 and no3 == 1.0 and used3 is True

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

    def mk(side, xs, ts, y):
        # y matters now: near/far is read off image y, so a fixture
        # with a flat y would be testing nothing
        s = _S(t=list(ts), cx=list(xs), ynorm=[y] * len(ts))
        s["side"] = side
        return s

    ts = [0.0, 1.0, 2.0]
    rd = {"tracks": {
        1: mk(0, [100.0, 100.0, 900.0], ts, 600.0),  # near, LEFT, crosses
        2: mk(0, [900.0, 900.0, 100.0], ts, 600.0),  # near, RIGHT
        3: mk(1, [100.0, 100.0, 100.0], ts, 200.0),  # far, steady
        4: mk(1, [900.0, 900.0, 900.0], ts, 200.0)}}
    rec = {"team_A_R": "ua", "team_A_L": "ub",
           "team_B_R": "uc", "team_B_L": "ud"}
    nm = {"ua": "Ann", "ub": "Bea", "uc": "Cal", "ud": "Dee"}
    lab, ok = label_tracks_at_serve(rd, 0.0, rec, "A", 0, nm)
    assert ok and lab[1] == "Bea" and lab[2] == "Ann", lab
    # side now comes from image y, not ser["side"]: corrupt the
    # tracker's field entirely and the labels must not move
    for _t, _s in rd["tracks"].items():
        _s["side"] = 0
    lab_y, ok_y = label_tracks_at_serve(rd, 0.0, rec, "A", 0, nm)
    assert ok_y and lab_y == lab, (lab_y, lab)
    # observe_quadrant must agree with the labeller's own geometry
    assert observe_quadrant(rd, 0.0, 2) == (True, True)
    assert observe_quadrant(rd, 0.0, 3) == (False, False)
    # after the crossing, track 1 is on the RIGHT — re-deriving
    # left/right at t=2 would now call it Ann, renaming both partners.
    # Track identity must not move.
    assert is_image_right_of_pair(rd, 0, 1, 2.0) is True
    assert lab[1] == "Bea", "identity must ride the track, not the side"
    # a side missing a player is reported, never guessed
    rd_bad = {"tracks": {1: rd["tracks"][1], 3: rd["tracks"][3],
                         4: rd["tracks"][4]}}   # only three on court
    _lab2, ok2 = label_tracks_at_serve(rd_bad, 0.0, rec, "A", 0, nm)
    assert ok2 is False

    print("selftest OK: quadrant round trip, end mirroring, orientation "
          "voting (clean + noisy), alternation overwrite, name "
          "assignment, label-at-serve survives a mid-rally switch")


if __name__ == "__main__":
    main()
