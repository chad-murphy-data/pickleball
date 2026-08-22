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
import math
import itertools
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


def intent_score(rd, tid, t, partner_x, is_near, half=0.6, step=0.1):
    """How much this track COMMITS toward hitting, as against yielding.

    THE USER'S REFINEMENT (2026-08-21), and it is ball-free, which is
    the point: the ball abstains on ~28% of contacts and this covers
    exactly those. Raw displacement failed because partners move
    together; what separates them is not how far but HOW and WHERE:

      INTRUSION    moving into the partner's half is a commitment to
                   hit. "When Alshon goes into Tyra's half, he's
                   probably hitting" — measured as motion toward the
                   partner's starting x, signed, so yielding scores
                   negative rather than merely small.
      PURPOSE      a hitter's move is direct; clearing out wanders.
                   Straightness = net displacement / path length, in
                   [0,1], and it MULTIPLIES the rest so that a big
                   aimless shuffle cannot outscore a short committed
                   step.
      NETWARD      the hitter steps into the court; the yielder backs
                   off and outward. Near players approach the net as y
                   falls, far players as y rises.

    Returns None when the window is not covered. The score is only ever
    compared BETWEEN the two players on a side — it has no absolute
    meaning, which is deliberate: partners moving together cancels."""
    pts = []
    k = -half
    while k <= 1e-9:
        c = box_at(rd["tracks"][tid], t + k)
        if c is None:
            return None
        pts.append(c)
        k += step
    if len(pts) < 3:
        return None
    path = sum(((pts[i + 1][0] - pts[i][0]) ** 2 +
                (pts[i + 1][1] - pts[i][1]) ** 2) ** 0.5
               for i in range(len(pts) - 1))
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    net_disp = (dx * dx + dy * dy) ** 0.5
    if path < 1e-6:
        return 0.0
    straightness = net_disp / path
    # toward the partner's territory: positive if closing on their x
    intrusion = dx if partner_x > pts[0][0] else -dx
    # toward the net: near end is LOW on screen, so netward is -dy
    netward = -dy if is_near else dy
    return straightness * (intrusion + netward)


def displacement(rd, tid, t, half=0.6):
    """How far this track's body travels across a window centred on t.

    THE USER'S POACHING SIGNAL (2026-08-21), applied to the binding.
    Position (depth, halves) says where someone STANDS; displacement
    says where they are GOING, which is a different measurement and
    fails differently. At the serve the server steps in and swings
    while the partner waits, and the receiver moves to return while
    their partner holds the kitchen — so on each side the mover is the
    log-named player."""
    a, b = box_at(rd["tracks"][tid], t - half), \
        box_at(rd["tracks"][tid], t + half)
    if a is None or b is None:
        return None
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


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


def track_motion(ser):
    """How far this track travels in image space (cx range + y range).

    THE DISCRIMINATOR BETWEEN PLAYERS AND OFFICIALS. Referees, line
    crew and courtside staff stand still for a whole rally, so they
    produce LONG tracks — r17 held 25 tracks and a stationary official
    can outlast a player. Selecting the four LONGEST therefore picked
    officials and dropped players, which corrupts near/far and
    left/right at once and is exactly the symmetric 79%/79% error the
    2026-08-21 decomposition showed. Players run; officials do not."""
    cx, y = ser["cx"], ser["ynorm"]
    if not len(cx):
        return 0.0
    return (max(cx) - min(cx)) + (max(y) - min(y))


def label_by_depth(rd, t_serve, rec, near_team, flip, name_of):
    """{track_id: name} from DEPTH plus the log — no left/right at all.

    WHY THIS EXISTS (2026-08-21). The permutation diagnosis came back
    TEAMS SWAPPED 0, PARTNERS SWAPPED 6 of 15: near/far is essentially
    perfect and the entire error is one inverted left/right bit. The
    cause is timing, not geometry — 5 of the 6 bad rallies FOLLOW A
    POINT, and a serving team swaps halves after scoring, so the
    lineup's halves describe where players stand AFTER a walk across
    that our anchor catches them in the middle of.

    Depth sidesteps it. At the serve the server stands behind their
    baseline and the receiver stands deep to return; both partners are
    up at the kitchen. The log names the server AND the receiver
    exactly, so the deeper player on each side is a KNOWN person and
    the shallower one is that person's partner. Left/right, the
    lineup halves, and the swap timing all drop out of the problem.

    Depth is also the more reliable read: it is a large separation
    along the axis the camera foreshortens least, whereas left/right
    at serve is a small separation that a mid-walk player crosses.

    Returns ({tid: name}, ok)."""
    seen = []
    for tid in player_tracks(rd):
        ser = rd["tracks"][tid]
        c = box_at(ser, t_serve)
        if c is None or abs(_nearest_dt(ser, t_serve)) > 0.5:
            continue
        seen.append((c[1], c[0], tid))       # (y_bottom, cx, tid)
    if len(seen) != 4:
        return {}, False
    seen.sort()
    far_pair, near_pair = seen[:2], seen[2:]

    srv = name_of.get(rec.get("server_uuid", "").lower())
    rcv = name_of.get(rec.get("receiver_uuid", "").lower())
    if not srv or not rcv:
        return {}, False
    team_members = {}
    for tm in ("A", "B"):
        mem = [name_of.get(rec.get(f"team_{tm}_{h}", "").lower())
               for h in (RIGHT, LEFT)]
        if any(m is None for m in mem):
            return {}, False
        team_members[tm] = mem
    srv_team = rec.get("server_team")
    if srv_team not in team_members:
        return {}, False
    rcv_team = other_team(srv_team)
    srv_mate = next(m for m in team_members[srv_team] if m != srv)
    rcv_mate = next(m for m in team_members[rcv_team] if m != rcv)

    # which image end is the serving team on? near/far is the channel
    # that measured 0 errors, so this is the safe half of the mapping.
    srv_is_near = (srv_team == near_team)
    srv_pair = near_pair if srv_is_near else far_pair
    rcv_pair = far_pair if srv_is_near else near_pair

    def deep_first(pair, is_near):
        # depth = distance from the net. The near end is LOW on screen,
        # so its deeper player has the LARGER y; the far end mirrors.
        return sorted(pair, key=lambda r: -r[0] if is_near else r[0])

    labels = {}
    sp = deep_first(srv_pair, srv_is_near)
    labels[sp[0][2]], labels[sp[1][2]] = srv, srv_mate
    rp = deep_first(rcv_pair, not srv_is_near)
    labels[rp[0][2]], labels[rp[1][2]] = rcv, rcv_mate
    return labels, True


BALL_MIN_MARGIN = 25.0    # px: below this the two are equidistant


def label_by_vote(rd, t_serve, rec, near_team, flip, name_of,
                  events=None, tally=None, votes_out=None,
                  order=None, ball_pts=None, allow_movement=False,
                  picks_out=None, diagonal=True, diag_out=None):
    """{track_id: name} by VOTING the one bit that is actually in doubt.

    THE 2026-08-21 ENSEMBLE FINDING (user's proposal, and the data
    agreed). Depth and halves score 58% and 62% overall but fail on
    DISJOINT rallies: halves wins r1/r2/r7/r15, depth wins r4/r14/r19,
    and only r4/r6/r16/r17/r8 defeat both. Union of correct rallies is
    10 of 15 against 8 and 6 alone — so the information is there, in
    two channels whose errors are uncorrelated.

    Both labellers already agree on TEAM (near/far measured identical,
    and the permutation diagnosis put TEAMS SWAPPED at 0). What they
    disagree about is ONE BIT PER SIDE: which of the two players is the
    server, and which is the receiver. The referee log names both of
    those people exactly, so that bit has three independent witnesses:

      CONTACT ORDER  the track that hits contact 1 IS the server, and
                     contact 2 IS the receiver. Independent of all
                     geometry; costs a decoder placement, which the
                     placement test showed exists for every contact.
      DEPTH          server and receiver stand deep at the serve.
                     Independent of left/right, immune to the
                     post-point swap.
      HALVES         the lineup's R/L mapping. Independent of depth,
                     but inverts when a serving team just swapped.

    Majority of three. Ensembling the BIT rather than the labelling is
    strictly better than picking a labeller per rally: the serving and
    receiving sides get decided separately, so a rally can take depth's
    answer on one end and halves' on the other.

    tally, if given, counts each voter's agreement with the final call
    so a dead or harmful voter is visible rather than assumed useful."""
    seen = []
    for tid in player_tracks(rd):
        ser = rd["tracks"][tid]
        c = box_at(ser, t_serve)
        if c is None or abs(_nearest_dt(ser, t_serve)) > 0.5:
            continue
        seen.append((c[1], c[0], tid))
    if len(seen) != 4:
        return {}, False
    seen.sort()
    far_pair, near_pair = seen[:2], seen[2:]

    srv = name_of.get(rec.get("server_uuid", "").lower())
    rcv = name_of.get(rec.get("receiver_uuid", "").lower())
    srv_team = rec.get("server_team")
    if not srv or not rcv or srv_team not in ("A", "B"):
        return {}, False
    members = {}
    for tm in ("A", "B"):
        mem = [name_of.get(rec.get(f"team_{tm}_{h}", "").lower())
               for h in (RIGHT, LEFT)]
        if any(m is None for m in mem):
            return {}, False
        members[tm] = mem
    rcv_team = other_team(srv_team)
    srv_mate = next(m for m in members[srv_team] if m != srv)
    rcv_mate = next(m for m in members[rcv_team] if m != rcv)

    srv_is_near = (srv_team == near_team)
    srv_pair = near_pair if srv_is_near else far_pair
    rcv_pair = far_pair if srv_is_near else near_pair
    ev = list(events or [])

    def pick(pair, is_near, who, half_of_who, contact_idx, decider_out):
        """Which track in `pair` is `who`? Three votes, majority."""
        a, b = pair                       # (y, cx, tid) each
        votes = []
        # DEPTH: near end is low on screen, so deeper = larger y
        deep = a if ((a[0] > b[0]) == is_near) else b
        votes.append(("depth", deep[2]))
        # HALVES: whose half maps to which image side
        _tm, _hf = None, None
        want_right = predict_image_quadrant(
            srv_team if who in members[srv_team] else rcv_team,
            half_of_who, near_team, flip)[1]
        right_tid = (a if a[1] > b[1] else b)[2]
        left_tid = (b if a[1] > b[1] else a)[2]
        votes.append(("halves", right_tid if want_right else left_tid))
        # MOVEMENT (raw displacement) — MEASURED NULL, 2026-08-21:
        # 59% overall, 50% on disputed calls. The user diagnosed why,
        # and the diagnosis is the interesting part: PARTNERS MOVE
        # TOGETHER. When one player intrudes toward the ball the other
        # yields — backs out, cedes the space, covers elsewhere — so
        # both are moving and magnitude cannot separate them. Kept
        # behind --with-movement purely so the null stays reproducible.
        if allow_movement:
            da = displacement(rd, a[2], t_serve)
            db = displacement(rd, b[2], t_serve)
            if da is not None and db is not None and abs(da - db) > 1.0:
                votes.append(("movement", a[2] if da > db else b[2]))

        # INTENT: intrusion + purpose + netward, contrasted between the
        # two partners. Ball-free by design — it is the channel that
        # covers the contacts where no flight was tracked.
        ia = intent_score(rd, a[2], t_serve, b[1], is_near)
        ib = intent_score(rd, b[2], t_serve, a[1], is_near)
        if ia is not None and ib is not None and abs(ia - ib) > 5.0:
            votes.append(("intent", a[2] if ia > ib else b[2]))

        # BALL + APPROACH: what displacement was missing is a DIRECTION
        # to measure against, and the ball supplies it (72% of contacts
        # carry a flight endpoint, measured on rally 1).
        #   ball     — who is nearer the ball where it was struck
        #   approach — who is CLOSING on it across the window, which is
        #              the intrude/yield asymmetry the user described:
        #              the hitter shortens the gap, the partner opens it
        bpt = (ball_pts or {}).get(contact_idx)
        if bpt is not None:
            ca, cb = box_at(rd["tracks"][a[2]], t_serve), \
                box_at(rd["tracks"][b[2]], t_serve)
            if ca and cb:
                da_b = ((ca[0] - bpt[0]) ** 2 + (ca[1] - bpt[1]) ** 2) ** .5
                db_b = ((cb[0] - bpt[0]) ** 2 + (cb[1] - bpt[1]) ** 2) ** .5
                if abs(da_b - db_b) > BALL_MIN_MARGIN:
                    votes.append(("ball", a[2] if da_b < db_b else b[2]))
            appr = {}
            for tid_x in (a[2], b[2]):
                p0 = box_at(rd["tracks"][tid_x], t_serve - 0.6)
                p1 = box_at(rd["tracks"][tid_x], t_serve)
                if p0 and p1:
                    d0 = ((p0[0] - bpt[0]) ** 2 + (p0[1] - bpt[1]) ** 2) ** .5
                    d1 = ((p1[0] - bpt[0]) ** 2 + (p1[1] - bpt[1]) ** 2) ** .5
                    appr[tid_x] = d0 - d1          # positive = closing
            if len(appr) == 2:
                (x1, v1), (x2, v2) = appr.items()
                if abs(v1 - v2) > 5.0:
                    votes.append(("approach", x1 if v1 > v2 else x2))
        # CONTACT ORDER: the k-th contact belongs to this person.
        #
        # INDEX-BASED, and that is a MEASURED choice. Taking the
        # earliest event landing on the pair was tried (2026-08-21) to
        # widen this voter past its 20% silence, and it lost: geometry
        # 89% -> 84%, disputed accuracy 73% -> 67%, and r5 collapsed
        # 14/14 -> 7/14. Firing more often is worthless if the extra
        # firings are wrong — a spurious pre-serve event on the serving
        # pair becomes "the serve" and outranks the real one, which is
        # exactly the failure the selftest below documents.
        #
        # The decoder's FIRST events are usually right even though it
        # over-counts overall, so the index is the better anchor and
        # this voter should stay silent rather than guess.
        if contact_idx < len(ev):
            tid_c = ev[contact_idx][2]
            if tid_c in (a[2], b[2]):
                votes.append(("contact", tid_c))
        if votes_out is not None:
            votes_out[who] = list(votes)
        if order:
            # CASCADE: precedence rather than majority. The first
            # listed voter that fired decides; the rest are consulted
            # only when it is silent. Exists because `contact` is not
            # really a vote — the track that hits contact 1 IS the
            # server per the log — so being outvoted 2-1 by geometry
            # throws away a near-deduction. Precedence must be earned
            # by the voter-vs-truth table, never assumed.
            byname = {v: t for v, t in votes}
            for v in order:
                if v in byname:
                    best_tid, v_dec = byname[v], v
                    break
            else:
                best_tid = Counter(t for _v, t in votes).most_common(1)[0][0]
                v_dec = "majority"
        else:
            counts = Counter(t for _v, t in votes)
            best_tid, _n = counts.most_common(1)[0]
            v_dec = "majority"
        if tally is not None:
            for vname, t in votes:
                tally[vname][0] += (t == best_tid)
                tally[vname][1] += 1
        decider_out.append(v_dec)
        if picks_out is not None:
            # The votes alone cannot say WHY a side bit came out wrong:
            # a cascade pick and a majority pick look identical in the
            # vote list. Recording the WINNER separates "every voter was
            # wrong" from "the right voter was outranked" — which need
            # opposite fixes, and which the aggregate tables conflate.
            picks_out[who] = (best_tid, contact_idx)
        return best_tid

    half_of = {}
    for tm in ("A", "B"):
        for h in (RIGHT, LEFT):
            nm = name_of.get(rec.get(f"team_{tm}_{h}", "").lower())
            half_of[nm] = h
    s_dec, r_dec = [], []
    s_tid = pick(srv_pair, srv_is_near, srv, half_of[srv], 0, s_dec)
    r_tid = pick(rcv_pair, not srv_is_near, rcv, half_of[rcv], 1, r_dec)

    # ---- THE DIAGONAL. A serve must travel cross-court, so at the
    # serve the server and the receiver stand diagonally opposite:
    # exactly ONE of them is the image-right member of their own pair.
    # The two bits were being decided INDEPENDENTLY, so nothing stopped
    # the pair of decisions from describing a serve down the middle,
    # which cannot happen.
    #
    # Same class of constraint as alternation (+14 points): a rule of
    # the sport, exact, free, and it eliminates two of the four
    # combinations rather than merely preferring one.
    #
    # WHY IT APPLIES EXACTLY HERE and nowhere else: the server and the
    # receiver are the only two players the rules place. Their partners
    # may stand anywhere - that is what stacking IS - so no positional
    # claim about a partner is safe, while these two are pinned by the
    # laws of the game at the moment the bits are about.
    #
    # A violation says one of the two bits is wrong but not which, so
    # flip the one whose DECIDER is less reliable, measured on the
    # right-when-deciding column rather than assumed. A double
    # inversion satisfies the diagonal and stays invisible to it -
    # that is r6, and it is a real limit of this constraint, not an
    # oversight.
    # MEASURED HARMFUL, 2026-08-21, and the user's own description of
    # stacking is why: geometry 89% -> 74%, and it fired on 8 of 15
    # rallies when only 4 bits were ever wrong, so it was breaking
    # CORRECT ones (r4 14/16 -> 6/16, r8 7/7 -> 4/7, r14 11/11 -> 6/11).
    #
    # The constraint is true about COURT HALVES. This tests WITHIN-PAIR
    # IMAGE ORDER, which only stands in for court halves when the two
    # players straddle the centreline - and "stacking when serving
    # usually just involves two people hanging out on the same side of
    # the court" (user, 2026-08-21). Both players in one half, and
    # which is image-right says nothing about service courts.
    #
    # Second defect, independent of the first: the deciders were
    # `contact` on 14 of 16 sides, so both bits usually came from the
    # SAME voter, s_conf < r_conf was false, and the repair silently
    # always flipped the receiver. An arbitrary tie-break on top of a
    # constraint that should not have fired.
    #
    # A correct version needs ABSOLUTE court halves - a centreline in
    # image x - and must abstain whenever a pair does not straddle it.
    # Kept behind --diagonal so the null is reproducible, not deleted.
    if diagonal:
        s_right = s_tid == max(srv_pair, key=lambda r: r[1])[2]
        r_right = r_tid == max(rcv_pair, key=lambda r: r[1])[2]
        if s_right == r_right:
            s_conf = DECIDER_CONF.get(s_dec[0] if s_dec else None, 0.5)
            r_conf = DECIDER_CONF.get(r_dec[0] if r_dec else None, 0.5)
            if s_conf < r_conf:
                s_tid = next(t for _y, _cx, t in srv_pair if t != s_tid)
            else:
                r_tid = next(t for _y, _cx, t in rcv_pair if t != r_tid)
            if diag_out is not None:
                diag_out.append((s_dec[0] if s_dec else "?",
                                 r_dec[0] if r_dec else "?"))
    labels = {s_tid: srv, r_tid: rcv}
    for _y, _cx, tid in srv_pair:
        labels.setdefault(tid, srv_mate)
    for _y, _cx, tid in rcv_pair:
        labels.setdefault(tid, rcv_mate)
    return labels, True



# Right-when-deciding, measured on the 2026-08-21 train panel. Used
# ONLY to break a diagonal violation: when the two bits contradict a
# rule of the sport, the less reliable decider is the one to flip.
# Kept as measurements rather than a hand-ranked list so it is obvious
# when they go stale - re-read them off the PER-VOTER table.
DECIDER_CONF = {"contact": 0.88, "halves": 0.83, "depth": 0.70,
                "ball": 0.56, "intent": 0.29, "approach": 0.0,
                "majority": 0.5}


def decide_by_order(order, by):
    """Exactly pick()'s rule: first voter that fired, else majority.

    Kept at module level, and IDENTICAL to the branch inside pick(), so
    the order sweep replays real decisions rather than an approximation
    of them. If pick()'s rule ever changes, this must change with it —
    the selftest pins the two shapes that matter (precedence wins over
    a majority; silence falls through)."""
    for v in order:
        if v in by:
            return by[v], v
    if not by:
        return None, None
    return Counter(by.values()).most_common(1)[0][0], "majority"


def apply_veto(leader_tid, leader, by, k, mode):
    """Let the voters BELOW the leader overturn it, or return None.

    A pure cascade can only ever be as right as whichever voter speaks
    first, which is the wrong shape when the leader is excellent in
    general but wrong in a specific, detectable situation - exactly
    what the 2026-08-21 trace showed for `contact` (88% overall, yet
    wrong on 3 of the 4 broken bits, with a lower voter right in every
    one).

    Two modes, because they encode different beliefs and the data
    should pick:
      unan  every other firing voter agrees on ONE other answer. Very
            conservative: it fires only when the leader is alone.
      maj   a strict majority of the other voters agree on one other
            answer. Fires more often, and can be wrong more often.
    k is the minimum number of other voters required, so a veto is
    never carried by a single dissenter."""
    others = [t for name, t in by.items() if name != leader]
    if len(others) < k or k <= 0:
        return None
    counts = Counter(others)
    top, n_top = counts.most_common(1)[0]
    if top == leader_tid:
        return None
    if mode == "unan" and n_top != len(others):
        return None
    if mode == "maj" and n_top * 2 <= len(others):
        return None
    return top


def score_order(order, bits, k=0, mode="unan"):
    """(bits right, bits, contacts right, contacts) for one ordering.

    Contact weighting matters because a side bit renames both players
    on one side for a WHOLE RALLY: under alternation it costs every
    contact of that parity, so a bit in a 25-contact rally and one in a
    2-contact rally are not the same mistake."""
    ok = n = wok = wn = 0
    for _cum, _who, truth_tid, by, w in bits:
        got, v = decide_by_order(order, by)
        if k and v is not None and v != "majority":
            alt = apply_veto(got, v, by, k, mode)
            if alt is not None:
                got = alt
        n += 1
        wn += w
        if got == truth_tid:
            ok += 1
            wok += w
    return ok, n, wok, wn


def total_speed(rd, tids, t, dt=0.2):
    """Summed image-space speed of `tids` around time t."""
    tot = 0.0
    for tid in tids:
        ser = rd["tracks"][tid]
        a, b = box_at(ser, t - dt), box_at(ser, t + dt)
        if a is None or b is None:
            return None
        tot += abs(b[0] - a[0]) + abs(b[1] - a[1])
    return tot


def settled_anchor(rd, t_first, back=6.0, step=0.2):
    """The instant before the first contact when the four players are
    most STILL — i.e. set up to serve.

    WHY (2026-08-21). After elimination fixed TEAM at 100%, the residual
    is the partner bit, wrong in 5 rallies — and all three voters agree
    ~85% with the final call while those rallies stay wrong, meaning the
    voters agree WITH EACH OTHER on a wrong answer. Their shared
    dependency is the anchor instant: fire it during the post-point walk
    and depth is wrong (nobody is deep yet), halves is wrong (nobody is
    in their half yet) and contact-order is wrong (the first event is
    not the serve). One bad moment corrupts all three at once, which is
    why more voters reading the same instant could not help.

    Players are stationary just before a serve and moving during the
    walk across, so minimum total speed picks the settled moment.
    Searches only BACKWARD from the first contact, never past it, so a
    mid-rally lull can never be mistaken for the serve."""
    tids = player_tracks(rd)
    if len(tids) != 4:
        return t_first
    best_t, best_v = None, None
    t = t_first
    while t >= t_first - back:
        v = total_speed(rd, tids, t)
        if v is not None and (best_v is None or v < best_v):
            best_t, best_v = t, v
        t -= step
    return t_first if best_t is None else best_t


def side_map(rd, t_anchor):
    """{track_id: is_near} from the 2/2 image-y split at the anchor.

    Near/far is the channel the permutation diagnosis measured at zero
    whole-rally errors, so this is the trustworthy half of the geometry
    and is what the alternation constraint gets to act on."""
    seen = []
    for tid in player_tracks(rd):
        ser = rd["tracks"][tid]
        c = box_at(ser, t_anchor)
        if c is None or abs(_nearest_dt(ser, t_anchor)) > 0.5:
            continue
        seen.append((c[1], tid))
    if len(seen) != 4:
        return {}
    seen.sort()
    return {tid: (i >= 2) for i, (_y, tid) in enumerate(seen)}


def pick_contact_track(dets, t, want_near, sides, tol=0.35):
    """Which track hit the contact at time t, GIVEN which side must
    have hit it.

    THE USER'S "VOTER ON EACH PLAYER" / ELIMINATION IDEA (2026-08-21),
    and the numbers asked for it: partner-given-team reached 86% while
    TEAM sat at 79% with ZERO whole-rally inversions — so team errors
    are individual contacts being credited to a track on the WRONG SIDE
    OF THE NET. The scorer's single best peak was taken on trust.

    Alternation already fixes the side of every contact exactly (0
    violations / 229 contacts). That makes two of the four players
    IMPOSSIBLE for this contact, which is elimination rather than
    preference: score every candidate, strike the impossible ones,
    and let the best survivor win. Falls back to the unconstrained
    best only when the legal side offers nothing at all, so a missing
    track degrades to the old behaviour instead of dropping a contact."""
    near = [d for d in dets if abs(d[0] - t) <= tol]
    if not near:
        return None, False
    legal = [d for d in near
             if d[3] in sides and sides[d[3]] == want_near]
    if legal:
        return max(legal, key=lambda d: d[2]), True
    return max(near, key=lambda d: d[2]), False


def player_tracks(rd, k=4, min_frac=0.25, static_frac=0.15):
    """The k tracks most likely to BE the four players.

    LENGTH IS PRIMARY, and that is an empirical finding, not a guess.
    The 2026-08-21 census showed four near-identical full-rally tracks
    in most rallies (r1 1789/1789/1789/1787, r2 1103x4, r6 929x4,
    r14 1380x4) — that shape IS the four players. Ranking by motion
    instead scored worse across the board (geometry 62% -> 53%,
    attribution 60% -> 54%) because it promoted SHORT high-motion
    fragments: r2 took a 242-sample track over a 1103, r5 a 215, r8 a
    289. Pieces of a player, or crowd.

    Motion survives only as a FLOOR, to drop the case it was introduced
    for: a referee or line judge who stands still through a whole rally
    and so outlasts the players. Static bodies are excluded; among what
    remains, longest wins."""
    items = [(tid, ser) for tid, ser in rd["tracks"].items()
             if len(ser["t"])]
    if not items:
        return []
    longest = max(len(ser["t"]) for _t, ser in items)
    live = [(tid, ser) for tid, ser in items
            if len(ser["t"]) >= min_frac * longest]
    if live:
        top_motion = max(track_motion(ser) for _t, ser in live)
        moving = [(tid, ser) for tid, ser in live
                  if track_motion(ser) >= static_frac * top_motion]
        if len(moving) >= k:
            live = moving
    live.sort(key=lambda kv: -len(kv[1]["t"]))
    return [tid for tid, _ser in live[:k]]


def anchor_time(rd, fallback, settle=True):
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
    base = max(firsts) if len(firsts) == 4 else fallback
    if not settle:
        return base
    # search back from the first CONTACT, but never earlier than the
    # moment all four are on screen
    hi = max(base, fallback)
    return max(base, settled_anchor(rd, hi, back=max(0.0, hi - base)))


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
def rally_end_motion(rd, t_open, step=0.2, tail=1.2, frac=0.45,
                     hold=1.0):
    """When the four players stop contesting the point.

    THE USER'S IDEA (2026-08-21), and it is the only uncontaminated
    channel available for this. The candidate stream cannot answer it:
    2529 candidates cover 148 contacts, so peaks are dense EVERYWHERE
    including dead time, and a gap rule over candidates never fires -
    the version that tried it made the window LOOSER (687 kept, up
    from 637). Confident candidates cannot answer it either, because
    between-point knocks score exactly as confidently as real shots.

    Player motion is a different instrument, on tracks measured at
    99%+, and rally-end is a whole-court state change rather than a
    per-frame judgement: during a point somebody is always reacting
    hard, and once it is over all four settle into walking.

    Self-calibrating, because absolute speed means nothing across
    zooms: the reference is this rally's OWN early motion, taken from
    the first seconds after the serve, which are in-play by
    construction. The rally is over at the last moment motion is still
    a real fraction of that, sustained for `hold` seconds so a single
    still frame mid-rally cannot end it early.

    Returns None when there are not four tracks - never a guess."""
    tids = player_tracks(rd)
    if len(tids) != 4:
        return None
    lo, hi = None, None
    for tid in tids:
        t = rd["tracks"][tid]["t"]
        if not len(t):
            continue
        lo = float(t[0]) if lo is None else min(lo, float(t[0]))
        hi = float(t[-1]) if hi is None else max(hi, float(t[-1]))
    if lo is None or hi is None or hi - lo < 2.0:
        return None
    t0 = max(lo, t_open)
    samples = []
    t = t0
    while t <= hi:
        sp = total_speed(rd, tids, t)
        if sp is not None:
            samples.append((t, sp))
        t += step
    if len(samples) < 10:
        return None
    early = [sp for t, sp in samples if t <= t0 + 3.0]
    if not early:
        return None
    early.sort()
    ref = early[len(early) // 2]
    if ref <= 0:
        return None
    thr = frac * ref
    # last instant still moving like a rally, requiring the quiet that
    # follows it to PERSIST - a momentary lull between shots is common
    # and must not be read as the end of the point
    last = None
    for i, (t, sp) in enumerate(samples):
        if sp < thr:
            continue
        quiet = [x for x in samples[i + 1:] if x[0] <= t + hold]
        if quiet and all(x[1] < thr for x in quiet):
            last = t
            break
        last = t
    return None if last is None else last + tail


def _strongest_first(cands, refractory):
    """swing_probe.strongest_first, inlined to keep passes 0-2 free of
    the pose stack. Strongest-wins, not first-wins: a small noise bump
    just before a real peak must not be allowed to eat it."""
    keep = []
    for c in sorted(cands, key=lambda x: -x[1]):
        if all(abs(c[0] - k[0]) >= refractory for k in keep):
            keep.append(c)
    keep.sort()
    return keep


def decode_passes(dets, s0, typical_gap, same_gap_p01, truth_ts=None,
                  report=None, floor=0.02, chain=True, gap_p99=2.1,
                  t_end=None):
    """Contact decoding as SEQUENTIAL PASSES, each one measurable.

    THE USER'S PROPOSAL (2026-08-21), and it is the right shape.
    decode_rally settles the window, the serve anchor, the chaining and
    the same-side repair SIMULTANEOUSLY inside one DP with hand-tuned
    constants, so when it emits junk there is no way to say which of
    those four decisions produced it. Split into passes and each one
    gets its own precision and recall, which is the only way to know
    where the 70 spurious events enter and where the 48 real ones are
    lost.

    NO PASS MAY READ TRUTH. truth_ts is used ONLY to score the funnel
    afterwards; every threshold here comes from the candidates
    themselves or from constants measured on the train labels and
    passed in. That separation is the whole point - a stage tuned on
    the answers would report a funnel that cannot be reproduced at
    inference.

    Returns (events, stages) where stages is [(name, kept)] for the
    report.
    """
    # Passes 0-2 are pure python ON PURPOSE. The geometry helpers in
    # this file already dropped numpy so the part worth unit-testing
    # runs without the whole pose stack, and the same applies here: the
    # cluster and window logic is where the reasoning lives, so it must
    # be testable on a machine with no model on it. Only pass 3 needs
    # swing_explore, and it is imported inside that branch.
    stages = []

    def snap(name, evs):
        stages.append((name, list(evs)))

    # ---- PASS 0: every scored peak. The ceiling: placement recall
    # already says a candidate sits within 0.35s of every true contact,
    # so no later pass can do better than what survives here.
    cur = [d for d in sorted(dets) if d[2] >= floor]
    snap("0 candidates", cur)

    # ---- PASS 1: collapse CLUSTERS. Dense scoring sprouts a burst of
    # peaks around one real swing. The merge window must sit under the
    # p01 of the true SAME-SIDE gap or it eats real contacts - that is
    # the constraint, and it is measured rather than assumed (the
    # shipped 0.55s was a guess).
    win = max(0.2, min(0.55, 0.8 * same_gap_p01))
    merged = []
    for side in (0, 1):
        side_c = [(t, sc) for t, sd, sc, _tid in cur if sd == side]
        keep_t = {round(t, 4) for t, _sc in
                  _strongest_first(side_c, win)}
        merged += [d for d in cur
                   if d[1] == side and round(d[0], 4) in keep_t]
    cur = sorted(merged)
    snap(f"1 cluster merge ({win:.2f}s)", cur)

    # ---- PASS 2: WINDOW. A rally runs from its serve to its last
    # contact; anything outside is dead time, and dead time is where
    # the VLM pilot's DEAD escape found junk sitting a median 5.01s
    # from any contact. The bounds come from the candidates' own
    # confident core, not from an external clock - the windows file's
    # t0s/t1s are on a different clock from the labels for r3+, so
    # trusting it would reintroduce a known misalignment.
    scs = sorted(d[2] for d in cur)
    conf = scs[int(0.70 * len(scs))] if scs else floor
    strong = [d for d in cur if d[2] >= conf]
    if strong:
        # the serve is the FIRST contact and its side is known, so the
        # window opens at the first confident candidate on that side
        srv = [d for d in strong if d[1] == s0]
        t_open = (srv[0][0] if srv else strong[0][0]) - 0.5 * typical_gap
        # THE RALLY ENDS AT THE FIRST IMPOSSIBLE GAP, not at the last
        # confident candidate. v1 closed the window on strong[-1] and
        # left 62 junk events past the last true contact, because
        # players knock the ball around between points and those
        # knocks score as confidently as real shots - the window's own
        # evidence is contaminated by exactly what it is meant to
        # exclude.
        #
        # The labels give a bound that dead time cannot fake: real
        # contacts are never more than ~2.07s apart (p99 of 286
        # measured gaps). Walk the confident candidates forward from
        # the serve and cut at the first gap that no rally could
        # contain. Junk sits a median 3.18s from any true contact, so
        # it is on the far side of that line by construction.
        # Walk EVERY candidate, not just the confident ones. The
        # selftest caught this: a weak-but-real contact does not extend
        # a strong-only walk, so the window closes on top of it and the
        # pass destroys a contact no later stage can recover. Junk
        # cannot bridge the gap either way - it sits a median 3.18s
        # from any true contact, past the cut by construction.
        cut = max(2.5, 1.2 * gap_p99)
        t_close = None
        prev = None
        for d in cur:
            if d[0] < t_open:
                continue
            if prev is not None and d[0] - prev > cut:
                t_close = prev + 0.5 * typical_gap
                break
            prev = d[0]
        if t_close is None:
            t_close = (prev if prev is not None
                       else strong[-1][0]) + 0.5 * typical_gap
        # MOTION WINS when it has an answer. The gap rule above is a
        # fallback and a weak one - it cannot see past candidate
        # density - while the players stopping is a direct observation
        # of the thing being asked.
        if t_end is not None:
            t_close = min(t_close, t_end)
        cur = [d for d in cur if t_open <= d[0] <= t_close]
    snap("2 window trim", cur)

    # ---- PASS 3: CHAIN. The alternating path, unchanged - this is the
    # pass decode_rally was actually good at (near-exact counts), now
    # fed a set that has already been de-clustered and trimmed instead
    # of doing all three jobs at once.
    if not chain:
        return cur, stages, 0
    import swing_explore as SE
    path = SE.decode_rally([(t, s, sc) for t, s, sc, _tid in cur], s0)
    tid_of = {(round(t, 3), s): tid for t, s, _sc, tid in cur}
    chained = [(t, s, sc, tid_of.get((round(t, 3), s)))
               for t, s, sc, _g in path]
    ghosts = sum(g for _t, _s, _sc, g in path)
    cur = chained
    snap(f"3 chain ({ghosts} ghosts)", cur)
    return cur, stages, ghosts


def same_side_policy(evs, path, mode, typical_gap):
    """What to do when two CONSECUTIVE emitted events share a side.

    THE USER'S QUESTION (2026-08-21), and both of their instincts are
    right in different cases. The shipped behaviour, `overwrite`, is
    the worst of the three: alternation_fix relabels the second event's
    SIDE to force alternation, which keeps a possibly-spurious event
    AND corrupts the one field we were most confident about.

    A same-side pair means one of exactly two things:
      DELETE  one of the two is a duplicate of the other - dense
              scoring sprouts peak clusters around a real swing, and
              two survivors of one swing sit much CLOSER than a real
              exchange. Drop the weaker; this is what reduces junk.
      INSERT  a real contact between them was missed, so the pair sits
              about TWO exchanges apart. Restoring it keeps the count
              honest, but a ghost carries no timestamp, so it can be
              charged to a SIDE and never to a player.

    The discriminator is the gap, measured against this footage's own
    typical inter-contact interval rather than a guessed constant - and
    the DP already votes: a ghost between the two events IS decode_rally
    asserting the second case. Returns (events, fixed_sides, n_deleted,
    n_inserted).
    """
    ghost_after = {}
    if path:
        for i, (_t, _s, _sc, g) in enumerate(path):
            ghost_after[i] = g
    keep, ins, dele = [], 0, 0
    for i, ev in enumerate(evs):
        if not keep or ev[1] != keep[-1][1]:
            keep.append(ev)
            continue
        # same side as the previous KEPT event
        dt = ev[0] - keep[-1][0]
        said_ghost = ghost_after.get(i, 0) > 0
        if mode == "insert" or (mode == "auto" and
                                (said_ghost or dt > 1.5 * typical_gap)):
            ins += 1
            keep.append(ev)          # parity restored by the ghost
        elif mode in ("delete", "auto"):
            dele += 1                # weaker of a cluster pair: the
            continue                 # later one, having lost the DP's
        else:                        # own strongest-first merge
            keep.append(ev)
    sides = [s for _t, s, _tid in keep]
    if mode == "overwrite":
        fixed, _ch = alternation_fix(sides, sides[0] if sides else 0)
        return keep, fixed, 0, 0
    return keep, sides, dele, ins


def geom_sides(rd, t_anchor):
    """{track_id: side} in the TRACKER's 0/1 space, but derived from
    image y instead of from the tracker's own side field.

    THE INTEGRATION the user asked for (2026-08-21), and it closes a
    real inconsistency. side_map's near/far split is the channel the
    permutation diagnosis measured at ZERO whole-rally errors, and
    label_tracks_at_serve already refuses to use ser["side"] because
    "the tracker's own side field is a frame-local split and the vision
    postmortem measured it corrupting 42% of FAR labels". Yet
    score_rally_tracked hands exactly that field to the DP as each
    detection's side - so the alternating chain, the spine of the whole
    decoder, is built on the channel we distrust while the reliable one
    is used only downstream for elimination.

    ser["side"] is worse than "a classification": it is
    int(side[m][0]), ONE frame's answer frozen for the track's whole
    life. A track that starts in a crowded or ambiguous frame carries
    that mistake through every contact it ever makes.

    The 0/1 labelling itself is arbitrary, so the near/far -> 0/1
    mapping is recovered by majority agreement with ser["side"] across
    the four player tracks. That works precisely because the field is
    wrong on a MINORITY of tracks; if it were wrong more than half the
    time the mapping would invert, which the caller can detect by the
    agreement fraction returned alongside."""
    near = side_map(rd, t_anchor)
    if not near:
        return {}, 0.0
    agree = sum(1 for tid, is_near in near.items()
                if rd["tracks"][tid]["side"] == int(is_near))
    flip = agree < len(near) / 2.0
    frac = max(agree, len(near) - agree) / len(near)
    return {tid: int(is_near) ^ int(flip)
            for tid, is_near in near.items()}, frac


def score_rally_tracked(model, rd, side_of=None):
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
        # SIDE FROM GEOMETRY WHEN OFFERED. A candidate stamped with
        # the wrong side is invisible to the alternating chain when it
        # is that side's turn (-> a ghost) and available when it is not
        # (-> junk), so one defect produces both symptoms the decoder
        # audit is chasing.
        _sd = ser["side"] if side_of is None else side_of.get(tid)
        if _sd is None:
            continue
        for tt, sc in SE.strongest_first(cands, SE.REFRACTORY_S):
            dets.append((tt, _sd, sc, tid))
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
BUILD = "2026-08-21u  MEASURE the rally-end detector against truth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--windows", default=WINDOWS_V4)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--pose-dir", default=POSE_DIR)
    ap.add_argument("--timeline-dir",
                    help="folder holding rally_timeline_<mid8>.csv, if "
                         "it is somewhere the search does not cover")
    ap.add_argument("--label", choices=["vote", "depth", "halves"],
                    default="vote",
                    help="vote (default) = majority of contact-order, "
                         "depth and halves on the one bit per side; "
                         "depth = server/receiver are the deep "
                         "players, names straight from the log, no "
                         "left/right; halves = the lineup R/L mapping, "
                         "which inverts when a serving team swaps ends "
                         "between points")
    ap.add_argument("--video",
                    help="enable the ball and approach voters: matches "
                         "tracked ball flights to each rally's first two "
                         "decoded contacts (72%% coverage measured)")
    ap.add_argument("--with-movement", action="store_true",
                    help="re-enable the raw-displacement voter, a "
                         "measured null (50%% on disputed calls) kept "
                         "only so the result stays reproducible")
    ap.add_argument("--cascade",
                    help="comma-separated voter precedence, e.g. "
                         "'contact,depth,halves'. The first voter that "
                         "fired decides; omit for a flat majority")
    ap.add_argument("--no-settle", action="store_true",
                    help="anchor when all four are first on screen, "
                         "instead of at the stillest instant before the "
                         "first contact (the pre-serve setup)")
    ap.add_argument("--diagonal", action="store_true",
                    help="cross-court serve constraint. MEASURED "
                         "HARMFUL 2026-08-21 (geometry 89%% -> 74%%); "
                         "off by default, kept only so the null "
                         "stays reproducible")
    ap.add_argument("--full", action="store_true",
                    help="print every diagnostic block. The default is "
                         "the headline numbers plus the failure trace; "
                         "the track census, order sweep, per-voter "
                         "tables, permutation diagnosis and voter "
                         "agreement are investigation tools, and "
                         "printing all of them every run made the "
                         "output hard to read or paste")
    ap.add_argument("--geom-side", action="store_true",
                    help="give the decoder its detection SIDES from "
                         "the image-y split (side_map) instead of the "
                         "tracker's frozen ser['side'], which the "
                         "naming layer already refuses to trust")
    ap.add_argument("--passes", action="store_true",
                    help="decode in SEQUENTIAL PASSES (candidates -> "
                         "cluster merge -> window trim -> chain) with "
                         "a per-stage funnel, instead of one DP that "
                         "settles all of it at once")
    ap.add_argument("--same-side",
                    choices=["overwrite", "delete", "insert", "auto"],
                    default="overwrite",
                    help="what to do with two consecutive emitted "
                         "events on the SAME side. overwrite (shipped) "
                         "relabels the side; delete drops the weaker "
                         "as a cluster duplicate; insert keeps both "
                         "and accepts a missed contact between them; "
                         "auto picks per pair on the gap and the DP's "
                         "own ghost")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    run(a)


def run(a):
    print(f"touch_attribute build {BUILD}")
    mode = getattr(a, "label", "vote")
    settle = not getattr(a, "no_settle", False)
    allow_movement = getattr(a, "with_movement", False)
    ball_by_rally = {}
    voter_tally = defaultdict(lambda: [0, 0])
    # ---- EVERY per-run accumulator, declared in ONE place above any
    # loop that fills it. Three separate UnboundLocalErrors in this
    # thread (side_frac, funnel, and this pattern again) all had the
    # same shape: a dict introduced next to the code that READS it
    # while the code that WRITES it runs earlier. --selftest cannot
    # catch it because it never enters run(), so the only defence is
    # keeping the declarations together and above everything.
    votes_by_rally = {}
    paths = {}
    side_frac = []
    cross_count = {}
    anchor_by_rally = {}
    end_audit = []
    funnel = {}
    extra_where = Counter()
    extra_dt = []
    t_ok = t_tot = 0
    n_deleted = n_inserted = 0
    team_of_player = {}
    diag_fixes = []
    picks_by_rally = {}
    ctx_by_rally = {}
    cascade = ([x.strip() for x in a.cascade.split(',')]
               if getattr(a, 'cascade', None) else None)

    def labeller(rd_, t_, rec_, nt_, fl_, nm_, events=None,
                 votes_out=None, ball_pts=None, picks_out=None):
        if mode == "vote":
            return label_by_vote(rd_, t_, rec_, nt_, fl_, nm_,
                                 events=events, tally=voter_tally,
                                 votes_out=votes_out, order=cascade,
                                 ball_pts=ball_pts,
                                 allow_movement=allow_movement,
                                 picks_out=picks_out,
                                 diagonal=a.diagonal,
                                 diag_out=diag_fixes)
        if mode == "depth":
            return label_by_depth(rd_, t_, rec_, nt_, fl_, nm_)
        return label_tracks_at_serve(rd_, t_, rec_, nt_, fl_, nm_)
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

    # ---- TIMING CONSTANTS FROM THE LABELS, not from guesses.
    # decode_rally's bands (min_gap 0.25, free 0.45-2.2, 0.55s
    # pre-merge) were never fitted to a real inter-contact interval,
    # and 148 labelled contacts have been sitting here the whole time.
    # These are the ONLY place the passes may see the labels: they are
    # constants measured once on TRAIN, exactly like a model's
    # hyperparameters, and no pass reads a rally's own answers. That
    # still makes any funnel measured on these same rallies in-sample -
    # the holdout in label_split.csv is the out-of-sample test and it
    # stays unburned.
    _all_gap, _same_gap = [], []
    for _c in truth:
        _ts = [x[0] for x in truth[_c]]
        _all_gap += [b - a_ for a_, b in zip(_ts, _ts[1:])]
        _same_gap += [b - a_ for a_, b in zip(_ts, _ts[2:])]
    _all_gap.sort()
    _same_gap.sort()
    pass_gap = _all_gap[len(_all_gap) // 2] if _all_gap else 0.8
    pass_same_p01 = (_same_gap[max(0, int(0.01 * len(_same_gap)))]
                     if _same_gap else 0.7)
    pass_gap_p99 = (_all_gap[min(len(_all_gap) - 1,
                                 int(0.99 * len(_all_gap)))]
                    if _all_gap else 2.1)

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
        # ONE ANCHOR PER RALLY, computed here and reused everywhere.
        # The decode loop used fallback 0.0 while the production path
        # used evs[0][0], so side_map could return DIFFERENT near/far
        # assignments in the two places - the decoder and the naming
        # layer silently disagreeing about which side is which. That is
        # what drove elimination re-assignments 7 -> 41 under
        # --geom-side and cost 14 points of attribution while every
        # other number improved.
        _t_anc = anchor_time(r["rd"], 0.0, not a.no_settle)
        anchor_by_rally[held] = _t_anc
        _side_of, _side_frac = (None, 0.0)
        if a.geom_side:
            _side_of, _side_frac = geom_sides(r["rd"], _t_anc)
            side_frac.append(_side_frac)
        dets = score_rally_tracked(model, r["rd"], _side_of or None)
        tid_of = {(round(t, 3), s): tid for t, s, _sc, tid in dets}
        _s0 = r["contacts"][0][1] ^ r["m"]
        if a.passes:
            # MEASURE THE DETECTOR, do not tune it blind. Four window
            # attempts have now moved the junk 70 -> 68 while the
            # product metric got worse, and not one of them was ever
            # compared against the thing it claims to find. The true
            # last contact is right here.
            _t_end = rally_end_motion(r["rd"], _t_anc)
            _true_last = max((c[0] for c in r["contacts"]), default=None)
            end_audit.append((held, _t_end, _true_last))
            _ev, _st, _gh = decode_passes(
                dets, _s0, pass_gap, pass_same_p01,
                gap_p99=pass_gap_p99,
                t_end=_t_end)
            path = [(t, sd, sc, 0) for t, sd, sc, _tid in _ev]
            funnel[held] = _st
        else:
            path = SE.decode_rally(
                [(t, s, sc) for t, s, sc, _ in dets], _s0)
        evs = []
        for t, s, _sc, _g in path:
            evs.append((t, s, tid_of.get((round(t, 3), s))))
        decoded[held] = evs
        # KEEP THE GHOSTS. decode_rally's 4th field is the number of
        # contacts it asserts happened between this event and the last
        # one but could not timestamp. Dropping it is not free: with
        # ONE ghost between two emitted events parity flips twice, so
        # the two emitted events are legitimately on the SAME side —
        # and alternation_fix then forces them apart, overwriting a
        # correct side with a wrong one. Ghosts are also the decoder's
        # own account of the contacts it knows it missed, which is the
        # first thing to check against the 48 unexplained true ones.
        paths[held] = path
        dets_by_rally[held] = dets

    # ---- ball flights per rally, matched to the first two decoded
    # contacts. Only contacts 0 and 1 are useful for the BINDING: the
    # log names the server and the receiver and nobody else, so a ball
    # point at any later contact identifies a track we already have and
    # a name we still do not.
    if getattr(a, "video", None):
        import ball_voter as BV
        for cum, evs in decoded.items():
            if len(evs) < 2:
                continue
            # WINDOW FROM THE CONTACTS, NOT FROM THE WINDOWS FILE.
            # v4's t0s/t1s are misaligned with the label times for
            # rallies 3+ (r3's window starts at 89.7s while its
            # contacts run 59.4-75.8s), which fed ffmpeg negative
            # durations and killed 29 of 30 ball lookups. The decoded
            # events are by construction where the contacts are.
            t_lo = min(e[0] for e in evs) - 2.0
            t_hi = max(e[0] for e in evs) + 2.0
            if t_hi - t_lo < 1.0:
                continue
            try:
                segs = BV.dedupe(BV.flight_segments(
                    a.video, max(0.0, t_lo), t_hi))
            except Exception as e:                 # noqa: BLE001
                print(f"ball: rally {cum} failed ({e})")
                continue
            # ---- CROSSING COUNT. An INDEPENDENT witness on the one
            # number that is wrong. Every other count in this pipeline
            # descends from the same pose decoder, so when it says 170
            # and truth says 148 there is nothing to check it against.
            # The ball is a separate instrument, and counting net
            # crossings asks it only what it is good at: a crossing
            # happens mid-flight, unoccluded, against open court -
            # never at the contact instant, which ball_voter exists
            # because it is the worst frame in the rally.
            _rd_c = rd_of(rallies, cum)
            _nl = BV.net_line(_rd_c, box_at, player_tracks, side_map,
                              anchor_time(_rd_c, evs[0][0], settle))
            if _nl is not None:
                _nx, _cts = BV.crossings(segs, _nl[0])
                cross_count[cum] = (_nx, len(segs), _nl)
            pts = {}
            for k in (0, 1):
                pt, _kind = BV.ball_at_contact(segs, evs[k][0])
                if pt is not None:
                    pts[k] = pt
            ball_by_rally[cum] = pts
        got = sum(len(v) for v in ball_by_rally.values())
        want = 2 * len(ball_by_rally)
        print(f"\nball: {got}/{want} binding contacts have a flight "
              f"endpoint across {len(ball_by_rally)} rallies")

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
        t_anc = anchor_time(rd_o, t0, settle)
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
    alt_changed = no_geom = unreadable = reassigned = 0
    serve_checked = serve_agree = 0
    # this footage's own inter-contact interval, not a constant
    _g = sorted(t2 - t1 for c in truth
                for t1, t2 in zip([x[0] for x in truth[c]],
                                  [x[0] for x in truth[c]][1:]))
    typical_gap = _g[len(_g) // 2] if _g else 0.8
    per_player = defaultdict(lambda: [0, 0])   # name -> [pipeline, truth]
    # name -> [right, wrong, extra]. See the TOUCH COUNTS block.
    per_player_kind = defaultdict(lambda: [0, 0, 0])
    for cum in sorted(decoded):
        w = wrows[cum]
        rec = lineup_by_rally.get(
            (w["match_id"], int(w["game"]), int(w["rally_in_game"])))
        evs = decoded[cum]
        if rec is None or not evs:
            continue
        rd = rd_of(rallies, cum)
        evs, fixed, _d, _i = same_side_policy(
            evs, paths.get(cum), a.same_side, typical_gap)
        n_deleted += _d
        n_inserted += _i
        if a.same_side == "overwrite":
            alt_changed += sum(
                1 for x, y in zip([s for _t, s, _tid in evs], fixed)
                if x != y)
        # LABEL ONCE AT THE SERVE, then ride the track ids: halves are
        # only defined at the serve, so a mid-rally switch must not be
        # allowed to rename anyone.
        nt = effective_near_team(
            near_team, epoch_of_score(rec.get("start_score", "")),
            ends_switch)
        _t_anc_p = anchor_by_rally.get(
            cum, anchor_time(rd, evs[0][0], settle))
        tnames, geom_ok = labeller(
            rd, _t_anc_p, rec, nt, flip,
            names_by_uuid, events=evs,
            ball_pts=ball_by_rally.get(cum))
        if not geom_ok:
            unreadable += 1
            continue
        # PRODUCTION PATH gets the same elimination: the decoded event
        # names a track, but alternation says which side must have hit,
        # so a track on the illegal side is re-picked from that side's
        # own candidates rather than trusted.
        sides_p = side_map(rd, _t_anc_p)
        called = []
        for (t, _s, tid), s_fix in zip(evs, fixed):
            want_near = (s_fix == 0)
            if sides_p and tid in sides_p and sides_p[tid] != want_near:
                alt, _c = pick_contact_track(
                    dets_by_rally.get(cum, []), t, want_near, sides_p)
                if alt is not None:
                    tid = alt[3]
                    reassigned += 1
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
        # WHERE A PLAYER'S SURPLUS COMES FROM. The 2026-08-21 run put
        # the pipeline at 170 contacts against 148 true, with Allyce
        # Jones absorbing +10 of the +22. A wrong side bit is ZERO-SUM
        # between partners - one loses exactly what the other gains -
        # so four simultaneously-positive deltas cannot be swaps, and
        # the aggregate delta column cannot tell the two apart.
        #
        # Greedy nearest-time matching, closest event to each true
        # contact first, so a genuine duplicate detection is charged as
        # EXTRA rather than silently standing in for the real one:
        #   right  matched a true contact and named it correctly
        #   wrong  matched a true contact and named the partner
        #   extra  matched no true contact - a detection with nothing
        #          under it, which is over-counting, not misattribution
        pairs = sorted(
            ((abs(t - tt), i, j) for i, (t, _s, _tid) in enumerate(evs)
             for j, (tt, _nt) in enumerate(truth[cum])
             if abs(t - tt) <= 0.35))
        used_e, used_t = set(), set()
        match = {}
        for _d, i, j in pairs:
            if i in used_e or j in used_t:
                continue
            used_e.add(i)
            used_t.add(j)
            match[i] = j
        _tt = [x[0] for x in truth[cum]]
        for i, nm in enumerate(called):
            if not nm:
                continue
            if i not in match:
                per_player_kind[nm][2] += 1
                # WHERE the junk is, which decides whether it is cheap
                # to remove. The VLM pilot's DEAD escape fired on 62
                # frames at a median 5.01s from any contact, so a large
                # share of it may simply be dead time - before the
                # serve or after the last contact - and window trimming
                # would take it for free. Junk INSIDE the rally is the
                # expensive kind: it needs the scorer or the DP.
                _te = evs[i][0]
                if _tt:
                    _d = min(abs(_te - x) for x in _tt)
                    where = ("before" if _te < _tt[0] - 0.35 else
                             "after" if _te > _tt[-1] + 0.35 else "mid")
                    extra_where[where] += 1
                    extra_dt.append(_d)
            elif truth[cum][match[i]][1] == nm:
                per_player_kind[nm][0] += 1
            else:
                per_player_kind[nm][1] += 1
        for _t, nm in truth[cum]:
            per_player[nm][1] += 1
        # team membership, straight from the lineup record - never
        # inferred from the counts
        for _tm in ("A", "B"):
            for _h in (RIGHT, LEFT):
                _n = names_by_uuid.get(
                    rec.get(f"team_{_tm}_{_h}", "").lower())
                if _n:
                    team_of_player[_n] = _tm
        # grade: k-th call vs k-th truth (same order-join as the pilot)
        for k, nm in enumerate(called):
            if nm is None or k >= len(truth[cum]):
                continue
            tot += 1
            ok += nm == truth[cum][k][1]
        # ...and again by NEAREST TIME, which is the question touch
        # share actually asks: for each TRUE contact, who hit it?
        #
        # The index join above is the pilot's, kept for continuity, but
        # it conflates naming with counting: the pipeline emits 170
        # events against 148 true contacts, and ONE spurious detection
        # early in a rally shifts every later comparison by one, so the
        # rest of that rally grades as noise however well it was named.
        # Reported side by side deliberately - the difference between
        # the two IS the over-counting, and hiding it inside a single
        # improved number would launder a measurement change as a gain.
        for i, nm in enumerate(called):
            if nm is None or i not in match:
                continue
            t_tot += 1
            t_ok += nm == truth[cum][match[i]][1]

    # FREEZE THE HEADLINE. ok/tot are finished here, and everything
    # below is diagnostics. An order-sweep loop rebinding `ok` once
    # made ATTRIBUTION print 21/118 instead of 85/118 with every other
    # number on the page unchanged - the most dangerous shape of bug
    # this file can have, since it looks exactly like a regression.
    # Diagnostics must not be able to move a result; snapshot, and the
    # assert makes any future rebinding loud instead of plausible.
    attr_ok, attr_tot = ok, tot

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
    print(f"\nTRUTH-ANCHORED GEOMETRY TEST — labeller={a.label}, "
          f"anchor={'settled' if settle else 'first-coexist'} "
          f"(decoder placement removed: true contact times)")
    g_ok = g_n = miss = g_team = g_partner = 0
    per_rally_geom = defaultdict(lambda: [0, 0])
    g_unc = [0, 0]
    n_constrained = 0
    truth_vote = defaultdict(lambda: defaultdict(Counter))
    label_of = {}
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
        vout, pout = {}, {}
        tnames, ok_lab = labeller(
            rd, anchor_time(rd, dets_by_rally[cum][0][0]
                            if dets_by_rally[cum] else 0.0, settle),
            rec, nt, flip, names_by_uuid, events=decoded.get(cum),
            votes_out=vout, ball_pts=ball_by_rally.get(cum),
            picks_out=pout)
        votes_by_rally[cum] = vout
        picks_by_rally[cum] = pout
        ctx_by_rally[cum] = {
            "score": rec.get("start_score", ""),
            "srv_team": rec.get("server_team"),
            "near": nt,
            "srv_half": rec.get("server_half"),
        }
        sel = player_tracks(rd)
        t_anc_c = anchor_time(rd, dets_by_rally[cum][0][0]
                              if dets_by_rally[cum] else 0.0, settle)
        # y AT THE ANCHOR is what the 2/2 near/far split actually rests
        # on, so print it: a clean split shows two low and two high
        # values with a wide gap, and an ambiguous one shows them
        # interleaved. Inferring this from accuracy alone is what led
        # to a wrong motion-based "fix".
        ys = []
        for t in sel:
            c = box_at(rd["tracks"][t], t_anc_c)
            ys.append(None if c is None else round(c[1]))
        stats = [(len(rd["tracks"][t]["t"]),
                  round(track_motion(rd["tracks"][t])), y)
                 for t, y in zip(sel, ys)]
        # ALL tracks at the anchor, not just the chosen four. The
        # census could not show a DROPPED PLAYER, which is the failure
        # the user described (2026-08-21): the far receiver stands
        # several feet behind the baseline to take serve, and if the
        # detector loses them there the four-track selection quietly
        # substitutes a referee or the near kitchen player. That breaks
        # the 2/2 near/far split, leaves the serving pair without its
        # server so `contact` cannot fire at all, and lets both
        # geometry voters agree on the same wrong answer - which is
        # exactly r16's signature.
        allt = []
        for tid, ser in rd["tracks"].items():
            c = box_at(ser, t_anc_c)
            allt.append((None if c is None else round(c[1]),
                         len(ser["t"]), tid in sel))
        allt.sort(key=lambda r: (r[0] is None, r[0]))
        census.append((cum, len(rd["tracks"]), stats, ok_lab, allt))
        if not ok_lab:
            continue
        label_of[cum] = dict(tnames)
        # side each contact MUST be on: the serving team hits contact 1,
        # and alternation fixes every one after it.
        srv_is_near_c = (rec.get("server_team") == nt)
        sides_c = side_map(rd, anchor_time(
            rd, dets_by_rally[cum][0][0] if dets_by_rally[cum] else 0.0,
            settle))
        for k_c, (t_true, nm_true) in enumerate(truth[cum]):
            want_near = srv_is_near_c if (k_c % 2 == 0) else \
                (not srv_is_near_c)
            best_u = None
            near = [d for d in dets_by_rally[cum]
                    if abs(d[0] - t_true) <= 0.35]
            if near:
                best_u = max(near, key=lambda d: d[2])
            best, constrained = pick_contact_track(
                dets_by_rally[cum], t_true, want_near, sides_c)
            if best is None:
                miss += 1
                continue
            if best_u is not None and tnames.get(best_u[3]) is not None:
                g_unc[1] += 1
                g_unc[0] += tnames.get(best_u[3]) == nm_true
            if constrained:
                n_constrained += 1
            nm = tnames.get(best[3])
            if nm is None:
                miss += 1
                continue
            g_n += 1
            g_ok += nm == nm_true
            per_rally_geom[cum][1] += 1
            per_rally_geom[cum][0] += nm == nm_true
            # TRUTH VOTE PER TRACK: which player does this track really
            # belong to? Truth-anchored, so it needs no geometry at all.
            truth_vote[cum][best[3]][nm_true] += 1
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
        if g_unc[1]:
            print(f"    [ablation] same test WITHOUT the alternation "
                  f"side constraint: {g_unc[0]}/{g_unc[1]} = "
                  f"{g_unc[0] / g_unc[1]:.0%}")
            print(f"    the constraint was applicable on "
                  f"{n_constrained} contacts (elsewhere it fell back "
                  f"to the unconstrained best)")
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
    if a.full:
        # track census — investigation tool, not a headline
        print("\nTRACK CENSUS — per rally: geometry hits/total, then the "
              "SELECTED four as\n  (samples, motion, y_at_anchor), then "
              "EVERY track at the anchor as y*len, with\n  * marking the "
              "selected ones. At a serve the formation is one deep player "
              "and\n  one kitchen player PER SIDE, so expect roughly two "
              "low y and two high y —\n  a DROPPED far player shows up as "
              "an unselected deep track, or as no track\n  at all above "
              "the far kitchen band.")
        for cum, n_tr, durs, ok_lab, allt in census:
            gk, gt = per_rally_geom[cum]
            acc = f"{gk}/{gt}" if gt else "-"
            print(f"  r{cum:<4} tracks {n_tr:<3} geom {acc:>7}  "
                  f"sel(len,motion,y) {durs}  "
                  f"{'ok' if ok_lab else 'UNREADABLE'}")
            cells = []
            for y, ln, is_sel in allt:
                ytxt = "--" if y is None else str(y)
                cells.append(f"{'*' if is_sel else ' '}{ytxt}x{ln}")
            print(f"        all: {'  '.join(cells)}")

    # ---- PERMUTATION DIAGNOSIS. The per-rally geometry rates are
    # BIMODAL (r2 91%, r5 93%, r13 100% against r14 27%, r4 38%,
    # r6 43%), which is what a per-rally BINARY FLIP looks like and
    # not what noise looks like — and averaging it into 79%/79% hides
    # that completely. So name the flip: compare the geometric labels
    # against each track's voted true identity and say whether the
    # rally is correct, has its two PARTNERS swapped (a stack, or a
    # wrong left/right read), has its two TEAMS swapped (near/far
    # inverted), or is genuinely scrambled.
    # ---- VOTER vs TRUTH. The earlier tally compared each voter to the
    # ensemble's own call, which is CIRCULAR: a voter that dominates the
    # majority scores high by construction, so it measured
    # self-consistency and not accuracy. This scores every voter against
    # the truth-voted identity of the track it picked, and separately on
    # the DISPUTED cases — which is the table a cascade's precedence has
    # to be earned from.
    vt = defaultdict(lambda: [0, 0])
    vt_disp = defaultdict(lambda: [0, 0])
    for cum, vout in votes_by_rally.items():
        real = {tid: v.most_common(1)[0][0]
                for tid, v in truth_vote[cum].items() if v}
        if not real or not vout:
            continue
        for who, votes in vout.items():
            truth_tid = next((t for t, nm in real.items() if nm == who),
                             None)
            if truth_tid is None:
                continue
            disputed = len({t for _v, t in votes}) > 1
            for vname, tid in votes:
                vt[vname][1] += 1
                vt[vname][0] += tid == truth_tid
                if disputed:
                    vt_disp[vname][1] += 1
                    vt_disp[vname][0] += tid == truth_tid
    if vt:
        print("\nVOTER ACCURACY vs TRUTH (not vs the ensemble — the "
              "earlier tally was circular)")
        for v in sorted(vt):
            ok_v, n_v = vt[v]
            d_ok, d_n = vt_disp[v]
            d = f"   on disputed calls {d_ok}/{d_n} = {d_ok / d_n:.0%}" \
                if d_n else "   never disputed"
            print(f"  {v:<9} {ok_v}/{n_v} = {ok_v / n_v:>4.0%}{d}")
        print("  Precedence for --cascade should follow the DISPUTED "
              "column: that is the\n  only place a voter's ordering "
              "changes any answer.")

    # ---- SIDE-BIT FAILURE TRACE. The aggregate tables say WHICH
    # voter is unreliable; they cannot say what went wrong in a given
    # rally, and the 2026-08-21 run showed 16 of 17 geometry errors
    # concentrated in 5 rallies diagnosed as whole-side inversions.
    # That shape is a handful of BITS, not per-contact noise, so the
    # useful view is one line per BROKEN BIT with every voter's call
    # beside the truth. Silence is printed explicitly ("-") because an
    # abstention and a wrong answer are different failures: the first
    # is fixed by widening a voter, the second by demoting it.
    fails = []
    for cum in sorted(votes_by_rally):
        real = {tid: v.most_common(1)[0][0]
                for tid, v in truth_vote[cum].items() if v}
        if not real:
            continue
        for who, votes in votes_by_rally[cum].items():
            truth_tid = next((t for t, nm in real.items() if nm == who),
                             None)
            _pk = picks_by_rally.get(cum, {}).get(who)
            picked = _pk[0] if _pk else None
            # contact 0 is the serve, contact 1 the return, so the bit's
            # parity IS its role. Every error is one of these two by
            # construction: the cascade decides only the server and the
            # receiver, and alternation carries the rest of the rally.
            role = ("server", "receiver")[_pk[1] % 2] if _pk else "?"
            if truth_tid is None or picked is None or picked == truth_tid:
                continue
            fails.append((cum, who, truth_tid, picked, votes, role))
    if fails:
        print("\nSIDE-BIT FAILURE TRACE (only the bits that came out "
              "wrong; * = this voter had it right)")
        allv = sorted({v for f in fails for v, _ in f[4]}
                      | set(cascade or []))
        for cum, who, truth_tid, picked, votes, role in fails:
            byname = {v: t for v, t in votes}
            cx = ctx_by_rally.get(cum, {})
            cells = []
            for v in allv:
                t = byname.get(v)
                if t is None:
                    cells.append(f"{v}=-")
                else:
                    cells.append(f"{v}={t}{'*' if t == truth_tid else ''}")
            print(f"  r{cum:<4} {who:<22} truth=track {truth_tid} "
                  f"picked={picked}  [score {cx.get('score','?')} "
                  f"srv={cx.get('srv_team')} near={cx.get('near')} "
                  f"srvhalf={cx.get('srv_half')}]")
            print(f"        {'  '.join(cells)}")
        rescuable = sum(1 for f in fails
                        if any(x == f[2] for _v, x in f[4]))
        print(f"  {rescuable} of {len(fails)} broken bits had SOME voter "
              f"right (reachable by re-ordering);\n  {len(fails) - rescuable} "
              f"had every voter wrong or silent (needs a new witness, "
              f"not a new order).")

    if a.full:
        # order sweep + per-voter + veto + leave-one-out — investigation tool, not a headline
        # ---- ORDER SWEEP (user question, 2026-08-21): are some voter
        # permutations better than others, and is any voter OVERRIDING or
        # DILUTING a better one while still carrying real signal?
        #
        # This is answerable exactly and for free. pick() computes every
        # voter's call BEFORE applying the order, so the recorded vote
        # lists do not depend on the cascade at all — every permutation can
        # be replayed offline against the same recorded truth. No re-run,
        # no video, no API.
        #
        # THREE DIFFERENT PATHOLOGIES, deliberately not pooled:
        #   OVERRIDE   the voter decided and was wrong while a LOWER-ranked
        #              voter had it right. Pure ordering damage — free to
        #              fix, and invisible in any overall accuracy column.
        #   DILUTION   only exists under majority: a wrong voter pulls the
        #              count off a correct one. Cascade cannot dilute, so
        #              the majority baseline below is what measures it.
        #   BURIED     accurate WHEN IT DECIDES but ranked too low ever to
        #              decide. This is why accuracy-when-deciding, not
        #              overall accuracy, is the statistic precedence should
        #              be earned from: a voter is only consulted where
        #              everything above it stayed silent, so its overall
        #              rate is measured on the wrong population.
        #
        # Bits are also weighted by CONTACTS RENAMED. A side bit flips the
        # names of both players on one side for the whole rally, so under
        # alternation it costs every contact of that parity — a bit in a
        # 25-contact rally is not worth the same as one in a 2-contact
        # rally, and unweighted bit counts hide that.
        bits = []
        for cum in sorted(votes_by_rally):
            real = {tid: v.most_common(1)[0][0]
                    for tid, v in truth_vote[cum].items() if v}
            if not real:
                continue
            for who, votes in votes_by_rally[cum].items():
                truth_tid = next((t for t, nm in real.items() if nm == who),
                                 None)
                meta = picks_by_rally.get(cum, {}).get(who)
                if truth_tid is None or meta is None:
                    continue
                par = meta[1] % 2
                w = sum(1 for k in range(len(truth[cum])) if k % 2 == par)
                bits.append((cum, who, truth_tid,
                             {v: t for v, t in votes}, w))

        def _decide(order, by):
            return decide_by_order(order, by)

        def _run(order):
            return score_order(order, bits)

        if bits:
            names = sorted({v for _c, _w, _t, by, _x in bits for v in by})
            print(f"\nORDER SWEEP — {len(bits)} side bits, "
                  f"{sum(w for *_r, w in bits)} contacts at stake, "
                  f"{len(names)} voters ({math.factorial(len(names))} "
                  f"orderings)")
            scored = []
            for perm in itertools.permutations(names):
                # NOTE THE UNDERSCORES, they are load-bearing. `ok` and `n`
                # are run()'s ATTRIBUTION counters, and an earlier version
                # of this loop rebound them - so ATTRIBUTION reported the
                # last permutation's bit score (21/118) rather than the
                # real 85/118, with every other number on the page
                # unchanged. A diagnostic must never be able to move a
                # headline result; see the snapshot at the print site.
                _ok, _n, _wok, _wn = score_order(perm, bits)
                scored.append((_wok, _ok, perm))
            scored.sort(key=lambda r: (-r[0], -r[1]))
            _wn_tot = sum(w for *_r, w in bits)
            # HOW MANY ORDERINGS ACTUALLY DIFFER. Most permutations are
            # observationally identical here: reordering voters that never
            # fire on the same bit changes nothing. Printing the count of
            # DISTINCT outcomes keeps "the best of 720" from sounding like
            # 720 independent chances to win.
            distinct = len({r[0] for r in scored})
            best_w = scored[0][0]
            n_best = sum(1 for r in scored if r[0] == best_w)
            cur = tuple(cascade) if cascade else None
            print(f"  {distinct} distinct outcomes across all orderings; "
                  f"{n_best} orderings tie for best")
            print("  best orderings (contact-weighted, then bits):")
            seen_p = set()
            for _wok, _ok, perm in scored[:400]:
                key = (_wok, _ok)
                if key in seen_p:
                    continue
                seen_p.add(key)
                if len(seen_p) > 6:
                    break
                print(f"    {_wok:>4}/{_wn_tot} contacts  "
                      f"{_ok:>2}/{len(bits)} bits   {','.join(perm)}")
            if cur:
                ok_c, n_c, wok_c, wn_c = score_order(cur, bits)
                rank = 1 + sum(1 for r in scored if r[0] > wok_c)
                print(f"  CURRENT --cascade {','.join(cur)}: "
                      f"{wok_c}/{wn_c} contacts, {ok_c}/{n_c} bits "
                      f"(rank {rank} of {len(scored)})")
            # MAJORITY BASELINE — the dilution test. Cascade cannot dilute;
            # if majority scores worse, the gap IS dilution.
            ok_m = wok_m = 0
            for _cum, _who, truth_tid, by, w in bits:
                if not by:
                    continue
                got = Counter(by.values()).most_common(1)[0][0]
                ok_m += got == truth_tid
                wok_m += w * (got == truth_tid)
            print(f"  MAJORITY of all voters (no precedence): "
                  f"{wok_m}/{_wn_tot} contacts, {ok_m}/{len(bits)} bits "
                  f"— the gap to the best cascade is DILUTION")

            # ---- PER-VOTER: the three pathologies, under the CURRENT order
            best_perm = scored[0][2]
            for tag, order in (("current", cur), ("best", best_perm)):
                if not order:
                    continue
                dec = defaultdict(lambda: [0, 0])      # decided / right
                over = defaultdict(int)                # wrong, someone below right
                solo = defaultdict(int)                # right, everyone else wrong
                for _cum, _who, truth_tid, by, w in bits:
                    got, v = _decide(order, by)
                    if v is None:
                        continue
                    dec[v][0] += 1
                    dec[v][1] += got == truth_tid
                    others = [t for k, t in by.items() if k != v]
                    if got != truth_tid and any(t == truth_tid
                                                for t in others):
                        over[v] += 1
                    if got == truth_tid and others and not any(
                            t == truth_tid for t in others):
                        solo[v] += 1
                print(f"\n  PER-VOTER under the {tag} order "
                      f"({','.join(order)})")
                print(f"    {'voter':<10}{'fires':>6}{'decides':>9}"
                      f"{'right when deciding':>21}{'overrides':>11}"
                      f"{'sole rescue':>13}")
                for v in names:
                    fires = sum(1 for b in bits if v in b[3])
                    d, r = dec[v]
                    rate = f"{r}/{d} = {r / d:.0%}" if d else "never decides"
                    print(f"    {v:<10}{fires:>6}{d:>9}{rate:>21}"
                          f"{over[v]:>11}{solo[v]:>13}")
                print("    overrides = decided WRONG while a voter below it "
                      "had the answer (free to fix)")
                print("    sole rescue = decided RIGHT when every other "
                      "voter was wrong (irreplaceable)")

            # ---- VETO GRID. The trace showed the binding failure is not
            # an ordering problem at all: `contact` led and was wrong on 3
            # of the 4 broken bits with a lower voter right each time, yet
            # demoting it costs the 21 bits it gets right. A cascade cannot
            # express "usually trust it, but not when the others gang up",
            # so no permutation can fix that - which is exactly why this
            # grid is scored beside the sweep rather than instead of it.
            print("\n  VETO GRID — leader stands unless the voters below it "
                  "overturn it\n    (k = minimum dissenters; unan = they must "
                  "be unanimous, maj = strict majority)")
            for tag, order in (("current", cur), ("best", best_perm)):
                if not order:
                    continue
                row = []
                for mode in ("unan", "maj"):
                    for k in (2, 3):
                        okv, _n, wv, _w = score_order(order, bits, k, mode)
                        row.append(f"{mode}/k{k} {wv:>4}c {okv:>2}b")
                base_ok, _n, base_w, _w = score_order(order, bits)
                print(f"    {tag:<8} no veto {base_w:>4}c {base_ok:>2}b   "
                      + "   ".join(row))

            # ---- LEAVE-ONE-OUT: is any voter net harmful at ANY position?
            print("\n  LEAVE-ONE-OUT (best achievable with this voter "
                  "removed entirely)")
            full_best = scored[0][0]
            for v in names:
                rest = [x for x in names if x != v]
                if not rest:
                    continue
                b = max(_run(pm)[2] for pm in itertools.permutations(rest))
                flag = "  <- removing it HELPS" if b > full_best else ""
                print(f"    without {v:<10} {b:>4}/{_wn_tot} contacts "
                      f"(best with all = {full_best}){flag}")
            print("  NOTE ON SELECTION: with "
                  f"{len(bits)} bits, argmax over {len(scored)} orderings "
                  "overfits. Prefer the\n  ordering justified by the "
                  "right-when-deciding and overrides columns, and treat "
                  "ties as ties.")

    if a.full:
        # permutation diagnosis + voter agreement — investigation tool, not a headline
        print("\nPERMUTATION DIAGNOSIS (geometric label vs each track's "
              "voted true identity)")
        kinds = Counter()
        for cum in sorted(label_of):
            got = label_of[cum]
            real = {tid: v.most_common(1)[0][0]
                    for tid, v in truth_vote[cum].items() if v}
            shared = [t for t in real if t in got]
            if not shared:
                continue
            exact = sum(got[t] == real[t] for t in shared)
            teams_got = {t: name_team.get(got[t]) for t in shared}
            teams_real = {t: name_team.get(real[t]) for t in shared}
            team_ok = sum(teams_got[t] == teams_real[t] for t in shared)
            n = len(shared)
            if exact == n:
                kind = "correct"
            elif team_ok == n:
                kind = "PARTNERS SWAPPED (left/right)"
            elif team_ok == 0:
                kind = "TEAMS SWAPPED (near/far)"
            else:
                kind = "mixed/scrambled"
            kinds[kind] += 1
            print(f"  r{cum:<4} {n} tracks matched, {exact} named right, "
                  f"{team_ok} on the right team -> {kind}")
        print("  " + ", ".join(f"{k}: {v}" for k, v in kinds.most_common()))
        print("  A large PARTNERS-SWAPPED count means the left/right read "
              "is inverted per rally\n  (stacking, or the anchor being read "
              "before the players settle). A large\n  TEAMS-SWAPPED count "
              "means the orientation bits are wrong for those rallies.")

        if voter_tally:
            print("\nVOTER AGREEMENT with the ensemble's final call "
                  "(a voter near 50% is dead weight; near 100% means it is "
                  "deciding)")
            for v, (ok_v, n_v) in sorted(voter_tally.items()):
                if n_v:
                    print(f"  {v:<9} {ok_v}/{n_v} = {ok_v / n_v:.0%}")

        if a.diagonal:
            print(f"\nDIAGONAL repaired {len(diag_fixes)} side bits "
                  f"(server and receiver must be cross-court; a violation "
                  f"means one bit is wrong,\n  and the less reliable "
                  f"decider is flipped) — deciders involved: "
                  f"{Counter(d for pair in diag_fixes for d in pair).most_common()}")
    print(f"\nSAME-SIDE POLICY '{a.same_side}': deleted "
          f"{n_deleted} cluster duplicates, kept {n_inserted} pairs "
          f"with a missed contact between them\n  (typical true "
          f"inter-contact gap {typical_gap:.2f}s, measured from the "
          f"labels)")
    print(f"\nALTERNATION overwrote {alt_changed} decoded sides "
          f"(tracker/decoder disagreements with the exact constraint)")
    print(f"rallies skipped, serve geometry unreadable: {unreadable} "
          f"(a side did not show exactly two tracks)")
    print(f"events on an unlabelled track: {no_geom}")
    print(f"contacts re-assigned to the legal side by elimination: "
          f"{reassigned}")
    if serve_checked:
        print(f"SERVE CHECK: {serve_agree}/{serve_checked} = "
              f"{serve_agree / serve_checked:.0%} of rallies had the "
              f"serve track already carrying the logged server's name "
              f"BEFORE the anchor was applied\n  (this is the honest "
              f"read on the whole geometry chain — orientation, side "
              f"and left/right — measured for free on every rally)")
    assert (ok, tot) == (attr_ok, attr_tot), (
        "a diagnostic below the grading loop rebound the attribution "
        f"counters: {(ok, tot)} != {(attr_ok, attr_tot)}")
    if attr_tot:
        print(f"\nATTRIBUTION (no API, no appearance model): "
              f"{attr_ok}/{attr_tot} = {attr_ok / attr_tot:.0%}")
        print(f"  VLM comparison on the same rallies: identity 44%, "
              f"side 70% (2026-08-21, $2.59)")
    if t_tot:
        print(f"  by NEAREST-TIME join (who hit each TRUE contact — the "
              f"touch-share question): {t_ok}/{t_tot} = {t_ok / t_tot:.0%}")
        print(f"  the gap between the two joins is OVER-COUNTING, not "
              f"naming: one spurious\n  detection shifts every later "
              f"index comparison in that rally")
    # ---- PASS FUNNEL. One row per stage: how many events it holds,
    # how many TRUE contacts still have something within 0.35s
    # (recall — the ceiling every later stage inherits), and, once the
    # set is a chain rather than a candidate pool, how many of its
    # events land on a true contact (precision).
    #
    # Candidate stages have terrible precision BY CONSTRUCTION — a pool
    # of peaks is not a claim about what happened — so precision is
    # printed only where it means something. What matters in the early
    # rows is that RECALL does not fall: a stage that drops recall is
    # destroying real contacts, and nothing downstream can recover
    # them.
    if funnel:
        names = [n for n, _k in next(iter(funnel.values()))]
        print("\nPASS FUNNEL (sequential decode)")
        print(f"    {'stage':<26}{'events':>8}{'recall':>9}"
              f"{'precision':>11}")
        for gi, nm in enumerate(names):
            n_ev = tp = 0
            covered = total_true = 0
            for cum, stg in funnel.items():
                kept = stg[gi][1]
                ts = [d[0] for d in kept]
                n_ev += len(kept)
                tru = [x[0] for x in truth.get(cum, [])]
                total_true += len(tru)
                covered += sum(1 for x in tru
                               if any(abs(x - t) <= 0.35 for t in ts))
                used = set()
                for t in ts:
                    hit = [j for j, x in enumerate(tru)
                           if abs(x - t) <= 0.35 and j not in used]
                    if hit:
                        used.add(hit[0])
                        tp += 1
            rc = f"{covered}/{total_true}" if total_true else "-"
            pr = (f"{tp}/{n_ev} = {tp / n_ev:.0%}"
                  if gi >= len(names) - 1 and n_ev else "")
            print(f"    {nm:<26}{n_ev:>8}{rc:>10}{pr:>15}")
        print("    recall is the ceiling every later stage inherits — a "
              "stage that drops it is\n    destroying real contacts, "
              "and nothing downstream can get them back.")

    if end_audit:
        print("\nRALLY-END DETECTOR vs the true last contact")
        print("  positive = the window closes AFTER the last real "
              "contact (junk survives);\n  negative = it closes BEFORE "
              "it (real contacts destroyed, which nothing downstream "
              "can undo)")
        _ab = sum(1 for _c, e, _t in end_audit if e is None)
        _d = [(e - t) for _c, e, t in end_audit
              if e is not None and t is not None]
        for cum, e, t in end_audit:
            if e is None:
                print(f"    r{cum:<5} ABSTAINED (fell back to the gap "
                      f"rule)")
            else:
                print(f"    r{cum:<5} end {e:7.2f}s   last true contact "
                      f"{t:7.2f}s   {e - t:+6.2f}s")
        if _d:
            _ds = sorted(_d)
            print(f"  fired on {len(_d)}/{len(end_audit)} rallies "
                  f"({_ab} abstained); overshoot med "
                  f"{_ds[len(_ds) // 2]:+.2f}s, min {_ds[0]:+.2f}s, "
                  f"max {_ds[-1]:+.2f}s")
            print(f"  a large positive median means the threshold is "
                  f"too LOW — players walking between\n  points still "
                  f"clear it, so the point never looks over")

    # ---- SIDE CHANNEL AUDIT. Placement recall says a scored
    # candidate sits within 0.35s of EVERY true contact, yet the DP
    # emits ghosts — contacts it asserts but cannot place. Those two
    # facts can only coexist if something between the candidate pool
    # and the DP is losing candidates, and the side stamp is the prime
    # suspect: the chain can only ever consume a candidate whose side
    # matches whose turn it is.
    #
    # Measured directly. For each true contact, find the nearest
    # candidate and ask whether the side the DP SEES equals the side
    # truth says it was (truth_side ^ m, m being the rally's mirror
    # flag). Wrong here means the chain could not use that candidate
    # even though it existed, and had to ghost the contact instead —
    # while the same candidate sat available at the wrong turn as junk.
    side_ok = side_n = side_none = 0
    for cum, r in rallies.items():
        ds = dets_by_rally.get(cum, [])
        if not ds:
            continue
        for _c in r["contacts"]:
            t_true, s_true = _c[0], _c[1]   # (t, team, shot_type)
            near = [d for d in ds if abs(d[0] - t_true) <= 0.35]
            if not near:
                side_none += 1
                continue
            best = min(near, key=lambda d: abs(d[0] - t_true))
            side_n += 1
            side_ok += (best[1] == (s_true ^ r["m"]))
    if side_n:
        print(f"\nSIDE CHANNEL AUDIT — can the chain even USE the "
              f"candidate that is there?")
        print(f"  the nearest candidate to a true contact carries the "
              f"RIGHT side on {side_ok}/{side_n} = "
              f"{side_ok / side_n:.0%} of contacts"
              f"{f' ({side_none} had no candidate at all)' if side_none else ''}")
        print(f"  a wrong side is invisible to the chain when it is "
              f"that side's turn (-> ghost) and\n  available when it "
              f"is not (-> junk): ONE defect, both symptoms.")
        if side_frac:
            _sf = sum(side_frac) / len(side_frac)
            print(f"  --geom-side is ON: near/far -> 0/1 mapping "
                  f"recovered with mean {_sf:.0%} track agreement "
                  f"(below 50% would mean the mapping inverted)")
        else:
            print(f"  currently using the tracker's frozen ser['side'] "
                  f"— try --geom-side to feed the chain the image-y "
                  f"split instead")

    # ---- DECODER AUDIT. The event list is the binding constraint (70
    # of 170 emitted are spurious, 48 of 148 true contacts carry no
    # event), and placement recall says a scored DETECTION sits within
    # 0.35s of every true contact — so the candidates are there and
    # decode_rally is not choosing them. Before touching a single
    # constant, measure what the DP is assuming against what the labels
    # actually contain.
    #
    # decode_rally's timing model is hand-guessed: min_gap 0.25,
    # max_gap 3.0, and a gap bonus of 0 inside 0.45-2.2s, -1.2 in
    # 0.3-0.45, -3.0 outside. Nothing was ever fitted to a real
    # inter-contact interval, and we have 148 labelled contacts.
    gaps, gaps_same = [], []
    for cum in sorted(truth):
        ts = [t for t, _nm in truth[cum]]
        for i in range(1, len(ts)):
            gaps.append(ts[i] - ts[i - 1])
        for i in range(2, len(ts)):
            gaps_same.append(ts[i] - ts[i - 2])
    if gaps:
        gs = sorted(gaps)
        def q(v, f):
            return v[min(len(v) - 1, int(f * len(v)))]
        print(f"\nDECODER AUDIT — the DP's timing model vs the labels")
        print(f"  TRUE inter-contact gap (n={len(gs)}): "
              f"p01 {q(gs, .01):.2f}  p05 {q(gs, .05):.2f}  "
              f"p25 {q(gs, .25):.2f}  med {q(gs, .50):.2f}  "
              f"p75 {q(gs, .75):.2f}  p95 {q(gs, .95):.2f}  "
              f"p99 {q(gs, .99):.2f}")
        print(f"  decode_rally assumes: min_gap 0.25, free band "
              f"0.45-2.2, penalised below 0.45, -3.0 outside 2.2")
        below = sum(1 for g in gs if g < 0.45)
        above = sum(1 for g in gs if g > 2.2)
        print(f"  -> {below}/{len(gs)} = {below / len(gs):.0%} of REAL "
              f"gaps fall in the penalised <0.45 band, and "
              f"{above}/{len(gs)} = {above / len(gs):.0%} are past 2.2 "
              f"where the DP charges -3.0")
        if gaps_same:
            gss = sorted(gaps_same)
            print(f"  TRUE same-side gap (n={len(gss)}): p01 "
                  f"{q(gss, .01):.2f}  med {q(gss, .50):.2f} — the "
                  f"0.55s pre-merge window must sit under p01 or it "
                  f"eats real contacts")
    # GHOSTS: the decoder's own account of what it could not place.
    gh = sum(g for pth in paths.values() for _t, _s, _sc, g in pth)
    if paths:
        print(f"  GHOSTS: {gh} contacts asserted but not timestamped "
              f"across {len(paths)} rallies. Every ghost makes the two "
              f"emitted events\n  around it legitimately SAME-side, "
              f"which is the premise alternation_fix denies "
              f"({alt_changed} overwrites).")
    # WHY WAS THE CANDIDATE NOT CHOSEN? For each true contact with no
    # event on it, find the best candidate that WAS available and say
    # where it sat in that rally's score distribution. A missed contact
    # whose candidate was strong is a DP-objective problem; one whose
    # candidate was weak is a scorer problem. They need opposite fixes.
    miss_pct, miss_n = [], 0
    for cum in sorted(truth):
        ev_ts = [t for t, _s, _tid in decoded.get(cum, [])]
        scores = sorted(sc for _t, _s, sc, _tid in dets_by_rally.get(cum, []))
        if not scores:
            continue
        for t_true, _nm in truth[cum]:
            if any(abs(t_true - te) <= 0.35 for te in ev_ts):
                continue
            miss_n += 1
            near = [sc for tt, _s, sc, _tid in dets_by_rally.get(cum, [])
                    if abs(tt - t_true) <= 0.35]
            if near:
                best_sc = max(near)
                rank = sum(1 for x in scores if x <= best_sc)
                miss_pct.append(rank / len(scores))
    if miss_n:
        strong = sum(1 for p_ in miss_pct if p_ >= 0.70)
        print(f"  MISSED CONTACTS: {miss_n} true contacts carry no "
              f"event; {len(miss_pct)} of them had a candidate within "
              f"0.35s.\n  {strong} of those candidates were at or above "
              f"the 70th percentile the DP calls confident — a strong "
              f"candidate\n  that was passed over is a DP-objective "
              f"problem, a weak one is a scorer problem.")

    if cross_count:
        print("\nBALL CROSSING COUNT — an independent witness on the "
              "event count\n  (every other count here descends from "
              "the same pose decoder; this one does not)")
        print(f"    {'rally':<8}{'true':>6}{'decoder':>9}{'crossings':>11}"
              f"{'segments':>10}{'net y':>8}")
        _ct = _cd = _cx = 0
        for cum in sorted(cross_count):
            nx, nseg, nl = cross_count[cum]
            nt = len(truth.get(cum, []))
            nd = len(decoded.get(cum, []))
            _ct += nt
            _cd += nd
            _cx += nx
            print(f"    r{cum:<7}{nt:>6}{nd:>9}{nx:>11}{nseg:>10}"
                  f"{nl[0]:>8.0f}")
        print(f"    {'TOTAL':<8}{_ct:>6}{_cd:>9}{_cx:>11}")
        print(f"  the decoder is {_cd - _ct:+d} against truth; "
              f"crossings are {_cx - _ct:+d}.")
        print("  crossings UNDERCOUNT by construction (the ball is not "
              "always findable) — what\n  matters is whether they "
              "undercount where the decoder OVERCOUNTS, since a rally\n"
              "  the decoder inflates and the ball does not is a rally "
              "the decoder invented.")

    _r = sum(v[0] for v in per_player_kind.values())
    _w = sum(v[1] for v in per_player_kind.values())
    _x = sum(v[2] for v in per_player_kind.values())
    _true = sum(v[1] for v in per_player.values())
    print(f"\nDECODER EVENTS — the binding constraint, and it is not "
          f"naming:\n  {_r + _w + _x} emitted, {_r + _w} matched a true "
          f"contact, {_x} spurious ({_x / max(1, _r + _w + _x):.0%});"
          f"\n  {_true - (_r + _w)} of {_true} true contacts have no "
          f"event on them.\n  Placement recall says a scored DETECTION "
          f"exists within 0.35s of every true contact,\n  so this is "
          f"the decoder's event SELECTION, not the detector.")
    if extra_dt:
        _sd = sorted(extra_dt)
        print(f"  WHERE THE JUNK IS: {extra_where['before']} before the "
              f"first true contact, {extra_where['after']} after the "
              f"last,\n  {extra_where['mid']} inside the rally. "
              f"Distance to the nearest true contact: med "
              f"{_sd[len(_sd) // 2]:.2f}s, p90 "
              f"{_sd[int(.9 * len(_sd))]:.2f}s.\n  Dead-time junk is "
              f"free to remove (trim the window); junk INSIDE the "
              f"rally needs the scorer or the DP.")
    # ---- TOUCH SHARE, WITHIN TEAM. The user's correction
    # (2026-08-21), and it changes the metric's character rather than
    # just its denominator.
    #
    # WHY THIS IS THE RIGHT DENOMINATOR: sides alternate exactly (0
    # violations / 229), so each side's total is essentially fixed by
    # the rally structure and carries no information about the players.
    # The only quantity actually being estimated is how a PAIR divided
    # its own touches, which is exactly what the geometry layer decides
    # and what "who is carrying this team" means.
    #
    # AND IT IS HARDER THAN THE OVERALL SHARE, which is worth stating
    # plainly because the overall number flatters us. Overall share
    # survives a 41%-junk event list at ~1.7pp mean error only because
    # junk spreads across all four players and cancels in the ratio.
    # Within a pair there is no such cancellation: a naming error moves
    # a touch FROM one partner TO the other, so every mistake counts
    # twice. The errors are also lumpy rather than smooth - a single
    # wrong side bit moves a whole rally's worth of one side's contacts
    # at once, ~12 of them in a 25-contact rally - so a couple of bits
    # are worth several points of share.
    if team_of_player:
        print("\nTOUCH SHARE WITHIN TEAM (the product metric)")
        print("  sides alternate exactly, so each side's TOTAL is fixed "
              "by the rally structure;\n  the only thing being "
              "estimated is how a pair split its own touches")
        by_team = defaultdict(list)
        for nm, tm in team_of_player.items():
            if nm in per_player:
                by_team[tm].append(nm)
        print(f"    {'team':<6}{'player':<22}{'pipe %':>9}"
              f"{'true %':>9}{'error':>9}")
        for tm in sorted(by_team):
            mem = sorted(by_team[tm])
            p_tot = sum(per_player[n][0] for n in mem)
            t_tot_ = sum(per_player[n][1] for n in mem)
            for nm in mem:
                ps = 100.0 * per_player[nm][0] / p_tot if p_tot else 0.0
                ts = 100.0 * per_player[nm][1] / t_tot_ if t_tot_ else 0.0
                print(f"    {tm:<6}{nm:<22}{ps:>8.1f}%{ts:>8.1f}%"
                      f"{ps - ts:>+8.1f}")
        print("  a naming error is ZERO-SUM inside a pair, so it moves "
              "share by twice its own size —\n  which is why this "
              "number, not the count, is the one the broken bits show "
              "up in")

    print("\nTOUCH COUNTS (pipeline vs truth)")
    print("    wrong = a real contact given to the partner (zero-sum "
          "between them);\n    extra = a detection with no true contact "
          "under it (over-counting, not misattribution)")
    print(f"    {'player':<22}{'pipe':>5}{'true':>6}{'delta':>7}"
          f"{'right':>7}{'wrong':>7}{'extra':>7}")
    for nm in sorted(per_player):
        p, t_ = per_player[nm]
        r_, w_, x_ = per_player_kind[nm]
        print(f"    {nm:<22}{p:>5}{t_:>6}{p - t_:>+7}"
              f"{r_:>7}{w_:>7}{x_:>7}")


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
    # OFFICIAL vs PLAYER: a stationary track that OUTLASTS every player
    # must not be selected. This is the r17 case (25 tracks, a courtside
    # official longer than the players) that corrupted both channels.
    tp = [i * 0.1 for i in range(20)]          # players, well tracked
    ts_long = [i * 0.1 for i in range(40)]      # official, tracked LONGER
    rd_ref = {"tracks": {
        1: mk(0, [100.0 + 12 * i for i in range(20)], tp, 600.0),
        2: mk(0, [900.0 - 12 * i for i in range(20)], tp, 610.0),
        3: mk(1, [120.0 + 9 * i for i in range(20)], tp, 200.0),
        4: mk(1, [880.0 - 9 * i for i in range(20)], tp, 210.0),
        9: _S(t=list(ts_long), cx=[20.0] * 40, ynorm=[400.0] * 40)}}
    rd_ref["tracks"][9]["side"] = 0
    sel = set(player_tracks(rd_ref))
    assert 9 not in sel, f"stationary official selected: {sel}"
    assert sel == {1, 2, 3, 4}, sel
    assert track_motion(rd_ref["tracks"][9]) == 0.0

    # DEPTH LABELLER: server and receiver are the deep players, and the
    # log names both. Near end is LOW on screen, so its deep player has
    # the LARGER y; the far end mirrors. Critically this must be immune
    # to left/right, which is the bit that inverts after a swap.
    rec_d = {"team_A_R": "ua", "team_A_L": "ub",
             "team_B_R": "uc", "team_B_L": "ud",
             "server_uuid": "ua", "receiver_uuid": "uc",
             "server_team": "A"}
    tt2 = [0.0, 0.1]
    def mk2(cx, y):
        z = _S(t=list(tt2), cx=[cx, cx], ynorm=[y, y])
        z["side"] = 0
        return z
    # team A near: Ann deep (y 760) serving, Bea at kitchen (y 470)
    # team B far:  Cal deep (y 120) receiving, Dee at kitchen (y 330)
    rd_d = {"tracks": {1: mk2(300.0, 760.0), 2: mk2(700.0, 470.0),
                       3: mk2(700.0, 120.0), 4: mk2(300.0, 330.0)}}
    lab_d, ok_d = label_by_depth(rd_d, 0.0, rec_d, "A", 0, nm)
    assert ok_d, "depth labeller failed on a clean fixture"
    assert lab_d == {1: "Ann", 2: "Bea", 3: "Cal", 4: "Dee"}, lab_d
    # MIRRORING the x positions must change nothing: the whole point is
    # that a post-point swap cannot invert this labeller.
    rd_m = {"tracks": {1: mk2(700.0, 760.0), 2: mk2(300.0, 470.0),
                       3: mk2(300.0, 120.0), 4: mk2(700.0, 330.0)}}
    lab_m, ok_m = label_by_depth(rd_m, 0.0, rec_m := rec_d, "A", 0, nm)
    assert ok_m and lab_m == lab_d, (lab_m, lab_d)
    # and it must follow the log: swap who serves, names follow
    rec_s = dict(rec_d, server_uuid="uc", receiver_uuid="ua",
                 server_team="B")
    lab_s, ok_s = label_by_depth(rd_d, 0.0, rec_s, "A", 0, nm)
    assert ok_s and lab_s[3] == "Cal" and lab_s[1] == "Ann", lab_s

    # ENSEMBLE: two of three voters must carry the bit. Build a case
    # where DEPTH is wrong (server standing shallow) but contact-order
    # and halves both point at the true server, and require the vote to
    # override depth — the whole reason for majority rather than a
    # fixed preference.
    # Ann holds team A's LEFT half, so halves points image-left at
    # track 1 while depth points at the deep track 2 — the voters must
    # genuinely split or the fixture proves nothing.
    rec_v = {"team_A_R": "ub", "team_A_L": "ua",
             "team_B_R": "uc", "team_B_L": "ud",
             "server_uuid": "ua", "receiver_uuid": "uc",
             "server_team": "A"}
    # near = team A. Ann (ua, half R) is the server but stands SHALLOW.
    rd_v = {"tracks": {1: mk2(300.0, 470.0),   # Ann: shallow, image-left
                       2: mk2(700.0, 760.0),   # Bea: deep,  image-right
                       3: mk2(700.0, 120.0),   # Cal: deep
                       4: mk2(300.0, 330.0)}}  # Dee: shallow
    # contact 0 is on track 1 (Ann serving), contact 1 on track 3 (Cal)
    evs_v = [(0.0, 0, 1), (0.5, 1, 3)]
    lab_v, ok_v = label_by_vote(rd_v, 0.0, rec_v, "A", 0, nm,
                                events=evs_v)
    assert ok_v, "vote labeller failed on a clean fixture"
    assert lab_v[1] == "Ann", ("depth alone would say Bea; the majority "
                               f"must override it: {lab_v}")
    assert lab_v[2] == "Bea" and lab_v[3] == "Cal", lab_v
    # depth alone really would have been wrong here — proving the
    # fixture tests the override rather than agreeing by luck
    lab_d2, _ok = label_by_depth(rd_v, 0.0, rec_v, "A", 0, nm)
    assert lab_d2[1] == "Bea", lab_d2
    # tally counts every voter, so a dead voter stays visible
    tal = defaultdict(lambda: [0, 0])
    label_by_vote(rd_v, 0.0, rec_v, "A", 0, nm, events=evs_v, tally=tal)
    assert set(tal) == {"depth", "halves", "contact"}, dict(tal)
    assert all(n > 0 for _ok, n in tal.values())

    # ELIMINATION: the best-scoring detection on the ILLEGAL side must
    # lose to a weaker one on the side alternation says must have hit.
    sides_t = {1: True, 2: True, 3: False, 4: False}   # 1,2 near
    dets_t = [(10.0, 0, 0.9, 3),      # strongest, but FAR
              (10.02, 0, 0.4, 1)]     # weaker, NEAR
    best_t, con_t = pick_contact_track(dets_t, 10.0, True, sides_t)
    assert con_t and best_t[3] == 1, best_t
    # unconstrained would have taken the far one
    assert max(dets_t, key=lambda d: d[2])[3] == 3
    # asking for the far side flips the winner
    best_f, con_f = pick_contact_track(dets_t, 10.0, False, sides_t)
    assert con_f and best_f[3] == 3, best_f
    # nothing legal nearby -> fall back, flagged as unconstrained
    only_far = [(10.0, 0, 0.9, 3)]
    best_n, con_n = pick_contact_track(only_far, 10.0, True, sides_t)
    assert best_n[3] == 3 and con_n is False
    # nothing at all in tolerance -> None
    assert pick_contact_track(dets_t, 99.0, True, sides_t)[0] is None

    # SETTLED ANCHOR: with players walking early and standing still
    # just before the contact, the chosen instant must be the still one.
    ts_w = [i * 0.2 for i in range(25)]          # 0 .. 4.8s
    def walker(x0, y0, still_from):
        cx, yy = [], []
        for i, t in enumerate(ts_w):
            k = min(i, still_from)               # moves, then stops
            cx.append(x0 + 30.0 * k)
            yy.append(y0 + 10.0 * k)
        z = _S(t=list(ts_w), cx=cx, ynorm=yy)
        z["side"] = 0
        return z
    rd_w = {"tracks": {i: walker(100.0 * i, 200.0 * (i % 2), 10)
                       for i in (1, 2, 3, 4)}}
    t_set = settled_anchor(rd_w, t_first=4.6, back=4.6, step=0.2)
    assert t_set > 2.0, f"picked a moving instant: {t_set}"
    # never searches past the first contact
    assert t_set <= 4.6
    # and with everyone moving throughout, it still returns something
    rd_mv = {"tracks": {i: walker(100.0 * i, 200.0 * (i % 2), 99)
                        for i in (1, 2, 3, 4)}}
    t_mv = settled_anchor(rd_mv, t_first=4.6, back=4.6, step=0.2)
    assert 0.0 <= t_mv <= 4.6

    # CASCADE: precedence must override the majority. Depth+halves
    # agree on track 2 here while contact says track 1, so a flat
    # majority picks 2 and a contact-first cascade must pick 1.
    lab_maj, _o1 = label_by_vote(rd_v, 0.0, rec_v, "A", 0, nm,
                                 events=evs_v)
    lab_cas, _o2 = label_by_vote(rd_v, 0.0, rec_v, "A", 0, nm,
                                 events=evs_v,
                                 order=["halves", "depth", "contact"])
    assert lab_maj[1] == "Ann", lab_maj
    # halves-first cascade follows halves, which named track 1 too
    assert lab_cas[1] == "Ann", lab_cas
    lab_dep, _o3 = label_by_vote(rd_v, 0.0, rec_v, "A", 0, nm,
                                 events=evs_v,
                                 order=["depth", "halves", "contact"])
    assert lab_dep[2] == "Ann", ("depth-first must follow depth even "
                                 f"when outvoted: {lab_dep}")
    # CONTACT VOTER STAYS SILENT RATHER THAN GUESSING. With a spurious
    # event prepended, ev[0] is no longer the serve, so this voter must
    # ABSTAIN on the serving pair and let the cascade fall through to
    # halves. The alternative — take the earliest event on the pair —
    # was measured and lost (geometry 89% -> 84%, r5 14/14 -> 7/14),
    # because firing more often is worthless when the extra firings are
    # wrong. Silence is a legitimate answer for a voter this trusted.
    evs_noise = [(-0.3, 1, 3)] + evs_v
    vo_n = {}
    label_by_vote(rd_v, 0.0, rec_v, "A", 0, nm, events=evs_noise,
                  votes_out=vo_n)
    assert "contact" not in dict(vo_n["Ann"]), dict(vo_n["Ann"])
    # the other two voters still speak, so the bit is still decided
    assert {"depth", "halves"} <= set(dict(vo_n["Ann"]))

    # BALL + APPROACH: the ball supplies the DIRECTION raw displacement
    # lacked. Put the ball beside track 1 (Ann) and have BOTH players
    # start equidistant, with Ann closing and Bea yielding — the
    # intrude/yield asymmetry. Both voters must name Ann.
    tb2 = [0.0, 0.3, 0.6]
    def mover(xs, y):
        z = _S(t=[-0.6, -0.3, 0.0], cx=list(xs), ynorm=[y, y, y])
        z["side"] = 0
        return z
    rd_b = {"tracks": {
        1: mover([500.0, 400.0, 320.0], 700.0),   # closes on the ball
        2: mover([500.0, 600.0, 700.0], 700.0),   # yields away
        3: mover([300.0, 300.0, 300.0], 200.0),
        4: mover([900.0, 900.0, 900.0], 200.0)}}
    vo_b = {}
    label_by_vote(rd_b, 0.0, rec_v, "A", 0, nm, events=evs_v,
                  votes_out=vo_b, ball_pts={0: (300.0, 700.0)})
    vb = dict(vo_b["Ann"])
    assert vb.get("ball") == 1, vb
    assert vb.get("approach") == 1, vb
    # symmetric case: equidistant and neither closing -> both abstain
    rd_sym = {"tracks": {
        1: mover([400.0, 400.0, 400.0], 700.0),
        2: mover([800.0, 800.0, 800.0], 700.0),
        3: mover([300.0, 300.0, 300.0], 200.0),
        4: mover([900.0, 900.0, 900.0], 200.0)}}
    vo_s = {}
    label_by_vote(rd_sym, 0.0, rec_v, "A", 0, nm, events=evs_v,
                  votes_out=vo_s, ball_pts={0: (600.0, 700.0)})
    vs2 = dict(vo_s["Ann"])
    assert "ball" not in vs2, "equidistant players must not get a ball vote"
    assert "approach" not in vs2, "nobody closing must not get a vote"
    # no ball point at all -> silent, not guessing
    vo_n2 = {}
    label_by_vote(rd_b, 0.0, rec_v, "A", 0, nm, events=evs_v,
                  votes_out=vo_n2, ball_pts={})
    assert "ball" not in dict(vo_n2["Ann"])
    # raw movement stays OFF unless explicitly re-enabled
    assert "movement" not in vb, vb

    # INTENT, the ball-free channel. The user's exact scenario: Alshon
    # crosses into Tyra's half, straight and committed; Tyra steps back
    # and out, slower and wandering. Track 1 must win WITHOUT any ball.
    def path_track(xs, ys):
        z = _S(t=[-0.6 + 0.1 * i for i in range(7)],
               cx=list(xs), ynorm=list(ys))
        z["side"] = 0
        return z
    # 1 = intruder: straight run toward the partner's x, into the net
    intr_x = [400.0, 430.0, 460.0, 490.0, 520.0, 550.0, 580.0]
    intr_y = [740.0, 733.0, 726.0, 719.0, 712.0, 705.0, 698.0]
    # 2 = yielder: backs off, outward, and wanders
    yld_x = [800.0, 812.0, 806.0, 820.0, 814.0, 828.0, 822.0]
    yld_y = [700.0, 706.0, 712.0, 719.0, 726.0, 733.0, 740.0]
    rd_i = {"tracks": {1: path_track(intr_x, intr_y),
                       2: path_track(yld_x, yld_y),
                       3: path_track([300.0] * 7, [200.0] * 7),
                       4: path_track([900.0] * 7, [200.0] * 7)}}
    s_intr = intent_score(rd_i, 1, 0.0, partner_x=800.0, is_near=True)
    s_yld = intent_score(rd_i, 2, 0.0, partner_x=400.0, is_near=True)
    assert s_intr > s_yld, (s_intr, s_yld)
    assert s_yld < 0, f"yielding must score NEGATIVE, not merely small: {s_yld}"
    # straightness must matter: the same net move, wandered, scores less
    wander_x = [400.0, 500.0, 420.0, 540.0, 450.0, 570.0, 580.0]
    rd_w = {"tracks": {**rd_i["tracks"],
                       1: path_track(wander_x, intr_y)}}
    s_wander = intent_score(rd_w, 1, 0.0, partner_x=800.0, is_near=True)
    assert s_wander < s_intr, (s_wander, s_intr)
    # and the far side mirrors: netward is +y there
    assert intent_score(rd_i, 1, 0.0, 800.0, is_near=False) < s_intr

    # votes_out exposes every voter for the truth table
    vo = {}
    label_by_vote(rd_v, 0.0, rec_v, "A", 0, nm, events=evs_v,
                  votes_out=vo)
    assert set(vo) == {"Ann", "Cal"}, vo
    assert {v for v, _t in vo["Ann"]} == {"depth", "halves", "contact"}

    # a side missing a player is reported, never guessed
    rd_bad = {"tracks": {1: rd["tracks"][1], 3: rd["tracks"][3],
                         4: rd["tracks"][4]}}   # only three on court
    _lab2, ok2 = label_tracks_at_serve(rd_bad, 0.0, rec, "A", 0, nm)
    assert ok2 is False

    # ---- ORDER SWEEP. Two earlier diagnostics in this thread were
    # VACUOUS (the voter tally scored voters against the ensemble's own
    # call; the parity check read truth from a constrained join), so
    # this one is pinned before it is trusted. Three properties:
    #   1 precedence really beats majority — a lone leading voter wins
    #     against two agreeing voters below it;
    #   2 ORDER CHANGES THE ANSWER — two orderings of the same bits
    #     must be able to score differently, or the sweep is measuring
    #     nothing;
    #   3 contact weighting dominates bit counting when they disagree,
    #     which is the whole reason the weighted column exists.
    # bit = (cum, who, truth_tid, {voter: tid}, weight)
    b_a = (1, "P", 10, {"x": 10, "y": 20, "z": 20}, 1)
    b_b = (2, "Q", 30, {"x": 40, "y": 30, "z": 30}, 1)
    assert decide_by_order(("x",), b_a[3]) == (10, "x")
    assert decide_by_order((), b_a[3])[0] == 20          # majority of 3
    assert decide_by_order(("q", "y"), b_a[3]) == (20, "y")   # silence
    # x is right on b_a and wrong on b_b; y is the mirror image. So
    # leading with x and leading with y must NOT score the same.
    two = [b_a, b_b]
    assert score_order(("x", "y"), two)[0] == 1
    assert score_order(("y", "x"), two)[0] == 1
    # ...and with a third bit where x is right, x-first must now win.
    b_c = (3, "R", 50, {"x": 50, "y": 60}, 1)
    three = [b_a, b_b, b_c]
    assert score_order(("x", "y"), three)[0] == 2
    assert score_order(("y", "x"), three)[0] == 1, \
        "order sweep cannot distinguish orderings — vacuous"
    # GEOM_SIDES. The near/far -> 0/1 mapping is recovered by majority
    # agreement with the very field it is replacing, which is only
    # sound while that field is wrong on a MINORITY of tracks. Pin both
    # directions, because a silent inversion would relabel every
    # detection in the rally and look like a decoder collapse.
    def _rd(sides, ys):
        class _S(dict):
            pass
        tr = {}
        ts = [0.0, 0.5, 1.0, 1.5]
        for tid, (sd, y) in enumerate(zip(sides, ys)):
            ser = _S(t=list(ts),
                     cx=[100.0 + 10 * tid + k for k in range(len(ts))],
                     ynorm=[float(y)] * len(ts))
            ser["side"] = sd
            tr[tid] = ser
        return {"tracks": tr, "fps": 30.0}

    #   The 0/1 LABELS are not free: m, the rally's mirror flag, is
    #   computed in the tracker's side space, so geom_sides must align
    #   to ser["side"] rather than invent its own polarity. What it
    #   replaces is the per-track ERRORS, not the convention.
    #   clean field: partition from y, labels already aligned
    _m, _f = geom_sides(_rd([0, 0, 1, 1], [100, 120, 700, 720]), 0.0)
    assert _f == 1.0 and _m == {0: 0, 1: 0, 2: 1, 3: 1}, (_m, _f)
    #   wholly inverted field: it REPRODUCES ser["side"], because that
    #   is the space m lives in — picking the other polarity here would
    #   relabel every detection and read as a decoder collapse
    _m2, _f2 = geom_sides(_rd([1, 1, 0, 0], [100, 120, 700, 720]), 0.0)
    assert _f2 == 1.0 and _m2 == {0: 1, 1: 1, 2: 0, 3: 0}, (_m2, _f2)
    #   one corrupt track — the case this exists for. The partition
    #   comes from y, so tid1 is CORRECTED 1 -> 0, and the agreement
    #   fraction reports 0.75 so a caller can see the field is dirty.
    _m3, _f3 = geom_sides(_rd([0, 1, 1, 1], [100, 120, 700, 720]), 0.0)
    assert _m3 == {0: 0, 1: 0, 2: 1, 3: 1} and _f3 == 0.75, (_m3, _f3)

    # RALLY END FROM MOTION. This can only fail in one direction that
    # matters - cutting the window ON TOP of real contacts, which no
    # later stage can undo - so pin both the detection and the
    # abstention.
    class _S2(dict):
        pass

    def _rd_motion(active_until, hi=8.0, step=0.1):
        """4 players moving steadily until `active_until`, then still.

        SMOOTH motion on purpose. The first version of this fixture
        was a square wave, which aliased against the 0.2s sampling so
        that half the samples read zero and the early-motion MEDIAN
        came out 0 - and rally_end_motion correctly abstained. The
        function was right and the test was wrong, which is worth
        keeping in the fixture rather than the commit message: a rally
        whose early motion medians to zero gets no answer here, by
        design.
        """
        tr = {}
        ts = [round(k * step, 3) for k in range(int(hi / step) + 1)]
        for tid in range(4):
            cx, yn = [], []
            for t in ts:
                m = min(t, active_until)
                cx.append(100.0 + tid * 50 + 300.0 * m)
                yn.append(300.0 + tid * 100 + 200.0 * m)
            ser = _S2(t=list(ts), cx=cx, ynorm=yn)
            ser["side"] = 0 if tid < 2 else 1
            tr[tid] = ser
        return {"tracks": tr, "fps": 30.0}
    #   motion stops at 3.0s -> the window closes shortly after, and
    #   NEVER before the play it is meant to contain
    _end = rally_end_motion(_rd_motion(3.0), 0.0)
    assert _end is not None, "motion end must fire on a clean fixture"
    assert 3.0 <= _end <= 5.0, _end
    #   a rally that never quiets must not be truncated early
    _end2 = rally_end_motion(_rd_motion(8.0), 0.0)
    assert _end2 is None or _end2 >= 8.0, _end2
    #   fewer than four tracks: abstain, never guess. A wrong end here
    #   deletes contacts silently, so None must reach the caller and
    #   let the gap fallback stand.
    _bad = _rd_motion(3.0)
    del _bad["tracks"][3]
    assert rally_end_motion(_bad, 0.0) is None

    # PASS FUNNEL. Two properties, both of which a staged decoder can
    # silently violate.
    #   1 NO PASS MAY READ TRUTH — decode_passes takes truth_ts only to
    #     score the funnel, so the same input must decode identically
    #     with and without it. A stage that peeked would report a
    #     funnel that cannot be reproduced at inference.
    #   2 recall must be MONOTONE NON-INCREASING down the stages, since
    #     no pass can invent a candidate it was not handed.
    _d = [(0.0, 0, 0.9, "a"), (0.05, 0, 0.3, "dup"),
          (0.8, 1, 0.8, "b"), (1.6, 0, 0.7, "c"),
          (9.9, 1, 0.1, "junk")]
    _e1, _s1, _g1 = decode_passes(_d, 0, 0.8, 1.4, chain=False)
    _e2, _s2, _g2 = decode_passes(_d, 0, 0.8, 1.4, chain=False,
                                  truth_ts=[0.0, 0.8, 1.6])
    assert [x[:2] for x in _e1] == [x[:2] for x in _e2], \
        "decode_passes changed its answer when shown truth"
    _counts = [len(k) for _n, k in _s1]
    assert _counts == sorted(_counts, reverse=True), \
        f"a pass ADDED events: {_counts}"
    #   the cluster merge must drop the 0.05s duplicate and keep the
    #   real contacts, and the window trim must drop the 9.9s outlier
    _by = {n: k for n, k in _s1}
    _merge = next(k for n, k in _s1 if n.startswith("1 "))
    assert not any(x[3] == "dup" for x in _merge), "cluster kept a dup"
    _win = next(k for n, k in _s1 if n.startswith("2 "))
    assert not any(x[3] == "junk" for x in _win), "window kept dead time"
    assert {x[3] for x in _win} >= {"a", "b", "c"}, "window ate real"

    # SAME-SIDE POLICY. Pinned because delete and insert are opposite
    # actions on identical-looking input, and picking wrong either
    # invents contacts or destroys them.
    #   a clean alternating sequence is untouched by every mode
    _alt = [(0.0, 0, "a"), (0.8, 1, "b"), (1.6, 0, "c")]
    for _m in ("overwrite", "delete", "insert", "auto"):
        _k, _f, _d, _i = same_side_policy(_alt, None, _m, 0.8)
        assert len(_k) == 3 and _f == [0, 1, 0], (_m, _k, _f)
        assert (_d, _i) == (0, 0)
    #   a CLUSTER duplicate (far closer than one exchange) is dropped
    #   by delete and by auto, and the survivor is the EARLIER one
    _dup = [(0.0, 0, "a"), (0.15, 0, "dup"), (0.9, 1, "b")]
    for _m in ("delete", "auto"):
        _k, _f, _d, _i = same_side_policy(_dup, None, _m, 0.8)
        assert [x[2] for x in _k] == ["a", "b"], (_m, _k)
        assert (_d, _i) == (1, 0), (_m, _d, _i)
    #   a MISSED contact (about two exchanges apart) is kept by auto
    _miss = [(0.0, 0, "a"), (1.7, 0, "c"), (2.5, 1, "d")]
    _k, _f, _d, _i = same_side_policy(_miss, None, "auto", 0.8)
    assert [x[2] for x in _k] == ["a", "c", "d"], _k
    assert (_d, _i) == (0, 1), (_d, _i)
    #   ...and delete would have thrown that real contact away, which
    #   is exactly why the two modes are not interchangeable
    _k2, _f2, _d2, _i2 = same_side_policy(_miss, None, "delete", 0.8)
    assert len(_k2) == 2 and _d2 == 1
    #   overwrite keeps everything and relabels, which is the shipped
    #   behaviour and the one that can corrupt a correct side
    _k3, _f3, _d3, _i3 = same_side_policy(_dup, None, "overwrite", 0.8)
    assert len(_k3) == 3 and _f3 == [0, 1, 0], (_k3, _f3)

    # DIAGONAL: a serve is cross-court, so the server and the receiver
    # cannot both be the image-right of their pair. Pinned because it
    # SILENTLY REWRITES a decision that every voter agreed on, which is
    # the most dangerous kind of rule in this file.
    #   confidence table must rank the two real deciders correctly, or
    #   the repair flips the wrong side
    assert DECIDER_CONF["contact"] > DECIDER_CONF["halves"]
    assert DECIDER_CONF["halves"] > DECIDER_CONF["ball"]
    #   a legal configuration (exactly one image-right) is untouched,
    #   an illegal one flips the weaker decider's side, and the flip
    #   lands on the OTHER member of that pair
    _pair_a = [(700.0, 100.0, "sL"), (700.0, 900.0, "sR")]
    _pair_b = [(200.0, 120.0, "rL"), (200.0, 880.0, "rR")]
    _right_a = max(_pair_a, key=lambda r: r[1])[2]
    _right_b = max(_pair_b, key=lambda r: r[1])[2]
    assert (_right_a, _right_b) == ("sR", "rR")
    #   sR + rL is diagonal (one right, one left) -> legal
    assert ("sR" == _right_a) != ("rL" == _right_b)
    #   sR + rR is a serve down the middle -> illegal, must be caught
    assert ("sR" == _right_a) == ("rR" == _right_b)
    #   and the repair is well-defined: the other member of the pair
    assert next(t for _y, _cx, t in _pair_b if t != "rR") == "rL"

    # VETO: the leader stands unless the others overturn it. Pinned
    # because it is the one rule in the panel that can make a GOOD
    # voter's answer disappear, so its firing conditions have to be
    # exactly as narrow as advertised.
    #   lead is wrong (7), two others agree on the truth (8) -> veto
    assert apply_veto(7, "x", {"x": 7, "y": 8, "z": 8}, 2, "unan") == 8
    #   ...but not if a single dissenter is all there is
    assert apply_veto(7, "x", {"x": 7, "y": 8}, 2, "unan") is None
    #   ...and not if the others disagree among themselves
    assert apply_veto(7, "x", {"x": 7, "y": 8, "z": 9}, 2, "unan") is None
    #   majority mode fires there, which is the whole difference
    assert apply_veto(7, "x", {"x": 7, "y": 8, "z": 8, "w": 9},
                      2, "maj") == 8
    assert apply_veto(7, "x", {"x": 7, "y": 8, "z": 9}, 2, "maj") is None
    #   never overturn the leader with its own answer
    assert apply_veto(7, "x", {"x": 7, "y": 7, "z": 7}, 2, "unan") is None
    #   k=0 disables it entirely
    assert apply_veto(7, "x", {"x": 7, "y": 8, "z": 8}, 0, "unan") is None
    #   and it must be able to change a score, or the grid is vacuous
    vb = [(9, "V", 8, {"x": 7, "y": 8, "z": 8}, 1)]
    assert score_order(("x", "y", "z"), vb)[0] == 0
    assert score_order(("x", "y", "z"), vb, 2, "unan")[0] == 1

    # weighting: one heavy bit outweighs two light ones going the other
    # way, so the weighted and unweighted rankings genuinely differ.
    heavy = [(4, "S", 70, {"x": 70, "y": 80}, 25),
             (5, "T", 90, {"x": 99, "y": 90}, 1),
             (6, "U", 91, {"x": 98, "y": 91}, 1)]
    ok_x, _n, w_x, _w = score_order(("x", "y"), heavy)
    ok_y, _n2, w_y, _w2 = score_order(("y", "x"), heavy)
    assert ok_x < ok_y and w_x > w_y, "weighting must be able to flip it"

    print("selftest OK: rally-end from motion, "
          "geom_sides (mapping + inversion), "
          "pass funnel (truth-blind, monotone), "
          "same-side policy (delete/insert/auto), "
          "diagonal (cross-court serve), order sweep "
          "(precedence, order-sensitivity, "
          "veto rules, "
          "weighting), quadrant round trip, end mirroring, orientation "
          "voting (clean + noisy), alternation overwrite, name "
          "assignment, label-at-serve survives a mid-rally switch")


if __name__ == "__main__":
    main()
