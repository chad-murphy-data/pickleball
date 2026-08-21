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


def label_by_vote(rd, t_serve, rec, near_team, flip, name_of,
                  events=None, tally=None, votes_out=None,
                  order=None):
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

    def pick(pair, is_near, who, half_of_who, contact_idx):
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
        # MOVEMENT: who is going somewhere, as against who is standing
        # where. Independent of depth and halves by construction — both
        # read a position at one instant, this reads a change across a
        # window — so its errors should not line up with theirs.
        da = displacement(rd, a[2], t_serve)
        db = displacement(rd, b[2], t_serve)
        if da is not None and db is not None and abs(da - db) > 1.0:
            votes.append(("movement", a[2] if da > db else b[2]))
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
                    best_tid = byname[v]
                    break
            else:
                best_tid = Counter(t for _v, t in votes).most_common(1)[0][0]
        else:
            counts = Counter(t for _v, t in votes)
            best_tid, _n = counts.most_common(1)[0]
        if tally is not None:
            for vname, t in votes:
                tally[vname][0] += (t == best_tid)
                tally[vname][1] += 1
        return best_tid

    half_of = {}
    for tm in ("A", "B"):
        for h in (RIGHT, LEFT):
            nm = name_of.get(rec.get(f"team_{tm}_{h}", "").lower())
            half_of[nm] = h
    s_tid = pick(srv_pair, srv_is_near, srv, half_of[srv], 0)
    r_tid = pick(rcv_pair, not srv_is_near, rcv, half_of[rcv], 1)
    labels = {s_tid: srv, r_tid: rcv}
    for _y, _cx, tid in srv_pair:
        labels.setdefault(tid, srv_mate)
    for _y, _cx, tid in rcv_pair:
        labels.setdefault(tid, rcv_mate)
    return labels, True


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
    ap.add_argument("--label", choices=["vote", "depth", "halves"],
                    default="vote",
                    help="vote (default) = majority of contact-order, "
                         "depth and halves on the one bit per side; "
                         "depth = server/receiver are the deep "
                         "players, names straight from the log, no "
                         "left/right; halves = the lineup R/L mapping, "
                         "which inverts when a serving team swaps ends "
                         "between points")
    ap.add_argument("--cascade",
                    help="comma-separated voter precedence, e.g. "
                         "'contact,depth,halves'. The first voter that "
                         "fired decides; omit for a flat majority")
    ap.add_argument("--no-settle", action="store_true",
                    help="anchor when all four are first on screen, "
                         "instead of at the stillest instant before the "
                         "first contact (the pre-serve setup)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    run(a)


def run(a):
    mode = getattr(a, "label", "vote")
    settle = not getattr(a, "no_settle", False)
    voter_tally = defaultdict(lambda: [0, 0])
    votes_by_rally = {}
    cascade = ([x.strip() for x in a.cascade.split(',')]
               if getattr(a, 'cascade', None) else None)

    def labeller(rd_, t_, rec_, nt_, fl_, nm_, events=None,
                 votes_out=None):
        if mode == "vote":
            return label_by_vote(rd_, t_, rec_, nt_, fl_, nm_,
                                 events=events, tally=voter_tally,
                                 votes_out=votes_out, order=cascade)
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
        tnames, geom_ok = labeller(
            rd, anchor_time(rd, evs[0][0], settle), rec, nt, flip, names_by_uuid,
            events=evs)
        if not geom_ok:
            unreadable += 1
            continue
        # PRODUCTION PATH gets the same elimination: the decoded event
        # names a track, but alternation says which side must have hit,
        # so a track on the illegal side is re-picked from that side's
        # own candidates rather than trusted.
        sides_p = side_map(rd, anchor_time(rd, evs[0][0], settle))
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
        vout = {}
        tnames, ok_lab = labeller(
            rd, anchor_time(rd, dets_by_rally[cum][0][0]
                            if dets_by_rally[cum] else 0.0, settle),
            rec, nt, flip, names_by_uuid, events=decoded.get(cum),
            votes_out=vout)
        votes_by_rally[cum] = vout
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
        census.append((cum, len(rd["tracks"]), stats, ok_lab))
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
    print("\nTRACK CENSUS — per rally: geometry hits/total, then the "
          "SELECTED four as\n  (samples, motion, y_at_anchor). A clean "
          "near/far split shows two low y and\n  two high y with a wide "
          "gap; interleaved y means the split is guessing.")
    for cum, n_tr, durs, ok_lab in census:
        gk, gt = per_rally_geom[cum]
        acc = f"{gk}/{gt}" if gt else "-"
        print(f"  r{cum:<4} tracks {n_tr:<3} geom {acc:>7}  "
              f"sel(len,motion,y) {durs}  "
              f"{'ok' if ok_lab else 'UNREADABLE'}")

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

    print("selftest OK: quadrant round trip, end mirroring, orientation "
          "voting (clean + noisy), alternation overwrite, name "
          "assignment, label-at-serve survives a mid-rally switch")


if __name__ == "__main__":
    main()
