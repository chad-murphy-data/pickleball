"""Rallies into shots: contacts, who hit them, and what kind they were.

CONTACT DETECTION, DONE PHYSICALLY
    The earlier detector looked for vertical reversals in PIXELS of at least
    22 px, then threw away anything inside a hand-drawn "kitchen band" to
    avoid counting bounces.  Both halves of that are guesses about a
    particular camera.  In court coordinates there is an actual physical
    discriminator:

        a PADDLE reverses the ball's direction ACROSS the court;
        a BOUNCE does not.

    A ball travelling toward the far baseline keeps travelling that way
    after it bounces — the bounce flips its VERTICAL velocity, which the
    camera sees as a wiggle, not its court-y velocity.  So a contact is a
    sign change in d(court_y)/dt, and bounces are excluded by physics
    rather than by masking off a region of the image.

    This also removes the dink/bounce confusion that made the pixel
    detector inflate contact counts fourfold: a dink and its bounce look
    alike vertically, and nothing alike across the court.

ATTRIBUTION, WITH THE LOG AS THE ANCHOR
    Two independent channels, because each fails where the other works:

    1. POSITION.  The contact's court position gives a side (near/far by
       y, net at 22) and a half (left/right by x).  vision/lineup.py says
       who stands in that half at this moment.  No appearance model, no
       player detection at all.
    2. NEAREST PLAYER.  When the tracker has a player blob near the
       contact in IMAGE space, use it.  Image space matters here: the ball
       is airborne, so its ground back-projection runs long, while the
       player's does not — comparing them in feet would compare two
       different things.

    Channel 1 is available always and is wrong when players cross over.
    Channel 2 is exact when it fires and is unavailable whenever the
    broadcast has cropped the player out of frame, which this camera does
    constantly behind the baselines.  They are reported separately AND
    together, so their disagreement rate is visible rather than hidden.

THE HONEST TEST, AND WHAT IT SAYS
    Two scores, neither of which needs a hand label:

      * SIDE ALTERNATION.  Consecutive contacts must land on opposite
        sides, because the ball crosses the net between them.  Chance is
        50%.  This scores the contact detector directly.
      * SERVE SIDE / SERVE PLAYER.  The log names the server, so the first
        contact of a rally has a free label.  Chance 50% and 25%.

    Run on MLP Chicago (32 rallies, women's doubles) these come back at
    ~35% alternation on ~1 contact per rally, against roughly 12 shots per
    rally actually played.  That is the honest state of this pipeline: the
    court fit and the identity anchor are solid, and the BALL DETECTOR is
    the binding constraint.  See vision/mvp_findings.md.

    python vision/shots.py --ball P_ball.csv --players P_players.csv \\
        --court court.json --rallies windows.csv --lineup lineup_x.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import court as C                                            # noqa: E402

NET_Y = C.NET_Y


# --------------------------------------------------------------------------
def load_csv(path, cols, play_region=False):
    a = np.genfromtxt(path, delimiter=",", names=True)
    if a.ndim == 0:
        a = a.reshape(1)
    d = {c: np.asarray(a[c], float) for c in cols}
    if play_region:
        # The ball is airborne, so its ground back-projection runs long and
        # the region has to be generous — but not unbounded: without this,
        # candidates from the crowd and the far signage land at things like
        # court (31, -51) and get attributed to a player as confidently as
        # a real shot does.
        k = ((d["x_ft"] > -12) & (d["x_ft"] < C.W_FT + 12)
             & (d["y_ft"] > -22) & (d["y_ft"] < C.L_FT + 16))
        d = {c: v[k] for c, v in d.items()}
        d["_dropped"] = float(1.0 - k.mean())
    return d


def link_tracks(fr, t, xi, yi, max_gap=4, gate=46.0):
    """Nearest-neighbour tracks in IMAGE space.

    Gating in pixels rather than feet is deliberate: the ball is above the
    plane, so its court-space speed is not physical and varies with height
    even at constant true speed.
    """
    order = np.argsort(fr)
    fr, t, xi, yi = fr[order], t[order], xi[order], yi[order]
    tracks, open_tr = [], []
    i = 0
    n = len(fr)
    while i < n:
        f = fr[i]
        j = i
        while j < n and fr[j] == f:
            j += 1
        cur = list(range(i, j))
        used, nxt = set(), []
        for tr in open_tr:
            k = tr[-1]
            if f - fr[k] > max_gap:
                tracks.append(tr)
                continue
            best, bj = gate * (f - fr[k]), None
            for c in cur:
                if c in used:
                    continue
                d = np.hypot(xi[c] - xi[k], yi[c] - yi[k])
                if d < best:
                    best, bj = d, c
            if bj is not None:
                used.add(bj)
                tr.append(bj)
            nxt.append(tr)
        for c in cur:
            if c not in used:
                nxt.append([c])
        open_tr = nxt
        i = j
    tracks.extend(open_tr)
    return tracks, (fr, t, xi, yi)


def contacts_from_track(idx, t, cy, min_run=3, min_travel=2.0):
    """Sign changes of d(court_y)/dt — i.e. the ball turned around.

    `min_run` frames of consistent direction on each side and `min_travel`
    feet of court travel are required, which is what keeps detection noise
    on a 2-8 px blob from manufacturing contacts.
    """
    if len(idx) < 2 * min_run + 1:
        return []
    y = cy[idx]
    ts = t[idx]
    # smooth lightly: the ball's ground back-projection is noisy in height
    k = np.ones(3) / 3.0
    ys = np.convolve(y, k, mode="same")
    ys[0], ys[-1] = y[0], y[-1]
    v = np.sign(np.diff(ys))
    out = []
    i = min_run
    while i < len(v) - min_run:
        before = v[i - min_run:i]
        after = v[i:i + min_run]
        if (before.mean() * after.mean() < 0 and abs(before.sum()) == min_run
                and abs(after.sum()) == min_run):
            a = ys[max(0, i - min_run)]
            b = ys[min(len(ys) - 1, i + min_run)]
            if abs(ys[i] - a) > min_travel and abs(b - ys[i]) > min_travel:
                out.append(i)
                i += min_run
        i += 1
    return out


# --------------------------------------------------------------------------
def half_of(x_ft):
    return "img_right" if x_ft > C.W_FT / 2 else "img_left"


def occupant(lineup_row, near_team, side, x_ft):
    """Who stands in this half of this side, per the log state machine.

    Geometry, not a free parameter: the camera sits behind the near team,
    so a near player facing the net has their right on the IMAGE right,
    while a far player faces the camera and has their right on the image
    LEFT.  That is also what makes the serve diagonal come out right.
    """
    team = near_team if side == "near" else ("B" if near_team == "A" else "A")
    right_is_img_right = side == "near"
    want_R = (x_ft > C.W_FT / 2) == right_is_img_right
    return lineup_row[f"team_{team}_{'R' if want_R else 'L'}"]


def analyse(ball, players, rallies, lineup, near_team, H,
            player_gate_px=110.0):
    Hi = np.linalg.inv(H)
    fr, t = ball["frame"], ball["t_s"]
    xi, yi, cy_all = ball["x_img"], ball["y_img"], ball["y_ft"]
    cx_all = ball["x_ft"]
    pt = players["t_s"]
    px, py = players["x_img"], players["y_img"]

    shots = []
    for r in rallies:
        m = (t >= r["t0"]) & (t < r["t1"])
        if m.sum() < 8:
            shots.append(None)
            continue
        sel = np.nonzero(m)[0]
        tracks, (F, T, X, Y) = link_tracks(fr[sel], t[sel], xi[sel], yi[sel])
        # re-index court coords to the sorted order used inside link_tracks
        order = np.argsort(fr[sel])
        CY = cy_all[sel][order]
        CX = cx_all[sel][order]
        tracks = [tr for tr in tracks if len(tr) >= 8]
        # WHICH TRACK IS THE BALL: it crosses the net.  Nothing else in the
        # frame does — players hold their side, and colour noise sits still.
        # This is the only ball identifier here that is not a tuned
        # threshold, and it is what lifts the contact detector's
        # side-alternation score out of the noise (9% -> ~35%).
        tracks = [tr for tr in tracks
                  if np.sum(np.diff(np.sign(CY[tr] - NET_Y)) != 0) >= 1]
        tracks.sort(key=lambda tr: T[tr[0]])
        lr = lineup.get(r["rally"])
        for tr in tracks:
            for ci in contacts_from_track(tr, T, CY):
                k = tr[ci]
                cyf, cxf = float(CY[k]), float(CX[k])
                side = "near" if cyf > NET_Y else "far"
                by_pos = occupant(lr, near_team, side, cxf) if lr else ""
                # nearest player blob, in image space
                w = np.abs(pt - T[k]) < 0.10
                by_near, dpx = "", np.nan
                if w.any():
                    d = np.hypot(px[w] - X[k], py[w] - Y[k])
                    j = int(np.argmin(d))
                    if d[j] < player_gate_px:
                        dpx = float(d[j])
                        pcx = players["x_ft"][w][j]
                        pcy = players["y_ft"][w][j]
                        pside = "near" if pcy > NET_Y else "far"
                        by_near = occupant(lr, near_team, pside,
                                           float(pcx)) if lr else ""
                shots.append({
                    "rally": r["rally"], "t_s": float(T[k]),
                    "x_ft": round(cxf, 2), "y_ft": round(cyf, 2),
                    "side": side, "half": half_of(cxf),
                    "by_position": by_pos, "by_nearest": by_near,
                    "nearest_px": None if np.isnan(dpx) else round(dpx, 1),
                })
    return [s for s in shots if s]


# --------------------------------------------------------------------------
def resolve_near_team(shots_a, shots_b):
    """Pick the near-team hypothesis that explains the serves better."""
    return shots_a if shots_a >= shots_b else shots_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ball", required=True)
    ap.add_argument("--players", required=True)
    ap.add_argument("--court", required=True)
    ap.add_argument("--rallies", required=True,
                    help="csv: rally,t0,t1 in VIDEO seconds")
    ap.add_argument("--lineup", required=True)
    ap.add_argument("--out")
    ap.add_argument("--names", action="store_true")
    a = ap.parse_args()

    d = json.load(open(a.court))
    H = np.array(d["H_court_to_img"])
    ball = load_csv(a.ball, ["frame", "t_s", "x_img", "y_img", "x_ft", "y_ft"],
                    play_region=True)
    players = load_csv(a.players, ["t_s", "x_img", "y_img", "x_ft", "y_ft"])
    rallies = [{"rally": int(r["rally"]), "t0": float(r["t0"]),
                "t1": float(r["t1"])}
               for r in csv.DictReader(open(a.rallies))]
    lineup = {int(r["rally"]): r for r in csv.DictReader(open(a.lineup))}

    best = None
    for near_team in ("A", "B"):
        shots = analyse(ball, players, rallies, lineup, near_team, H)
        # score on serves: first contact of each rally should be the server
        acc, n = serve_accuracy(shots, lineup)
        if best is None or acc > best[0]:
            best = (acc, n, near_team, shots)
    acc, n, near_team, shots = best
    alt, nalt = alternation(shots)
    sacc, sn = serve_side_accuracy(shots, lineup, near_team)
    drop = ball.get("_dropped", 0.0)
    print(f"contacts: {len(shots)} over {len(rallies)} rallies "
          f"({len(shots)/max(len(rallies),1):.1f}/rally); "
          f"{drop*100:.0f}% of ball candidates dropped as off-court")
    print(f"  side alternation   {alt*100:.0f}% of {nalt} consecutive pairs "
          f"(chance 50%) — label-free score for the contact detector")
    print(f"  near team = {near_team}")
    print(f"  serve SIDE         {sacc*100:.0f}% of {sn} (chance 50%)")
    print(f"  serve PLAYER       {acc*100:.0f}% of {n} (chance 25%)")
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(shots[0]))
            w.writeheader()
            w.writerows(shots)
        print(f"wrote {a.out}")


def alternation(shots):
    """Consecutive contacts in a rally must swap sides — the ball crosses
    the net every time.  This needs no labels at all, so it scores the
    CONTACT DETECTOR itself: spurious contacts (a bounce read as a hit, a
    tracking gap read as a new shot) show up as same-side repeats."""
    ok = tot = 0
    by = {}
    for s in shots:
        by.setdefault(s["rally"], []).append(s)
    for r, ss in by.items():
        ss.sort(key=lambda s: s["t_s"])
        for a, b in zip(ss, ss[1:]):
            tot += 1
            ok += a["side"] != b["side"]
    return (ok / tot if tot else 0.0), tot


def serve_side_accuracy(shots, lineup, near_team):
    """Coarse, robust check: the first contact of a rally should be on the
    SERVER'S side of the net.  Two-way (chance 50%) and immune to the
    left/right half being wrong."""
    seen, hit, tot = set(), 0, 0
    for s in sorted(shots, key=lambda s: (s["rally"], s["t_s"])):
        if s["rally"] in seen:
            continue
        seen.add(s["rally"])
        lr = lineup.get(s["rally"])
        if not lr:
            continue
        srv_team = "A" if lr["server_uuid"] in (
            lr["team_A_R"], lr["team_A_L"]) else "B"
        want = "near" if srv_team == near_team else "far"
        tot += 1
        hit += s["side"] == want
    return (hit / tot if tot else 0.0), tot


def serve_accuracy(shots, lineup):
    seen, hit, tot = set(), 0, 0
    for s in shots:
        if s["rally"] in seen:
            continue
        seen.add(s["rally"])
        lr = lineup.get(s["rally"])
        if not lr:
            continue
        tot += 1
        hit += s["by_position"] == lr["server_uuid"]
    return (hit / tot if tot else 0.0), tot


if __name__ == "__main__":
    main()
