"""Court-dominance measures — the EPISODIC view of "how much court".

The shipped width share is a time-average, and "he's pushing his
partner off the court" is episodic: most seconds of most rallies
everyone stands in normal position, so the mean boundary sits near 0.5
even when one player repeatedly barges into the partner's zone at the
moments that matter.  These four measures target the episodes.

PRE-REGISTERED 2026-08-18 in chat BEFORE any number was computed;
definitions restated here unchanged:

1. OFF-COURT DISPLACEMENT  (offcourt_frac)
   Fraction of a player's rally-phase frames with court x outside
   [0, 20] — standing beyond a sideline.  Uses each player's own
   frames; needs no partner visibility.
2. PARTNER-HALF INCURSION  (incursion_frac)
   On paired frames (both partners visible, same end) where BOTH stand
   on the same side of the court's centerline (x = 10), the INCURSOR
   is the player closer to the centerline — the displaced partner hugs
   the sideline.  Straddling the centerline = no incursion.
3. DOMINANCE MOMENTS  (share_p90, share_gt65)
   The per-frame instantaneous width share (identical midpoint mapping
   to coverage.rally_share); report its 90th percentile and
   P(share > 0.65) instead of only the mean.
4. SPACE CONTROL — Voronoi  (vor_mean, vor_p90, vor_gt65)
   Per paired frame, the fraction of the team's own in-bounds half
   (20 x 22 ft, net line to their baseline, 0.5 ft grid) closer to the
   player than to the partner.  Soccer-analytics pitch control,
   restricted to the pair.

Frame set = EXACTLY the rally-phase frames the shipped metrics used:
coverage.run()'s own rally_tracks_by_game, handed over via its collect
hook (same gates, same serve-phase exclusion, same identity chain).
Anti-cooking protocol: these definitions were frozen before
computation; if they fail the eyeball test they are REPORTED as
failing, not tuned until they agree.

    python vision/coverage_dominance.py --pose-dir ... --court ... \
        --windows ... --lineup ... [--cam ...] [--match-id ...]
    python vision/coverage_dominance.py --selftest

Output: printed table + data/coverage_dominance.csv (per player-game
rows plus a whole-match row per player, frame-weighted).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coverage as C
from coverage import NET_Y, W_FT

ROOT = Path(__file__).resolve().parent.parent
SHARE_HI = 0.65


def paired(ts_a, xy_a, ts_b, xy_b):
    """Common-frame positions (same round(t,3) matching as
    coverage.rally_share)."""
    ta = {round(t, 3): i for i, t in enumerate(ts_a)}
    A, B = [], []
    for j, t in enumerate(ts_b):
        i = ta.get(round(t, 3))
        if i is not None:
            A.append(xy_a[i])
            B.append(xy_b[j])
    return np.array(A), np.array(B)


def inst_share(A, B):
    """Instantaneous width share of a vs partner b (midpoint mapping)."""
    m = (A[:, 0] + B[:, 0]) / 2
    return np.where(A[:, 0] < B[:, 0], m / W_FT, (W_FT - m) / W_FT)


def incursion(A, B):
    """Bool per frame: a is the incursor (both same side of center,
    a nearer the centerline)."""
    da = A[:, 0] - W_FT / 2
    db = B[:, 0] - W_FT / 2
    same = da * db > 0
    return same & (np.abs(da) < np.abs(db))


_GRID = {}


def _grid(end):
    if end not in _GRID:
        ys = np.arange(0.25, NET_Y, 0.5) + (NET_Y if end == "near" else 0.0)
        xs = np.arange(0.25, W_FT, 0.5)
        _GRID[end] = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    return _GRID[end]


def voronoi_frac(A, B, end):
    """Per frame: fraction of the team's own half closer to a than b."""
    G = _grid(end)
    d = B - A
    c = (np.sum(B ** 2, 1) - np.sum(A ** 2, 1)) / 2
    return (G @ d.T < c[None, :]).mean(0)


def compute(rally_tracks_by_game):
    per = defaultdict(lambda: {"off": [], "share": [], "inc": [],
                               "vor": [], "own": 0})
    for game, rallies in rally_tracks_by_game.items():
        for cum, rd, lin in rallies:
            for u, (ts, xy, end) in rd.items():
                if len(ts):
                    p = per[(game, u)]
                    p["off"].append(np.sum((xy[:, 0] < 0)
                                           | (xy[:, 0] > W_FT)))
                    p["own"] += len(ts)
            if lin is None:
                continue
            for ta, tb in (("team_A_R", "team_A_L"),
                           ("team_B_R", "team_B_L")):
                ua, ub = lin[ta], lin[tb]
                if ua not in rd or ub not in rd:
                    continue
                ts_a, xy_a, end_a = rd[ua]
                ts_b, xy_b, end_b = rd[ub]
                if end_a != end_b or not len(ts_a) or not len(ts_b):
                    continue
                A, B = paired(ts_a, xy_a, ts_b, xy_b)
                if len(A) < 2:
                    continue
                sa = inst_share(A, B)
                va = voronoi_frac(A, B, end_a)
                ia = incursion(A, B)
                per[(game, ua)]["share"].append(sa)
                per[(game, ua)]["vor"].append(va)
                per[(game, ua)]["inc"].append(ia)
                per[(game, ub)]["share"].append(1.0 - sa)
                per[(game, ub)]["vor"].append(1.0 - va)
                per[(game, ub)]["inc"].append(incursion(B, A))
    return per


def summarize(per):
    rows = []
    whole = defaultdict(lambda: defaultdict(list))
    for (game, u), p in sorted(per.items()):
        share = np.concatenate(p["share"]) if p["share"] else np.array([])
        vor = np.concatenate(p["vor"]) if p["vor"] else np.array([])
        inc = np.concatenate(p["inc"]) if p["inc"] else np.array([])
        off = float(np.sum(p["off"])) / p["own"] if p["own"] else np.nan
        rows.append(dict(
            game=game[1], player_uuid=u, n_pair_frames=len(share),
            offcourt_frac=off,
            incursion_frac=float(inc.mean()) if len(inc) else np.nan,
            share_p90=(float(np.percentile(share, 90))
                       if len(share) else np.nan),
            share_gt65=(float((share > SHARE_HI).mean())
                        if len(share) else np.nan),
            vor_mean=float(vor.mean()) if len(vor) else np.nan,
            vor_p90=(float(np.percentile(vor, 90))
                     if len(vor) else np.nan),
            vor_gt65=(float((vor > SHARE_HI).mean())
                      if len(vor) else np.nan)))
        w = whole[u]
        w["share"].append(share)
        w["vor"].append(vor)
        w["inc"].append(inc)
        w["offn"].append(float(np.sum(p["off"])))
        w["own"].append(p["own"])
    for u, w in sorted(whole.items()):
        share = np.concatenate(w["share"])
        vor = np.concatenate(w["vor"])
        inc = np.concatenate(w["inc"])
        rows.append(dict(
            game="MATCH", player_uuid=u, n_pair_frames=len(share),
            offcourt_frac=sum(w["offn"]) / sum(w["own"]),
            incursion_frac=float(inc.mean()),
            share_p90=float(np.percentile(share, 90)),
            share_gt65=float((share > SHARE_HI).mean()),
            vor_mean=float(vor.mean()),
            vor_p90=float(np.percentile(vor, 90)),
            vor_gt65=float((vor > SHARE_HI).mean())))
    return rows


def names_from_players_csv():
    p = ROOT / "data/coverage_players.csv"
    if not p.exists():
        return {}
    return {r["player_uuid"]: r["player"]
            for r in csv.DictReader(open(p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-dir")
    ap.add_argument("--court")
    ap.add_argument("--windows")
    ap.add_argument("--lineup")
    ap.add_argument("--cam", default="")
    ap.add_argument("--no-cam-gate", action="store_true")
    ap.add_argument("--spotcheck", default="/nonexistent.csv")
    ap.add_argument("--swaps", default="")
    ap.add_argument("--vod", default="")
    ap.add_argument("--event", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--match-id", default="")
    ap.add_argument("--out", default=str(ROOT / "data/coverage_dominance.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    for req in ("pose_dir", "court", "windows", "lineup"):
        if not getattr(a, req):
            ap.error(f"--{req} required")
    got = {}
    C.run(a, collect=lambda rt: got.update(rt))
    per = compute(got)
    rows = summarize(per)
    names = names_from_players_csv()
    hdr = ("player", "game", "frames", "offcourt%", "incursion%",
           "share_p90", "P(s>.65)", "vor_mean", "vor_p90", "P(v>.65)")
    print(("{:<22} {:>5} {:>7} {:>9} {:>10} {:>9} {:>8} {:>8} {:>7} "
           "{:>8}").format(*hdr))
    for r in rows:
        print("{:<22} {:>5} {:>7} {:>9.1%} {:>10.1%} {:>9.3f} {:>8.1%} "
              "{:>8.3f} {:>7.3f} {:>8.1%}".format(
                  names.get(r["player_uuid"], r["player_uuid"][:8]),
                  r["game"], r["n_pair_frames"], r["offcourt_frac"],
                  r["incursion_frac"], r["share_p90"], r["share_gt65"],
                  r["vor_mean"], r["vor_p90"], r["vor_gt65"]))
    fields = ["player"] + list(rows[0])
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            r["player"] = names.get(r["player_uuid"], "")
            wr.writerow({k: r[k] for k in fields})
    print(f"-> {a.out}")


def selftest():
    # symmetric pair: a left at (5, 33), b right at (15, 33), near end
    A = np.array([[5.0, 33.0]] * 4)
    B = np.array([[15.0, 33.0]] * 4)
    s = inst_share(A, B)
    assert np.allclose(s, 0.5), s
    v = voronoi_frac(A, B, "near")
    assert np.all(np.abs(v - 0.5) < 0.03), v
    assert not incursion(A, B).any() and not incursion(B, A).any()
    # a barges to (12, 33): past center into b's half, b pushed to 19
    A2 = np.array([[12.0, 33.0]] * 4)
    B2 = np.array([[19.0, 33.0]] * 4)
    s2 = inst_share(A2, B2)
    assert np.allclose(s2, 15.5 / 20), s2       # midpoint 15.5 from left
    assert incursion(A2, B2).all() and not incursion(B2, A2).any()
    v2 = voronoi_frac(A2, B2, "near")
    assert np.all(v2 > 0.70), v2
    # off-court: x = -1 and 21 are out, 0.5 in
    xs = np.array([[-1.0, 30], [21.0, 30], [0.5, 30]])
    assert int(np.sum((xs[:, 0] < 0) | (xs[:, 0] > W_FT))) == 2
    # paired alignment drops non-common frames
    A3, B3 = paired(np.array([1.0, 1.1]), np.array([[1, 2], [3, 4.]]),
                    np.array([1.1, 1.2]), np.array([[5, 6], [7, 8.]]))
    assert len(A3) == 1 and A3[0][0] == 3 and B3[0][0] == 5
    print("SELFTEST OK (share, incursion, voronoi, offcourt, pairing)")


if __name__ == "__main__":
    main()
