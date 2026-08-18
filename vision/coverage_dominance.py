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

SECOND BATCH — PRE-REGISTERED 2026-08-18 in chat BEFORE computing
(user design: time-averages dilute episodic poaching; a 2 s poach in a
20 s rally is ~10% of frames, so per-frame means sit near 0.5 even
when the poach decides the rally):

5. RALLY PEAK EXCURSION  (pk_share_mean, pk_share_p90, peak_inc_mean_ft,
   deep_poach_frac) — "furthest ventured is the rally's number".
   Per rally the player's score is the MAXIMUM instantaneous width
   share reached at any paired frame, and the maximum depth in feet
   the player extends past the court centerline (x = 10) into the side
   the partner currently occupies (0 if never).  Depth counts only on
   frames where the player is measure 2's incursor (both partners on
   the same side of the centerline, player the nearer to it) — the
   displaced partner hugging the sideline is NOT incursing, and a
   completed side swap is a switch, not a poach.  (Sharpened to the
   measure-2 convention at selftest time, before any real number: the
   raw past-centerline formula scores the DISPLACED player as the
   incursor.)  Across rallies report the mean and p90 of the peaks and
   the fraction of rallies with peak incursion >= DEEP_POACH_FT = 2 ft.
6. CLAIMED TERRITORY WITH HYSTERESIS  (terr_mean, terr_end_mean) —
   "once he poaches, it's his space until she reclaims it".
   Pitch control with memory: each paired frame stamps every grid cell
   of the team's own half within REACH = 4.0 ft of a player as
   last-occupied by that player (same-frame overlaps go to the closer
   player); a cell belongs to its most recent occupant until the
   partner overwrites it.  Per-frame claim share = own claimed cells /
   all claimed cells; report the frame-weighted mean and the
   end-of-rally value averaged across rallies.
   REACH and DEEP_POACH_FT were fixed before any real number existed.
   These measure SPACE TAKEN, not balls taken — true poaching (hitting
   the partner's ball) needs ball data this stack does not have.

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
REACH = 4.0            # ft; a stamped cell's claim radius (measure 6)
DEEP_POACH_FT = 2.0    # ft past centerline = a "deep poach" (measure 5)


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


def peak_incursion(A, B):
    """Per frame: how far a extends past the centerline while being
    the incursor per measure 2's rule (ft, >= 0)."""
    return np.where(incursion(A, B),
                    np.abs(A[:, 0] - W_FT / 2), 0.0)


def territory(A, B, end):
    """Hysteresis claim shares for a, per frame (measure 6).

    A cell within REACH of a player is stamped as last-occupied by
    them (same-frame overlaps -> closer player) and stays theirs until
    the partner overwrites it.  Returns the per-frame array of
    a's claimed / all-claimed; frames with nothing claimed yet are
    skipped (cannot happen after frame 1 unless both are off-half).
    """
    G = _grid(end)
    owner = np.zeros(len(G), np.int8)          # 0 none, 1 a, 2 b
    out = []
    for i in range(len(A)):
        da = np.linalg.norm(G - A[i], axis=1)
        db = np.linalg.norm(G - B[i], axis=1)
        ina, inb = da <= REACH, db <= REACH
        owner[ina & ~inb] = 1
        owner[inb & ~ina] = 2
        both = ina & inb
        owner[both] = np.where(da[both] <= db[both], 1, 2)
        n = np.sum(owner > 0)
        if n:
            out.append(np.sum(owner == 1) / n)
    return np.array(out)


def compute(rally_tracks_by_game):
    per = defaultdict(lambda: {"off": [], "share": [], "inc": [],
                               "vor": [], "own": 0,
                               "pk_share": [], "pk_inc": [],
                               "terr": [], "terr_end": []})
    rally_rows = []
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
                ta_terr = territory(A, B, end_a)
                for u, S, V, I, T, INC in (
                        (ua, sa, va, ia, ta_terr, peak_incursion(A, B)),
                        (ub, 1.0 - sa, 1.0 - va, incursion(B, A),
                         1.0 - ta_terr, peak_incursion(B, A))):
                    p = per[(game, u)]
                    p["share"].append(S)
                    p["vor"].append(V)
                    p["inc"].append(I)
                    pk_s, pk_i = float(S.max()), float(INC.max())
                    p["pk_share"].append(pk_s)
                    p["pk_inc"].append(pk_i)
                    p["terr"].append(T)
                    p["terr_end"].append(float(T[-1]) if len(T)
                                         else np.nan)
                    rally_rows.append(dict(
                        game=game, rally=cum, player_uuid=u,
                        n_frames=len(S), peak_share=round(pk_s, 4),
                        peak_inc_ft=round(pk_i, 2),
                        terr_mean=(round(float(T.mean()), 4)
                                   if len(T) else ""),
                        terr_end=(round(float(T[-1]), 4)
                                  if len(T) else "")))
    return per, rally_rows


def _peaks(pk_share, pk_inc, terr, terr_end):
    """Measure 5 + 6 summary fields from per-rally lists."""
    ps = np.array(pk_share)
    pi = np.array(pk_inc)
    t = np.concatenate(terr) if terr else np.array([])
    te = np.array([x for x in terr_end if not np.isnan(x)])
    return dict(
        n_rallies=len(ps),
        pk_share_mean=float(ps.mean()) if len(ps) else np.nan,
        pk_share_p90=(float(np.percentile(ps, 90))
                      if len(ps) else np.nan),
        peak_inc_mean_ft=float(pi.mean()) if len(pi) else np.nan,
        deep_poach_frac=(float((pi >= DEEP_POACH_FT).mean())
                         if len(pi) else np.nan),
        terr_mean=float(t.mean()) if len(t) else np.nan,
        terr_end_mean=float(te.mean()) if len(te) else np.nan)


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
                      if len(vor) else np.nan),
            **_peaks(p["pk_share"], p["pk_inc"], p["terr"],
                     p["terr_end"])))
        w = whole[u]
        w["share"].append(share)
        w["vor"].append(vor)
        w["inc"].append(inc)
        w["offn"].append(float(np.sum(p["off"])))
        w["own"].append(p["own"])
        w["pk_share"].extend(p["pk_share"])
        w["pk_inc"].extend(p["pk_inc"])
        w["terr"].extend(p["terr"])
        w["terr_end"].extend(p["terr_end"])
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
            vor_gt65=float((vor > SHARE_HI).mean()),
            **_peaks(w["pk_share"], w["pk_inc"], w["terr"],
                     w["terr_end"])))
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
    ap.add_argument("--track-map", default="")
    ap.add_argument("--vod", default="")
    ap.add_argument("--event", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--match-id", default="")
    ap.add_argument("--out", default=str(ROOT / "data/coverage_dominance.csv"))
    ap.add_argument("--rally-out",
                    default=str(ROOT / "data/coverage_dominance_rallies.csv"))
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
    per, rally_rows = compute(got)
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
    hdr2 = ("player", "game", "rallies", "pk_share", "pk_p90",
            "pk_inc_ft", "deep%", "terr", "terr_end")
    print()
    print(("{:<22} {:>5} {:>7} {:>8} {:>7} {:>9} {:>6} {:>6} "
           "{:>8}").format(*hdr2))
    for r in rows:
        print("{:<22} {:>5} {:>7} {:>8.3f} {:>7.3f} {:>9.2f} {:>6.1%} "
              "{:>6.3f} {:>8.3f}".format(
                  names.get(r["player_uuid"], r["player_uuid"][:8]),
                  r["game"], r["n_rallies"], r["pk_share_mean"],
                  r["pk_share_p90"], r["peak_inc_mean_ft"],
                  r["deep_poach_frac"], r["terr_mean"],
                  r["terr_end_mean"]))
    fields = ["player"] + list(rows[0])
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            r["player"] = names.get(r["player_uuid"], "")
            wr.writerow({k: r[k] for k in fields})
    print(f"-> {a.out}")
    rfields = ["player"] + list(rally_rows[0]) if rally_rows else []
    if rally_rows:
        with open(a.rally_out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=rfields)
            wr.writeheader()
            for r in rally_rows:
                r["player"] = names.get(r["player_uuid"], "")
                wr.writerow({k: r[k] for k in rfields})
        print(f"-> {a.rally_out}")


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
    # measure 5: peak incursion — a at 12, b at 19 (b's side is right)
    pi = peak_incursion(A2, B2)
    assert np.allclose(pi, 2.0), pi
    assert np.allclose(peak_incursion(B2, A2), 0.0)   # b never crosses
    assert np.allclose(peak_incursion(A, B), 0.0)     # symmetric pair
    # a spikes to 13 for one frame -> rally peak 3 ft, most frames 0
    A4 = np.array([[5.0, 33], [13.0, 33], [5.0, 33]])
    B4 = np.array([[15.0, 33]] * 3)
    assert peak_incursion(A4, B4).max() == 3.0
    # measure 6: symmetric static pair -> claims split ~50/50
    t = territory(A, B, "near")
    assert len(t) == 4 and abs(t[-1] - 0.5) < 0.05, t
    # a sweeps through b's zone and retreats: the swept cells STAY a's
    # (hysteresis) until b, static, overwrites only its own disc
    A5 = np.array([[5.0, 33], [14.0, 33], [5.0, 33]])
    B5 = np.array([[16.5, 33]] * 3)
    t5 = territory(A5, B5, "near")
    assert t5[-1] > 0.55, t5     # a keeps most of what it swept
    t5b = territory(B5, A5, "near")
    assert abs(t5[-1] + t5b[-1] - 1.0) < 1e-9   # complements
    print("SELFTEST OK (share, incursion, voronoi, offcourt, pairing, "
          "peak, territory)")


if __name__ == "__main__":
    main()
