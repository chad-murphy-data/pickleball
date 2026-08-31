"""Contact territory — where each player hits the ball, from labels.

User request 2026-08-31: "look at the coded rallies and create some
sort of metric of where players are hitting the ball to see how much
court someone's taking." This is the CONTACT-side complement to the
coverage model (branch claude/court-coverage-model-8rg94l): coverage
measures space OCCUPIED (pose), this measures space USED AT CONTACT
(ball position when a player hits). It is also the direct product
face of finding 11's w-dial: v2's pooled weakest-link gamma implies
the stronger player takes w = 0.59 of the court; a within-team
contact-width split is that number observed instead of inferred.

Inputs, all committed (no tracker, no sealed data):
- ball passes  data/vision/ball_path_r{1,6,7,8}.csv (rally 9 EXCLUDED
  until its seal/train designation lands — its pass stays unpeeked)
- hitter taps  r1: state_labels (kind=impact, per-episode player);
               r6-8: contact_labels manual/divergent taps, contact!=0
- court        court_landmarks + court3d DLT (one-time calibration)

Method: ball position at each tap (nearest labeled V/S/I frame within
+/-0.15 s) -> pixel -> court feet via the calibrated projection,
intersecting the pixel ray with a horizontal plane. Ground plane
(z=0) is the primary read; z=2.5 ft re-projection is printed as the
height-bias band (a contact happens above the ground, so the ground
projection slides the point away from the camera along the ray —
WIDTH is barely affected near mid-frame, DEPTH wears the bias).

Metrics per player (serves and returns excluded from territory —
their positions are rule-fixed; reported separately):
- touch share within team (rally shots only)
- contact-width distribution: median + 10-90 span, share of the
  team's combined 10-90 width span  = "how much court they take"
- the team SPLIT LINE: midpoint of the two medians, vs finding 11's
  w = 0.41/0.59 prior
- median depth behind the kitchen line (with the height-bias band)

Honest scope: n = 4 rallies / one women's matchup (Nelson/Wei vs
Jones/Tuionetoa, Chicago 0725 game 1). A demo instrument for the
metric's shape, not a publishable read on any player.

Usage: python3 vision/contact_territory.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from court3d import load_landmarks, dlt  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "vision"
RALLIES = (1, 6, 7, 8)          # r9 excluded: designation pending
MATCH_S = 0.15                  # tap <-> ball-frame match window
NET_Y = 22.0                    # court is 20 x 44 ft, net at y=22
KITCHEN = 7.0                   # kitchen line is 7 ft from the net

TEAMS = {"Emma Nelson": "NEL/WEI", "Ting Chieh Wei": "NEL/WEI",
         "Allyce Jones": "JON/TUI", "Etta Tuionetoa": "JON/TUI"}


def load_contacts(rally):
    """[(t, hitter)] time-ordered; serve = ordinal 1, return = 2."""
    out = []
    if rally == 1:
        seen = []
        for r in csv.DictReader(open(DATA / "state_labels_chicago0725.csv")):
            if (int(r["rally_cum"]) == 1 and r["kind"] == "impact"
                    and r["player"] and r["t_s"]):
                t = float(r["t_s"])
                if all(abs(t - s) > 0.05 for s in seen):   # duplicate rows
                    seen.append(t)
                    out.append((t, r["player"]))
    else:
        for r in csv.DictReader(open(DATA / "contact_labels_chicago0725.csv")):
            if (int(r["rally_cum"]) == rally
                    and r["source"] in ("manual", "divergent")
                    and r.get("contact", "1") != "0" and r["hitter_name"]):
                out.append((float(r["t_refined_s"] or r["t_tap_s"]),
                            r["hitter_name"]))
    return sorted(out)


def ball_at(path_rows, t):
    best, bd = None, MATCH_S + 1
    for tt, x, y in path_rows:
        d = abs(tt - t)
        if d < bd:
            bd, best = d, (x, y)
    return best if bd <= MATCH_S else None


def ray_plane(P, px, py, z):
    """Intersect the pixel's back-projected ray with the z=const
    plane: solve P [X Y z 1]^T ~ [px py 1] for (X, Y)."""
    A = np.array([[P[0, 0] - px * P[2, 0], P[0, 1] - px * P[2, 1]],
                  [P[1, 0] - py * P[2, 0], P[1, 1] - py * P[2, 1]]])
    b = -np.array([P[0, 2] * z + P[0, 3] - px * (P[2, 2] * z + P[2, 3]),
                   P[1, 2] * z + P[1, 3] - py * (P[2, 2] * z + P[2, 3])])
    X, Y = np.linalg.solve(A, b)
    return float(X), float(Y)


def q(vals, p):
    return float(np.percentile(vals, p)) if vals else float("nan")


def main():
    X3, x2, _ = load_landmarks()
    P = dlt(X3, x2)

    hits = []          # (rally, ordinal, hitter, X0, Y0, X25, Y25)
    for rally in RALLIES:
        rows = [(float(r["t_s"]), float(r["x"]), float(r["y"]))
                for r in csv.DictReader(open(DATA / f"ball_path_r{rally}.csv"))
                if r["x"]]
        contacts = load_contacts(rally)
        miss = 0
        for i, (t, who) in enumerate(contacts, start=1):
            b = ball_at(rows, t)
            if b is None:
                miss += 1
                continue
            X0, Y0 = ray_plane(P, b[0], b[1], 0.0)
            X2, Y2 = ray_plane(P, b[0], b[1], 2.5)
            hits.append((rally, i, who, X0, Y0, X2, Y2))
        print(f"rally {rally}: {len(contacts)} taps, "
              f"{len(contacts) - miss} located"
              + (f" ({miss} no ball frame within {MATCH_S}s)" if miss
                 else ""))

    # which end does each team defend (constant within the game)
    end = {}
    for team in set(TEAMS.values()):
        ys = [h[4] for h in hits if TEAMS[h[2]] == team]
        end[team] = "far" if np.median(ys) > NET_Y else "near"

    def norm(team, X, Y):
        """Team-local frame: width 0-20 left-to-right FROM THAT
        TEAM'S VIEW; depth = feet behind the net (positive)."""
        if end[team] == "far":
            return 20.0 - X, Y - NET_Y
        return X, NET_Y - Y

    rally_shots = [h for h in hits if h[1] >= 3]
    print(f"\n{len(hits)} contacts located; {len(rally_shots)} rally "
          f"shots (serves/returns excluded from territory — "
          f"rule-fixed positions)")

    print("\n=== per player (rally shots, ground-plane read; depth "
          "band = z=2.5ft re-projection) ===")
    by_team = defaultdict(list)
    stats = {}
    for h in rally_shots:
        by_team[TEAMS[h[2]]].append(h)
    for team in sorted(by_team):
        th = by_team[team]
        print(f"\n  {team} ({end[team]} end, {len(th)} rally shots)")
        for who in sorted({h[2] for h in th}):
            ph = [h for h in th if h[2] == who]
            W = [norm(team, h[3], h[4])[0] for h in ph]
            D0 = [norm(team, h[3], h[4])[1] for h in ph]
            D2 = [norm(team, h[5], h[6])[1] for h in ph]
            w10, w50, w90 = q(W, 10), q(W, 50), q(W, 90)
            stats[who] = dict(n=len(ph), share=len(ph) / len(th),
                              w10=w10, w50=w50, w90=w90,
                              span=w90 - w10, d0=q(D0, 50), d2=q(D2, 50))
            s = stats[who]
            print(f"    {who:<16} {s['n']:>3} shots ({100*s['share']:.0f}% "
                  f"of team) | width med {w50:4.1f} ft, 10-90 "
                  f"[{w10:4.1f},{w90:4.1f}] span {s['span']:4.1f} | "
                  f"depth med {s['d0']:4.1f} ft behind net "
                  f"(z2.5: {s['d2']:4.1f})")
        pair = sorted({h[2] for h in th},
                      key=lambda p: stats[p]["w50"])
        if len(pair) == 2:
            a, b = pair
            split = (stats[a]["w50"] + stats[b]["w50"]) / 2
            wa = split / 20.0
            lo = min(stats[a]["w10"], stats[b]["w10"])
            hi = max(stats[a]["w90"], stats[b]["w90"])
            spansum = stats[a]["span"] + stats[b]["span"]
            print(f"    -> split line at {split:.1f} ft: {a} takes "
                  f"{100*wa:.0f}% / {b} {100*(1-wa):.0f}% of court "
                  f"width (v2 w-dial prior: 41/59 stronger-takes-59)")
            if spansum > 0:
                print(f"    -> width-span shares: {a} "
                      f"{100*stats[a]['span']/spansum:.0f}% / {b} "
                      f"{100*stats[b]['span']/spansum:.0f}% of the "
                      f"pair's covered width [{lo:.1f},{hi:.1f}]")

    print("\n=== serves + returns (rule-fixed; for the record) ===")
    for h in hits:
        if h[1] <= 2:
            team = TEAMS[h[2]]
            W, D = norm(team, h[3], h[4])
            kind = "serve " if h[1] == 1 else "return"
            print(f"    r{h[0]} {kind} {h[2]:<16} width {W:4.1f} "
                  f"depth {D:4.1f}")

    print("\nCaveats: n=4 rallies, one matchup, one camera. Depth "
          "wears the contact-height bias (z0 vs z2.5 columns); width "
          "is the trustworthy axis. Serves/returns excluded above. "
          "A licensed tracker mints this for every logged rally; "
          "the coverage model measures the same question from pose.")


if __name__ == "__main__":
    main()
