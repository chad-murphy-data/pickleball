"""Replay ONE real game rally-by-rally and grade skill-only vs skill+clutch.

The game: MLP Orlando playoff, 2026-08-02, STL vs NJ, mixed doubles —
Anna Bright / Hayden Patriquin vs Anna Leigh Waters / Noe Khlif, 13-15.
A to-11 win-by-two that ran five points deep into deuce, so it passes
through 9-9 and every tied state above it: the exact situation this whole
thread started from.

Two things this answers that the aggregate work cannot:

1. **Does the clutch rating actually price this game better than skill
   alone?**  Both models produce a per-rally P(serving side wins); score
   them by log-loss on the 134 real rallies.  n is tiny — this is an
   illustration on one game, NOT a validation — but it is the real thing
   rather than a simulation.
2. **What did each model say at 9-9**, and what happened.

Clutch enters exactly as the regime model defines it: each rally is
classified high- or low-leverage by its within-game standardised leverage,
and the four players contribute their fitted level for THAT regime.  No new
machinery — the ratings are read from data/clutch_regimes_unanchored.csv.

Run: python model/replay_mxd_clutch.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "scraper"))

from sitelib import race                                          # noqa: E402
from sitelib.winprob import (A1, A2, B1, B2, ServeDP, _table,      # noqa: E402
                             eta_anchor, serve_probs)
from pb_api import PBClient                                        # noqa: E402

MATCH = "900ceede-b4fb-44d6-80b8-e95543b13a9b"
K_LEAGUE = 0.443
RALLY_START, POINT, SIDEOUT, SWITCH, SECOND = 12, 14, 16, 17, 23

MATCHUP = "1c53dea3-1f7a-4741-be32-8e6b63157121"   # STL vs NJ playoff


def lineup(client):
    """The four uuids, READ FROM THE MATCH RECORD.  Hardcoding them is how you
    end up with Kate Fahey standing in for Hayden Patriquin — the women's
    doubles pair sits two matches earlier in the same payload."""
    d = client.matchup_data(MATCHUP, volatile=False)
    m = next(x for x in d["matches"] if x["matchUuid"] == MATCH)
    s1 = [m["teamOnePlayerOneUuid"].lower(), m["teamOnePlayerTwoUuid"].lower()]
    s2 = [m["teamTwoPlayerOneUuid"].lower(), m["teamTwoPlayerTwoUuid"].lower()]
    return s1, s2, m


def parse(logs, S1, S2):
    """(server_score, receiver_score, server_no, server_uuid, receiver_uuid,
    serving_side, won) per rally, from the referee log."""
    rows = logs.get("data") if isinstance(logs, dict) else logs
    out = []
    for r in sorted(rows, key=lambda x: x.get("log_index", 0)):
        if r.get("log_type") not in (POINT, SIDEOUT, SECOND):
            continue
        s = r.get("start_score_current_game_string")
        if not s:
            continue
        try:
            a, b, n = (int(x) for x in s.split("-"))
        except ValueError:
            continue
        srv = (r.get("server_uuid") or "").lower()
        rcv = (r.get("receiver_uuid") or "").lower()
        if not srv:
            continue
        side = 0 if srv in S1 else 1
        out.append((a, b, n, srv, rcv, side, 1 if r["log_type"] == POINT else 0))
    return out


def leverage(V, T, a, b, state, side_A_serving):
    def val(aa, bb, ss):
        if aa >= T and aa - bb >= 2:
            return 1.0
        if bb >= T and bb - aa >= 2:
            return 0.0
        return V.get((aa, bb, ss), 0.5)
    if side_A_serving:
        w, l = val(a + 1, b, state), (val(a, b, 1) if state == 0 else val(a, b, 2))
    else:
        w, l = val(a, b + 1, state), (val(a, b, 3) if state == 2 else val(a, b, 0))
    return abs(w - l)


def main():
    cal = json.loads((ROOT / "web" / "calibration.json").read_text())
    race.set_calibration(cal["a"], cal["b"], cal["eps"])
    v2 = {r["player_id"].lower(): (r["full_name"], float(r["value_now_mean"]))
          for r in csv.DictReader((ROOT / "data" / "v2_players.csv").open())}
    cl = {r["uuid"].lower(): r for r in
          csv.DictReader((ROOT / "data" / "clutch_regimes_components.csv").open())}

    client = PBClient()
    S1, S2, rec = lineup(client)
    logs = client.match_logs(MATCH)
    rallies = parse(logs, S1, S2)
    print(f"Parsed {len(rallies)} rallies from the referee log.\n")

    nm = {u: v2[u][0] for u in S1 + S2}
    v = [v2[u][1] for u in S1 + S2]
    eta = race.team_eta(v[0], v[1], v[2], v[3])
    p0 = race.calibrate(race.game_win_prob_uncertain(eta, race.SD_MATCH, 11))
    print(f"  STL {nm[S1[0]]}/{nm[S1[1]]}   vs   NJ {nm[S2[0]]}/{nm[S2[1]]}")
    print(f"  v2 values: {v[0]:+.3f}/{v[1]:+.3f}  vs  {v[2]:+.3f}/{v[3]:+.3f}")
    print(f"  pre-game, skill only: STL {100 * p0:.1f}%\n")

    kA, kB = serve_probs(eta_anchor(p0), K_LEAGUE)
    V = _table(round(kA, 6), round(kB, 6), 11, 51)

    # within-game standardised leverage, exactly as the ratings were fitted
    levs = []
    for (a, b, n, srv, rcv, side, won) in rallies:
        state = (0 if n == 1 else 1) if side == 0 else (2 if n == 1 else 3)
        aa, bb = (a, b) if side == 0 else (b, a)
        levs.append(leverage(V, 11, aa, bb, state, side == 0))
    mu = sum(levs) / len(levs)
    sd = (sum((x - mu) ** 2 for x in levs) / len(levs)) ** 0.5
    levz = [(x - mu) / sd for x in levs]
    # the fit's own global cut: top quartile of levz is "big"
    cut = sorted(levz)[int(0.75 * len(levz))]

    def clutch_p(side, hi):  # noqa: E306
        """P(serving side wins this rally) under the regime model — the SAME
        algebra the fit used: serving pair's serve-levels add, receiving
        pair's return-levels subtract, on top of that regime's league rate.

        The earlier version halved a combined serve+return column, which is
        not the model: the CSV's `regular`/`big` are mL+nL and mH+nH, already
        summed across the two roles and not separable back out. Hence
        clutch_regimes_components.csv, which keeps all four.
        """
        srv_pair = S1 if side == 0 else S2
        rcv_pair = S2 if side == 0 else S1
        sc = "serve_big" if hi else "serve_regular"
        rc = "return_big" if hi else "return_regular"
        base = math.log(0.4001 / 0.5999) if hi else math.log(0.4418 / 0.5582)
        e = (base
             + sum(float(cl[u][sc]) for u in srv_pair if u in cl)
             - sum(float(cl[u][rc]) for u in rcv_pair if u in cl))
        return 1 / (1 + math.exp(-e))

    ll_skill = ll_clutch = 0.0
    at99 = []
    print(f"  {'score':>8} {'srv':>4} {'server':<20}{'levz':>7}{'skill':>8}"
          f"{'clutch':>8}  result")
    traj = []
    for i, (a, b, n, srv, rcv, side, won) in enumerate(rallies):
        hi = levz[i] > cut
        ps = kA if side == 0 else kB
        pc = clutch_p(side, hi)
        ll_skill += math.log(max(ps if won else 1 - ps, 1e-12))
        ll_clutch += math.log(max(pc if won else 1 - pc, 1e-12))
        s1s, s2s = (a, b) if side == 0 else (b, a)
        traj.append((s1s, s2s))
        if s1s >= 9 and s2s >= 9:
            at99.append((s1s, s2s, nm.get(srv, srv[:8]), side, won, ps, pc, levz[i]))

    # verify the trajectory against the remembered one
    seen = []
    for s in traj:
        if not seen or seen[-1] != s:
            seen.append(s)
    marks = [s for s in seen if s in ((6, 0), (7, 10), (11, 10))]
    print(f"\n  trajectory check — reached {marks} ; final "
          f"{max(t[0] for t in traj)}-{max(t[1] for t in traj)} plus the "
          f"winning point")
    print(f"  first 24 distinct scores (STL-NJ): "
          f"{' '.join(f'{x}-{y}' for x, y in seen[:24])}")

    print(f"\n{'=' * 70}\nFIT ON THE REAL RALLIES (log-loss, higher = better)"
          f"\n{'=' * 70}")
    print(f"  skill only     {ll_skill / len(rallies):+.4f} per rally")
    print(f"  skill + clutch {ll_clutch / len(rallies):+.4f} per rally"
          f"   ({'BETTER' if ll_clutch > ll_skill else 'WORSE'} by "
          f"{abs(ll_clutch - ll_skill) / len(rallies):.4f}/rally, "
          f"{abs(ll_clutch - ll_skill):.2f} total)")
    print(f"  n={len(rallies)} rallies on ONE game — an illustration, "
          "not a validation.")

    print(f"\n{'-' * 70}\nEVERY RALLY WITH BOTH SIDES ON 9+ ({len(at99)} of them)"
          f"\n{'-' * 70}")
    print(f"  {'STL-NJ':>7} {'server':<20}{'levz':>6}{'P(hold) skill':>15}"
          f"{'clutch':>9}  won?")
    for (x, y, who, side, won, ps, pc, lz) in at99:
        print(f"  {x:>3}-{y:<3} {who:<20}{lz:+6.2f}{100 * ps:>14.1f}%"
              f"{100 * pc:>8.1f}%   {'YES' if won else 'no'}")

    dp = ServeDP(eta_anchor(p0), K_LEAGUE, 11)
    print(f"\n  DP at 9-9, STL serving (#2): STL wins {100 * dp.p(9, 9, A2):.1f}%")
    print(f"  DP at 9-9, NJ serving  (#2): STL wins {100 * dp.p(9, 9, B2):.1f}%")
    print(f"  ACTUAL: NJ won 15-13.")


if __name__ == "__main__":
    main()
