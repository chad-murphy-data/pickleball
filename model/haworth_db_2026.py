"""Is Haworth's 2026 DreamBreaker record bad luck or bad play? First pass.

User question (2026-08-24, late-night, "just get this started"): Haworth's
DreamBreaker results have felt bad this season. Given he's presumably
facing strong opposition, what should we expect (singles model), what did
he actually do, and can we tell chance from a real deficit?

Data path (see model/haworth_db_2026.md for the full writeup):
  - data/dreambreakers.csv: all MLP DreamBreaker match records (score only).
  - Brooklyn's DBs found via matchup_id -> data/mlp_matchups_2026.csv join.
  - Player-level rally attribution: reconstructed live from referee logs
    (scraper/pb_api.PBClient.match_logs). Two log schemas coexist in 2026:
      * OLD (pre ~7/23): rotating 4-player lineup, subs are separate
        log_type=32 rows -> need model/db_impute.py's parse_db to track
        who's on court (already cached in data/db_rallies.csv for 6/28).
      * NEW (7/23 onward, observed here): log_type=14 POINT rows carry
        server_uuid/receiver_uuid INLINE, no log_type=32 rows at all, and
        the whole DB is ONE continuous 1v1 (no rotation) -- both new-schema
        matches validate exactly against the official final score using
        just the type-14 rows. This looks like a genuine rule change
        (rotating quartet -> single designated champion) somewhere around
        7/23; the schema change is presumably downstream of that (no more
        need to log substitutions if nobody substitutes mid-DB).
      This script re-fetches the two NEW-schema matches live (network) and
      reuses the cached OLD-schema reconstruction for 6/28. A 4th Brooklyn
      DB (7/9, lost 21-23) has NO digital referee log at all and is
      excluded -- so this is a lower bound on the sample, not the full
      season.
  - Expectation: db_model.md's singles-value rally model, P(win) =
    sigmoid(K * (Haworth_singles - opponent_singles)), K = 0.42 (the
    house-fit rally-level coefficient, held fixed here -- refitting K on
    Haworth's own 78 rallies would be circular).

Method: 1-parameter logistic MLE for a Haworth-specific additive offset
delta on top of the fixed-K singles-gap baseline (same shape as the
clutch/gap-exploit player-level estimators elsewhere in this repo).
delta's SE comes from the fitted Hessian; z and a flat-prior normal-approx
posterior P(delta<0 | data) are both reported, NOT just a p-value, per the
project's usual "how confident, not just significant" framing.

Run: python model/haworth_db_2026.py  (hits the network for 2 fresh log
fetches; ~1s, logs are small and cached).
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scraper"))
sys.path.insert(0, str(ROOT / "model"))
from pb_api import PBClient                    # noqa: E402
from harvest_logs import _point_delta          # noqa: E402
from db_impute import load_players             # noqa: E402

HAWORTH = "a91e2b68-7357-4518-baf9-f59b0b4c2477"
BROOKLYN = "fe9c66f9-c52c-4153-89bc-207d3bb5d934"
K = 0.42  # db_model.md v2 (singles-gap) rally-level fit

NEW_SCHEMA_MATCHES = {
    "3c5e3a2b-70e2-4b35-8f9d-bd28d3445ef1": "2026-07-23",  # Brooklyn 21-14 Miami
    "309943a0-bd66-4575-90c9-7a23d9607691": "2026-07-26",  # Dallas 21-10 Brooklyn
}
OLD_SCHEMA_MATCH = "52469f10-1b72-4a61-9f84-28bfd7048485"   # 2026-06-28, cached
OLD_OPPONENT = "bd061b42-2f82-4bbe-82be-d8bcfb37f9e1"        # JW Johnson


def reconstruct_new_schema(mid, db_by_mid, client):
    rows = sorted(client.match_logs(mid) or [], key=lambda r: r.get("log_index", 0))
    md = client.matchup_data(db_by_mid[mid]["matchup_id"], volatile=False)
    team1 = (md.get("teamOneUuid") or "").lower()
    team2 = (md.get("teamTwoUuid") or "").lower()
    totals = {team1: 0, team2: 0}
    rallies = []
    for r in rows:
        if r.get("log_type") != 14:
            continue
        delta, team = _point_delta(r)
        if team not in totals:
            continue
        end = (r.get("point_log") or {}).get("end_score")
        if delta >= 0 and end is not None:
            if end == totals[team] + 1:
                delta = 1
            elif end <= totals[team]:
                delta = 0
            else:
                gap = end - totals[team] - 1
                totals[team] += gap
                delta = 1
        if delta > 0:
            srv = (r.get("server_uuid") or "").lower()
            rcv = (r.get("receiver_uuid") or "").lower()
            for _ in range(delta):
                rallies.append((srv, rcv, team))
                totals[team] += 1
        elif delta < 0:
            for _ in range(-delta):
                for j in range(len(rallies) - 1, -1, -1):
                    if rallies[j][2] == team:
                        rallies.pop(j)
                        totals[team] -= 1
                        break
    return totals, rallies


def main():
    sv, sg, gen, dbl, names = load_players()

    def singles_value(u):
        if u in sv and sg.get(u, 0) >= 10:
            return sv[u]
        if u in dbl:
            return 0.28 + 1.14 * dbl[u]          # make_forecast's un-shrunk imputation
        return sv.get(u)

    db_by_mid = {r["match_id"]: r for r in csv.DictReader(open(ROOT / "data" / "dreambreakers.csv"))}
    client = PBClient()

    h_rallies = []  # (opponent_uuid, won: bool, date)
    old = [r for r in csv.DictReader(open(ROOT / "data" / "db_rallies.csv"))
           if r["match_id"] == OLD_SCHEMA_MATCH and r["player_team1"] == HAWORTH]
    for r in old:
        h_rallies.append((OLD_OPPONENT, bool(int(r["team1_won"])), "2026-06-28"))

    for mid, date in NEW_SCHEMA_MATCHES.items():
        totals, rallies = reconstruct_new_schema(mid, db_by_mid, client)
        official = db_by_mid[mid]
        assert (totals == {} or
                sorted(totals.values()) == sorted([int(official["t1_score"]), int(official["t2_score"])])), \
            f"score mismatch on {mid}: {totals} vs official"
        for s, rcv, tm in rallies:
            if s == HAWORTH or rcv == HAWORTH:
                opp = rcv if s == HAWORTH else s
                h_rallies.append((opp, tm == BROOKLYN, date))

    hv = singles_value(HAWORTH)
    print(f"Haworth singles value: {hv:.4f} ({sg.get(HAWORTH)} games, ranked)")
    print(f"{len(h_rallies)} rallies across {len({d for _, _, d in h_rallies})} reconstructed "
          f"DreamBreakers (a 4th, 7/9 loss to Chicago, has no digital referee log)\n")

    by_match = {}
    for opp, won, date in h_rallies:
        by_match.setdefault(date, []).append((opp, won))
    for date, rows in sorted(by_match.items()):
        opp = rows[0][0]
        ov = singles_value(opp)
        gap = hv - ov
        p = 1 / (1 + math.exp(-K * gap))
        w = sum(1 for _, won in rows if won)
        n = len(rows)
        print(f"  {date}  vs {names.get(opp, opp):20} (opp singles {ov:.3f}, gap {gap:+.3f}, "
              f"model p={p:.3f})  actual {w}-{n - w} ({w / n:.3f})")

    gaps = [hv - singles_value(opp) for opp, won, _ in h_rallies]
    ys = [1 if won else 0 for _, won, _ in h_rallies]

    delta, H = 0.0, 1.0
    for _ in range(100):
        g = H = 0.0
        for gap, y in zip(gaps, ys):
            p = 1 / (1 + math.exp(-(K * gap + delta)))
            g += y - p
            H += p * (1 - p)
        step = g / H if H > 1e-12 else 0.0
        delta += step
        if abs(step) < 1e-10:
            break
    se = 1 / math.sqrt(H)
    z = delta / se
    p_two = math.erfc(abs(z) / math.sqrt(2))
    post_below0 = 0.5 * (1 + math.erf((0 - delta) / (se * math.sqrt(2))))

    print(f"\nCombined observed win rate: {sum(ys)}/{len(ys)} = {sum(ys) / len(ys):.3f}")
    print(f"Fitted Haworth DB offset (K={K} fixed): delta = {delta:+.3f} +/- {se:.3f} logit")
    print(f"  z = {z:.2f}, two-sided p = {p_two:.3f}")
    print(f"  95% CI: [{delta - 1.96 * se:+.3f}, {delta + 1.96 * se:+.3f}]")
    print(f"  flat-prior P(true offset < 0 | data) = {post_below0:.3f}")
    print(f"  scale check: {delta:+.3f} logit ~= {(1 / (1 + math.exp(-delta)) - 0.5) * 100:+.1f}pp "
          f"win prob at an even matchup")


if __name__ == "__main__":
    main()
