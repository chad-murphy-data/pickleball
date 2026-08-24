"""Is Haworth's 2026 DreamBreaker record bad luck or bad play? First pass.

User question (2026-08-24, late-night, "just get this started"): Haworth's
DreamBreaker results have felt bad this season. Given he's presumably
facing strong opposition, what should we expect (singles model), what did
he actually do, and can we tell chance from a real deficit?

CORRECTION (same session, same day): the first version of this script
claimed a mid-season DreamBreaker format change (rotating quartet ->
single 1v1 champion) based on the 7/23 and 7/26 matches showing only two
player uuids in the referee logs the whole way through. That claim was
WRONG -- caught by the user ("it was always rotating, we just calculated
it wrong"). Checked: the matchup record's own tieBreaker rotation config
(tieBreakerTeamRotation = TEAM_ROTATION_COMBINED_SCORE_EQUALS_POINTS,
tieBreakerRotationCombinedPoints = 4 -- rotate every 4 combined points) is
IDENTICAL between 6/28 (confirmed rotating via explicit log_type=32 SUB
events) and 7/23 / 7/26 (zero log_type=32 rows). Rotation was never
switched off. What actually happened: MLP's newer referee-log schema
(observed from ~7/23 on) stopped emitting substitution events entirely,
so the server_uuid/receiver_uuid fields on POINT rows stay pinned to
the OPENING pair for the whole match instead of tracking real
substitutions -- a logging gap, not a rule change. That means the two
new-schema matches CANNOT be attributed to individual players from this
log stream: there is no signal distinguishing "Haworth played the whole
DB" from "Haworth played 25% of it and the log just never updated."

Net effect: only the 6/28 match (12 rallies, old schema, SUB-log
validated) is usable player-level data. That collapses the sample from
78 rallies to 12 -- nowhere near enough to say anything, which is exactly
what db_model.md already warned ("player-level DB effects are hopeless at
this n"). This script now reports ONLY the 6/28 match and is explicit
about why the other two are excluded.

Run: python model/haworth_db_2026.py  (offline -- reuses the cached
data/db_rallies.csv reconstruction, no network needed since the
new-schema matches are no longer fetched/used).
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scraper"))
sys.path.insert(0, str(ROOT / "model"))
from db_impute import load_players             # noqa: E402

HAWORTH = "a91e2b68-7357-4518-baf9-f59b0b4c2477"
OLD_SCHEMA_MATCH = "52469f10-1b72-4a61-9f84-28bfd7048485"   # 2026-06-28, SUB-log validated
OLD_OPPONENT = "bd061b42-2f82-4bbe-82be-d8bcfb37f9e1"        # JW Johnson
K = 0.42  # db_model.md v2 (singles-gap) rally-level fit

EXCLUDED_NO_SUB_LOG = {
    "3c5e3a2b-70e2-4b35-8f9d-bd28d3445ef1": "2026-07-23 vs Miami -- new log schema, "
        "no log_type=32 SUB rows anywhere in the log, so on-court identity beyond the "
        "opening pair (Haworth/Delgado) cannot be verified even though rotation is "
        "configured on (see module docstring). NOT usable for player attribution.",
    "309943a0-bd66-4575-90c9-7a23d9607691": "2026-07-26 vs Dallas -- same issue.",
}


def main():
    sv, sg, gen, dbl, names = load_players()

    def singles_value(u):
        if u in sv and sg.get(u, 0) >= 10:
            return sv[u]
        if u in dbl:
            return 0.28 + 1.14 * dbl[u]
        return sv.get(u)

    hv = singles_value(HAWORTH)
    print(f"Haworth singles value: {hv:.4f} ({sg.get(HAWORTH)} games, ranked)\n")

    print("Excluded matches (rotation configured on, but no substitution log to verify "
          "who was actually on court -- see docstring):")
    for mid, why in EXCLUDED_NO_SUB_LOG.items():
        print(f"  {mid}: {why}")
    print()

    rows = [r for r in csv.DictReader(open(ROOT / "data" / "db_rallies.csv"))
            if r["match_id"] == OLD_SCHEMA_MATCH and r["player_team1"] == HAWORTH]
    ys = [int(r["team1_won"]) for r in rows]
    n = len(ys)
    w = sum(ys)

    ov = singles_value(OLD_OPPONENT)
    gap = hv - ov
    p = 1 / (1 + math.exp(-K * gap))
    print(f"2026-06-28 vs {names.get(OLD_OPPONENT, OLD_OPPONENT)} (opp singles {ov:.3f}, "
          f"gap {gap:+.3f}, model p={p:.3f}): actual {w}-{n - w} ({w / n:.3f}), "
          f"model-expected {p * n:.2f}/{n}")

    delta, H = 0.0, 1.0
    for _ in range(100):
        eta = K * gap + delta
        pp = 1 / (1 + math.exp(-eta))
        g = n * (w / n - pp)
        H = n * pp * (1 - pp)
        step = g / H if H > 1e-12 else 0.0
        delta += step
        if abs(step) < 1e-10:
            break
    se = 1 / math.sqrt(H)
    z = delta / se
    p_two = math.erfc(abs(z) / math.sqrt(2))
    print(f"\nFitted Haworth DB offset (K={K} fixed, n={n} rallies, ONE opponent): "
          f"delta = {delta:+.3f} +/- {se:.3f} logit")
    print(f"  z = {z:.2f}, two-sided p = {p_two:.3f}")
    print(f"  95% CI: [{delta - 1.96 * se:+.3f}, {delta + 1.96 * se:+.3f}]")
    print("\nn=12 against a single opponent is not enough to say anything about Haworth's "
          "DreamBreaker skill either way -- see model/haworth_db_2026.md for the honest "
          "bottom line and what would actually resolve the question.")


if __name__ == "__main__":
    main()
