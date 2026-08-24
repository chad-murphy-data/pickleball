"""Is Haworth's 2026 DreamBreaker record bad luck or bad play? First pass.

User question (2026-08-24, late-night, "just get this started"): Haworth's
DreamBreaker results have felt bad this season. Given he's presumably
facing strong opposition, what should we expect (singles model), what did
he actually do, and can we tell chance from a real deficit?

HOW THIS ANALYSIS GOT HERE (four passes, same session -- read before
touching this again). DreamBreakers rotate 4 players per team every 4
combined points; this is a MANDATORY rule, not a team option (confirmed
by the user directly -- there is no "team chooses not to rotate" mode).
Haworth's 6/28 DB shows the rotation properly logged: explicit
log_type=32 substitution events name all 4 players per side, and the
reconstruction validates exactly against the official score. His 7/23
and 7/26 DBs show ZERO log_type=32 events and only 2 distinct player
uuids each, ALSO matching the official final score exactly.
  Pass 1: read that as a mid-season format change to single-champion
    1v1. Wrong -- there's no such format; rotation is mandatory.
  Pass 2: read the missing log_type=32 rows as a referee-logging schema
    regression, and excluded both matches. Right call (exclude), wrong
    mechanism.
  Pass 3: found that OTHER post-7/23 DreamBreakers -- including one
    worked by the SAME referee who officiated Brooklyn's 7/26 game --
    show full 8-player rotation logged correctly. Read that as "so
    Brooklyn's games genuinely didn't rotate" (team discretion). WRONG:
    rotation is mandatory, so this reasoning doesn't hold no matter how
    clean the log looks.
  Pass 4 (current, correct): rotation happened in both 7/23 and 7/26 (it
    has to -- it's not optional), but the referee logs for those two
    matches simply FAILED to capture it -- a logging/attribution error,
    not a data artifact that can be trusted just because the final score
    happens to validate. A log can be internally consistent (clean
    side-outs, exact score match) and still be wrong about WHO was on
    court; score validation only proves the running total is right, not
    that server_uuid/receiver_uuid tracked real substitutions. There is
    currently no reliable way to recover true on-court identity for
    these two matches from this log stream.

Net: ONLY 6/28 (12 rallies, one opponent) is trustworthy player-level
data. 7/23, 7/26, and 7/09 (no digital log at all) are all excluded.
That's a genuinely small sample -- exactly what db_model.md already
warned ("player-level DB effects are hopeless at this n").

Expectation: db_model.md's singles-value rally model, P(win) =
sigmoid(K * (Haworth_singles - opponent_singles)), K = 0.42 (the
house-fit rally-level coefficient, held fixed here -- refitting K on
Haworth's own rallies would be circular).

Method: 1-parameter logistic MLE for a Haworth-specific additive offset
delta on top of the fixed-K singles-gap baseline. delta's SE comes from
the fitted Hessian; z and a flat-prior normal-approx posterior
P(delta<0 | data) are both reported, not just a p-value.

Run: python model/haworth_db_2026.py (offline -- reuses the cached
data/db_rallies.csv reconstruction of the 6/28 match only).
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

EXCLUDED = {
    "2026-07-09 vs John Lucian Goins (L 21-23 team)": "no digital referee log at all",
    "2026-07-23 vs James Delgado (W 21-14 team)": "log exists but has no substitution "
        "events; rotation is mandatory so this can't be a real 1v1, meaning the log failed "
        "to capture who was actually on court",
    "2026-07-26 vs JW Johnson rematch (L 10-21 team)": "same issue as 7/23",
}


def main():
    sv, sg, gen, dbl, names = load_players()

    def singles_value(u):
        if u in sv and sg.get(u, 0) >= 10:
            return sv[u]
        if u in dbl:
            return 0.28 + 1.14 * dbl[u]          # make_forecast's un-shrunk imputation
        return sv.get(u)

    hv = singles_value(HAWORTH)
    print(f"Haworth singles value: {hv:.4f} ({sg.get(HAWORTH)} games, ranked)\n")

    print("Excluded (rotation is mandatory, so any match that isn't SUB-log-verified "
          "cannot be trusted for player attribution, however clean the log otherwise looks):")
    for k, why in EXCLUDED.items():
        print(f"  {k}: {why}")
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
