# One-off partnerships pooled: familiarity, and forced vs chosen (2026-10-10)

Reproduce: `python model/chem_oneoff.py` (out-of-sample, post-Jun-1 vs frozen pre-June
fit) and `python model/chem_oneoff.py --full` (all 2024-26 vs month-of-game ratings).
Side residual = side point share - sigmoid(eta), x1e-3, cluster-robust SE by event.
Other side restricted to settled pairs (>=15 games together). Familiarity = prior
games together, counted from the past only (forward-clean; a stopping rule can't bias it).

## Familiarity curve (full history, in-sample monthly ratings)

| prior games together | n | residual |
|---|---|---|
| 0 (first-ever) | 899 | -7.4 +/- 5.4 |
| 1-2 | 1,905 | -12.0 +/- 3.8 |
| 3-5 | 2,611 | -5.4 +/- 3.6 |
| 6-14 | 4,185 | -7.7 +/- 2.7 |
| 15-39 | 3,742 | -2.3 +/- 1.8 |
| 40+ | 4,270 | +2.0 +/- 1.6 |

A smooth gradient: pairs with under ~15 games together run ~0.7-1.2pp of point
share below their ratings; gone by 15-39, flat by 40+. Same size and sign as the
chem_survival.md scratch penalty (-8 +/- 3.7), now with a dose-response shape
(~0.15 pts/game, ~1-1.5pp of game win probability).

## Forced (usual partner absent from the event) vs chosen (usual partner playing elsewhere)

First-ever pairings: FORCED n=380 -9.9 +/- 7.5; CHOSEN n=439 -3.3 +/- 7.8.
Difference -6.6 +/- 10.8: NOT distinguishable. Out-of-sample post-June alone is
n=113 first-ever sides (+/-15-20): too thin to say anything.

## Reading

- Generic FAMILIARITY (more games together -> better) is estimable by pooling and
  shows up as a small, smooth penalty that heals over ~15 games.
- Pair-SPECIFIC chemistry (this pair is special) remains uncertifiable (finding 2).
- Caveats: ratings are in-sample and track form (tracking bias is negative at low
  tenure, per chem_survival.md), so the curve's level is approximate; the
  forced-vs-chosen contrast is underpowered. Candidate v2 cleanup stays as in
  finding 5: replace beta_new (selection-shaped, positive) with an unconditional
  familiarity ramp; gate on the holdout.
