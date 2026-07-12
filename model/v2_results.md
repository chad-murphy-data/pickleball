# v2 model results (dynamic skill + race-to-T likelihood + weakest link)

One joint 2024-2026 fit, 37k games (to-15 Challenger rounds included
natively), monthly random-walk skill for >=60-game players, per-point
Binomial race likelihood. 0 divergences, max rhat 1.06.

## Validation gate — PASSED
Frozen before 2026-06-01, predicting all later games (n=884, more than v1's
686 because v2 knows 2024-25 players and handles to-15 games):

| model | accuracy | Brier | log loss |
|:--|--:|--:|--:|
| v2 | **77.4%** | **0.165** | 0.506 |
| v1 (Gaussian, static, 2026-only) | 75.2% | 0.178 | 0.536 |
| DUPR (as-of-match) | 64.7% | 0.229 | — |

## Headline scalar results (per-point logit scale)
- gamma (weakest link) = -0.183 ± 0.047: survives removal of scoreboard
  truncation at ~half its Gaussian-model size — half real targeting, half
  ceiling artifact, now separated.
- sd_d (chemistry) = 0.053 ± 0.011 (~1/3 point per game): small, again.
- tau = 0.038/month skill drift; sd_v = 0.38.
- beta_new = +0.088 ± 0.011: new pairings overperform their first ~6 games,
  but pairs predating Jan 2024 have window-edge games mislabeled "new" —
  keep the asterisk.

## Trajectory revisions (the dynamic model's payoff)
- Ben Johns NEVER declined in absolute terms (+1.02 -> +1.11 over 2.5 yrs);
  the field's top rose to meet him. "Decline of Johns" was pool drift.
- Tardio and Patriquin climbed smoothly (+0.29 and +0.25) — no single
  breakout, a continuous ascent.
- Waters is still pulling away: steepest curve in the data; her gap over
  Bright nearly doubled since early 2024.
- Parenteau is a genuine absolute decliner (-0.20 over two years).

Common-to-everyone drift is unidentified (margins are relative); curves are
movement relative to the average tour regular.

Artifacts: data/v2_players.csv (current form), data/v2_trajectories.csv
(monthly curves), data/v2_dyads.csv, model/v2_fit_summary.json,
model/v2_holdout_summary.json.
