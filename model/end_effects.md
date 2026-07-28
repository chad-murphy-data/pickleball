# End effects (good side / bad side) — inferred from switch structure

Levels are NOT interpretable alone (they mix skill continuity with end effects); read the CONTRASTS between rows. End effects push every statistic DOWN.

## Design A — corr of game-1 vs game-2 residual margins (same 4 players, ends switched)

| group | matches | corr(r1, r2) [95% CI] | cov (pts²) |
|---|---|---|---|
| INDOOR (all wind — no exposure expected) | 4152 | +0.066 [+0.039, +0.091] | +1.34 |
| OUTDOOR calm <8 mph | 2970 | +0.028 [-0.022, +0.080] | +0.56 |
| OUTDOOR moderate 8–14 | 5096 | +0.071 [+0.038, +0.104] | +1.51 |
| OUTDOOR windy 14+ | 1641 | +0.058 [+0.000, +0.129] | +1.22 |

cov(calm) − cov(windy) estimates Var(end adv | windy) − Var(end adv | calm) in points²; its square root is the typical per-game end advantage the wind adds.

## Design B — decider game 3: point share before vs after the end switch at 6

| group | deciders | corr(pre, post) [95% CI] |
|---|---|---|
| INDOOR (all wind — no exposure expected) | 884 | +0.335 [+0.292, +0.377] |
| OUTDOOR calm <8 mph | 616 | +0.355 [+0.265, +0.442] |
| OUTDOOR moderate 8–14 | 990 | +0.333 [+0.283, +0.382] |
| OUTDOOR windy 14+ | 340 | +0.347 [+0.266, +0.444] |

Binomial noise in the point shares attenuates all rows toward zero equally; again, read contrasts, not levels.

---
*Caveats: day-level wind (attenuates the windy-vs-calm contrast); indoor/outdoor labels heuristic; Design A assumes match-level form variance is similar across groups. Hour-level wind (data/match_times.csv + event_weather_hourly.csv) is the designed upgrade.*
