# End effects (good side / bad side) — inferred from switch structure

Wind source: **at the match start hour (data/match_times.csv)**.

Levels are NOT interpretable alone (they mix skill continuity with end effects); read the CONTRASTS between rows. End effects push every statistic DOWN.

## Design A — corr of game-1 vs game-2 residual margins (same 4 players, ends switched)

| group | matches | corr(r1, r2) [95% CI] | cov (pts²) |
|---|---|---|---|
| INDOOR (all wind — no exposure expected) | 4152 | +0.066 [+0.039, +0.091] | +1.34 |
| OUTDOOR calm <8 mph | 6623 | +0.048 [+0.017, +0.078] | +1.00 |
| OUTDOOR moderate 8–14 | 2603 | +0.069 [+0.021, +0.119] | +1.44 |
| OUTDOOR windy 14+ | 481 | +0.091 [+0.012, +0.192] | +1.94 |

cov(calm) − cov(windy) estimates Var(end adv | windy) − Var(end adv | calm) in points²; its square root is the typical per-game end advantage the wind adds.

## Design B — decider game 3: point share before vs after the end switch at 6

| group | deciders | corr(pre, post) [95% CI] |
|---|---|---|
| INDOOR (all wind — no exposure expected) | 884 | +0.335 [+0.292, +0.377] |
| OUTDOOR calm <8 mph | 1299 | +0.357 [+0.305, +0.411] |
| OUTDOOR moderate 8–14 | 536 | +0.300 [+0.235, +0.372] |
| OUTDOOR windy 14+ | 111 | +0.368 [+0.227, +0.531] |

Binomial noise in the point shares attenuates all rows toward zero equally; again, read contrasts, not levels.

---
*Caveats: indoor/outdoor labels heuristic; Design A assumes match-level form variance is similar across groups.*
