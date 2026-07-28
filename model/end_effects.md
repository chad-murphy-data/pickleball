# End effects (good side / bad side) — inferred from switch structure

Wind source: **at the match start hour (data/match_times.csv)**.

End effects push the paired-swing VARIANCE up (primary tables) and the correlations down (secondary). Levels are contaminated — Design A by game noise, Design B by serve-streak clustering — so read CONTRASTS between rows; the contaminants have no reason to vary with wind.

## Design A — game 1 vs game 2 (same 4 players, ends switched between games)

Primary: variance of the paired swing d = margin(g1) − margin(g2) (skill cancels; Var(d) = 4·Var(end adv) + 2·Var(game noise)).

| group | matches | Var(d) pts² [95% CI] | vs calm: implied end-adv sd (pts/game) |
|---|---|---|---|
| INDOOR (all wind — no exposure expected) | 4152 | 38.05 [36.25, 39.63] | ≤0 (Var Δ -1.54) |
| OUTDOOR calm <8 mph | 6623 | 39.59 [38.32, 40.89] | — (baseline) |
| OUTDOOR moderate 8–14 | 2603 | 38.70 [36.25, 41.07] | ≤0 (Var Δ -0.89) |
| OUTDOOR windy 14+ | 481 | 38.98 [34.91, 42.65] | ≤0 (Var Δ -0.61) |

Secondary (older view — levels not interpretable, contrasts only):

| group | corr(r1, r2) [95% CI] | cov (pts²) |
|---|---|---|
| INDOOR (all wind — no exposure expected) | +0.066 [+0.039, +0.091] | +1.34 |
| OUTDOOR calm <8 mph | +0.048 [+0.017, +0.078] | +1.00 |
| OUTDOOR moderate 8–14 | +0.069 [+0.021, +0.119] | +1.44 |
| OUTDOOR windy 14+ | +0.091 [+0.012, +0.192] | +1.94 |

## Design B — decider game 3: point share before vs after the mid-game end switch at 6

Primary: mean of swing² − binomial noise, where swing = point share pre-switch − post-switch (skill cancels; mean swing ≡ 0 by end-assignment symmetry; excess = 4·Var(end adv) in share² — LEVEL inflated by serve-streak clustering, read contrasts).

| group | deciders | RMS swing | noise RMS | mean excess ×10³ [95% CI] |
|---|---|---|---|---|
| INDOOR (all wind — no exposure expected) | 884 | 0.303 | 0.223 | +42.22 [+36.47, +48.01] |
| OUTDOOR calm <8 mph | 1299 | 0.303 | 0.223 | +42.03 [+35.65, +49.05] |
| OUTDOOR moderate 8–14 | 536 | 0.315 | 0.223 | +49.32 [+38.52, +60.95] |
| OUTDOOR windy 14+ | 111 | 0.324 | 0.223 | +55.46 [+34.85, +73.43] |

Secondary (older correlation view):

| group | corr(pre, post) [95% CI] |
|---|---|
| INDOOR (all wind — no exposure expected) | +0.335 [+0.292, +0.377] |
| OUTDOOR calm <8 mph | +0.357 [+0.305, +0.411] |
| OUTDOOR moderate 8–14 | +0.300 [+0.235, +0.372] |
| OUTDOOR windy 14+ | +0.368 [+0.227, +0.531] |

---
*Caveats: indoor/outdoor labels heuristic; Design A assumes match-level form variance is similar across groups.*
