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

## Design A2 — game-1 vs game-2 POINT SHARES (share/z² framework, no ratings needed)

Same object as Design B but across the between-game end switch: swing = team 1's share of points in game 1 minus game 2 (scores ARE the point counts). 'null z²' = the same statistic in games simulated from the winprob.py serve-state model (k = 0.43, match etas, NO momentum, NO end effects) — the mechanical serve-clustering floor.

| group | matches | mean z² [95% CI] | null z² (sim) | mean excess ×10³ [95% CI] |
|---|---|---|---|---|
| INDOOR (all wind — no exposure expected) | 4152 | 1.83 [1.76, 1.91] | 1.88 | +27.31 [+24.92, +29.55] |
| OUTDOOR calm <8 mph | 6653 | 1.87 [1.82, 1.93] | 1.88 | +28.88 [+27.18, +30.68] |
| OUTDOOR moderate 8–14 | 2638 | 1.90 [1.81, 2.00] | 1.86 | +29.09 [+26.14, +32.25] |
| OUTDOOR windy 14+ | 483 | 1.81 [1.64, 1.97] | 1.84 | +26.89 [+21.73, +31.64] |

## Design B — point share before vs after the mid-game end switch at 6 (PPA deciders + ALL MLP games)

Primary: the swing = TEAM A's point share on its first end minus its share on its second end (the 6-0-then-5-7 comparison; team B is the mirror image, so side A alone carries all the information). Its mean is 0 by end-assignment symmetry, so the tests are (i) mean of swing² − binomial noise (= 4·Var(end adv) in share²) and (ii) mean z², each game standardized by its own sampling noise — 1.00 under the null, so games with decisive halves count for more. LEVELS are inflated by serve-streak clustering; read contrasts.

| group | games | RMS swing | noise RMS | mean excess ×10³ [95% CI] | mean z² [95% CI] | null z² (sim) |
|---|---|---|---|---|---|---|
| INDOOR (all wind — no exposure expected) | 2383 | 0.297 | 0.220 | +39.81 [+36.28, +43.50] | 1.67 [1.61, 1.73] | 1.76 |
| OUTDOOR calm <8 mph | 1636 | 0.302 | 0.221 | +42.31 [+36.82, +48.16] | 1.72 [1.63, 1.83] | 1.78 |
| OUTDOOR moderate 8–14 | 656 | 0.306 | 0.222 | +44.54 [+34.04, +56.92] | 1.76 [1.57, 1.98] | 1.79 |
| OUTDOOR windy 14+ | 111 | 0.324 | 0.223 | +55.46 [+34.85, +73.43] | 1.95 [1.58, 2.28] | 1.79 |
### The weather test — Δ mean z² vs OUTDOOR calm (reference)

| contrast | Δ mean z² [95% CI] |
|---|---|
| INDOOR (all wind — no exposure expected) − calm | -0.056 [-0.177, +0.064] |
| OUTDOOR moderate 8–14 − calm | +0.034 [-0.184, +0.285] |
| OUTDOOR windy 14+ − calm | +0.224 [-0.160, +0.585] |

Continuous: slope of z² on match-hour wind, outdoor only (2403 games): +0.165 per +10 mph [-0.053, +0.364]


Secondary (older correlation view):

| group | corr(pre, post) [95% CI] |
|---|---|
| INDOOR (all wind — no exposure expected) | +0.438 [+0.403, +0.468] |
| OUTDOOR calm <8 mph | +0.381 [+0.333, +0.426] |
| OUTDOOR moderate 8–14 | +0.342 [+0.272, +0.407] |
| OUTDOOR windy 14+ | +0.368 [+0.227, +0.531] |

## Design C — rally level: SERVE-RALLY WIN RATE before vs after the switch

Each serve rally is close to an independent Bernoulli given the serve state, so the mechanical side-out clustering that inflates point-share swings mostly vanishes — the null sits near 1.00 on its own, and each team's serve rate is separate information (two observations per game, unlike point shares where B mirrors A). The statistic is channel-agnostic: an end that hurts the SERVING team shows up in its own serve-rate swing, one that hurts the RECEIVING team (sun on the return, resets into wind) shows up in the opponent's — both sides are observed, so either lands here. Same games as Design B; data/decider_serve_splits.csv (pb_rally, rallies + wins per side per half).

| group | team-halves | mean z² [95% CI] | null z² (sim) |
|---|---|---|---|
| INDOOR (all wind — no exposure expected) | 3857 | 1.10 [1.06, 1.15] | 1.15 |
| OUTDOOR calm <8 mph | 2659 | 1.15 [1.09, 1.22] | 1.16 |
| OUTDOOR moderate 8–14 | 1065 | 1.09 [0.99, 1.20] | 1.16 |
| OUTDOOR windy 14+ | 162 | 1.34 [1.10, 1.66] | 1.20 |

### The weather test at rally level — Δ mean z² vs OUTDOOR calm

| contrast | Δ mean z² [95% CI] |
|---|---|
| INDOOR (all wind — no exposure expected) − calm | -0.052 [-0.130, +0.023] |
| OUTDOOR moderate 8–14 − calm | -0.067 [-0.188, +0.062] |
| OUTDOOR windy 14+ − calm | +0.190 [-0.067, +0.544] |

Continuous: slope of serve-rate z² on match-hour wind, outdoor only (3886 team-halves): +0.038 per +10 mph [-0.093, +0.166]

---
*Caveats: indoor/outdoor labels heuristic; Design A assumes match-level form variance is similar across groups.*
