# Wind skill as a RARE trait — spike-and-slab / tail / select-then-verify

24819 outdoor games with match-hour wind; 552 players with ≥40 games (identical panel to wind_skill.py); 60 within-player-era permutation replicates as the null.

Null calibration: var(z) across replicates = 0.97 (should be ≈1; the observed-vs-null comparisons below are valid either way since both sides get identical treatment).

## [1] Spike and slab — what fraction of players have ANY wind skill?

| arm | players | pi | slab mu (/mph) | slab sd | LR | null LR |
|---|---|---|---|---|---|---|
| full | 552 | 1.000 | -0.00028 | 0.00105 | **5.8** | 1.6±1.8 (max 7.2) |

(clutch, same test: LR 72.2 doubles / 18.2 singles vs null ≈1. pi alone is not quotable — the LR against its own null is the test.)

## [2] Tail counts — players past |z| bars vs null seasons

| tail | bar | observed | null median | null 95% | p |
|---|---|---|---|---|---|
| wind-strong (z>bar) | 2.0 | 11 | 12 | 18 | 0.717 |
| wind-strong (z>bar) | 2.5 | 3 | 3 | 7 | 0.683 |
| wind-strong (z>bar) | 3.0 | 0 | 1 | 2 | 1.000 |
| wind-fragile (z<-bar) | 2.0 | 16 | 12 | 17 | 0.117 |
| wind-fragile (z<-bar) | 2.5 | 5 | 3 | 6 | 0.267 |
| wind-fragile (z<-bar) | 3.0 | 1 | 1 | 2 | 0.600 |

## [3] Select then verify — name them on 2024-25, test on 2026

(both directions: 'strong' selects the top wind-positive tail, 'fragile' the wind-negative tail; obs z > null in EITHER row means the trait persisted across eras)

| tail K | obs z | null mean | null sd | p | n avail |
|---|---|---|---|---|---|
| strong 5 | -1.91 | -0.04 | 1.10 | 0.967 | 281 |
| strong 10 | -1.22 | -0.09 | 1.06 | 0.883 | 281 |
| strong 20 | -0.13 | -0.06 | 1.00 | 0.550 | 281 |
| strong 40 | -0.12 | -0.01 | 1.07 | 0.500 | 281 |
| fragile 5 | +0.58 | +0.01 | 1.05 | 0.383 | 281 |
| fragile 10 | +0.95 | -0.08 | 1.01 | 0.183 | 281 |
| fragile 20 | -0.01 | -0.00 | 0.92 | 0.533 | 281 |
| fragile 40 | +0.91 | +0.09 | 0.85 | 0.167 | 281 |

(clutch, same test: K=40 z=3.77 vs null 0.29±1.00; K=5 3.31 vs 0.02±0.97.)

## [4] Top posterior P(slab) — DO NOT PUBLISH unless the LR/tail/STV tests above cleared their nulls

| player | games | slope/10mph | z | P(slab) |
|---|---|---|---|---|
| Catherine Parenteau | 579 | -0.0413 | -2.49 | 1.00 |
| Jessie Irvine | 418 | -0.0510 | -2.67 | 1.00 |
| Collin Johns | 329 | -0.0607 | -2.28 | 1.00 |
| Mya Bui | 154 | -0.0900 | -2.69 | 1.00 |
| Anna Leigh Waters | 726 | -0.0200 | -1.62 | 1.00 |
| Jaeda Minniefield | 221 | -0.0765 | -2.43 | 1.00 |
| Andre Mercado | 249 | -0.0809 | -2.34 | 1.00 |
| Martina Frantova | 181 | -0.0826 | -2.35 | 1.00 |

## Power

Median per-player slope se = 0.0473 share per +10 mph. A wind specialist worth +0.02 share at +10 mph (≈ +0.4–0.5 points in a 22-point game) sits at z ≈ 0.4 for a median player — individually invisible, but the battery aggregates: the spike-slab LR and tail counts see a 10-15% minority of that size if it exists, and select-then-verify sees whether the SAME names repeat across eras.

---
*Panel identical to wind_skill.py: current-form v2 values applied retroactively, outdoor labels heuristic, match-hour wind. Null = wind permuted within player-era. Verdict changes vs wind_skill.py are attributable to the instrument, not the data.*
