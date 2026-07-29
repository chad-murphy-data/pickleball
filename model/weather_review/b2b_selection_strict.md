# B2b parts 3-4 — decider selection + strict correction handling

## Part 4a — integrity of data/decider_splits.csv vs a fresh pb_rally rebuild

- committed rows: 5061; reproduced exactly (or mirrored): 5061 (100.0%); differ: 0; not in rebuild: 0
- of the committed rows, the fresh rebuild flags 1013 (20.0%) with a score-sequence correction/rewind, 333 (6.6%) with the switch boundary NOT at exactly 6, and 983 (19.4%) whose derived final score disagrees with games.csv

## Part 3 — decider conditioning: switch games vs the no-switch placebo

Statistic identical in both arms (point share before vs after the score first reaches 6). In SWITCH games the ends actually change there; in NO-SWITCH games (PPA games 1-2 of a best-of-3) nothing changes, so any wind-driven excess there cannot be an end effect.


### published labels

| arm | n games | mean z2 | by group: INDOOR / calm <8 / moderate 8-14 / windy 14+ |
|---|---|---|---|
| SWITCH (MLP all + PPA deciders) | 4786 | 1.707 | 1.669 (n=2383) / 1.725 (n=1636) / 1.759 (n=656) / 1.949 (n=111) |
| NO-SWITCH placebo (PPA non-deciders) | 20311 | 1.642 | 1.657 (n=6415) / 1.637 (n=9253) / 1.639 (n=3902) / 1.579 (n=741) |

PPA only, decider minus non-decider mean z2: +0.120 (2830 vs 20311 games) — this is the SELECTION (collider) magnitude, not an end effect.

**Wind contrasts inside each arm** (same estimators as part 1):

| contrast (SWITCH) | n_t | unpaired | paired-FE [95% CI] | paired-ATT [95% CI] | events | perm p (1s) |
|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | 656 | +0.034 | -0.142 [-0.340, +0.082] | -0.169 [-0.621, +0.210] | 48 | 0.887 |
| OUTDOOR windy 14+ - calm | 111 | +0.224 | +0.375 [-0.087, +0.739] | +0.401 [+0.035, +0.683] | 15 | 0.084 |

| contrast (NO-SWITCH placebo) | n_t | unpaired | paired-FE [95% CI] | paired-ATT [95% CI] | events | perm p (1s) |
|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | 3902 | +0.002 | -0.026 [-0.126, +0.079] | -0.070 [-0.181, +0.047] | 49 | 0.698 |
| OUTDOOR windy 14+ - calm | 741 | -0.058 | -0.046 [-0.251, +0.138] | -0.102 [-0.281, +0.074] | 19 | 0.656 |

**Difference-in-differences** (windy-calm in switch games minus windy-calm in no-switch games), event-paired both sides:
- DiD OUTDOOR moderate 8-14 vs calm: -0.116 [-0.271, +0.069] (53 events)
- DiD OUTDOOR windy 14+ vs calm: +0.421 [-0.022, +0.841] (19 events)

### corrected labels

| arm | n games | mean z2 | by group: INDOOR / calm <8 / moderate 8-14 / windy 14+ |
|---|---|---|---|
| SWITCH (MLP all + PPA deciders) | 4319 | 1.700 | 1.674 (n=1277) / 1.719 (n=2063) / 1.655 (n=866) / 2.006 (n=113) |
| NO-SWITCH placebo (PPA non-deciders) | 17606 | 1.648 | 1.659 (n=3337) / 1.645 (n=9711) / 1.656 (n=3829) / 1.603 (n=729) |

PPA only, decider minus non-decider mean z2: +0.110 (2473 vs 17606 games) — this is the SELECTION (collider) magnitude, not an end effect.

**Wind contrasts inside each arm** (same estimators as part 1):

| contrast (SWITCH) | n_t | unpaired | paired-FE [95% CI] | paired-ATT [95% CI] | events | perm p (1s) |
|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | 866 | -0.063 | -0.197 [-0.370, +0.000] | -0.182 [-0.566, +0.119] | 52 | 0.976 |
| OUTDOOR windy 14+ - calm | 113 | +0.287 | +0.346 [-0.113, +0.725] | +0.419 [+0.023, +0.697] | 14 | 0.097 |

| contrast (NO-SWITCH placebo) | n_t | unpaired | paired-FE [95% CI] | paired-ATT [95% CI] | events | perm p (1s) |
|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | 3829 | +0.011 | -0.019 [-0.107, +0.073] | -0.054 [-0.168, +0.062] | 45 | 0.657 |
| OUTDOOR windy 14+ - calm | 729 | -0.042 | -0.041 [-0.250, +0.129] | -0.088 [-0.283, +0.092] | 17 | 0.635 |

**Difference-in-differences** (windy-calm in switch games minus windy-calm in no-switch games), event-paired both sides:
- DiD OUTDOOR moderate 8-14 vs calm: -0.178 [-0.312, -0.012] (57 events)
- DiD OUTDOOR windy 14+ vs calm: +0.386 [-0.018, +0.804] (18 events)

## Part 4b — the paired windy contrast under STRICT correction handling

STRICT = fresh pb_rally rebuild AND no score-sequence correction/rewind AND switch boundary exactly at 6 AND derived final score equals games.csv.


### published labels


**rebuilt, all rows** (Design B n=4786, Design C n=7743)

| contrast (Design B) | n_t | unpaired | paired-FE [95% CI] | paired-ATT [95% CI] | events | perm p (1s) |
|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | 656 | +0.034 | -0.142 [-0.340, +0.082] | -0.169 [-0.621, +0.210] | 48 | 0.887 |
| OUTDOOR windy 14+ - calm | 111 | +0.224 | +0.375 [-0.087, +0.739] | +0.401 [+0.035, +0.683] | 15 | 0.084 |

| contrast (Design C) | n_t | unpaired | paired-FE [95% CI] | paired-ATT [95% CI] | events | perm p (1s) |
|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | 1065 | -0.067 | -0.154 [-0.292, -0.021] | -0.137 [-0.324, +0.044] | 48 | 0.990 |
| OUTDOOR windy 14+ - calm | 162 | +0.190 | +0.220 [-0.133, +0.561] | +0.251 [-0.027, +0.566] | 15 | 0.106 |

**STRICT** (Design B n=3846, Design C n=6135)

| contrast (Design B) | n_t | unpaired | paired-FE [95% CI] | paired-ATT [95% CI] | events | perm p (1s) |
|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | 538 | +0.081 | -0.062 [-0.308, +0.231] | -0.075 [-0.589, +0.348] | 46 | 0.693 |
| OUTDOOR windy 14+ - calm | 94 | +0.204 | +0.336 [-0.093, +0.715] | +0.291 [-0.054, +0.587] | 15 | 0.125 |

| contrast (Design C) | n_t | unpaired | paired-FE [95% CI] | paired-ATT [95% CI] | events | perm p (1s) |
|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | 862 | -0.043 | -0.094 [-0.239, +0.053] | -0.052 [-0.251, +0.149] | 46 | 0.900 |
| OUTDOOR windy 14+ - calm | 137 | +0.184 | +0.275 [-0.043, +0.625] | +0.233 [+0.016, +0.509] | 15 | 0.077 |

### corrected labels


**rebuilt, all rows** (Design B n=4319, Design C n=6982)

| contrast (Design B) | n_t | unpaired | paired-FE [95% CI] | paired-ATT [95% CI] | events | perm p (1s) |
|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | 866 | -0.063 | -0.197 [-0.370, +0.000] | -0.182 [-0.566, +0.119] | 52 | 0.976 |
| OUTDOOR windy 14+ - calm | 113 | +0.287 | +0.346 [-0.113, +0.725] | +0.419 [+0.023, +0.697] | 14 | 0.097 |

| contrast (Design C) | n_t | unpaired | paired-FE [95% CI] | paired-ATT [95% CI] | events | perm p (1s) |
|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | 1392 | -0.089 | -0.143 [-0.244, -0.031] | -0.097 [-0.252, +0.050] | 52 | 0.995 |
| OUTDOOR windy 14+ - calm | 173 | +0.108 | +0.098 [-0.247, +0.425] | +0.134 [-0.169, +0.434] | 14 | 0.266 |

**STRICT** (Design B n=3460, Design C n=5512)

| contrast (Design B) | n_t | unpaired | paired-FE [95% CI] | paired-ATT [95% CI] | events | perm p (1s) |
|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | 697 | -0.024 | -0.124 [-0.315, +0.109] | -0.099 [-0.482, +0.233] | 51 | 0.866 |
| OUTDOOR windy 14+ - calm | 95 | +0.287 | +0.271 [-0.126, +0.649] | +0.282 [-0.043, +0.587] | 14 | 0.188 |

| contrast (Design C) | n_t | unpaired | paired-FE [95% CI] | paired-ATT [95% CI] | events | perm p (1s) |
|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | 1101 | -0.065 | -0.089 [-0.197, +0.036] | -0.024 [-0.178, +0.132] | 51 | 0.938 |
| OUTDOOR windy 14+ - calm | 143 | +0.094 | +0.084 [-0.287, +0.471] | +0.067 [-0.235, +0.398] | 14 | 0.326 |
