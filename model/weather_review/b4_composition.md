# B4 addendum — event composition vs the H4 interaction

Corrected outdoor pool: 69 events / 24718 games — 36 web-verified (11435 games), 33 unaudited-heuristic (13283 games).

| arm | events | games | d (skill×wind, share outcome) ± CR1 se |
|---|---|---|---|
| web-verified outdoor | 36 | 11435 | +0.0592 ± 0.0270 |
| unaudited heuristic outdoor | 33 | 13283 | -0.1105 ± 0.0362 |
| both (corrected pool) | 69 | 24718 | -0.0376 ± 0.0285 |

Observed arm gap (verified − unaudited) = **+0.1697** (B2a reported ≈ +0.225 on its own variant).

## 1. How much do individual EVENTS disagree about d?

*(within-event fits use CLASSICAL OLS standard errors — the cluster-robust sandwich is degenerate with one cluster.)*

41 events with ≥60 games and an estimable within-event d. Fixed-effect mean -0.0403. Cochran Q = 54.8 on 40 df (expected 40 under no true heterogeneity) → **τ = 0.1346** (between-event sd of the true interaction slope).

Median within-event standard error 0.262 — a single event pins d only to about ±0.51. Between-event heterogeneity τ = 0.135 is the number that matters: it is the sd of the per-event slope, and it is 2× the effect size under debate (the binned −2.0 pp windy drift maps to d ≈ −0.072).

## 2. Is a gap of that size remarkable? (event-label permutation)

5,000 random re-labellings of which events are 'verified' (36 of 69), same estimator:

| null gap sd | 2.5% | 50% | 97.5% | P(|random gap| ≥ observed 0.170) | P(|random gap| ≥ 0.225) |
|---|---|---|---|---|---|
| 0.0586 | -0.1155 | -0.0013 | +0.1141 | 0.002 | 0.000 |

Size-stratified permutation (events split into 3 size terciles, verified-count held fixed inside each — the audit was not random, it targeted the bigger label-supplying events): null sd 0.0596, 95% band [-0.1162, +0.1170], P(|random gap| ≥ 0.170) = 0.002.

### Same split, by tour and by outcome (is it an audit effect or an event-type effect?)

| subset | verified d | unaudited d | gap |
|---|---|---|---|
| all events (36v/33u events) | +0.0592 ± 0.0270 | -0.1105 ± 0.0362 | +0.1697 |
| PPA events only (23v/33u events) | +0.0759 ± 0.0245 | -0.1105 ± 0.0362 | +0.1865 |
| events ≥300 games (19v/28u events) | +0.0974 ± 0.0207 | -0.1032 ± 0.0361 | +0.2006 |
| events <300 games (17v/5u events) | -0.1244 ± 0.1357 | +0.4608 ± 0.1385 | -0.5851 |

## 3. Does the arm gap survive EVENT fixed effects?

| arm | d, no FE | d, event FE (within-event wind variation only) |
|---|---|---|
| web-verified | +0.0592 ± 0.0270 | +0.0638 ± 0.0281 |
| unaudited | -0.1105 ± 0.0362 | -0.1157 ± 0.0362 |
| pool | -0.0376 ± 0.0285 | -0.0396 ± 0.0296 |

Gap under event FE = **+0.1795** vs +0.1697 without. Event FE removes all BETWEEN-event variation, so a gap that survives is within-event heterogeneity, not composition.

## 4. Which events move the arm gap most (leave-one-out)?

| event | games | arm | Δ gap when dropped |
|---|---|---|---|
| PPA Tour: Fasenra Virginia Beach Cup presented by Joola (2025-10-06) | 383 | unaudited | +0.0188 |
| PPA Tour: North Carolina Cup (2024-04-01) | 356 | unaudited | -0.0167 |
| PPA Tour: Selkirk Kansas City Open (2024-08-07) | 534 | verified | -0.0129 |
| PPA Tour: CIBC Texas Open - 2025 (2025-03-12) | 506 | unaudited | -0.0116 |
| Charleston Metro PPA Challenger (2025-10-04) | 339 | unaudited | +0.0110 |
