# B6 — extra upset rate in wind, WITHIN-EVENT paired

y = (win − predicted win prob) signed so positive = the underdog beat its prediction. Event and predicted-probability-decile effects removed; the coefficient is on 1[wind ≥ W]. 587 usable resamples of the 99 events (cluster bootstrap, seed 1234).

| setting | threshold | games ≥ W | paired extra upset rate [95% CI] |
|---|---|---|---|
| outdoor | ≥14 mph | 1,133 | +1.16 pp [-0.95, +3.36] |
| indoor | ≥14 mph | 713 | +0.77 pp [-2.31, +5.99] |
| **outdoor − indoor** | ≥14 mph | 1,846 | +0.39 pp [-5.29, +3.94] |
| outdoor | ≥12 mph | 2,508 | +1.77 pp [-0.23, +3.92] |
| indoor | ≥12 mph | 1,228 | -1.71 pp [-5.70, +2.60] |
| **outdoor − indoor** | ≥12 mph | 3,736 | +3.47 pp [-1.21, +8.08] |

## Reliability-slope contrast (windy 14+ minus calm <8)

Logistic recalibration slope beta of the outcome on logit(p_v2). beta < 1 = over-confident. A NEGATIVE contrast means predictions get relatively MORE over-confident in wind — the variance channel's signature. Absolute levels are not interpretable (retroactive current-form ratings inflate beta everywhere); the contrast is.

| arm | Δbeta (windy − calm) [95% CI] |
|---|---|
| outdoor | -0.120 [-0.245, +0.014] |
| indoor | -0.150 [-0.241, +0.099] |
| **outdoor − indoor** | +0.031 [-0.228, +0.195] |

