# B6 — the variance channel: does wind add noise at unchanged mean?

*(model/weather_review/b6_variance.py; corrected venue labels from data/venue_overrides.csv; z = (observed point share − exact race-DP expected share)/race-DP sd, so the mechanical null is E[z²] = 1 and any excess is unmodelled dispersion — rating error plus serve clustering — which is common to all bins.)*

- outdoor: 24,718 games, mean z² = 1.863
- indoor: 7,217 games, mean z² = 1.792

## V1. Mean z² by wind bin (unpaired and within-event paired)

| setting | bin | games | mean z² [95% CI] | Δ vs calm, UNPAIRED [95% CI] | Δ vs calm, PAIRED within event [95% CI] |
|---|---|---|---|---|---|
| outdoor | 0–8 | 17,083 | 1.841 [1.793, 1.889] | (reference) | (reference) |
| outdoor | 8–14 | 6,502 | 1.924 [1.854, 2.004] | +0.083 [+0.004, +0.168] | +0.067 [-0.030, +0.173] |
| outdoor | 14–20 | 1,051 | 1.807 [1.605, 2.053] | -0.033 [-0.236, +0.226] | -0.051 [-0.305, +0.304] |
| outdoor | 20+ | 82 | 2.286 [0.881, 2.897] | +0.445 [-0.945, +1.070] | +0.445 [-0.467, +1.210] |
| indoor | 0–8 | 4,145 | 1.807 [1.710, 1.897] | (reference) | (reference) |
| indoor | 8–14 | 2,359 | 1.704 [1.599, 1.825] | -0.103 [-0.213, +0.026] | -0.138 [-0.288, +0.021] |
| indoor | 14–20 | 514 | 1.832 [1.659, 1.944] | +0.025 [-0.169, +0.173] | -0.058 [-0.296, +0.190] |
| indoor | 20+ | 199 | 2.417 [2.417, 2.417] | +0.610 [+0.519, +0.696] | +0.190 [+0.190, +0.190] |

## V2. Continuous z² slope on wind (with composition controls)

OLS of z² on wind/10 plus controls for tour, race length and |skill| decile — so the slope cannot be a composition artefact of which matches happen to be played in wind.

| setting | games | z² slope per +10 mph [95% CI] | same, no controls |
|---|---|---|---|
| outdoor | 24,718 | +0.0632 [-0.0235, +0.1656] | +0.0618 [-0.0248, +0.1597] |
| indoor | 7,217 | +0.0593 [-0.1898, +0.2158] | +0.0587 [-0.1936, +0.2011] |

## V3. Upset rate at matched predicted probability

Games are stratified into 10 bins of the race-DP predicted win probability of team 1; within each stratum the observed win rate is compared calm (<8 mph) vs windy (≥14 mph). The reported number is the stratum-size-weighted mean of (observed − predicted) windy minus the same calm — i.e. the extra upset rate in wind at equal skill gap. Positive = favourites lose more in wind.

| setting | windy threshold | extra upset rate in wind [95% CI] |
|---|---|---|
| outdoor | ≥14 mph | +1.85 pp [-0.06, +3.84] |
| outdoor | ≥12 mph | +1.89 pp [+0.31, +3.59] |
| indoor | ≥14 mph | +2.00 pp [-2.37, +4.56] |
| indoor | ≥12 mph | -0.20 pp [-5.00, +3.01] |

## V4. Reliability slope and Brier decomposition by wind bin

Logistic recalibration logit P(win) = alpha + beta·logit(p_v2) per bin. beta < 1 means predictions are too confident for that bin — the signature of extra outcome noise. (The absolute level of beta is not interpretable here: v2 values are current-form applied retroactively, which inflates beta everywhere. The comparison ACROSS bins within a setting is the test.)

| setting | bin | games | reliability slope beta [95% CI] | Brier | Brier of a p=½ forecast |
|---|---|---|---|---|---|
| outdoor | 0–8 | 17,083 | 0.673 [0.642, 0.707] | 0.1554 | 0.2500 |
| outdoor | 8–14 | 6,502 | 0.612 [0.562, 0.665] | 0.1636 | 0.2500 |
| outdoor | 14–20 | 1,051 | 0.549 [0.426, 0.715] | 0.1680 | 0.2500 |
| outdoor | 20+ | 82 | (too few) | | |
| indoor | 0–8 | 4,145 | 0.687 [0.622, 0.770] | 0.1585 | 0.2500 |
| indoor | 8–14 | 2,359 | 0.707 [0.641, 0.797] | 0.1564 | 0.2500 |
| indoor | 14–20 | 514 | 0.596 [0.515, 0.758] | 0.1723 | 0.2500 |
| indoor | 20+ | 199 | (too few) | | |

## V5. What the variance null is worth (power translation)

Inflating the outcome sd by a factor f multiplies mean z² by f². For the reference favourite (v2 expected share 0.60, an 83.6% favourite in a race to 11), an sd inflation of f raises the upset rate to Phi(−(mu−½)/(f·sd)) — the table converts the CI edges of V2 into that currency at 20 mph vs 5 mph.

Reference: mu = 0.615, sd = 0.129, upset rate 18.7% (normal approximation; the exact race-DP value is 16.4% — the CHANGE column is the object, and both ends use the same approximation).

| z² slope per 10 mph | z² at 20 mph vs 5 mph | sd inflation f | upset rate | change |
|---|---|---|---|---|
| point estimate +0.0632 | +0.0948 | 1.0251 | 19.25% | +0.59 pp |
| lower CI edge -0.0235 | -0.0352 | 0.9905 | 18.43% | -0.23 pp |
| upper CI edge +0.1656 | +0.2485 | 1.0646 | 20.14% | +1.48 pp |
| upper CI edge, de-attenuated (lambda_T = 0.941) +0.1760 | +0.2640 | 1.0685 | 20.23% | +1.57 pp |

Mean outdoor z² = 1.863 is the denominator: the excess over 1.0 is rating error + serve clustering, present in every bin.

