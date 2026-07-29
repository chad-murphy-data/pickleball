# TASK B1 part 2 — attacking the positive heat interaction

Mean skill (team-1 minus team-2 expected share) outdoors: +0.0760 — NOT centred, so the intercept and heat main effect can entangle with the interaction (test A).

## A. Antisymmetric (symmetrized) specification

Doubling with flipped labels forces the fit to be odd in skill: y = b*skill + d*skill*h (intercept and heat main effect are exactly 0 by construction).

| setting | spec | b (skill) | d (skill x heat) [95% CI] |
|---|---|---|---|
| outdoor | with intercept + heat main (published form) | 1.033 | +0.031 [+0.005, +0.058] |
| outdoor | symmetrized, odd only | 1.087 | +0.017 [-0.005, +0.041] |
| indoor | with intercept + heat main (published form) | 1.081 | -0.015 [-0.044, +0.008] |
| indoor | symmetrized, odd only | 1.110 | -0.013 [-0.036, +0.001] |

## B. Functional form — is the interaction just concavity x composition?

share ~ skill is concave near the bounds. If cool hours hold the big mismatches, a linear model reads that as skill x heat. skill^3 (odd, so antisymmetry survives) absorbs the curvature.

| setting | spec | games | d (skill x heat) [95% CI] |
|---|---|---|---|
| outdoor | odd linear | 24718 | +0.017 [-0.005, +0.041] |
| outdoor | odd + skill^3 | 24718 | +0.016 [-0.005, +0.040] |
| outdoor | odd + skill^3 + skill^3 x heat | 24718 | +0.020 [-0.012, +0.050] |
| indoor | odd linear | 7127 | -0.013 [-0.036, +0.001] |
| indoor | odd + skill^3 | 7127 | -0.013 [-0.035, +0.001] |
| indoor | odd + skill^3 + skill^3 x heat | 7127 | -0.015 [-0.039, +0.014] |

### B2. Inside |skill| terciles (each band is nearly linear)

| setting | band | games | mean|skill| | b | d (skill x heat) [95% CI] |
|---|---|---|---|---|---|
| outdoor | low |skill| (<0.065) | 8238 | 0.032 | 1.247 | +0.072 [-0.029, +0.170] |
| outdoor | mid | 8238 | 0.103 | 1.175 | +0.034 [-0.016, +0.085] |
| outdoor | high |skill| (>=0.147) | 8242 | 0.223 | 1.064 | +0.010 [-0.009, +0.035] |
| indoor | low |skill| (<0.066) | 2374 | 0.032 | 1.168 | -0.097 [-0.229, -0.008] |
| indoor | mid | 2376 | 0.103 | 1.127 | +0.005 [-0.033, +0.029] |
| indoor | high |skill| (>=0.144) | 2377 | 0.220 | 1.104 | -0.015 [-0.040, +0.001] |

## C. Non-parametric dose-response: skill slope b inside temperature bins

A real effect should climb monotonically; a composition artefact need not.

| setting | temp bin | games | b (skill slope) [95% CI] |
|---|---|---|---|
| outdoor | <60F | 2393 | 1.025 [0.964, 1.090] |
| outdoor | 60-70F | 6840 | 1.100 [1.073, 1.126] |
| outdoor | 70-80F | 11068 | 1.078 [1.053, 1.104] |
| outdoor | 80-90F | 3753 | 1.096 [1.049, 1.149] |
| outdoor | 90F+ | 664 | 1.116 [1.034, 1.212] |
| indoor | <60F | 2156 | 1.168 [1.109, 1.231] |
| indoor | 60-70F | 684 | 1.120 [1.019, 1.187] |
| indoor | 70-80F | 1356 | 1.145 [1.062, 1.221] |
| indoor | 80-90F | 1731 | 1.058 [0.963, 1.159] |
| indoor | 90F+ | 1200 | 1.081 [0.989, 1.174] |

## D. Round/stage composition: within (event x stage) fixed effects

| setting | games | d (skill x heat) [95% CI] |
|---|---|---|
| outdoor | 24718 | +0.017 [-0.005, +0.041] |
| indoor | 7127 | -0.013 [-0.036, +0.001] |

## E. Venue labels

Same events, audited vs heuristic label (isolates relabelling from sample change); then confidence and the mixed/unknown arms.

| arm | games | events | d (skill x heat) [95% CI] |
|---|---|---|---|
| audited events, AUDITED label = outdoor | 11435 | 36 | +0.022 [+0.001, +0.046] |
| audited events, HEURISTIC label = outdoor | 11536 | 34 | -0.011 [-0.032, +0.020] |
| audited outdoor, HIGH confidence only | 6138 | 20 | +0.051 [+0.015, +0.091] |
| unaudited events (heuristic outdoor) | 13283 | 33 | +0.011 [-0.027, +0.052] |
| mixed-venue events | 3231 | 7 | +0.001 [-0.055, +0.048] |
| unknown-venue events | 1442 | 4 | -0.140 [-0.199, -0.022] |

## F. Splits (outdoor, audited labels, symmetrized odd spec)

| split | games | events | d (skill x heat) [95% CI] |
|---|---|---|---|
| PPA only | 23546 | 56 | +0.015 [-0.009, +0.040] |
| MLP only | 1172 | 13 | +0.080 [-0.001, +0.158] |
| ACTUAL start times only | 18325 | 68 | +0.021 [-0.004, +0.048] |
| PLANNED start times only | 6393 | 53 | +0.028 [+0.005, +0.051] |
| season 2024 | 7660 | 17 | +0.040 [+0.011, +0.085] |
| season 2025 | 9426 | 32 | +0.043 [-0.004, +0.079] |
| season 2026 | 7632 | 20 | -0.014 [-0.043, +0.024] |

## G. Does the FAVOURITE actually WIN more in heat?

Logistic P(favourite wins the game) = a + b*|skill| + c*h + d*|skill|*h. This is the quantity a reader cares about; d > 0 = fewer upsets in heat. Cluster bootstrap over events, 300 draws.

| setting | games | favourite win rate | d (|skill| x heat) [95% CI] | upset shift, 75F -> 95F, median favourite |
|---|---|---|---|---|
| outdoor | 24718 | 0.773 | +0.156 [-0.519, +0.737] | 0.788 -> 0.807 (+1.92 pp) |
| indoor | 7127 | 0.768 | -0.080 [-0.499, +0.331] | 0.781 -> 0.769 (-1.21 pp) |

---
*Deterministic; cluster bootstrap over events (1200 draws, seeded). model/weather_review/heat_robust.py*
