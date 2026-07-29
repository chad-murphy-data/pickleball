# B2b — paired + corrected-label re-run of the end-effect designs

Pre-specified signal: a real bad-end effect makes the PAIRED pre/post swing VARIANCE exceed its sampling/serve-clustering null, and that excess GROWS with wind. Concretely: (i) mean z2 in the outdoor windy bin above its simulated null, and (ii) the windy-minus-calm contrast positive with a CI excluding 0, in the PRE-SPECIFIED within-event paired form, surviving the corrected venue labels. A contrast that flips sign or loses its CI when the labels are fixed is not a signal.

## Pairing estimator (stated before results)

Treatment arm = games in wind bin g at event e; control arm = games in the OUTDOOR calm bin at the SAME event e. Only events contributing games to both arms enter. Per event

    d_e = mean z2(treated games at e) - mean z2(calm games at e)

and the reported contrast is a weighted mean of d_e. Three weightings are reported because the weighting is a real choice:

* **FE** w_e = n_t n_c/(n_t+n_c) - the inverse-variance weight under homoskedastic per-game z2; algebraically the OLS coefficient on the wind dummy in a regression with event fixed effects. **Pre-specified primary**: it is the efficient estimator, and it is fixed by the design, not by the data.

* **ATT** w_e = n_t - 'effect for the average windy game observed' (the weighting used in the phase-1 audit).

* **unit** w_e = 1 - 'average event-level effect'.

Why the weighting cannot manufacture the result: all three weight vectors are functions of arm SIZES only, fixed before any z2 is looked at, and none can change the sign of a common effect - they only trade efficiency against which events dominate. The honest check is that they are reported side by side; a result that lives in only one weighting is a composition artifact, not an effect. Inference: (a) cluster bootstrap resampling EVENTS from the paired set, 2000 draws; (b) a t interval on the G event-level d_e with G-1 df (small-G honest); (c) a within-event permutation test that reshuffles the wind label across each event's games holding n_t, n_c fixed - exact under the sharp null.

Residual confounding the pairing does NOT remove: within an event, windy games are a non-random subset of DAYS and HOURS (and court assignments). The paired contrast is 'windy vs calm at the same tournament', not 'same match under two winds'.


---

## Label arm: `published`  (Design B n=4786 games, Design C n=7743 team-halves)

Design B group sizes: INDOOR=2383, OUTDOOR calm <8=1636, OUTDOOR moderate 8-14=656, OUTDOOR windy 14+=111


**Design B (point share) — per-group level and excess over the simulated null**

| group | n | mean z2 [95% CI] | sim null z2 | excess [95% CI] |
|---|---|---|---|---|
| INDOOR | 2383 | 1.669 [1.606, 1.732] | 1.765 | -0.096 [-0.163, -0.032] |
| OUTDOOR calm <8 | 1636 | 1.725 [1.627, 1.830] | 1.803 | -0.078 [-0.175, +0.026] |
| OUTDOOR moderate 8-14 | 656 | 1.759 [1.560, 1.988] | 1.828 | -0.069 [-0.271, +0.151] |
| OUTDOOR windy 14+ | 111 | 1.949 [1.590, 2.272] | 1.832 | +0.117 [-0.178, +0.398] |

**Design B (point share) — contrast vs OUTDOOR calm <8: UNPAIRED (published design) vs PAIRED (within-event)**

| contrast | unpaired Δz2 [95% CI] | paired-FE Δ [boot 95%] | paired-FE [t, G-1 df] | paired-ATT | paired-unit | events | perm p (1-sided / 2-sided) |
|---|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | +0.034 [-0.188, +0.287] | -0.142 [-0.349, +0.087] | [-0.358, +0.074] | -0.169 [-0.628, +0.197] | -0.099 [-0.623, +0.430] | 48 | 0.881 / 0.230 |
| OUTDOOR windy 14+ - calm | +0.224 [-0.186, +0.578] | +0.375 [-0.087, +0.739] | [-0.078, +0.827] | +0.401 [+0.035, +0.683] | +0.343 [-0.237, +0.885] | 15 | 0.076 / 0.151 |

Design B: WITHIN-EVENT slope of z2 on match-hour wind (outdoor, 2403 obs / 63 events): -0.009 per +10 mph [-0.285, +0.249]

**Design C (serve-rally rate) — per-group level and excess over the simulated null**

| group | n | mean z2 [95% CI] | sim null z2 | excess [95% CI] |
|---|---|---|---|---|
| INDOOR | 3857 | 1.103 [1.059, 1.147] | 1.156 | -0.053 [-0.099, -0.008] |
| OUTDOOR calm <8 | 2659 | 1.155 [1.093, 1.220] | 1.155 | -0.000 [-0.062, +0.063] |
| OUTDOOR moderate 8-14 | 1065 | 1.087 [0.989, 1.208] | 1.159 | -0.072 [-0.175, +0.054] |
| OUTDOOR windy 14+ | 162 | 1.344 [1.101, 1.660] | 1.128 | +0.217 [-0.026, +0.514] |

**Design C (serve-rally rate) — contrast vs OUTDOOR calm <8: UNPAIRED (published design) vs PAIRED (within-event)**

| contrast | unpaired Δz2 [95% CI] | paired-FE Δ [boot 95%] | paired-FE [t, G-1 df] | paired-ATT | paired-unit | events | perm p (1-sided / 2-sided) |
|---|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | -0.067 [-0.191, +0.066] | -0.154 [-0.292, -0.021] | [-0.287, -0.022] | -0.137 [-0.324, +0.044] | -0.097 [-0.301, +0.114] | 48 | 0.990 / 0.022 |
| OUTDOOR windy 14+ - calm | +0.190 [-0.077, +0.527] | +0.220 [-0.133, +0.561] | [-0.162, +0.602] | +0.251 [-0.027, +0.566] | +0.437 [-0.224, +1.246] | 15 | 0.109 / 0.210 |

Design C: WITHIN-EVENT slope of z2 on match-hour wind (outdoor, 3886 obs / 63 events): -0.042 per +10 mph [-0.228, +0.149]

---

## Label arm: `corrected_all`  (Design B n=4319 games, Design C n=6982 team-halves)

Design B group sizes: INDOOR=1277, OUTDOOR calm <8=2063, OUTDOOR moderate 8-14=866, OUTDOOR windy 14+=113


**Design B (point share) — per-group level and excess over the simulated null**

| group | n | mean z2 [95% CI] | sim null z2 | excess [95% CI] |
|---|---|---|---|---|
| INDOOR | 1277 | 1.674 [1.584, 1.774] | 1.758 | -0.084 [-0.182, +0.009] |
| OUTDOOR calm <8 | 2063 | 1.719 [1.637, 1.807] | 1.783 | -0.064 [-0.154, +0.027] |
| OUTDOOR moderate 8-14 | 866 | 1.655 [1.511, 1.833] | 1.775 | -0.119 [-0.266, +0.065] |
| OUTDOOR windy 14+ | 113 | 2.006 [1.629, 2.322] | 1.751 | +0.254 [-0.083, +0.545] |

**Design B (point share) — contrast vs OUTDOOR calm <8: UNPAIRED (published design) vs PAIRED (within-event)**

| contrast | unpaired Δz2 [95% CI] | paired-FE Δ [boot 95%] | paired-FE [t, G-1 df] | paired-ATT | paired-unit | events | perm p (1-sided / 2-sided) |
|---|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | -0.063 [-0.243, +0.136] | -0.197 [-0.363, -0.002] | [-0.381, -0.012] | -0.182 [-0.537, +0.133] | -0.137 [-0.612, +0.342] | 52 | 0.980 / 0.043 |
| OUTDOOR windy 14+ - calm | +0.287 [-0.081, +0.644] | +0.346 [-0.108, +0.728] | [-0.106, +0.797] | +0.419 [+0.024, +0.706] | +0.372 [-0.235, +0.939] | 14 | 0.098 / 0.185 |

Design B: WITHIN-EVENT slope of z2 on match-hour wind (outdoor, 3042 obs / 68 events): -0.111 per +10 mph [-0.404, +0.154]

**Design C (serve-rally rate) — per-group level and excess over the simulated null**

| group | n | mean z2 [95% CI] | sim null z2 | excess [95% CI] |
|---|---|---|---|---|
| INDOOR | 2076 | 1.130 [1.070, 1.191] | 1.151 | -0.020 [-0.085, +0.045] |
| OUTDOOR calm <8 | 3341 | 1.144 [1.086, 1.200] | 1.163 | -0.019 [-0.076, +0.036] |
| OUTDOOR moderate 8-14 | 1392 | 1.054 [0.978, 1.149] | 1.148 | -0.094 [-0.170, +0.000] |
| OUTDOOR windy 14+ | 173 | 1.251 [1.019, 1.505] | 1.190 | +0.061 [-0.184, +0.325] |

**Design C (serve-rally rate) — contrast vs OUTDOOR calm <8: UNPAIRED (published design) vs PAIRED (within-event)**

| contrast | unpaired Δz2 [95% CI] | paired-FE Δ [boot 95%] | paired-FE [t, G-1 df] | paired-ATT | paired-unit | events | perm p (1-sided / 2-sided) |
|---|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | -0.089 [-0.183, +0.017] | -0.143 [-0.244, -0.031] | [-0.250, -0.035] | -0.097 [-0.252, +0.050] | -0.058 [-0.231, +0.116] | 52 | 0.995 / 0.014 |
| OUTDOOR windy 14+ - calm | +0.108 [-0.114, +0.386] | +0.098 [-0.247, +0.425] | [-0.260, +0.456] | +0.134 [-0.169, +0.434] | +0.340 [-0.406, +1.248] | 14 | 0.264 / 0.551 |

Design C: WITHIN-EVENT slope of z2 on match-hour wind (outdoor, 4906 obs / 68 events): -0.067 per +10 mph [-0.217, +0.081]

---

## Label arm: `corrected_hi`  (Design B n=3400 games, Design C n=5498 team-halves)

Design B group sizes: INDOOR=1170, OUTDOOR calm <8=1482, OUTDOOR moderate 8-14=647, OUTDOOR windy 14+=101


**Design B (point share) — per-group level and excess over the simulated null**

| group | n | mean z2 [95% CI] | sim null z2 | excess [95% CI] |
|---|---|---|---|---|
| INDOOR | 1170 | 1.660 [1.569, 1.752] | 1.748 | -0.088 [-0.199, +0.022] |
| OUTDOOR calm <8 | 1482 | 1.692 [1.602, 1.783] | 1.789 | -0.097 [-0.194, -0.005] |
| OUTDOOR moderate 8-14 | 647 | 1.676 [1.487, 1.902] | 1.782 | -0.106 [-0.287, +0.111] |
| OUTDOOR windy 14+ | 101 | 2.082 [1.756, 2.405] | 1.817 | +0.265 [-0.078, +0.610] |

**Design B (point share) — contrast vs OUTDOOR calm <8: UNPAIRED (published design) vs PAIRED (within-event)**

| contrast | unpaired Δz2 [95% CI] | paired-FE Δ [boot 95%] | paired-FE [t, G-1 df] | paired-ATT | paired-unit | events | perm p (1-sided / 2-sided) |
|---|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | -0.016 [-0.221, +0.235] | -0.176 [-0.340, +0.042] | [-0.369, +0.017] | -0.168 [-0.627, +0.187] | -0.088 [-0.682, +0.535] | 39 | 0.941 / 0.118 |
| OUTDOOR windy 14+ - calm | +0.390 [+0.001, +0.728] | +0.398 [-0.072, +0.807] | [-0.106, +0.902] | +0.473 [+0.098, +0.768] | +0.352 [-0.361, +0.986] | 12 | 0.098 / 0.200 |

Design B: WITHIN-EVENT slope of z2 on match-hour wind (outdoor, 2230 obs / 52 events): -0.089 per +10 mph [-0.390, +0.181]

**Design C (serve-rally rate) — per-group level and excess over the simulated null**

| group | n | mean z2 [95% CI] | sim null z2 | excess [95% CI] |
|---|---|---|---|---|
| INDOOR | 1903 | 1.126 [1.069, 1.178] | 1.154 | -0.028 [-0.094, +0.031] |
| OUTDOOR calm <8 | 2403 | 1.150 [1.084, 1.216] | 1.166 | -0.016 [-0.078, +0.045] |
| OUTDOOR moderate 8-14 | 1037 | 1.070 [0.970, 1.183] | 1.164 | -0.094 [-0.192, +0.015] |
| OUTDOOR windy 14+ | 155 | 1.251 [1.025, 1.500] | 1.179 | +0.072 [-0.155, +0.319] |

**Design C (serve-rally rate) — contrast vs OUTDOOR calm <8: UNPAIRED (published design) vs PAIRED (within-event)**

| contrast | unpaired Δz2 [95% CI] | paired-FE Δ [boot 95%] | paired-FE [t, G-1 df] | paired-ATT | paired-unit | events | perm p (1-sided / 2-sided) |
|---|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | -0.080 [-0.197, +0.052] | -0.149 [-0.270, -0.028] | [-0.270, -0.028] | -0.090 [-0.283, +0.094] | -0.063 [-0.275, +0.181] | 39 | 0.986 / 0.030 |
| OUTDOOR windy 14+ - calm | +0.101 [-0.151, +0.367] | +0.023 [-0.327, +0.360] | [-0.374, +0.419] | +0.092 [-0.234, +0.434] | -0.052 [-0.530, +0.456] | 12 | 0.438 / 0.901 |

Design C: WITHIN-EVENT slope of z2 on match-hour wind (outdoor, 3595 obs / 52 events): -0.082 per +10 mph [-0.260, +0.094]

---

## Label arm: `audited_hi`  (Design B n=2308 games, Design C n=3720 team-halves)

Design B group sizes: INDOOR=1170, OUTDOOR calm <8=744, OUTDOOR moderate 8-14=353, OUTDOOR windy 14+=41


**Design B (point share) — per-group level and excess over the simulated null**

| group | n | mean z2 [95% CI] | sim null z2 | excess [95% CI] |
|---|---|---|---|---|
| INDOOR | 1170 | 1.660 [1.569, 1.752] | 1.758 | -0.098 [-0.207, +0.001] |
| OUTDOOR calm <8 | 744 | 1.683 [1.563, 1.786] | 1.774 | -0.091 [-0.211, +0.006] |
| OUTDOOR moderate 8-14 | 353 | 1.435 [1.295, 1.623] | 1.807 | -0.372 [-0.509, -0.194] |
| OUTDOOR windy 14+ | 41 | 2.063 [1.398, 2.277] | 1.865 | +0.198 [-0.522, +0.431] |

**Design B (point share) — contrast vs OUTDOOR calm <8: UNPAIRED (published design) vs PAIRED (within-event)**

| contrast | unpaired Δz2 [95% CI] | paired-FE Δ [boot 95%] | paired-FE [t, G-1 df] | paired-ATT | paired-unit | events | perm p (1-sided / 2-sided) |
|---|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | -0.248 [-0.432, -0.016] | -0.252 [-0.406, -0.021] | [-0.451, -0.052] | -0.185 [-0.388, +0.075] | -0.148 [-0.414, +0.128] | 18 | 0.969 / 0.065 |
| OUTDOOR windy 14+ - calm | +0.380 [-0.575, +0.656] | +0.237 [-0.364, +0.581] | [-0.637, +1.111] | +0.331 [-0.353, +0.600] | +0.159 [-0.243, +0.517] | 4 | 0.303 / 0.615 |

Design B: WITHIN-EVENT slope of z2 on match-hour wind (outdoor, 1138 obs / 20 events): -0.311 per +10 mph [-0.745, +0.000]

**Design C (serve-rally rate) — per-group level and excess over the simulated null**

| group | n | mean z2 [95% CI] | sim null z2 | excess [95% CI] |
|---|---|---|---|---|
| INDOOR | 1903 | 1.126 [1.069, 1.178] | 1.163 | -0.037 [-0.100, +0.021] |
| OUTDOOR calm <8 | 1192 | 1.137 [1.068, 1.209] | 1.161 | -0.025 [-0.096, +0.048] |
| OUTDOOR moderate 8-14 | 557 | 1.025 [0.922, 1.142] | 1.140 | -0.114 [-0.229, +0.015] |
| OUTDOOR windy 14+ | 68 | 1.108 [0.894, 1.229] | 1.135 | -0.027 [-0.237, +0.048] |

**Design C (serve-rally rate) — contrast vs OUTDOOR calm <8: UNPAIRED (published design) vs PAIRED (within-event)**

| contrast | unpaired Δz2 [95% CI] | paired-FE Δ [boot 95%] | paired-FE [t, G-1 df] | paired-ATT | paired-unit | events | perm p (1-sided / 2-sided) |
|---|---|---|---|---|---|---|---|
| OUTDOOR moderate 8-14 - calm | -0.112 [-0.241, +0.029] | -0.077 [-0.157, +0.021] | [-0.170, +0.017] | -0.015 [-0.141, +0.134] | -0.041 [-0.156, +0.078] | 18 | 0.824 / 0.357 |
| OUTDOOR windy 14+ - calm | -0.029 [-0.653, +0.154] | -0.102 [-0.287, +0.023] | [-0.442, +0.238] | -0.096 [-0.289, +0.023] | -0.090 [-0.239, +0.023] | 4 | 0.616 / 0.722 |

Design C: WITHIN-EVENT slope of z2 on match-hour wind (outdoor, 1817 obs / 20 events): -0.118 per +10 mph [-0.233, -0.016]
