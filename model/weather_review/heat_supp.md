# TASK B1 part 3 — joint controls, corrected-label binned stat, power

## 1. Everything at once: event fixed effects + hour-of-day controls, odd (symmetrized) spec

| setting | games | d (skill x heat) [95% CI] |
|---|---|---|
| outdoor | 24718 | +0.029 [+0.005, +0.058] |
| indoor | 7127 | -0.012 [-0.035, +0.002] |

## 2. The PUBLISHED statistic (favourite obs-pred edge by temp bin, match hour) recomputed with AUDITED venue labels

Published outdoor (heuristic labels): -0.039 / -0.041 / -0.044 / -0.049 across <70 / 70-82 / 82-92 / 92+ F. The overall -4 pp is a level miscalibration of the race DP, not weather; only the DRIFT across bins is a heat test.

| setting | bin | games | pred fav % | obs fav % | edge (obs-pred) [95% CI] | edge minus <70F bin [95% CI] |
|---|---|---|---|---|---|---|
| outdoor | <70F | 9233 | 0.817 | 0.772 | -0.044 [-0.053, -0.035] | +0.000 [+0.000, +0.000] |
| outdoor | 70-82F | 12277 | 0.811 | 0.772 | -0.039 [-0.047, -0.032] | +0.005 [-0.005, +0.016] |
| outdoor | 82-92F | 2913 | 0.806 | 0.775 | -0.031 [-0.044, -0.015] | +0.014 [-0.001, +0.032] |
| outdoor | 92F+ | 295 | 0.803 | 0.793 | -0.010 [-0.044, +0.078] | +0.039 [-0.001, +0.121] |
| indoor | <70F | 2840 | 0.815 | 0.790 | -0.025 [-0.039, -0.007] | +0.000 [+0.000, +0.000] |
| indoor | 70-82F | 1724 | 0.798 | 0.749 | -0.049 [-0.077, -0.018] | -0.026 [-0.057, +0.010] |
| indoor | 82-92F | 1689 | 0.810 | 0.745 | -0.066 [-0.091, -0.040] | -0.042 [-0.074, -0.011] |
| indoor | 92F+ | 874 | 0.822 | 0.773 | -0.049 [-0.097, -0.025] | -0.029 [-0.076, +0.002] |

## 3. Attenuation bound from the actual-vs-planned start split

Planned-start rows carry extra hour error, so their slope should be smaller; the ratio bounds how much the pooled estimate is attenuated.

| arm | games | d (skill x heat) [95% CI] |
|---|---|---|
| actual start time | 18325 | +0.021 [-0.004, +0.048] |
| planned start time | 6393 | +0.028 [+0.005, +0.051] |

## 4. What the duration nulls can still hide (H3 power)

From heat_test.py section 4 (within event x format cells, outdoor):

| outcome | mean | CI per +10F | largest effect still allowed at +20F |
|---|---|---|---|
| rallies per match | 79.350 | [-0.426, +1.388] | -0.852 to +2.776 (-1.1% to +3.5%) |
| points per match | 35.640 | [-0.290, +0.347] | -0.580 to +0.694 (-1.6% to +1.9%) |
| rallies per point | 2.210 | [-0.001, +0.025] | -0.002 to +0.050 (-0.1% to +2.3%) |
| games per match | 1.990 | [-0.017, +0.004] | -0.034 to +0.008 (-1.7% to +0.4%) |
| 3-game rate (bo3) | 0.283 | [-0.018, +0.009] | -0.036 to +0.018 (-12.8% to +6.2%) |

## 5. Real-world translation of the primary estimate

Primary spec = antisymmetric (odd) outdoor fit: share = b*skill + d*skill*h. Below, a favourite is described by its 75F game win probability; the +20F column applies d*skill*2 to the expected point share and re-runs the race DP.

Outdoor d = +0.0165 [+1.0690, +1.1070] (n = 24718 games, 69 events); b = 1.087.

| favourite at 75F | share at 75F | win prob at 95F (point est) | (CI low = most leveling allowed) | (CI high) |
|---|---|---|---|---|
| 60% | 0.526 | 0.603 (+0.32 pp) | 0.788 (+18.77 pp) | 0.793 (+19.33 pp) |
| 75% | 0.569 | 0.757 (+0.70 pp) | 0.986 (+23.62 pp) | 0.988 (+23.81 pp) |
| 90% | 0.630 | 0.907 (+0.73 pp) | 1.000 (+10.00 pp) | 1.000 (+10.00 pp) |
| 97% | 0.688 | 0.974 (+0.42 pp) | 1.000 (+3.00 pp) | 1.000 (+3.00 pp) |

---
*model/weather_review/heat_supp.py*
