# B6 — settling the two phase-1 claims about H4

## Claim 1: does the binned −2.0pp drift really map to d = −0.072?

**heuristic labels (as published)** — outdoor 24,909 games (calm 17,139, 14–20 1,151); mean wind 4.4 vs 15.7 mph; mean favourite |skill| 0.119.

- A LINEAR interaction reproducing a −2.0 pp calm→windy drift in the favourite's win rate needs **d = -0.1040**.
- A STEP interaction switched on at 14 mph reproducing the same −2.0 pp drift needs **d_step = -0.1252** (i.e. the favourite's edge shrinks by 12.5% above 14 mph).

- If the TRUTH is that step, the continuous reg-1 coefficient on skill×wind converges to d_step × **0.316** = **-0.0396** — because only 5.0% of outdoor games are ≥14 mph and the step is nearly orthogonal to the linear term after partialling out skill and wind. The binned→continuous translation therefore only holds if the effect really is linear in wind from 0 mph up.

| spec fitted to the data | coefficient | 95% CI (event bootstrap) | value implied by the −2.0 pp binned drift | is that value excluded? |
|---|---|---|---|---|
| continuous d (skill×wind/10) | +0.0097 | [-0.0507, +0.0764] | -0.1040 (if linear) / -0.0396 (if a 14 mph step) | yes / no |
| step d (skill×1[w≥14]) | +0.0178 | [-0.0989, +0.1129] | -0.1252 | yes |

**corrected labels (venue_overrides)** — outdoor 24,718 games (calm 17,083, 14–20 1,051); mean wind 4.6 vs 15.8 mph; mean favourite |skill| 0.119.

- A LINEAR interaction reproducing a −2.0 pp calm→windy drift in the favourite's win rate needs **d = -0.1034**.
- A STEP interaction switched on at 14 mph reproducing the same −2.0 pp drift needs **d_step = -0.1231** (i.e. the favourite's edge shrinks by 12.3% above 14 mph).

- If the TRUTH is that step, the continuous reg-1 coefficient on skill×wind converges to d_step × **0.328** = **-0.0404** — because only 4.6% of outdoor games are ≥14 mph and the step is nearly orthogonal to the linear term after partialling out skill and wind. The binned→continuous translation therefore only holds if the effect really is linear in wind from 0 mph up.

| spec fitted to the data | coefficient | 95% CI (event bootstrap) | value implied by the −2.0 pp binned drift | is that value excluded? |
|---|---|---|---|---|
| continuous d (skill×wind/10) | -0.0322 | [-0.0865, +0.0181] | -0.1034 (if linear) / -0.0404 (if a 14 mph step) | yes / no |
| step d (skill×1[w≥14]) | -0.0218 | [-0.1413, +0.0662] | -0.1231 | no |

### Did the rally-level logit constrain anything?

Published: outdoor d = −0.017 [−0.098, +0.058] on adv×(wind/10), where adv is the serving team's v2 eta advantage. Translate the same −2.0 pp binned favourite drift into that parameterisation.

Put both regressions on ONE scale: *fraction of the skill edge lost per +10 mph*, which is d divided by the main skill coefficient. It is scale-free, so the game-level share regression and the rally-level logit become directly comparable.

| test | d | main skill coef | fractional compression per 10 mph [95% CI] |
|---|---|---|---|
| reg 1, game level (published, heuristic labels) | +0.002 | 1.040 | +0.002 [-0.058, +0.062] |
| reg 1, game level (corrected labels) | -0.038 | 1.051 | -0.036 [-0.091, +0.016] |
| reg 3, rally logit (published, heuristic labels) | -0.017 | 0.458 | -0.037 [-0.214, +0.127] |

The binned −2.0 pp drift corresponds to a fractional compression of -0.098 per 10 mph (using the corrected-label d above and its own skill coefficient). The rally logit's CI on that scale is [−0.214, +0.127] — six times wider than the effect it was cited as ruling out. **Confirmed: the rally-level logit never constrained the hypothesis.** Its apparent agreement with reg 1 is agreement between a tight estimate and a very loose one.

## Claim 2: is '14+ mph is hot in every design' false?

| design | rows with a wind join | outdoor (heuristic) | outdoor & ≥14 mph | distinct matches in that windy cell |
|---|---|---|---|---|
| B (point share, pre/post switch) | 5,061 | 2,571 | 118 | 118 |
| C (serve-rally rate, pre/post switch) | 5,061 | 2,571 | 118 | 118 |

Overlap of the two windy-14+ cells: **118** of 118 (B) and 118 (C) game-rows — 100% / 100%.

Under the corrected venue labels the same cells are 122 (B) and 122 (C) rows with 122 shared.

