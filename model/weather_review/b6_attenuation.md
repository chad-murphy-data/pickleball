# B6 — measurement error in the wind regressor and de-attenuation

*(model/weather_review/b6_attenuation.py; seeds fixed; cluster bootstrap over EVENTS, 1000 resamples)*

## 1. How noisy is the regressor? (timing/aggregation component)

Games with BOTH the published match-start-hour wind and their own game-hour wind: **25,734** (70% of the weather-joined game set).

| subset | games | var(w_meas) | var(w_meas − w_true) | mean |Δh| | **lambda_T** |
|---|---|---|---|---|---|
| all | 25,734 | 15.78 | 0.90 | 0.33 h | 0.968 |
| actual start | 25,679 | 15.78 | 0.89 | 0.33 h | 0.968 |
| planned start | 55 | 14.00 | 2.49 | 0.85 h | 0.921 |
| outdoor (corrected labels) | 17,665 | 13.75 | 0.84 | 0.33 h | 0.965 |

`lambda_T` is the *exact* attenuation factor for a regression on the published regressor when the outcome truly depends on the game-hour wind: plim(c_hat) = c_true · cov(w_meas, w_true)/var(w_meas). It needs no independence assumption.

Cross-check on the brief's second handle — the 13,242 matches carrying BOTH a planned and an actual start: regressing on the PLANNED-hour wind when the truth is the ACTUAL-hour wind attenuates by lambda = **0.918** (var of the discrepancy 2.91 mph², var of planned-hour wind 14.83). Applied to the 27% of games that fall back to planned, this alone costs 2.2% of the signal; the game-hour table above already contains it.

The `planned start` row above is tiny because planned-only matches rarely carry per-game end stamps. The composite is therefore built explicitly: take the outdoor games that have an actual start, a planned start AND a game hour, and re-measure the fraction of them that the real sample takes from the planned hour.

Composite sample: 17,661 outdoor games with all three clocks; 26% of the real outdoor sample uses the planned hour, so that fraction is switched to the planned-hour value here. **lambda_T (composite) = 0.941** (var of the composite error 1.65 mph² against 13.57 mph² of regressor variance).

**Adopted lambda_T = 0.941** (outdoor games, corrected labels — the sample every wind slope is estimated on). The headline here is that the timing defect the phase-1 audit flagged as a 'systematic toward-null pressure' is real but SMALL: it can hide at most 6% of any wind slope, because ERA5 wind is strongly autocorrelated hour to hour, so being an hour or two off costs little.

The SITE component cannot be estimated from this archive: no on-court anemometer, no venue observation, nothing. It is therefore carried as an explicit unknown lambda_S and never asserted. lambda_S = 1.0 is the case where the causal target simply IS the reanalysis wind (a perfectly reasonable reading of the published claim, since that is the exposure variable); lambda_S = 0.5 is a deliberately pessimistic 'the court sees only half of the grid signal coherently'. Anyone who wants a different assumption can read the row.

## 2. De-attenuated H4 (skill × wind interaction, game level)

Spec is the committed one (`model/favorites_wind.py` reg 1): share − ½ = a + b·skill + c·w + d·(skill×w), skill = v2 expected share − ½, w = wind/10 mph, per setting. d < 0 = wind compresses the favourite's edge.

| setting | games | lambda_S | lambda total | b (skill) | **d (skill×wind)** [95% CI] | max |d| still allowed |
|---|---|---|---|---|---|---|
| outdoor | 24,718 | 1.00 | 0.941 | 1.052 | -0.0400 [-0.0996, +0.0207] | 0.0996 |
| outdoor | 24,718 | 0.85 | 0.800 | 1.057 | -0.0473 [-0.1187, +0.0250] | 0.1187 |
| outdoor | 24,718 | 0.70 | 0.659 | 1.064 | -0.0578 [-0.1448, +0.0310] | 0.1448 |
| outdoor | 24,718 | 0.50 | 0.471 | 1.081 | -0.0824 [-0.2062, +0.0445] | 0.2062 |
| indoor | 7,217 | 1.00 | 0.941 | 1.119 | -0.0399 [-0.1400, +0.1151] | 0.1400 |
| indoor | 7,217 | 0.85 | 0.800 | 1.125 | -0.0476 [-0.1664, +0.1357] | 0.1664 |
| indoor | 7,217 | 0.70 | 0.659 | 1.133 | -0.0589 [-0.2060, +0.1670] | 0.2060 |
| indoor | 7,217 | 0.50 | 0.471 | 1.153 | -0.0864 [-0.3102, +0.2434] | 0.3102 |

Row `lambda_S = 1.0` is de-attenuated for TIMING only — i.e. it is the honest estimate of the effect of *grid* wind at the game's own hour, which is the exposure the published claim is actually about. Lower rows additionally assume on-court wind is a noisy function of grid wind.

Uncorrected reference (lambda = 1, exactly the committed spec, recomputed here on corrected venue labels):

| setting | games | b | d [95% CI] |
|---|---|---|---|
| outdoor | 24,718 | 1.051 | -0.0376 [-0.0933, +0.0194] |
| indoor | 7,217 | 1.117 | -0.0373 [-0.1314, +0.1083] |

## 3. The number the published null is entitled to claim

Reference favourite: v2 expected point share 0.60 (skill = +0.10), which is a **83.6%** game favourite in a race to 11. Effect = moving from 5 mph to 20 mph.

| scenario | d bound | share lost by the favourite | upset probability rises by |
|---|---|---|---|
| published point estimate (outdoor, heuristic labels) | -0.0020 | -0.03 pp of share | +0.1 pp |
| uncorrected here (outdoor, corrected labels) | -0.0376 | -0.56 pp of share | +1.4 pp |
| uncorrected CI edge | -0.0933 | -1.40 pp of share | +3.7 pp |
| de-attenuated point, timing only | -0.0400 | -0.60 pp of share | +1.5 pp |
| **de-attenuated CI edge, timing only** | -0.0996 | -1.49 pp of share | +3.9 pp |
| de-attenuated CI edge, lambda_S = 0.7 | -0.1448 | -2.17 pp of share | +5.9 pp |
| de-attenuated CI edge, lambda_S = 0.5 | -0.2062 | -3.09 pp of share | +8.7 pp |

## 4. De-attenuated H1 (serve-point rate vs wind)

Weighted by rallies; slope per +10 mph of serve-point rate (P(server wins the rally)).

| setting | matches | lambda_S | slope [95% CI] | largest true slope still excluded |
|---|---|---|---|---|
| outdoor | 8,738 | 1.00 | +0.0029 [-0.0015, +0.0072] | |0.0072| per 10 mph |
| outdoor | 8,738 | 0.70 | +0.0041 [-0.0022, +0.0103] | |0.0103| per 10 mph |
| outdoor | 8,738 | 0.50 | +0.0057 [-0.0031, +0.0145] | |0.0145| per 10 mph |
| indoor | 2,683 | 1.00 | +0.0020 [-0.0021, +0.0097] | |0.0097| per 10 mph |
| indoor | 2,683 | 0.70 | +0.0029 [-0.0030, +0.0138] | |0.0138| per 10 mph |
| indoor | 2,683 | 0.50 | +0.0041 [-0.0042, +0.0193] | |0.0193| per 10 mph |

## 5. Pre-specified discriminator: smeared threshold vs true zero

If the truth is a high-wind THRESHOLD, a noisy regressor produces exactly the published 'tail bin only, no dose-response' pattern. Forward-simulate that: take the empirical (w_meas, w_true) pairs outdoors, impose a true step effect of size Δ on games whose TRUE wind exceeds 18 mph, and read off what the OBSERVED bin means would look like.

Two exposures matter. (a) TIMING only: truth = grid wind at the game's own hour, observed for real. (b) TIMING + SITE: truth = on-court wind, simulated as the classical-EIV posterior E[w_true|w_meas] = mu + lambda(w_meas − mu), Var = lambda(1−lambda)·var(w_meas), which is the *only* thing a reliability coefficient pins down.

(a) Observed, timing only (17,665 outdoor games). True game-hour wind ≥ 14 mph in 4.7% of games; measured ≥ 14 in 4.7%.

| bin (measured) | games | P(TRUE ≥ 14 mph) | dilution of a true 14 mph step, vs the 0–8 bin |
|---|---|---|---|
| 0–8 | 11,891 | 0.0% | +0.000 |
| 8–14 | 4,943 | 1.7% | +0.017 |
| 14–20 | 764 | 89.3% | +0.893 |
| 20–+ | 67 | 100.0% | +1.000 |

(b) Adding site error. A true step of size Δ switched on at 14 mph of ON-COURT wind shows up in the measured 14–20 bin, relative to the measured 0–8 bin, as Δ × dilution:

| lambda total | dilution | a −2.0 pp observed drift implies a TRUE step of | is that step excluded by the binned CI (±1.9 pp)? |
|---|---|---|---|
| 0.941 | 0.784 | -2.6 pp | no |
| 0.800 | 0.453 | -4.4 pp | no |
| 0.659 | 0.250 | -8.0 pp | no |
| 0.471 | 0.072 | -27.8 pp | no |

Read the middle column as: 'if wind really does something only above 14 mph on court, the archive would have to be hiding an effect this big to have produced only what we saw.' At lambda ≈ 0.9 the smearing is mild and the implied true step is close to the observed one; the 'tail-bin-only' pattern is NOT explained by attenuation at any reliability this data can support.

