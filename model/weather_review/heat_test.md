# TASK B1 — continuous HEAT test at match-hour resolution

Pre-specified (exertion physiology): heat is a LEVELER (d < 0 outdoors, ~0 indoors) and heat LENGTHENS matches (more rallies/points/3-game matches at fixed skill gap).

Games with a match-hour weather join: 36518 (outdoor 24718, indoor 7127, mixed 3231, unknown 1442).
Outdoor temperature (F): min 41, p10 60, median 73, p90 84, max 102; 664 games at 90F+, 126 at 95F+.
Match-hour joins using an ACTUAL start time: 73% (rest use planned start — classical measurement error, which attenuates every slope below toward zero).

## 1. Game level: share - 1/2 = a + b*skill + c*h + d*skill*h

h = (temperature_2m at match hour - 75F)/10. d is the test: negative = heat compresses the favorite's conversion of skill into points.

### 1a. Pooled (audited venue labels), temperature_2m

| setting | games | events | b (skill slope) | d (skill x heat) per +10F [95% CI] | c (heat main) | slope at 95F |
|---|---|---|---|---|---|---|
| outdoor | 24718 | 69 | 1.033 | +0.031 [+0.005, +0.058] | -0.0035 [-0.0070, +0.0001] | 1.094 |
| indoor | 7127 | 30 | 1.081 | -0.015 [-0.044, +0.007] | +0.0009 [-0.0029, +0.0038] | 1.051 |

### 1b. Within-EVENT (event fixed effects; identified off hour-to-hour and day-to-day heat swings inside one event, so no venue/season/field confound can enter)

| setting | games | events | b (skill) | d (skill x heat) [95% CI] |
|---|---|---|---|---|
| outdoor | 24718 | 69 | 1.032 | +0.030 [+0.004, +0.058] |
| indoor | 7127 | 30 | 1.069 | -0.019 [-0.049, +0.005] |

### 1c. Same, using APPARENT temperature (heat index: temp + humidity + wind + radiation — the exertion-relevant variable)

### apparent_temperature, pooled

| setting | games | events | b (skill slope) | d (skill x heat) per +10F [95% CI] | c (heat main) | slope at 95F |
|---|---|---|---|---|---|---|
| outdoor | 24718 | 69 | 1.036 | +0.033 [+0.015, +0.052] | -0.0027 [-0.0052, -0.0000] | 1.103 |
| indoor | 7127 | 30 | 1.084 | -0.008 [-0.030, +0.010] | +0.0001 [-0.0030, +0.0023] | 1.067 |

### 1d. Sensitivity: the OLD heuristic venue labels (what every published test used; ~26% of games are mislabeled)

### heuristic labels, temperature_2m

| setting | games | events | b (skill slope) | d (skill x heat) per +10F [95% CI] | c (heat main) | slope at 95F |
|---|---|---|---|---|---|---|
| outdoor | 24819 | 67 | 1.040 | -0.002 [-0.030, +0.032] | -0.0004 [-0.0046, +0.0032] | 1.035 |
| indoor | 11699 | 43 | 1.046 | -0.003 [-0.017, +0.023] | -0.0004 [-0.0046, +0.0016] | 1.041 |

## 2. Is it heat, or is it the afternoon? (H4)

Adds hour-of-day as a control AND as its own interaction with skill, so d is identified off heat variation at a GIVEN hour (different days / different venues).

| setting | games | d (skill x heat) alone | d (skill x heat) with hour controls | skill x hour |
|---|---|---|---|---|
| outdoor | 24718 | +0.031 | +0.036 [+0.009, +0.064] | -0.034 [-0.080, +0.019] |
| indoor | 7127 | -0.015 | -0.015 [-0.047, +0.007] | -0.018 [-0.073, +0.049] |

## 3. Rally level: favorite-minus-underdog serve-rally win gap vs heat (H2)

Sample = data/decider_serve_splits.csv (PPA deciders + all MLP games) — close-match-selected, so the heat SLOPE is the object, not the level. v2 only labels who the favorite is (|eta| >= 0.1).

| setting | games | mean gap | heat slope per +10F [95% CI] |
|---|---|---|---|
| outdoor | 2448 | +0.105 | +0.0058 [-0.0022, +0.0145] |
| indoor | 1010 | +0.106 | -0.0013 [-0.0098, +0.0028] |

### 3b. Rally-level binomial logit: logit P(server wins rally) = a + b*adv + c*h + d*adv*h

| setting | rallies | b (adv) | d (adv x heat) [95% CI] |
|---|---|---|---|
| outdoor | 123582 | 0.456 | +0.029 [-0.002, +0.052] |
| indoor | 50534 | 0.461 | -0.010 [-0.025, +0.006] |

## 4. Does heat LENGTHEN matches? (H3 — the channel the wind work never had)

Within (event x format) cells, so format/venue/season are differenced out; identified off heat swings inside one event. Controls: |skill gap|. Positive = heat lengthens.

**rallies per match**

| setting | matches | mean | heat slope per +10F [95% CI] | + hour control |
|---|---|---|---|---|
| outdoor | 8072 | 79.35 | +0.577 [-0.329, +1.389] | +0.727 [-0.210, +1.738] |
| indoor | 2505 | 70.53 | -0.105 [-1.314, +1.025] | -0.334 [-1.616, +0.733] |

**points per match**

| setting | matches | mean | heat slope per +10F [95% CI] | + hour control |
|---|---|---|---|---|
| outdoor | 8072 | 35.64 | +0.053 [-0.249, +0.371] | +0.099 [-0.241, +0.454] |
| indoor | 2505 | 31.68 | +0.056 [-0.485, +0.547] | -0.018 [-0.517, +0.414] |

**rallies per point**

| setting | matches | mean | heat slope per +10F [95% CI] | + hour control |
|---|---|---|---|---|
| outdoor | 8072 | 2.21 | +0.012 [-0.001, +0.025] | +0.013 [+0.000, +0.026] |
| indoor | 2505 | 2.22 | -0.013 [-0.032, +0.007] | -0.014 [-0.042, +0.011] |

**games per match**

| setting | matches | mean | heat slope per +10F [95% CI] | + hour control |
|---|---|---|---|---|
| outdoor | 12390 | 1.99 | -0.006 [-0.016, +0.005] | -0.008 [-0.019, +0.004] |
| indoor | 4019 | 1.77 | +0.007 [-0.019, +0.025] | +0.005 [-0.018, +0.021] |

**3-game rate (best-of-3 matches only; 1 = went to a decider)**

| setting | matches | rate | heat slope per +10F [95% CI] |
|---|---|---|---|
| outdoor | 9378 | 0.283 | -0.0065 [-0.0184, +0.0075] |
| indoor | 2366 | 0.289 | +0.0114 [-0.0254, +0.0391] |

## 5. Secondary channels: humidity, apparent temp, precipitation

Same reg-1 form with the channel in place of h (units: RH per +10 pct pts, apparent per +10F, precip per +1 mm/h).

| channel | setting | games | d (skill x channel) [95% CI] | main effect c [95% CI] |
|---|---|---|---|---|
| apparent temp /10F | outdoor | 24718 | +0.033 [+0.017, +0.053] | -0.0027 [-0.0053, -0.0000] |
| apparent temp /10F | indoor | 7127 | -0.008 [-0.029, +0.009] | +0.0001 [-0.0029, +0.0024] |
| humidity /10pp | outdoor | 24718 | +0.010 [-0.001, +0.020] | -0.0000 [-0.0018, +0.0017] |
| humidity /10pp | indoor | 7127 | +0.021 [-0.001, +0.035] | -0.0019 [-0.0041, +0.0013] |
| precip mm/h | outdoor | 24718 | +0.173 [-2.239, +0.872] | +0.0193 [-0.0399, +0.2964] |
| precip mm/h | indoor | 7127 | -0.477 [-2.572, +0.284] | -0.0233 [-0.2275, +0.0976] |

## 6. Translation: what the outdoor CI still allows

Outdoor d = +0.031 share per unit skill per +10F, 95% CI [+0.005, +0.058] (n = 24718 games, 69 events).
  - a 75% favorite at 75F -> 76.3% at 95F (point est: share shift +0.42 pp)
  - a 75% favorite at 75F -> 75.2% at 95F (CI low: share shift +0.07 pp)
  - a 75% favorite at 75F -> 77.5% at 95F (CI high: share shift +0.81 pp)
  - a 90% favorite at 75F -> 91.3% at 95F (point est: share shift +0.80 pp)
  - a 90% favorite at 75F -> 90.2% at 95F (CI low: share shift +0.12 pp)
  - a 90% favorite at 75F -> 92.4% at 95F (CI high: share shift +1.52 pp)

---
*Deterministic: all RNGs seeded. Written by model/weather_review/heat_test.py.*
