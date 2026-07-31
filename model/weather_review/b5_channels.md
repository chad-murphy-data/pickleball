# B5 — untested weather channels: gusts, rain, cold, swirl, day/night

Match-hour joins: 12885 matches with rally logs, 36518 games with full v2 ratings. Dropped: 57 matches with no start time, 1 with no hourly weather row.

## Pre-specification (written before reading any result below)

**PRIMARY channel: GUSTINESS = windgusts_10m - windspeed_10m at match hour,
controlling for sustained wind.** Two pre-registered coefficients:

* **S** (serve-point rate): `serve_rate ~ 1 + sust + gustiness`, WLS by rallies.
  Signal = gustiness slope outdoors whose 95% cluster-bootstrap CI excludes 0
  AND is |>= 0.010| per 10 mph of gustiness (1.0 pp of serve-point rate --
  the smallest change that would matter for the live win-prob DP), AND the
  indoor falsification arm does NOT move the same way.
* **F** (favorite compression): `share-1/2 ~ 1 + skill + sust + gustiness
  + skill*sust + skill*gustiness`. Signal = `skill x gustiness` coefficient
  d < 0 outdoors with CI excluding 0, |d| >= 0.05 (i.e. >= 5% of the skill
  slope destroyed per 10 mph of gustiness), and indoor d not equally negative.

Secondary channels, same two outcomes each, same rules: plain gust speed,
wet ball (recent precipitation), cold tail, direction swirl, night session.
Family-wise: 12 outdoor primary coefficients (6 channels x 2 outcomes);
Holm-Bonferroni across the family is reported alongside raw.

Everything runs on CORRECTED labels (data/venue_overrides.csv, mixed/unknown
events dropped). Robustness arms: published labels, high-confidence-only
labels, and actual-start-time-only matches.

## Exposure: what the untested channels actually look like

Corrected labels: 24718 outdoor games / 7127 indoor games; 8738 outdoor matches / 2666 indoor matches with logs.

| channel (outdoor games) | p10 | p50 | p90 | p99 | max | n>threshold |
|---|---|---|---|---|---|---|
| sust | 2.3 | 6.1 | 12.0 | 16.8 | 25.5 | 1133 >= 14 mph |
| gust | 8.9 | 15.4 | 25.5 | 36.9 | 53.7 | 2839 >= 25 mph |
| gustiness | 5.6 | 9.4 | 14.2 | 21.1 | 28.6 | 5410 >= 12 mph |
| swirl | 2.7 | 10.1 | 39.9 | 83.6 | 187.9 | 2103 >= 45 deg |
| temperature (F) | 60.3 | 73.2 | 83.9 | 92.3 | 102.2 | 2393 < 60F |
| wet (precip h-2..h) | - | - | - | - | - | 4627 any / 3675 light / 952 >0.05in |
| night (local start >= 17h) | - | - | - | - | - | 3453 of 24718 |

**Concealment check**: of 2839 outdoor games at gust >= 25 mph, 1713 (60%) sit in the published calm/moderate SUSTAINED bins (<14 mph). The published binning did hide most of the gust exposure.

corr(sustained, gustiness) outdoor = +0.710 -- they are far from collinear, so the control is identified.

## Results — outdoor (test) vs indoor (falsification), corrected labels

### S. Serve-point rate (WLS by rallies)

| channel | arm | n matches | events | slope | 95% CI | raw p |
|---|---|---|---|---|---|---|
| gustiness (gust-sust), per 10 mph | outdoor | 8738 | 68 | -0.0049 | [-0.0109, +0.0005] | 0.073 |
| gustiness (gust-sust), per 10 mph | indoor | 2666 | 28 | -0.0002 | [-0.0158, +0.0115] | 0.937 |
| gust speed, per 10 mph | outdoor | 8738 | 68 | +0.0007 | [-0.0016, +0.0031] | 0.595 |
| gust speed, per 10 mph | indoor | 2666 | 28 | +0.0011 | [-0.0016, +0.0043] | 0.394 |
| wet ball: precip>0.01in in h-2..h | outdoor | 8738 | 68 | +0.0025 | [-0.0014, +0.0061] | 0.187 |
| wet ball: precip>0.01in in h-2..h | indoor | 2666 | 28 | -0.0058 | [-0.0153, +0.0016] | 0.124 |
| cold: degrees below 60F, per 10F | outdoor | 8738 | 68 | +0.0056 | [-0.0035, +0.0169] | 0.231 |
| cold: degrees below 60F, per 10F | indoor | 2666 | 28 | -0.0002 | [-0.0017, +0.0074] | 0.853 |
| swirl: circular sd of wind dir, per 30 deg | outdoor | 8738 | 68 | -0.0023 | [-0.0045, +0.0000] | 0.050 |
| swirl: circular sd of wind dir, per 30 deg | indoor | 2666 | 28 | +0.0006 | [-0.0033, +0.0048] | 0.673 |
| night session (local start >= 17h) | outdoor | 8738 | 68 | +0.0006 | [-0.0027, +0.0040] | 0.742 |
| night session (local start >= 17h) | indoor | 2666 | 28 | -0.0067 | [-0.0137, +0.0003] | 0.060 |

### F. Favorite compression (game-level skill x channel interaction)

Coefficient shown = `skill x channel`. d<0 means the channel eats the favourite's edge. Skill slope b (at channel=0) given for scale.

| channel | arm | n games | events | b (skill) | d (skill x chan) | 95% CI | raw p |
|---|---|---|---|---|---|---|---|
| gustiness (gust-sust), per 10 mph | outdoor | 24718 | 69 | 1.039 | +0.0214 | [-0.0711, +0.1120] | 0.646 |
| gustiness (gust-sust), per 10 mph | indoor | 7127 | 30 | 1.102 | +0.0240 | [-0.1039, +0.1560] | 0.777 |
| gust speed, per 10 mph | outdoor | 24718 | 69 | 1.051 | -0.0154 | [-0.0448, +0.0158] | 0.328 |
| gust speed, per 10 mph | indoor | 7127 | 30 | 1.120 | -0.0171 | [-0.0738, +0.0677] | 0.780 |
| wet ball: precip>0.01in in h-2..h | outdoor | 24718 | 69 | 1.051 | +0.0098 | [-0.0722, +0.0976] | 0.830 |
| wet ball: precip>0.01in in h-2..h | indoor | 7127 | 30 | 1.114 | +0.0724 | [-0.0516, +0.1613] | 0.272 |
| cold: degrees below 60F, per 10F | outdoor | 24718 | 69 | 1.055 | -0.0941 | [-0.2398, +0.0264] | 0.117 |
| cold: degrees below 60F, per 10F | indoor | 7127 | 30 | 1.113 | +0.0166 | [-0.0448, +0.1076] | 0.483 |
| swirl: circular sd of wind dir, per 30 deg | outdoor | 24718 | 69 | 1.068 | -0.0175 | [-0.0643, +0.0222] | 0.382 |
| swirl: circular sd of wind dir, per 30 deg | indoor | 7127 | 30 | 1.185 | -0.0717 | [-0.1274, +0.0080] | 0.073 |
| night session (local start >= 17h) | outdoor | 24718 | 69 | 1.050 | +0.0165 | [-0.0355, +0.0747] | 0.515 |
| night session (local start >= 17h) | indoor | 7127 | 30 | 1.125 | -0.0733 | [-0.1487, -0.0037] | 0.038 |

## Family-wise multiplicity

Family = 12 pre-declared OUTDOOR coefficients (6 channels x 2 outcomes). Holm-Bonferroni:

| test | raw p | Holm-adjusted p | survives 0.05? |
|---|---|---|---|
| S:swirl | 0.050 | 0.606 | no |
| S:gustiness | 0.073 | 0.808 | no |
| F:cold | 0.117 | 1.000 | no |
| S:wet | 0.187 | 1.000 | no |
| S:cold | 0.231 | 1.000 | no |
| F:gust | 0.328 | 1.000 | no |
| F:swirl | 0.382 | 1.000 | no |
| F:night | 0.515 | 1.000 | no |
| S:gust | 0.595 | 1.000 | no |
| F:gustiness | 0.646 | 1.000 | no |
| S:night | 0.742 | 1.000 | no |
| F:wet | 0.830 | 1.000 | no |

## Minimum detectable effect (80% power, two-sided 0.05)

MDE = 2.80 x cluster-bootstrap SE, translated into real units.

| test | channel | SE | MDE (coef) | MDE in real-world units |
|---|---|---|---|---|
| S | gustiness | 0.0029 | 0.0082 | 0.82 pp of serve-point rate per 10 mph |
| F | gustiness | 0.0471 | 0.1318 | 1.98 pp of point share for a 65% favourite, per 10 mph |
| S | gust | 0.0012 | 0.0034 | 0.34 pp of serve-point rate per 10 mph |
| F | gust | 0.0157 | 0.0440 | 0.66 pp of point share for a 65% favourite, per 10 mph |
| S | wet | 0.0019 | 0.0053 | 0.53 pp of serve-point rate (on/off) |
| F | wet | 0.0436 | 0.1221 | 1.83 pp of point share for a 65% favourite, per on/off |
| S | cold | 0.0051 | 0.0144 | 1.44 pp of serve-point rate per 10F |
| F | cold | 0.0691 | 0.1935 | 2.90 pp of point share for a 65% favourite, per 10F |
| S | swirl | 0.0012 | 0.0032 | 0.32 pp of serve-point rate per 30 deg |
| F | swirl | 0.0219 | 0.0614 | 0.92 pp of point share for a 65% favourite, per 30 deg |
| S | night | 0.0017 | 0.0048 | 0.48 pp of serve-point rate (on/off) |
| F | night | 0.0277 | 0.0777 | 1.17 pp of point share for a 65% favourite, per on/off |

## Binned view of the PRIMARY channel (gustiness)


**outdoor** serve-point rate by gustiness bin

| gustiness (mph) | matches | rallies | serve-point rate | 95% CI |
|---|---|---|---|---|
| 0-4 | 226 | 17854 | 0.4550 | [0.4476, 0.4597] |
| 4-7 | 1280 | 105004 | 0.4498 | [0.4461, 0.4529] |
| 7-10 | 3027 | 242625 | 0.4486 | [0.4459, 0.4514] |
| 10-14 | 3134 | 245386 | 0.4476 | [0.4454, 0.4497] |
| 14-+ | 1071 | 84009 | 0.4513 | [0.4462, 0.4563] |

**indoor** serve-point rate by gustiness bin

| gustiness (mph) | matches | rallies | serve-point rate | 95% CI |
|---|---|---|---|---|
| 0-4 | 55 | 4610 | 0.4534 | [0.4438, 0.4637] |
| 4-7 | 328 | 24439 | 0.4504 | [0.4437, 0.4557] |
| 7-10 | 755 | 54679 | 0.4496 | [0.4435, 0.4551] |
| 10-14 | 940 | 65788 | 0.4462 | [0.4409, 0.4516] |
| 14-+ | 588 | 40198 | 0.4518 | [0.4446, 0.4556] |

**Favourite point-share edge by gustiness bin** (obs share of the v2 favourite minus its model-expected share; negative = favourites underperform)


*outdoor*

| gustiness (mph) | games | events | obs-exp share | 95% CI |
|---|---|---|---|---|
| 0-4 | 1005 | 27 | +1.61 pp | [+0.60, +2.41] |
| 4-7 | 4113 | 53 | +1.54 pp | [+0.93, +2.08] |
| 7-10 | 8739 | 67 | +1.17 pp | [+0.78, +1.57] |
| 10-14 | 8057 | 65 | +1.63 pp | [+1.20, +2.05] |
| 14-+ | 2778 | 41 | +1.00 pp | [+0.33, +1.76] |

*indoor*

| gustiness (mph) | games | events | obs-exp share | 95% CI |
|---|---|---|---|---|
| 0-4 | 286 | 12 | +3.19 pp | [+1.90, +4.49] |
| 4-7 | 1138 | 23 | +1.13 pp | [-0.60, +2.47] |
| 7-10 | 2017 | 29 | +1.45 pp | [+0.51, +2.50] |
| 10-14 | 2310 | 29 | +1.73 pp | [+0.83, +2.55] |
| 14-+ | 1376 | 22 | +1.66 pp | [+0.23, +3.49] |

## Robustness on the PRIMARY channel

| arm / variant | S: gustiness slope [CI] | F: skill x gustiness [CI] |
|---|---|---|
| corrected labels (primary) — outdoor | -0.0049 [-0.0109,+0.0005] (n=8738) | +0.0214 [-0.0711,+0.1120] (n=24718) |
| corrected labels (primary) — indoor | -0.0002 [-0.0158,+0.0115] (n=2666) | +0.0240 [-0.1039,+0.1560] (n=7127) |
| published labels — outdoor | -0.0049 [-0.0114,+0.0011] (n=7905) | +0.0533 [-0.0404,+0.1503] (n=24819) |
| published labels — indoor | -0.0039 [-0.0161,+0.0063] (n=4980) | -0.0178 [-0.1455,+0.1098] (n=11699) |
| high-confidence labels only — outdoor | -0.0123 [-0.0257,+0.0014] (n=2546) | +0.1230 [-0.1576,+0.3259] (n=6138) |
| high-confidence labels only — indoor | -0.0005 [-0.0183,+0.0113] (n=2239) | +0.0109 [-0.1356,+0.1550] (n=5986) |
| corrected + ACTUAL start times only — outdoor | -0.0059 [-0.0123,+0.0001] (n=8558) | +0.0451 [-0.0514,+0.1417] (n=18325) |
| corrected + ACTUAL start times only — indoor | +0.0004 [-0.0154,+0.0118] (n=2620) | +0.0248 [-0.0990,+0.1832] (n=5009) |
| corrected + PPA only — outdoor | -0.0046 [-0.0110,+0.0010] (n=7691) | +0.0129 [-0.0800,+0.1045] (n=23546) |
| corrected + PPA only — indoor | +0.0021 [-0.0166,+0.0137] (n=1897) | +0.0501 [-0.0741,+0.2028] (n=6212) |

## Extra cuts

### Temperature, both tails (serve rate; outdoor)


*outdoor*

| temp (F) | matches | rallies | serve-point rate |
|---|---|---|---|
| <-55 | 226 | 17650 | 0.4495 |
| 55-65 | 1191 | 98317 | 0.4529 |
| 65-75 | 3356 | 280013 | 0.4491 |
| 75-85 | 3125 | 244812 | 0.4479 |
| 85-92 | 719 | 46594 | 0.4446 |
| 92-+ | 121 | 7492 | 0.4481 |

*indoor*

| temp (F) | matches | rallies | serve-point rate |
|---|---|---|---|
| <-55 | 726 | 59162 | 0.4509 |
| 55-65 | 126 | 8753 | 0.4608 |
| 65-75 | 278 | 15923 | 0.4527 |
| 75-85 | 583 | 38751 | 0.4428 |
| 85-92 | 601 | 40555 | 0.4466 |
| 92-+ | 352 | 26570 | 0.4521 |

### Favourite edge (obs-exp point share) by temperature and session


*outdoor*

| cut | games | events | obs-exp share | 95% CI |
|---|---|---|---|---|
| temp <-55F | 1112 | 15 | +0.43 pp | [-0.96, +1.42] |
| temp 55-65F | 4055 | 39 | +1.05 pp | [+0.35, +1.75] |
| temp 65-75F | 9166 | 61 | +1.42 pp | [+1.03, +1.83] |
| temp 75-85F | 8403 | 59 | +1.55 pp | [+1.12, +1.98] |
| temp 85-92F | 1687 | 25 | +1.71 pp | [+0.60, +2.87] |
| temp 92-+F | 295 | 7 | +1.83 pp | [+0.61, +3.98] |
| night (>=17h) | 3453 | 66 | +1.36 pp | [+0.83, +1.92] |
| day (<17h) | 21265 | 69 | +1.39 pp | [+1.10, +1.69] |
| wet h-2..h | 2917 | 33 | +1.50 pp | [+0.76, +2.31] |
| dry | 21801 | 69 | +1.37 pp | [+1.10, +1.65] |
| gust >= 25 mph | 2839 | 37 | +1.07 pp | [+0.37, +1.79] |
| gust < 25 mph | 21879 | 69 | +1.43 pp | [+1.15, +1.70] |

*indoor*

| cut | games | events | obs-exp share | 95% CI |
|---|---|---|---|---|
| temp <-55F | 2003 | 7 | +2.04 pp | [+1.19, +2.84] |
| temp 55-65F | 553 | 10 | +3.62 pp | [+2.12, +4.70] |
| temp 65-75F | 983 | 14 | +1.60 pp | [-0.34, +3.43] |
| temp 75-85F | 1454 | 23 | +1.04 pp | [+0.01, +2.08] |
| temp 85-92F | 1260 | 15 | +0.61 pp | [-0.54, +1.80] |
| temp 92-+F | 874 | 10 | +1.70 pp | [-0.11, +3.72] |
| night (>=17h) | 926 | 28 | +0.46 pp | [-0.63, +1.64] |
| day (<17h) | 6201 | 30 | +1.77 pp | [+1.09, +2.44] |
| wet h-2..h | 1071 | 18 | +1.49 pp | [+0.27, +2.45] |
| dry | 6056 | 30 | +1.62 pp | [+0.91, +2.34] |
| gust >= 25 mph | 1406 | 16 | +1.23 pp | [+0.07, +2.70] |
| gust < 25 mph | 5721 | 30 | +1.69 pp | [+1.00, +2.32] |

### COLD deep dive (the outdoor temp gradient above is monotone)

Linear (un-hinged) temperature interaction, `skill x temp/10F`. Positive = favourites do BETTER as it warms = cold compresses skill.

| arm | n games | events | skill x temp/10F | 95% CI | raw p |
|---|---|---|---|---|---|
| outdoor | 24718 | 69 | +0.0305 | [+0.0040, +0.0586] | 0.022 |
| indoor | 7127 | 30 | -0.0164 | [-0.0425, +0.0106] | 0.213 |
| **DiD (out-in)** | 31845 | 99 | +0.0470 | [+0.0084, +0.0846] | 0.018 |
| outdoor, within-event | 24718 | 69 | +0.0294 | [+0.0026, +0.0572] | 0.029 |

Cold-tail exposure outdoors: 2393 games below 60F over 25 events; the 5 biggest contribute 1428 (60%). Leave-one-event-out on the hinged `skill x cold` coefficient:
- full-sample coefficient -0.0941; leave-one-out range [-0.1637, -0.0780] across 69 events — no single event drives it.

#### Temperature interaction — spec curve and confound kill-list

The `skill x temp` result was NOT in the pre-registered family: the
linear (un-hinged) parameterisation was chosen after seeing the monotone
binned table. It must therefore clear a higher bar. The two confounds that
would manufacture it:

* **Seasonal form lookahead.** v2 values are CURRENT form applied
  retroactively. Cold games are January-March; if the field improves over a
  season, a current rating overstates the January favourite and the
  favourite "underperforms in the cold" for reasons that have nothing to do
  with temperature. Controlled by adding `skill x days-since-2024-01-01`.
* **Day/night and hour.** Cold hours inside an event are early mornings and
  late nights, which get different draws/rounds. Controlled by adding
  `skill x night` and `skill x hour`.

| spec | skill x temp/10F (outdoor) | 95% CI | raw p | indoor same spec |
|---|---|---|---|---|
| base (outdoor) | +0.0305 | [+0.0040, +0.0586] | 0.022 | -0.0164 [-0.0425, +0.0106] |
| + skill x season-time | +0.0290 | [+0.0070, +0.0545] | 0.005 | -0.0146 [-0.0377, +0.0072] |
| + skill x night | +0.0304 | [+0.0037, +0.0582] | 0.028 | -0.0158 [-0.0417, +0.0116] |
| + skill x hour | +0.0349 | [+0.0070, +0.0628] | 0.011 | -0.0160 [-0.0446, +0.0114] |
| + skill x gustiness | +0.0310 | [+0.0048, +0.0599] | 0.023 | -0.0198 [-0.0464, +0.0103] |
| all controls | +0.0330 | [+0.0108, +0.0603] | 0.003 | -0.0154 [-0.0436, +0.0071] |
| event FE | +0.0294 | [+0.0026, +0.0572] | 0.029 | -0.0201 [-0.0460, +0.0069] |
| event FE + all controls | +0.0332 | [+0.0112, +0.0596] | 0.001 | -0.0181 [-0.0462, +0.0037] |

Hinge-point sweep (outdoor, `skill x max(0, T0-temp)/10`):

| hinge T0 | games below | coef | 95% CI | raw p |
|---|---|---|---|---|
| 55F | 1112 | -0.1123 | [-0.4105, +0.1284] | 0.342 |
| 60F | 2393 | -0.0941 | [-0.2398, +0.0264] | 0.117 |
| 65F | 5167 | -0.0665 | [-0.1472, +0.0074] | 0.079 |
| 70F | 9233 | -0.0413 | [-0.0914, +0.0102] | 0.110 |
| 75F | 14333 | -0.0361 | [-0.0729, +0.0008] | 0.056 |

Heat-only hinge (`skill x max(0, temp-T0)/10`, the published heat channel re-expressed as an interaction):

| hinge T0 | games above | coef | 95% CI | raw p |
|---|---|---|---|---|
| 75F | 10207 | +0.0556 | [-0.0004, +0.1244] | 0.052 |
| 82F | 3172 | +0.0851 | [-0.0227, +0.2194] | 0.120 |
| 88F | 1206 | +0.1815 | [-0.0446, +0.4490] | 0.074 |

#### Temperature interaction — remaining confound kill-list

Three more ways to manufacture a `skill x temp` slope without any
physics: (a) **draw composition** — cold hours are early-morning qualifiers
with badly-rated players, and v2's calibration differs by round; (b) **skill
misspecification** — if the true share-vs-skill curve is not linear and the
skill distribution shifts with temperature, a linear `skill` main effect
leaks into the interaction; (c) **label error** — 26% of games carry the
wrong indoor/outdoor tag.

| spec | skill x temp/10F (outdoor) | 95% CI | raw p |
|---|---|---|---|
| + skill x qualifier | +0.0216 | [-0.0053, +0.0497] | 0.118 |
| + nonlinear skill (skill*|skill|) | +0.0300 | [+0.0035, +0.0573] | 0.026 |
| + nonlinear skill x temp | +0.0515 | [+0.0018, +0.0987] | 0.043 |
| kitchen sink | +0.0370 | [-0.0101, +0.0853] | 0.124 |

Label arms (base spec):

| label arm | outdoor coef [CI] | indoor coef [CI] |
|---|---|---|
| corrected (primary) | +0.0305 [+0.0040,+0.0586] (n=24718) | -0.0164 [-0.0425,+0.0106] (n=7127) |
| published heuristic | -0.0022 [-0.0305,+0.0327] (n=24819) | -0.0063 [-0.0193,+0.0201] (n=11699) |
| high-confidence audited only | +0.0763 [+0.0383,+0.1321] (n=6138) | -0.0231 [-0.0572,+0.0027] (n=5986) |

Tour arms and per-year replication (outdoor, base spec):

| subset | n games | events | coef | 95% CI | raw p |
|---|---|---|---|---|---|
| PPA only | 23546 | 56 | +0.0239 | [-0.0046, +0.0534] | 0.100 |
| MLP only | 1172 | 13 | +0.0933 | [-0.0149, +0.1878] | 0.073 |
| 2024 | 7660 | 17 | +0.0519 | [+0.0208, +0.1048] | 0.001 |
| 2025 | 9426 | 32 | +0.0602 | [+0.0106, +0.0970] | 0.023 |
| 2026 | 7632 | 20 | +0.0037 | [-0.0293, +0.0489] | 0.806 |
| non-qualifier rounds | 19100 | 69 | +0.0234 | [-0.0082, +0.0570] | 0.143 |

#### Temperature vs THE CALENDAR — the decisive test

Outdoors, temperature IS the calendar: cold = Jan-Mar/Nov-Dec, hot =
Jun-Aug. v2 values are end-of-2026 form applied retroactively, so if v2's
calibration sags early in each season the favourite's edge is smaller in
January for reasons that have nothing to do with the ball. A LINEAR
season-time control cannot absorb an annual sawtooth. Two tests that can:

1. Replace temperature with a pure seasonal wave `seas = cos(2pi(doy-200)/365)`
   (peaks mid-July). If the effect is calendar, `skill x seas` reproduces it
   OUTDOORS **and appears INDOORS too** (indoor venues have the same
   calendar but a thermostat).
2. Horse-race: put `skill x temp` and `skill x seas` in together, outdoors.
   Temperature survives only if within-season temperature deviations —
   a cold snap at a July event, a warm February — carry the effect.

| test | arm | coef of interest | 95% CI | raw p |
|---|---|---|---|---|
| 1. skill x SEASON only | outdoor | +0.0243 | [-0.0158, +0.0706] | 0.246 |
| 1. skill x SEASON only | indoor | -0.0780 | [-0.1458, +0.0069] | 0.075 |
| 2. horse-race: temp | outdoor | +0.0293 | [+0.0020, +0.0593] | 0.035 |
| 2. horse-race: season | outdoor | +0.0064 | [-0.0356, +0.0540] | 0.819 |
| 2. horse-race: temp | indoor | +0.0024 | [-0.0308, +0.0326] | 0.863 |
| 2. horse-race: season | indoor | -0.0839 | [-0.1788, +0.0355] | 0.145 |
| 3. event FE + ALL controls + season | outdoor | +0.0183 | [-0.0056, +0.0458] | 0.134 |
| 3. event FE + ALL controls + season | indoor | +0.0198 | [-0.0145, +0.0434] | 0.248 |

Within-event temperature spread that identifies spec 3 (outdoor): sd of temp deviation from the event mean = 6.0 F, range [-23, 20] F.

**Saturated DiD (event FE, every control, both arms pooled): outdoor-minus-indoor `skill x temp/10F` = -0.0014 [-0.0349, +0.0436], p = 0.958, n = 31845.** The identifying contrast is gone: once season, round, hour and a nonlinear skill term are in, the sheltered arm moves exactly as much as the exposed one.

Bound: the CI allows an outdoor-specific temperature effect up to 0.044 per 10F, i.e. at most 2.6 pp of point share for a 65% favourite across the full 55F-to-95F range — roughly a 4-5 pp swing in game win probability at the extreme, and nothing at all is equally consistent with the data.

**Caveat on spec 3 / the saturated DiD**: inside a 4-day event the
seasonal wave and season-time are nearly constant, so `skill x seas` and
`skill x days` are near-zero-variance regressors under event FE — including
them there is an unstable over-control, not a clean adjustment. The event FE
already absorbs everything between events (calendar, venue, field). The
right within-event saturation keeps only controls that actually vary inside
an event: hour, night, round, and the nonlinear skill term.

| arm | within-event saturated `skill x temp/10F` | 95% CI | raw p |
|---|---|---|---|
| outdoor | +0.0318 | [+0.0029, +0.0602] | 0.029 |
| indoor | -0.0222 | [-0.0529, +0.0057] | 0.111 |
| **DiD (out - in)** | +0.0540 | [+0.0140, +0.0952] | 0.011 |


### Rain intensity split (outdoor)

- wet_any: slope -0.0010 [-0.0045, +0.0023], n=8738, exposed=1709
- wet_light: slope -0.0015 [-0.0049, +0.0017], n=8738, exposed=1369
- wet_heavy: slope +0.0009 [-0.0048, +0.0071], n=8738, exposed=340

## Difference-in-differences: outdoor MINUS indoor

The single cleanest statistic per channel. Pool both arms, add an
`out` indicator, and read the `out x channel` interaction: how much MORE the
channel moves outcomes outdoors than in the sheltered control. This absorbs
any season/venue/format confound that moves both arms together (which the
binned tables below show is real for gustiness).

| channel | outcome | DiD coefficient | 95% CI | raw p | MDE (real units) |
|---|---|---|---|---|---|
| gustiness | serve rate | -0.0048 | [-0.0177, +0.0127] | 0.613 | 2.23 pp |
| gustiness | favourite compression | -0.0026 | [-0.1634, +0.1618] | 0.977 | 3.48 pp share @65% fav |
| gust | serve rate | -0.0004 | [-0.0044, +0.0032] | 0.825 | 0.54 pp |
| gust | favourite compression | +0.0018 | [-0.0909, +0.0687] | 0.935 | 1.76 pp share @65% fav |
| wet | serve rate | +0.0083 | [-0.0002, +0.0186] | 0.057 | 1.33 pp |
| wet | favourite compression | -0.0627 | [-0.1805, +0.0912] | 0.466 | 2.92 pp share @65% fav |
| cold | serve rate | +0.0058 | [-0.0061, +0.0178] | 0.322 | 1.67 pp |
| cold | favourite compression | -0.1107 | [-0.2772, +0.0175] | 0.099 | 3.21 pp share @65% fav |
| swirl | serve rate | -0.0029 | [-0.0077, +0.0014] | 0.178 | 0.65 pp |
| swirl | favourite compression | +0.0542 | [-0.0339, +0.1221] | 0.245 | 1.68 pp share @65% fav |
| night | serve rate | +0.0073 | [-0.0006, +0.0153] | 0.070 | 1.10 pp |
| night | favourite compression | +0.0898 | [+0.0042, +0.1883] | 0.038 | 1.97 pp share @65% fav |

Holm across the 12 DiD coefficients: min adjusted p = 0.462 (smallest raw p = 0.038, F:night). 0 survive 0.05.

## Within-event (event fixed effects)

Cold hours and gusty hours are not randomly assigned to events: a
cold event is an early-season northern stop with its own field, format and
court. Demeaning every variable within event (Frisch-Waugh; exactly the
event-dummy estimator) throws away all between-event variation and asks
whether the channel still moves outcomes ACROSS HOURS AND DAYS OF THE SAME
EVENT. Cluster bootstrap still over events.

Within-event spread of the channels (weighted sd of the deviation from the event mean, outdoor):

- gustiness mph: within-event sd = 2.70 (total sd 3.64)
- sustained mph: within-event sd = 2.74 (total sd 3.74)
- deg below 60F: within-event sd = 1.55 (total sd 2.06)
- swirl deg: within-event sd = 15.51 (total sd 18.07)

| channel | outcome | arm | within-event coef | 95% CI | raw p |
|---|---|---|---|---|---|
| gustiness | serve rate | outdoor | -0.0045 | [-0.0122, +0.0028] | 0.228 |
| gustiness | favourite compression | outdoor | +0.0087 | [-0.0843, +0.1003] | 0.833 |
| gustiness | serve rate | indoor | +0.0047 | [-0.0129, +0.0145] | 0.588 |
| gustiness | favourite compression | indoor | +0.0061 | [-0.1225, +0.1396] | 1.000 |
| gust | serve rate | outdoor | -0.0004 | [-0.0029, +0.0021] | 0.716 |
| gust | favourite compression | outdoor | -0.0185 | [-0.0478, +0.0132] | 0.253 |
| gust | serve rate | indoor | -0.0006 | [-0.0051, +0.0037] | 0.810 |
| gust | favourite compression | indoor | -0.0188 | [-0.0745, +0.0673] | 0.752 |
| wet | serve rate | outdoor | +0.0022 | [-0.0026, +0.0073] | 0.365 |
| wet | favourite compression | outdoor | -0.0022 | [-0.0824, +0.0864] | 0.972 |
| wet | serve rate | indoor | -0.0034 | [-0.0140, +0.0050] | 0.448 |
| wet | favourite compression | indoor | +0.0770 | [-0.0465, +0.1672] | 0.245 |
| cold | serve rate | outdoor | +0.0085 | [-0.0032, +0.0193] | 0.145 |
| cold | favourite compression | outdoor | -0.0787 | [-0.2208, +0.0417] | 0.181 |
| cold | serve rate | indoor | -0.0004 | [-0.0119, +0.0067] | 0.861 |
| cold | favourite compression | indoor | +0.0201 | [-0.0409, +0.1109] | 0.382 |
| swirl | serve rate | outdoor | -0.0027 | [-0.0055, +0.0003] | 0.069 |
| swirl | favourite compression | outdoor | -0.0162 | [-0.0609, +0.0231] | 0.418 |
| swirl | serve rate | indoor | -0.0000 | [-0.0040, +0.0047] | 0.977 |
| swirl | favourite compression | indoor | -0.0731 | [-0.1329, +0.0094] | 0.081 |
| night | serve rate | outdoor | -0.0000 | [-0.0040, +0.0042] | 0.977 |
| night | favourite compression | outdoor | +0.0169 | [-0.0349, +0.0738] | 0.507 |
| night | serve rate | indoor | -0.0047 | [-0.0114, +0.0020] | 0.141 |
| night | favourite compression | indoor | -0.0721 | [-0.1458, -0.0021] | 0.043 |

## What the control arm says about our false-positive rate

Indoor is where rain, gusts and swirl CANNOT touch the ball. Running the identical 12 tests there returns 1 coefficient(s) at raw p<0.05 and 3 at raw p<0.10 (chance: 0.6 and 1.2). Smallest indoor p = 0.038 (F:night). Any outdoor result at raw p in the 0.05-0.15 range is therefore inside the noise this pipeline generates on variables that provably do nothing.

## Attenuation: how noisy is the match-hour channel?

On 13242 matches with BOTH a planned and an actual start time, correlation between the channel measured at the planned hour and at the actual hour:

- gustiness: r = +0.874  -> if a share s of rows use planned times, the reliability of the pooled regressor is about 1-s(1-r); at s=0.30 that is 0.962, inflating any true slope by 1.04x.
- sustained: r = +0.904  -> if a share s of rows use planned times, the reliability of the pooled regressor is about 1-s(1-r); at s=0.30 that is 0.971, inflating any true slope by 1.03x.
- temperature: r = +0.981  -> if a share s of rows use planned times, the reliability of the pooled regressor is about 1-s(1-r); at s=0.30 that is 0.994, inflating any true slope by 1.01x.

Hour-to-hour persistence (a match spans ~1-1.5 h, so the hour stamp is itself only a sample of the exposure window):
- gustiness h vs h+1: r = +0.873
- sustained h vs h+1: r = +0.920

Both sources bias slopes TOWARD ZERO. Combined with ERA5 grid error (unmeasurable here), the honest reading is that every slope below is a LOWER bound on |true effect| by roughly 10-25%, and the CIs should be widened correspondingly before being read as 'nothing bigger than X is possible'.

## Swirl, conditioned on there being wind to swirl

corr(swirl, sustained) outdoor = -0.461 — direction wanders most when the wind is LIGHT, so raw swirl is partly an inverse wind proxy. Restricting to hours with real wind:

| subset | n matches | swirl slope on serve rate | 95% CI |
|---|---|---|---|
| all outdoor | 8738 | -0.0023 | [-0.0045, +0.0000] |
| sustained >= 8 mph | 2863 | -0.0027 | [-0.0124, +0.0073] |
| sustained >= 12 mph | 930 | +0.0008 | [-0.0176, +0.0210] |


### Hour of day, continuous (outdoor games, favourite edge)

- outdoor: skill x (hour-14)/6 = -0.0076 [-0.0536, +0.0437], n=24718
- indoor: skill x (hour-14)/6 = -0.0260 [-0.0866, +0.0287], n=7127

## Verdict

**The pre-registered primary (gustiness) is a null, and a reasonably tight
one.** Outdoor serve-point rate moves −0.49 pp per +10 mph of gustiness
[−1.09, +0.05]; within-event −0.45 pp; the indoor control is flat; the DiD
is −0.48 pp [−1.77, +1.27]. The binned pattern is non-monotone AND mirrored
bin-for-bin in the indoor arm, which is what a shared season/venue confound
looks like, not a wind effect. MDE 0.82 pp per 10 mph, so over the realistic
5.6→21 mph gustiness range we can exclude anything bigger than ~1.2 pp of
serve-point rate. Favourite compression by gustiness is +0.02 [−0.07, +0.11]
— dead centre on zero, MDE ≈ 2 pp of point share per 10 mph (≈3 pp of game
win probability for a 65%-share favourite).

**Plain gust speed, rain, swirl and night are nulls too**, none surviving
Holm, and two of them (swirl, night) have a same-sized or larger twin in the
indoor arm where the mechanism cannot operate. Rain in particular: only 340
outdoor matches saw >0.05 in in the three hours before start, so "they stop
play when it really rains" caps the exposure.

**The one live thread is TEMPERATURE, and it was not pre-registered.** The
outdoor favourite edge is monotone across all six temperature bins (+0.43 pp
at <55 °F rising to +1.83 pp at 92 °F+), the linear `skill x temp`
interaction is +0.031 per 10 °F [+0.004, +0.059], it survives event fixed
effects, leave-one-event-out, a season-wave horse race and every control
except `skill x qualifier` (which shrinks it 30%), and it strengthens as the
venue labels get cleaner (+0.076 on high-confidence labels only). It fails
Holm against its own family, is absent in 2026, and its indoor placebo arm
sits at −0.022 rather than the 0 physics demands — so half of the +0.054 DiD
comes from a control arm that should not be moving at all. Suggestive, not
established.

**If I got exactly one shot**: pre-register the FULL-RANGE temperature
interaction, not gusts. Exact spec, fixed in advance:
`share − ½ ~ skill + sust + temp + night + hour + qualifier + skill·|skill|
+ skill×(each of those)`, event fixed effects, cluster bootstrap over events,
outdoor arm with corrected labels, indoor arm as the falsification, and the
DiD `out × skill × temp` as the single reported number. Direction
pre-committed positive (favourites convert skill better as it warms).
Declare a hit only if the outdoor coefficient AND the DiD both clear zero
and the indoor arm is inside ±0.015. Gustiness is the runner-up on physics
but it has already spent its power here: the estimate is half its MDE, the
binned shape is confounded, and unlike temperature nothing about it
coheres across specifications.
