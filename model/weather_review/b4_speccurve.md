# B4 (3-4) — specification curves


# Part 3 — specification curve, H4 outdoor skill×wind interaction

Model per spec (OUTDOOR games only): `y = a + b·skill + c·(wind/10) + d·skill·(wind/10)`, skill = v2-expected point share − ½. **d** is the compression coefficient (negative = wind flattens the favourite). Because the three outcomes live on different scales, the scale-free curve statistic is **r = d/b**, the fraction of the favourite's skill edge erased per +10 mph; the raw-d curve is also given for the 216 point-share specs, directly comparable to the published +0.002 [−0.060, +0.064].

Grid: 864 estimable specifications.

PUBLISHED spec reproduces at d = +0.0017 [-0.0595, +0.0629] (CR1), vs published +0.002 [−0.060, +0.064] (event bootstrap) — n=24819, 67 events.

**FULL CURVE — r = d/b (fraction of favourite's edge erased per +10 mph)** — 864 specs

| median | IQR | min | max | frac CI excludes 0 | frac negative (compression) |
|---|---|---|---|---|---|
| -0.0059 | [-0.0285, +0.0131] | -0.1211 | +0.1253 | 59/864 (7%) | 499/864 (58%) |

PUBLISHED spec = +0.0016 → percentile 59 of its own curve (middle; 0 = most negative/compression end, 100 = most positive).

**POINT-SHARE outcome only — raw d** — 288 specs

| median | IQR | min | max | frac CI excludes 0 | frac negative (compression) |
|---|---|---|---|---|---|
| -0.0025 | [-0.0231, +0.0141] | -0.0801 | +0.0607 | 15/288 (5%) | 151/288 (52%) |

PUBLISHED spec = +0.0017 → percentile 54 of its own curve (middle; 0 = most negative/compression end, 100 = most positive).

## Curve by analyst choice (median r, and how many of that slice's CIs exclude zero)

| dimension | level | specs | median r | IQR | CI≠0 |
|---|---|---|---|---|---|
| timing | daily_max | 216 | +0.0039 | [-0.0187, +0.0136] | 0/216 |
| timing | hour_actual | 216 | -0.0124 | [-0.0318, +0.0090] | 7/216 |
| timing | hour_planned | 216 | -0.0003 | [-0.0209, +0.0183] | 5/216 |
| timing | hour_pub | 216 | -0.0173 | [-0.0377, +0.0047] | 47/216 |
| metric | gust | 432 | -0.0008 | [-0.0157, +0.0142] | 26/432 |
| metric | sustained | 432 | -0.0183 | [-0.0424, +0.0104] | 33/432 |
| sample | all | 288 | -0.0090 | [-0.0329, +0.0135] | 29/288 |
| sample | g1_only | 288 | -0.0095 | [-0.0297, +0.0092] | 13/288 |
| sample | no_deciders | 288 | -0.0029 | [-0.0234, +0.0181] | 17/288 |
| labels | audited_hi | 216 | +0.0199 | [-0.0010, +0.0417] | 14/216 |
| labels | corrected_all | 216 | -0.0124 | [-0.0234, -0.0021] | 0/216 |
| labels | corrected_hi | 216 | -0.0366 | [-0.0637, -0.0240] | 45/216 |
| labels | published | 216 | +0.0088 | [-0.0016, +0.0166] | 0/216 |
| outcome | margin | 288 | -0.0034 | [-0.0270, +0.0167] | 31/288 |
| outcome | share | 288 | -0.0024 | [-0.0221, +0.0139] | 13/288 |
| outcome | win | 288 | -0.0125 | [-0.0341, +0.0071] | 15/288 |
| fe | event | 288 | -0.0071 | [-0.0287, +0.0119] | 19/288 |
| fe | none | 288 | -0.0045 | [-0.0270, +0.0139] | 20/288 |
| fe | tour | 288 | -0.0055 | [-0.0290, +0.0135] | 20/288 |

## The composition question — event fixed effects

- **FE = none**: median r -0.0045, IQR [-0.0270, +0.0139], range [-0.1113, +0.1058], median CI width 0.1193, 20/288 CIs exclude zero.
- **FE = tour**: median r -0.0055, IQR [-0.0290, +0.0135], range [-0.1149, +0.1077], median CI width 0.1198, 20/288 CIs exclude zero.
- **FE = event**: median r -0.0071, IQR [-0.0287, +0.0119], range [-0.1211, +0.1253], median CI width 0.1218, 19/288 CIs exclude zero.

### Extreme specs (the ones that would make a headline)

| r | 95% CI | timing | metric | sample | labels | outcome | FE | n |
|---|---|---|---|---|---|---|---|---|
| -0.1211 | [-0.1974, -0.0449] | hour_pub | sustained | g1_only | corrected_hi | margin | event | 9482 |
| -0.1149 | [-0.1916, -0.0382] | hour_pub | sustained | g1_only | corrected_hi | margin | tour | 9482 |
| -0.1113 | [-0.1872, -0.0354] | hour_pub | sustained | g1_only | corrected_hi | margin | none | 9482 |
| -0.1095 | [-0.1870, -0.0321] | hour_pub | sustained | all | corrected_hi | win | tour | 19421 |
| +0.1051 | [-0.0462, +0.2564] | hour_actual | sustained | no_deciders | audited_hi | win | event | 4602 |
| +0.1058 | [-0.0453, +0.2569] | hour_actual | sustained | all | audited_hi | win | none | 5160 |
| +0.1077 | [-0.0464, +0.2618] | hour_actual | sustained | all | audited_hi | win | tour | 5160 |
| +0.1253 | [-0.0246, +0.2752] | hour_actual | sustained | all | audited_hi | win | event | 5160 |

### CR1 vs event cluster bootstrap (published spec, 2,000 reps)

bootstrap d CI [-0.0588, +0.0638] vs CR1 [-0.0595, +0.0629] — the analytic intervals used across the grid are the honest twin of the published bootstrap.


# Part 4 — specification curve, H1 serve-point rate vs wind (outdoor)

Model per spec: `serve_rate = a + s·(wind/10)` on OUTDOOR matches with rally logs; **s** is the slope per +10 mph (published +0.0030 [−0.0009, +0.0072] at match hour, +0.0017 daily). serve_rate = n_points / n_rallies.

Grid: 384 specifications. PUBLISHED spec reproduces at s = +0.0030 [-0.0015, +0.0074] (CR1) vs published +0.0030 [−0.0009, +0.0072].

| median | IQR | min | max | frac CI excludes 0 | published pctile |
|---|---|---|---|---|---|
| +0.0004 | [-0.0014, +0.0012] | -0.0079 | +0.0053 | 21/384 (5%) | 89 |

| dimension | level | specs | median s | IQR | CI≠0 |
|---|---|---|---|---|---|
| timing | daily_max | 96 | -0.0000 | [-0.0023, +0.0005] | 8/96 |
| timing | hour_actual | 96 | +0.0006 | [-0.0006, +0.0023] | 0/96 |
| timing | hour_planned | 96 | +0.0006 | [-0.0020, +0.0015] | 8/96 |
| timing | hour_pub | 96 | +0.0007 | [-0.0007, +0.0013] | 5/96 |
| metric | gust | 192 | +0.0000 | [-0.0014, +0.0007] | 21/192 |
| metric | sustained | 192 | +0.0010 | [-0.0015, +0.0028] | 0/192 |
| minr | 20 | 192 | +0.0003 | [-0.0011, +0.0011] | 9/192 |
| minr | 40 | 192 | +0.0004 | [-0.0019, +0.0013] | 12/192 |
| weight | equal | 192 | +0.0004 | [-0.0014, +0.0014] | 9/192 |
| weight | rallies | 192 | +0.0004 | [-0.0014, +0.0010] | 12/192 |
| labels | audited_hi | 96 | -0.0033 | [-0.0044, -0.0023] | 21/96 |
| labels | corrected_all | 96 | +0.0005 | [-0.0000, +0.0023] | 0/96 |
| labels | corrected_hi | 96 | +0.0010 | [+0.0001, +0.0021] | 0/96 |
| labels | published | 96 | +0.0007 | [-0.0005, +0.0025] | 0/96 |
| fe | event | 128 | -0.0007 | [-0.0015, +0.0001] | 2/128 |
| fe | none | 128 | +0.0008 | [-0.0003, +0.0027] | 8/128 |
| fe | tour | 128 | +0.0008 | [-0.0001, +0.0025] | 11/128 |

| s | 95% CI | timing | metric | min rallies | weight | labels | FE | n |
|---|---|---|---|---|---|---|---|---|
| -0.0079 | [-0.0170, +0.0013] | daily_max | sustained | 40 | equal | audited_hi | event | 2154 |
| -0.0078 | [-0.0159, +0.0002] | daily_max | sustained | 40 | equal | audited_hi | tour | 2154 |
| -0.0075 | [-0.0177, +0.0028] | daily_max | sustained | 40 | equal | audited_hi | none | 2154 |
| +0.0049 | [-0.0002, +0.0101] | hour_planned | sustained | 40 | equal | corrected_hi | tour | 6299 |
| +0.0051 | [-0.0015, +0.0117] | hour_planned | sustained | 20 | equal | corrected_hi | tour | 6793 |
| +0.0053 | [-0.0013, +0.0118] | hour_planned | sustained | 20 | equal | corrected_hi | none | 6793 |
