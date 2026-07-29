# B2a — corrected-venue-label re-run (H1 serve rate, H4 favorites)

rows: 12885 matches (H1), 36518 games (H4a), 3826 games (H4b); all with a match-hour wind join.

## Label arms — game counts (H4a sample)

| arm | outdoor games | indoor games | dropped | outdoor events | indoor events |
|---|---|---|---|---|---|
| a | 24819 | 11699 | 0 | 67 | 43 |
| b | 26451 | 8785 | 1282 | 72 | 35 |
| c | 24718 | 7127 | 4673 | 69 | 30 |
| d | 27949 | 7127 | 1442 | 76 | 30 |

## What the correction is confounded with

Flip class = heuristic label -> verified label (arm c mapping), game-weighted over the H4a sample.

| heuristic -> verified | games | %MLP | mean match-hour wind | mean temp |
|---|---|---|---|---|
| outdoor -> (unaudited) | 13283 | 0% | 6.2 mph | 73 °F |
| indoor -> outdoor | 5989 | 10% | 7.5 mph | 73 °F |
| outdoor -> outdoor | 5446 | 10% | 7.1 mph | 70 °F |
| indoor -> indoor | 3747 | 24% | 8.0 mph | 66 °F |
| outdoor -> indoor | 3380 | 0% | 7.0 mph | 72 °F |
| indoor -> mixed | 1963 | 6% | 8.1 mph | 76 °F |
| outdoor -> unknown | 1442 | 0% | 6.4 mph | 75 °F |
| outdoor -> mixed | 1268 | 0% | 7.0 mph | 67 °F |

| arm | pool | games | %MLP | mean wind | P(wind>=14) | mean temp |
|---|---|---|---|---|---|---|
| a | outdoor | 24819 | 2% | 6.5 mph | 5.0% | 72 °F |
| a | indoor | 11699 | 14% | 7.7 mph | 6.8% | 71 °F |
| b | outdoor | 26451 | 3% | 6.6 mph | 4.6% | 73 °F |
| b | indoor | 8785 | 14% | 7.9 mph | 8.2% | 69 °F |
| c | outdoor | 24718 | 5% | 6.7 mph | 4.6% | 72 °F |
| c | indoor | 7127 | 13% | 7.5 mph | 10.0% | 69 °F |
| d | outdoor | 27949 | 5% | 6.8 mph | 4.4% | 72 °F |
| d | indoor | 7127 | 13% | 7.5 mph | 10.0% | 69 °F |

## H1. Serve-point rate vs match-hour wind (WLS slope per +10 mph)

| arm | outdoor slope [95% CI] | n_out | indoor slope [95% CI] | n_in | OUTDOOR − INDOOR [95% CI] |
|---|---|---|---|---|---|
| a | +0.0030 [-0.0013, +0.0076] | 7905 | +0.0018 [-0.0021, +0.0073] | 4980 | +0.0011 [-0.0059, +0.0075] |
| b | +0.0026 [-0.0012, +0.0072] | 8955 | +0.0010 [-0.0029, +0.0070] | 3417 | +0.0016 [-0.0053, +0.0079] |
| c | +0.0027 [-0.0014, +0.0072] | 8738 | +0.0019 [-0.0022, +0.0099] | 2666 | +0.0007 [-0.0075, +0.0067] |
| d | +0.0030 [-0.0010, +0.0077] | 9868 | +0.0019 [-0.0023, +0.0095] | 2666 | +0.0011 [-0.0074, +0.0074] |

## H4a. Game level: share−½ = a + b·skill + c·w + d·skill·w  (d)

| arm | outdoor d [95% CI] | b_out | indoor d [95% CI] | b_in | OUTDOOR − INDOOR d [95% CI] |
|---|---|---|---|---|---|
| a | +0.002 [-0.061, +0.069] | 1.040 | -0.080 [-0.146, +0.017] | 1.108 | +0.082 [-0.034, +0.172] |
| b | -0.018 [-0.081, +0.038] | 1.045 | -0.050 [-0.131, +0.076] | 1.113 | +0.032 [-0.112, +0.131] |
| c | -0.038 [-0.096, +0.017] | 1.051 | -0.038 [-0.129, +0.116] | 1.117 | +0.000 [-0.168, +0.107] |
| d | -0.042 [-0.095, +0.008] | 1.059 | -0.038 [-0.129, +0.114] | 1.117 | -0.004 [-0.166, +0.102] |

## H4b. Rally level: favourite−underdog serve-rate gap slope per +10 mph

| arm | outdoor c [95% CI] | n_out | indoor c [95% CI] | n_in | OUTDOOR − INDOOR [95% CI] |
|---|---|---|---|---|---|
| a | -0.0215 [-0.0414, -0.0012] | 1922 | -0.0311 [-0.0526, -0.0101] | 1904 | +0.0096 [-0.0205, +0.0388] |
| b | -0.0191 [-0.0379, -0.0007] | 2297 | -0.0275 [-0.0513, -0.0032] | 1359 | +0.0084 [-0.0230, +0.0372] |
| c | -0.0211 [-0.0390, -0.0027] | 2448 | -0.0249 [-0.0510, +0.0065] | 1010 | +0.0038 [-0.0297, +0.0351] |
| d | -0.0248 [-0.0426, -0.0064] | 2737 | -0.0249 [-0.0505, +0.0047] | 1010 | +0.0002 [-0.0334, +0.0325] |

## H4c (context). Favourite obs−pred edge by match-hour wind bin

| arm | setting | 0–8 | 8–14 | 14–20 | 20+ |
|---|---|---|---|---|---|
| a | outdoor | +0.0144 (17139) | +0.0165 (6447) | +0.0104 (1151) | +0.0057 (82) |
| a | indoor | +0.0159 (6847) | +0.0140 (4062) | +0.0086 (568) | -0.0247 (222) |
| b | outdoor | +0.0142 (18615) | +0.0148 (6626) | +0.0092 (1128) | +0.0057 (82) |
| b | indoor | +0.0162 (4544) | +0.0176 (3525) | +0.0175 (517) | -0.0265 (199) |
| c | outdoor | +0.0147 (17083) | +0.0127 (6502) | +0.0071 (1051) | +0.0057 (82) |
| c | indoor | +0.0140 (4145) | +0.0228 (2269) | +0.0183 (514) | -0.0265 (199) |
| d | outdoor | +0.0156 (18786) | +0.0130 (7930) | +0.0040 (1128) | +0.0024 (105) |
| d | indoor | +0.0140 (4145) | +0.0228 (2269) | +0.0183 (514) | -0.0265 (199) |
