# Rare wind skill — injection power test

Real panel (552 players), trait planted in 13% of players with slope ±s (random sign, persistent across eras), 20 sims per size. Detection = spike-slab LR > 7.2 (wind_rare null max) or STV K=40 z > 2.5 (either direction).

| s (share/10mph) | LR mean | P(LR fires) | STV z mean | P(STV fires) | P(either) |
|---|---|---|---|---|---|
| 0.000 | 6.0 | 0.00 | +1.39 | 0.00 | 0.00 |
| 0.010 | 6.1 | 0.10 | +1.44 | 0.00 | 0.10 |
| 0.020 | 8.6 | 0.75 | +1.63 | 0.05 | 0.75 |
| 0.030 | 18.7 | 1.00 | +2.10 | 0.30 | 1.00 |
| 0.040 | 25.3 | 1.00 | +2.33 | 0.45 | 1.00 |

*Sign persistent across eras = a genuine trait; random sign mirrors the two-sided tail structure. s = 0 row is the false-positive check. Null replicates are the real-data permutations (injection slightly widens injected players' true se, so detection rates are, if anything, optimistic — fine for an upper bound on the telescope).*
