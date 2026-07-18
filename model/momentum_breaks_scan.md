# Break-type momentum scan — EXPLORATION (sample A, 2,500 matches)

Not a pre-registration. All cells reported with denominators; nothing here is an asserted finding without a pre-registered shot on the untouched pool. ~24 break estimands below — under the null expect ~1 in 20 to cross p<.05 by chance.

## T1 — Icing: adjusted effect of a break on the receiver's next-W-rally share (matched moments)

| break | W | theta | 99% CI | z | p | n_treat | n_ctrl | naive Δ |
|---|---|---|---|---|---|---|---|---|
| timeout | 3 | -0.41pp | [-1.21, +0.40] | -1.30 | 0.19 | 6961 | 158290 | +37.7pp |
| switch | 3 | -2.82pp | [-4.77, -0.86] | -3.71 | 0.00021 | 1149 | 158290 | +33.4pp |
| challenge | 3 | +0.38pp | [-2.32, +3.07] | +0.36 | 0.72 | 545 | 158290 | +24.3pp |
| timeout | 5 | -0.32pp | [-0.92, +0.28] | -1.36 | 0.17 | 6318 | 138720 | +27.2pp |
| switch | 5 | -3.16pp | [-4.60, -1.72] | -5.66 | 1.5e-08 | 1147 | 138720 | +23.2pp |
| challenge | 5 | +0.55pp | [-1.63, +2.74] | +0.65 | 0.51 | 485 | 138720 | +14.6pp |
| timeout | 10 | -0.11pp | [-0.57, +0.36] | -0.59 | 0.55 | 4251 | 90745 | +15.0pp |
| switch | 10 | -1.51pp | [-2.50, -0.52] | -3.93 | 8.3e-05 | 967 | 90745 | +14.9pp |
| challenge | 10 | +0.34pp | [-1.42, +2.11] | +0.50 | 0.62 | 305 | 90745 | +7.2pp |

## T2 — Fragility: adjusted lag-1 rally autocorrelation, by whether a break separates the two rallies

| pair type | lag-1 APE | 99% CI | z | p | n_pairs |
|---|---|---|---|---|---|
| no_break | -0.41pp | [-1.28, +0.45] | -1.23 | 0.22 | 177857 |
| after_any_break | +0.56pp | [-5.26, +6.37] | +0.25 | 0.8 | 8576 |
| after_timeout | +1.80pp | [-5.05, +8.65] | +0.68 | 0.5 | 7600 |
| after_switch | -1.44pp | [-57.61, +54.73] | -0.07 | 0.95 | 1151 |
| after_challenge | -8.06pp | [-19.46, +3.33] | -1.82 | 0.068 | 624 |

## Follow-up: is the switchover ping real or confounded? (still exploration, sample A)

The 6-point switchover was the one non-null: receiver's next-5-rally share −3.16pp (z=−5.66). Three confound attacks, all survived:

1. **Cumulative dominance** (adding both sides' absolute scores as controls): −3.16 → −3.07pp. Unmoved. Not "the team that raced to 6 keeps rolling."
2. **Score-6 position** (natural control — moments at score 6 with NO logged switch, since switch-logging is spotty): pure switch −2.87pp vs at-6-no-switch −0.48pp. The switch is ~6× the mere-position effect.
3. **Cross-event coverage** (restrict to the 1,097 matches that log ≥1 switch, compare switch moments to baseline from the same matches): −3.33pp (z=−3.90). Not a between-event artifact.

**Key internal contrast**: of the three break types, only the switchover involves an END CHANGE — and it's the only one with an effect (timeout ~0, challenge ~0). This points at the physical end-swap, not "a break" per se, disadvantaging the RECEIVER (harder task: reading serve/third-shot from a freshly swapped end) for a few rallies, fading by W=10.

**Status**: exploratory signal that survived the obvious confounds — NOT an asserted finding. Earns one pre-registered confirmatory shot on the untouched pool. Likely a mechanical end-change micro-effect (win-prob-relevant), NOT psychological momentum. Remaining caveats: switch-moment n is modest (475 clean); "≥W rallies after" selection at score 6 not fully excluded; pool replication tests noise-vs-real but a shared-mechanism artifact would survive it too.
