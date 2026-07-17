# Momentum pre-registration — RESULTS (run once)

Sample: 2500 matches (0 dropped), 191,500 resolved rallies; timeout events recv/srv 6417/512; game-point rallies 10,651; possession-3rd+ rallies 31,129.

| estimand | APE | 99% CI | z | p |
|---|---|---|---|---|
| H1 phi_poss (poss2p) | +0.11pp | [-0.72, +0.94] | +0.34 | 0.74 |
| H2 phi5 (trail5c +0.2) | +0.17pp | [-0.19, +0.52] | +1.19 | 0.23 |
| H2s phi1 (lag1_A) | -0.18pp | [-1.01, +0.64] | -0.57 | 0.57 |
| H3 tau_recv | +1.53pp | [-0.07, +3.12] | +2.46 | 0.014 |
| H3s tau_srv | -3.24pp | [-8.74, +2.26] | -1.52 | 0.13 |
| H4 beta_gp (gp_srv) | -0.01pp | [-1.31, +1.29] | -0.02 | 0.99 |
| H4s beta_gp_recv | -0.30pp | [-1.80, +1.20] | -0.51 | 0.61 |

## Verdicts (per the frozen criteria)

- H1 within-possession momentum: **no meaningful effect (bounded)** — 99% CI inside ±1pp.
- H2 cross-possession momentum: **no meaningful effect (bounded)**.
- H3 receiving-side timeout: **no effect established; formally inconclusive** (CI reaches +3.1pp) — direction mildly AGAINST the caller.
- H4 game-point pressure: **no meaningful effect (bounded)** — servers convert game points at exactly their usual rate.

Registered predictions: 4/4 HIT, mean Brier 0.044. Descriptive bonus: timeouts
are called by the receiving side 12.5× as often as the serving side (6,417 vs
512) — momentum-stopping attempts by the team being scored on, with no
detectable next-rally payoff.
