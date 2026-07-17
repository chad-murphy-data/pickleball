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

## ADDENDUM H3b — windowed timeout effect (single frozen run)

H3b W=10 (primary): theta +0.060 pts (se 0.037, 99% CI [-0.034, +0.155], z=+1.64, p=0.1) | n_treat 3400, n_ctrl 91957 | trigger profile: mean prew -2.61 (treat) vs -0.25 (ctrl) | NAIVE after-minus-before +2.30

H3b W=5 (secondary): theta +0.044 pts (se 0.024, 99% CI [-0.017, +0.105], z=+1.87, p=0.062) | n_treat 5213, n_ctrl 140273 | trigger profile: mean prew -2.48 (treat) vs -0.23 (ctrl) | NAIVE after-minus-before +2.09

## ADDENDUM H3c — rally-share steelman (single frozen run; FINAL timeout test on this sample)

H3c PRIMARY share10: theta +0.20pp (99% CI [-0.30, +0.71], z=+1.02, p=0.31) | n_treat 3400, n_ctrl 91957 | naive Δshare +16.4pp
S1 share3: theta +0.86pp (99% CI [-0.24, +1.96], z=+2.01, p=0.045) | n_treat 3400, n_ctrl 91957 | naive Δshare +20.5pp
S2 possession-killer: theta -0.05pp (99% CI [-2.19, +2.08], z=-0.06, p=0.95) | n_treat 3400, n_ctrl 91957 | naive Δshare +12.2pp
S3 share10 | severe trigger (prew_pts <= -4): theta -0.11pp (99% CI [-1.03, +0.80], z=-0.31, p=0.75) | n_treat 1100, n_ctrl 9519 | naive Δshare +25.5pp
