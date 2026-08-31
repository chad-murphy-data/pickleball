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

## ADDENDUM H3d (amended): window-profile confirmation on the untouched pool (single frozen run)

moments: 12782 treated / 348811 control
| W | theta_W | 99% CI | z | test |
|---|---|---|---|---|
| 1 | +0.42pp | [-0.73, +1.57] | +0.94 | family: ns (|z|>=2.69) |
| 2 | +0.14pp | [-0.60, +0.88] | +0.49 | family: ns (|z|>=2.69) |
| 3 | +0.26pp | [-0.29, +0.82] | +1.21 | PRIMARY one-sided p=0.113 (alpha .01) |
| 4 | +0.11pp | [-0.35, +0.56] | +0.60 | family: ns (|z|>=2.69) |
| 5 | +0.01pp | [-0.38, +0.39] | +0.05 | family: ns (|z|>=2.69) |
| 7 | +0.09pp | [-0.23, +0.41] | +0.73 | family: ns (|z|>=2.69) |
| 10 | +0.13pp | [-0.13, +0.39] | +1.27 | family: ns (|z|>=2.69) |

All seven 99% CIs inside ±2.5pp: YES — the 1–10 rally window family is BOUNDED

### H3d verdict (per frozen criteria)

- PRIMARY (window 3): θ₃ = +0.26pp, one-sided p = .113 — **FAILS to replicate** at α=.01. Sample A's +0.86pp did not survive on 4× the data.
- FAMILY (windows 1,2,4,5,7,10): no window clears Bonferroni |z|≥2.69 (max |z| = 1.27 at W=10) — **no discovery**.
- BOUNDING: all seven 99% CIs inside ±2.5pp — **the 1–10 rally window family is bounded**. No window from 1 to 10 rallies hides a meaningful timeout effect.
- SHAPE: flat/incoherent (+0.42, +0.14, +0.26, +0.11, +0.01, +0.09, +0.13), not the decaying profile a real short-window huddle effect would produce. This is the noise branch.

The 3-rally hint was noise. Both registered predictions HIT. Momentum program: 8/8 pre-registered predictions correct, mean Brier ≈ 0.048.
