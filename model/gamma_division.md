# Weakest-link gamma by division

Games with month-of-game v2 values for all four players (349 skipped for missing values):
  mens: 13928
  womens: 7459
  mixed: 15562

| division | games | gamma (MLE) | 95% CI (match bootstrap) | better wt | worse wt |
|---|---|---|---|---|---|
| pooled | 36949 | -0.120 | [-0.150, -0.093] | 0.440 | 0.560 |
| mens | 13928 | -0.066 | [-0.117, -0.013] | 0.467 | 0.533 |
| womens | 7459 | -0.155 | [-0.226, -0.086] | 0.422 | 0.578 |
| mixed | 15562 | -0.143 | [-0.183, -0.103] | 0.429 | 0.571 |

Pairwise differences (bootstrap of the difference):

| contrast | Δgamma | 95% CI |
|---|---|---|
| mens − womens | +0.090 | [+0.007, +0.174] |
| mens − mixed | +0.077 | [+0.009, +0.146] |
| womens − mixed | -0.013 | [-0.094, +0.069] |

*Values fixed at v2 month-of-game means (fitted with the pooled gamma), so divisions' values have partially adapted to the shared gamma — this is a conditional test. Match-cluster bootstrap (n=1000) absorbs the match random effect. Per-division |gap| spread differs (mixed pairs are wider), which is the power driver.*

| division | mean team |gap| | sd of Δgap (identifying spread) |
|---|---|---|
| mens | 0.170 | 0.200 |
| womens | 0.181 | 0.220 |
| mixed | 0.205 | 0.236 |

---

**SUPERSEDED (same day): the joint refit (`fit_v2_gamma_div.py` →
`gamma_division_refit.md`) reverses this table's ordering — mens −0.280,
womens −0.204, mixed −0.091, with mens−mixed the one credible contrast
(P(Δ>0)=0.008). The conditional-on-pooled-values circularity warned about
above is material. Quote the refit, not this file.**
