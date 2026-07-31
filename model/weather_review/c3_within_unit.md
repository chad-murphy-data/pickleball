# C3 — within-unit identification of the wind effect, and wind skill as a variance component

Scripts (all new, none of the committed weather scripts touched):

* `model/weather_review/c3_lib.py` — frame + sparse fixed-effect absorber
* `model/weather_review/c3a_fixed_effects.py` → `c3a_fixed_effects.json`
* `model/weather_review/c3a_gap.py` — paired audited/unaudited contrast
* `model/weather_review/c3a_nonlinear.py` → `c3a_nonlinear.txt`
* `model/weather_review/c3b_wind_skill.py` → `c3b_wind_skill.json`

Sample: 36,518 games with a match-hour wind join (73% actual start times);
corrected venue labels, arm c (web-verified high/medium; mixed+unknown
dropped) → 24,718 outdoor games / 69 events, 7,127 indoor / 30 events.
Inference: cluster bootstrap over EVENTS (1,000 draws; 400 for the heavy
FE specs, 200 for L6/L7), with event-nested fixed effects re-keyed per
drawn copy. All RNGs seeded.

---

## Pre-specification (written before reading the results)

**A. Fixed effects.** The estimand is `d` in the published spec
`share − ½ = a + b·skill + c·w + d·skill·w` (w = match-hour wind/10 mph,
skill = v2 expected share − ½). Wind compresses the favourite ⇒ `d < 0`.

*What would have counted as a signal:* `d` stays in the same place — same
sign, point estimate inside the pooled CI — as the identification is
tightened from pooled → event → player → pair → pair×event, with each CI
excluding zero as far as power allows, and the indoor placebo staying at
zero at every level. *What would have counted as "composition drove it":*
`d` collapsing toward zero or reversing as fixed effects are added, and
the audited-vs-unaudited gap shrinking under within-unit identification.

**B. Wind skill.** The estimand is `tau1`, the sd across players of a
personal wind slope, in point share per 10 mph. *Signal:* `tau1`
significantly > 0 outdoors (LR vs 0 beyond the simulated null band),
**and** ≈ 0 in the indoor placebo, where ERA5 wind cannot touch the
court. *Null:* an upper bound on `tau1` below the size that would matter.
For style, signal = a nonzero style×wind interaction outdoors with an
indoor placebo at zero.

---

## A. The interaction under eight identification schemes (outdoor, arm c)

`d` per +10 mph; translation = change in win probability at 15 mph for a
favourite v2 prices at .600 expected share (83.6% to win a game to 11).
"keep" = sd of the interaction regressor surviving the FE projection —
the honest cost of each design.

| spec | fixed effects | d | 95% CI (event boot) | keep | Δ win prob @15 mph |
|---|---|---|---|---|---|
| L0 | none (published spec) | −0.038 | [−0.092, +0.019] | 100% | −1.4pp [−3.6, +0.7] |
| L1 | event | −0.040 | [−0.094, +0.019] | 96% | −1.5pp [−3.7, +0.7] |
| L2 | event + event×skill | −0.050 | [−0.119, +0.013] | 39% | −1.9pp [−4.8, +0.5] |
| L3 | player (antisymmetric) | **−0.065** | **[−0.108, −0.009]** | 48% | −2.5pp [−4.3, −0.4] |
| L4 | pair / dyad | −0.004 | [−0.079, +0.067] | 35% | −0.1pp [−3.1, +2.3] |
| L5 | pair × event | +0.056 | [−0.165, +0.241] | 10% | +2.0pp [−6.8, +7.4] |
| L6 | player + event×skill | **−0.078** | **[−0.138, −0.003]** | 34% | −3.0pp [−5.6, −0.1] |
| L7 | pair×event + event×skill | +0.076 | [−0.181, +0.269] | 9% | +2.7pp [−7.5, +8.1] |

Indoor placebo, same ladder: −0.038, −0.036, +0.065, −0.004, −0.070,
−0.255, −0.008, −0.178 — **every** CI spans zero, and the tight designs
swing by ±0.25, which is the best available read on how noisy L5/L7 are.

Identifying variation that survives each restriction:

| spec | units | units with within-unit wind sd ≥ 2 mph | share of observations they carry |
|---|---|---|---|
| L1/L2 | 69 events | 43 | 65% of games |
| L3/L6 | 2,748 players | 1,236 | 91% of player-appearances |
| L4 | 6,507 pairs | 1,591 | 54% of pair-appearances |
| L5/L7 | 10,347 pair×event cells | 1,644 | 23% of pair-appearances |

**Read.** The estimate is *not* stable. It runs from −0.078 to +0.076
across the ladder, and the two specifications whose CI excludes zero
(player FE, player FE + event×skill) are the ones that are *not* the
tightest identification — pair and pair×event FE, which use strictly
within-pair contrasts, give ≈0 and +0.06. The strongest design the data
support ("same pair, same event, calm vs windy hours") keeps 10% of the
identifying variation and returns +0.056 [−0.165, +0.241]: it can neither
confirm nor exclude compression up to ±7pp of win probability, so it is
uninformative rather than confirmatory. Where the data ARE well powered
(L0–L3, MDE ≈ 0.07–0.09 in d), the answer is a small negative point
estimate whose interval mostly excludes compression larger than ~5pp of
win probability at 15 mph for a .600 favourite.

### The B2a diagnostic, done as a paired contrast (`c3a_gap.py`)

One regression on the outdoor pool with everything interacted with an
`unaudited` dummy, bootstrapped over events, so the gap has its own CI:

| spec | d (audited events) | gap (unaudited − audited) | implied d (unaudited) |
|---|---|---|---|
| L0 pooled | +0.059 [−0.006, +0.108] | **−0.170 [−0.254, −0.068]** | −0.111 |
| L1 event FE | +0.064 [+0.000, +0.112] | **−0.180 [−0.263, −0.074]** | −0.116 |
| L2 event FE + event×skill | +0.013 [−0.080, +0.090] | **−0.134 [−0.246, −0.010]** | −0.121 |
| L3 player FE | −0.047 [−0.110, +0.012] | −0.042 [−0.139, +0.076] | −0.089 |

**This is the answer to B2a.** The gap is real (z ≈ 4 pooled) and it is
*not* between-event composition: putting event and event×skill fixed
effects in leaves it at −0.134, still excluding zero. It is **player**
composition — comparing every player with himself shrinks the gap to
−0.042 with a CI that spans zero, and the two event sets then agree on a
mildly negative d. The published pooled numbers were comparing different
people, and the "audited vs unaudited" split was a proxy for which people.

### Is the one significant spec an artefact? (`c3a_nonlinear.py`)

The mechanical route to d < 0 with no wind effect is a compressive
nonlinearity in the skill response combined with windy hours holding
different skill gaps. The composition premise is weak — mean |skill| by
wind bin outdoors is 0.120 (0–8 mph) / 0.119 (8–14) / 0.124 (14+) — but
the test still matters:

| spec | linear | + skill³, skill³·w | + skill⁵, skill⁵·w |
|---|---|---|---|
| L0 | −0.038 [−0.095, +0.016] | −0.054 [−0.134, +0.039] | −0.065 [−0.184, +0.048] |
| L2 | −0.050 [−0.120, +0.016] | −0.071 [−0.158, +0.018] | −0.089 [−0.208, +0.028] |
| L3 | −0.065 [−0.109, −0.014] | −0.067 [−0.146, +0.026] | −0.082 [−0.215, +0.049] |

The point estimate is stable-to-more-negative under a flexible skill
response, but the interval opens up and the player-FE significance
disappears. So: not an artefact, but not robust either — the only
CI-excludes-zero cell in the whole ladder is one functional-form choice
away from spanning zero.

---

## B. Wind skill as a variance component (`c3b_wind_skill.py`)

Model (match level — every regressor and every player effect is constant
within a match, so collapsing games to matches keeps all the information
about `tau1` and discards the within-match replication a game-level fit
would mistake for player structure):

```
y_m = a + b·skill + c·w + d·skill·w
      + Σ_{T1}(u0_i + u1_i·wc_m) − Σ_{T2}(u0_j + u1_j·wc_m) + ε_m
u0 ~ N(0, tau0²)   u1 ~ N(0, tau1²)   ε_m ~ N(0, φ_bin · v_m)
```

`v_m` = the exact iid race-to-T variance of the match point share, so
close-game/blowout heteroskedasticity is not read as player structure;
`φ` is free per wind bin (0–8 / 8–14 / 14+ mph) so wind-driven *noise*
cannot be read as wind *skill*. ML by Woodbury; interval by profile
likelihood; `wind_mode = within_event` re-defines the slope covariate as
the deviation from the event's own mean wind, so a player's event-to-event
form cannot masquerade as a wind slope. Players with ≥20 matches in the
pool get random effects (526 outdoor, 223 indoor) — the same threshold in
matches that `model/wind_skill.py` used in games.

| arm | tau1 (share per 10 mph) | 95% profile CI | LR vs 0 |
|---|---|---|---|
| **outdoor, global wind** | 0.0127 | [0.0055, 0.0180] | 6.52 |
| **outdoor, within-event wind** | 0.0114 | [—, 0.0197] | 2.24 |
| **indoor placebo, global** | 0.0154 | [0.0030, 0.0236] | 4.24 |
| **indoor placebo, within-event** | 0.0147 | [—, 0.0304] | 0.58 |

Calibration by parametric bootstrap on the real design (6 sims per cell):
under `tau1 = 0` the estimator returns 0.000–0.013 with LR ≤ 2.8; under
`tau1 = 0.020` it returns 0.014–0.020 with LR 4.9–38.1 (12/12 above 3.84).
So the design detects a 0.02 wind-skill dimension essentially always, and
the MDE is ≈ 0.015–0.020.

**Read.** Outdoors the variance component is nominally nonzero (LR 6.5,
p ≈ 0.005 on the 50:50 boundary mixture) — but the indoor placebo, where
wind cannot reach the court, returns the *same* number on a third of the
data (0.0154 vs 0.0127). Whatever the estimator is picking up is generic
player heterogeneity that projects onto the wind variable (time of day,
day of the weekend, venue, season), not wind. That is exactly the
falsification pattern the pre-specification called for, and it fails.

**The publishable claim** (what the split-half design could never say):

> Across 12,390 outdoor matches and 526 players, the dispersion of
> personal wind slopes is at most **0.018 point share per 10 mph** (95%
> profile upper bound). A player one sd above average on this dimension
> is worth at most **+1.5pp of his team's point share at 15 mph**; a pair
> of two such players beats an even-matched pair **61.5% instead of 50%**
> at 15 mph, and that ceiling includes whatever the indoor placebo is
> measuring, so the true wind-specific part is smaller still.

For comparison, the committed split-half test (r = +0.06 vs a
permutation band [−0.07, +0.12]) is consistent with everything from
tau1 = 0 to tau1 well above 0.02 — it never produced a bound.

### Style × wind (pooled, hypothesis-driven)

Team style gap (mean of team 1 − mean of team 2, standardised) × wind,
same regression, event-cluster bootstrap (1,000). Player-FE rows absorb
each player's mean (the FE projection is held fixed inside that
bootstrap, so those CIs are slightly optimistic).

| index (split-half reliability) | pool | f (share per 1 sd gap per 10 mph) | 95% CI |
|---|---|---|---|
| pace = mean k_match (r = 0.90) | outdoor | −0.0008 | [−0.0074, +0.0052] |
| pace, + player FE | outdoor | +0.0024 | [−0.0065, +0.0108] |
| pace | indoor | +0.0078 | [−0.0024, +0.0227] |
| serve-rally win rate (r = 0.53) | outdoor | +0.0033 | [−0.0082, +0.0165] |
| serve rate, + player FE | outdoor | +0.0064 | [−0.0041, +0.0189] |
| serve − return (r = 0.19) | outdoor | +0.0013 | [−0.0058, +0.0074] |
| serve − return | indoor | +0.0081 | [−0.0048, +0.0221] |

All null; the indoor placebo is, if anything, the more positive arm.
Best-powered bound (pace, the most reliable index, 18,752 games): at
15 mph a one-sd banger-vs-grinder style gap moves the point share by
−0.1pp [−1.1, +0.8], i.e. **at most ~3pp of win probability**. The
serve-rate bound is looser (+2.5pp share at the CI edge, ~+4.7pp after
dividing by its 0.53 reliability). The serve−return index is too
unreliable (r = 0.19) to bound anything: its interval should be read
five times wider.

---

## Caveats

* ERA5 grid wind is a noisy proxy for on-court wind and 27% of joins use
  planned rather than actual start times; both attenuate `d`, `tau1` and
  `f` toward zero, so the *upper* bounds above are the safe side of the
  error and the point estimates are, if anything, too small. Restricting
  to actual start times moved d by 0.002 (L0) and 0.013 (L2).
* Outdoor 14+ mph is 4.6% of games and 20+ mph is nearly unobserved; none
  of this licenses a claim about genuinely severe conditions.
* L5/L7 keep <10% of the interaction variation; they are reported for
  completeness, not as evidence either way.
* The style indices are crude and partly encode tour/level rather than
  personal style (pace's 0.90 reliability is suspiciously high for a
  "style" measure); a real banger-vs-grinder axis needs shot-level data
  that does not exist in this stack.
* Three style indices × two pools were tested; no multiplicity correction
  is applied because nothing came close to significance.
