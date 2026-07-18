# The specification shootout — did we settle on v2 too fast?

*2026-07-18. Code: `model/spec_shootout.py` (results in
`model/spec_shootout_summary.json`). Question posed: "we got to the model
awfully quickly — try every reasonably defensible strategy, measure them
against the current one, and describe each." This file is that.*

## Protocol (one paragraph)

Every strategy prices the same frozen holdout — all non-forfeit side-out
games dated on/after 2026-06-01 whose four players each have ≥10 games in
the pre-June training data (n = 926; the published 884 from
`model/v2_holdout.py` was the same filter before the last week of games
landed) — and every strategy goes through the same final step, the exact
race-to-11/15 win-by-2 DP, so nothing wins on plumbing. All free
parameters (scales, loadings, Elo K, everything) are fit on training data
only: rating-like values on the full 2024–26 train window, small
prediction-stage parameters on its 2026 portion (n = 8,862 games).
Differences vs the reference get paired-bootstrap 95% CIs on ΔBrier
(negative = better than v2), plus P(better) = the bootstrap share in
which the challenger wins. The reference `v2_plugin` is v2 exactly as the
holdout script computes it, minus posterior draws (posterior means only —
the draws file isn't in this clone); it lands at 77.1% / 0.1665 vs the
published draws version's 77.4% / 0.1653 on the pre-refresh 884, i.e. the
same model to within the noise the draws themselves add.

One honesty note up front: this is ~30 strategies against one holdout.
The winner of a 30-way race on 926 games is expected to look better than
it is (winner's curse), and nothing below survives at the "gate a model
change" bar of the working rules. Reading guide: P(better) ≥ 0.95 is
"interesting, verify next window", not "ship it".

## Scoreboard

Full-holdout strategies (n = 926 unless noted). ΔBrier vs `v2_plugin`;
negative favors the challenger.

| strategy | n | acc | Brier | log loss | ΔBrier | 95% CI | P(better) |
|---|---|---|---|---|---|---|---|
| **X_ensemble** | 926 | **77.4%** | **0.1638** | **0.4991** | −0.0027 | [−0.0060, +0.0005] | 0.95 |
| A_uncert_shrink | 926 | 77.1% | 0.1660 | 0.5095 | −0.0005 | [−0.0011, +0.0000] | 0.96 |
| v2_mixture | 926 | 77.1% | 0.1658 | 0.5068 | −0.0008 | [−0.0019, +0.0002] | 0.93 |
| A_gamma_by_ctx | 926 | 76.7% | 0.1660 | 0.5052 | −0.0005 | [−0.0023, +0.0012] | 0.72 |
| A_v2_rescaled | 926 | 77.1% | 0.1661 | 0.5058 | −0.0004 | [−0.0020, +0.0011] | 0.71 |
| A_seed_order | 926 | 76.3% | 0.1661 | 0.5051 | −0.0004 | [−0.0025, +0.0015] | 0.66 |
| A_weak_gamma | 926 | 76.9% | 0.1662 | 0.5058 | −0.0003 | [−0.0021, +0.0014] | 0.64 |
| A_sum | 926 | 76.7% | 0.1663 | 0.5064 | −0.0002 | [−0.0025, +0.0019] | 0.58 |
| **v2_plugin (reference)** | 926 | 77.1% | 0.1665 | 0.5144 | — | — | — |
| A_experience | 926 | 76.6% | 0.1670 | 0.5071 | +0.0005 | [−0.0018, +0.0026] | 0.34 |
| C_margin_level | 926 | 75.9% | 0.1683 | 0.5096 | +0.0017 | [−0.0033, +0.0066] | 0.24 |
| A_chem | 926 | 75.2% | 0.1696 | 0.5181 | +0.0031 | [−0.0008, +0.0072] | 0.06 |
| B_refit_decay | 926 | 76.2% | 0.1696 | 0.5183 | +0.0031 | [−0.0009, +0.0071] | 0.07 |
| B_refit_sum | 926 | 76.2% | 0.1697 | 0.5158 | +0.0032 | [−0.0019, +0.0080] | 0.11 |
| A_momentum | 926 | 75.6% | 0.1700 | 0.5157 | +0.0035 | [+0.0009, +0.0059] | 0.00 |
| B_refit_gamma | 926 | 76.6% | 0.1703 | 0.5187 | +0.0037 | [−0.0011, +0.0085] | 0.06 |
| E_elo_game | 926 | 74.6% | 0.1748 | 0.5255 | +0.0083 | [−0.0001, +0.0167] | 0.03 |
| A_min_only | 926 | 72.7% | 0.1781 | 0.5319 | +0.0116 | [+0.0054, +0.0176] | 0.00 |
| C_v1_published | 823 | 74.7% | 0.1807 | 0.5408 | +0.0147 | [+0.0063, +0.0227] | 0.00 |
| A_max_only | 926 | 73.1% | 0.1824 | 0.5450 | +0.0159 | [+0.0074, +0.0244] | 0.00 |
| B_refit_min | 926 | 73.3% | 0.1827 | 0.5490 | +0.0162 | [+0.0083, +0.0240] | 0.00 |
| B_refit_max | 926 | 71.2% | 0.1853 | 0.5497 | +0.0187 | [+0.0084, +0.0289] | 0.00 |
| E_elo_point | 926 | 71.2% | 0.1855 | 0.5496 | +0.0190 | [+0.0080, +0.0300] | 0.00 |
| C_game_level | 926 | 70.1% | 0.1936 | 0.5827 | +0.0271 | [+0.0151, +0.0391] | 0.00 |
| C_match_level | 926 | 69.7% | 0.2075 | 0.6288 | +0.0410 | [+0.0261, +0.0554] | 0.00 |
| E_coin | 926 | 42.2%¹ | 0.2500 | 0.6931 | +0.0835 | [+0.0684, +0.0979] | 0.00 |

¹ the coin's "accuracy" is the share of games the second-listed team won
(p = 0.5 ties break against team 1); its Brier/log-loss rows are the real
floor.

**TL;DR: v2 survives.** Nothing beats it significantly; two dozen
defensible alternatives lose by amounts ranging from a rounding error to
DUPR-sized. The only challengers that even lean positive are (a) an
equal-weight ensemble of v2 with two worse models, and (b) v2 with its
own uncertainty integrated — which is not a rival model but a nicer way
to serve the same one, and is what the site's calibration layer already
approximates in production.

---

## Family A — same ratings, different aggregation

*"Once we have the individual player ratings, try different things."
Everything here keeps the frozen v2 player values and only changes how
the four numbers become a prediction. Scales/loadings fit on 2026
training games through the race DP.*

**A_sum — just add the two players.** No weakest link, no chemistry.
Loses to full v2 by 0.0002 Brier — i.e., nearly nothing. The honest
reading: on top of good player values, the structural extras are worth a
hair, not a headline.

**A_weak_gamma — sum + free γ·|gap|.** The game-winner-fit γ comes out
−0.157 (v2's point-level posterior: −0.166) — the weakest-link dial is
stable across objectives, which is reassuring, and worth about +0.0001
Brier over raw sum out of sample. A useful algebraic note for everything
below: `sum + γ|gap|` is *identical* to `(1+γ)·max + (1−γ)·min`, so γ is
exactly a min/max re-weighting. v2's γ = −0.166 means the team value is
58% weaker player, 42% stronger player.

**A_min_only / A_max_only — "weaker player only" and "stronger player
only".** These are the endpoints of that same dial (γ = −1 and γ = +1),
and the data rejects both hard: +0.0116 and +0.0159 Brier, CIs excluding
zero by a wide margin, accuracy down 4½ points. Pickleball teams are
mostly the sum of their players; the weakest link tilts the blend
slightly toward the weaker one, and either extreme throws away real
information. (In mixed games "the weaker player" also silently depends on
the cross-gender offset convention — one more reason these were never
going to be publishable models.)

**A_gamma_by_ctx — separate γ for mixed vs same-gender.** Fits
γ_mixed = −0.19 vs γ_same = −0.12: a hint that the weak-link effect is
stronger in mixed (makes pickleball sense — targeting the weaker player
is a cleaner strategy when the gap is structural), but the improvement is
−0.0005 [−0.0023, +0.0012]. Not established; parked as a hypothesis for
the season-end refit.

**A_chem — free loading on the dyad chemistry term.** The training
window wanted loading ≈ 7.9 (i.e., "chemistry matters 8× more than v2
says") and then lost by +0.0031 on the holdout — a textbook overfit of a
tiny, noisy component. Meanwhile v2's fixed loading of 1 (in the
reference) vs dropping chem entirely (A_weak_gamma) differ by ~0.0001.
Consistent with the established finding: chemistry is real-ish but small
(sd ≈ 0.05 logit), and predictively it's decoration.

**A_seed_order — does the listed order (seeding) carry information?**
Adds a per-tour intercept for the first-listed team. Fitted intercepts
are −0.011 (MLP) and +0.016 (PPA) logit — essentially zero; the bracket
already tells you nothing the ratings don't. Clean null.

**A_experience — career games as a feature.** +0.019 per log-game
differential in training, no holdout gain (+0.0005). The ratings already
know who the veterans are.

**A_momentum — the hot hand.** EWMA (45-day half-life) of each player's
recent over/under-performance vs model expectation, walked forward with
no leakage. The training fit actually assigns it a *negative*
coefficient (recent overperformers regress), and it loses +0.0035
[+0.0009, +0.0059] out of sample. Form beyond the rating is noise here —
same conclusion the dynamic random walk already encodes (tau ≈ 0.04
logit/month is *slow*).

**A_uncert_shrink / v2_mixture / A_v2_rescaled — not rivals, but the
most useful row in the family.** The plugin's raw etas are ~16% too hot
(free rescale fits 0.84): posterior means fed through a convex race
curve overstate favorites relative to averaging the race probability
over the posterior, which is exactly what the official draws-based
holdout does. Integrating over each matchup's value uncertainty
(v2_mixture: −0.0008, P = 0.93; the fitted-width version A_uncert_shrink:
−0.0005, P = 0.96, width ≈ 0.72× the naive Σsd²) recovers it. This
*validates* the site's existing out-of-sample-fitted calibration layer
(`web/calibration.json`) — production numbers already pass through
`calibrate()`, which is the global-average version of this fix. A
possible refinement (per-matchup width instead of global) is logged in
Verdicts below.

## The mixed arena — who carries mixed doubles?

*438 of the 926 holdout games are mixed. Every strategy here is priced
on exactly those games; the reference is v2 on the same subset
(76.7% / 0.1703).*

| strategy | acc | Brier | ΔBrier vs MX_v2 | 95% CI | P(better) |
|---|---|---|---|---|---|
| MX_v2 (reference) | 76.7% | 0.1703 | — | — | — |
| MX_gender_free | 75.3% | 0.1717 | +0.0014 | [−0.0026, +0.0050] | 0.24 |
| MX_weaker_gender | 73.3% | 0.1793 | +0.0089 | [−0.0011, +0.0180] | 0.04 |
| MX_women_only | 74.0% | 0.1828 | +0.0124 | [+0.0004, +0.0243] | 0.02 |
| MX_men_only | 65.1% | 0.2178 | +0.0475 | [+0.0295, +0.0644] | 0.00 |

**The single most interesting result of the shootout.** Price mixed
doubles from the two women alone and you get 74.0% accuracy — within
shouting distance of the full model. Price it from the two men alone and
you get 65.1%, barely better than DUPR prices full games. "Men don't
matter in mixed" is NOT the right reading, though: the free-loading
model (MX_gender_free) fits nearly *equal* per-rating-point weights
(men 0.75, women 0.77) and only matches — doesn't beat — the equal-weight
full model. The mechanism is spread, not importance: in these mixed
matchups the women's rating differences run 1.5× wider than the men's
(sd 0.364 vs 0.240 logit). Pro men in mixed are more interchangeable;
the women's gap is usually the biggest rating fact on the court. A rating
point of man is worth the same as a rating point of woman — there are
just fewer of them separating the men.

(These three rows are offset-safe: within-gender differences cancel any
cross-gender offset convention. MX_weaker_gender — "each team is its
weaker member" — is *not* offset-safe, inherits the prior convention, and
loses anyway; house rule stands.)

## Family B — refit the ratings under each structure

*Family A reused v2's ratings, which were themselves fit under sum+γ —
arguably unfair to the alternatives. Here the per-player values are
re-estimated from scratch under each structure: fast MAP, point-binomial
likelihood (v2's own), ridge prior at v2's sd_v = 0.38, static (no
monthly walk), no dyad/match effects, then a train-fit output scale.*

| structure | acc | Brier | vs v2 | vs its own family |
|---|---|---|---|---|
| B_refit_sum | 76.2% | 0.1697 | +0.0032 | family baseline |
| B_refit_gamma (γ → −0.26) | 76.6% | 0.1703 | +0.0037 | ≈ baseline |
| B_refit_decay (12-mo half-life) | 76.2% | 0.1696 | +0.0031 | ≈ baseline |
| B_refit_min | 73.3% | 0.1827 | +0.0162 | −0.013 |
| B_refit_max | 71.2% | 0.1853 | +0.0187 | −0.016 |

Two takeaways. First, the structure ranking is the same whether you
re-aggregate v2's ratings (family A) or refit values natively under each
structure: sum ≈ sum+γ ≫ min-only ≫ max-only. "Maybe min-only would work
if the ratings were fit for it" — no; even with values estimated under
the weakest-link-only likelihood, it's a DUPR-sized step backwards.
Second, the whole static-MAP family sits ~+0.003 Brier behind shipped v2.
That gap *is* the measured value of v2's remaining machinery — monthly
dynamics, the match random effect, full posterior — since likelihood and
structure are otherwise identical. Exponential recency-weighting
(B_refit_decay), the cheap imitation of dynamics, recovers essentially
none of it.

## Family C — what's the right level of analysis?

*Points, games, matches — same sum+γ structure, same data, only the
likelihood changes. (Rallies get their own family below.)*

| level | observation | acc | Brier | vs v2 |
|---|---|---|---|---|
| C_match_level | match winner (best-of collapsed) | 69.7% | 0.2075 | +0.0410 |
| C_game_level | game winner only | 70.1% | 0.1936 | +0.0271 |
| C_margin_level | game point margin (Gaussian) | 75.9% | 0.1683 | +0.0017 |
| B_refit_gamma | every point (binomial) | 76.6% | 0.1703 | +0.0037 |

The information ladder is unambiguous below the top: collapsing 35k
games to 20k match outcomes costs ~0.04 Brier; scoring only game winners
costs ~0.03. Once you use the *score* the ladder flattens — the Gaussian
margin model and the point-binomial land statistically on top of each
other (the margin model even noses ahead; its per-game noise term
absorbs the match-to-match overdispersion that the raw binomial
pretends away, which is the job v2's match random effect does properly).
Two footnotes for the record: (1) the shipped v1 (same margin likelihood
plus per-season hierarchy, 74.7% / 0.1807 on its 823 evaluable games) is
*worse* than this bare static twin, mostly because per-season fitting
throws away cross-year pooling; (2) "points as binomial vs the exact
race likelihood" is a non-question at fit time — under iid points the
race's stopping rule changes the likelihood only by a parameter-free
combinatorial constant (plus negligible deuce bookkeeping), which is why
`fit_v2.py` fits plain binomial and saves the race DP for prediction.

So: **fit on points (or margins — the data agrees they're nearly the same
information), never on bare winners.** The user asked "points as the
level of analysis? games?" — this table is the answer, and it's why v2
was built at the point level.

## Family E — different rating systems entirely

**E_elo_game — prequential Elo, K tuned on train (K = 0.2), updating
through the holdout as a live system would.** 74.6% / 0.1748: better
than shipped v1, ~0.008 Brier behind v2. Honestly strong for a
zero-inference online system, and a good reminder that most of the value
is "any decent rating on lots of games"; the Bayesian machinery buys the
last ~0.008.

**E_elo_point — Elo updated on point shares instead of wins.** Worse
(0.1855), which surprised me for a margin-aware variant: per-game point
share is noisy enough that with the larger K the tuner wants, blowout
games slosh ratings around; game-level Elo's binary signal is better
behaved at this K range.

**Platform (DUPR-synced) ratings, as-of-match** — the site's benchmark
nemesis, now in three flavors on the 824 rated holdout games: sum 67.0%
/ 0.2124, min-only 66.0%, max-only 63.1% (v2 on the same subset: 76.8% /
0.1668). No aggregation rescues it; the gap is the rating, not the
formula.

**E_coin** — 0.2500 Brier, as the universe requires.

## Family X — the ensemble

**X_ensemble — equal-weight logit average of v2 + the static margin
refit + game Elo. The only strategy that beat the reference: 77.4% /
0.1638 / 0.4991, ΔBrier −0.0027 [−0.0060, +0.0005], P(better) = 0.95.**
No weights were fit (so nothing leaked); the gain is textbook ensemble
diversity — three systems with decorrelated errors, two of which are
individually worse by 0.003–0.008. Under multiplicity this is exactly
the row winner's curse loves, so it does NOT clear the working-rules
gate today. It earns a shadow ledger: run it alongside v2 on the next
event windows and let receipts decide (see Verdicts).

## Family D — rally level: the tug of war

*The user's framing: "team 1 wins X% of points/rallies/serving
points/serving rallies, team 2 wins Y%, and we tug."*

First, the pleasing identity: for plain points, the tug of war IS the
current model. "Team 1 wins X, team 2 wins Y, normalize" is
p = X/(X+Y); write each team's X as the exponential of a team rating and
you get exactly the logistic p = σ(η₁ − η₂) that v2 fits per point
(Bradley–Terry). The tug-of-war intuition only becomes a *new* model
when X and Y are measured on different serve states — pickleball is
side-out scoring, so "win rally while serving" (scores a point) and
"win rally while returning" (wins back the serve) are different skills
with different consequences. That version needs rally-level data.

RALLY_SECTION_PENDING

## Verdicts

1. **Keep v2 as the model.** After ~30 defensible challengers on 926
   held-out games: no structural change is supported. Sum-of-players
   carries almost everything; weakest-link γ is a small real refinement;
   min-only/max-only/men-only are decisively worse; chemistry, momentum,
   experience, and seeding are predictive nulls.
2. **The women-carry-mixed result is publishable** (offset-safe form
   only): women-alone prices mixed at 74% vs 65% for men-alone, because
   women's rating spread is 1.5× wider — per-point loadings are equal.
   Candidate EXPLAINER/social item.
3. **Uncertainty handling matters more than specification.** The plugin
   is ~16% overconfident on the logit scale; posterior
   integration/calibration fixes it. Production already does this via
   `web/calibration.json`; a per-matchup mixture
   (`race.game_win_prob_uncertain` with each pairing's value sd) is the
   one cheap upgrade worth trying at the next calibration refit.
4. **Ensemble to the shadow ledger.** Register equal-weight
   v2+margin+Elo forecasts alongside v2 for upcoming events in
   `model/receipts.json`-style records (not on the site), grade after a
   few windows, adopt only if the edge survives contact with new data.
5. **γ-by-context** (stronger weak-link in mixed) is a hypothesis for
   the season-end refit, not a change today.

## Fine print

Fitting: family A/E scales and loadings by Nelder–Mead on game-winner
Bernoulli NLL through the race DP, on 2026 train games (n = 8,862);
families B/C by L-BFGS MAP with analytic gradients on all train games
(n = 34,811; players = every UUID appearing, ridge prior N(0, 0.3765) on
the point-logit scale — v2's posterior sd_v; min/max via softmin,
τ = 0.05; γ prior N(0, 0.3) as in v2; margin model prior N(0, 2.5) pts,
σ fit jointly, output rescaled to point logits on train). Elo
prequential with K from {0.02…0.3} by sequential train log loss
(2024-H1 burn-in excluded from the criterion). Platform arena uses
strictly-before-game-date snapshots from `data/per_match_ratings.json`.
Bootstrap: 4,000 paired resamples, seed 20260718. All rows deterministic;
rerun with `python model/spec_shootout.py` (add `--rally` for family D,
which needs `raw/match_logs/` from `scraper/harvest_logs.py`).
