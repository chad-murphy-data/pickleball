# The comparison we refuse to publish — and the three ways we tried to break our own refusal

*Session analysis, 2026-07-13. Methodological exercise only. House rule
unchanged: no cross-gender ranking is published as fact. This file
documents why the rule survives contact with every identification trick
the data permits.*

## 1. The impossibility, stated properly

Every sanctioned game has equal women on each side (womens 2W–2W, mens
0–0, mixed 1W–1W). Add a constant c to every woman's value and every
predicted score in 36,000 games is unchanged: the likelihood is exactly
flat along the men-vs-women offset. Verified numerically: profiling c on
womens-only games moves the log-likelihood by 0.000000 across c ∈ [−2, +2].

**Graph form (the useful way to see it).** Pair-vs-pair games estimate
differences of pair sums. Individuals are recoverable from pair sums iff
the pairing graph (players = nodes, observed pairings = edges) contains an
odd cycle. Mixed-only pairing edges all cross the M/W bipartition → the
graph is bipartite → even cycles only → a one-parameter family (+t all
men, −t all women) fits identically. That free parameter IS the
cross-gender offset. More data shrinks noise; it never touches a null
direction.

## 2. Attempt 1 — the weakest-link loophole (γ-channel profile)

The one term that breaks exact flatness: γ·|partner gap|. In mixed, the
within-team gap |v_M − (v_W + c)| depends on c, so mixed games carry
*functional-form-borne* information. Profiling the mixed-game likelihood
over c (values frozen at the joint-fit posterior means, monthly values,
14,042 mixed games):

- **c\* = +0.08 logit, nominal 95% CI [+0.06, +0.12]** — the women's
  scale sits almost exactly where the joint fit put it, ~1:1 with the men's
  (median active man 0.38, median active woman 0.39).
- Robust in-form: γ at ±1 sd, half, double → c\* ∈ [+0.08, +0.10];
  2024–25 vs 2026 → +0.10 / +0.06; PPA +0.08. MLP alone is uninformative
  (CI [−0.12, +0.24]) — drafted rosters are balanced, so there is little
  partner-gap variance for the channel to use. The signal lives in
  lopsided PPA pairings.

Taken literally this puts three women (Waters +1.80, Bright +1.33,
J. Johnson +1.17) above every active man (Johns +1.11), Waters #1 overall
by the width of the men's entire top-25 spread.

**Why we don't believe the precision.** (a) Circularity: the values were
fitted with mixed games included; the profile holds 3,559 values fixed, so
prior-centering and the γ channel — which contribute same-order location
information (~50 log-lik units per 0.1 shift each) — cannot be separated
without an explicit-offset refit. (b) 100% form-borne: a small
mixed-specific deviation in how partner gaps convert to points moves c\*
arbitrarily. (c) Even if true, "same per-point logit" is a within-population
statement; nobody in the data has ever played across the net at the other
population. Registered follow-up: at the season-end refit, fit γ separately
for mixed vs same-gender contexts — if they differ, this channel is
formally dead; if they match, the exercise earns a smaller asterisk.

## 3. Attempt 2 — the head-to-head kink test

Waters/Johns vs Bright/Patriquin, the sport's premier mixed rivalry:
38 games (2025-03 → 2026-05), Waters/Johns 25–13, 56.9% of points.

Mixed-vs-mixed is bipartite, so the offset cancels in the raw win rate —
EXCEPT through the weakest-link geometry, and only via *kinks*: the
likelihood is exactly flat in c wherever the within-team pecking order is
the same on both teams, and moves only where c flips someone's internal
ordering. Here the informative window is −0.75 < c < −0.25 (Bright drops
below Patriquin while Waters still tops Johns):

| hypothesis | Δ log-lik | predicted W/J point share |
|:--|--:|--:|
| c ≈ 0 (scales aligned) | 0 (best) | 61.3% |
| c = −0.5 | −2.3 (~10:1 against) | 63.3% |
| c ≤ −0.75 | −4.5 (~90:1 against) | 64.9% |

Observed 56.9% sits closest to the aligned-scales prediction. Caveat: a
single-pair H2H confounds chemistry with offset (W/J underperforming all
hypotheses slightly is consistent with B/P owning the good chemistry).

## 4. Attempt 3 — what would actually settle it (the designed experiment)

Any same-gender pairing creates the odd cycle. The maximal single game is
**2W vs 2M** (offset enters the margin with coefficient 2): one game to 11
carries se(c) ≈ 0.24 logit of DIRECT evidence; ten games ≈ ±0.08 — a
single exhibition afternoon outweighs all 14k sanctioned mixed games,
because its information is likelihood-borne, not form-borne.

King-of-the-beach rotation of {Waters, Johns, Bright, Tardio} does it
automatically — the three pairings-of-four include exactly one
same-gender-vs-same-gender game. Model predictions by hypothesis (win
prob and expected margin, race to 11, first-listed team):

| pairing | c = 0 | c = −0.5 | c = −1.0 |
|:--|--:|--:|--:|
| Waters/Johns v Bright/Tardio | 85% (+3.8) | 90% (+4.4) | 92% (+4.9) |
| Waters/Tardio v Bright/Johns | 79% (+3.0) | 86% (+3.8) | 90% (+4.4) |
| **Waters/Bright v Johns/Tardio** | **98% (+6.4)** | **37% (−1.3)** | **<1% (−7.5)** |

The two mixed rows are near-invariant (they serve as controls, pinning
within-gender contrasts and the γ nuisance); the whole experiment lives in
row three, where hypotheses are separated by ~7 points per game. A
best-of-five decides.

## 5. Where this leaves us

Three fragile instruments — the corpus-wide γ profile, its perturbations,
and the H2H kink test — independently point the same direction: *the
fitted scales look roughly aligned*. None of them is evidence of the kind
we publish. The refusal stands until someone plays the odd cycle.

DreamBreakers do not help (segments are same-gender by rule); the singles
corpus does not help (singles draws are gender-segregated too — same
bipartite structure, trivially).
