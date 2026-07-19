# Decomposing the surprise of a single result

*2026-07-19. Code: `model/iceout_waters.py` (`--json` →
`model/iceout_waters_summary.json`). Case study: the Mid-Season
women's-doubles final game, **Waters / Johnson lost 6-11 to Bright /
Fahey** — the weekend's headline upset and a graded 88%-favorite MISS on
the site's receipts.*

Reusable idea, not a one-off: point the tool at any four players and an
observed result and it answers *given the ratings we had, how surprised
should the model have been, and which source explains it?* — then lets you
turn each dial. The generalizable lessons are at the bottom, and one of
them is a correction we only found by testing.

## The four sources of surprise

1. **Race variance** — even at a fixed, correctly-known gap, a
   first-to-11 sprint is noisy; a favorite loses some fixed fraction.
2. **Skill-estimation error** — values carry posterior SDs; maybe
   favorites were overrated / underdogs underrated.
3. **Structure (γ, the weak-link dial)** — targeting / freeze-out.
4. **Match shock** — v2's fitted per-match random effect, sd_m ≈ 0.35.

The important, non-obvious result is about which of these you may legally
use to price a *future* game — see "The number we report" below.

## What the model actually said (plug-in)

Values (per-point logit): Waters 1.80±0.10, Johnson 1.17±0.09,
Bright 1.33±0.09, Fahey 1.08±0.08. With γ = −0.183 and dyad chemistry,
per-point p(Waters/Johnson) = 0.630:

| outcome | probability |
|---|---|
| **Waters/Johnson WIN** | **90.0%** |
| lose (any score) | 10.0% |
| lose 11-7 or worse (score ≤ 7) | 3.2% |
| the exact 6-11 | 0.9% |

## Given a loss, close scores dominate

A natural trap (which an earlier draft of this note fell into): "11-7 or
worse" covers ≤7, most of the *range* of losing scores, so it sounds like
the typical way to lose. It isn't — the *probability* is heavily
back-loaded onto the near-misses, because a per-point favorite that loses
usually only just lost:

| their score in a loss | share of all losses |
|---|---|
| 11-9 | 26% |
| 11-8 | 19% |
| deuce (10-12, 11-13, …) | 22% |
| 11-7 | 14% |
| **11-6 (actual)** | **9%** |
| 11-5 or worse | 10% |

So **~67% of losses are 11-8-or-closer**, the modal loss is 11-9, and only
**~19% of losses are 11-6 or worse**. The 6-11 that happened sat around
the 80th percentile of loss *severity* — a moderately lopsided loss, not a
typical one. (Reusable takeaway: for a favorite, "how bad was the loss"
carries information beyond "did they lose" — a blowout loss is a stronger
signal than a squeaker, which is why v2 fits on points, not just wins.)

## The number we report — and a correction

Here is the subtle part, and it corrects an earlier version of this
analysis. Add the uncertainty sources one at a time
(`--json` reproduces this):

| model of the world | P(loss) | P(11-7 or worse) |
|---|---|---|
| plug-in (skills known exactly) | 10.0% | 3.2% |
| **+ skill-estimation error** | **11.9%** | 4.2% |
| + γ uncertainty | 11.9% | 4.2% |
| + match shock (sd_m) | 17.6% | 7.9% |

It is tempting to call the last row (adding v2's per-match random effect)
"the most honest" — bottom-up, a new game *does* draw a fresh match shock.
**But that's wrong for prediction, and the holdout proves it.** The match
random effect is fit in-sample to absorb game-to-game wobble; adding it at
predict time double-counts noise. `holdout_calibration_test()` (baked into
the script) scores every method on the frozen June+ holdout, n = 926:

| method | Brier | log-loss |
|---|---|---|
| plug-in | 0.1666 | 0.5145 |
| **+ value uncertainty** | **0.1658** | 0.5069 |
| + value + match shock | 0.1668 | 0.5052 |
| + match shock only | 0.1665 | 0.5049 |

Adding the match shock makes **Brier worse than even the plug-in**
(0.1668 vs 0.1658) — it over-disperses, softening the many
confident-correct calls more than it helps the few misses. The
independent check agrees: in `web/calibration.json`, games predicted at
90% actually **won 91.7%** (n = 230) — a 90% favorite loses about **1 in
11**, right at the ~10–12% we reported.

**So the honest number for this game is ~12% (value-uncertainty), which is
essentially what the site's calibration layer already shows — not the 17%
an earlier draft claimed.** The match-shock row is kept in the code only
to demonstrate what over-dispersion looks like and why we don't ship it.
The methodological receipt: *when in doubt about a modeling choice, the
out-of-sample holdout is the judge* — and here it vetoed a plausible-sounding
correction.

## Both dials at once

`P(lose 11-7 or worse)`, sweeping the skill dial (Waters/Johnson down
*k*·SD, Bright/Fahey up *k*·SD) against the weak-link dial γ:

| k \ γ | −0.18 (v2) | −0.35 | −0.50 | −1.00 (min-only) |
|---|---|---|---|---|
| 0.0 | 3.2% | 4.4% | 5.6% | 11.9% |
| 0.5 | 7.1% | 9.0% | 11.1% | 20.4% |
| 1.0 | 13.7% | 16.6% | 19.6% | 31.7% |
| 1.5 | 23.6% | 27.6% | 31.3% | 45.1% |
| 2.0 | 36.6% | 41.1% | 45.2% | 58.9% |

Neither dial alone gets you to the observed lopsided loss. γ cranked to
min-only (a spec the shootout *rejected*, 0.178 Brier) only reaches 12%;
the skill dial is stiffer still — moving all four players 1σ the
convenient way at once is a ~1-in-1,500 joint coincidence, 1.5σ ≈
1-in-50,000. You need the bottom-right corner — heavy freeze-out **and** a
multi-σ joint skill miss — before Bright/Fahey are even coin-flips, and
building that after seeing the box score is drawing the target around the
arrow.

## The lessons that generalize

1. **For well-observed players, a lone upset is race variance + tight
   skill uncertainty** — "the Tuesday the model budgets for." Dials 2 and
   3 are stiff (these four play constantly; γ can't leave the data-supported
   range). To indict the ratings or the structure you need **many** games
   pointing the same way (the convex-gap / freeze-out test in
   `spec_shootout.md`), never one. For a *low-game* player, dial 2 widens
   and "we had the skill wrong" genuinely can carry an upset — which is why
   separating the sources is worth a script.
2. **Don't add the match random effect at prediction time.** It
   over-disperses out of sample. The honest predictive uncertainty is
   value-estimation error (what the calibration layer already encodes).
   This is the one thing we'd have gotten wrong by reasoning bottom-up
   instead of testing.
3. **How badly a favorite lost is signal.** Loss severity is
   back-loaded onto close scores; a blowout is rarer and more informative
   than a squeaker.
