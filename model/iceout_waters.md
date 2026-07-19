# Decomposing the surprise of a single result

*2026-07-19. Code: `model/iceout_waters.py` (`--json` →
`model/iceout_waters_summary.json`). Case study: the Mid-Season
women's-doubles final game, **Waters / Johnson lost 6-11 to Bright /
Fahey** — the weekend's headline upset and a graded 88%-favorite MISS on
the site's receipts.*

This is a reusable idea, not a one-off. Point the tool at any four
players and an observed result and it answers: **given the ratings we
had, how surprised should the model have been — and which dial explains
the surprise?** The generalizable lesson is at the bottom.

## The four sources of surprise

When a favorite loses, exactly four things could be "responsible," and
they are not equal:

1. **Race variance** — even at a fixed, correctly-known skill gap, a
   first-to-11 sprint is noisy. A favorite loses some fixed fraction
   outright, full stop.
2. **Skill-estimation error** — our values have posterior SDs; maybe the
   favorites were a bit overrated and/or the underdogs underrated.
3. **Structure (γ, the weak-link dial)** — how far a team is dragged
   toward its weaker player by targeting / a freeze-out.
4. **Match shock** — v2's *fitted* per-match random effect, sd_m ≈ 0.35
   logit: real game-to-game variation beyond skill (off day, tactics,
   adrenaline, a hot opponent). Already in the model — it just collapses
   when we quote a single-number win probability.

The tool turns each dial and reports the probability of the outcome
bucket you care about.

## What the model actually said (plug-in, ratings as of that day)

Values (per-point logit): Waters 1.80±0.10, Johnson 1.17±0.09,
Bright 1.33±0.09, Fahey 1.08±0.08. With γ = −0.183 and dyad chemistry,
per-point p(Waters/Johnson) = 0.630, and:

| outcome | probability |
|---|---|
| **Waters/Johnson WIN** | **90.0%** (modal score 11-5, 12.9%) |
| lose (any score) | 10.0% |
| **lose 11-7 or worse** (score ≤ 7) | **3.2%** |
| the exact 6-11 | 0.9% |

The most likely single thing that could happen, by a mile, was a
comfortable Waters/Johnson win around 11-5. The realized 6-11 sat in a
thin tail.

## How surprised should we honestly have been?

Add the uncertainty sources one at a time (Monte-Carlo posterior
predictive, independent normal draws per player):

| model of the world | P(loss) | P(lose 11-7 or worse) |
|---|---|---|
| plug-in (best-guess skills) | 10.0% | 3.2% |
| + skill-estimation error | 11.9% | 4.2% |
| + γ uncertainty | 11.9% | 4.2% |
| **+ match shock (sd_m — the model's own)** | **17.6%** | **7.9%** |

So the fully honest read is: an outright loss was about **1 in 6**, and a
loss *this lopsided* about **1 in 13**. Uncommon — but the kind of thing
that happens somewhere on nearly every event slate. And a telling
conditional: **given that they lost at all, there was a 33% chance it was
11-7 or worse.** When a favorite loses to a genuinely good team, it is
often *not* close — the same bad-day/tactics shock that flips the result
also tends to widen it. 6-11 was not a freak way to lose; it was a fairly
ordinary one.

## Both dials at once

`P(lose 11-7 or worse)`, sweeping the skill dial (Waters/Johnson each down
*k*·SD, Bright/Fahey each up *k*·SD — "favorites at the bottom of their
range, underdogs at the top") against the weak-link dial γ:

| k \ γ | −0.18 (v2) | −0.35 | −0.50 | −1.00 (min-only) |
|---|---|---|---|---|
| 0.0 | 3.2% | 4.4% | 5.6% | 11.9% |
| 0.5 | 7.1% | 9.0% | 11.1% | 20.4% |
| 1.0 | 13.7% | 16.6% | 19.6% | 31.7% |
| 1.5 | 23.6% | 27.6% | 31.3% | 45.1% |
| 2.0 | 36.6% | 41.1% | 45.2% | 58.9% |

Reading the grid:

- **Neither dial alone gets you there.** Cranking γ all the way to
  min-only (a spec the shootout *rejected* on the full holdout, 0.178 vs
  0.166 Brier) only lifts a lopsided loss to 12%. Pushing skills is
  stiffer still: these four play constantly, so their SDs are tiny
  (0.08–0.10), and moving all four by 1σ *in the convenient direction at
  once* is already a ~1-in-1,500 joint coincidence (1.5σ ≈ 1-in-50,000).
- **You need the bottom-right corner** — a heavy freeze-out *and* a
  multi-σ joint skill miss — before Bright/Fahey become real favorites,
  and even the extreme corner (k=2, γ=−0.5) only reaches p = 0.43 /
  75% loss. Manufacturing that after seeing the box score is drawing the
  target around the arrow.
- **The dominant term is the one that isn't a "tweak" at all:** the
  match shock. It alone moves the lopsided-loss probability more than
  skill-error and γ-uncertainty combined, because sd_m ≈ 0.35 is large
  and it is the model's own honest statement that one game to 11 is
  noisy.

## The lesson that generalizes

For **well-observed players, a single upset is almost entirely dials 1
and 4** — race variance plus the match shock the model already budgets
for. Dials 2 and 3 are stiff: tight rating SDs make "we misjudged them"
a weak explanation, and γ can't honestly leave the range the full-season
data supports. So the correct response to one 6-11 is *not* to move a
dial — it's to note the model called it a ~1-in-6, and 1-in-6 things
happen. To actually indict the ratings or the structure you need **many**
games pointing the same way (the convex-gap / freeze-out test in
`spec_shootout.md`), never one.

The flip side, and why this tool is worth keeping: when the same
decomposition is run on a *low-game* player (rookie, small sample), dial 2
becomes wide and "we had the skill wrong" genuinely can carry an upset.
The point of separating the four is that the answer is different for
different players — and now we can just compute it.
