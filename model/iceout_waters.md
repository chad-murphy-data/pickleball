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

But the grid above is *symmetric* — it penalizes both teams' skill gaps
equally, which is not what "ice out Waters" means. The freeze is one-sided,
and one-sided is a different, much sharper instrument.

## The freeze-out mechanism — the actual "ice-out" story

The eye-test claim is that Bright/Fahey *funneled every ball to Johnson and
kept it away from Waters*. We take that as the **premise** (we don't try to
prove it — see the data ceiling below) and ask the model one question: **if
the freeze was real, what does it do to the odds?** The answer is the
striking part of this whole file.

**Why 88% and not 99% to begin with.** The weak-link term
`stronger + weaker + γ·|gap|` with γ = −0.18 is *algebraically identical* to
the stronger player only covering a share `w = (1+γ)/2 ≈ 0.41` of the court:

```
2·[(1−w)·weaker + w·stronger]  =  stronger + weaker + (2w−1)·(stronger−weaker),
which equals the weak-link value exactly when w = (1+γ)/2.
```

So **the model already assumes Waters covers only ~41% of the court** —
opponents already tilt ~59% of the load onto Johnson, and *that* is why they
were 88% and not 99%. Sweeping w (`coverage_curve`) walks the same line all
the way to the freeze:

| Waters's court share w | team-1 value | P(win) |
|---|---|---|
| **0.00** — fully iced, touches nothing | +2.37 | **52.0%** |
| **0.41** — realized γ tilt (= the pre-match number) | +2.88 | **90.0%** |
| 0.50 — equal share | +3.00 | 94.0% |
| 0.75 | +3.31 | 98.8% |
| 1.00 — Waters does everything | +3.62 | 99.8% |

The pre-match 90% *is* the w = 0.41 point (an internal consistency check —
same γ, reached two independent ways). Ice Waters completely and you slide
down the same curve to a **coin flip**.

**The asymmetric freeze.** Set team-1 = `2·Johnson` (Waters iced to zero
touches), team-2 at normal weighting, then add a skill dial k
(`asymmetric_freeze`):

| k (skill nudge on top of the freeze) | P(win) |
|---|---|
| 0.0 — freeze only, every rating left as-is | **52.0%** |
| 0.5 | 35.8% |
| 1.0 | 22.0% |
| 2.0 | 5.7% |

And the identity behind it (`two_of_weaker`): literally **two Jorja Johnsons
vs the real Bright/Fahey = 49.7%.** A dead coin flip. **The freeze alone —
with Waters's rating and everyone else's left exactly where the model had
them — erases the 88%.** It needs no "Waters had a bad day" and no rating
error; it needs her to not touch the ball. That is the one-sided instrument
the symmetric grid couldn't represent, and it's the parsimonious explanation
for the upset: not that the model was wrong, but that the game it priced
(Waters covers 41%) is not the game that was played (Waters covers ~0%).

## The data ceiling (why we assume the freeze rather than measure it)

**No shot-level data exists anywhere in this stack** (a CLAUDE.md house
rule). The referee log (`getListLogs`, cached under `raw/match_logs/`)
records per rally only the **server** and the **receiver of the serve** —
the first ball. The freeze happens *mid-rally* (the 3rd/5th/7th shots at the
weaker player), which is invisible to every feed we have. As a sanity check,
the serve-receiver split in this exact match was ~even (Johnson received 10,
Waters 9) — uninformative, precisely because it only sees the return, not
where the rally went afterward. So Part B is honestly a **model mechanism
study** — "if the freeze was real, here is what it does" — not a measurement.
Any real targeting signal would need shot tracking we don't have (ceiling: a
vision pipeline on broadcasts).

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
4. **A single result can illustrate a mechanism without measuring it.**
   Part B is defensible *only* because it's framed as "what the model says
   the freeze does," never "here's proof they froze her." Keep that wall up.

## Reproduce (handoff for a new thread)

Everything above comes from one self-contained script:

```bash
python model/iceout_waters.py            # all tables + the holdout validation
python model/iceout_waters.py --json     # + model/iceout_waters_summary.json
```

Pure stdlib + numpy, ~5 s. To retarget **any** game, edit the `MATCHUP` /
`OBSERVED` / `LOPSIDED_PTS` constants at the top of the script — player
values are looked up by name from `data/v2_players.csv`, chem from
`data/v2_dyads.csv`, and the posterior scalars (γ, sd_m) are constants from
`model/v2_fit_summary.json`. The race DP mirrors `model/v2_holdout.py` and
`web/sitelib/race.py` — keep the three in sync if you touch it.

A fresh run should print, within Monte-Carlo noise:

| number | expected |
|---|---|
| plug-in P(win) | 90.0% |
| honest P(loss), value-uncertainty | ~11.9% (→ 88.1% win = the site's 88%) |
| match-shock P(loss) *(the trap, do not ship)* | ~17.6% |
| coverage w=0 / 0.41 / 1.0 | 52.0% / 90.0% / 99.8% |
| asymmetric freeze k=0 / 1 / 2 | 52.0% / 22.0% / 5.7% |
| two-Johnson identity | 49.7% |
| holdout Brier: value-unc / match-shock | 0.1658 (best) / 0.1668 (worse) |

**Story layer** (built from these numbers for a general audience) lives in
`content/waters_iceout/`: `analysis_dossier.md` (7-lens menu + all numbers),
`stats_summary.md` (source of truth for the drafts), `explainer_iceout.md`
(long-form), `reddit_post.md` + `threads_post.md` (no-hedge voice),
`iceout_infographic.html` (+ `iceout_preview.png`). Deeper method context:
`model/spec_shootout.md` (the weak-link / convex-gap finding, #1).
