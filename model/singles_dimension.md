# Is "singles surplus" a real second dimension of skill?

*Session 2026-08-09. Script: `model/singles_dimension.py`. Artifacts:
`model/singles_dimension_summary.json`, `data/singles_surplus.csv`.*

**Verdict: it is a REAL and highly reliable property of a player — and it
says nothing measurable about doubles. A second dimension of the PLAYER,
not a second dimension of DOUBLES SKILL.**

---

## Why the question needs an auxiliary channel

v2 gives each player one number, and that scalar is a sufficient statistic
for doubles outcomes *by construction*. Decomposing it into physical /
selection / strategy is likelihood-flat — every triple that sums to `d_p`
fits identically. Same class of non-identifiability as the cross-gender
offset. Escaping it requires a channel that loads on the components
**differently**, and the test of a candidate dimension is not "can I
compute it" but reliability plus incremental validity.

Singles is the best channel in this archive: 26k games, and
singles~doubles r = 0.74, so ~45% of singles variance is orthogonal to
doubles. Candidate = **singles surplus**: a player's singles value minus
what their doubles value predicts, residualised **within gender** (men and
women never play each other in singles, so the two singles scales are
connected only through the prior and their relative level is arbitrary).

## Gate 1 — reliability: PASSED, comfortably

Both disciplines are refit independently on each half with the same static
MAP estimator, then each half's surplus is computed from that half alone.

Two traps had to be handled, and both bite hard:

1. **Shared rating error.** Residualising both halves against ONE doubles
   value puts that value's error into both residuals and fakes the
   correlation. Hence a full independent refit of *both* disciplines per
   half.
2. **Shrinkage.** A player with few singles games is pulled to the prior,
   manufacturing a surplus that tracks game count — and game count is
   stable across halves, so it would fake reliability directly. The
   residualisation therefore carries log-count covariates. This is not
   hypothetical: **it inflates the answer from 0.562 to 0.776.**

| split | n | surplus reliability | null 95% | singles-rating ceiling | doubles-rating ceiling |
|---|---|---|---|---|---|
| random halves | 199 | **+0.562** (0.776 uncontrolled) | [−0.138, +0.132] | 0.949 | 0.812 |
| **era: 2024-25 vs 2026** | 149 | **+0.355** (0.671 uncontrolled) | [−0.145, +0.156] | 0.871 | 0.648 |

Both p = 0.002 (permutation). For scale against this project's own
precedents: clutch and durability cleared split-half r ≈ 0.13–0.15 and were
kept; wind skill failed at 0.06 and was shelved. The pre-registered
threshold for this one was r ≈ 0.4, and the random split clears it. The era
split is strictly harder — it must survive ageing, form and a changing
field, with no shared event context — and still lands at 0.355 against its
own ceiling of 0.87/0.65.

**Face validity** (not a ranking claim — reported because it is the kind of
check that catches a broken estimator): the men's top is Medina Alvarez,
Haworth, Crum, Ignatowich; the women's is Schmidt, **Genie Bouchard**,
Erokhina, Imparato. Bouchard is a former WTA world #5 — an elite athlete
whose tools transfer to singles faster than to kitchen craft — and Haworth
is already on record in finding 10 as singles-clutch-high while sitting
near the *bottom* in doubles clutch. The axis is measuring something
recognisable.

## Gate 2 — incremental validity inside doubles: NULL

What does **not** count as evidence: finding 6 (singles value predicts
DreamBreakers better than the doubles proxy). DreamBreakers *are* singles,
so that is close to tautological. The real question is whether the surplus
says anything about **doubles**.

And the main effect cannot, by construction — the doubles rating already
absorbs it. Observed main-effect slope ≈ −0.005, i.e. ~0, which is the
design's sanity check passing. So the test has to be an **interaction**: does
the surplus predict doubles performance in physically harder conditions
relative to the same players in easier ones?

Panel: 20,928 games, sd(x) = 0.314 where x = (team A summed surplus − team
B summed surplus). Players with no singles record get surplus 0 (= "average
surplus", the honest imputation); at least two of the four must be
measured. Side orientation randomised (finding 11).

| arm | slope difference (hard − easy) | 95% CI | reading |
|---|---|---|---|
| **decider, within-match** | **−0.0146** | [−0.0403, +0.0080] | null |
| same-day match load ≥ 2 | −0.0002 | [−0.0174, +0.0177] | dead null |
| heat > 80°F | +0.0122 | [−0.0050, +0.0327] | null |
| *decider, naive between-games* | *−0.0211* | *[−0.0445, +0.0005]* | **artifact — see below** |

Nothing. And the **same-day match-load arm is the one to weigh most**: pro
players run several disciplines a day, so "how many matches has this player
already finished today" applies to every game rather than just game 3, it
is the best-powered fatigue probe available, and it comes back at −0.0002.
If the surplus were fitness, that arm would show it.

Power: the CI half-widths are ≈0.02 on the slope difference, so with
sd(x) = 0.314 this rules out conditional effects larger than about **0.6
percentage points of point share per sd of surplus difference**. Small
enough to be a real bound rather than a shrug.

### The decider trap, again

The naive between-games decider contrast returned −0.021 with a CI that
grazed zero (and flipped in and out of significance across seeds). It is
the artifact `clutch_decider.py` documents: **reaching 1-1 selects on the
match-level shock**, which is common to all three games.

Two things worth recording. First, **controlling for eta does not remove
it** (−0.0213 → −0.0204) — the artifact is the match shock, not the skill
gap, so the obvious control is the wrong one. Second, differencing *inside*
the match — game 3's residual minus the mean of games 1-2 for the same team
— cancels the shock exactly, and the effect dies: −0.0146 [−0.0403,
+0.0080]. That within-match version is now the arm the script reports as
trustworthy, with the naive one kept beside it as a warning.

Independent corroboration that this had to be an artifact: `clutch_decider`
already established that between-game decider performance is **not a player
trait at all** (tau = 0 [0, 0.0043], 2 players past |z| > 1.96 where 8 are
expected). If no individual has decider ability, a *group* of players
cannot either.

## What this means

The surplus is real, stable, and face-valid. It is a legitimate second axis
to put on a player page — "how much of this player's game survives without
a partner" — and it is genuinely orthogonal information about the person.

It is **not** a hidden lever on doubles. The doubles rating already contains
everything the surplus knows about doubles outcomes, on average *and*
conditionally, down to a tight bound. So this does not become a model
feature, and it does not reopen the spec shootout.

The honest summary of the dimensionality question: **the second dimension
exists and is measurable, but it points away from doubles rather than
inside it.** For explanation and player pages, useful. For prediction,
nothing — which is the same answer the spec shootout keeps giving, now with
one more candidate ruled out at a measured floor.

What would move it: the surplus is the *physical/self-sufficiency* channel.
The channel with no proxy at all is **shot selection**, where execution and
decision are perfectly confounded in the result — a ball in the net is a
bad choice or a bad swing and the scoreboard records neither. That one is
vision or nothing.

Reproduce: `python model/singles_dimension.py --perms 400` (~10 min).
