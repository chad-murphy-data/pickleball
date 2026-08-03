# What this investigation found

*Companion to `model/clutch_leverage.md`, which is organised around the
statistics. This one is organised around what we learned. Session 2026-08-03.*

The technical file necessarily reads like a demolition log — artifact found,
estimator replaced, null recalibrated. That undersells it. Seven things came
out of this thread that did not exist before, and four of them are useful
outside the clutch question entirely.

---

## 1. Pickleball now has a leverage scale

Every rally in the archive can be priced by the exact amount of win
probability riding on it — computed from the serve-aware side-out DP at equal
strength, so it is a property of the *situation* and never of the players in
it. 1,464,258 rallies carry one.

The scale runs from about 0.02 to 0.463. The single biggest rally in the
sport is **being down 9-10 and receiving against the opponent's second
server: a 0.457 swing**, nearly half the game on one point. Its mirrors at
10-9 and 11-10 are the same size.

This is reusable. Any question of the form "when does this matter" — live
win-prob graphics, broadcast callouts, which points to feature in a
highlight, how to weight a rally in a model — now has a number attached.
The measured league constants came out of the same pass: **k = 0.4383 for
doubles, 0.5254 for singles** (the serve-rally win rate between equal sides),
confirming the 0.43 the repo had been assuming.

## 2. Side-out scoring leaves a fingerprint on any timing statistic

This is the finding I would keep if I could keep only one, because it
generalises past this project.

In side-out scoring a server keeps serving while winning, so **a service run
is a string of wins terminated by exactly one loss — and that loss lands at
the run's highest score, which is usually its highest leverage.** Every run
ends badly, structurally, with no psychology involved. Any statistic that
correlates "how big was the point" with "did you win it" picks this up.

It would be harmless if it hit everyone equally. It does not: a better server
has longer runs, so the single terminating loss is diluted. The artifact
therefore **scales with serve quality — correlation +0.87 with a player's own
serve-win rate, +0.65 with overall rating.**

That is exactly the shape of a real clutch finding, which is why the earlier
pass (`model/clutch.md`) produced a plausible list of stars. Any sport with
side-out scoring — pickleball, volleyball, older badminton and squash
scoring, racquetball — has this waiting for whoever measures clutch next.

## 3. A doubles rally belongs to four players, and the estimator has to say so

We could have argued about whether crediting a rally to its server is
defensible. Instead we measured it: inject a known team-level clutch effect
of tau = 0.010 into simulated seasons and read it back through
server/receiver attribution.

**It recovers 0.0042 — 41% — and names zero significant players.**

So the fix is not a refinement, it is a different estimator: the cell is the
whole game, the outcome is the *side's*, both partners get the same number,
and individuals separate only through partner rotation. That is the same
identification channel the v2 rating model already uses, which is a
reassuring place to land rather than a compromise.

The same test says singles is unaffected — there the side is one person and
the statistic is exact. That is why singles is the clean arm.

## 4. Clutch is real, and it is a minority trait

Not "small and universal" — the distinction matters and the two are
arithmetically indistinguishable from a single summary number. A field where
5% of players carry +0.03 and 95% carry exactly zero has the same tau as
everyone carrying a uniform 0.0067.

Asking the rare-trait question directly:

- **Spike-and-slab**: likelihood ratio **72.2** against "nobody is clutch",
  versus a null distribution of 1.0 ± 1.3. Fitted shape: about an eighth of
  doubles players carrying an effect of sd 0.014, the rest carrying nothing.
- **Tail counts**: 6 players past z = 2.5 where the null median is 2; 3 past
  z = 3 where the null median is 0. And the excess sits in the **200-500 and
  500+ game bins** (11 vs 4, 5 vs 1), not the thin bin — precision sharpens
  it, which is what a real effect does and miscalibrated errors do not.
- **Select then verify**, the test that establishes individuals: name the top
  40 on **2024-25 alone**, measure that fixed roster on **2026 alone**, and it
  returns **z = 3.77** against a null centred at 0.29 ± 1.00. No no-clutch
  season in 30 ever reached it.

Players named on the first two seasons deliver in the third. That is the
claim, tested the way it should be.

## 5. CWPA: a unit for the record, separate from the forecast

Two questions hide under one word and they need different statistics.
*Is this player clutch?* is a forecast — it needs replication and shrinkage.
*How much did this player actually win in big moments?* is a record. It
happened, and demanding that a record replicate is a category error.

The ledger unit is **CWPA — Clutch Win Probability Added, denominated in
games.** +1.0 means one full game's worth of win probability banked purely
from *when* a side's rally wins landed, at the same total rallies won. It is
not shrunk (a record is not an estimate) but it is baseline-corrected,
because subtracting the side-out artifact sets where zero is rather than
pulling anyone toward the field.

Career doubles: **Johns +26.3, Waters +21.0, Tardio +15.4, Alshon +10.6,
Erokhina +10.0** (best rate at +2.30 per 100 games). 2026 alone: Johns +12.2.
Same family as finding 9's MVP outcome accounting.

## 6. How to read a team-attributed number

Johns and Waters are each other's partner in 500 games — 44% of his doubles
record and 46% of hers. Their two apparently independent 5-sigma results
shared nearly half their data.

Splitting by partner: **Johns survives with her removed (z 3.27 on 648
games); Waters does not with him removed (z 1.14 on 577 games)**, and 577
games at se 2.93 is a real null rather than thin data. Tardio is intermediate
at 2.25.

The transferable lesson is procedural and now sits in CLAUDE.md: with a
team-attributed statistic, **one dominant pairing can carry an entire career
total, so the partner split runs before any name is published.** The check
costs nothing and it changed a headline.

## 7. The variance decomposition of a pro pickleball match

This fell out of the deciders work and is useful well beyond it.

To ask whether anyone elevates in a deciding game, the null has to reproduce
how often deciders happen and how they are selected. The race model's nominal
per-match sd of 0.352 does neither — it predicts a 21.9% decider rate against
a real 28.1%, and a selection structure of −0.213 against a real −0.016.

Two components are needed and **they pull in opposite directions**: a
per-MATCH effect makes a match's games agree, which *lowers* the decider rate
and drives a skill-linked selection bias; a per-GAME effect makes them
disagree, which *raises* the decider rate and biases nothing. Real pickleball
wants mostly the second: **sd_match ≈ 0.15, sd_game ≈ 0.35**, which
reproduces the decider rate to 0.281 vs 0.281 and the selection to −0.016 vs
−0.016.

That is a real measurement of how much of a match's outcome is "today" versus
"this game", and it belongs in any future simulation of PPA matches — title
odds, draw simulations, live win probability — not just this analysis.

Against that calibrated null, nobody elevates in deciders: var(z) = 0.827
against chance 1.000, tau = 0 with CI [0, 0.0043], and 2 players past
|z| = 1.96 where 8 are expected. **A tight zero is a measurement, not a
failure** — it says the folk version of clutch, the player who is different
when the match is on the line, is not in this data, while the within-game
timing version is.

---

## The rule that came out of it

Three times in one session an assumed-parameter null nearly produced a
confident false positive, and the pattern was identical each time: an
artifact correlated with skill, subtracted with the wrong magnitude, handing
the best players free credit. Once too little (the attributed estimator saw
41% of the effect and reported none), twice too much (sd_match 0.352 in the
deciders; the flat-rate null in the cross-era test).

**A null must reproduce the data's own structure before it is allowed to
correct anything.** For the deciders that meant matching both the decider
rate and the selection correlation before believing a single player number.
Where a null cannot be calibrated, the honest move is a positive control —
inject a known effect and check the machinery returns it — which is what
`clutch_power.py` and `clutch_circularity.py` do.

That rule is worth more than any of the individual numbers above.
