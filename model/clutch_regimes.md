# Is clutch forced to be zero-sum?  No — and the first answer here was wrong

*Session 2026-08-02. Code: `model/clutch_regimes.py`. **Not adopted**;
`data/clutch_players.csv` remains the index of record.*

## The question

Both existing constructions measure a leverage **gradient** against a
baseline fitted on ALL of a player's points, big ones included. That pins
their average residual near zero, so beating the baseline on high-leverage
points mechanically implies falling short on low-leverage ones. Is "wins the
big points" doomed to mean "loses the small ones"?

## First attempt, and the error in it

I fitted two levels per player on **disjoint** rally sets (top-quartile
leverage vs bottom half) so nothing would constrain their difference, and got
an impossible result: corr(v2, low-leverage level) = **−0.747**, corr(v2,
high) = **+0.792**. Elite players supposedly performing below their own
baseline on ordinary points.

I diagnosed that as endogeneity in the regime label — the score path is a
function of the rally outcomes — and concluded the zero-sum property was "the
price of identification."

**That conclusion was wrong.** The user caught it: the per-rally offset
`logit(k_side)` is derived from v2 and calibrated so each player's OVERALL
rally-win rate reproduces their v2 point share. **The offset pins the sum.**
Disjoint regime fitting was irrelevant — any high-leverage gain still had to
be paid for at low leverage to keep the total matching v2. The constraint was
never removed; it moved from the within-game centring into the skill anchor,
and I read the resulting artefact as a substantive finding.

## Corrected construction

Drop the skill anchor. Replace the per-rally v2 offset with a **per-regime
league constant**, so each player's regular-point and big-point abilities
float freely relative to the field and nothing pins their sum. Opponent
adjustment is retained — all four players are still in every rally.

## Results

**Global regime effect** (the reason the per-regime constant is needed — a
common shift in the player terms cancels and cannot absorb this):

| regime | serving side wins |
|---|---|
| low leverage | 44.18% |
| high leverage | 40.01% |
| | **−4.17 pp** |

Holding serve is materially harder on big points, for everybody.

**The answer to the question:**

| | value |
|---|---|
| corr(regular-point ability, big-point ability) | **+0.515** |
| corr(v2, regular-point) | +0.563 |
| corr(v2, big-point) | +0.825 |
| corr(v2, lift = big − regular) | +0.721 |
| sd(regular) / sd(big) / sd(lift) | 0.133 / 0.383 / 0.335 |

**Regular-point and big-point ability are POSITIVELY correlated (+0.52).**
Players who are good on ordinary points are good on big ones. The zero-sum
tradeoff was an artefact of the baseline from start to finish — it is not a
property of pickleball, and it is not the price of identification.

**Player spread is ~3× larger on big points** (sd 0.383 vs 0.133). The field
separates far more when the point matters. Suggestive, not established — see
the caveat below.

## Caveats — these are load-bearing

* **The lift still correlates +0.72 with skill.** Clutch-beyond-skill is
  still not cleanly separated. This construction fixes the sign problem; it
  does not solve the identification problem.
* **The permutation nulls are muddy and should not be leaned on.** Shuffling
  the regime label within game gives corr(regular, big) = +0.705 — *higher*
  than the real +0.515 — and null sd(regular) = 0.240, *exceeding* the real
  0.133. Shuffling mixes the regimes so each parameter fits a blend, which
  distorts the reference rather than providing a clean one. The real/null
  ratio on the lift (1.96×) is the only null statistic here worth quoting,
  and only weakly.
* The regime-label endogeneity I raised last time is still real. It just was
  not what produced the impossible numbers.

## On bracket-stakes as an instrument

Also flagged by the user, also correct: bracket-level stakes restrict the
sample to players whose matches carry stakes, and the cleanest instruments
are the most selective — gold-medal matches only exist for players who reach
them. "Deciding game of a matchup" and "elimination vs round robin" apply to
everyone in team play, so it is not fatal, but the population narrows exactly
where the identification improves.

## Status

Not adopted. Recorded because the sequence — plausible fix → impossible
result → wrong diagnosis → correct diagnosis — is the useful part.

## Reproduce

```
python model/clutch_regimes.py      # anchored (the flawed version, kept for
                                    # comparison); pass anchor=False to
                                    # fit_regimes for the corrected fit
```
