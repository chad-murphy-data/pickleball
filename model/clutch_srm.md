# Clutch, rebuilt as an SRM over all four players in a rally

*Session 2026-08-02. Code: `model/clutch_srm.py`. Output:
`data/clutch_srm.csv` (1,148 players). Supersedes nothing yet — the frozen
index in `data/clutch_players.csv` is still what the site and
`clutch_at_99.py` use.*

## What changed and why

The frozen index is `mean(levz × residual)` over a player's **own serving
rallies**. Two structural problems:

1. **It sees about a quarter of a player's rallies.** Return rallies are
   dropped entirely — and in side-out scoring you cannot score on return, so
   every comeback is *built* on return rallies. The metric is blind to the
   half of the game where deficits get erased. The stated reason ("you can't
   pin a return rally on one of two receivers") does not hold: `pb_rally`
   carries `receiver_uuid` on **100%** of doubles rallies. And the argument
   was self-inconsistent — the server's partner plays the rally too, so serve
   attribution is exactly as team-contaminated as return attribution.
2. **It never adjusts for who else was on court.** It is raw plus-minus, not
   a rating.

The replacement makes every rally one observation constraining all four
players at once:

```
logit P(serving side wins) = offset(skill, via the serve-aware DP)
                           + (m_s1 + m_s2) - (n_r1 + n_r2)        <- LEVEL
                           + levz * ((a_s1 + a_s2) - (d_r1 + d_r2))  <- CLUTCH
```

`a` = attack clutch (side serving), `d` = defend clutch (side receiving).
Gaussian prior → partial pooling, so the noisy middle shrinks on its own
instead of being cut off post hoc at |z| > 1.5.

**970,747 rallies, 1,148 players** (vs 182). Waters goes from 1,695 rallies
of evidence to 37,333.

## Two traps hit on the way

**The level terms are not decoration.** A first cut without them produced
z-scores near +30 in exact skill order — it was re-estimating skill, not
clutch. `levz` is mean-zero within a *game*, but not within the subset of
rallies where one side serves, so any miscalibration in the skill offset
flows straight into `a` and `d`. With `m`/`n` in, the clutch terms are
identified only off the leverage **gradient**.

**A malformed diagnostic nearly hid it.** The first skill check compared
`m − n`, but a strong player has a high serve level *and* a high return
level (return enters negated), so it cancels by construction and read
+0.05 — falsely reassuring. The skill-facing combination is `m + n`.

## Permutation null — the decisive check

Shuffle `levz` within each game and refit. Everything else is preserved:
same players, same rallies, same outcomes, same offsets. Any spread that
survives is not a leverage effect.

| | sd(attack) | max abs z |
|---|---|---|
| real | 0.0928 | 32.9 |
| null | 0.0412 | 3.0 |
| **ratio** | **2.25×** | — |

The null's max |z| of 3.0 is what you'd expect for the maximum of ~1,148
standard normals, so **the standard errors are calibrated** and the real
spread is genuine leverage signal.

## Attack vs defend: one trait, not two

**corr(attack, defend) = +0.80** across 1,148 players. Correcting crudely
for measurement error pushes the estimate above 1.0 — which is impossible,
and is itself the finding: **attack and defend clutch are indistinguishable
from the same underlying trait.** (The >1 means the correction is crude —
ridge shrinkage makes the naive reliability formula invalid, and the two
estimates share rallies so their errors are correlated. Don't quote the
number; quote the conclusion.)

So "clutch defenders" and "clutch attackers" are the same people. That is a
real answer to the question, and it is not the answer I expected — it was
worth building the two parameters to find out they collapse.

## The open problem: it still tracks skill

**corr(v2 skill, total clutch) = +0.80** — *higher* than the frozen index's
0.58–0.71. The permutation null rules out "pure artifact", but it does not
rule out **skill that varies with leverage**: if better players' edge grows
in late-and-close states for any reason other than clutch, or if the DP's
constant-k assumption misprices those states as a function of skill gap, the
interaction absorbs it.

**This is the reason not to publish the new index yet.** The skill-adjusted
residual (below) is the defensible cut, but the whole ranking should be
treated as provisional until the leverage-interaction-with-skill channel is
tested directly.

Top 10 skill-adjusted: Ben Johns +0.459, Thomas Wilson +0.366, Waters
+0.360, Alshon +0.346, Daescu +0.329, Tardio +0.305, Jorja Johnson +0.303,
Bright +0.302, JW Johnson +0.300, Parenteau +0.275. Johns on top matches the
frozen index's residual finding exactly — one of the few clean
cross-validations available.

## Comparison to the frozen index

corr with the frozen index = **+0.59** on the 183 overlapping players. Not a
re-scaling — a different measurement.

**Population caveat: the two indices are not rank-comparable.** The frozen
index required ≥300 *serving* rallies (182 players, essentially the top
pros); this one requires ≥300 rallies *on court* (1,148 players, a much
wider net). A percentile in one is not a percentile in the other.

## Status

Not adopted. `clutch_players.csv` remains the index of record. Before this
replaces it: resolve the skill-tracking question above, and re-run
`clutch_at_99.py` against these values (the defender half may carry 9-9
signal the server-only index cannot see, which was the original motivation).

## Reproduce

```
python model/clutch_srm.py         # needs SUPABASE_ANON_KEY, ~5 min
```
