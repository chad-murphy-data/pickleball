# Bucketing the big points — eight definitions, one answer

*Session 2026-08-03. Companion to `model/clutch_state_srm.py` (the state-FE
SRM) and `model/clutch_endogeneity.py` (the zero-clutch simulation).*

After the DP-leverage construction failed the simulated null, the natural
question was whether the *definition* of "big point" was the problem. It
isn't. Eight definitions, every one fitted with score-state fixed effects
(so players are compared within the same state, never against their own
regular rallies), every one gated against the zero-clutch simulation:

| rule | big share | real sd | corr(skill) | split-half | sim/real |
|---|---|---|---|---|---|
| DP leverage, top quartile | 13% | 0.0803 | +0.086 | **+0.036** | 1.01 |
| anyone ≥ 8 | 31% | 0.0880 | +0.098 | +0.089 | 0.95 |
| anyone ≥ 9 | 21% | 0.0858 | +0.059 | +0.066 | 0.95 |
| anyone ≥ 10 | 12% | 0.0809 | +0.034 | +0.023 | 0.95 |
| both ≥ 9 | 3.5% | 0.0646 | +0.039 | +0.060 | 0.95 |
| 9+ and other side 6+ | 11% | 0.0774 | +0.060 | +0.047 | 1.00 |
| 9+ and other side 7+ | 8% | 0.0746 | +0.049 | +0.069 | 1.01 |
| 10+ and within 2 | 3.8% | 0.0645 | +0.038 | +0.093 | 0.97 |

## Read the reliability column

The `sim/real` ratios hover at 1.0, but that comparison is arguable — model
misfit could put real a few percent above a simulation. **Split-half
reliability cannot be argued with.** It runs 0.02–0.09 across every
definition, meaning the per-player numbers do not replicate between disjoint
halves of the matches. Skill replicates above 0.9. A real trait replicates.
These don't.

## The definitions were never the problem

The list deliberately includes the most human definition available — *9+ with
the other side at 6+*, i.e. late AND close, excluding 9-1 where the only
thing at stake is avoiding a pickle. It performs no better than the rest, and
its simulation ratio is exactly 1.00.

What changed the answer was not the bucket but the **score-state fixed
effects**. Without them the same data yields sd 0.335, skill correlation
+0.72 and reliability 0.63 — all of which the zero-clutch simulation
reproduces. With them, every bucket collapses to noise. The earlier signal
was composition: which game situations a player is observed in is itself
determined by how good they are.

## Status

Consistent with `clutch.md` §5 and now far better supported: there is no
measurable clutch-beyond-skill in this archive under any of these
definitions. No index is adopted.

## Reproduce

The sweep is a thin loop over `clutch_state_srm.fit` with the rule swapped;
see that module for the model and `clutch_endogeneity.py` for the simulation.
