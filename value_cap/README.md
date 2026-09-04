# MLP Value Cap — Transfermarkt for MLP

Two columns per player: what our model says they're worth under a $1M
roster cap (value price), and what MLP actually charges once that's
published (league price). Surplus is the headline stat. Consumes
PICKLES (`data/v2_players.csv` etc.) and, where useful, MOTSON — does
not rebuild either.

**Read `HANDOFF.md` first** — dated 2026-09-04 (evening), it's the live
status snapshot and next-thread to-do. `phase2_pricing.md` is the
current Phase 2 result (joint pool, context-averaged value `phi` from
`shapley_value.py`, the must-buy test, and the Waters window). The rest
of this README and the phase docs below are the layered record
underneath it.

Read `phase0_bench_value.md` first for the phase-by-phase background. It's a scoping document, not a
finished phase: three of the four bench-value rules questions are still
open pending Chad's confirmation, and the write-up deliberately stops
short of picking a strategy-determining assumption. See its "working
rule" section before adding any parameter to later phases. One rules
question (DreamBreaker eligibility) is now confirmed and turned out to
matter a lot -- see the DreamBreaker-specialist section there and
`phase1_first_cut.md` for the model result it produces.

`phase1_value_model.py` is a rough first cut at the value model
(run: `python value_cap/phase1_value_model.py`, writes
`player_value.csv`). It's dyad- and role-aware and reuses the
production PICKLES race engine; every simplifying assumption is stated
in its docstring and in `phase1_first_cut.md`.

`phase2_pricing.py` turns value into dollars (run:
`python value_cap/phase2_pricing.py --quick`; default basis is phi from
`shapley_value.py`, `--value total` reproduces the Phase 1 V_total basis).
It sweeps the star-premium exponent alpha instead of picking one, prices
both genders from one league pool, and tests a price list with the
must-buy instrument (best $1M roster with a player vs best without, both
sides optimizing) rather than hand-picked rosters. `pool.py` defines the
priced pool. `phase2_notes.md` is the record of the first split-pool pass
(its script, `phase2_price_model.py`, was removed once `phase2_pricing.py`
covered it): it surfaced the alpha ceiling (a high enough alpha prices
one player above the team cap) and the benchmark problem (rosters must be
compared to each other, not to an all-replacement team), both of which
`phase2_pricing.md` builds on. Not yet built: Phase 3 (site).
