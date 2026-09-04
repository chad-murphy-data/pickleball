# MLP Value Cap — Transfermarkt for MLP

Two columns per player: what our model says they're worth under a $1M
roster cap (value price), and what MLP actually charges once that's
published (league price). Surplus is the headline stat. Consumes
PICKLES (`data/v2_players.csv` etc.) and, where useful, MOTSON — does
not rebuild either.

**Read `HANDOFF.md` first** — dated 2026-09-04 (evening), it's the live
status snapshot and next-thread to-do. `phase2_joint_pool.md` is the
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

`phase2_price_model.py` is a first pass at turning V into dollars
(run: `python value_cap/phase2_price_model.py`). It sweeps the
star-premium exponent instead of picking one, and searches every
possible $1M roster instead of checking hand-picked examples. Read
`phase2_notes.md` before touching alpha or the floor -- it surfaced a
hard ceiling constraint (a high enough alpha prices a single player
above the entire team cap) and a benchmark problem (comparing rosters
against an all-replacement team can't tell a stars-heavy build from a
balanced one; candidate rosters need to be compared to each other, not
to replacement level) that the real archetype fit still needs to
account for. Not yet built: Phase 3 (site).
