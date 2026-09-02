# MLP Value Cap — Transfermarkt for MLP

Two columns per player: what our model says they're worth under a $1M
roster cap (value price), and what MLP actually charges once that's
published (league price). Surplus is the headline stat. Consumes
PICKLES (`data/v2_players.csv` etc.) and, where useful, MOTSON — does
not rebuild either.

Read `phase0_bench_value.md` first. It's a scoping document, not a
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
in its docstring and in `phase1_first_cut.md`. Not yet built: Phase 2
(price curve fit against real roster archetypes), Phase 3 (site).
