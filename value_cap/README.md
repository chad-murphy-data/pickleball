# MLP Value Cap — Transfermarkt for MLP

Two columns per player: what our model says they're worth under a $1M
roster cap (value price), and what MLP actually charges once that's
published (league price). Surplus is the headline stat. Consumes
PICKLES (`data/v2_players.csv` etc.) and, where useful, MOTSON — does
not rebuild either.

Read `phase0_bench_value.md` first. It's a scoping document, not a
finished phase: the bench-value question (do roster spots 5 and 6 carry
real win value, or are they pure injury insurance) is still open
pending MLP rules confirmation, and the write-up deliberately stops
short of picking a strategy-determining assumption. See its "working
rule" section before adding any parameter to later phases.

Not yet built: Phase 1 (per-player expected-wins-added value model,
dyad-aware), Phase 2 (price curve fit), Phase 3 (site).
