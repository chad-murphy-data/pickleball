# MLP Value Cap — HANDOFF (2026-09-04, evening; supersedes the morning snapshot)

Dated status snapshot + next-thread to-do. Read this first, then
`phase2_pricing.md` (this session's result, with every number), then
`phase0_bench_value.md`, `phase1_first_cut.md`, `phase2_notes.md` for the
layered record underneath.

## Where we are

Phase 0 (bench value) and Phase 1 (V per player) stand. Phase 2 (V → dollars)
is now in a much better place than this morning: the price formula's last
two arbitrary inputs — the 50/50 gender split and Phase 1's replacement-
context V as the value basis — were both tested and both replaced, and the
"alpha" question has a precise answer instead of a range. All of it reuses
the production tie model; nothing is a proxy.

## Session findings (2026-09-04, evening) — details in phase2_pricing.md

1. **Joint pool, not two $10M sub-pools.** Price everyone from one $20M
   pool; the gender split falls out as ~57/43 women (V is offset-free, so
   this is legitimate). On the indifference pairs it barely matters, but
   the 50/50 split overprices every man by ~14% (men's total value 3.42 vs
   women's 4.80) and makes Ben Johns a must-avoid at EVERY alpha; joint
   fixes him. Adopted.

2. **Phase 1's V is the wrong basis for pricing.** It measures each player
   next to a #60 partner, where the weakest-link gap penalty damps stars in
   proportion to their value. No single power law on V reconciles the
   pairs (needed alpha climbs with anchor rank: #10 → 0.70, #3 → 0.86,
   #1 → 1.39), and Waters is a bargain at every alpha while Johns/Bright are
   overpriced. Replaced by **phi** (`shapley_value.py` →
   `player_value_shapley.csv`): context-averaged over league rosters,
   Shapley-style, 3,000 common-random-number draws, over a pool that is
   SOLVED FOR (top 60 per gender by phi, iterated until it reproduces
   itself — converges in 3, churn only at #56-60; `pool.py` defines it).
   phi/V ≈ 1.04 Waters, 0.72-0.85 rest of top-10, 0.17-0.27 at #60 —
   more convex than V. DB specialists (Barlow #117, Bouchard #216) price
   into the pool on the singles channel.

3. **The must-buy test is the instrument** (`must_buy` in
   `phase2_pricing.py`): best $1M roster WITH a player vs best WITHOUT,
   both best-responding, scored head-to-head. 0.5 = fair. Injection floor:
   ±50% mispricing moves it ±0.15-0.18, so 0.49 means "right within a few
   percent". The morning's indifference pairs A/B anchor on star-plus-two-
   scrubs builds and reward a GM for a bad build; don't fit alpha to them.

4. **On phi + joint, there are exactly two regimes and one player decides
   which.** alpha ≤ 0.8: every star reads 0.33-0.40 because the best
   roster without them is a WATERS roster (her own line 0.60-0.66) — one
   dominant strategy. alpha ≥ 0.9: Waters is unrosterable and #1-#10 in
   both genders all sit at 0.45-0.51 out to alpha 1.6 — no dominant
   strategy, alpha nearly irrelevant to fairness.

5. **The Waters fact.** Her phi is 5.3% of the league's total; a team is
   5%. She is worth more than a whole payroll by the model's own
   accounting. Fair-and-rosterable window is a knife-edge: **alpha
   0.84-0.85, ~$720-730k, the five cheapest pool players (~$53k each)
   around her**; fair at 0.84, infeasible from 0.853. Below: dominant
   strategy. Above: not in the league. The "star discount" (alpha < 1)
   exists only to make her rosterable; for everyone else alpha above 0.9
   is a free dial. NOTE the edge moved from 0.91 (first draft, V_total
   pool with negative-phi members priced at exactly the floor) to 0.85
   (self-consistent pool, all members positive) — the value weight of the
   cheapest cast is what binds, so the pool's bottom definition matters.

6. **The floor is cosmetic.** Beyond the morning's "cancels out of pair
   comparisons": with 6×20 priced players and cap = pool/20, any roster
   is legal iff its value share ≤ 1/20, for every floor. Floor/ceiling
   decoupling (morning finding 5) is moot. The floor only matters for the
   $500k min-spend arithmetic (phi joint pool: binds at every alpha —
   cheapest legal roster $440k at 0.6 down to $240k at 1.2).

7. Chad's tiered-pricing constants (morning finding 3) are withdrawn by
   Chad ("more hypothetical than anything") — not retested, and phi +
   alpha ≈ 0.845 lands Waters $725k / Bright $520k / Johns $420k on its own.

## Next steps

1. **Phase 3 (site) can start** from the alpha ≈ 0.845, phi-basis, joint-pool
   price list (phase2_pricing.md §7). Present alpha as pinned by the
   Waters feasibility edge, not fitted — and say so on the page.
2. **Injury/absence Monte Carlo layer** (Phase 1's open item). phi assumes
   everyone is always available; a Waters roster of five floor players is
   the most availability-fragile build in the league, which may be the
   thing that legitimately closes her window. Sweep the absence rate, don't
   pick it.
   The same layer should draw the pool's bottom too: the Waters edge is
   set by the value weight of the five cheapest pool members (finding 5),
   so a realistic bottom (who actually gets rostered at $50k) is the
   first thing that moves it.
3. **Realistic-roster weighting for phi.** Uniform draws from the priced
   pool; teams actually cluster talent. Top-10 ordering is robust (flat
   phi/V there); depth prices at the margin may move.
4. **Confirm 20 teams and the $500k min-spend rule officially.** The Waters
   edge moves with 1/N_teams — 24 teams would widen her window, 16 would
   close it.

## Reusable code

- `value_cap/shapley_value.py` — phi (context-averaged value) AND the
  self-consistent priced pool (top 60 per gender by phi, iterated until
  the pool reproduces itself), cached to `player_value_shapley.csv`
  (`in_pool`, `pool_rank` columns). Re-run after any v2/singles refit.
- `value_cap/pool.py` — `load_pool("phi"|"total")`: the one place the
  priced pool is defined. "total" keeps the morning's V_total pool
  reproducible.
- `value_cap/phase2_pricing.py` — `prices(pool, alpha, mode)`,
  `find_crossover(anchor, block_fn)`, `alpha_reconciling(a, b, mode)`,
  `best_roster(price, opp, must, exclude)`, `must_buy(pid, alpha, mode)`,
  `crossover_alpha`. CLI: `--quick`, `--value phi|total` (default phi),
  `--floor N`, `--must-buy "Name" --modes joint,split`. The rank-sweep
  boilerplate the morning handoff asked to promote now lives here.
- `value_cap/phase2_pricing_results.txt` — raw output of every sweep
  quoted in the write-up.
- `value_cap/phase1_value_model.py` — unchanged; still the tie model and
  the V_total/V_regular/V_db table. `phase2_price_model.py` (the morning's
  split-pool first pass) was REMOVED 2026-09-04 evening — everything it
  did is in `phase2_pricing.py` under `--value total --modes split`;
  `phase2_notes.md` records what that pass found.
