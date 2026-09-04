# MLP Value Cap — HANDOFF (2026-09-04, evening; supersedes the morning snapshot)

Dated status snapshot + next-thread to-do. Read this first, then
`phase2_joint_pool.md` (this session's result, with every number), then
`phase0_bench_value.md`, `phase1_first_cut.md`, `phase2_notes.md` for the
layered record underneath.

## Where we are

Phase 0 (bench value) and Phase 1 (V per player) stand. Phase 2 (V → dollars)
is now in a much better place than this morning: the price formula's last
two arbitrary inputs — the 50/50 gender split and Phase 1's replacement-
context V as the value basis — were both tested and both replaced, and the
"alpha" question has a precise answer instead of a range. All of it reuses
the production tie model; nothing is a proxy.

## Session findings (2026-09-04, evening) — details in phase2_joint_pool.md

1. **Joint pool, not two $10M sub-pools.** Price everyone from one $20M
   pool; the gender split falls out as ~57/43 women (V is offset-free, so
   this is legitimate). On the indifference pairs it barely matters, but
   the 50/50 split overprices every man by ~16% (men's total value 3.22 vs
   women's 4.74) and makes Ben Johns a must-avoid at EVERY alpha; joint
   fixes him. Adopted.

2. **Phase 1's V is the wrong basis for pricing.** It measures each player
   next to a #60 partner, where the weakest-link gap penalty damps stars in
   proportion to their value. No single power law on V reconciles the
   pairs (needed alpha climbs with anchor rank: #10 → 0.70, #3 → 0.86,
   #1 → 1.39), and Waters is a bargain at every alpha while Johns/Bright are
   overpriced. Replaced by **phi** (`shapley_value.py` →
   `player_value_shapley.csv`): context-averaged over league rosters,
   Shapley-style, 3,000 common-random-number draws. phi/V ≈ 1.03 Waters,
   0.7-0.8 rest of top-10, 0.1-0.3 at #60 — more convex than V.

3. **The must-buy test is the instrument** (`must_buy` in
   `phase2_joint_pool.py`): best $1M roster WITH a player vs best WITHOUT,
   both best-responding, scored head-to-head. 0.5 = fair. Injection floor:
   ±50% mispricing moves it ±0.15-0.18, so 0.49 means "right within a few
   percent". The morning's indifference pairs A/B anchor on star-plus-two-
   scrubs builds and reward a GM for a bad build; don't fit alpha to them.

4. **On phi + joint, there are exactly two regimes and one player decides
   which.** alpha ≤ 0.8: every star reads 0.37-0.43 because the best
   roster without them is a WATERS roster (her own line 0.60-0.65) — one
   dominant strategy. alpha ≥ 0.9: Waters is unrosterable and #1-#10 in
   both genders all sit at 0.47-0.51 out to alpha 1.6 — no dominant
   strategy, alpha nearly irrelevant to fairness.

5. **The Waters fact.** Her phi is 5.5% of the league's total; a team is
   5%. She is worth more than a whole payroll by the model's own
   accounting. Fair-and-rosterable window ≈ **alpha 0.88-0.91, ~$780-810k,
   five floor players around her**. Below: dominant strategy. Above: not
   in the league. The "star discount" (alpha < 1) exists only to make her
   rosterable; for everyone else alpha above 0.9 is a free dial. Free
   search at 0.89 cycles through three unrelated rosters within a point
   of each other — the fair-price signature.

6. **The floor is cosmetic.** Beyond the morning's "cancels out of pair
   comparisons": with 6×20 priced players and cap = pool/20, any roster
   is legal iff its value share ≤ 1/20, for every floor. Floor/ceiling
   decoupling (morning finding 5) is moot. The floor only matters for the
   $500k min-spend arithmetic (joint pool: binds from alpha ≈ 0.9).

7. Chad's tiered-pricing constants (morning finding 3) are withdrawn by
   Chad ("more hypothetical than anything") — not retested, and phi +
   alpha ≈ 0.89 lands Waters $793k / Bright $558k on its own.

## Next steps

1. **Phase 3 (site) can start** from the alpha ≈ 0.89, phi-basis, joint-pool
   price list (phase2_joint_pool.md §7). Present alpha as pinned by the
   Waters feasibility edge, not fitted — and say so on the page.
2. **Injury/absence Monte Carlo layer** (Phase 1's open item). phi assumes
   everyone is always available; a Waters roster of five floor players is
   the most availability-fragile build in the league, which may be the
   thing that legitimately closes her window. Sweep the absence rate, don't
   pick it.
3. **Realistic-roster weighting for phi.** Uniform draws from the priced
   pool; teams actually cluster talent. Top-10 ordering is robust (flat
   phi/V there); depth prices at the margin may move.
4. **Confirm 20 teams and the $500k min-spend rule officially.** The Waters
   edge moves with 1/N_teams — 24 teams would widen her window, 16 would
   close it.

## Reusable code

- `value_cap/shapley_value.py` — phi (context-averaged value), cached to
  `player_value_shapley.csv`. Re-run after any v2/singles refit.
- `value_cap/phase2_joint_pool.py` — `prices(pool, alpha, mode)`,
  `find_crossover(anchor, block_fn)`, `alpha_reconciling(a, b, mode)`,
  `best_roster(price, opp, must, exclude)`, `must_buy(pid, alpha, mode)`,
  `crossover_alpha`. CLI: `--quick`, `--value shapley|total`,
  `--floor N`, `--must-buy "Name" --modes joint,split`. The rank-sweep
  boilerplate the morning handoff asked to promote now lives here.
- `value_cap/phase2_joint_pool_results.txt` — raw output of every sweep
  quoted in the write-up.
- `value_cap/phase1_value_model.py` / `phase2_price_model.py` — unchanged;
  still the value model and the original split-pool pricing.
