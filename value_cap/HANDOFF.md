# MLP Value Cap — HANDOFF (2026-09-04, night; supersedes the evening snapshot)

Dated status snapshot + next-thread to-do. Read this first, then
`phase2_pricing.md` §8 (the shipped rule and every number behind it),
then `price_list.md` (the list itself), then `phase0_bench_value.md`,
`phase1_first_cut.md`, `phase2_notes.md`, `phase2_pricing.md` §0-7 for
the layered record underneath.

## Where we are

Phase 0 (bench value), Phase 1 (V per player), Phase 2 (V → dollars)
stand. **Phase 2 is SHIPPED**: alpha 1, one joint $20M pool, $30k floor,
Anna Leigh Waters franchise-tagged at $769k (user call 2026-09-04 night,
after the draft simulations and the pool/floor sweep). The price list is
`price_list.md` / `price_list.csv` with a doubles-only column so the
DreamBreaker lift is visible. Phase 3 (site) has not started.

## The shipped rule in one paragraph

Prices are proportional to phi (context-averaged tie value, `shapley_value.py`)
over the self-consistent top-60-per-gender pool, one joint pool for both
genders (women's share falls out at 56.6%). Waters' phi is 5.3% of the
league's value and a team is 5%, so her curve price ($903k) fits no legal
roster; she is tagged at cap minus the cheapest legal completion (two
cheapest priced women + three cheapest priced men) and the $134k gap is
redistributed over the other 119 by value. Everyone else is on the plain
alpha-1 curve. Full statement: `phase2_pricing.md` §8.

## What the sessions after the evening snapshot established

1. **Draft simulator built** (`draft_sim.py`): 20 teams, 6 rounds (3M+3W),
   snake or linear, owners with belief noise (0/10/25% of the gender
   pool spread) projecting a greedy final roster and picking the one with
   the best believed tie probability vs a reference roster; season =
   double round robin + top-4 playoff on TRUE values. Reads: parity
   spread, slot win/title %, star rows, undrafted priced players (INFO,
   near the top of the output — user rule), blueprint mix, spend.
   Deterministic (pid tie-breaks everywhere; verified across
   PYTHONHASHSEED). Three boards: `best60` (priced + next 60/gender by
   phi at the floor), `mlp2026` (priced + every real 2026 MLP participant
   outside the pool at the floor: 35M/28F — the default), `mlp2026only`
   (real participants only: 38M/39F priced + fill-ins; talent-starved).
   `KEEP` pins Grayson Goldin in the pool (user call: good value, was out
   for health).
2. **Tag list vs alpha 0.845 list in a real league**: tag wins every read
   (spread 4.3-4.6 vs 9.6-12.8; nobody in the top 30 undrafted vs Bellamy
   #29M left every time; cap spent). Slot 1 = Waters + cheap DB singles
   specialists at 66% / ~36% title; slots 2-20 at 47-53% / ~3.4%.
   `draft_sim_tag.md`, `draft_sim_tag_mlp2026.md`,
   `draft_sim_tag_mlp2026only.md`, `draft_sim_a0.845.md`.
3. **The real 2026 roster board does not shrink Waters' prize** (user
   question): 66% / 36% on mlp2026, 72% / 47% on mlp2026only.
4. **Pool size and floor are not levers** (`pool_floor_sweep.md`): floor
   $10k-$75k gives identical drafts; pool 80/100 per gender drops slot 1
   to 57% only by leaving $90-330k of every cap unspent and hands the
   prize to Bright's team (61-66%); a deeper replacement line makes
   Waters cheaper AND stronger (73% / 50% title vs #100). Fast
   self-consistent phi for any P is in `pool_floor_sweep.py` (`phi_pool{P}.csv`
   cached; FastTie twin of `shapley_value.phi_for`, self-tested on
   identical draws).
5. **Speed layer** (`fast_tie.py`): pair-strength table + interpolated
   game-probability grid, ~85x faster than the exact tie, max error 8e-5;
   `pair_table.csv` is the human-readable version. Everything above runs
   on it.
6. **Template strategies without scarcity** (`draft_strategies*.md`,
   `template_season.md`): no dominant blueprint on the tag list. Kept as a
   blueprint-vs-blueprint read; the draft sim is the league read.
7. **35% title odds is normal** (user, after the write-up): top NBA/EPL
   preseason favorites carry 30-50%. Ship it and say so on the page.

## Next steps

1. **Owner personas in the draft sim** (user request, words as given):
   "owner who overvalues men"; "guy who is trying to save money and draft
   only $500k worth of players"; "marketing guy who values big names";
   "person who wants real teams" (dyad/2026-pairing preference);
   "persona who picks up bargains first, then goes for high value
   players". Build as `Owner` subclasses with a belief/objective twist,
   mix them into a league of quants, read the same tables. Sweep the
   mix, don't pick it.
2. **Phase 3 (site)** off `price_list.csv`: two columns per player (value
   price, league price once MLP publishes), surplus as the headline;
   the doubles-only column next to the price; state the tag rule and the
   "pick 1 is a ~1-in-3 title shot because of one player" fact plainly.
3. **Injury/absence Monte Carlo** (Phase 1's open item): sweep the rate;
   the Waters-plus-five-cheap build is the most availability-fragile
   roster in the league and this is the one layer that could
   legitimately narrow her prize.
4. **Gamma switch sweep**: `FastTie` takes the per-division gamma dict
   (finding 1); rerun phi + prices + draft on it and report the delta.
5. **Realistic-roster weighting for phi** (teams cluster talent); top-10
   ordering is robust, depth prices may move.
6. **Confirm 20 teams and the $500k min-spend rule.** The tag moves with
   the cheapest completion, i.e. with the floor and the pool bottom; the
   prize moves with 1/N_teams.

## Reusable code

- `price_list.py` → `price_list.md/.csv` — the shipped list.
- `phase2_pricing.py` — `prices`, `prices_tagged`, `must_buy`, `best_roster`,
  `find_crossover`; CLI `--quick`, `--value phi|total`, `--floor`, `--must-buy`.
- `shapley_value.py` → `player_value_shapley.csv` (phi + self-consistent
  pool; ~15 min); `pool.py` `load_pool("phi"|"total")`.
- `fast_tie.py` — `FastTie(DOUBLES, SINGLES, gamma=float|dict)`; grid cached
  under `cache/` (gitignored).
- `draft_sim.py` — `--tag`, `--board best60|mlp2026|mlp2026only`,
  `--drafts`, `--seasons`, `--noise`; `set_board(mode, pool)` for scripted use.
- `pool_floor_sweep.py` — `phi_pool(P)`, `run_cell(P, floor, pool, ...)`.
- `draft_strategies.py`, `simulate_templates.py` — template rosters and
  their season.
- `phase1_value_model.py` — unchanged; still the exact tie model.
