# MLP Value Cap — HANDOFF (2026-09-05, rev 3 after the market-limit run; supersedes the 2026-09-04 rev 2 snapshot)

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
DreamBreaker lift is visible. **Phase 3 (site) first cut is BUILT**
(same night): `web/build_site.py:build_valuecap` renders `site/valuecap.html`
off `price_list.csv` (gender tabs, TAG chip, share of $20M, doubles pts +
rank, lift arrows, singles games; the tag rule and the one-favorite fact
stated on the page), linked from the site nav and a landing-page door.
The league-price / surplus column waits on MLP publishing real prices.
**Owner personas are DONE** (`personas.py` -> `personas.md`, summary in
`phase2_pricing.md` §9): the one persona that breaks the league is the
$500k cheapskate (alone 21% / no title; five of them spread 13.7, the
rest of the league 58%), which is the case for a $500k min-spend;
bargains-first is the worst sensible-looking strategy (24%, the top 60
are gone before round 4); overvaluing a gender, chasing names and mild
loyalty are all free; nothing dents Waters' team (63-68%). **No persona
creates a second contender** (title-odds concentration table in
`personas.md`: one team at 10%+ in every cell she is drafted).
**Dials probe DONE** (`dials_probe.py` -> `dials.md`, `phase2_pricing.md`
§10): on the price side there is exactly one lever (charge her above what
the cap allows); what actually moves her team is playing time (67% of
ties -> 48%; she is a single point of failure, team without her 12%), a
coin-flip DreamBreaker (gap over Bright/Johns 14.6 -> 8.6), and targeted
rivals (42% head-to-head at no season cost). Split gender caps make it
worse (82% vs reference; Bright and Johns go over cap too). Format
changes the title lottery (37% -> 14%) but never lifts a second team past
10%. **Auction draft DONE** (`auction_sim.py` -> `auction.md`,
`phase2_pricing.md` §11): same owners and personas, prices set by the
room (rotation nominations under the auction's own scarcity,
indifference-price ceilings, second price + $5k, ceilings capped at
budget minus the cheapest legal completion = $850k for a first buy).
Waters is not dented -- $850k at sale 1 in every quant cell and seed,
five floor teammates, 67-68% / 33-39% title (snake 66 / 36) -- but the
auction makes a CHASE: runner-up 11.5-16% (snake 7.3%), 1.8-3.0 teams
at 10%+ (snake 1.0). The chasers are one build a snake cannot make: two
$390-490k players plus four floor slots, a man and a woman (the tie
model pairs them in mixed, so a star plays three of the four games;
Patriquin + Rohrabacher 62% vs the field). The room re-prices the list's
middle (second star at 111-130% of list, depth at the floor, floored DB
specialists at 3-6x) and forgives individual persona mistakes while
punishing shared ones (twenty bargain hunters at $250k: Waters
$515-547k, her team 84-88%, spread 15-16). Expectations and noise are
second order once nominations run under the auction's own scarcity.
**Market limit DONE** (`market_eq.py` -> `market_eq.md`,
`market_prices.csv`; write-up in `auction.md` "The market limit",
`phase2_pricing.md` §12): the user's backward induction (owner 19
stops overpaying, then 18, then 17 ...) run to its fixed point with
owners who plan whole rosters (exhaustive 3M+3W search, brute-force
checked). Waters is the ONLY rationed player (at the $850k maximum
with excess demand, every run, both price rules); Bright is bid to
the price where a Bright team is average ($697-760k, not rationed),
Johns $501-545k, and every other star likewise; the other nineteen
teams are equals (second-best 50.6-51.1%, runner-up title 6-8%, one
team at 10%+). Her team 62-63% / 24-38%. List vs market: top 15 at
112-117% of list, #16-30 at list, #31-60 at 53-61% (27-28 of 120 at
the floor), pool total $19.84-19.98M vs $20.00M -- the list is right
in level and rank, the market steepens the shape. The auction's
two-star chase is a greedy-owner artefact (owners in `draft_sim` /
`auction_sim` project one slot at a time; a roster planner beats
them). "Buy whole teams" as a solver IS this solver: the
Waters+Johns+Bright+JW chain costs $2.29M at list / $2.65M at market
and falls back to one star plus fill-ins at 49%. `two_star.py` (auction.md
"Why the chasers buy a man AND a woman") decomposes the two-star
builds game by game vs the market-limit field: M+W 50-51%, M+M 44-46%,
F+F 52-54%, all $1.06-1.18M at market (over the cap) -- M+W because the
two stars share the mixed court (77%), the punt lands in MXD2 (30%)
not WD (19%), and floor men are DreamBreaker specialists.

**Toy auction solved exactly DONE** (2026-09-05 late; `toy_auction.py`
-> `toy_auction.md`, `phase2_pricing.md` §13): typed players, 2-4
strategic teams + a fixed Waters team, money grid, fixed sale order,
backward induction over the whole auction under two stage conventions
(english alternating-raise strict, second-price indifference bids).
Feasible range: 2 teams with one or two stars per gender, 3-4 teams
with one (up to ~4.4M states, 14 min for 4 teams); 3 teams with two
stars per gender (2,2,4,4 and 2,2,6,6) is out of reach in Python. Reads: allocation robust to the convention, prices
not; small rooms accommodate (stars split one per team, star teams no
better than the no-star team, demand reduction without a Waters team);
the truthful-planner benchmark is NOT the equilibrium at 2-3 teams
under either expectation; nobody buys two stars of one gender; sale
order is a first-order dial; a fourth bidder pushes star prices to the
list ($500k/$400k vs $300-400k at 3 teams) while the truthful planner's
prices stay at 2 units -- the N-trend argues against truthful planner
owners at twenty without a strategic layer. Numbers and caveats in the
file.

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
8. **Auction simulator** (`auction_sim.py`, user question "can we try an
   auction with the personas we have"): summary above, full grid and the
   hand-written reads in `auction.md`. A nomination defect was found and
   fixed on the way: nominations projected under the SNAKE's scarcity let
   the stars come up after the money was gone (Bright at sale 113 for
   $107k, JW Johnson unsold) and produced a "fragile to expectations" read
   (Waters' team 52-65%) that is RETRACTED. Now each owner nominates its
   snake pick under the rotation's own scarcity (gaps = (j+1) x active
   others), anything openable at the floor is nominatable, and the stars
   sell in the first ~13 sales at 0.95-1.1x list. Seeds 2 and 3 agree with
   seed 1 on the quant cells (67-69% / 32-40%). Truthful bidding, rotation
   nomination, no learning within an auction -- all stated in the
   docstring; the star rows carry what her other five cost (`mates_paid`,
   `mates_floor`).

## Next steps

1. ~~Owner personas in the draft sim~~ DONE (see above; `personas.md`).
   Left open: the marketing fame table is built from TRUE doubles rank
   (a leak toward the truth, stated in the write-up) -- a fame proxy
   from outside the model (social following, broadcast mentions) would
   make that persona honest. Personas as `Owner` subclasses
   (`draft_sim.Owner` hooks: beliefs, cap, score, filter_cands) are the
   reusable part; a mixed league of several personas at once is one
   `owner_factory` away.
1b. **Confirm MLP's rotation / playing-time rules** (Phase 0's open
   item, now the biggest known lever on the one-favourite problem):
   measured 2026 shows contenders' stars at 100% of matchups, so today
   there is no rotation; if MLP has or adds a minimum-starts rule, sweep
   it (share of ties the star plays) through the draft sim, and re-price
   (bench value rises). Also worth a cell each: DreamBreaker rule
   variants, and a "rival" persona that best-responds to the Waters
   roster (the spoiler exists; see `dials.md`).
1c. **Auction follow-ups** (`auction.md`): (i) the first-buy maximum
   (cap minus the cheapest legal completion) is the only rule that ever
   binds in a mixed room -- confirm whether MLP's real mechanism is a
   draft or an auction, and whether it carries a per-player maximum or
   a min-spend; (ii) run the rotation / playing-time sweep of 1b at
   auction too: the two-star build is the most rotation-sensitive roster
   in the league after Waters' own; (iii) strategic nomination and bid
   shading are the two mechanism gaps to close if the auction ever
   becomes the recommendation rather than a probe; (iv) the surplus
   column should expect the room's convex curve (second star and fit
   above list, depth at the floor); (v) **swap `market_eq.solve` (exact
   roster planner) into `draft_sim.Owner`'s projection** and re-run the
   persona and auction grids -- the two-star chase and the persona
   rankings were measured with greedy owners, and the market limit says
   the chase does not survive owners who plan whole rosters; (vi) the
   surplus column should use `market_prices.csv` (market and
   average-team price per pool player) as its second reference next to
   the list; (vii) `market_eq.py --noise` (belief noise) and a deeper
   `--k` are the two unswept dials on the market limit; (viii) **the
   strategic-auction search is BUILT** (2026-09-05, `strategic_auction.py`
   -> `strategic_auction.md`, phase2_pricing §14): planner owners, an
   8-dial strategy family (ceiling multipliers by tier incl. a
   no-star-team multiplier, price expectations incl. fictitious-play
   learned prices, greedy/planner projection, nomination policy,
   shortlist/all bidding), coordinate-ascent self-play to a symmetric
   dial equilibrium with the best remaining single-dial gain as the
   exploitability. Validated on `toy_auction.py`'s solved cells on
   ROSTERS under both conventions: MATCH only at 2 teams, DIFFER at 3-4
   (the SPE's asymmetric hold-backs and deep-pocket buys by the no-star
   team are outside a symmetric dial family; coarse money grid) -- so
   20-team reads are the FAMILY's equilibrium, not the game's. Real
   board from the truthful start: `auction_sim`'s owner is exploitable
   by 8 pp (learned prices) then 20 pp (roster planner); the search
   CYCLES on the no-star multiplier (1.0 <-> 2.0) at a_star 1.25 /
   learned / planners, exploitability ~10-12 pp; Waters $850k at sale 1
   and 67-68% / 37% at every node (no dial moves her); the middle is
   repriced (top 5 116% of list, #31-60 75%, 22 floor-priced vs 39) and
   the two-star chase SURVIVES planners in a sequential room (2.1 teams
   at 10%+, second-best 58.7%) -- the market limit's "chase dies"
   holds at market-fixed-point prices, which a rotation-nominated room
   does not reach. Unswept: `--noise` (belief dispersion), nomination
   dials from the truthful start (snake stayed), deviator slot
   asymmetry, and the toy's own lesson -- a non-symmetric (role-typed)
   profile search is the next instrument if anyone wants the game's
   equilibrium rather than the family's. Two cheap probes from the
   plan (price-enforcement free-rider null, drain nomination only
   around Waters/Bright) are still unrun.
2. **Phase 3 follow-ups**: the league-price + surplus column once MLP
   publishes prices (the page already says it is coming); a per-player
   line on player pages ("value price $X, #N among women"); an insights
   article on the tag rule and the one favorite. Optional: LLM owners as
   persona GENERATORS for item 1 (the user asked; judged not worth it as
   drafters — repeated play in the sim answers the learning question
   cheaper).
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
  `--drafts`, `--seasons`, `--noise`; `set_board(mode, pool)` for scripted use;
  `Owner` hooks (beliefs `dbl`/`sgl` + `rebuild()`, `cap`, `score(proj)`,
  `filter_cands`) and `run_variant(..., owner_factory=)` for persona leagues.
- `personas.py` → `personas.md` — the five persona classes + `franchises_2026()`
  (pid → modal 2026 MLP team); `--only`, `--counts`, `--rerender`;
  `concentration(r)` = title-odds concentration of a run_variant result.
- `dials_probe.py` → `dials.md` — DB coin flip, availability, measured 2026
  playing time, rival best response, split caps, season formats.
- `auction_sim.py` → `auction.md` — `run_auction(...)`, `run_variant(...,
  expect="list"|"inflated", owner_factory=)`, `render`; CLI `--expect list
  inflated`, `--counts`, `--only`, `--perfect` (quant room at noise 0),
  `--rerender` (rows cached in `cache/auction_rows.pkl`, gitignored).
  Nominations = the owner's snake pick under the auction's scarcity
  (`nominate`), ceilings by bisection (`ceiling`), stranded fallback.
- `pool_floor_sweep.py` — `phi_pool(P)`, `run_cell(P, floor, pool, ...)`.
- `draft_strategies.py`, `simulate_templates.py` — template rosters and
  their season.
- `phase1_value_model.py` — unchanged; still the exact tie model.
