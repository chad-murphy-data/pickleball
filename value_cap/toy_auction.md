# The toy auction, solved exactly (2026-09-05; `toy_auction.py`)

**What this is.** The 20-team auction cannot be solved for an equilibrium
(auction.md, "The market limit"; the reasons are the state space, the
exposure problem, budgets coupling every sale, equilibrium multiplicity and
PPAD-hardness). This toy CAN be: typed players (star / good / floor per
gender, each type carrying its tier's mean doubles value, singles value,
uncertainty and list price; the production FastTie on those representative
players), a few strategic teams, one deterministic Waters team (Waters +
five floor players, not in the auction), money on a $50k or $100k grid,
a fixed sale order, and an English auction played as an alternating-raise
game with perfect information -- solved by backward induction over the
whole auction. Payoff = in-league win% (mean tie probability against every
other team, the Waters team included). Unspent money is worth nothing.

The types, from the pool (values on the per-point logit scale):

| type | doubles | singles | list | who |
|---|---|---|---|---|
| star man (SM) | +1.07 | +1.68 | $427k | Johns, JW Johnson, Patriquin, Tardio, Alshon |
| good man (GM) | +0.82 | +1.37 | $182k | #6-30 |
| floor man (FM) | +0.55 | +1.30 | $37k | Truong, Joseph, Emmrich |
| star woman (SF) | +1.18 | +1.50 | $488k | Bright, Jorja Johnson, Jade Kawamoto, Todd, Pisnik |
| good woman (GF) | +0.90 | +1.17 | $235k | #6-30 |
| floor woman (FF) | +0.60 | +1.04 | $39k | Padegimaite, Erokhina, Schull |
| Waters (W) | +1.80 | +2.52 | $769k | fixed team, not auctioned |

Two stage mechanisms, because a complete-information English auction has a
continuum of equilibria and the solver has to pick one:

- `english` (default): alternating raises in rotation, a team raises only
  when strictly better off, the standing bidder wins when everyone else
  passes in a row. Selects the LOWEST-price equilibrium: a loser never
  runs the winner up unless draining them is strictly better for the
  loser later. `--tie aggressive` (raise when weakly better) is the other
  end of that rule.
- `secondprice`: each eligible team bids its indifference price against
  what happens if it abstains (computed with the equilibrium continuation
  values, by induction on the bidder set); the highest ceiling wins at
  the second-highest + 1 unit. Selects the truthful equilibrium. This is
  the convention `auction_sim.py`'s owners use, with exact values.

One-item sanity check (2 teams, one star woman): english gives her to the
opener at the reserve, secondprice at the first-buy maximum -- both as the
theory says.

## What it costs (measured; 4 cores, pure Python)

| teams | board SM,SF,GM,GF | unit | mechanism | between-sale states | in-sale states | time |
|---|---|---|---|---|---|---|
| 2 | 1,1,4,4 | $50k | english | 510k | 11.8M | 31 s |
| 2 | 1,1,4,4 | $50k | secondprice | 552k | 2.0M | 21 s |
| 2 | 2,2,4,4 | $50k | english | 1.85M | 37.7M | 103 s |
| 3 | 1,1,3,3 | $100k | english | 401k | 12.1M | 28 s |
| 3 | 1,1,3,3 | $100k | secondprice | 406k | 3.2M | 27 s |
| 3 | 1,1,3,3 goods first | $100k | either | 150k | 1-4M | 12 s |
| 3 | 1,1,2,2 | $50k | english | 994k | 72.9M | 129 s |
| 4 | 1,1,3,3 | $100k | secondprice | 4.37M | 68.9M | 829 s |
| 3 | 2,2,4,4 | $100k | secondprice | > 6.8M / 47M in-sale at 14 min (~8k states/s, > 1 GB), process died without a result -- out of reach | | |
| 3 | 2,2,4,4 | $50k | english | > 1.8M in 2 min, killed | | |
| 3 | 2,2,6,6 | $100k | english | > 1.4M in 2 min, killed | | |
| 3 | 2,2,6,6 | $50k | english | > 2M in 3 min, killed | | |

Memory is ~200 bytes per between-sale state plus the in-sale memo, so
a few million states fit in a gigabyte. The count is driven by the money
grid (every combination of what the teams have spent is a distinct
state) times the typed holdings; halving the unit roughly triples the
count, one more team at the same board multiplies it several-fold, and
the 2,2,6,6 board with 3 teams -- the size I first proposed -- is out of
reach in Python at either grid. The honest working range is 2-3 teams
with one or two stars per gender, 4 teams with one.

## What the equilibria look like

Prices in grid units (unit $50k: budget 20, star maximum 15; unit
$100k: budget 10, maximum 5). Win% = in-league, the Waters team included
where present. "Bench" = the truthful-planner benchmark on the same toy
(ceiling = indifference vs a reference roster at expected list-price
types, second price + 1; what `auction_sim.py` owners do, with exact
roster planning).

| run | stars sold at | rosters (strategic teams) | win% | Waters team |
|---|---|---|---|---|
| 2 teams, 1,1,4,4, $50k, english | SF 1, SM 1 | one star each (T0 the woman, T1 the man), goods, money left on the table | 57.4 / 55.2 | 37.5 |
| same, english aggressive | SF 15, SM 1 | identical rosters | 57.4 / 55.2 | 37.5 |
| same, secondprice | SF 15, SM 2 | identical rosters | 57.4 / 55.2 | 37.5 |
| same, bench (list expectations) | SF 5, SM 4 | identical playing lineups, more goods bought | 57.4 / 55.2 | 37.5 |
| 2 teams, 2,2,4,4, $50k, english | SF 5-6, SM 1 | each team one star man AND one star woman | 64.4 / 64.4 | 21.2 |
| same, bench | SF 5, SM 4 | T0 two star women + goods, T1 two star men + goods | 65.4 / 58.8 | -- |
| 3 teams, 1,1,3,3, $100k, english | SF 4, SM 4 | T2 the woman, T1 the man, T0 no star (3 goods) | 46.5 (no star) / 40.9 / 45.4 | 67.2 |
| same, secondprice | SF 4, SM 3 | same split | 44.8 (no star) / 43.3 / 45.0 | 66.9 |
| same, bench (either expectation) | SF 2, SM 2 | T0 the woman, T1 the man, T2 stranded on goods | 60.3 / 57.7 / 20.3 | 61.6 |
| 3 teams, 1,1,3,3, goods first, secondprice | SF 4, SM 2 | T0 buys goods early, then the star woman | 59.2 / 41.0 / 37.4 | 62.5 |
| same, english | SF 3, SM 1 | one star each for T0, T1; T2 goods | 41.3 / 45.8 / 45.4 | 67.6 |
| 3 teams, 1,1,3,3, secondprice, NO Waters team | SF 2, SM 5 | T0 gets the star woman for $200k, others' ceilings 1 | 66.5 / 43.0 / 40.5 | -- |
| 3 teams, 1,1,2,2, $50k, english | SF unsold, SM 9 | nobody takes the star woman at the reserve | 42.8 / 41.1 / 38.1 | 78.0 |
| 4 teams, 1,1,3,3, $100k, secondprice | SF 5, SM 4 (= list) | T1 the man + a good woman, T2 the woman + five floors, T0 two good women, T3 three good men | 41.6 / 56.0 / 37.9 / 38.7 | 75.8 |
| same, bench (list expectations) | SF 2, SM 2 | T0 the woman, T1 the man, T2 and T3 stranded on goods | 53.5 / 53.3 / 34.8 / 34.8 | 73.6 |
| same, bench (equilibrium expectations) | SF 3, SM 2 | T1 the man + goods at 69%, T2 stranded at 20% | 52.9 / 69.4 / 19.7 / 35.4 | 72.6 |

## What it says

1. **Allocation is robust to the equilibrium concept; prices are not.**
   In the 2-team toy the three conventions (english strict, english
   aggressive, second-price) produce the SAME rosters and the same win%,
   and sell the star woman for $50k, $750k and $750k respectively. In a
   complete-information sequential English auction the price is a
   selection, not a prediction. Any validation of a heuristic strategy
   search against this toy must be scored on rosters, not prices.
2. **Small rooms accommodate.** With three teams and one star per gender
   (stars first), both conventions split the stars one per team, sell
   them at $300-400k against a $427-488k list, and leave the star teams
   NO better off than the team that bought three goods (43-46% each, the
   market limit's "equals" already at n=3). With no Waters team the star
   woman goes to the opener for $200k because the other two teams' true
   ceilings are one unit: contesting her leads to a worse continuation
   for them than letting her go. That is demand reduction, the textbook
   small-number result, and it is exactly what a 20-team room dilutes --
   the toy overstates it, as expected.
3. **The truthful planner is not the equilibrium at 2 or 3 teams.** The
   benchmark (what `auction_sim.py` owners do, with exact roster
   planning) overpays the english-strict price 4-5x and underpays the
   second-price 3x at 2 teams; at 3 teams it underprices both stars 2x
   and strands one team on goods at 20% -- with EITHER expectation
   (list prices, or the equilibrium's own prices). Correct price
   expectations do not rescue truthful bidding in a small room; the
   missing piece is the strategic response, not the forecast.
4. **Nobody chases two stars of one gender.** With two stars per gender
   and two teams, each team takes one man and one woman (64.4% each,
   Waters' team 21%), the same M+W shape `two_star.py` found in the real
   room. The 3-team, two-stars-per-gender case is out of reach (the
   solve died past 6.8M states without a result), so the M+W shape is
   established here at two teams only.
5. **Sale order is a first-order dial.** Goods first hands the opener a
   59% team at second-price (buy goods while they are cheap, then the
   star woman at 4 with a roster that makes her worth it); stars first
   gives three equals. The room's nomination order is therefore not a
   detail, and it is the one strategic dimension this toy deliberately
   fixes.
6. **Strict-raise english produces odd paths.** Goods go unsold at the
   reserve and are bought later at higher prices; in the 1,1,2,2 board
   the star woman goes UNSOLD at $50k because the opener's continuation
   after taking her is worse than the continuation without her. These
   are genuine subgame-perfect outcomes of the lowest-price selection,
   and they are the reason to report both conventions rather than one.

7. **A fourth bidder pushes star prices to the list.** At 4 teams and
   one star per gender (second-price) the stars sell at exactly their
   list prices ($500k / $400k) where 3 teams paid $300-400k: the
   accommodation of finding 2 weakens as bidders are added, which is
   the direction the 20-team market limit needs. The split is no longer
   equal, though -- the star man's buyer adds a good woman and wins 56%,
   the star woman's buyer is left with five floor players at 38%, below
   the two no-star teams (42%, 39%). The truthful-planner benchmark does
   NOT converge toward this as teams are added: at 4 teams it still buys
   both stars at 2 units and strands two teams at 35% (list
   expectations) or one at 20% (equilibrium expectations). Three points
   on the N-trend (2, 3, 4) say prices move toward the list and the
   benchmark's prices do not move at all.

## Caveats

Identical, perfectly informed teams on true values; typed players (no
within-tier differences, so the "second star" is exactly as good as the
first); fixed sale order (nomination is not a choice); payoff = win%
(no title nonlinearity); the two stage conventions bracket a continuum
of equilibria; 2-4 teams overstate every strategic effect relative to
twenty. The toy is a check on METHOD (does a heuristic search find these
rosters?) and on DIRECTION (what few-bidder strategy looks like); it is
not a forecast of MLP's room.

## Next

- Score a strategy search (fictitious play over a small dial set) on
  this toy: it passes if it reproduces the equilibrium ROSTERS at 2 and
  3 teams under both conventions.
- The N-trend at one star per gender is now measured at 2, 3 and 4
  teams (finding 7): equilibrium star prices rise toward the list, the
  truthful-planner benchmark's do not. That is the case AGAINST trusting
  truthful planner owners at twenty without a strategic layer; the
  strategy search above is the fix.
- A faster solver (the between-sale recursion in a compiled loop) would
  buy one more team or one more star per gender, not twenty teams.
