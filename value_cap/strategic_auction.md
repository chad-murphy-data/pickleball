# Strategic auction: dial agents, self-play, and what the toy says about them (2026-09-05; `strategic_auction.py`)

**Question.** `auction_sim.py`'s owners bid truthfully off a greedy
projection; `market_eq.py` showed a whole-roster planner beats them;
`toy_auction.py` solved small rooms exactly and found that truthful
planners are *not* the equilibrium and that a 20-team equilibrium is out of
computational reach. HANDOFF item 1c(viii) asked for the next honest
instrument: planner owners, a small family of strategies (dials), self-play
to a symmetric equilibrium in the dial space, and the size of the best
remaining deviation -- validated on the toy before anything runs at twenty.

**Short answer.** _(filled in below once the runs finish)_

## Instrument

Every owner plays the same dial vector theta (a symmetric profile). Dials:

| dial | values | meaning |
|---|---|---|
| `a_star`, `a_good`, `a_depth` | 0.6 .. 2.0 | multiplier on the indifference ceiling, by the tier of the player on the block (list rank within gender: 1-5 star, 6-30 good, 31+ depth) |
| `a_good0` | 0.6 .. 2.0 | the good-tier multiplier used INSTEAD of `a_good` by a team holding no star (role-conditional bidding, see the toy) |
| `expect` | list, inflate, rivals, learned | prices assumed for players not yet bought: the cheat sheet; the sheet scaled by money-left / value-left (`auction_sim`'s `inflated`); the sheet capped at what the richest rival could still pay; the prices the room paid last time (fictitious play -- an all-learned profile is iterated to a price fixed point, a deviator expects the incumbent room's prices) |
| `plan` | greedy, planner | the projection behind the indifference ceiling: `auction_sim`'s one-slot-at-a-time fill, or the exhaustive completion of the roster held so far (`market_eq.solve` generalised to a partly filled roster; K=10 candidates per gender, planner ceilings for list rank <= 40, greedy beyond) |
| `nom` | snake, want, cheap, drain, dear | (real board) what to nominate: `auction_sim`'s snake pick under the auction's own scarcity / the dearest player in my plan / the cheapest / the dearest player NOT in my plan (make rivals spend) / the dearest available |
| `bid` | shortlist, all | (real board) bid only on players I would shortlist for a snake pick (`auction_sim`'s rule: top 30 by value per gender, top 6 singles, 5 cheapest) / bid on everyone |

Mechanism as in `auction_sim` and the toy's `secondprice`: sealed ceilings,
highest wins at the second-highest + $5k (one unit on the toy), nominator
opens at the floor, nobody bids past budget minus the cheapest legal
completion ($850k first buy), rotation nomination. Payoff = mean tie
probability against the other teams (the toy includes the fixed Waters
team). Self-play = coordinate ascent: every single-dial deviation is
evaluated with the deviator's slot rotating over the seeds (paired against
the incumbent room at the same seed), the best gain is adopted if it clears
the tolerance (and, on the real board, twice its standard error over seeds),
until nothing improves; the best remaining single-dial gain is the
exploitability. Starts: `truthful` = `auction_sim`'s owner transplanted
(greedy, list, snake nomination, shortlist bidding -- it reproduces that
room: Waters $850k at sale 1 with five floor teammates and a 69% team,
Bright $632k, Johns $507k), and `planner` (planner, list, nominate my
dearest planned player, bid on everyone).

Cost: one 20-team auction is ~2 s with greedy owners, ~36 s with twenty
planners (K=10, planner ceilings for the top 40 list ranks per gender); a
real-board self-play round is 27 deviations x 8 seeds = 216 auctions
(2.5-4 min with greedy owners, ~45 min once the profile is all-planner).
On the toy a deviator sits in every slot in turn and the mean is exact,
so the se gate is off there.

## Toy validation (rosters, both conventions)

The toy (`toy_auction.py`) is the only place the true equilibrium is
known, so this is the calibration: run the same self-play on each solved
cell and compare what the dial family lands on against the exact
subgame-perfect equilibrium (SPE) of the same auction. Rosters are the
target, per `toy_auction.md` (allocation is robust to the stage
convention, prices are not). Board `a,b,c,d` = star women, star men, good
women, good men; fill-ins are free at the floor; the Waters team is fixed
and outside the search; unit = money grid.

| cell | start | dial equilibrium | vs exact second-price | vs exact english |
|---|---|---|---|---|
| 2 teams, 1,1,4,4, $50k | truthful | a_star 1.25, list, greedy (converged) | rosters MATCH, star prices MATCH (SF 15, SM 2) | rosters differ (star split same) |
| 2 teams, 1,1,4,4, $50k | planner | a_star 2.0, inflate, planner (converged) | rosters MATCH, star prices MATCH | rosters differ |
| 2 teams, 2,2,4,4, $50k | truthful | a_star 0.8, list, greedy (converged) | rosters MATCH (stars split 1+1 each), SF 7/SM 4 vs exact 6/5 | rosters differ |
| 2 teams, 2,2,4,4, $50k | planner | a_star 1.25, list, planner (converged) | rosters MATCH, SF 6/SM 6 vs exact 6/5 | rosters differ |
| 3 teams, 1,1,3,3, $100k | truthful | learned prices, greedy (converged) | rosters DIFFER; star split and star prices MATCH (SF 4, SM 3) | differ |
| 3 teams, 1,1,3,3, $100k | planner | a_star 1.25, list, planner (converged) | rosters DIFFER; star split MATCH; SF 3/SM 3 vs 4/3 | differ |
| 3 teams, 1,1,2,2, $50k | truthful | CYCLE (list -> learned -> planner -> inflate -> a_star 1.25 -> back) | rosters, prices and star split all DIFFER | differ |
| 3 teams, 1,1,2,2, $50k | planner | max rounds at a_star 2, a_good 1.5, a_good0 0.8, inflate, planner | all DIFFER | differ |
| 4 teams, 1,1,3,3, $100k (known SPE) | truthful | CYCLE (a_good0 1.5 <-> 1.0) | rosters DIFFER | -- |
| 4 teams, 1,1,3,3, $100k (known SPE) | planner | a_star 1.25, learned, planner (converged) | rosters DIFFER | -- |

Read: the dial family reproduces the exact allocation only in the
two-team rooms. There the search also finds the right prices from a
truthful start (2,2,4,4 lands within one grid unit), and the two starts
reach the same rosters and payoffs by different dial settings (a_star 1.25
greedy vs a_star 2.0 inflate planner), which is the expected
non-identification: several dial vectors are the same bidding function on
a small board. From three teams up every cell differs from the SPE on
rosters, and the two starts usually land in different places.

Why it differs, from the paths:

- **3 teams 1,1,3,3 ($100k).** The SPE has the no-star team (T0) buying
  three good men plus a good woman with deep pockets (8 of 10 units) and
  the room ends balanced at 44.8 / 43.3 / 45.0. The dial search from both
  starts hands the star man's team a good man too and ends 40 / 58 / 39
  (truthful) or 38 / 58 / 41 (planner): the star-man team runs away. Star
  split and star prices are right; what the family cannot express is the
  SPE's role-specific aggression of the team that missed the stars. That
  is exactly what `a_good0` was added for, and on this cell it does not
  bite: at list expectations a planner's ceiling on a good man is two
  units on the $100k grid, and the rotation tie-break on the coarse grid
  decides who gets him, not the multiplier.
- **3 teams 1,1,2,2 ($50k).** The exact second-price SPE puts BOTH stars
  on one team (SF at 11, SM at 2; payoffs 31.6 / 66.0 / 33.9): T1 buys the
  star woman near her max and then takes the star man cheaply because
  the other two teams, who planned around the woman, no longer want him
  at any price above two units. The dial search never produces that
  hold-back: symmetric profiles either all fight for the star man (stars
  at 10 and 10 from the truthful cycle, 15 and 15 from the planner run)
  or all defer. The truthful start cycles through the price-expectation
  dials because each expectation is best only against the previous
  one -- there is no symmetric fixed point in this family on this board.
- **4 teams 1,1,3,3 ($100k).** The known SPE has one team on three good
  men and no women beyond fill-ins. The planner-start equilibrium
  (a_star 1.25, learned prices, planner) instead spreads the good players
  and the two star teams end 34% (star woman, no good men) and 53% (star
  man), with a goods-only team at 54%. The truthful start cycles on the
  no-star multiplier: a_good0 1.5 is best against 1.0 and 1.0 is best
  against 1.5.

Two structural reasons, both visible above. (1) The SPE is asymmetric:
after the first sale the room is in different positions and the exact
solution has each team best-respond to its own position, including
credible hold-backs and deep-pocket buys by whoever missed a star. A
symmetric dial profile can only condition on role through `a_good0`, one
number for one situation. (2) The money grid is coarse: at $50-100k units
a good player's whole ceiling range is one or two units, so rotation
tie-breaks (who nominated, who bid first) allocate more than the dials
do. Both matter less at twenty teams with $5k increments, but the toy
cannot show that; it can only show that the family is not the game's
equilibrium once three or more teams bid.

What this licenses for the real board: the 20-team self-play below finds
a symmetric equilibrium of the DIAL FAMILY and the size of the best
single-dial deviation from it. It is a strategy-search result about
`auction_sim`'s owner population, not the auction's equilibrium. Rosters
and prices it produces are where such a room ends up, with the toy saying
the true equilibrium can be elsewhere in exactly the 3+-team direction:
more role-specific aggression from teams that miss the stars.
