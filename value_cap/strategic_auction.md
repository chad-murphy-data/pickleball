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

## The real board (20 teams, `mlp2026` board, list expectations = the shipped list, noise 0)

### From the truthful start (8 seeds, 8 rounds allowed, 4.5 h)

Path, with the best single-dial gain at each step (pp of mean tie
probability for the deviating team; se over the 8 seeds):

| round | profile | best deviation | gain |
|---|---|---|---|
| 0 | truthful: list, greedy, snake, shortlist (= `auction_sim`) | expect=learned | +8.0 (se 2.1); inflate +6.5, a_star 1.25 +6.3, nom=cheap +5.9 |
| 1 | learned, greedy | plan=planner | +20.2 (se 4.2); inflate +16.0, rivals +12.7, list +11.1 |
| 2 | learned, planner | a_star=1.25 | +22.7 (se 2.8); every other expectation and plan=greedy +20-21 |
| 3 | a_star 1.25, learned, planner | a_good0=2.0 | +12.2 (se 1.2); a_good0 1.25-1.5 +11.8, rivals/list +11.4 |
| 4 | a_star 1.25, a_good0 2.0, learned, planner | a_good0=1.0 | +9.6 (se 4.4); a_good0 0.6-0.8 +8.4 |

Round 4 returns to the round-3 profile: **CYCLE**, no symmetric
equilibrium in the dial family from this start. The two nodes are
"a_star 1.25, learned prices, planners, snake nomination, shortlist
bidding" with the no-star multiplier at 1.0 or 2.0; the exploitability at
either node is ~10-12 pp for a single deviating team, and it is the same
dial both ways: a team that missed the stars wants to bid twice the
indifference price for good players when nobody else does, and wants to
stop when everybody does. That is the toy's 4-team truthful cycle
(a_good0 1.5 <-> 1.0) reappearing at twenty.

The room at the reported node (a_good0 1.0; all clones, 8 seeds), next to
`auction_sim`'s truthful room:

| | truthful room (`auction_sim`) | dial-cycle node |
|---|---|---|
| price / list, #1-5 | 1.06 | 1.16 |
| price / list, #6-15 | 1.21 | 1.05 |
| price / list, #16-30 | 1.06 | 0.99 |
| price / list, #31-60 | 0.53 | 0.75 |
| Waters price | $850k (sale 1) | $850k (sale 1) |
| Waters' team win / title | 68.8% / 37% | 67.5% / 37% |
| second-best team | 59.7% | 58.7% |
| teams at 10%+ title | 2.6 | 2.1 |
| parity spread (pts) | 7.3 | 6.0 |
| floor-priced players (of 120) | 38.6 | 21.8 |
| unspent per team | $4k | $44k |

Reads:

- **Waters is not dented by strategy either.** At the cap's first-buy
  maximum at sale 1, five floor teammates, 67-68% and a 37% title shot,
  in the truthful room, at every node of the search and in every
  deviation table (no dial ever moved her price or her team). She is
  rationed by the rule, not the room, and no strategy layer changes that:
  it is the same read as `market_eq.md` and `auction.md`, now with owners
  who plan whole rosters and bid strategically.
- **The strategic layer moves the middle, not the top.** Planners with
  learned prices push the top 5 from 6% to 16% over list (Bright $774k
  vs $697-760k in the market limit; Johns $614k), pull the second tier
  back to list (105% vs 121%), and lift depth from 53% to 75% of list:
  fewer players at the floor (22 vs 39), and about $44k per team unspent
  because ceilings on depth players are planner-exact and the shortlist
  rule keeps most owners out of those sales. The shape is the market
  limit's shape (`market_prices.csv`: top 15 at 112-117%, #31-60 at
  53-61%) with a flatter tail.
- **The two-star chase survives planners here.** The seed-0 room at the
  cycle node has Pisnik + Rohrabacher ($390k + $389k, 60.8%), Sewing +
  Black ($365k + $363k, 53.9%), Jade Kawamoto + Daescu ($440k + $338k,
  54.4%), Alshon + Jackie Kawamoto ($486k + $394k) -- the second-best
  team is 58.7% and 2.1 teams sit at 10%+ title odds, against 1.0 in the
  snake. `market_eq.md` said the chase does not survive roster planners;
  what it does not survive is planners at MARKET prices (the fixed point
  where every second star is bid to the price that makes their team
  average). In an actual auction with sequential sales and rotation
  nomination, the second stars come up before their buyers' money is
  gone and the room does not reach that fixed point; two of them for
  ~$750k is still the best no-Waters build. The chasers are not all
  man + woman here (Pisnik + Rohrabacher is two women, Sewing + Black
  two women), which is the planner's completion finding a cheap man to
  pair with in mixed; the M+W shape from `two_star.py` is the greedy
  owner's version.
- **What the dials say about `auction_sim`'s owner.** Its truthful
  ceiling is exploitable by 8 pp at round 0 and its greedy projection by
  20 pp at round 1: a single roster planner in a room of nineteen
  truthful greedy owners gains a fifth of a tie probability. So the
  persona and auction grids measured with greedy owners are not
  equilibrium reads (as HANDOFF 1c(v) suspected). But once the room is
  planners with learned prices the remaining single-dial gains are the
  role-conditional a_good0 flip-flop and ~7 pp on the star multiplier,
  and none of them touch Waters' team.

Caveats specific to this run: 8 seeds, so a deviation gain under ~5 pp is
inside its own error bar (the round-4 table is all se 3-5); the deviator
sits in slot (7 x seed) mod 20, so its position in the nomination
rotation varies with the seed; noise 0 (every owner shares the true
values), so the room has no belief dispersion at all -- `auction_sim`'s
10% noise was not swept here for cost.
