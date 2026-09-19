# Strategic auction: dial agents, self-play, and what the toy says about them (2026-09-05; `strategic_auction.py`)

**Question.** `auction_sim.py`'s owners bid truthfully off a greedy
projection; `market_eq.py` showed a whole-roster planner beats them;
`toy_auction.py` solved small rooms exactly and found that truthful
planners are *not* the equilibrium and that a 20-team equilibrium is out of
computational reach. HANDOFF item 1c(viii) asked for the next honest
instrument: planner owners, a small family of strategies (dials), self-play
to a symmetric equilibrium in the dial space, and the size of the best
remaining deviation -- validated on the toy before anything runs at twenty.

**Short answer.** Built (`strategic_auction.py`), validated on the toy,
run at twenty from both starts. The toy says the symmetric dial family
reproduces the exact equilibrium's rosters only in two-team rooms; from
three teams up the true equilibrium has asymmetric moves (hold-backs,
deep-pocket good-player buys by whoever missed the stars) that one shared
dial vector cannot express, so the 20-team result is the FAMILY's
equilibrium, not the game's. On the real board `auction_sim`'s truthful
greedy owner is exploitable by 8 pp (learn last time's prices) and then
20 pp (plan the whole roster); the search from that start cycles on the
no-star team's good-player multiplier with 10-12 pp of single-dial gain
left, and from the planner start it walks back to greedy + list with
~7 pp left -- there is no low-exploitability symmetric point in this
family. What does not move under any of it: Waters at the cap's $850k
maximum at sale 1, her team 67-69% / 33-37% title, a second-best team at
58.7-59.7%, 2-3 teams at 10%+ title odds. Strategy reshapes the middle of
the price curve (whether the top 5 or the second star carries the
premium; how much depth sells at the floor), and the two-star chase
survives roster planners in a sequential room -- the market limit's
"chase dies" holds at its fixed-point prices, which a rotation-nominated
auction never reaches.

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

### From the planner start (4 seeds, 4 rounds allowed, 45 min)

Start = whole-roster planners with list expectations, nominating their
dearest planned player, bidding on everyone.

| round | profile | best deviation | gain |
|---|---|---|---|
| 0 | planner, list, want, all | nom=snake | +6.6 (se 2.9); inflate +6.4, nom=dear +3.5 |
| 1 | planner, list, snake, all | plan=greedy | +9.6 (se 4.2); a_star 1.25-2.0 +9.4, learned +9.1 |
| 2 | greedy, list, snake, all | inflate +7.5 (se 5.2) | fails the 2-se gate: stop, "converged" |

So from the planner start the search walks BACK to `auction_sim`'s owner
(greedy projection, list expectations, snake nomination), differing only
in bidding on everyone rather than the shortlist. The stop is a gate
artifact, not a fixed point: the rule tests only the single best gain
(inflate, se 5.2 on 4 seeds) and the second-best, expect=learned at
+7.4 (se 2.9), would have cleared it. Read the exploitability at this
profile as ~7 pp, and the two starts as disagreeing about where the
family settles: planners + learned prices cycling on the no-star
multiplier from one side, greedy + list from the other, each with 7-12 pp
of single-dial gain left. The dial family has no low-exploitability
symmetric point on this board; both runs stopped at whatever the
adoption rule let through.

Room at the planner-start's end point (all clones, 4 seeds): top 5 at
106% of list, #6-15 at 120%, #16-30 at 107%, #31-60 at 54%, 36.5
floor-priced, $6k unspent -- `auction_sim`'s room again (106 / 121 / 106 /
53). Waters $850k at sale 1, 67.2% / 33% title, second-best 59.5%,
3.0 teams at 10%+, spread 6.8. Seed 0 has Bright + Garnett 61.4%, JW
Johnson + Glozman + Leigh Waters 58.1%, Alshon + Oncins + Dennehy 58.2%,
Johns + Parenteau 55.8%: the second stars sold at $414-632k with a
$225-330k partner, the chase again.

The STARTING room is worth one line because it is the failure mode of a
naive planner league: twenty planners at list expectations nominating
what they want and bidding on everyone leaves $316k per team unspent,
prices #31-60 at 121% of list with 1.5 players at the floor, and Waters'
buyer -- forced to five floor slots -- gets the dregs: her team 39%,
title 0.5%. Planner-exact ceilings at list prices are conservative on
stars and generous on depth, and the room never corrects because nobody
learns within the auction. It is not an equilibrium (snake nomination
gains 6.6 pp at once) and no run ended there, but it is the shape a
room of over-careful quants would produce, and it is the only room in
this whole project where Waters' team is not the favourite.

## Price-order nomination (league rule; in-session 2026-09-05, 8 seeds, noise 0)

Question (user): what if nominations were in PRICE order -- the dearest
remaining player by list price is always the next sale, nobody chooses?
In the instrument that is `nom=dear` frozen as a rule: `--room` reports
the room at a profile, `--fix nom --start nom=dear` runs the self-play
with the dial removed from the search. Two owner types, each held fixed
across the rule change (the bidding rule must be held fixed too; see
the shortlist caveat below):

| | greedy owners, snake (= `auction_sim`) | greedy owners, PRICE ORDER | planner owners, own choice | planner owners, PRICE ORDER |
|---|---|---|---|---|
| price / list, #1-5 | 1.06 | 1.10 | 0.96 | 0.91 |
| price / list, #6-15 | 1.21 | **1.31** | 0.98 | 0.97 |
| price / list, #16-30 | 1.06 | 0.92 | 1.01 | 1.11 |
| price / list, #31-60 | 0.53 | 0.49 | 1.21 | 0.99 |
| Waters price | $850k, sale 1 | $850k, sale 1 | $850k, sale 1 | $850k, sale 1 |
| Waters' team win / title | 68.8% / 37.6% | 66.2% / **30.2%** | 39.3% / 0.5% | 56.5% / **7.1%** |
| best team | 68.8% (hers) | 66.4% | 61.7% | **64.7% (not hers)** |
| second-best team | 59.7% | **61.6%** | 57.8% | **63.2%** |
| teams at 10%+ title | 2.9 | **3.9** | 3.5 | 3.75 |
| parity spread (pts) | 7.3 | 8.0 | 5.9 | 7.4 |
| floor-priced players (of 120) | 38.6 | 45.3 | 1.5 | 15.1 |
| unspent per team | $4k | $8k | $316k | $56k |

(greedy = `a_star 1, list, greedy, shortlist`, price order with
`bid=all` gives IDENTICAL numbers -- with the dearest player always on
the block the shortlist filter never excludes a contested sale; planner
= `plan=planner, list, bid=all`, own choice = `nom=want`.)

Reads:

- **Waters' price does not move.** She is sale 1 under both rules
  (owner-chosen nomination already puts her up first) and the $850k
  first-buy maximum is the binding constraint, not the order. Price
  order changes what happens AFTER her.
- **It widens the chase.** With `auction_sim`'s greedy owners the
  second-best team goes 59.7% -> 61.6% and the count of teams at 10%+
  title odds 2.9 -> 3.9; her title share falls 37.6% -> 30.2%. The
  second-star tier is where the money goes: #6-15 sell at 131% of list
  (121% under snake). Mechanism: the nineteen owners who missed her
  face the next-dearest players immediately, all with full budgets, so
  every star sale is contested by everyone; under snake nomination the
  stars trickle out between cheaper picks and some go uncontested.
  Depth pays for it: #16-30 slip to 92% of list and 45 players sell at
  the floor (39 under snake), and 1 pool player goes unsold (the
  cheapest come last, when rosters are full).
- **With planner owners it makes the two-star build the favourite.**
  Seed 0: Jade Kawamoto + JW Johnson 65.3%, Parris Todd + Kate Fahey
  64.1%, Tina Pisnik + Gabriel Tardio 62.5% -- all ahead of Waters +
  five floor players at 59.3%. Over 8 seeds her team is 56.5% with a
  7% title shot; the best team is 64.7% and the second 63.2%. Stars
  sell to planners BELOW list (#1-5 at 91%, #6-15 at 97%) and the
  mid-tier ABOVE (#16-30 at 111%): a whole-roster planner knows the
  star's price only pays if the completion is cheap, and with the stars
  gone first the completions are what twenty planners fight over.
  3 pool players unsold in seed 0.
- **Caveat on the planner column.** Its reference room (planner owners,
  own-choice nomination) is the anomalous 39% start already flagged
  under Caveats ($316k per team unspent) and neither planner room is a
  self-play end point, so read the planner comparison as "price order
  gets planners to spend and hands the room to two-star teams", not as
  a Waters-team number to quote. The greedy column is the like-for-like
  test against the shipped `auction_sim` room.
### What people actually paid (mean over 8 seeds; `--room` now writes `cache/strategic_room<tag>_prices.csv`)

| # | player | list | greedy, snake | greedy, PRICE ORDER | planner, PRICE ORDER | adapted, PRICE ORDER (path end) |
|---|---|---|---|---|---|---|
| 1 | Anna Leigh Waters (F) | $769k | $850k | $850k | $850k | $850k |
| 1 | Ben Johns (M) | $474k | $507k | $520k | $414k | $478k |
| 2 | Anna Bright (F) | $613k | $629k | $607k | $568k | $630k |
| 2 | JW Johnson (M) | $430k | $466k | $466k | $373k | $441k |
| 3 | Parris Todd (F) | $489k | $491k | $491k | $466k | $486k |
| 3 | Hayden Patriquin (M) | $421k | $440k | $466k | $339k | $419k |
| 4 | Jorja Johnson (F) | $475k | $488k | $491k | $466k | $479k |
| 4 | Gabriel Tardio (M) | $409k | $443k | $520k | $342k | $432k |
| 5 | Kate Fahey (F) | $453k | $488k | $488k | $392k | $452k |
| 5 | Christian Alshon (M) | $400k | $420k | $488k | $338k | $407k |
| 6 | Jade Kawamoto (F) | $444k | $466k | $466k | $389k | $380k |
| 6 | Andrei Daescu (M) | $323k | $347k | $322k | $312k | $293k |
| 7 | Rachel Rohrabacher (F) | $421k | $491k | $491k | $368k | $346k |
| 7 | Federico Staksrud (M) | $315k | $372k | $293k | $312k | $320k |
| 8 | Tina Pisnik (F) | $416k | $499k | $491k | $363k | $363k |
| 8 | Eric Oncins (M) | $307k | $322k | $334k | $312k | $270k |
| 9 | Jackie Kawamoto (F) | $391k | $555k | $555k | $363k | $327k |
| 9 | Jay Devilliers (M) | $288k | $373k | $434k | $286k | $335k |
| 10 | Tyra Hurricane Black (F) | $386k | $445k | $555k | $338k | $358k |
| 10 | Riley Newman (M) | $251k | $294k | $393k | $261k | $353k |
| 11 | Sofia Sewing (F) | $350k | $434k | $555k | $338k | $333k |
| 11 | Jack Sock (M) | $226k | $272k | $373k | $242k | $199k |
| 12 | Danni-Elle Townsend (F) | $325k | $389k | $345k | $286k | $304k |
| 12 | Phuc Huynh (M) | $210k | $324k | $246k | $211k | $310k |
| 13 | Mari Humberg (F) | $303k | $364k | $392k | $286k | $295k |
| 13 | Thomas Wilson (M) | $209k | $261k | $354k | $234k | $157k |
| 14 | Vivian Glozman (F) | $271k | $312k | $434k | $270k | $287k |
| 14 | Quang Duong (M) | $203k | $251k | $294k | $185k | $157k |
| 15 | Catherine Parenteau (F) | $267k | $300k | $317k | $261k | $236k |
| 15 | Connor Garnett (M) | $202k | $264k | $229k | $218k | $182k |

Reads: greedy owners under price order pay MORE for every man in the
top 5 (Tardio $520k vs $443k, Alshon $488k vs $420k) and the same for
the top women, then bid the second-tier women to a $555k ceiling
(Jackie Kawamoto, Hurricane Black, Sewing -- the most a team that has
already bought one player can pay) and the #9-14 men to 130-170% of
list (Devilliers $434k, Newman $393k, Sock $373k, Wilson $354k);
#21-40 collapse to 63% of list and four of #58-60 go unsold in most
seeds (they come up last, when rosters are full). Planner owners under
price order pay BELOW list for every star (Johns $414k, JW $373k,
Patriquin $339k, Bright $568k, Fahey $392k), list for #16-20, 104% for
#21-40, and double the $80k tier (Widdershoven, Petrei, Wall, Brascia,
Walczak at $145-160k) -- the completions are what planners fight over.
DreamBreaker specialists split the same way: Haworth $43k with greedy
owners vs $137k with planners (list $95k). Bright is the one star who
goes for LESS under price order with either owner type ($607k / $568k
vs $629k under snake): she is sale 2, when the room has just watched
Waters go for $850k and nobody wants two stars of one gender.

- **Self-play with the rule frozen** (`--start nom=dear --fix nom`,
  7 dials searched, 8 seeds, 4 rounds, ~1 h): owners adapt in this
  order -- inflate price expectations (+11.5 pp, se 1.9; the largest
  round-0 gain of any start), a_star 1.25, whole-roster planners,
  learned prices -- and hit the round limit with +5.2 pp (se 1.7) still
  on the table (a_good0 next), so this is a PATH END, not an
  equilibrium. The room at that end (all clones, 8 seeds):

  | | greedy, price order (start) | adapted owners, price order (path end) |
  |---|---|---|
  | price / list, #1-5 / #6-15 / #16-30 / #31-60 | 1.10 / 1.31 / 0.92 / 0.49 | 1.02 / 0.96 / 0.83 / 1.18 |
  | Waters price | $850k, sale 1 | $850k, sale 1 |
  | Waters' team win / title | 66.2% / 30.2% | 71.3% / 28.8% |
  | best team | 66.4% | **74.3% (not hers)** |
  | second-best team | 61.6% | 69.0% |
  | teams at 10%+ title | 3.9 | 3.9 |
  | parity spread (pts) | 8.0 | **12.8** |
  | floor-priced (of 120) | 45.3 | 19.9 |
  | unspent per team | $8k | **$156k** |

  Seed 0 is a league of super-teams and stragglers: Alshon + Sewing +
  Humberg 70.3%, Todd + JW Johnson 66.1%, Johns + Jade Kawamoto 64.2%,
  Waters + five floor players 64.0%, Fahey + Tardio 63.7%, Rohrabacher
  + Hurricane Black 60.8% -- and six teams at 33-42%. Adapted owners
  expect last room's prices and plan whole rosters, so they hold money
  back for completions that then sell above list (#31-60 at 118%,
  Katerina Stewart $65k -> $244k, Walczak $79k -> $220k, Haworth $200k)
  while 13 of the #29-60 go unsold in at least one seed (they come up
  last, after the planners have filled) and $156k per team is left
  over. Stars themselves sell at list (Johns $478k, Bright $630k, JW
  $441k, Tardio $432k), the second-star tier BELOW list (Jade Kawamoto
  $380k, Rohrabacher $346k, Jackie Kawamoto $327k). So the rule's
  effect with adapting owners is not "a fairer league" -- it is a
  wider one: more strong chasers (3.9 at 10%+, second-best 69%) AND
  more dead teams, spread 12.8 vs 7-8 in every other room in this
  file. Waters' team wins more ties (71%) because the bottom is
  weaker, but her title share falls (28.8%) because the top is
  stronger. Read with the gate caveat (path end, symmetric family,
  noise 0); the direction (chase widens, Waters unmoved in price) is
  the same at every node of the path.

## What is robust across both starts and every node

| | `auction_sim` truthful | truthful-start cycle node | planner-start end |
|---|---|---|---|
| Waters price | $850k, sale 1 | $850k, sale 1 | $850k, sale 1 |
| Waters' team win / title | 68.8% / 37% | 67.5% / 37% | 67.2% / 33% |
| second-best team | 59.7% | 58.7% | 59.5% |
| teams at 10%+ title | 2.6 | 2.1 | 3.0 |
| parity spread | 7.3 | 6.0 | 6.8 |
| #1-5 / #6-15 / #16-30 / #31-60 vs list | 1.06 / 1.21 / 1.06 / 0.53 | 1.16 / 1.05 / 0.99 / 0.75 | 1.06 / 1.20 / 1.07 / 0.54 |

Waters' price and team, the second-best team and the count of chasers
do not move with strategy. What strategy moves is the shape of the
middle: whether the second star or the top 5 carries the premium and
how much of depth sells at the floor.

## Caveats

- **Family, not game.** The toy says the symmetric dial family's
  equilibrium differs from the exact one at 3+ teams in a specific
  direction (asymmetric aggression from teams that missed the stars). The
  20-team numbers above are what this owner population does, and the
  robust rows are robust WITHIN it. The next instrument, if anyone wants
  the game's equilibrium, is role-typed profiles (the no-star team plays
  a different vector, not one extra multiplier) with the same self-play
  and the same toy check.
- **Gate and seeds.** 8 seeds on the truthful start, 4 on the planner
  start; single-dial gains under ~5 pp sit inside their error bars; the
  adoption rule tests only the best gain, which let the planner start
  stop while a smaller-se deviation still cleared the bar.
- **Cycles.** Coordinate ascent on a discrete profile has no
  convergence guarantee; the truthful start's a_good0 cycle is a
  real best-response cycle (bid up good players when nobody else does,
  stop when everyone does), not a numerical one.
- **Noise 0.** Every owner shares the true values; `auction_sim`'s 10%
  belief noise, `--noise`, was not swept for cost. With dispersion the
  second-price mechanism pays the second-highest BELIEF, which should
  raise mid prices and lower exploitability of the price-expectation
  dials; the Waters rows should not move (she is at the cap every time).
- **Shortlist artifact.** `auction_sim`'s 68% Waters team depends on
  its shortlist rule and snake nomination leaving floor-priced good
  players uncontested; bidding on everyone with the same greedy owner
  reproduces the room (planner-start end point), but planner owners
  bidding on everyone at list expectations do not (the 39% starting
  room). Any comparison across owner types must hold the bidding rule
  fixed.
- **Costs.** Truthful start 4.5 h at jobs 3 (four planner rounds at
  80-86 min each); planner start 45 min at 4 seeds / jobs 4. Cache:
  `cache/strategic_real_{truthful,planner}.json` (path, every
  deviation's gain and se, room stats at start and end),
  `cache/strategic_toy.json` (the ladder).
