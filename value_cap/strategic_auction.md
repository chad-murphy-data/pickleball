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
self-play round is 27 deviations x 4 seeds.
