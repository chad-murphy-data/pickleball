# The records-only owner — spec, 2026-09-05

(Naming, user call 2026-09-05: these are OWNERS — people who bought a pickleball team and follow the sport without analytics — not "fans". The `fan_` file prefix is kept from the first draft.)

Status: BUILT end to end (`fan_view.py` knowledge layer, `fan_auction.py` room, `fan_auction.md` results) 2026-09-05;
awaiting sign-off before the auction is built. This document is the design
conversation written down; the numbers are from `python value_cap/fan_view.py`.

## Why

Every room `strategic_auction.py` has run hands each owner OUR price list as
its picture of what the other slots will cost: `expect=list` uses it,
`inflate` scales it, `rivals` caps it, `learned` seeds from it. So the
indifference prices in the walk-through ($568k / $632k for Bright) are
"the owner's valuation GIVEN our sheet". The question this answers: what
does a first-ever MLP auction look like when nobody has a sheet and nobody
can compute one?

## What a records-only owner knows (user-set, 2026-09-05)

Four layers. The fog is in dollars and in what wins, not in who is good.

**1. Rules (everyone).** 6 players, 3M + 3W, $1M cap, $30k floor. A tie is
WD, MD, MXD1, MXD2, DreamBreaker singles at 2-2 with a team-named 4-player
lineup. Four of six play a tie; players 5 and 6 rarely play (owners know
this). Cap arithmetic: $167k average slot, $850k most anyone can put on one
player.

**2. Player facts (everyone, with sample-size sense).**
- Doubles win% and games, 2026, split by own-gender doubles and mixed.
- Singles win% and games (2026 and career) — so who plays singles and how
  well. Todd and Fahey being elite singles players is common knowledge; so
  is Hurricane Black not playing singles.
- An ORDINAL picture within gender: one joint draw from the v2 posterior
  (`value_now_mean ± value_now_sd`), ranked, and the owner keeps only the
  order. The fog is the model's own uncertainty, scaled by `--sd-mult`
  (1 = posterior; 2 = an owner who reads results less efficiently). Same for
  singles from the singles suite posterior.
- 2026 MLP franchise and usage (matchups appeared / franchise matchups).
  Bench and injury are indistinguishable in the data; only the risk-averse
  persona reads low usage as risk (Rohrabacher 19/27, Todd 23/33,
  Hurricane Black 22/33, Hunter Johnson 11/27).
- NOT known: any per-point value, tie probability, phi, price; any
  cross-gender comparison (Johns vs Bright is not a question such an owner can
  answer, and the owner is never asked it).

What the ordinal draw looks like (P(rank ≤ k), posterior ×1, mlp2026 board):

| | E[rank] | P(#1) | P(≤3) | P(≤5) | P(≤10) |
|---|---|---|---|---|---|
| Waters | 1.0 | 1.00 | 1.00 | 1.00 | 1.00 |
| Bright | 2.6 | 0.00 | 0.87 | 0.96 | 1.00 |
| Jorja Johnson | 5.9 | 0.00 | 0.25 | 0.55 | 0.91 |
| Todd | 7.2 | 0.00 | 0.15 | 0.39 | 0.82 |
| Fahey | 9.6 | 0.00 | 0.03 | 0.14 | 0.62 |
| Johns | 3.2 | 0.27 | 0.66 | 0.85 | 0.98 |
| JW Johnson | 4.3 | 0.20 | 0.52 | 0.73 | 0.95 |
| Patriquin | 4.4 | 0.14 | 0.46 | 0.71 | 0.95 |
| Tardio | 4.6 | 0.11 | 0.43 | 0.70 | 0.95 |
| Alshon | 6.9 | 0.04 | 0.21 | 0.43 | 0.84 |

Read: the women's top two are certain, #3-#10 is a pack. The men's #1 is a
genuine four-way question even to the model (Johns 27%). An owner's "ALW >
Bright > Jorja" is exactly what the posterior says; "Johns is the best man"
is a 27% opinion.

**3. Beliefs about what wins (opinion; heterogeneous; the personas).**
How much a star matters vs balance, whether the bench matters, whether the
DreamBreaker is worth planning for. The model has answers; owners have
none. First draft = a hand-picked MIX of personas (user call); learning
across seasons is a follow-up.

**4. Dollars (nobody).** No list, no past auction. Price information =
cap arithmetic + tonight's sales so far.

## Personas (Layer 3) — signed off 2026-09-05, two amendments

Each is a roster SHAPE: six roles with (gender, acceptable band in the
owner's own ordering, budget share). Shares are the owner's plan for
spreading $1M, not a valuation.

| persona | shape | shares | notes |
|---|---|---|---|
| star & scrubs | one top-3 (either gender) + five floor | 85 / 3×5 | the build our model says wins; unclear any new owner would try it |
| two stars | top-5 woman + top-5 man + four cheap | 40 / 40 / 5×4 | the auction's chase build |
| four starters | 2W + 2M from own top-15, bench at floor | 22×4 / 6 / 6 | the natural new-owner build ("four play") |
| balanced six | six from own top-30 | 17×6 | |
| singles-minded | four starters, plus ONE of the six must be a real singles player, any gender (user amendment: "tries to find a singles player in their top 6 somewhere") | 22×4 / 6 / 6 | real = active on the singles tour (≥10 games 2026 or ≥100 career) and in the owner's own singles top-10; the bench slot pays up to 6% for one if no starter is |
| risk-averse | four starters + a real bench, avoids <75% usage | 20×4 / 10 / 10 | the only persona that reads usage |

Default room: a mix (e.g. 2 / 3 / 6 / 4 / 3 / 2 of 20); sweeps: all-X rooms,
one-X-in-a-room-of-Y, sd-mult 1 / 2.

## Bid rule (Layer 4) — signed off 2026-09-05

For owner o and player x on the block:
1. **Role match.** x fills the most expensive unfilled role whose gender
   and rank band (own ordering) it satisfies; otherwise no bid (the
   nominator still opens at the floor if it can afford completion).
2. **Plan money.** ceiling = that role's share × $1M + savings carried from
   roles already filled under plan.
3. **Watch the room.** Once ≥3 players in the same band have sold tonight,
   going_rate = their median price. If going_rate < plan money, ceiling =
   going_rate × (1 + premium), premium a persona dial (sweep 0.1 / 0.3).
   Owners never bid above plan money for a role — except:
4. **Scarcity.** If the acceptable players left for the role ≤ the rivals
   who still need that role, bid full plan money (ignore the going rate).
5. **Hard cap.** Never above budget − floor × (slots left − 1).
Payment: second-highest + $5k, as in every room so far. Nomination:
rotation; the nominator names the top target of its dearest unfilled role
(owners nominate who they want) — `nom` sweepable as before.

No owner ever evaluates a roster's tie probability. Payoffs (win%, title
odds) are scored on the TRUE tie model, because that is the world.

## What the room will tell us

- Paid prices vs our list, by rank band: does a room with no sheet land
  near it, and where does it bend?
- Waters' price and buyer. Only star & scrubs can pay $850k; if the room
  has one such owner she goes for the second bid + $5k, i.e. the two-star
  owners' ~$400k. Her price is set by the persona mix, not the cap.
- Which philosophy wins a first draft (persona win% / title odds) — the
  thing the real league will learn.
- Title concentration and unspent money vs the quant rooms.

## Status

BUILT 2026-09-05: `fan_auction.py` (results in `fan_auction.md`). Two
user amendments folded in: (a) singles-minded = one real singles player
anywhere in the six, either gender; (b) each owner is handed ONE rank per
player — a single joint posterior draw per owner, ranked within gender
(`fan_view.draw_order`) — owner 1 has Fahey 10th, owner 2 14th, owner 3
6th; nobody sees a distribution. Build decisions taken while wiring the
rule (all dials, none picked as truth): savings from roles filled under
plan spread evenly over the OPEN targeted roles first, then the floor slots
(a star-and-scrubs owner who lost the star at $424k does not sit on $445k);
a role nobody left on the board can fill degrades to "anyone of that
gender" but keeps its money AND is capped by the going rate of the band the
player actually sits in (no $225k scrubs); floor slots bid only on the best
of what is left in the owner's own ordering (rank among available ≤
2×open slots + 2); per-owner share jitter sd 0.1 so identical textbook
plans do not tie to the dollar (`--jitter`, 0 = textbook).

REPRODUCIBILITY FIX 2026-09-05 (later): two tie-breaks (the cross-gender
coin flip in nomination, the fallback nomination) consumed random numbers
in Python set-iteration order, so the same seed gave a different room in a
different process (PYTHONHASHSEED). Fixed — candidates are sorted by pid
before the rng is touched; `fan_auction.md` was regenerated and the
headline numbers moved only at the margin (default mix still $570k,
68/78/113/149; jitter-0 price $794k not $850k; sd×2 $792k not $738k).

LEARNING ACROSS SEASONS BUILT 2026-09-05 (later): `owner_learning.py` →
`owner_learning.md`. Ten seasons of full re-auctions; three switchable
channels — P the room remembers last season's prices (going-rate seed +
budget shares re-anchored toward the band median, weight lam), S owners
copy a random playoff team's shape with probability p (or the champion),
K rank draws sharpen (sd × decay per season). Headline: S is the channel
that makes the room look like the list (p 0.5 → 95/100/93/106 of list by
season 10, the list's shape from a room that never saw it); P alone
deflates stars and inflates depth (the anchor is a band median that depth
buyers set, and no shape pays more for #1 than #15); K is agreement =
competition (unspent $0.84M → $0.18M). Every learning cell makes Waters
CHEAPER and her team STRONGER. Degenerate corner: identical plans + shared
ranking + last-season anchors → one owner sweeps Waters, Bright, Johns and
JW Johnson at $225k each (99% / 89%). The missing dial is a within-band
rank slope — open decision below.

W (RESULTS) CHANNEL ADDED 2026-09-05 (later still; user: a player who
wins every season should get dearer, and nothing in P/S/K could make that
happen — none connects a player to her team's result). Public reputation
multiplier per player, moved by the team's realised win% (credit eta,
attributed by price paid or equally, cumulative or last-season-only),
applied to every owner's ceiling (`Owner.rep`). Waters sells for the $850k
maximum from season 2 in every W cell; the curve un-inverts in one season;
at the cap with five floor teammates her team is 62-70% / 6-18% (control
72% / 25%). Attribution sets the league around her: by-price marks losers'
expensive players down and leaves the cap unspent ($1.84M); equal credit
gives the tightest league of any cell (spread 10.9) and every shape
converges to 46-52%. Reputation chases one season's luck. Picked, not
swept: the [1/5, 5] clip (irrelevant to her price; it sets how hard losers
are marked down).

## Open decisions (user)

- Persona set and default mix above; any persona missing (marketing /
  hometown from `personas.py` could be re-added as a target-list tilt).
- Premium and carry-over rules in step 3; whether a persona may exceed plan
  money for its star (a "stretch" dial).
- ~~Shared vs per-owner ordinal draw~~ — per-owner, one rank per player (user call).
- ~~Learning over seasons~~ — built (`owner_learning.py`, P/S/K/W); its
  open dials: anchor rule for P (band median vs own top target vs
  asymmetric "raise if I lost"), copy target for S, keepers/contracts
  across seasons; for W: symmetric vs winners-only credit, the clip, what
  counts as "winning" (win% vs playoffs vs title), one season vs a running
  record, public vs private reputation.
- **Within-band rank slope**: does an owner pay more for its #1 than its
  #15 inside a band, and how much? Every shape is flat today; the learning
  runs show that flatness is what makes price memory deflate the stars and
  what lets one owner sweep the top four at the going rate.
