# The MLP Value Cap, explained (for a smart 15-year-old)

This is the plain-language version of how the price list and the
auction experiments work. Every number here is stated more carefully
in `phase2_pricing.md`, `auction.md` and `strategic_auction.md`; this
file exists so you can follow the idea without reading those.

## The question

Major League Pickleball teams have six players: three men, three
women. Imagine the league gives every team $1 million to spend on
those six. What should each player cost?

The honest answer is "whatever makes the league fair": if you spend
your million well, your team should be about as good as anyone else's
who spent theirs well. A price list that does that is a fair price
list. Everything below is how we build one and then try to break it.

## Step 1: how good is each player?

We have every pro doubles game since 2024, about 37,000 of them. A
model (PICKLES, the same one the rest of this site runs on) turns them
into one number per player: how much stronger than average they are
at winning a single point. Ben Johns and Anna Leigh Waters are at the
top; the number tells you how far.

Two extra things the model learned that matter here:

- **A team is only as good as its weaker player, mostly.** When a great
  player pairs with a weak one, the pair is worse than the average of
  the two. Opponents pick on the weak player. The model measures this
  ("weakest link") and we use it as-is.
- **Singles is a separate skill.** MLP ties that reach 2-2 end with a
  DreamBreaker, which is singles. Some players are ordinary in doubles
  and excellent in singles (Christopher Haworth is the famous case), so
  we also have a singles rating for everyone.

## Step 2: what does a six-player team actually win?

An MLP tie is four doubles games (women's, men's, two mixed) and, if it
is 2-2, a DreamBreaker. Given two six-player rosters we can compute the
chance one beats the other: pick each team's best four for the doubles
games, the best four singles players for the DreamBreaker, work out
every game's odds from the ratings, and add it up. This "tie model" is
the engine everything else runs on. It is not new here; it is the same
math the site uses to predict real matchups.

## Step 3: how much is one player worth?

Here is the trick. You cannot ask "how good is Waters" in a vacuum;
you have to ask "how much better is a team WITH her than the same team
with a replacement-level player in her spot". But which team? A star
on a team of scrubs looks different from a star on a team of stars
(weakest link again).

So we do it thousands of times. Build a random team from the pool of
players good enough to be rostered, drop the player in, drop a
replacement in instead, play both versions against a random opponent,
take the difference. Average over all those random teams. That average
is the player's **value**, written phi. It is the same idea as the
Shapley value from game theory: your worth is what you add, averaged
over all the teams you could be on.

One wrinkle: "the pool of players good enough to be rostered" depends
on the values, and the values depend on the pool. We just loop: guess
the pool, compute values, re-pick the pool from the values, repeat
until it stops changing. It settles after two rounds.

## Step 4: turn value into dollars

Twenty teams times $1 million is $20 million. Add up everyone's value.
Your price is your share of the total value times $20 million. That is
it. Men and women are priced from the same pot, and the split comes
out about 57% to the women, because the women's top end is stronger
relative to its replacement level.

We tested whether stars should get a discount or a premium on top of
that (a "curvature" knob called alpha). Straight proportional pricing
(alpha = 1) is the one that makes equal-strength rosters cost the same,
which is what fair means, so we shipped it.

**The Waters problem.** Waters' value is 5.3% of the whole league.
A team is 5% of the league. Her fair price is $903k, and no team can
pay that and still buy five more players. So she gets a **franchise
tag**: her price is set to the most a team can possibly pay and still
fill a legal roster ($769k with the cheapest five teammates at $30k
each), and the money she "should" have cost is spread back over
everyone else so the list still adds up to twenty caps. One player,
one rule, everyone else priced normally.

## Step 5: does the list work? Fake leagues

A price list is a claim: "spend your million well and you'll be
competitive". We test claims by simulation. Twenty computer owners
draft off the list (snake draft), each trying to build the roster with
the best odds; then we play thousands of seasons on the true ratings.

What comes out: the team that gets Waters (plus cheap DreamBreaker
singles specialists with the change) wins about two-thirds of its ties
and about one title in three. The other nineteen teams are all around
50%. Nobody in the top 30 goes undrafted, everyone spends their cap.
The user's call: one 35% favourite is normal for a pro sport; ship it
and say so.

We also tried owners with personalities (the guy who overvalues men,
the bargain hunter, the $500k cheapskate). Only the cheapskate breaks
the league, by leaving money unspent, which is the case for a minimum
spend rule.

## Step 6: what if it is an auction instead?

In a snake draft prices are fixed. In an auction the room sets them.
So we built an auction: one player up at a time, every owner bids up to
the most they think the player is worth to THEIR roster, the winner
pays the second-highest bid plus $5k (like eBay). One hard rule binds:
you can never bid more than your remaining budget minus the cheapest
legal way to finish your roster, so the first purchase of any team
tops out at $850k.

Results: Waters sells for that $850k maximum, first sale, every time.
Her team is still a two-thirds team. But the auction creates something
the snake could not: a **chase**. Two or three other teams buy two
$400-500k stars, one man and one woman (they share the mixed court),
and fill the rest at the floor. Those teams get to 60%+ and a 10-15%
title shot. The room also re-prices the list's middle: second-tier
stars go above list, depth goes to the floor.

## A turn in the room: Bright is up

Step 6 says "every owner bids up to the most the player is worth to
their roster". Here is what that means, one sale at a time, using one
real traced auction (owners as built in `strategic_auction.py`, honest
bidders, list prices expected, seed 0; run it yourself with
`SA_TRACE="Anna Bright" python value_cap/strategic_auction.py --world
real --seeds 1 --time --start "nom=dear,plan=planner,bid=all"`).

**Sale 1, Waters.** All twenty owners have an empty roster and $1M.
Every one of them asks the same question: "what is the best team I can
build WITHOUT her at list prices, and at what price for her does a team
WITH her stop beating it?" For Waters that price is above what anyone
may pay, so every owner hits the hard rule instead: $1M minus five
floor players at $30k = $850k. Twenty identical $850k bids. The sim
breaks the tie with a coin flip; call the winner Team 15. It pays the
second-highest bid plus $5k, capped at its own bid, so $850k. Team 15
now has $150k for five slots. It will bid the $30k floor on everyone
for the rest of the night and end up with Waters plus five floor
players (59% in this room). Nobody else has spent anything.

**Sale 2, Bright (list $613k).** Now the owners differ. Three kinds of
owner are in the room, and each does something different.

1. **An owner with an empty roster (nineteen of them).** The owner first
   builds its best six without Bright, paying list for each slot, and
   records that team's expected win rate against the field. Call that
   the baseline. Then it tries rosters WITH Bright at a trial price:
   Bright at $p, plus the best five it can still afford from the
   money left over, again at list. It searches for the largest $p at
   which the Bright team still matches the baseline. That is the most
   it will pay, because one dollar more and it would rather not have
   her. In this room the answer is $568k, below her list price, and
   all nineteen empty owners land on the same number because they
   hold the same thing (nothing) and expect the same prices. Nineteen
   bids of $568k, one coin flip, Team 5 wins and pays $568k.

   Why below list? Because these owners plan the whole roster. A
   planner's best no-Bright team is quite good (two $300-400k players
   and real depth), so Bright has to be a bargain to beat it. In the
   same auction with owners who fill one slot at a time (the "greedy"
   owners of the earlier sims) the no-Bright baseline is weaker, the
   indifference price comes out at $632k, above list, and she sells
   for $632k. Same player, same money, different bid, and the only
   thing that changed is how well the owner imagines its fallback.

2. **An owner who already bought a star.** In the greedy auction
   Bright comes up at sale 6, after Johns, JW Johnson, Jorja Johnson
   and Todd have sold. The owner holding JW Johnson at $466k would
   value Bright at about the same $632k as everyone else, but it
   cannot pay it: $1M minus $466k minus four floor players leaves
   $414k, and that is its bid. The Jorja owner bids $392k, the Todd
   owner $389k, the Johns owner $373k, each exactly its remaining
   budget minus completions. Their bids are not a valuation, they are
   a wallet. This is why the sim's first-buy ceiling matters more than
   any strategy knob: once you own one $450k player you are out of the
   running for a second star unless you go floor everywhere else.

3. **Waters' owner.** Team 15 bids $30k, the floor, because $150k for
   five slots leaves nothing above it.

**Who wins and what they pay.** Bids are sorted; ties are broken by the
coin flip; the winner pays the second-highest bid plus $5k, never more
than its own bid. When fifteen owners tie at $632k the winner pays
$632k. In the planner room one owner (Team 11) later stretches into a
second star anyway, Johns at $414k on top of Jorja at $466k, and
finishes with four floor players and a 33% team. The two-star chase
the earlier sims found works only when the two stars are cheap enough
to leave real money for the other four, and in this seed they were
not.

**The knobs from Step 7, in this picture.** "Bid over list on stars"
multiplies that indifference price by, say, 1.25 before the budget
cap. "Expect inflated prices" changes the list prices the owner
plugs into its fallback team, which moves the baseline and so the
bid. "Plan the whole roster" versus "fill one slot at a time" is the
$568k versus $632k difference above. "Nominate by price" is what put
Bright up at sale 2 instead of sale 6. None of those knobs touch the
$850k first-buy rule, which is why Waters' price never moves.

## Step 7: what if the owners are clever?

The auction owners above are honest: they bid what a player is worth
to them and nothing else. Real owners scheme. Three instruments look at
this:

1. **The market limit.** Suppose every owner plans their whole roster
   and nobody overpays. Run that to its fixed point. Waters is still
   the only player where demand exceeds what a team may pay; everyone
   else is bid to the price where their team is average. Nineteen
   equal teams, one favourite.
2. **A toy solved exactly.** With 2-4 teams and a handful of typed
   players you can solve the auction perfectly by working backward
   from the last sale. Lesson: who ends up where is stable, but the
   PRICES they pay swing wildly with tiny rule details. So we judge
   strategies on rosters, not prices. Twenty teams cannot be solved
   this way; the game is too big.
3. **Strategy search on the real board.** Give every owner a strategy
   made of eight knobs (how much to bid over list on stars, on good
   players, when you have no star yet; what prices you expect; whether
   you plan the whole roster or one slot at a time; who you nominate;
   who you bother bidding on). Everyone plays the same knobs. Then ask:
   can one owner do better by turning a single knob? If yes, everyone
   adopts it, and we ask again. Stop when nobody can gain, or when it
   goes in a circle. The biggest gain still available is how
   "exploitable" the room is. We checked this on the toy (it matches
   the exact answer with 2 teams, not with 3-4, so we call the 20-team
   answer "the equilibrium of this strategy family", not of the game).

What survives all of that: Waters at $850k, first sale, her team at
67-69%, two or three chasers. What the schemers change is the middle
of the price list, not the top.

## Step 8: rule experiments

Once you have a room that behaves, you can change the league's rules
and watch. The one tried so far: **nominate in price order** (dearest
remaining player is always next, nobody chooses). Waters' price does
not move, she is first either way. But the chase gets wider and
stronger: more teams at 10%+ title odds, second-best team over 60%,
and with roster-planning owners the two-star build becomes the
favourite outright. So it is the first rule found that dents the
favourite without touching her price, and it does it by strengthening
the field rather than weakening her.

## What we are sure of, and not

Sure: the ratings (validated on unseen games), the tie model (same
one that grades real predictions), that Waters is worth more than a
team may pay, and that a fair list plus a snake draft gives one
favourite and nineteen equals.

Not sure, and said so in the write-ups: MLP's real rules (we do not
know if it is a draft or an auction, whether there is a per-player
maximum or a minimum spend, or how playing time is decided, which is
the biggest lever of all); injuries and absences (not modelled yet);
and the 20-team auction has no exact solution, so every "smart owner"
result is a best effort with a stated blind spot.

The working rule throughout: when a number is unknown, sweep it and
show the range rather than pick one and hide it.
