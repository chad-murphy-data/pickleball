# How to freeze out the best player alive

*What the numbers say about Anna Leigh Waters getting iced out — and,
just as importantly, what they can't say. For the number-curious. No
statistics degree required.*

## The upset

At the MLP Mid-Season final, Anna Leigh Waters and Jorja Johnson walked
onto the women's doubles court as heavy favorites. Waters is the most
dominant player in the sport — not the best woman, *the best*, by a margin
that doesn't really have a comparison. Our model, which has watched every
professional game since 2024, made her team **88% favorites.**

They lost. 6-11. It was the single biggest miss on our public
prediction ledger that weekend — the game where the model looked most
foolish. Which makes it the most interesting game to explain, because you
learn nothing from the predictions that come true.

The word going around was that Bright and Fahey **iced Waters out**:
played the whole game to Johnson, kept the ball away from the superstar,
and dared the weaker half of the team to beat them. It's an old strategy
with a new nickname. The question we wanted to answer wasn't "did the
model blow it" — favorites lose sometimes, that's what *favorite* means.
The question was: **if you really can freeze out the best player alive, how
much is that actually worth?** And can we see it in the data at all?

## First, the thing that makes freezing possible

Here's the finding that everything hangs on, and it's the sleeper result
from our whole project: **pro doubles is a weakest-link game.** A team is
not the sum of its two players. It's more like **59% the weaker player and
41% the stronger one.**

That sounds backwards until you think about who's holding the paddle.
*You* don't get to decide how involved your star is — your opponents do.
They aim at the weak link. Every third ball goes to the player they'd
rather hit, which means a superstar's greatness is capped by a simple
physical fact: **she can only cover so much of the court.** The rest of the
time, she's watching her partner play.

So the 88% was never Waters playing singles. It already assumed the
normal amount of "hide from the superstar" that every opponent tries.
Baked into that number is Waters covering roughly **41%** of the meaningful
balls, and Johnson covering the other **59%** — because that's the tilt
opponents can usually force.

The ice-out is just that tilt, cranked to the extreme.

## The dial

Because the model knows how much a team's value depends on who covers the
court, we can put that on a dial and watch the win probability move. Let
"Waters's share of the court" slide from a normal game down toward zero:

- **Waters covers ~41% (a normal game):** 88% to win. *(The real number.)*
- **Waters covers 50% (an even split):** 93%.
- **Waters covers 0% (frozen out completely):** **48%.**

Read that last line again. If you could truly erase Anna Leigh Waters from
a game — not injure her, just never let her touch a meaningful ball — her
team goes from an 88% favorite to a **coin flip.**

And here's the cleanest way to see *why*, the number that stuck with me:
freeze Waters out entirely, and mathematically her team stops being
"Waters and Johnson" and becomes **two Jorja Johnsons.** Two copies of the
weaker player. Because if the star never plays, the team performs at the
partner's level, twice over.

So we asked the model the silliest possible version of the question:
**two Jorja Johnsons versus Bright and Fahey — who wins?** Answer: **48%.**
A coin flip. Exactly the same as the full freeze, as it has to be. Two
Johnsons *is* the frozen team.

That's the whole strategy in one sentence: **you don't beat the GOAT by
outplaying her. You beat her by turning her team into two of her partner —
and two of her partner is a coin flip.**

## So did Bright and Fahey actually do it?

This is where an honest project has to slow down, because the exciting
answer and the true answer aren't the same.

We have the referee's log for that game — every rally, who served, who
received. So we looked: did Bright and Fahey actually funnel the ball at
Johnson? On serve, the answer is **no, not really** — they served about
evenly, 10 balls at Johnson and 9 at Waters. But that's not damning,
because you don't *get* to choose who you serve to; the rotation decides
it.

The real ice-out doesn't happen on the serve. It happens on the third
ball, and the fifth, and the seventh — in where you place a volley, who
you make hit the awkward shot, which player you quietly refuse to engage.
And **none of that is in any dataset that exists.** No feed tracks where
each shot in a rally goes. The only thing that could confirm an ice-out is
watching the tape.

So here's the honest shape of what we can claim. **We can prove the
strategy is *sufficient* — that a full freeze, all by itself, is enough to
turn this 88% game into a coin flip. We cannot prove it *happened*.** The
math says the door was unlocked; it can't tell you they walked through it.

## How surprised should we have been, really?

Not very, and this matters for keeping the story honest. An 88% favorite
loses roughly **one game in eight.** That's not a fluke or a model failure
— it's just what 88% means. We checked it the boring, rigorous way: across
hundreds of held-out games, teams the model called at 90% actually won
about 92% of the time. The favorites are real; they also lose sometimes.

We even poked at "maybe we just had the players rated wrong that night."
But these four are among the most-played athletes in the sport, so their
ratings are pinned down tight. To manufacture a Bright/Fahey win out of
*rating error alone*, you'd need all four players to be misjudged by a
large margin, all in the convenient direction, at the same time — a
one-in-thousands coincidence. It's a much smaller stretch to say: someone
executed a good game plan, and/or an 88% favorite ran into its 1-in-8.

And for what it's worth, **6-11 wasn't even a freakish scoreline.** When a
favorite loses, it's usually close — the most likely losing score in this
matchup is 11-9, and about two-thirds of losses are 11-8 or nearer. A 6-11
is on the lopsided side of normal, the kind of beating that's consistent
with a real tactical edge on the night, but nowhere near a miracle.

## A note on holding ourselves to it

One confession, because it's the part I'm proudest of. When we first
wrote down "how surprised should we be," we reached for the model's
built-in game-to-game noise and got a loss probability around **17%.** It
felt right — games are messy, add some slop.

Then we tested it against reality instead of trusting the vibe, and the
data said **no**: adding that extra noise made our predictions measurably
*worse*, not better. The honest number is about **12%** — a 1-in-8, not a
1-in-6. We'd talked ourselves into a plausible-sounding correction, and
the out-of-sample check vetoed it. That's the whole discipline in
miniature: the receipts outrank the story, even when the story is your
own.

## What it means

Icing out the best player alive is not cope, and it's not a hot take. It's
a structurally sound strategy, and the reason it works is subtle and kind
of beautiful: because pro teams are mostly their weaker player, neutralizing
the star doesn't just subtract her shots — it drags the whole team down to
the partner's level. Freeze Anna Leigh Waters and, on the scoreboard, you
are playing two Jorja Johnsons. Two Johnsons is a coin flip. Bright and
Fahey won the coin flip.

Did they earn it with a perfect game plan, or did an 88% favorite simply
hit its 1-in-8? One game can't tell you, and we're not going to pretend
otherwise. But the ceiling of the strategy — the best case for "just ice
her out" — is exactly the game we watched. And now we know precisely how
high that ceiling is.

---

*Numbers from the v2 model (every MLP + PPA pro game since 2024; 77%
winner accuracy out of sample, versus 65% for the tours' official rating).
Reproduce them yourself: `model/iceout_waters.py`. Every prediction in
this piece was on our public ledger before the match, including the one we
got wrong.*
