# What we did, in plain language

*A walkthrough for the number-curious. No statistics degree required.*

## The question

Pro pickleball doubles has an obvious puzzle: when a team wins, who gets the
credit? Anna Bright and Hayden Patriquin dominate mixed doubles — is that
because they're both great, or because they're great *together*? Announcers
love the chemistry story. We wanted to know if the numbers do.

## Step 1: Get every game

The tours' results website quietly loads its data from a public feed. We
wrote a polite little program (one request per second) that walked through
every date since January 2024 and collected every professional doubles game
— both tours, every player, every game score. About **36,000 games** and
**3,600 players**, each with a permanent ID number so that "Tyra Hurricane
Black" and "Tyra Black" never get counted as two people.

## Step 2: Rate the players — all at once

Here's the whole model in one sentence: **every game's point margin should
equal the value of your two players, minus the value of their two players,
plus luck.** Write that equation down for all 36,000 games, then ask a
computer to find the player values that explain the results best.

Doing it this way — everyone at once — is the trick. Beating someone 11-7
means more if their teammates were strong, and the model knows exactly how
strong, because their games are in the system too. Every rating is
automatically adjusted for the quality of every partner and every opponent.

One more ingredient: skepticism. The model treats every player as average
until the evidence pushes it off that assumption. Twenty games of hot streak
move your rating a little; two hundred games move it a lot. (Statisticians
call this *shrinkage*. Think of it as the model saying "prove it.")

## Step 3: Check that it actually works

A rating system that can't predict the future is just vibes with decimals.
So we froze the model on June 1st and made it predict every game for the
next six weeks — games it had never seen.

**It called 75% of winners correctly.** For comparison, a coin flip gets
50%, and the official rating system used by the tours (using ratings that
kept updating all summer, an unfair advantage) got 65% on the same games.

## What we found

**1. Anna Leigh Waters is on her own planet.** She adds about 7.7 points per
game to a team, roughly 1.7 more than the #2 woman (Anna Bright). That gap —
between #1 and #2 — is about the same size as the *entire spread of the
men's top 25*. The men's #1 spot is a genuine pileup: Tardio, Johns,
Patriquin, JW Johnson and Alshon are all within a whisker of each other.

**2. Chemistry is mostly a myth.** How good the players are matters about
**five times more** than how well they fit together. Bright & Patriquin's
dominance? Two top-5 players standing on the same side of the net. Their
special sauce is worth maybe a tenth of a point per game — and we can't even
be sure it's real. Nobody's chemistry cleared that bar. To *prove* a typical
chemistry effect exists for one specific pair, you'd need about a thousand
games together; the busiest pair in pickleball played 138.

**3. Pickleball is a weakest-link game.** This was the sleeper finding.
A team isn't worth the sum of its players — every point of skill gap between
partners costs about half a point of team strength. Star-plus-passenger
loses to balanced-and-solid at the same total talent. And in mixed doubles,
the target on someone's back follows *skill*, not gender: the data rejects
the old "attack the woman" doctrine in favor of "attack the weaker player,
whoever that is."

**4. The stories in the ratings are real.** Gabriel Tardio climbed from #8
to #4 to #1 among men across three seasons. Ben Johns slid from clear #1 —
the kind of gap Waters has now — to "member of a five-way tie." The official
rating, for what it's worth, had Tardio *declining* through his breakout,
which tells you something about official ratings.

## What we can't know (and refuse to pretend we do)

- **Can't split "great player" from "great partner."** A game score only
  tells you what the *team* did, so a player's own skill and their boost to
  a partner are mathematically welded together. We say so instead of making
  it up. (This is also why the weakest-link discovery was exciting — it's
  the closest thing to a real "partner effect" the data can support.)
- **Can't compare men to women.** Men and women never play *against* each
  other — every game has the same number of women on each side — so the data
  literally contains no information about how the two rankings line up.
  Waters #1-overall is a convention, not a measurement.
- **Chemistry verdicts on specific pairs are soft.** We wrote down our best
  guesses (in public, with timestamps) and the rest of the season will grade
  them.

## The punchline

If you remember one thing: **in pro doubles, who you are beats who you're
with — and if you must choose a partner, choose your equal.**

*(All code, data, and the full technical writeups live in this repository.
A fancier version of the model — with month-by-month skill tracking — is
fitting as this file is written.)*
