# Phase 2 — quick pass: pricing sweep + a real (not cherry-picked) search

Status: 2026-09-02, rough pass. Run: `python value_cap/phase2_price_model.py`.

Formula: `price = floor + (V^alpha / sum(V^alpha)) * pool`, fit separately
per gender. Floor fixed at $30,000 (the original brief's example number,
not fit), alpha swept across 0.5/1.0/1.5/2.0/3.0 rather than picked, 20
teams assumed, $1M cap split 50/50 into men's/women's league sub-pools.
All placeholders for tomorrow's real fit -- see the script docstring for
the full assumption list.

Two real findings came out of this pass, both more useful than a clean
result would have been.

## 1. Alpha has a hard ceiling, and it's lower than you'd guess

At alpha=2.0, Anna Leigh Waters alone prices at **$1,001,237 -- over the
entire $1,000,000 team cap.** At alpha=3.0, Ben Johns and Waters both
individually exceed the cap. This isn't a rare edge case: even at
alpha=0.5, the mildest setting tried, pricing the single best man
(Ben Johns) and single best woman (Waters) together plus four
bare-minimum-priced teammates already runs $5,291 over the cap.

So "does it need to be exponential" has a sharper answer than "yes/no":
**whatever alpha ends up chosen has to satisfy a real constraint** (no
single player, or reasonable combination, can price above what a team
could ever pay), not just "does it feel like a fair star premium." That
constraint depends on the floor and the pool size too -- it's a
three-way relationship, not a single free dial. This bounds tomorrow's
fit before any sample roster gets involved.

## 2. "Win probability vs. a replacement-level team" isn't the right test for comparing $1M rosters

The search found the best achievable $1M roster under each alpha and
checked its real win probability (the full joint model, not just added
V) against the Phase 1 replacement-level opponent (a team of nothing but
60th-ranked players). Every alpha's best roster came back around
90-98% -- including rosters made of comfortably-good-but-not-elite
players (Rafael Lenhard, Alexander Crum, Mohaned Alhouni; Kate Fahey,
Mari Humberg, Vivienne David at alpha=1.0).

That's not a bug, it's a saturation problem: once a roster has SIX
players who are all meaningfully better than replacement (not just one,
which is what Phase 1's V was built to measure), the edge compounds
across all four regular games at once, and beating an all-replacement
team stops being a hard test almost immediately. It means this
benchmark can't distinguish a stars-heavy build from a balanced one --
both stomp a scrub team. Confirmed directly: at alpha 1.0/1.5/2.0/3.0,
excluding the top 3 priciest players per gender from the search changed
NOTHING -- the optimizer was never using them anyway, because against a
scrub opponent they aren't worth their price. Only at alpha=0.5 (where
Waters is cheap enough to include) did excluding the top-3 actually cost
win probability (0.981 -> 0.914).

**The fix, needed before any real archetype comparison means anything:**
score candidate rosters against each other (or against one fixed strong
$1M reference roster), not against Phase 1's all-replacement team. That
benchmark was the right one for computing individual V -- it is not the
right one for judging which roster-building strategy wins under a cap.
Flagging this now so tomorrow's sample-roster fit uses a fair test from
the start.

## What this pass did NOT do

- Did not fit alpha or the floor to anything -- swept, not chosen.
- Did not build the roster-vs-roster benchmark described above.
- Did not use real archetype rosters (waiting on Chad's picks -- rankings
  are fine, translated to real names against `data/v2_players.csv`
  order).
