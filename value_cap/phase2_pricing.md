# Phase 2 — joint pool, context-averaged value, and the must-buy test

Status: 2026-09-04 (evening; numbers refreshed on the self-consistent
pool). Picks up HANDOFF.md's morning Next Step 1 (test the 50/50 gender
split) and ends somewhere different: the split was a side issue, the value
basis was the problem, and the "star discount" has a precise meaning.
Reproduce with

    python value_cap/shapley_value.py                  # ~15 min -> player_value_shapley.csv (phi + pool)
    python value_cap/phase2_pricing.py --quick         # implied split + indifference pairs, phi basis
    python value_cap/phase2_pricing.py --quick --value total       # same on Phase 1's V (morning numbers)
    python value_cap/phase2_pricing.py --must-buy "Anna Leigh Waters" --modes joint

Raw sweep output for every number below is in `phase2_pricing_results.txt`.
Everything runs the production tie model (`phase1_value_model.tie_win_prob`:
race-to-11 DP + weakest-link gamma, DreamBreaker singles-gap model); nothing
here is a proxy calculation. The roster optimizer scores REAL tie win
probability over the top-25 proxy candidates per gender per $10k budget
split, plus the must-player's cheapest legal completion so "infeasible"
means infeasible; K=25 vs K=80 candidates gives 0.481 vs 0.473 on the
same reading.

## 0. The handoff's instruments reproduce

Same anchor shapes (doubles ranks; #79/#80 as the "near-useless" bench,
#60 as replacement), same $30k floor, same two $10M sub-pools, Phase 1's
V as the basis (`--value total`):

| pair | ties at | handoff alpha | this run |
|---|---|---|---|
| A  #3M/#2F + 2 scrubs vs (k, k+1, #60) | k=17 | 0.88 | 0.879 |
| B  #10 + 2 scrubs vs (k, k+1, #60) | k=24 | 0.76 | 0.746 |

So the machinery matches and everything below is comparable.

## 1. The joint pool: the split is an output, and it fixes the men's market

`price_i = floor + v_i^alpha / sum_j v_j^alpha * (20*cap - 120*floor)` with j
over all 120 priced players. Value is defined relative to the same-gender
replacement player, so it is offset-free and comparable across genders
(the cross-gender house rule is about ratings, not about win-probability
contributions, which cancel the offset by construction).

Implied women's share of the $20M on phi: 54% (alpha 0.6) → 56% (0.88) →
57% (1.0) → 58% (1.2); the same 54-58% on V. The data want roughly 57/43,
not 50/50 and not 60/40.

On the indifference pairs the joint pool barely matters (alphas move by
0.02-0.08). Where it matters is the must-buy test (§4) for MEN: total
context-averaged value in the men's pool is 3.42 vs 4.80 for the women,
and the 50/50 split hands both markets $10M, so every man is ~14%
overpriced relative to every woman. Under the split Ben Johns is a
must-AVOID at every alpha tried (P = 0.30-0.41, never fair); under the
joint pool he is fairly priced from alpha 0.9 up (0.47-0.49). Keep the
joint pool.

## 2. No single power law on Phase 1's V reconciles the pairs

Adding a #1-anchored pair (C) and a #1-plus-two-replacements pair (F) to
the handoff's A and B, the alpha that price-matches a pair climbs
monotonically with the anchor's rank, on either basis:

| pair (anchor + 2 scrubs vs k, k+1, #60) | ties at | V split | V joint | phi split | phi joint |
|---|---|---|---|---|---|
| B  #10 | k=24 | 0.746 | 0.703 | 0.525 | 0.503 |
| A  #3M / #2F | k=17 | 0.879 | 0.855 | 0.614 | 0.598 |
| C  #1 (Johns + Waters) | k=8 | 1.470 | 1.393 | 0.939 | 0.891 |
| F  #1 + 2 replacements vs (k, k+1, k+2) | k=10 | 1.683 | 1.757 | 1.212 | 1.204 |

"Two stars + replacement vs depth" shapes never tie at any k: the third
roster slot only enters the DreamBreaker, so no depth block beats two
stars. Those pairs carry no price information and were dropped.

The must-buy panel on V says the same thing from the other side (§4,
"V basis" rows): Waters is a bargain at every alpha ≤ 1.2 (P = 0.59-0.72)
while Johns and Bright are must-avoids until ~1.07-1.09, and that flip
is Waters becoming unaffordable, not their own price. There is no alpha
at which V-pricing is fair to both the #1 and the #2.

## 3. Why: V is measured next to a replacement partner

Phase 1's V drops a player into a roster of #60s. Their doubles partner
is the #60, the weakest-link gap penalty (GAMMA*|gap|) is at its largest,
and it damps a player in proportion to how good they are. That is a
measurement-context artifact, not a property of the player.

`shapley_value.py` replaces the context with the league itself: draw 2
same-gender + 3 other-gender teammates and a 6-player opponent uniformly
from the priced pool (60 per gender = exactly what 20 teams roster), and
average P(with player) − P(with replacement) over 3,000 draws (common
random numbers across players, se ≈ 0.001-0.003). Call it phi.

The pool is part of the definition, so it is solved for rather than
inherited from V: start from the top 60 per gender by V_total, measure
phi for the top 80, re-pick the top 60 by phi, repeat. It converges in
three iterations and all the churn is at the #56-60 edge, where phi is
0.005-0.015 and the price is within $25k of the floor (5 men and 1 woman
swap in; the departing men had NEGATIVE phi — worse than the #60 in a
real roster despite a positive V). `pool.py` is the one place the pool
is defined; `--value total` keeps the V_total pool for §0.

| player | V | phi | phi/V |
|---|---|---|---|
| Anna Leigh Waters | 0.422 | 0.437 | 1.04 |
| Anna Bright | 0.342 | 0.290 | 0.85 |
| Ben Johns | 0.269 | 0.221 | 0.82 |
| rest of each top-10 | | | 0.72-0.81 |
| pool rank ~30 | | | 0.46 |
| pool rank 60 | | | 0.17-0.27 |

phi is more convex than V: the very top is worth MORE than V says, depth
is worth a lot less (in a real roster the third player usually doesn't
play). Waters is the only player whose context-averaged value exceeds
her replacement-context value. The pool also stops being a doubles
top-60: Matthew Barlow (doubles #117), Genie Bouchard (#216) and Hoang
Nam Ly (#102) price in on the DreamBreaker channel — the Phase 1
smell-test (Haworth-style DB specialists) showing up in the pool itself.

## 4. The must-buy test — the right instrument, and what it says on phi

`must_buy(pid, alpha)`: best $1M roster that MUST include the player vs
best $1M roster that must EXCLUDE them, each side best-responding to the
other for two rounds, scored head-to-head. 0.5 = fairly priced; above =
bargain (a must-buy); below = overpriced (a must-avoid). This is the
brief's actual question ("does a dominant strategy fall out") asked with
both sides playing optimally. The indifference pairs are not: A and B
anchor on star-plus-two-scrubs builds, which a GM should be charged for.

Injection floor at alpha 1.0, phi, joint: halving Patriquin's price →
0.627, ×1.5 → 0.332 (base 0.481); Newman 0.580 / 0.415 (base 0.505). A
reading near 0.49 means the price is right to within a few percent; the
instrument has teeth.

P(best roster with player beats best roster without), phi basis, joint pool, $30k floor:

| player | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 | 1.4 | 1.6 |
|---|---|---|---|---|---|---|---|---|---|
| Waters (#1F) | .646 | .656 | .602 | INF | INF | INF | INF | INF | INF |
| Bright (#2F) | .353 | .361 | .402 | .511 | .483 | .455 | .417 | INF | INF |
| Johns (#1M) | .350 | .359 | .400 | .489 | .477 | .474 | .466 | .446 | .409 |
| JW Johnson (#2M) | .347 | .359 | .392 | .491 | .481 | .481 | .481 | .477 | .456 |
| Patriquin (#3M) | .343 | .356 | .388 | .490 | .481 | .479 | .481 | .480 | .461 |
| Jorja Johnson (#3F) | .362 | .365 | .383 | .505 | .497 | .482 | .474 | .463 | .432 |
| Fahey (#10F) | .371 | .359 | .392 | .486 | .493 | .484 | .484 | .473 | .448 |
| Newman (#10M) | .332 | .374 | .393 | .483 | .505 | .512 | .509 | .519 | .517 |

INF = no legal roster contains the player (the optimizer always tries
their cheapest completion). The $5,000-floor runs of the same panel
agree to within 0.01 everywhere (§6 explains why they must).

Same panel on the V basis (joint pool) for the three marquee names:
Waters .72/.69/.69/.67/.66/.66/.59/INF/INF (never fair);
Bright .34/.33/.32/.31/.34/.56/.55/.49/.44 (crossover 1.07);
Johns .35/.36/.35/.38/.36/.52/.54/.49/.48 (crossover 1.09).
And on phi under the SPLIT pool: Bright fair only at 1.0-1.2, Johns never
(0.30-0.41), Waters a bargain through 0.9 and infeasible from 1.0.

Read the phi table by column, not by row. Two regimes:

- **alpha ≤ 0.8: one dominant strategy — buy Waters.** Every other
  star reads 0.33-0.40 for the same reason: the best roster WITHOUT them
  is a Waters roster, and it wins ~60/40. Waters' own line (0.60-0.66)
  is the mirror image.
- **alpha ≥ 0.9: no dominant strategy.** Waters is no longer rosterable,
  and everyone from #1 to #10 in both genders sits at 0.45-0.51 out to
  alpha 1.6 (Johns drifts to 0.41 by 1.6; Newman to 0.52). Alpha is
  nearly irrelevant to fairness in this regime; it only decides how
  steep the dollar curve looks.

So the whole competitive-balance question is one player.

## 5. The Waters fact, stated plainly

Waters' context-averaged value is 5.3% of the league's total (0.437 of
8.21 across the 120 rostered players). A team is 5% of the league (1 of
20). **By the model's own accounting she is worth more than an entire
team's payroll.** Any alpha ≥ 1.08 prices her alone above the cap
(1.22 under the split); any alpha below ~0.84 discounts her enough that
a Waters roster beats the best roster without her.

Fine scan (budget grid $5k), phi, joint, $30k floor:

| alpha | Waters price | cheapest Waters roster | Waters must-buy P | Patriquin must-buy P |
|---|---|---|---|---|
| 0.80 | $678k | | .605 | .390 |
| 0.82 | $698k | | .575 | .414 |
| 0.835 | $714k | $985k | .529 | |
| 0.84 | $719k | $989k | .495 | .482 |
| 0.845 | $725k | $993k | .464 | |
| 0.85 | $730k | $997k | .459 | |
| 0.855 | $735k | $1,001k | INF | |
| 0.86-0.92 | $741-808k | > cap | INF | .487-.491 |

Waters crosses 0.5 at alpha ≈ 0.84 and drops out of the feasible set at
0.853; Patriquin (standing in for every other star) becomes fair at the
same 0.84 — the two lines are the same event seen from both sides.

The "star discount" the handoff found (alpha < 1) is therefore not a
statement about how stars should be priced in general — for everyone
else alpha is a free dial above 0.9 — it is the exact amount by which
the league's best player must be discounted to be rosterable at all,
and the window where she is rosterable AND not a must-buy is a
knife-edge: **alpha ≈ 0.84-0.85, price ≈ $720-730k, supporting cast =
the five cheapest players in the pool at ~$53k each**. Below the window
she is a dominant strategy; above it she is not in the league.

Where the window sits depends on what the cheapest cast costs. The
morning's V_total pool included men with negative phi, priced at
exactly the floor (zero value share), which made a Waters roster
feasible up to alpha 0.91 and put the window at 0.88-0.91 in this
document's first draft. The self-consistent pool has only positive-phi
members, and at a sub-linear alpha five near-zero values still carry
~0.7% of the league's value weight — enough to move the edge from 0.91
to 0.85. The floor identity (§6) is untouched by this; it is the value
weight of the cast, not its dollar floor, that binds.

This is a real property of a $1M / 6-player / 20-team cap with this
talent distribution, not a modeling artifact: the same test names no
other player as a dominant strategy at any alpha, and the free search in
the window cycles through unrelated rosters (Bright-anchored,
depth-anchored, Johns-anchored) within a point of each other — the
signature of a fair price list.

## 6. The floor is cosmetic

The handoff proved the floor cancels out of any same-size roster
comparison. It also cancels out of the cap constraint: with a pool of
exactly 6×20 priced players and cap = pool/20, a 6-player roster costs
`6f + S(20C − 120f)` where S is its share of total value, so cost ≤ C
iff S ≤ 1/20 — for every floor. Off-pool players (share 0, price f)
obey the same identity. The floor decides nothing about which rosters
are legal or which strategy wins; it only labels the bottom of the
dollar scale. The handoff's finding 5 (floor/ceiling decoupling) is moot
here, and the $500k minimum-spend rule is the one place the floor still
does real work: the cheapest legal roster on the phi joint pool is $440k
at alpha 0.6, $345k at 0.8, $309k at 0.9, $281k at 1.0, $240k at 1.2 —
the rule BINDS at every alpha on this basis (on V it only bound from
~0.9, because V's depth is worth more).

## 7. Price list at alpha 0.845, phi basis, joint pool, $30k floor

Women's share 55.8%. Men: Johns $420k, JW Johnson $387k, Patriquin
$380k, Tardio $371k, Alshon $364k, Daescu $304k, Staksrud $298k, Oncins
$291k, Devilliers $276k, Newman $246k, Sock $225k, Huynh $211k … #57-60
(Barlow, Alhouni, Ly, Goldin) $51-55k.
Women: Waters $725k, Bright $520k, Todd $431k, Jorja Johnson $421k, Fahey
$404k, Jade Kawamoto $397k, Rohrabacher $380k, Pisnik $376k, Jackie
Kawamoto $357k, Black $353k, Sewing $325k, Townsend $306k … #57-60
(Bouchard, Maddox, C. Smith, Campbell) $53-65k.

## What this does NOT settle

- alpha ≈ 0.845 is pinned by Waters' feasibility edge, not fitted to
  anything else; if the league's real cap, team count, or roster size
  differs, recompute the edge (it moves with 1/N_teams, not with the
  floor). 20 teams and the $500k min-spend rule are still unconfirmed
  (HANDOFF.md Next Step 3).
- The edge also moves with the value weight of the five cheapest pool
  members (§5), i.e. with how the pool's bottom is defined. A realistic-
  roster weighting for phi (below) would change that bottom first.
- No injury/absence draw yet (Phase 1's open item); phi assumes every
  player is available. A Waters roster of five ~$53k players is the most
  availability-fragile build in the league.
- phi is uniform over the priced pool. A realistic-roster weighting
  (teams cluster talent) would change depth values at the margin; the
  top-10 ordering is robust to it because the top-10's phi/V ratios are
  flat.
- The must-buy test has ~±0.01 search noise (budget grid $10k, K=25
  candidates); readings within 0.48-0.52 are "fair", not ranked.

## 8. SHIPPED (2026-09-04, night): the franchise tag at alpha 1

User call after the draft simulations and the pool/floor sweep: ship the
tag rule. The rule, in plain language (`prices_tagged` in
`phase2_pricing.py`):

1. Price everyone off the curve at alpha 1 -- share of the $20M league
   pool = share of total phi, one joint pool for men and women.
2. Franchise-tag the one player the curve cannot fit. Waters' curve price
   is $903k of a $1M cap; no legal six-player roster can carry it. Her
   price is set directly to the most a team can pay and still field a
   legal roster: cap minus the cheapest legal completion (the two cheapest
   priced women plus the three cheapest priced men). At a $30k floor
   that is **$769k**.
3. The $134k gap between her curve price and her tag is spread back over
   the other 119 priced players in proportion to their value, so the
   pool still sums to twenty caps (+1% each, iterated because the
   cheapest completion itself moves a little).
4. Nobody else changes rules: one tagged player, one price list, alpha
   stays at 1. This replaces the §5/§7 alpha ≈ 0.845 list, which bought
   Waters' rosterability by discounting every other star and
   surcharging every role player 10-15%.

The list: `price_list.md` / `price_list.csv` (`price_list.py`), with the
doubles-only value and rank next to each price so the DreamBreaker
(singles) lift is visible -- Todd (#5F doubles) and Fahey (#10F) price
3rd and 6th among women because they are the best two singles players
in the field after Waters; Jade Kawamoto and Pisnik have 0 singles games
and sit on the imputed prior. Top of the list: Waters $769k, Bright
$613k, Todd $489k, Jorja Johnson $475k, Johns $474k, Fahey $453k, Jade
Kawamoto $444k, JW Johnson $430k, Patriquin $421k, Rohrabacher $421k.
Women's share 56.6%.

What the tag list does in a league with scarcity (`draft_sim.py`, 20
teams, 6 rounds, owners with belief noise, seasons on true values;
`draft_sim_tag*.md`, `draft_sim_a0.845.md`):

| price list / board | snake, perfect info: spread / strongest | snake, 10% owner error: spread / strongest / slot-1 title | top-30 undrafted | mean spend |
|---|---|---|---|---|
| tag, priced 60+60 + real 2026 MLP fill-ins (`mlp2026`, the default board) | 4.3 pts / 66.1% | 4.4 / 65.7% / 36% | none | $975k |
| tag, priced 60+60 + best-60 free agents (`best60`) | 4.2 / 65.3% | 4.6 / 65.7% / 34% | none | $970k |
| tag, real 2026 MLP participants only (`mlp2026only`) | 4.6 / 68.2% | 7.0 / 72.3% / 47% | -- (77 priced for 120 slots; talent-starved) | $838k |
| alpha 0.845, `best60` | 9.6 / 67.2% | 9.7 / 67.7% | Bellamy #29M in 100% of drafts | $926k |

Reads: (a) the strongest team is always the slot-1 pick, Waters plus the
cheap DreamBreaker singles specialists the spare ~$230k buys (Gabriel
Joseph, Bellamy); slots 2-20 sit at 47-53% and ~3.4% title each. (b) The
tag list beats the alpha 0.845 list on every read: half the parity
spread, no top-30 player left on the board, the cap actually spent.
(c) The real-2026-participant board does not shrink Waters' prize (the
user asked; it makes it bigger, because the fill-in tail is weaker).
(d) A 66% team with a ~1-in-3 title shot at pick 1 is a normal pro-sports
favorite (top NBA/EPL preseason favorites carry 30-50%); it is a lottery
prize attached to one player being 5.3% of the league's value, not a
dynasty -- user call: "it is what it is", ship it and say so.

Levers that do NOT move it (`pool_floor_sweep.md`, `pool_floor_sweep.py`):
the floor ($10k-$75k: identical drafts at every floor); the priced-pool
size (80 or 100 per gender drops Waters' team to 57% only by spreading
the $20M over players nobody drafts -- spend falls to $670-910k and the
prize moves to Bright's team at 61-66%; pricing the 120 draftable against
a deeper replacement makes her CHEAPER and her team STRONGER, 73% / 50%
title vs #100). The tag accounting alternative (charge the full $903k,
lower floor for the tagged team) reaches the low 50s and is the only
thing that has ever moved her; rejected because it prices one player
above a team's fair share, which is what the tag exists to avoid.

Template-strategy rosters without scarcity (`draft_strategies_tag.md`,
`template_season.md`): no dominant blueprint on the tag list -- Balanced
four / DreamBreaker specialist / Quant within a point of each other,
Superstar-Waters 12.3% title in an 8-roster field (parity 12.5%). Those
rosters share players, so the draft sim above is the real league read.

Probe the user asked about: Waters plus the five BEST leftover fill-ins
after a full draft beats the drafted field 55.6% (best60 board) / 56.1%
(mlp2026) / 42.7% (mlp2026only); plus the five worst, 16%. Same story as
the draft: the value is in her, the cheap completion decides the margin.

Reproduce:

    python value_cap/price_list.py                              # the shipped list
    python value_cap/draft_sim.py --tag --board mlp2026         # the league read (~2 min)
    python value_cap/pool_floor_sweep.py                        # pool x floor grid (~20 min; phi_pool{60,80,100}.csv cached)
