# Phase 2b — joint pool, context-averaged value, and the must-buy test

Status: 2026-09-04. Picks up HANDOFF.md's Next Step 1 (test the 50/50
gender split) and ends somewhere different: the split was a side issue,
the value basis was the problem, and the "star discount" has a precise
meaning. Reproduce with

    python value_cap/shapley_value.py                          # ~4 min -> player_value_shapley.csv
    python value_cap/phase2_joint_pool.py --quick              # pairs + implied split, V basis
    python value_cap/phase2_joint_pool.py --quick --value shapley
    python value_cap/phase2_joint_pool.py --value shapley --must-buy "Anna Leigh Waters" --modes joint

Raw sweep output for every number below is in `phase2_joint_pool_results.txt`.
Everything runs the production tie model (`phase1_value_model.tie_win_prob`:
race-to-11 DP + weakest-link gamma, DreamBreaker singles-gap model); nothing
here is a proxy calculation. The roster optimizer scores REAL tie win
probability over the top-25 proxy candidates per gender per budget split;
K=25 vs K=80 gives the same answer to 3 decimals (0.597 vs 0.600).

## 0. The handoff's instruments reproduce

Same anchor shapes (doubles ranks; #79/#80 as the "near-useless" bench,
#60 as replacement), same $30k floor, same two $10M sub-pools:

| pair | ties at | handoff alpha | this run |
|---|---|---|---|
| A  #3M/#2F + 2 scrubs vs (k, k+1, #60) | k=17 | 0.88 | 0.879 |
| B  #10 + 2 scrubs vs (k, k+1, #60) | k=24 | 0.76 | 0.746 |

So the machinery matches and everything below is comparable.

## 1. The joint pool: the split is an output, and it fixes the men's market

`price_i = floor + V_i^alpha / sum_j V_j^alpha * (20*cap - 120*floor)` with j
over all 120 priced players. V is defined relative to the same-gender
replacement player, so it is offset-free and comparable across genders
(the cross-gender house rule is about ratings, not about win-probability
contributions, which cancel the offset by construction).

Implied women's share of the $20M: 54% (alpha 0.6) → 57% (1.0) → 58% (1.2)
on V; 55-59% on phi (below). The data want roughly 57/43, not 50/50 and
not 60/40.

On the indifference pairs the joint pool barely matters (alphas move by
0.02-0.08). Where it matters is the must-buy test (§4) for MEN: total
context-averaged value in the men's top-60 is 3.22 vs 4.74 for the women,
and the 50/50 split hands both markets $10M, so every man is ~16%
overpriced relative to every woman. Under the split Ben Johns is a
must-AVOID at every alpha tried (P = 0.32-0.41, never fair); under the
joint pool he is fairly priced from alpha 0.9 up (0.47-0.50). Keep the
joint pool.

## 2. No single power law on Phase 1's V reconciles the pairs

Adding a #1-anchored pair (C) and a #1-plus-two-replacements pair (F) to
the handoff's A and B, the alpha that price-matches a pair climbs
monotonically with the anchor's rank:

| pair (anchor + 2 scrubs vs k, k+1, #60) | ties at | alpha split | alpha joint |
|---|---|---|---|
| B  #10 | k=24 | 0.746 | 0.703 |
| A  #3M / #2F | k=17 | 0.879 | 0.855 |
| C  #1 (Johns + Waters) | k=8 | 1.470 | 1.393 |
| F  #1 + 2 replacements vs (k, k+1, k+2) | k=10 | 1.683 | 1.757 |

"Two stars + replacement vs depth" shapes never tie at any k: the third
roster slot only enters the DreamBreaker, so no depth block beats two
stars. Those pairs carry no price information and were dropped.

The must-buy panel on V says the same thing from the other side (§4
table, "V basis" rows): Waters is a bargain at every alpha ≤ 1.2 (P =
0.62-0.71) while Johns and Bright are must-avoids until ~1.07-1.09, and
that flip is Waters becoming unaffordable, not their own price. There
is no alpha at which V-pricing is fair to both the #1 and the #2.

## 3. Why: V is measured next to a replacement partner

Phase 1's V drops a player into a roster of #60s. Their doubles partner
is the #60, the weakest-link gap penalty (GAMMA*|gap|) is at its largest,
and it damps a player in proportion to how good they are. That is a
measurement-context artifact, not a property of the player.

`shapley_value.py` replaces the context with the league itself: draw 2
same-gender + 3 other-gender teammates and a 6-player opponent uniformly
from the priced pool (top 60 per gender = exactly what 20 teams roster),
and average P(with player) − P(with replacement) over 3,000 draws (common
random numbers across players, se ≈ 0.0005). Call it phi.

| player | V | phi | phi/V |
|---|---|---|---|
| Anna Leigh Waters | 0.422 | 0.435 | 1.03 |
| Anna Bright | 0.342 | 0.288 | 0.84 |
| Ben Johns | 0.269 | 0.215 | 0.80 |
| rest of each top-10 | | | 0.71-0.79 |
| rank ~30 | | | 0.45-0.58 |
| rank 60 | | | 0.11-0.29 |

phi is more convex than V: the very top is worth MORE than V says, depth
is worth a lot less (in a real roster the third player usually doesn't
play). Waters is the only player whose context-averaged value exceeds
her replacement-context value.

## 4. The must-buy test — the right instrument, and what it says on phi

`must_buy(pid, alpha)`: best $1M roster that MUST include the player vs
best $1M roster that must EXCLUDE them, each side best-responding to the
other for two rounds, scored head-to-head. 0.5 = fairly priced; above =
bargain (a must-buy); below = overpriced (a must-avoid). This is the
brief's actual question ("does a dominant strategy fall out") asked with
both sides playing optimally. The indifference pairs are not: A and B
anchor on star-plus-two-scrubs builds, which a GM should be charged for.

Injection floor at alpha 1.0, phi, joint: halving Patriquin's price →
0.665, ×1.5 → 0.339 (Newman 0.587 / 0.423). A reading near 0.49 means
the price is right to within a few percent; the instrument has teeth.

P(best roster with player beats best roster without), phi basis, joint pool, $30k floor:

| player | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 | 1.4 | 1.6 |
|---|---|---|---|---|---|---|---|---|---|
| Waters (#1F) | .642 | .647 | .597 | edge | INF | INF | INF | INF | INF |
| Bright (#2F) | .414 | .383 | .379 | .497 | .485 | .449 | .430 | INF | INF |
| Johns (#1M) | .385 | .395 | .428 | .503 | .481 | .471 | .467 | .452 | .424 |
| JW Johnson (#2M) | .373 | .388 | .407 | .490 | .492 | .484 | .490 | .476 | .457 |
| Patriquin (#3M) | .371 | .384 | .384 | .497 | .485 | .487 | .492 | .479 | .466 |
| Jorja Johnson (#4F) | .406 | .384 | .389 | .489 | .484 | .478 | .489 | .456 | .429 |
| Fahey (#10F) | .412 | .391 | .386 | .486 | .488 | .485 | .500 | .464 | .455 |
| Newman (#10M) | .478 | .387 | .397 | .497 | .508 | .515 | .519 | .520 | .512 |

INF = the player plus the five cheapest teammates already exceeds the cap.
The $5,000-floor runs of the same panel are identical to within the
optimizer's budget-grid noise (§6 explains why they must be).

Same panel on the V basis (joint pool) for the three marquee names:
Waters .70/.70/.69/.68/.66/.66/.62 (never fair, INF from 1.26);
Bright .36/.32/.32/.32/.36/.56/.55 (crossover 1.07);
Johns .36/.36/.35/.38/.38/.52/.50 (crossover 1.09).

Read the phi table by column, not by row. Two regimes:

- **alpha ≤ 0.8: one dominant strategy — buy Waters.** Every other
  star reads 0.37-0.43 for the same reason: the best roster WITHOUT them
  is a Waters roster, and it wins ~60/40. Waters' own line (0.60-0.65)
  is the mirror image.
- **alpha ≥ 0.9: no dominant strategy.** Waters is no longer rosterable
  with any cast worth having, and everyone from #1 to #10 in both
  genders sits at 0.47-0.51 out to alpha 1.6. Alpha is nearly
  irrelevant to fairness in this regime; it only decides how steep the
  dollar curve looks.

So the whole competitive-balance question is one player.

## 5. The Waters fact, stated plainly

Waters' context-averaged value is 5.5% of the league's total (0.435 of
7.96 across the 120 rostered players). A team is 5% of the league (1 of
20). **By the model's own accounting she is worth more than an entire
team's payroll.** Any alpha ≥ 1 prices her above what a team can spend
(alone above the cap from alpha 1.06 joint / 1.21 split); any alpha
below ~0.88 discounts her enough that a Waters roster beats the best
roster without her.

Fine scan (budget grid $5k), phi, joint:

| alpha | Waters price | Waters must-buy P | Patriquin must-buy P |
|---|---|---|---|
| 0.80 | $696k | .605 | .394 |
| 0.82 | $717k | .588 | .405 |
| 0.84 | $738k | .589 | .417 |
| 0.86 | $759k | .550 | .444 |
| 0.88 | $781k | .508 | .491 |
| 0.90 | $804k | .454 | .497 |
| 0.92 | $827k | INF | .501 |

Waters crosses 0.5 at alpha ≈ 0.88 and drops out of the feasible set at
0.91; Patriquin (standing in for every other star) becomes fair at the
same 0.88 — the two lines are the same event seen from both sides.

The "star discount" the handoff found (alpha < 1) is therefore not a
statement about how stars should be priced in general — for everyone
else alpha is a free dial above 0.9 — it is the exact amount by which
the league's best player must be discounted to be rosterable at all,
and the fair window where she is rosterable AND not a must-buy is
narrow: alpha ≈ 0.88-0.91, price ≈ $780-810k, supporting cast = five
floor players. Below the window she is a dominant strategy; above it
she is not in the league.

This is a real property of a $1M / 6-player / 20-team cap with this
talent distribution, not a modeling artifact: the same test names no
other player as a dominant strategy at any alpha, and the free search at
alpha 0.89 cycles through three unrelated rosters (Bright-anchored,
depth-anchored, Johns-anchored) that each beat the previous by 0.2-1.0
points — the signature of a fair price list.

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
does real work: cheapest legal roster on the joint pool is $623k at
alpha 0.6, $529k at 0.8, $451k at 1.0, $387k at 1.2 (binds from ~0.9).

## 7. Price list at alpha 0.89, phi basis, joint pool, $30k floor

Women's share 57.0%. Men: Johns $437k, JW Johnson $402k, Patriquin
$393k, Tardio $383k, Alshon $378k, Daescu $311k, Staksrud $305k, Oncins
$299k, Devilliers $282k, Newman $251k … #58-60 at the $30k floor.
Women: Waters $793k, Bright $558k, Todd $453k, Jorja Johnson $446k, Fahey
$427k, Jade Kawamoto $418k, Rohrabacher $400k, Pisnik $394k, Jackie
Kawamoto $376k, Black $369k … Bouchard $53k, H. Jansen $45k.

## What this does NOT settle

- alpha ≈ 0.89 is pinned by Waters' feasibility edge, not fitted to
  anything else; if the league's real cap, team count, or roster size
  differs, recompute the edge (it moves with 1/N_teams, not with the
  floor). 20 teams and the $500k min-spend rule are still unconfirmed
  (HANDOFF.md Next Step 3).
- No injury/absence draw yet (Phase 1's open item); phi assumes every
  player is available.
- phi is uniform over the priced pool. A realistic-roster weighting
  (teams cluster talent) would change depth values at the margin; the
  top-10 ordering is robust to it because the top-10's phi/V ratios are
  flat.
- The must-buy test has ~±0.01 search noise (budget grid $20k, K=25
  candidates); readings within 0.48-0.52 are "fair", not ranked.
