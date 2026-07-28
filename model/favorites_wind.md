# Does wind flatten the favorite's edge? (data-referenced nulls)

## 1. Game level: share − ½ = a + b·skill + c·(wind/10) + d·skill·(wind/10)

skill = v2-expected share − ½. d is the test: negative = wind compresses the favorite's conversion of skill into points. b at 0 mph vs b + 1.5d at 15 mph shows the size.

| setting | games | b (skill slope) | d (skill×wind) [95% CI] | slope at 15 mph |
|---|---|---|---|---|
| outdoor | 24819 | 1.040 | +0.002 [-0.060, +0.064] | 1.042 |
| indoor | 11699 | 1.108 | -0.080 [-0.150, +0.020] | 0.988 |

## 2. Rally level: favorite−underdog SERVE-RALLY win-rate gap vs wind

v2 only labels which side is the favorite (|eta| ≥ 0.1 required); gap uses full-game serve tallies from decider_serve_splits (deciders + all MLP games — close-match-selected, so read the wind slope, not the level).

| setting | games | mean gap | wind slope c per +10 mph [95% CI] |
|---|---|---|---|
| outdoor | 1922 | +0.098 | -0.0215 [-0.0406, +0.0001] |
| indoor | 1904 | +0.113 | -0.0311 [-0.0518, -0.0096] |

## 3. Rally level proper: P(server wins THIS rally) — binomial logit

Each serve rally is a 0/1; with covariates constant within a match-side the Bernoulli series collapses losslessly to its (wins, attempts) sufficient statistic, so this fits the rally series exactly while weighting every rally once (fixing the equal-weight-per-game approximation of regression 2): logit p = a + b·adv + c·(wind/10) + d·adv·(wind/10), where adv = serving team's v2 eta advantage (signed). d < 0 = wind erodes the better team's rally edge. Cluster bootstrap by event.

| setting | rallies | b (adv) | d (adv×wind) [95% CI] |
|---|---|---|---|
| outdoor | 98341 | 0.458 | -0.017 [-0.098, +0.058] |
| indoor | 94842 | 0.510 | -0.060 [-0.129, +0.014] |

---
*All nulls are within-data: the interaction's zero, the indoor arm, and calm games. No simulation used. Caveats: current-form v2 retroactive (fine for interactions); outdoor labels heuristic; regressions 2–3 use the close-match-selected serve-splits sample (deciders + MLP) — slopes/interactions are the objects, not levels.*
