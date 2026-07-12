# analysis.md — player value & pair chemistry, 2026 MLP + PPA

## Headlines

1. **Individual value dwarfs chemistry.** Player-value spread is sd = 2.21 points/game; pair chemistry spread is sd = 0.47. Who you are matters ~5× more than who you're standing next to. Chemistry exists league-wide (sd_d is bounded away from zero) but is small, and no single pair's chemistry is certifiable with high confidence — even 100+ game dyads carry ±0.45 posteriors.
2. **Anna Leigh Waters is #1 by a wide margin** (+7.74 pts/game vs +6.09 for #2 Anna Bright) — but see the cross-gender caveat: rankings are rock-solid *within* gender, while the men-vs-women alignment is a modeling convention, not a data statement.
3. **Bright + Patriquin's dominance is star power, not magic.** Their mixed chemistry is +0.10 ± 0.46 (88th percentile, P(>0) = 0.57) over 138 games — mildly positive, far from certain. Their expected margin comes almost entirely from both being top-5 players.
4. **Waters + Johns are exactly the sum of their parts**: chemistry -0.00 (56th pct) across 111 games.
5. **The Waters + Bright superteam slightly *under*-performs its parts** (-0.15, 3rd percentile, 98 games) — two #1-caliber players don't add a bonus on top of being overwhelming favorites.
6. **Skill transfers across contexts almost perfectly**: the mixed/mens/womens deviation scale is sd_w = 0.13 — negligible. A player's gendered-doubles level is their mixed level.

## What is (and is not) identifiable — the "actor vs partner" question

The original brief asked to separate each player's **actor effect** (own skill) from their **partner effect** (how much they elevate whoever stands next to them). With team-level margins that split is **not identifiable**: if a team's strength is `actor_i + partner_j→i + actor_j + partner_i→j`, then only the sums `(actor_i + partner_i)` and `(actor_j + partner_j)` ever enter the likelihood — any reallocation between a player's actor and partner components produces identical predictions for every game, including counterfactual pairings. No amount of data fixes this; it's structural. (Kenny's classic SRM separates them because dyadic outcomes there are *directional* — i's rating of j differs from j's rating of i. A game margin has no direction within a team.)

### The cross-gender flat direction

A second structural non-identifiability: **every observed game has an equal number of women on each side** — womens (2v2), mens (0v0), mixed (1v1). Verified across all modeling rows. Consequence: adding any constant *c* to every woman's value changes no predicted margin anywhere, so the *offset* between the men's and women's value blocks is invisible to the data. The zero-mean prior resolves it by (approximately) equating the two pools' averages. Therefore:

- comparisons **within** a gender are data-driven and safe;
- comparisons **across** genders ("is Waters better than Johns?") reflect the equal-pools convention, not evidence;
- hypothetical cross-gender matchups ("Waters/Bright vs Johns/Tardio") shift by 2c along the flat direction — the model has **no** data-driven prediction for them, and no such game exists in the data. Mixed-team predictions (1M+1F vs 1M+1F) are safe: the offset cancels.

What the data **does** identify:

1. **Player value** `v_i` — total points per game a player adds to their team's margin (actor + partner combined), with context-specific deviations (mixed/mens/womens).
2. **Dyad chemistry** `d_ij` — how far a specific pair deviates from the sum of its parts. This is exactly the "relationship effect" of the SRM.
3. **Partner-dependence** (descriptive) — the spread of a player's dyad effects: players whose pairs consistently over/under-perform additivity vs. players who are partner-proof.

## Model

```
margin_g ~ Normal(mu_g, sigma_e)
mu_g = beta_tour + sum(v_i + w_i,ctx | team1) - sum(... | team2)
     + d_dyad1 - d_dyad2 + m_match
v ~ N(0, sd_v)   w ~ N(0, sd_w)   d ~ N(0, sd_d)   m ~ N(0, sd_m)
```
Fit with NUTS (numpyro), 2 chains × 700/700, non-centered, on the 8,875 side-out-to-11 non-forfeit games (to-15 Challenger rounds excluded — different margin scale).

Convergence: 0 divergences; max R̂ over all player values 1.006, over all dyad effects 1.008.

### Variance decomposition (posterior means of scales)

| component | sd (points) | interpretation |
|:--|--:|:--|
| player value sd_v | 2.21 | spread of individual ability |
| context deviation sd_w | 0.13 | how much skill shifts across mixed/mens/womens |
| dyad chemistry sd_d | 0.47 | typical size of pair-specific synergy |
| match intercept sd_m | 1.18 | shared match-day component (Bo3 correlation) |
| residual sd_e | 4.55 | game-to-game noise |

Team-one bias: MLP +0.11 ± 0.16 (≈0, as expected — home/away is arbitrary), PPA +1.13 ± 0.09 (small residual seeding bias after conditioning on player values; compare +2.79 raw).

## Leaderboard — player value (≥40 games)

`value` = points per game added to team margin vs. an average pro in this pool. A pairing's predicted margin ≈ (v+w of your two) − (v+w of theirs) + chemistry terms.

Shown separately by gender — within-gender order is data-driven; the alignment *between* the two lists is the equal-pools prior convention (see above).

### Men (top 25 of 138 with ≥40 games)

| rank | player | value | ±sd | games |
|--:|:--|--:|--:|--:|
| 1 | Gabriel Tardio *(focal)* | +5.55 | 0.52 | 265 |
| 2 | Ben Johns *(focal)* | +5.47 | 0.58 | 263 |
| 3 | Hayden Patriquin *(focal)* | +5.42 | 0.61 | 259 |
| 4 | JW Johnson | +5.39 | 0.55 | 221 |
| 5 | Christian Alshon | +5.27 | 0.53 | 224 |
| 6 | Eric Oncins | +4.98 | 0.49 | 205 |
| 7 | Andrei Daescu | +4.74 | 0.49 | 269 |
| 8 | Federico Staksrud *(focal)* | +4.54 | 0.48 | 240 |
| 9 | Jay Devilliers | +4.39 | 0.55 | 149 |
| 10 | Riley Newman | +4.34 | 0.62 | 92 |
| 11 | Nicolas Acevedo | +4.29 | 0.56 | 110 |
| 12 | Jack Sock | +4.28 | 0.60 | 107 |
| 13 | Bruno Faletto | +4.20 | 0.75 | 63 |
| 14 | Noe Khlif *(focal)* | +4.00 | 0.46 | 190 |
| 15 | Armaan Bhatia | +3.92 | 0.56 | 97 |
| 16 | Juan Benitez | +3.92 | 0.66 | 76 |
| 17 | Connor Garnett | +3.82 | 0.52 | 162 |
| 18 | CJ Klinger | +3.71 | 0.53 | 185 |
| 19 | Oscar Serra | +3.68 | 1.15 | 52 |
| 20 | Tama Shimabukuro | +3.65 | 0.52 | 167 |
| 21 | Will Howells *(focal)* | +3.64 | 0.62 | 87 |
| 22 | Mark Dancuart | +3.56 | 0.83 | 55 |
| 23 | Hunter Johnson | +3.55 | 0.52 | 112 |
| 24 | Yuta Funemizu | +3.50 | 0.61 | 108 |
| 25 | Dekel Bar | +3.47 | 0.51 | 135 |

### Women (top 25 of 114 with ≥40 games)

| rank | player | value | ±sd | games |
|--:|:--|--:|--:|--:|
| 1 | Anna Leigh Waters *(focal)* | +7.74 | 0.54 | 259 |
| 2 | Anna Bright *(focal)* | +6.09 | 0.64 | 261 |
| 3 | Parris Todd | +5.17 | 0.51 | 202 |
| 4 | Jorja Johnson *(focal)* | +5.15 | 0.54 | 224 |
| 5 | Jade Kawamoto *(focal)* | +5.07 | 0.64 | 94 |
| 6 | Rachel Rohrabacher | +5.07 | 0.51 | 208 |
| 7 | Tina Pisnik | +5.02 | 0.46 | 223 |
| 8 | Sofia Sewing | +5.00 | 0.79 | 48 |
| 9 | Jackie Kawamoto | +4.82 | 0.62 | 84 |
| 10 | Tyra Hurricane Black *(focal)* | +4.81 | 0.49 | 214 |
| 11 | Mari Humberg | +4.62 | 0.48 | 161 |
| 12 | Danni-Elle Townsend | +4.51 | 0.66 | 75 |
| 13 | Kate Fahey *(focal)* | +4.40 | 0.52 | 186 |
| 14 | Ewa Radzikowska | +4.28 | 0.80 | 65 |
| 15 | Lucy Kovalova | +3.77 | 0.93 | 43 |
| 16 | Chao Yi Wang | +3.73 | 0.53 | 112 |
| 17 | Lacy Schneemann | +3.72 | 0.47 | 174 |
| 18 | Sahra Dennehy | +3.49 | 0.76 | 51 |
| 19 | Meghan Dizon | +3.45 | 0.53 | 134 |
| 20 | Marisa Ruiz | +3.44 | 0.93 | 51 |
| 21 | Catherine Parenteau | +3.42 | 0.53 | 200 |
| 22 | Estee Widdershoven | +3.37 | 0.63 | 111 |
| 23 | Allyce Jones | +3.36 | 0.57 | 132 |
| 24 | Carlota Trevino | +3.25 | 0.59 | 108 |
| 25 | Isabella Dunlap | +3.23 | 0.47 | 198 |

## Focal players

| player | value | ±sd | games | value rank (≥40g) |
|:--|--:|--:|--:|--:|
| Ben Johns | +5.47 | 0.58 | 263 | 4/252 |
| Anna Leigh Waters | +7.74 | 0.54 | 259 | 1/252 |
| Anna Bright | +6.09 | 0.64 | 261 | 2/252 |
| Hayden Patriquin | +5.42 | 0.61 | 259 | 5/252 |
| Gabriel Tardio | +5.55 | 0.52 | 265 | 3/252 |
| Federico Staksrud | +4.54 | 0.48 | 240 | 19/252 |
| Jade Kawamoto | +5.07 | 0.64 | 94 | 10/252 |
| Jorja Johnson | +5.15 | 0.54 | 224 | 9/252 |
| Will Howells | +3.64 | 0.62 | 87 | 38/252 |
| Noe Khlif | +4.00 | 0.46 | 190 | 28/252 |
| Kate Fahey | +4.40 | 0.52 | 186 | 21/252 |
| Tyra Hurricane Black | +4.81 | 0.49 | 214 | 16/252 |

## Pair chemistry (dyads with ≥15 games)

`chemistry` = points per game beyond the sum of the two players' values. `P(>0)` = posterior probability the synergy is real rather than shrinkage noise.

### Best chemistry

| pair | context | chem | ±sd | P(>0) | pct | games |
|:--|:--|--:|--:|--:|--:|--:|
| Alexa Schull + Darrian Young | mixed | +0.46 | 0.53 | 0.81 | 100 | 26 |
| Andrea Koop + Lauren Stratman | womens | +0.45 | 0.57 | 0.78 | 100 | 16 |
| Lacy Schneemann + Tina Pisnik | womens | +0.44 | 0.49 | 0.82 | 100 | 38 |
| Zoeya Khan + Lina Padegimaite | womens | +0.41 | 0.54 | 0.78 | 100 | 15 |
| Allison Phillips + Samantha Parker | womens | +0.40 | 0.51 | 0.78 | 100 | 18 |
| Ivan Jakovljevic + Judit Castillo | mixed | +0.40 | 0.51 | 0.79 | 100 | 46 |
| Mya Bui + Paula Rives | womens | +0.39 | 0.53 | 0.76 | 100 | 19 |
| Ewa Radzikowska + Tamaryn Emmrich | womens | +0.39 | 0.50 | 0.78 | 100 | 30 |
| Callie Smith + Lea Jansen | womens | +0.38 | 0.52 | 0.78 | 100 | 56 |
| Ashley Griffith + George Rangelov | mixed | +0.35 | 0.52 | 0.75 | 100 | 51 |
| Anna Leigh Waters + Jorja Johnson | womens | +0.34 | 0.49 | 0.75 | 100 | 25 |
| Kate Fahey + Anna Bright | womens | +0.34 | 0.51 | 0.73 | 99 | 25 |

### Worst chemistry

| pair | context | chem | ±sd | P(>0) | pct | games |
|:--|:--|--:|--:|--:|--:|--:|
| Ewa Radzikowska + Martin Emmrich | mixed | -0.18 | 0.48 | 0.37 | 2 | 35 |
| Gregory Dow + Anderson Scarpa | mens | -0.20 | 0.49 | 0.34 | 1 | 28 |
| Eric Oncins + Tina Pisnik | mixed | -0.20 | 0.45 | 0.34 | 1 | 34 |
| Tama Shimabukuro + Connor Garnett | mens | -0.21 | 0.47 | 0.33 | 1 | 20 |
| Christian Alshon + Hayden Patriquin | mens | -0.21 | 0.47 | 0.32 | 1 | 95 |
| Isabella Dunlap + Estee Widdershoven | womens | -0.21 | 0.48 | 0.34 | 1 | 18 |
| Jay Devilliers + Dekel Bar | mens | -0.22 | 0.50 | 0.34 | 0 | 17 |
| Isabella Dunlap + Christopher Haworth | mixed | -0.26 | 0.47 | 0.29 | 0 | 19 |

### Focal dyads

| pair | context | chem | ±sd | P(>0) | pct | games |
|:--|:--|--:|--:|--:|--:|--:|
| Anna Leigh Waters + Jorja Johnson | womens | +0.34 | 0.49 | 0.75 | 100 | 25 |
| Kate Fahey + Anna Bright | womens | +0.34 | 0.51 | 0.73 | 99 | 25 |
| Kate Fahey + Gabriel Tardio | mixed | +0.31 | 0.49 | 0.74 | 99 | 19 |
| Gabriel Tardio + Hayden Patriquin | mens | +0.30 | 0.49 | 0.73 | 99 | 25 |
| Jade Kawamoto + Ben Johns | mixed | +0.23 | 0.48 | 0.69 | 98 | 19 |
| Anna Leigh Waters + Noe Khlif | mixed | +0.22 | 0.48 | 0.66 | 98 | 25 |
| Parris Todd + Tyra Hurricane Black | womens | +0.22 | 0.48 | 0.68 | 98 | 22 |
| Jade Kawamoto + Jackie Kawamoto | womens | +0.20 | 0.46 | 0.65 | 97 | 29 |
| Christian Alshon + Tyra Hurricane Black | mixed | +0.19 | 0.47 | 0.65 | 96 | 21 |
| Federico Staksrud + CJ Klinger | mens | +0.18 | 0.49 | 0.63 | 95 | 13 |
| Kate Fahey + Federico Staksrud | mixed | +0.16 | 0.43 | 0.64 | 95 | 78 |
| Eric Oncins + Tyra Hurricane Black | mixed | +0.15 | 0.46 | 0.62 | 94 | 14 |
| Rafa Hewett + Noe Khlif | mens | +0.12 | 0.46 | 0.62 | 91 | 10 |
| Jorja Johnson + Will Howells | mixed | +0.12 | 0.48 | 0.59 | 91 | 21 |
| Ben Johns + Max Freeman | mens | +0.10 | 0.46 | 0.59 | 88 | 20 |
| Anna Bright + Hayden Patriquin | mixed | +0.10 | 0.46 | 0.57 | 88 | 138 |
| Federico Staksrud + Milan Rane | mixed | +0.07 | 0.46 | 0.57 | 82 | 18 |
| Noe Khlif + Tyra Hurricane Black | mixed | +0.07 | 0.48 | 0.56 | 80 | 17 |
| Andrei Daescu + Federico Staksrud | mens | +0.06 | 0.43 | 0.55 | 78 | 101 |
| Parris Todd + Kate Fahey | womens | +0.03 | 0.43 | 0.51 | 69 | 28 |
| Noe Khlif + Tina Pisnik | mixed | +0.01 | 0.43 | 0.50 | 61 | 46 |
| Anna Leigh Waters + Ben Johns | mixed | -0.00 | 0.44 | 0.48 | 56 | 111 |
| Catherine Parenteau + Gabriel Tardio | mixed | -0.00 | 0.46 | 0.50 | 55 | 79 |
| Jorja Johnson + Tyra Hurricane Black | womens | -0.03 | 0.43 | 0.46 | 43 | 83 |
| Jade Kawamoto + Catherine Parenteau | womens | -0.04 | 0.48 | 0.46 | 35 | 29 |
| Jack Sock + Federico Staksrud | mens | -0.05 | 0.46 | 0.46 | 31 | 15 |
| Andrei Daescu + Gabriel Tardio | mens | -0.05 | 0.46 | 0.47 | 29 | 15 |
| Ben Johns + Gabriel Tardio | mens | -0.07 | 0.44 | 0.43 | 18 | 113 |
| Andrei Daescu + Tyra Hurricane Black | mixed | -0.11 | 0.47 | 0.40 | 8 | 21 |
| Brooke Buckner + Kate Fahey | womens | -0.12 | 0.49 | 0.42 | 6 | 11 |
| Jorja Johnson + JW Johnson | mixed | -0.14 | 0.44 | 0.37 | 4 | 95 |
| Anna Leigh Waters + Anna Bright | womens | -0.15 | 0.46 | 0.36 | 3 | 98 |
| Noe Khlif + Will Howells | mens | -0.18 | 0.48 | 0.35 | 2 | 41 |
| Rachel Rohrabacher + Gabriel Tardio | mixed | -0.18 | 0.50 | 0.35 | 2 | 14 |
| Christian Alshon + Hayden Patriquin | mens | -0.21 | 0.47 | 0.32 | 1 | 95 |
| Kate Fahey + Lacy Schneemann | womens | -0.32 | 0.48 | 0.25 | 0 | 11 |

## Partner-dependence (descriptive)

Std-dev of a player's dyad-chemistry estimates (dyads with ≥8 games, players with ≥3 such dyads). High = chemistry-sensitive; low = partner-proof. Descriptive only — see the identifiability note.

| player | dyads | mean chem | spread (sd) |
|:--|--:|--:|--:|
| Chao Yi Wang | 3 | +0.15 | 0.32 |
| Paula Rives | 3 | +0.10 | 0.28 |
| Roscoe Bellamy | 5 | +0.09 | 0.28 |
| Ivan Jakovljevic | 3 | +0.11 | 0.28 |
| Collin Johns | 3 | +0.03 | 0.27 |
| Andrea Koop | 3 | +0.21 | 0.27 |
| Hayden Patriquin *(focal)* | 3 | +0.06 | 0.26 |
| Kate Fahey *(focal)* | 6 | +0.07 | 0.26 |
| Ashley Griffith | 4 | +0.12 | 0.25 |
| Anna Bright *(focal)* | 3 | +0.10 | 0.25 |
| Camden Chaffin | 4 | +0.06 | 0.23 |
| Lacy Schneemann | 8 | +0.04 | 0.23 |

*Most partner-proof (lowest spread):*

| player | dyads | mean chem | spread (sd) |
|:--|--:|--:|--:|
| Kyle Koszuta | 3 | +0.06 | 0.05 |
| Kayla Williams | 3 | -0.01 | 0.05 |
| Luc Pham | 4 | +0.07 | 0.05 |
| James Delgado | 4 | +0.08 | 0.04 |
| Alex Walker | 4 | +0.04 | 0.04 |
| Angie Walker | 3 | +0.04 | 0.04 |
| Lyn Yuen Choo | 3 | +0.07 | 0.03 |
| Mark Dancuart | 5 | +0.06 | 0.02 |

## Unshrunk ("fixed effects") chemistry check

OLS with a fixed effect per player + one dyad dummy, cluster-robust SEs by match — no shrinkage prior at all (`model/fixed_effects_dyads.py`, all dyads with ≥30 games). If the Bayesian prior were burying real chemistry, it would show up here. It doesn't:

| pair | context | games | unshrunk est. | ±se | t | Bayesian (shrunk) |
|:--|:--|--:|--:|--:|--:|--:|
| Kate Fahey + Gabriel Tardio | mixed | 19 | +1.78 | 1.13 | +1.6 | +0.31 |
| Gabriel Tardio + Hayden Patriquin | mens | 25 | +1.76 | 1.18 | +1.5 | +0.30 |
| Kate Fahey + Anna Bright | womens | 25 | +1.59 | 1.11 | +1.4 | +0.34 |
| Anna Leigh Waters + Jorja Johnson | womens | 25 | +1.49 | 0.85 | +1.8 | +0.34 |
| Jade Kawamoto + Jackie Kawamoto | womens | 29 | +1.31 | 1.24 | +1.1 | +0.20 |
| Christian Alshon + Tyra Hurricane Black | mixed | 21 | +1.23 | 1.20 | +1.0 | +0.19 |
| Jade Kawamoto + Ben Johns | mixed | 19 | +1.20 | 1.35 | +0.9 | +0.23 |
| Anna Leigh Waters + Noe Khlif | mixed | 25 | +1.05 | 1.08 | +1.0 | +0.22 |
| Jorja Johnson + Will Howells | mixed | 21 | +0.84 | 1.27 | +0.7 | +0.12 |
| Parris Todd + Tyra Hurricane Black | womens | 22 | +0.72 | 0.81 | +0.9 | +0.22 |
| Kate Fahey + Federico Staksrud | mixed | 78 | +0.59 | 0.84 | +0.7 | +0.16 |
| Anna Bright + Hayden Patriquin | mixed | 138 | +0.54 | 0.92 | +0.6 | +0.10 |
| Ben Johns + Max Freeman | mens | 20 | +0.24 | 1.04 | +0.2 | +0.10 |
| Federico Staksrud + Milan Rane | mixed | 18 | +0.13 | 0.94 | +0.1 | +0.07 |
| Parris Todd + Kate Fahey | womens | 28 | +0.05 | 1.03 | +0.1 | +0.03 |
| Noe Khlif + Tyra Hurricane Black | mixed | 17 | +0.02 | 1.02 | +0.0 | +0.07 |
| Andrei Daescu + Federico Staksrud | mens | 101 | -0.02 | 0.77 | -0.0 | +0.06 |
| Andrei Daescu + Gabriel Tardio | mens | 15 | -0.04 | 1.12 | -0.0 | -0.05 |
| Jorja Johnson + Tyra Hurricane Black | womens | 83 | -0.04 | 0.88 | -0.1 | -0.03 |
| Noe Khlif + Tina Pisnik | mixed | 46 | -0.08 | 0.93 | -0.1 | +0.01 |
| Anna Leigh Waters + Ben Johns | mixed | 111 | -0.20 | 0.88 | -0.2 | -0.00 |
| Ben Johns + Gabriel Tardio | mens | 113 | -0.51 | 0.87 | -0.6 | -0.07 |
| Catherine Parenteau + Gabriel Tardio | mixed | 79 | -0.52 | 0.90 | -0.6 | -0.00 |
| Jade Kawamoto + Catherine Parenteau | womens | 29 | -0.70 | 1.25 | -0.6 | -0.04 |
| Andrei Daescu + Tyra Hurricane Black | mixed | 21 | -0.81 | 1.26 | -0.6 | -0.11 |
| Jack Sock + Federico Staksrud | mens | 15 | -0.81 | 1.31 | -0.6 | -0.05 |
| Jorja Johnson + JW Johnson | mixed | 95 | -1.27 | 0.93 | -1.4 | -0.14 |
| Noe Khlif + Will Howells | mens | 41 | -1.58 | 1.13 | -1.4 | -0.18 |
| Anna Leigh Waters + Anna Bright | womens | 98 | -1.72 | 0.92 | -1.9 | -0.15 |
| Christian Alshon + Hayden Patriquin | mens | 95 | -1.72 | 0.90 | -1.9 | -0.21 |

Across all 200 high-volume dyads, the t-statistics have mean +0.44 and sd 1.32 (pure noise would give ≈0 and ≈1). The mild overdispersion is the small league-wide chemistry variance; the positive mean hints at survivorship (pairs that keep playing together are pairs it's working for). No individual pair separates from the pack.

Note for Bright + Patriquin specifically: Patriquin is Bright's only mixed partner, so her personal mixed-context shift and the pair dummy are the same regression column — the unshrunk estimate is their *sum*, i.e. if anything an overstatement of pure pair chemistry.

### Why a player's dyad estimates see-saw (read before over-interpreting)

A player's value is fitted to their *average* performance across partnerships, so their dyad effects are deviations around that average: one strongly negative pairing mechanically implies their other pairings lean positive. The apparent "outliers" (Waters+Bright −1.7, Alshon+Patriquin −1.7) and the mirrored positives (Bright+Fahey +1.6, Patriquin+Tardio +1.8) are therefore **not independent facts** — each within-player set is one identified contrast: *Bright's games with Waters run ~2–3 points worse than her games with her other partners* (≈2.3σ), and likewise Patriquin with Alshon vs Tardio (≈2.3σ). Which pairing deserves the label "bad" vs "good" is not identified — only the difference is. Candidate real mechanisms (court-side preference conflicts, role redundancy) and plain multiple-comparisons luck are both live; the season's remaining games are the honest test.

A further confound: for both stars the negative-chemistry partner is PPA-only (Waters, Alshon) and the positive-chemistry partner is MLP-only (Fahey, Tardio), so the partner contrast is collinear with a player-by-tour contrast the model doesn't include. The one testable implication — Bright+Patriquin's own games split by tour — shows no MLP advantage (MLP +0.04 ± 0.94 vs PPA +0.36 ± 0.77, unshrunk), which disfavors but cannot rule out the tour story (only 23 MLP games together).

## Does it predict? Temporal holdout

Model refit on games before 2026-06-01 only, then used to predict every later game whose four players all had ≥10 training games (n = 686 — mostly MLP, predicted from PPA-heavy training):

| metric | model | coin flip |
|:--|--:|--:|
| winner accuracy | 74.9% | 50% |
| Brier score | 0.177 | 0.250 |
| log loss | 0.533 | 0.693 |
| margin MAE | 4.38 | 5.69 (predict 0) |

Calibration is *under*confident (e.g. games called ~65% go the favorite's way ~76% of the time) — the conservative direction: player values generalize at least as well as their posteriors claim.

## Benchmark: platform rating vs this model

pickleball.com embeds its own per-player rating (its in-house system — not DUPR, which requires an authenticated API) as an as-of-match snapshot in the raw payloads. Head-to-head on the same holdout games (≥ 2026-06-01, n = 518), predicting each game's winner:

| predictor | accuracy | Brier |
|:--|--:|--:|
| this model (frozen 2026-06-01) | 73.7% | 0.179 |
| platform rating (as-of-match, updates all season) | 64.7% | 0.229 |

The two systems agree on 69% of games; correlation between model value and latest rating is 0.65 (men, n=137) / 0.53 (women, n=112). The model wins despite the rating having an information edge (it updates through the test window; the model is frozen at the split). Notable rating oddities the model avoids: Gabriel Tardio (#1 here) ranked ~#30 by rating; Jackie Kawamoto's rating collapsed mid-season from 6.13 to 3.50 — an apparent reset/identity glitch in the rating engine.

## Robustness: core-pool refit ("real pros only")

Same model refit on games where **all four players have ≥30 appearances** (4,698 games, 312 players — drops Challenger one-weekenders and qualifier cannon fodder). Result:

- **Rankings are stable**: Spearman ρ = 0.966 between the two fits over the 167 shared regulars. The leaderboard's composition is unchanged.
- **The zero point moves up ~1.8 points** (values are relative to the pool average, and the pool got stronger). E.g. Waters +7.74 → +6.41. Differences *between* players are what carry meaning; the absolute level is pool-dependent.
- **Chemistry shrinks further** (sd_d 0.47 → 0.33): part of the apparent synergy in the full pool was pairs feasting together on weak fields.
- One notable mover: players who log many Challenger games (e.g. Patriquin) give back a fraction of a point relative to peers who don't — mild "Challenger farming" inflation in the full-pool fit.

## Caveats

- **Cross-gender comparisons are convention** (see the flat-direction section): read the leaderboard as two interleaved within-gender rankings aligned by prior.
- Single 2026 season, mid-season snapshot (through Jul 11): no time-varying skill; Patriquin-type trajectories are averaged over the window.
- Margins treated as Gaussian; to-11 games truncate blowouts (±11-ish cap), so elite values are mildly compressed relative to "true" dominance.
- Anna Bright's mixed games are 100% with Patriquin: her mixed context deviation and that dyad's chemistry are separated only by the pooled hierarchical structure.
- Selection: PPA game 3 exists only after 1–1 splits; handled by the match intercept, not modeled as an explicit selection process.
- Qualifier and Challenger main-draw games are included; they mostly inform the tail of the player pool and tighten opponent-quality adjustment for focal players.