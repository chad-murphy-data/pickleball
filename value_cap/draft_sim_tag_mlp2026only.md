# Draft simulation -- 20 teams, scarcity, varied draft and information

Prices: alpha = 1.0, one joint $20M pool, **Anna Leigh Waters franchise-tagged at $769,149** (`phase2_pricing.prices_tagged`).

Board: ONLY 2026 MLP participants: 38M/39F priced + 35M/28F real fill-ins at the $30k floor (`--board mlp2026only`).

20 teams, $1M cap, 6 rounds (3M+3W), one pick per turn. Owners project a final roster for each affordable candidate (candidate + greedy fill of the best-believed players still available) and take the one whose projection has the highest believed tie probability against a reference roster of doubles ranks (10, 30, 50) per gender. Noise = sd of each owner's belief error as a fraction of the gender's pool spread (men 0.143, women 0.232 logit), fixed per owner per draft. Seasons: double round robin (38 ties) + top-4 playoff, scored with the TRUE values. Parity = every team 50% expected wins, 5% title. Built by `draft_sim.py`.

## Summary

| draft | owner noise | drafts | parity spread (sd of team win%) | strongest team win% | mean spend | blueprint mix |
|---|---|---|---|---|---|---|
| snake | 0% | 1 | 4.6 pts | 68.2% | $838k | star-led 60%, superstar 30%, anchor 10% |
| snake | 10% | 30 | 7.0 pts | 72.3% | $838k | star-led 63%, superstar 26%, anchor 11% |
| snake | 25% | 30 | 7.9 pts | 72.5% | $838k | star-led 64%, superstar 26%, anchor 10% |
| linear | 0% | 1 | 12.3 pts | 74.9% | $838k | star-led 90%, superstar 10% |
| linear | 10% | 30 | 11.8 pts | 73.8% | $838k | star-led 88%, superstar 12% |
| linear | 25% | 30 | 11.5 pts | 73.7% | $838k | star-led 84%, superstar 16%, anchor 0% |

## snake draft, owner noise 0% (1 draft(s) x 200 seasons, 1 s)

**Undrafted priced players (info, not a test):** the board is ONLY 2026 MLP participants: 38M/39F priced + 35M/28F real fill-ins at the $30k floor; teams took 43.0 floor players per draft, leaving 0 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: none. All: none.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 68.2% | 42.5% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 2.0 | 49.1% | 3.5% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 3.0 | 52.9% | 6.0% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 4.0 | 52.3% | 7.0% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 5.0 | 49.8% | 3.0% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 6.0 | 44.8% | 1.0% | YES |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 7.0 | 47.3% | 1.5% | YES |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 8.0 | 49.3% | 3.0% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 9.0 | 46.6% | 1.5% | YES |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 10.0 | 50.7% | 4.0% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 12.0 | 49.6% | 4.5% | no |
| Andrei Daescu (#6M) | $323k | 1.0 | 0% | 19.0 | 46.8% | 1.5% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 68 | 49 | 53 | 52 | 50 | 45 | 47 | 49 | 47 | 51 | 46 | 50 | 48 | 50 | 49 | 50 | 50 | 48 | 47 | 52 |
| title% | 42 | 4 | 6 | 7 | 3 | 1 | 2 | 3 | 2 | 4 | 2 | 4 | 1 | 2 | 2 | 2 | 4 | 4 | 2 | 6 |

## snake draft, owner noise 10% (30 draft(s) x 200 seasons, 26 s)

**Undrafted priced players (info, not a test):** the board is ONLY 2026 MLP participants: 38M/39F priced + 35M/28F real fill-ins at the $30k floor; teams took 43.0 floor players per draft, leaving 0 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: none. All: none.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 72.3% | 46.5% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 2.0 | 57.5% | 11.6% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 4.1 | 50.9% | 3.4% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 4.7 | 49.5% | 2.8% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 7.7 | 48.7% | 2.5% | YES |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 8.7 | 49.1% | 2.6% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 4.4 | 49.2% | 2.7% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 6.9 | 44.9% | 1.4% | YES |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 8.0 | 43.8% | 1.1% | YES |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 9.5 | 46.6% | 1.6% | YES |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 10.3 | 46.4% | 1.6% | YES |
| Andrei Daescu (#6M) | $323k | 1.0 | 0% | 17.0 | 47.9% | 2.0% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 72 | 58 | 50 | 49 | 47 | 49 | 45 | 46 | 46 | 48 | 48 | 49 | 48 | 49 | 50 | 49 | 48 | 48 | 51 | 51 |
| title% | 46 | 12 | 3 | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 3 | 3 |

## snake draft, owner noise 25% (30 draft(s) x 200 seasons, 26 s)

**Undrafted priced players (info, not a test):** the board is ONLY 2026 MLP participants: 38M/39F priced + 35M/28F real fill-ins at the $30k floor; teams took 43.0 floor players per draft, leaving 0 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: none. All: none.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 72.5% | 44.6% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 2.4 | 58.4% | 12.2% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 7.1 | 51.4% | 4.1% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 6.8 | 50.7% | 3.6% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 8.8 | 51.3% | 4.2% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 10.4 | 50.0% | 2.8% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 3.9 | 48.5% | 2.2% | YES |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 5.9 | 46.0% | 1.5% | YES |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 7.2 | 46.8% | 2.4% | YES |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 8.2 | 44.2% | 1.0% | YES |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 9.1 | 45.6% | 1.7% | YES |
| Andrei Daescu (#6M) | $323k | 1.0 | 0% | 15.6 | 45.7% | 1.4% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 73 | 54 | 50 | 49 | 48 | 48 | 48 | 44 | 48 | 49 | 48 | 49 | 49 | 47 | 49 | 51 | 48 | 48 | 50 | 50 |
| title% | 45 | 8 | 5 | 3 | 2 | 2 | 2 | 1 | 2 | 3 | 4 | 3 | 2 | 1 | 2 | 4 | 2 | 2 | 3 | 3 |

## linear draft, owner noise 0% (1 draft(s) x 200 seasons, 1 s)

**Undrafted priced players (info, not a test):** the board is ONLY 2026 MLP participants: 38M/39F priced + 35M/28F real fill-ins at the $30k floor; teams took 43.0 floor players per draft, leaving 0 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: none. All: none.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 74.9% | 42.5% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 3.0 | 63.5% | 11.5% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 4.0 | 58.7% | 6.0% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 2.0 | 63.5% | 13.5% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 7.0 | 60.9% | 6.5% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 10.0 | 53.9% | 3.0% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 5.0 | 62.1% | 8.5% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 6.0 | 56.6% | 3.5% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 8.0 | 54.4% | 0.5% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 9.0 | 52.4% | 1.5% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 11.0 | 51.3% | 2.0% | no |
| Andrei Daescu (#6M) | $323k | 1.0 | 0% | 17.0 | 40.0% | 0.0% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 75 | 64 | 63 | 59 | 62 | 57 | 61 | 54 | 52 | 54 | 51 | 50 | 49 | 44 | 32 | 30 | 40 | 36 | 34 | 34 |
| title% | 42 | 14 | 12 | 6 | 8 | 4 | 6 | 0 | 2 | 3 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## linear draft, owner noise 10% (30 draft(s) x 200 seasons, 25 s)

**Undrafted priced players (info, not a test):** the board is ONLY 2026 MLP participants: 38M/39F priced + 35M/28F real fill-ins at the $30k floor; teams took 43.0 floor players per draft, leaving 0 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: none. All: none.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 73.8% | 39.7% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 3.0 | 61.5% | 10.1% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 3.5 | 61.5% | 10.2% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 4.2 | 61.2% | 9.2% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 8.1 | 57.0% | 5.3% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 8.6 | 55.4% | 3.8% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 4.1 | 60.6% | 7.9% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 6.8 | 57.8% | 5.6% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 8.0 | 54.9% | 3.6% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 9.8 | 50.7% | 1.0% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 9.9 | 50.7% | 1.7% | no |
| Andrei Daescu (#6M) | $323k | 1.0 | 0% | 16.1 | 37.4% | 0.4% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 74 | 62 | 61 | 61 | 61 | 60 | 57 | 55 | 53 | 53 | 50 | 47 | 45 | 43 | 41 | 38 | 38 | 35 | 34 | 32 |
| title% | 40 | 11 | 10 | 9 | 8 | 8 | 5 | 3 | 2 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## linear draft, owner noise 25% (30 draft(s) x 200 seasons, 26 s)

**Undrafted priced players (info, not a test):** the board is ONLY 2026 MLP participants: 38M/39F priced + 35M/28F real fill-ins at the $30k floor; teams took 43.0 floor players per draft, leaving 0 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: none. All: none.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 73.7% | 39.8% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 3.9 | 59.7% | 8.7% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 6.7 | 59.4% | 8.1% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 6.6 | 58.8% | 7.7% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 7.9 | 56.6% | 5.0% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 10.3 | 52.3% | 2.6% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 3.1 | 59.2% | 8.1% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 5.8 | 57.1% | 5.6% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 6.9 | 56.3% | 4.5% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 8.7 | 51.3% | 2.3% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 8.7 | 52.5% | 3.7% | no |
| Andrei Daescu (#6M) | $323k | 1.3 | 0% | 12.3 | 43.6% | 2.2% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 73 | 61 | 60 | 59 | 59 | 58 | 57 | 55 | 54 | 54 | 49 | 47 | 45 | 44 | 43 | 40 | 37 | 36 | 36 | 33 |
| title% | 39 | 10 | 9 | 8 | 8 | 7 | 6 | 4 | 3 | 3 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
