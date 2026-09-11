# Draft simulation -- 20 teams, scarcity, varied draft and information

Prices: alpha = 1.0, one joint $20M pool, **Anna Leigh Waters franchise-tagged at $769,149** (`phase2_pricing.prices_tagged`).

Board: the priced 60+60 plus every 2026 MLP participant outside the priced pool: 60M/60F priced + 35M/28F real fill-ins at the $30k floor (`--board mlp2026`).

20 teams, $1M cap, 6 rounds (3M+3W), one pick per turn. Owners project a final roster for each affordable candidate (candidate + greedy fill of the best-believed players still available) and take the one whose projection has the highest believed tie probability against a reference roster of doubles ranks (10, 30, 50) per gender. Noise = sd of each owner's belief error as a fraction of the gender's pool spread (men 0.143, women 0.232 logit), fixed per owner per draft. Seasons: double round robin (38 ties) + top-4 playoff, scored with the TRUE values. Parity = every team 50% expected wins, 5% title. Built by `draft_sim.py`.

## Summary

| draft | owner noise | drafts | parity spread (sd of team win%) | strongest team win% | mean spend | blueprint mix |
|---|---|---|---|---|---|---|
| snake | 0% | 1 | 4.3 pts | 66.1% | $976k | star-led 65%, anchor 25%, superstar 10% |
| snake | 10% | 30 | 4.4 pts | 65.7% | $975k | star-led 59%, anchor 31%, superstar 10% |
| snake | 25% | 30 | 4.9 pts | 65.4% | $969k | star-led 60%, anchor 30%, superstar 10% |
| linear | 0% | 1 | 5.2 pts | 66.4% | $973k | star-led 60%, anchor 30%, superstar 10% |
| linear | 10% | 30 | 5.5 pts | 65.9% | $969k | star-led 58%, anchor 32%, superstar 10% |
| linear | 25% | 30 | 5.4 pts | 65.8% | $966k | star-led 61%, anchor 29%, superstar 10%, balanced 0% |

## snake draft, owner noise 0% (1 draft(s) x 200 seasons, 1 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus every 2026 MLP participant outside the priced pool: 60M/60F priced + 35M/28F real fill-ins at the $30k floor; teams took 16.0 floor players per draft, leaving 16 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: none. All: Shelby Bates #47F, Zoey Weil #50F, Allie Reichert #51F, Connor Mogle #42M, Max Freeman #43M, Eugenia Carolina Lopez Ascarate #55F, Donald Young #44M, Harsh Mehta #45M, Alex Emery #46M, Genie Erokhina #56F, Etienne Blaszkewycz #48M, Juan Benitez #49M, Tyson McGuffin #50M, Martin Emmrich #53M, Patrick Kawka #56M, Cailyn Campbell #60F.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 66.1% | 37.5% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 2.0 | 51.5% | 5.0% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 11.0 | 48.5% | 2.0% | YES |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 8.0 | 50.6% | 2.0% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 7.0 | 49.0% | 4.0% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 10.0 | 48.7% | 6.0% | YES |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 3.0 | 52.4% | 6.5% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 4.0 | 49.5% | 4.0% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 5.0 | 52.1% | 6.0% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 6.0 | 51.2% | 3.0% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 9.0 | 50.6% | 3.5% | no |
| Andrei Daescu (#6M) | $323k | 1.0 | 0% | 18.0 | 49.1% | 4.0% | no |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 66 | 51 | 52 | 50 | 52 | 51 | 49 | 51 | 51 | 49 | 48 | 48 | 46 | 46 | 45 | 46 | 49 | 49 | 48 | 52 |
| title% | 38 | 5 | 6 | 4 | 6 | 3 | 4 | 2 | 4 | 6 | 2 | 2 | 1 | 0 | 1 | 1 | 4 | 4 | 2 | 4 |

## snake draft, owner noise 10% (30 draft(s) x 200 seasons, 38 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus every 2026 MLP participant outside the priced pool: 60M/60F priced + 35M/28F real fill-ins at the $30k floor; teams took 15.0 floor players per draft, leaving 40 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: none. All: Martin Emmrich #53M, Genie Erokhina #56F, Patrick Kawka #56M, Zoey Weil #50F, Donald Young #44M, Juan Benitez #49M, Alex Emery #46M, Connor Mogle #42M, Max Freeman #43M, Shelby Bates #47F, Cailyn Campbell #60F, Tyson McGuffin #50M, Eugenia Carolina Lopez Ascarate #55F, Harsh Mehta #45M, Allie Reichert #51F, Daria Walczak #44F, Callie Smith #59F, Milan Rane #40F, Etienne Blaszkewycz #48M, Lucy Kovalova #46F, Hien Truong #37M, Pablo Tellez #38M, Tristan Dussault #55M, Luca Mack #40M, Liz Truluck #48F, Estee Widdershoven #42F, Anderson Scarpa #39M, Joseph Wild #41M, Thomas Yu #34M, Samantha Parker #35F, George Wall #36M, Maggie Brascia #41F, Rafa Hewett #52M, Carlota Trevino #34F, Jessie Irvine #37F, Isabella Dunlap #38F, Mohaned Alhouni #58M, Ting Chieh Wei #33F, Oscar Serra #33M, Hoang Nam Ly #59M.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 65.7% | 36.4% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 2.8 | 51.2% | 4.3% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 6.6 | 50.9% | 4.6% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 6.7 | 51.1% | 4.6% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 9.0 | 50.3% | 4.2% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 10.8 | 50.1% | 4.0% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 2.5 | 50.8% | 4.2% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 4.5 | 50.0% | 3.6% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 6.2 | 49.4% | 3.5% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 8.5 | 49.0% | 3.2% | YES |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 8.1 | 48.8% | 3.2% | YES |
| Andrei Daescu (#6M) | $323k | 1.1 | 0% | 17.0 | 47.8% | 2.7% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 66 | 51 | 51 | 50 | 51 | 50 | 50 | 49 | 49 | 50 | 50 | 50 | 49 | 47 | 47 | 47 | 47 | 48 | 49 | 49 |
| title% | 36 | 4 | 4 | 3 | 4 | 4 | 4 | 4 | 3 | 4 | 4 | 3 | 3 | 2 | 2 | 2 | 2 | 2 | 3 | 3 |

## snake draft, owner noise 25% (30 draft(s) x 200 seasons, 37 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus every 2026 MLP participant outside the priced pool: 60M/60F priced + 35M/28F real fill-ins at the $30k floor; teams took 15.7 floor players per draft, leaving 56 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Dekel Bar (#27M, $111k, 20% of drafts), Yuta Funemizu (#26M, $119k, 13% of drafts), Yana Newell (#28F, $144k, 10% of drafts), Layne Sleeth (#27F, $147k, 7% of drafts), CJ Klinger (#28M, $111k, 7% of drafts), Ewa Radzikowska (#22F, $176k, 3% of drafts), Leigh Waters (#24F, $157k, 3% of drafts), Allyce Jones (#25F, $154k, 3% of drafts), Jillian Braverman (#26F, $149k, 3% of drafts), Vanshik Kapadia (#24M, $132k, 3% of drafts), Alix Truong (#30F, $132k, 3% of drafts), Roscoe Bellamy (#29M, $107k, 3% of drafts), Augustus Ge (#30M, $102k, 3% of drafts). All: Martin Emmrich #53M, Patrick Kawka #56M, Zoey Weil #50F, Alex Emery #46M, Juan Benitez #49M, Genie Erokhina #56F, Cailyn Campbell #60F, Harsh Mehta #45M, Estee Widdershoven #42F, Shelby Bates #47F, Max Freeman #43M, Tyson McGuffin #50M, Thomas Yu #34M, Lucy Kovalova #46F, Hien Truong #37M, Allie Reichert #51F, Eugenia Carolina Lopez Ascarate #55F, Callie Smith #59F, Anderson Scarpa #39M, Isabella Dunlap #38F, Pablo Tellez #38M, Donald Young #44M, Tristan Dussault #55M, Connor Mogle #42M, Etienne Blaszkewycz #48M, George Wall #36M, Daria Walczak #44F, Joseph Wild #41M, Samantha Parker #35F, Oscar Serra #33M, Liz Truluck #48F, Dekel Bar #27M, Jessie Irvine #37F, Carlota Trevino #34F, Milan Rane #40F, Maggie Brascia #41F, Yuta Funemizu #26M, Yana Newell #28F, Ting Chieh Wei #33F, Yufei Long #45F, Luca Mack #40M, Tama Shimabukuro #47M, Rafa Hewett #52M, Layne Sleeth #27F, CJ Klinger #28M, Christine Maddox #58F, Mohaned Alhouni #58M, Ewa Radzikowska #22F, Leigh Waters #24F, Allyce Jones #25F, Jillian Braverman #26F, Vanshik Kapadia #24M, Alix Truong #30F, Roscoe Bellamy #29M, Augustus Ge #30M, Marianna Petrei #43F.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 65.4% | 34.4% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 4.4 | 51.1% | 4.2% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 5.8 | 50.8% | 4.1% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 7.8 | 50.7% | 4.7% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 9.2 | 50.1% | 3.6% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 9.9 | 50.1% | 4.0% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 3.1 | 50.3% | 3.8% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 5.6 | 49.0% | 3.6% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 6.5 | 48.9% | 2.8% | YES |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 8.8 | 50.3% | 4.0% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 8.2 | 49.0% | 3.7% | YES |
| Andrei Daescu (#6M) | $323k | 1.1 | 0% | 15.7 | 47.4% | 2.5% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 65 | 50 | 50 | 51 | 50 | 51 | 49 | 49 | 50 | 50 | 49 | 49 | 49 | 48 | 49 | 48 | 48 | 48 | 47 | 49 |
| title% | 34 | 4 | 4 | 4 | 4 | 4 | 3 | 3 | 4 | 4 | 3 | 4 | 3 | 4 | 3 | 3 | 3 | 3 | 3 | 3 |

## linear draft, owner noise 0% (1 draft(s) x 200 seasons, 1 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus every 2026 MLP participant outside the priced pool: 60M/60F priced + 35M/28F real fill-ins at the $30k floor; teams took 16.0 floor players per draft, leaving 16 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: none. All: Carlota Trevino #34F, Estee Widdershoven #42F, Daria Walczak #44F, Joseph Wild #41M, Liz Truluck #48F, Connor Mogle #42M, Max Freeman #43M, Donald Young #44M, Harsh Mehta #45M, Alex Emery #46M, Etienne Blaszkewycz #48M, Juan Benitez #49M, Tyson McGuffin #50M, Tristan Dussault #55M, Patrick Kawka #56M, Cailyn Campbell #60F.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 66.4% | 37.0% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 8.0 | 53.4% | 6.5% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 10.0 | 50.5% | 2.0% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 4.0 | 51.3% | 4.0% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 9.0 | 48.7% | 3.5% | YES |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 11.0 | 49.7% | 3.5% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 2.0 | 51.4% | 5.0% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 3.0 | 51.3% | 5.5% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 7.0 | 49.6% | 1.0% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 5.0 | 51.4% | 5.5% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 6.0 | 52.6% | 3.5% | no |
| Andrei Daescu (#6M) | $323k | 1.0 | 0% | 17.0 | 46.3% | 1.0% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 66 | 51 | 51 | 51 | 51 | 53 | 50 | 53 | 49 | 50 | 50 | 54 | 50 | 50 | 50 | 46 | 46 | 45 | 42 | 40 |
| title% | 37 | 5 | 6 | 4 | 6 | 4 | 1 | 6 | 4 | 2 | 4 | 8 | 2 | 5 | 5 | 0 | 1 | 1 | 0 | 0 |

## linear draft, owner noise 10% (30 draft(s) x 200 seasons, 38 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus every 2026 MLP participant outside the priced pool: 60M/60F priced + 35M/28F real fill-ins at the $30k floor; teams took 17.2 floor players per draft, leaving 46 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Alix Truong (#30F, $132k, 3% of drafts). All: Martin Emmrich #53M, Patrick Kawka #56M, Connor Mogle #42M, Juan Benitez #49M, Liz Truluck #48F, Shelby Bates #47F, Genie Erokhina #56F, Tyson McGuffin #50M, Max Freeman #43M, Anderson Scarpa #39M, Rafa Hewett #52M, Callie Smith #59F, Daria Walczak #44F, Pablo Tellez #38M, Joseph Wild #41M, Zoey Weil #50F, Milan Rane #40F, Yufei Long #45F, Alex Emery #46M, Cailyn Campbell #60F, Carlota Trevino #34F, Donald Young #44M, Tristan Dussault #55M, Isabella Dunlap #38F, Hien Truong #37M, Estee Widdershoven #42F, Lucy Kovalova #46F, Eugenia Carolina Lopez Ascarate #55F, Samantha Parker #35F, Jessie Irvine #37F, Etienne Blaszkewycz #48M, Ting Chieh Wei #33F, Thomas Yu #34M, Maggie Brascia #41F, Allie Reichert #51F, Christine Maddox #58F, George Wall #36M, Luca Mack #40M, Mohaned Alhouni #58M, Roos Van Reek #36F, Harsh Mehta #45M, Alix Truong #30F, Oscar Serra #33M, Marianna Petrei #43F, Tama Shimabukuro #47M, Hoang Nam Ly #59M.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 65.9% | 34.8% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 6.6 | 52.1% | 5.3% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 9.1 | 51.5% | 4.7% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 8.9 | 52.0% | 5.1% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 9.5 | 51.7% | 4.9% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 8.7 | 51.6% | 4.5% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 3.2 | 51.5% | 4.6% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 5.0 | 50.9% | 3.6% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 5.6 | 51.9% | 4.9% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 4.8 | 51.3% | 3.9% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 6.2 | 51.4% | 4.6% | no |
| Andrei Daescu (#6M) | $323k | 1.0 | 0% | 16.9 | 45.1% | 1.2% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 66 | 52 | 52 | 52 | 51 | 52 | 51 | 51 | 52 | 52 | 51 | 52 | 51 | 49 | 49 | 47 | 45 | 44 | 42 | 40 |
| title% | 35 | 5 | 5 | 4 | 4 | 5 | 5 | 5 | 4 | 5 | 4 | 4 | 4 | 3 | 3 | 2 | 1 | 1 | 0 | 0 |

## linear draft, owner noise 25% (30 draft(s) x 200 seasons, 37 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus every 2026 MLP participant outside the priced pool: 60M/60F priced + 35M/28F real fill-ins at the $30k floor; teams took 16.5 floor players per draft, leaving 57 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Augustus Ge (#30M, $102k, 17% of drafts), Yana Newell (#28F, $144k, 13% of drafts), CJ Klinger (#28M, $111k, 13% of drafts), Layne Sleeth (#27F, $147k, 10% of drafts), Vanshik Kapadia (#24M, $132k, 10% of drafts), Dekel Bar (#27M, $111k, 10% of drafts), Jillian Braverman (#26F, $149k, 7% of drafts), Alix Truong (#30F, $132k, 7% of drafts), Yuta Funemizu (#26M, $119k, 7% of drafts), Leigh Waters (#24F, $157k, 3% of drafts), Dylan Frazier (#23M, $133k, 3% of drafts). All: Juan Benitez #49M, Hien Truong #37M, Shelby Bates #47F, Martin Emmrich #53M, Patrick Kawka #56M, Genie Erokhina #56F, Lucy Kovalova #46F, Max Freeman #43M, Tyson McGuffin #50M, Zoey Weil #50F, Allie Reichert #51F, Connor Mogle #42M, Alex Emery #46M, Cailyn Campbell #60F, Joseph Wild #41M, Callie Smith #59F, Thomas Yu #34M, Anderson Scarpa #39M, Liz Truluck #48F, Donald Young #44M, Carlota Trevino #34F, Jessie Irvine #37F, George Wall #36M, Daria Walczak #44F, Harsh Mehta #45M, Estee Widdershoven #42F, Oscar Serra #33M, Pablo Tellez #38M, Eugenia Carolina Lopez Ascarate #55F, Rafa Hewett #52M, Tristan Dussault #55M, Isabella Dunlap #38F, Maggie Brascia #41F, Yufei Long #45F, Etienne Blaszkewycz #48M, Ting Chieh Wei #33F, Augustus Ge #30M, Samantha Parker #35F, Luca Mack #40M, Yana Newell #28F, CJ Klinger #28M, Milan Rane #40F, Layne Sleeth #27F, Vanshik Kapadia #24M, Dekel Bar #27M, Christine Maddox #58F, Jillian Braverman #26F, Alix Truong #30F, Yuta Funemizu #26M, Tama Shimabukuro #47M, Grayson Goldin #60M, Leigh Waters #24F, Dylan Frazier #23M, Roos Van Reek #36F, Marianna Petrei #43F, Mohaned Alhouni #58M, Hoang Nam Ly #59M.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.1 | 65.8% | 35.2% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 7.8 | 51.7% | 4.8% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 8.9 | 51.2% | 4.2% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 8.3 | 50.8% | 3.7% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 8.9 | 51.1% | 4.4% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 10.4 | 51.1% | 4.6% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 3.5 | 50.8% | 4.6% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 4.9 | 51.2% | 4.6% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 5.6 | 51.0% | 4.3% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 5.9 | 50.9% | 4.1% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 6.1 | 51.2% | 4.4% | no |
| Andrei Daescu (#6M) | $323k | 1.1 | 0% | 15.6 | 46.2% | 2.2% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 64 | 53 | 51 | 51 | 51 | 51 | 51 | 52 | 50 | 51 | 50 | 51 | 49 | 50 | 49 | 47 | 47 | 45 | 43 | 42 |
| title% | 31 | 8 | 5 | 5 | 4 | 4 | 4 | 5 | 3 | 5 | 4 | 4 | 3 | 3 | 3 | 2 | 2 | 1 | 1 | 1 |
