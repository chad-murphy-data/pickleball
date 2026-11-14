# Draft simulation -- 20 teams, scarcity, varied draft and information

Prices: alpha = 1.0, one joint $20M pool, **Anna Leigh Waters franchise-tagged at $769,149** (`phase2_pricing.prices_tagged`).

Board: the priced 60+60 plus 60 free agents per gender (next by phi / doubles value) at the $30k floor (`--board best60`).

20 teams, $1M cap, 6 rounds (3M+3W), one pick per turn. Owners project a final roster for each affordable candidate (candidate + greedy fill of the best-believed players still available) and take the one whose projection has the highest believed tie probability against a reference roster of doubles ranks (10, 30, 50) per gender. Noise = sd of each owner's belief error as a fraction of the gender's pool spread (men 0.143, women 0.232 logit), fixed per owner per draft. Seasons: double round robin (38 ties) + top-4 playoff, scored with the TRUE values. Parity = every team 50% expected wins, 5% title. Built by `draft_sim.py`.

## Summary

| draft | owner noise | drafts | parity spread (sd of team win%) | strongest team win% | mean spend | blueprint mix |
|---|---|---|---|---|---|---|
| snake | 0% | 1 | 4.2 pts | 65.3% | $971k | star-led 65%, anchor 25%, superstar 10% |
| snake | 10% | 30 | 4.6 pts | 65.7% | $970k | star-led 60%, anchor 30%, superstar 10% |
| snake | 25% | 30 | 5.0 pts | 65.0% | $962k | star-led 60%, anchor 30%, superstar 10%, balanced 0% |
| linear | 0% | 1 | 5.8 pts | 67.8% | $967k | star-led 55%, anchor 35%, superstar 10% |
| linear | 10% | 30 | 5.9 pts | 66.8% | $965k | star-led 60%, anchor 30%, superstar 10% |
| linear | 25% | 30 | 5.7 pts | 65.3% | $959k | star-led 61%, anchor 28%, superstar 10%, balanced 1% |

## snake draft, owner noise 0% (1 draft(s) x 200 seasons, 2 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender (next by phi / doubles value) at the $30k floor; teams took 18.0 floor players per draft, leaving 18 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: none. All: Daria Walczak #44F, Hien Truong #37M, Pablo Tellez #38M, Shelby Bates #47F, Liz Truluck #48F, Zoey Weil #50F, Connor Mogle #42M, Max Freeman #43M, Eugenia Carolina Lopez Ascarate #55F, Donald Young #44M, Harsh Mehta #45M, Alex Emery #46M, Genie Erokhina #56F, Juan Benitez #49M, Tyson McGuffin #50M, Martin Emmrich #53M, Patrick Kawka #56M, Cailyn Campbell #60F.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 65.3% | 34.0% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 2.0 | 51.0% | 5.5% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 8.0 | 53.1% | 6.0% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 6.0 | 52.3% | 3.5% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 10.0 | 49.9% | 6.5% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 11.0 | 48.5% | 2.0% | YES |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 3.0 | 53.3% | 7.0% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 4.0 | 50.8% | 6.0% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 5.0 | 49.3% | 3.5% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 7.0 | 48.6% | 3.0% | YES |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 9.0 | 47.9% | 4.0% | YES |
| Andrei Daescu (#6M) | $323k | 1.0 | 0% | 18.0 | 48.5% | 3.5% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 65 | 51 | 53 | 51 | 49 | 52 | 49 | 53 | 48 | 50 | 49 | 50 | 50 | 46 | 45 | 47 | 48 | 49 | 47 | 50 |
| title% | 34 | 6 | 7 | 6 | 4 | 4 | 3 | 6 | 4 | 6 | 2 | 4 | 1 | 2 | 0 | 1 | 3 | 4 | 2 | 3 |

## snake draft, owner noise 10% (30 draft(s) x 200 seasons, 53 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender (next by phi / doubles value) at the $30k floor; teams took 18.6 floor players per draft, leaving 42 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Augustus Ge (#30M, $102k, 3% of drafts). All: Genie Erokhina #56F, Patrick Kawka #56M, Cailyn Campbell #60F, Shelby Bates #47F, Martin Emmrich #53M, Callie Smith #59F, Zoey Weil #50F, Tyson McGuffin #50M, Liz Truluck #48F, Juan Benitez #49M, Max Freeman #43M, Alex Emery #46M, Tristan Dussault #55M, Pablo Tellez #38M, Connor Mogle #42M, Donald Young #44M, Etienne Blaszkewycz #48M, Allie Reichert #51F, Hien Truong #37M, Anderson Scarpa #39M, Eugenia Carolina Lopez Ascarate #55F, Joseph Wild #41M, Milan Rane #40F, Harsh Mehta #45M, Rafa Hewett #52M, Estee Widdershoven #42F, Thomas Yu #34M, Daria Walczak #44F, George Wall #36M, Maggie Brascia #41F, Lucy Kovalova #46F, Luca Mack #40M, Isabella Dunlap #38F, Mohaned Alhouni #58M, Samantha Parker #35F, Oscar Serra #33M, Augustus Ge #30M, Ting Chieh Wei #33F, Carlota Trevino #34F, Jessie Irvine #37F, Tama Shimabukuro #47M, Hoang Nam Ly #59M.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 65.7% | 34.1% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 2.5 | 52.4% | 6.5% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 6.3 | 51.2% | 4.5% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 7.3 | 51.3% | 4.9% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 9.6 | 50.2% | 3.9% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 9.8 | 49.8% | 3.8% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 2.7 | 50.7% | 4.6% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 4.4 | 49.8% | 3.4% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 6.1 | 50.2% | 3.4% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 7.9 | 48.2% | 2.6% | YES |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 9.2 | 48.8% | 3.2% | YES |
| Andrei Daescu (#6M) | $323k | 1.0 | 0% | 16.6 | 46.9% | 2.6% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 66 | 51 | 51 | 51 | 50 | 50 | 51 | 49 | 50 | 50 | 49 | 48 | 49 | 47 | 48 | 47 | 47 | 48 | 49 | 49 |
| title% | 34 | 5 | 6 | 4 | 4 | 4 | 4 | 3 | 4 | 4 | 3 | 3 | 3 | 3 | 2 | 3 | 2 | 2 | 3 | 4 |

## snake draft, owner noise 25% (30 draft(s) x 200 seasons, 54 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender (next by phi / doubles value) at the $30k floor; teams took 19.7 floor players per draft, leaving 53 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Jillian Braverman (#26F, $149k, 17% of drafts), Leigh Waters (#24F, $157k, 10% of drafts), Yana Newell (#28F, $144k, 10% of drafts), Alix Truong (#30F, $132k, 10% of drafts), CJ Klinger (#28M, $111k, 10% of drafts), Armaan Bhatia (#25M, $130k, 7% of drafts), Dekel Bar (#27M, $111k, 7% of drafts), Roscoe Bellamy (#29M, $107k, 7% of drafts), Allyce Jones (#25F, $154k, 3% of drafts), Augustus Ge (#30M, $102k, 3% of drafts). All: Patrick Kawka #56M, Callie Smith #59F, Martin Emmrich #53M, Tyson McGuffin #50M, Shelby Bates #47F, Liz Truluck #48F, Cailyn Campbell #60F, Pablo Tellez #38M, Anderson Scarpa #39M, Zoey Weil #50F, Max Freeman #43M, Genie Erokhina #56F, Juan Benitez #49M, Rafa Hewett #52M, Lucy Kovalova #46F, Hien Truong #37M, Allie Reichert #51F, Connor Mogle #42M, Alex Emery #46M, Tristan Dussault #55M, Thomas Yu #34M, Joseph Wild #41M, Harsh Mehta #45M, Etienne Blaszkewycz #48M, Ting Chieh Wei #33F, Eugenia Carolina Lopez Ascarate #55F, Donald Young #44M, George Wall #36M, Milan Rane #40F, Daria Walczak #44F, Isabella Dunlap #38F, Maggie Brascia #41F, Estee Widdershoven #42F, Luca Mack #40M, Mohaned Alhouni #58M, Oscar Serra #33M, Jessie Irvine #37F, Carlota Trevino #34F, Christine Maddox #58F, Jillian Braverman #26F, Yufei Long #45F, Leigh Waters #24F, Yana Newell #28F, Alix Truong #30F, CJ Klinger #28M, Samantha Parker #35F, Armaan Bhatia #25M, Dekel Bar #27M, Roscoe Bellamy #29M, Tama Shimabukuro #47M, Allyce Jones #25F, Augustus Ge #30M, Roos Van Reek #36F.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 65.0% | 33.0% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 3.2 | 51.3% | 4.7% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 7.3 | 50.5% | 4.3% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 7.8 | 51.0% | 4.5% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 9.3 | 50.7% | 4.6% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 9.8 | 49.6% | 3.5% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 2.9 | 50.5% | 3.9% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 5.3 | 49.9% | 4.0% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 6.9 | 47.7% | 2.6% | YES |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 7.0 | 48.9% | 2.9% | YES |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 9.0 | 49.4% | 3.2% | no |
| Andrei Daescu (#6M) | $323k | 1.1 | 0% | 16.9 | 48.1% | 3.3% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 65 | 51 | 50 | 51 | 50 | 49 | 49 | 50 | 49 | 50 | 49 | 50 | 49 | 49 | 48 | 49 | 48 | 48 | 49 | 49 |
| title% | 33 | 4 | 4 | 4 | 4 | 3 | 4 | 3 | 4 | 4 | 4 | 4 | 4 | 4 | 3 | 3 | 3 | 3 | 3 | 3 |

## linear draft, owner noise 0% (1 draft(s) x 200 seasons, 2 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender (next by phi / doubles value) at the $30k floor; teams took 20.0 floor players per draft, leaving 20 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: none. All: Thomas Yu #34M, Estee Widdershoven #42F, Daria Walczak #44F, Hien Truong #37M, Shelby Bates #47F, Joseph Wild #41M, Zoey Weil #50F, Connor Mogle #42M, Max Freeman #43M, Donald Young #44M, Harsh Mehta #45M, Alex Emery #46M, Genie Erokhina #56F, Etienne Blaszkewycz #48M, Juan Benitez #49M, Tyson McGuffin #50M, Martin Emmrich #53M, Patrick Kawka #56M, Callie Smith #59F, Cailyn Campbell #60F.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 67.8% | 37.5% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 6.0 | 51.9% | 2.0% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 5.0 | 52.4% | 4.0% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 4.0 | 50.4% | 5.5% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 9.0 | 53.1% | 7.5% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 10.0 | 53.7% | 7.5% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 2.0 | 51.6% | 5.0% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 3.0 | 52.8% | 5.5% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 7.0 | 52.6% | 2.5% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 8.0 | 52.6% | 5.5% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 11.0 | 50.5% | 5.5% | no |
| Andrei Daescu (#6M) | $323k | 1.0 | 0% | 17.0 | 45.6% | 0.5% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 68 | 52 | 53 | 50 | 52 | 52 | 53 | 53 | 53 | 54 | 50 | 51 | 49 | 45 | 48 | 47 | 46 | 43 | 43 | 39 |
| title% | 38 | 5 | 6 | 6 | 4 | 2 | 2 | 6 | 8 | 8 | 6 | 4 | 1 | 2 | 2 | 1 | 0 | 0 | 0 | 0 |

## linear draft, owner noise 10% (30 draft(s) x 200 seasons, 57 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender (next by phi / doubles value) at the $30k floor; teams took 20.4 floor players per draft, leaving 45 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Alix Truong (#30F, $132k, 3% of drafts). All: Patrick Kawka #56M, Cailyn Campbell #60F, Tyson McGuffin #50M, Martin Emmrich #53M, Callie Smith #59F, Liz Truluck #48F, Genie Erokhina #56F, Zoey Weil #50F, Max Freeman #43M, Anderson Scarpa #39M, Shelby Bates #47F, Connor Mogle #42M, Donald Young #44M, Juan Benitez #49M, Tristan Dussault #55M, Hien Truong #37M, Pablo Tellez #38M, Joseph Wild #41M, Etienne Blaszkewycz #48M, Rafa Hewett #52M, Daria Walczak #44F, Yufei Long #45F, Alex Emery #46M, Milan Rane #40F, Eugenia Carolina Lopez Ascarate #55F, Harsh Mehta #45M, Isabella Dunlap #38F, Jessie Irvine #37F, Thomas Yu #34M, Estee Widdershoven #42F, Allie Reichert #51F, Mohaned Alhouni #58M, Carlota Trevino #34F, Oscar Serra #33M, Lucy Kovalova #46F, Luca Mack #40M, Christine Maddox #58F, Ting Chieh Wei #33F, George Wall #36M, Maggie Brascia #41F, Roos Van Reek #36F, Samantha Parker #35F, Mary Brascia #52F, Alix Truong #30F, Tama Shimabukuro #47M.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 66.8% | 34.9% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 4.4 | 52.6% | 5.5% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 8.1 | 51.8% | 4.0% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 8.4 | 52.1% | 5.1% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 8.3 | 51.7% | 4.6% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 9.4 | 52.2% | 5.0% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 3.5 | 52.2% | 5.1% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 5.1 | 52.2% | 5.0% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 6.0 | 51.9% | 4.7% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 6.1 | 51.0% | 3.8% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 7.6 | 52.1% | 5.2% | no |
| Andrei Daescu (#6M) | $323k | 1.0 | 0% | 16.5 | 44.0% | 1.5% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 67 | 53 | 52 | 52 | 52 | 52 | 52 | 52 | 52 | 52 | 51 | 52 | 50 | 50 | 48 | 46 | 44 | 43 | 42 | 39 |
| title% | 35 | 6 | 5 | 5 | 5 | 5 | 5 | 4 | 5 | 4 | 4 | 5 | 3 | 3 | 2 | 2 | 1 | 1 | 1 | 0 |

## linear draft, owner noise 25% (30 draft(s) x 200 seasons, 57 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender (next by phi / doubles value) at the $30k floor; teams took 20.7 floor players per draft, leaving 58 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Alix Truong (#30F, $132k, 17% of drafts), Dekel Bar (#27M, $111k, 17% of drafts), Yuta Funemizu (#26M, $119k, 13% of drafts), Augustus Ge (#30M, $102k, 13% of drafts), Layne Sleeth (#27F, $147k, 10% of drafts), CJ Klinger (#28M, $111k, 10% of drafts), Leigh Waters (#24F, $157k, 7% of drafts), Jillian Braverman (#26F, $149k, 7% of drafts), Ewa Radzikowska (#22F, $176k, 3% of drafts), Allyce Jones (#25F, $154k, 3% of drafts), Yana Newell (#28F, $144k, 3% of drafts), Vanshik Kapadia (#24M, $132k, 3% of drafts), Armaan Bhatia (#25M, $130k, 3% of drafts). All: Patrick Kawka #56M, Martin Emmrich #53M, Cailyn Campbell #60F, Tyson McGuffin #50M, Anderson Scarpa #39M, Genie Erokhina #56F, Callie Smith #59F, Hien Truong #37M, Juan Benitez #49M, Tristan Dussault #55M, Shelby Bates #47F, Liz Truluck #48F, Zoey Weil #50F, Alex Emery #46M, Thomas Yu #34M, Pablo Tellez #38M, Connor Mogle #42M, Max Freeman #43M, Donald Young #44M, Lucy Kovalova #46F, Etienne Blaszkewycz #48M, Rafa Hewett #52M, Isabella Dunlap #38F, Milan Rane #40F, Allie Reichert #51F, Eugenia Carolina Lopez Ascarate #55F, Ting Chieh Wei #33F, George Wall #36M, Daria Walczak #44F, Joseph Wild #41M, Mohaned Alhouni #58M, Maggie Brascia #41F, Oscar Serra #33M, Jessie Irvine #37F, Estee Widdershoven #42F, Harsh Mehta #45M, Christine Maddox #58F, Carlota Trevino #34F, Yufei Long #45F, Samantha Parker #35F, Alix Truong #30F, Dekel Bar #27M, Yuta Funemizu #26M, Augustus Ge #30M, Luca Mack #40M, Layne Sleeth #27F, CJ Klinger #28M, Leigh Waters #24F, Jillian Braverman #26F, Tama Shimabukuro #47M, Hoang Nam Ly #59M, Grayson Goldin #60M, Ewa Radzikowska #22F, Allyce Jones #25F, Yana Newell #28F, Vanshik Kapadia #24M, Armaan Bhatia #25M, Jaume Martinez Vich #31M.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.1 | 65.3% | 32.1% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 6.2 | 52.4% | 5.5% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 8.2 | 51.0% | 4.2% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 8.8 | 51.4% | 4.5% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 9.3 | 50.4% | 4.3% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 10.3 | 50.6% | 3.6% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 3.5 | 51.3% | 5.0% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 5.0 | 51.7% | 5.1% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 5.8 | 51.0% | 4.4% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 5.6 | 51.4% | 4.3% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 6.5 | 51.8% | 4.8% | no |
| Andrei Daescu (#6M) | $323k | 1.1 | 0% | 15.1 | 45.3% | 1.6% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 65 | 52 | 51 | 52 | 51 | 51 | 51 | 52 | 52 | 51 | 51 | 51 | 50 | 49 | 48 | 48 | 45 | 44 | 44 | 42 |
| title% | 30 | 6 | 4 | 5 | 5 | 5 | 5 | 5 | 5 | 4 | 4 | 4 | 4 | 3 | 2 | 3 | 2 | 1 | 1 | 1 |
