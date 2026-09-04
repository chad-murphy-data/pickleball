# Draft simulation -- 20 teams, scarcity, varied draft and information

Prices: alpha = 1.0, one joint $20M pool, **Anna Leigh Waters franchise-tagged at $769,149** (`phase2_pricing.prices_tagged`).

20 teams, $1M cap, 6 rounds (3M+3W), one pick per turn. Owners project a final roster for each affordable candidate (candidate + greedy fill of the best-believed players still available) and take the one whose projection has the highest believed tie probability against a reference roster of doubles ranks (10, 30, 50) per gender. Noise = sd of each owner's belief error as a fraction of the gender's pool spread (men 0.143, women 0.232 logit), fixed per owner per draft. Seasons: double round robin (38 ties) + top-4 playoff, scored with the TRUE values. Parity = every team 50% expected wins, 5% title. Built by `draft_sim.py`.

## Summary

| draft | owner noise | drafts | parity spread (sd of team win%) | strongest team win% | mean spend | blueprint mix |
|---|---|---|---|---|---|---|
| snake | 0% | 1 | 7.1 pts | 67.8% | $960k | star-led 55%, anchor 30%, superstar 10%, balanced 5% |
| snake | 10% | 30 | 6.2 pts | 66.1% | $963k | star-led 58%, anchor 28%, superstar 10%, balanced 4% |
| snake | 25% | 30 | 5.9 pts | 65.1% | $958k | star-led 60%, anchor 26%, superstar 10%, balanced 4% |
| linear | 0% | 1 | 5.5 pts | 67.9% | $966k | star-led 70%, anchor 20%, superstar 10% |
| linear | 10% | 30 | 5.9 pts | 66.5% | $963k | star-led 63%, anchor 24%, superstar 10%, balanced 3% |
| linear | 25% | 30 | 6.1 pts | 65.5% | $955k | star-led 61%, anchor 24%, superstar 10%, balanced 4% |

## snake draft, owner noise 0% (1 draft(s) x 200 seasons, 2 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender at the $30k floor; teams took 23.0 floor players per draft, leaving 23 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Roscoe Bellamy (#29M, $107k, 100% of drafts). All: Roscoe Bellamy #29M, Oscar Serra #33M, John Lucian Goins #35M, Daria Walczak #44F, Pablo Tellez #38M, Shelby Bates #47F, Luca Mack #40M, Liz Truluck #48F, Zoey Weil #50F, Max Freeman #43M, Eugenia Carolina Lopez Ascarate #55F, Donald Young #44M, Alex Emery #46M, Genie Erokhina #56F, Etienne Blaszkewycz #48M, Juan Benitez #49M, Tyson McGuffin #50M, Rafa Hewett #52M, Christine Maddox #58F, Martin Emmrich #53M, Patrick Kawka #56M, Callie Smith #59F, Cailyn Campbell #60F.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 67.8% | 38.5% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 2.0 | 52.1% | 5.0% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 8.0 | 52.2% | 3.0% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 6.0 | 52.3% | 2.0% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 14.0 | 51.4% | 3.5% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 13.0 | 49.2% | 2.5% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 3.0 | 52.2% | 4.5% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 4.0 | 53.2% | 6.5% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 5.0 | 51.6% | 3.5% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 7.0 | 53.6% | 4.0% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 9.0 | 48.7% | 2.5% | YES |
| Andrei Daescu (#6M) | $323k | 1.0 | 0% | 18.0 | 53.3% | 8.5% | no |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 68 | 52 | 52 | 53 | 52 | 52 | 54 | 52 | 49 | 42 | 35 | 32 | 49 | 51 | 48 | 52 | 51 | 53 | 52 | 51 |
| title% | 38 | 5 | 4 | 6 | 4 | 2 | 4 | 3 | 2 | 1 | 0 | 0 | 2 | 4 | 2 | 2 | 4 | 8 | 2 | 4 |

## snake draft, owner noise 10% (30 draft(s) x 200 seasons, 46 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender at the $30k floor; teams took 21.5 floor players per draft, leaving 49 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Alix Truong (#30F, $132k, 3% of drafts), Augustus Ge (#30M, $102k, 3% of drafts). All: Tyson McGuffin #50M, Rafa Hewett #52M, Patrick Kawka #56M, Cailyn Campbell #60F, Callie Smith #59F, Liz Truluck #48F, Genie Erokhina #56F, Martin Emmrich #53M, Donald Young #44M, Shelby Bates #47F, Juan Benitez #49M, Zoey Weil #50F, Tristan Dussault #55M, Pablo Tellez #38M, Anderson Scarpa #39M, Connor Mogle #42M, Max Freeman #43M, Yufei Long #45F, Hien Truong #37M, Luca Mack #40M, Allie Reichert #51F, Harsh Mehta #45M, Alex Emery #46M, Joseph Wild #41M, Isabella Dunlap #38F, Daria Walczak #44F, Eugenia Carolina Lopez Ascarate #55F, Thomas Yu #34M, Etienne Blaszkewycz #48M, Mohaned Alhouni #58M, Milan Rane #40F, Estee Widdershoven #42F, Oscar Serra #33M, Roos Van Reek #36F, Lucy Kovalova #46F, Christine Maddox #58F, George Wall #36M, Jessie Irvine #37F, Marianna Petrei #43F, Ting Chieh Wei #33F, Carlota Trevino #34F, Samantha Parker #35F, Maggie Brascia #41F, Hoang Nam Ly #59M, Alix Truong #30F, Judit Castillo #32F, Augustus Ge #30M, John Lucian Goins #35M, Tama Shimabukuro #47M.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 66.1% | 34.4% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 2.5 | 51.9% | 4.3% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 6.6 | 51.8% | 4.4% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 7.4 | 51.6% | 4.7% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 10.6 | 50.2% | 3.8% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 10.9 | 49.9% | 3.4% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 2.7 | 51.0% | 4.2% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 4.4 | 51.7% | 4.2% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 6.1 | 51.0% | 4.1% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 8.2 | 51.0% | 3.6% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 9.8 | 50.5% | 4.0% | no |
| Andrei Daescu (#6M) | $323k | 1.1 | 0% | 18.3 | 50.6% | 3.9% | no |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 66 | 51 | 52 | 52 | 51 | 52 | 52 | 51 | 51 | 44 | 40 | 42 | 49 | 47 | 49 | 49 | 50 | 49 | 51 | 51 |
| title% | 34 | 4 | 5 | 4 | 4 | 5 | 5 | 4 | 4 | 2 | 1 | 1 | 3 | 2 | 3 | 3 | 4 | 3 | 4 | 5 |

## snake draft, owner noise 25% (30 draft(s) x 200 seasons, 47 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender at the $30k floor; teams took 21.5 floor players per draft, leaving 60 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Augustus Ge (#30M, $102k, 20% of drafts), Jillian Braverman (#26F, $149k, 13% of drafts), Yana Newell (#28F, $144k, 13% of drafts), Alix Truong (#30F, $132k, 13% of drafts), Yuta Funemizu (#26M, $119k, 13% of drafts), Allyce Jones (#25F, $154k, 10% of drafts), CJ Klinger (#28M, $111k, 10% of drafts), Roscoe Bellamy (#29M, $107k, 10% of drafts), Dekel Bar (#27M, $111k, 7% of drafts), Leigh Waters (#24F, $157k, 3% of drafts), Layne Sleeth (#27F, $147k, 3% of drafts), Vanshik Kapadia (#24M, $132k, 3% of drafts). All: Rafa Hewett #52M, Callie Smith #59F, Tyson McGuffin #50M, Martin Emmrich #53M, Patrick Kawka #56M, Genie Erokhina #56F, Hien Truong #37M, Pablo Tellez #38M, Tristan Dussault #55M, Cailyn Campbell #60F, Liz Truluck #48F, Zoey Weil #50F, Shelby Bates #47F, Donald Young #44M, Allie Reichert #51F, Alex Emery #46M, Juan Benitez #49M, Isabella Dunlap #38F, Anderson Scarpa #39M, Connor Mogle #42M, Yufei Long #45F, Luca Mack #40M, Thomas Yu #34M, Estee Widdershoven #42F, Daria Walczak #44F, Max Freeman #43M, Harsh Mehta #45M, Mohaned Alhouni #58M, Milan Rane #40F, George Wall #36M, Joseph Wild #41M, Eugenia Carolina Lopez Ascarate #55F, Ting Chieh Wei #33F, Oscar Serra #33M, Samantha Parker #35F, Jessie Irvine #37F, Maggie Brascia #41F, Lucy Kovalova #46F, Tama Shimabukuro #47M, Etienne Blaszkewycz #48M, Augustus Ge #30M, Carlota Trevino #34F, Jillian Braverman #26F, Yana Newell #28F, Alix Truong #30F, Yuta Funemizu #26M, Christine Maddox #58F, Allyce Jones #25F, CJ Klinger #28M, Roscoe Bellamy #29M, Dekel Bar #27M, Roos Van Reek #36F, Hoang Nam Ly #59M, Leigh Waters #24F, Layne Sleeth #27F, Vanshik Kapadia #24M, Jaume Martinez Vich #31M, Marianna Petrei #43F, Matthew Barlow #57M, Grayson Goldin #60M.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 65.1% | 32.1% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 3.2 | 51.4% | 4.2% | no |
| Parris Todd (#3F) | $489k | 1.1 | 0% | 8.5 | 51.3% | 4.6% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 8.1 | 51.0% | 4.0% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 10.1 | 50.8% | 3.5% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 11.4 | 49.7% | 3.8% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 2.9 | 51.1% | 4.2% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 5.3 | 50.6% | 4.0% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 7.1 | 50.3% | 4.0% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 7.9 | 50.5% | 3.9% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 9.5 | 50.2% | 3.7% | no |
| Andrei Daescu (#6M) | $323k | 1.4 | 0% | 17.2 | 50.9% | 4.2% | no |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 65 | 51 | 51 | 51 | 51 | 51 | 51 | 51 | 50 | 46 | 42 | 45 | 49 | 46 | 49 | 49 | 49 | 49 | 51 | 51 |
| title% | 32 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 2 | 2 | 2 | 4 | 2 | 3 | 4 | 4 | 4 | 5 | 5 |

## linear draft, owner noise 0% (1 draft(s) x 200 seasons, 2 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender at the $30k floor; teams took 21.0 floor players per draft, leaving 21 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: none. All: Estee Widdershoven #42F, Daria Walczak #44F, Yufei Long #45F, Pablo Tellez #38M, Shelby Bates #47F, Joseph Wild #41M, Liz Truluck #48F, Zoey Weil #50F, Connor Mogle #42M, Max Freeman #43M, Donald Young #44M, Harsh Mehta #45M, Alex Emery #46M, Genie Erokhina #56F, Etienne Blaszkewycz #48M, Tyson McGuffin #50M, Rafa Hewett #52M, Martin Emmrich #53M, Patrick Kawka #56M, Callie Smith #59F, Cailyn Campbell #60F.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 67.9% | 38.0% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 4.0 | 50.4% | 3.5% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 10.0 | 47.5% | 3.0% | YES |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 12.0 | 50.3% | 4.5% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 11.0 | 51.9% | 4.0% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 14.0 | 51.7% | 4.5% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 2.0 | 51.4% | 5.0% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 5.0 | 53.3% | 6.0% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 7.0 | 52.5% | 3.5% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 13.0 | 50.9% | 1.5% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 15.0 | 47.6% | 2.0% | YES |
| Andrei Daescu (#6M) | $323k | 2.0 | 0% | 6.0 | 50.5% | 3.0% | no |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 68 | 51 | 56 | 50 | 53 | 50 | 53 | 51 | 42 | 47 | 52 | 50 | 51 | 52 | 48 | 49 | 46 | 47 | 44 | 41 |
| title% | 38 | 5 | 10 | 4 | 6 | 3 | 4 | 4 | 0 | 3 | 4 | 4 | 2 | 4 | 2 | 0 | 1 | 5 | 0 | 0 |

## linear draft, owner noise 10% (30 draft(s) x 200 seasons, 47 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender at the $30k floor; teams took 21.7 floor players per draft, leaving 55 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Brooke Buckner (#29F, $134k, 3% of drafts), Roscoe Bellamy (#29M, $107k, 3% of drafts), Augustus Ge (#30M, $102k, 3% of drafts). All: Patrick Kawka #56M, Callie Smith #59F, Rafa Hewett #52M, Liz Truluck #48F, Donald Young #44M, Genie Erokhina #56F, Tyson McGuffin #50M, Cailyn Campbell #60F, Max Freeman #43M, Martin Emmrich #53M, Pablo Tellez #38M, Zoey Weil #50F, Connor Mogle #42M, Tristan Dussault #55M, Shelby Bates #47F, Juan Benitez #49M, Hien Truong #37M, Luca Mack #40M, Joseph Wild #41M, Alex Emery #46M, Daria Walczak #44F, Yufei Long #45F, Eugenia Carolina Lopez Ascarate #55F, Milan Rane #40F, Anderson Scarpa #39M, Harsh Mehta #45M, Etienne Blaszkewycz #48M, Mohaned Alhouni #58M, Allie Reichert #51F, Isabella Dunlap #38F, Thomas Yu #34M, Lucy Kovalova #46F, Oscar Serra #33M, Maggie Brascia #41F, Estee Widdershoven #42F, Jessie Irvine #37F, Christine Maddox #58F, Roos Van Reek #36F, George Wall #36M, Ting Chieh Wei #33F, Carlota Trevino #34F, Tama Shimabukuro #47M, Genie Bouchard #57F, Samantha Parker #35F, Marianna Petrei #43F, Brooke Buckner #29F, Lea Jansen #31F, Roscoe Bellamy #29M, Judit Castillo #32F, Augustus Ge #30M, John Lucian Goins #35M, Kiora Kunimoto #49F, Mary Brascia #52F, Hoang Nam Ly #59M, Grayson Goldin #60M.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.0 | 66.5% | 34.5% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 4.7 | 52.1% | 5.1% | no |
| Parris Todd (#3F) | $489k | 1.0 | 0% | 9.9 | 50.9% | 4.4% | no |
| Jorja Johnson (#4F) | $475k | 1.0 | 0% | 9.7 | 50.3% | 3.4% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 10.4 | 50.3% | 3.7% | no |
| Jade Kawamoto (#6F) | $444k | 1.0 | 0% | 12.2 | 50.7% | 4.5% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 3.7 | 52.0% | 4.6% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 6.4 | 52.3% | 5.1% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 7.6 | 52.1% | 5.0% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 14.4 | 50.3% | 4.2% | no |
| Christian Alshon (#5M) | $400k | 1.0 | 0% | 11.2 | 51.4% | 4.4% | no |
| Andrei Daescu (#6M) | $323k | 1.6 | 0% | 10.8 | 47.9% | 3.4% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 67 | 52 | 52 | 52 | 51 | 51 | 51 | 50 | 47 | 49 | 51 | 46 | 50 | 49 | 50 | 50 | 47 | 46 | 45 | 43 |
| title% | 34 | 6 | 6 | 5 | 4 | 4 | 4 | 4 | 3 | 3 | 5 | 2 | 4 | 3 | 3 | 3 | 2 | 2 | 1 | 1 |

## linear draft, owner noise 25% (30 draft(s) x 200 seasons, 47 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender at the $30k floor; teams took 22.2 floor players per draft, leaving 67 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Alix Truong (#30F, $132k, 20% of drafts), Yuta Funemizu (#26M, $119k, 20% of drafts), Augustus Ge (#30M, $102k, 20% of drafts), Layne Sleeth (#27F, $147k, 10% of drafts), Meghan Dizon (#23F, $171k, 7% of drafts), Leigh Waters (#24F, $157k, 7% of drafts), Allyce Jones (#25F, $154k, 7% of drafts), Jillian Braverman (#26F, $149k, 7% of drafts), Yana Newell (#28F, $144k, 7% of drafts), Vanshik Kapadia (#24M, $132k, 7% of drafts), Dekel Bar (#27M, $111k, 7% of drafts), Roscoe Bellamy (#29M, $107k, 7% of drafts), Etta Tuionetoa (#21F, $187k, 3% of drafts), Ewa Radzikowska (#22F, $176k, 3% of drafts), Will Howells (#20M, $154k, 3% of drafts), Dylan Frazier (#23M, $133k, 3% of drafts), CJ Klinger (#28M, $111k, 3% of drafts). All: Tyson McGuffin #50M, Patrick Kawka #56M, Callie Smith #59F, Rafa Hewett #52M, Genie Erokhina #56F, Cailyn Campbell #60F, Martin Emmrich #53M, Pablo Tellez #38M, Zoey Weil #50F, Hien Truong #37M, Shelby Bates #47F, Max Freeman #43M, Alex Emery #46M, Juan Benitez #49M, Liz Truluck #48F, Donald Young #44M, Anderson Scarpa #39M, Oscar Serra #33M, Isabella Dunlap #38F, Thomas Yu #34M, Luca Mack #40M, Tristan Dussault #55M, Milan Rane #40F, Daria Walczak #44F, Lucy Kovalova #46F, Connor Mogle #42M, Eugenia Carolina Lopez Ascarate #55F, Mohaned Alhouni #58M, George Wall #36M, Maggie Brascia #41F, Estee Widdershoven #42F, Joseph Wild #41M, Allie Reichert #51F, Harsh Mehta #45M, Ting Chieh Wei #33F, Carlota Trevino #34F, Yufei Long #45F, Christine Maddox #58F, Jessie Irvine #37F, Tama Shimabukuro #47M, Alix Truong #30F, Yuta Funemizu #26M, Augustus Ge #30M, Etienne Blaszkewycz #48M, Hoang Nam Ly #59M, Layne Sleeth #27F, Judit Castillo #32F, Samantha Parker #35F, Marianna Petrei #43F, Meghan Dizon #23F, Leigh Waters #24F, Allyce Jones #25F, Jillian Braverman #26F, Yana Newell #28F, Vanshik Kapadia #24M, Dekel Bar #27M, Roscoe Bellamy #29M, Etta Tuionetoa #21F, Ewa Radzikowska #22F, Will Howells #20M, Dylan Frazier #23M, CJ Klinger #28M, Jaume Martinez Vich #31M, Roos Van Reek #36F, Kiora Kunimoto #49F, Seone Mendez #54F, Grayson Goldin #60M.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $769k | 1.0 | 0% | 1.1 | 65.5% | 32.5% | no |
| Anna Bright (#2F) | $613k | 1.0 | 0% | 7.2 | 51.8% | 4.9% | no |
| Parris Todd (#3F) | $489k | 1.1 | 0% | 9.9 | 51.3% | 4.2% | no |
| Jorja Johnson (#4F) | $475k | 1.1 | 0% | 11.1 | 51.3% | 4.1% | no |
| Kate Fahey (#5F) | $453k | 1.0 | 0% | 12.3 | 50.4% | 3.7% | no |
| Jade Kawamoto (#6F) | $444k | 1.1 | 0% | 13.2 | 50.0% | 3.3% | no |
| Ben Johns (#1M) | $474k | 1.0 | 0% | 4.3 | 52.4% | 5.6% | no |
| JW Johnson (#2M) | $430k | 1.0 | 0% | 6.6 | 51.8% | 4.9% | no |
| Hayden Patriquin (#3M) | $421k | 1.0 | 0% | 8.0 | 51.4% | 4.5% | no |
| Gabriel Tardio (#4M) | $409k | 1.0 | 0% | 14.3 | 49.6% | 3.9% | no |
| Christian Alshon (#5M) | $400k | 1.1 | 0% | 9.1 | 51.7% | 4.6% | no |
| Andrei Daescu (#6M) | $323k | 1.6 | 0% | 11.7 | 47.3% | 2.6% | YES |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 65 | 53 | 51 | 52 | 51 | 51 | 51 | 51 | 49 | 51 | 48 | 50 | 48 | 46 | 49 | 50 | 47 | 46 | 47 | 44 |
| title% | 31 | 7 | 4 | 5 | 4 | 4 | 5 | 4 | 4 | 4 | 3 | 4 | 3 | 2 | 4 | 4 | 2 | 3 | 2 | 2 |
