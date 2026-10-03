# Draft simulation -- 20 teams, scarcity, varied draft and information

Prices: alpha = 0.845, one joint $20M pool, $30k floor (`phase2_pricing.prices`).

20 teams, $1M cap, 6 rounds (3M+3W), one pick per turn. Owners project a final roster for each affordable candidate (candidate + greedy fill of the best-believed players still available) and take the one whose projection has the highest believed tie probability against a reference roster of doubles ranks (10, 30, 50) per gender. Noise = sd of each owner's belief error as a fraction of the gender's pool spread (men 0.143, women 0.232 logit), fixed per owner per draft. Seasons: double round robin (38 ties) + top-4 playoff, scored with the TRUE values. Parity = every team 50% expected wins, 5% title. Built by `draft_sim.py`.

## Summary

| draft | owner noise | drafts | parity spread (sd of team win%) | strongest team win% | mean spend | blueprint mix |
|---|---|---|---|---|---|---|
| snake | 0% | 1 | 9.6 pts | 67.2% | $932k | anchor 60%, star-led 25%, balanced 10%, superstar 5% |
| snake | 10% | 30 | 9.7 pts | 67.7% | $926k | anchor 54%, star-led 29%, balanced 12%, superstar 5% |
| snake | 25% | 30 | 9.2 pts | 66.7% | $926k | anchor 50%, star-led 30%, balanced 15%, superstar 5% |
| linear | 0% | 1 | 12.8 pts | 71.5% | $852k | anchor 55%, star-led 25%, balanced 15%, superstar 5% |
| linear | 10% | 30 | 10.6 pts | 69.8% | $900k | anchor 51%, star-led 28%, balanced 14%, superstar 6% |
| linear | 25% | 30 | 9.8 pts | 67.9% | $913k | anchor 49%, star-led 32%, balanced 14%, superstar 6% |

## snake draft, owner noise 0% (1 draft(s) x 200 seasons, 2 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender at the $30k floor; teams took 31.0 floor players per draft, leaving 31 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Roscoe Bellamy (#29M, $119k, 100% of drafts). All: Roscoe Bellamy #29M, Roos Van Reek #36F, Oscar Serra #33M, George Wall #36M, Daria Walczak #44F, Yufei Long #45F, Hien Truong #37M, Pablo Tellez #38M, Shelby Bates #47F, Luca Mack #40M, Joseph Wild #41M, Liz Truluck #48F, Zoey Weil #50F, Connor Mogle #42M, Max Freeman #43M, Harsh Mehta #45M, Alex Emery #46M, Genie Erokhina #56F, Etienne Blaszkewycz #48M, Juan Benitez #49M, Tyson McGuffin #50M, Rafa Hewett #52M, Christine Maddox #58F, Martin Emmrich #53M, Tristan Dussault #55M, Patrick Kawka #56M, Callie Smith #59F, Mohaned Alhouni #58M, Cailyn Campbell #60F, Hoang Nam Ly #59M, Grayson Goldin #60M.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $725k | 1.0 | 0% | 1.0 | 67.2% | 29.0% | no |
| Anna Bright (#2F) | $520k | 1.0 | 0% | 3.0 | 57.5% | 7.5% | no |
| Parris Todd (#3F) | $431k | 1.0 | 0% | 7.0 | 57.2% | 9.5% | no |
| Jorja Johnson (#4F) | $421k | 1.0 | 0% | 10.0 | 58.2% | 7.0% | no |
| Kate Fahey (#5F) | $404k | 1.0 | 0% | 16.0 | 52.2% | 3.0% | no |
| Jade Kawamoto (#6F) | $397k | 2.0 | 0% | 17.0 | 60.2% | 12.0% | no |
| Ben Johns (#1M) | $420k | 2.0 | 0% | 8.0 | 56.5% | 8.5% | no |
| JW Johnson (#2M) | $387k | 2.0 | 0% | 10.0 | 58.2% | 7.0% | no |
| Hayden Patriquin (#3M) | $380k | 2.0 | 0% | 6.0 | 55.9% | 6.0% | no |
| Gabriel Tardio (#4M) | $371k | 1.0 | 0% | 17.0 | 60.2% | 12.0% | no |
| Christian Alshon (#5M) | $364k | 2.0 | 0% | 14.0 | 57.1% | 3.5% | no |
| Andrei Daescu (#6M) | $304k | 2.0 | 0% | 2.0 | 54.3% | 2.0% | no |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 67 | 54 | 57 | 55 | 38 | 56 | 57 | 56 | 40 | 58 | 40 | 43 | 56 | 57 | 47 | 52 | 60 | 36 | 35 | 35 |
| title% | 29 | 2 | 8 | 4 | 0 | 6 | 10 | 8 | 0 | 7 | 0 | 0 | 6 | 4 | 2 | 3 | 12 | 0 | 0 | 0 |

## snake draft, owner noise 10% (30 draft(s) x 200 seasons, 47 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender at the $30k floor; teams took 29.8 floor players per draft, leaving 79 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Brooke Buckner (#29F, $145k, 20% of drafts), Allyce Jones (#25F, $163k, 17% of drafts), Layne Sleeth (#27F, $156k, 13% of drafts), Yana Newell (#28F, $153k, 13% of drafts), Dekel Bar (#27M, $123k, 13% of drafts), Chao Yi Wang (#18F, $216k, 10% of drafts), CJ Klinger (#28M, $123k, 10% of drafts), Augustus Ge (#30M, $114k, 10% of drafts), Etta Tuionetoa (#21F, $192k, 7% of drafts), Meghan Dizon (#23F, $177k, 7% of drafts), Leigh Waters (#24F, $165k, 7% of drafts), Jillian Braverman (#26F, $158k, 7% of drafts), Yuta Funemizu (#26M, $130k, 7% of drafts), Roscoe Bellamy (#29M, $119k, 7% of drafts), Jackie Kawamoto (#9F, $357k, 3% of drafts), Danni-Elle Townsend (#12F, $306k, 3% of drafts), Catherine Parenteau (#15F, $259k, 3% of drafts), Lacy Schneemann (#17F, $224k, 3% of drafts), Ewa Radzikowska (#22F, $183k, 3% of drafts), Alix Truong (#30F, $142k, 3% of drafts). All: Donald Young #44M, Tristan Dussault #55M, Patrick Kawka #56M, Tyson McGuffin #50M, Mohaned Alhouni #58M, Liz Truluck #48F, Pablo Tellez #38M, Rafa Hewett #52M, Martin Emmrich #53M, Callie Smith #59F, Hien Truong #37M, Joseph Wild #41M, Cailyn Campbell #60F, Harsh Mehta #45M, Genie Erokhina #56F, Luca Mack #40M, Christine Maddox #58F, Shelby Bates #47F, Etienne Blaszkewycz #48M, Juan Benitez #49M, Marianna Petrei #43F, Yufei Long #45F, Zoey Weil #50F, Connor Mogle #42M, Alex Emery #46M, Thomas Yu #34M, Max Freeman #43M, Hoang Nam Ly #59M, Grayson Goldin #60M, Daria Walczak #44F, Anderson Scarpa #39M, Isabella Dunlap #38F, Mary Brascia #52F, Tama Shimabukuro #47M, Carlota Trevino #34F, Milan Rane #40F, George Wall #36M, Allie Reichert #51F, Eugenia Carolina Lopez Ascarate #55F, Genie Bouchard #57F, Roos Van Reek #36F, Oscar Serra #33M, Maggie Brascia #41F, Estee Widdershoven #42F, Brooke Buckner #29F, Jessie Irvine #37F, Kiora Kunimoto #49F, Allyce Jones #25F, Judit Castillo #32F, Samantha Parker #35F, John Lucian Goins #35M, Oliver Frank #54M, Matthew Barlow #57M, Layne Sleeth #27F, Yana Newell #28F, Dekel Bar #27M, Jaume Martinez Vich #31M, Kaitlyn Christian #39F, Lucy Kovalova #46F, Chao Yi Wang #18F, CJ Klinger #28M, Augustus Ge #30M, Etta Tuionetoa #21F, Meghan Dizon #23F, Leigh Waters #24F, Jillian Braverman #26F, Yuta Funemizu #26M, Lea Jansen #31F, Roscoe Bellamy #29M, Christopher Haworth #32M, Katerina Stewart #53F, Seone Mendez #54F, Jackie Kawamoto #9F, Danni-Elle Townsend #12F, Catherine Parenteau #15F, Lacy Schneemann #17F, Ewa Radzikowska #22F, Alix Truong #30F, Zane Ford #51M.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $725k | 1.0 | 0% | 1.0 | 67.7% | 30.9% | no |
| Anna Bright (#2F) | $520k | 1.0 | 0% | 3.8 | 58.5% | 9.7% | no |
| Parris Todd (#3F) | $431k | 1.4 | 0% | 14.4 | 56.0% | 7.0% | no |
| Jorja Johnson (#4F) | $421k | 1.4 | 0% | 14.2 | 55.1% | 6.1% | no |
| Kate Fahey (#5F) | $404k | 1.8 | 0% | 12.6 | 53.2% | 4.1% | no |
| Jade Kawamoto (#6F) | $397k | 2.0 | 0% | 12.4 | 52.2% | 4.9% | no |
| Ben Johns (#1M) | $420k | 3.1 | 0% | 11.7 | 55.0% | 5.2% | no |
| JW Johnson (#2M) | $387k | 1.7 | 0% | 12.7 | 57.1% | 8.0% | no |
| Hayden Patriquin (#3M) | $380k | 1.9 | 0% | 10.9 | 55.8% | 6.3% | no |
| Gabriel Tardio (#4M) | $371k | 2.0 | 0% | 11.9 | 56.2% | 6.8% | no |
| Christian Alshon (#5M) | $364k | 2.3 | 0% | 9.3 | 55.7% | 6.0% | no |
| Andrei Daescu (#6M) | $304k | 3.0 | 0% | 9.2 | 53.0% | 4.2% | no |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 68 | 57 | 57 | 52 | 53 | 52 | 50 | 52 | 47 | 49 | 47 | 48 | 47 | 47 | 49 | 47 | 42 | 42 | 46 | 48 |
| title% | 31 | 7 | 7 | 5 | 5 | 4 | 4 | 4 | 2 | 3 | 2 | 3 | 2 | 3 | 3 | 3 | 2 | 2 | 2 | 5 |

## snake draft, owner noise 25% (30 draft(s) x 200 seasons, 46 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender at the $30k floor; teams took 27.3 floor players per draft, leaving 90 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Augustus Ge (#30M, $114k, 37% of drafts), CJ Klinger (#28M, $123k, 27% of drafts), Roscoe Bellamy (#29M, $119k, 27% of drafts), Layne Sleeth (#27F, $156k, 20% of drafts), Allyce Jones (#25F, $163k, 17% of drafts), Yana Newell (#28F, $153k, 17% of drafts), Dekel Bar (#27M, $123k, 17% of drafts), Jillian Braverman (#26F, $158k, 13% of drafts), Meghan Dizon (#23F, $177k, 10% of drafts), Leigh Waters (#24F, $165k, 10% of drafts), Alix Truong (#30F, $142k, 10% of drafts), Sofia Sewing (#11F, $325k, 7% of drafts), Mari Humberg (#13F, $288k, 7% of drafts), Catherine Parenteau (#15F, $259k, 7% of drafts), Sahra Dennehy (#20F, $205k, 7% of drafts), Dylan Frazier (#23M, $143k, 7% of drafts), Yuta Funemizu (#26M, $130k, 7% of drafts), Tyra Hurricane Black (#10F, $353k, 3% of drafts), Danni-Elle Townsend (#12F, $306k, 3% of drafts), Vivian Glozman (#14F, $262k, 3% of drafts), Bobbi Oshiro (#16F, $243k, 3% of drafts), Lacy Schneemann (#17F, $224k, 3% of drafts), Chao Yi Wang (#18F, $216k, 3% of drafts), Vivienne David (#19F, $207k, 3% of drafts), Etta Tuionetoa (#21F, $192k, 3% of drafts), Ewa Radzikowska (#22F, $183k, 3% of drafts), Bruno Faletto (#21M, $151k, 3% of drafts), Hunter Johnson (#22M, $147k, 3% of drafts), Brooke Buckner (#29F, $145k, 3% of drafts), Vanshik Kapadia (#24M, $142k, 3% of drafts). All: Patrick Kawka #56M, Tyson McGuffin #50M, Donald Young #44M, Callie Smith #59F, Rafa Hewett #52M, Tristan Dussault #55M, Liz Truluck #48F, Genie Erokhina #56F, Christine Maddox #58F, Mohaned Alhouni #58M, Hien Truong #37M, Zoey Weil #50F, Cailyn Campbell #60F, Connor Mogle #42M, Juan Benitez #49M, Yufei Long #45F, Shelby Bates #47F, Eugenia Carolina Lopez Ascarate #55F, Alex Emery #46M, Etienne Blaszkewycz #48M, Martin Emmrich #53M, Milan Rane #40F, George Wall #36M, Estee Widdershoven #42F, Harsh Mehta #45M, Isabella Dunlap #38F, Pablo Tellez #38M, Joseph Wild #41M, Jessie Irvine #37F, Anderson Scarpa #39M, Luca Mack #40M, Augustus Ge #30M, Ting Chieh Wei #33F, Thomas Yu #34M, Max Freeman #43M, Tama Shimabukuro #47M, Hoang Nam Ly #59M, Oscar Serra #33M, Maggie Brascia #41F, Lucy Kovalova #46F, Marianna Petrei #43F, Allie Reichert #51F, CJ Klinger #28M, Roscoe Bellamy #29M, Carlota Trevino #34F, Daria Walczak #44F, Oliver Frank #54M, Grayson Goldin #60M, Layne Sleeth #27F, Allyce Jones #25F, Yana Newell #28F, Dekel Bar #27M, John Lucian Goins #35M, Seone Mendez #54F, Matthew Barlow #57M, Jillian Braverman #26F, Mary Brascia #52F, Genie Bouchard #57F, Meghan Dizon #23F, Leigh Waters #24F, Alix Truong #30F, Jaume Martinez Vich #31M, Samantha Parker #35F, Sofia Sewing #11F, Mari Humberg #13F, Catherine Parenteau #15F, Sahra Dennehy #20F, Dylan Frazier #23M, Yuta Funemizu #26M, Judit Castillo #32F, Roos Van Reek #36F, Kaitlyn Christian #39F, Kiora Kunimoto #49F, Tyra Hurricane Black #10F, Danni-Elle Townsend #12F, Vivian Glozman #14F, Bobbi Oshiro #16F, Lacy Schneemann #17F, Chao Yi Wang #18F, Vivienne David #19F, Etta Tuionetoa #21F, Ewa Radzikowska #22F, Bruno Faletto #21M, Hunter Johnson #22M, Brooke Buckner #29F, Vanshik Kapadia #24M, Lea Jansen #31F, Christopher Haworth #32M, Katerina Stewart #53F, Zane Ford #51M.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $725k | 1.0 | 0% | 1.1 | 66.7% | 28.5% | no |
| Anna Bright (#2F) | $520k | 1.0 | 0% | 3.5 | 57.6% | 8.8% | no |
| Parris Todd (#3F) | $431k | 1.4 | 0% | 12.5 | 56.4% | 7.6% | no |
| Jorja Johnson (#4F) | $421k | 1.4 | 0% | 13.0 | 55.8% | 6.6% | no |
| Kate Fahey (#5F) | $404k | 1.7 | 0% | 13.3 | 55.6% | 6.5% | no |
| Jade Kawamoto (#6F) | $397k | 1.7 | 0% | 11.7 | 54.5% | 5.4% | no |
| Ben Johns (#1M) | $420k | 2.7 | 0% | 13.2 | 55.6% | 7.5% | no |
| JW Johnson (#2M) | $387k | 1.4 | 0% | 11.3 | 55.9% | 6.3% | no |
| Hayden Patriquin (#3M) | $380k | 1.4 | 0% | 13.7 | 55.3% | 6.7% | no |
| Gabriel Tardio (#4M) | $371k | 1.6 | 0% | 11.8 | 56.3% | 6.8% | no |
| Christian Alshon (#5M) | $364k | 1.7 | 0% | 9.6 | 55.6% | 6.1% | no |
| Andrei Daescu (#6M) | $304k | 2.2 | 0% | 11.9 | 51.9% | 3.8% | no |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 66 | 52 | 53 | 49 | 52 | 48 | 49 | 52 | 47 | 49 | 48 | 49 | 45 | 47 | 46 | 47 | 51 | 49 | 48 | 53 |
| title% | 28 | 5 | 7 | 3 | 5 | 3 | 4 | 5 | 2 | 4 | 2 | 3 | 3 | 2 | 2 | 3 | 5 | 4 | 3 | 6 |

## linear draft, owner noise 0% (1 draft(s) x 200 seasons, 2 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender at the $30k floor; teams took 25.0 floor players per draft, leaving 25 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Rachel Rohrabacher (#7F, $380k, 100% of drafts), Tina Pisnik (#8F, $376k, 100% of drafts), Jackie Kawamoto (#9F, $357k, 100% of drafts), Tyra Hurricane Black (#10F, $353k, 100% of drafts), Sofia Sewing (#11F, $325k, 100% of drafts), Danni-Elle Townsend (#12F, $306k, 100% of drafts), Mari Humberg (#13F, $288k, 100% of drafts), Alix Truong (#30F, $142k, 100% of drafts). All: Rachel Rohrabacher #7F, Tina Pisnik #8F, Jackie Kawamoto #9F, Tyra Hurricane Black #10F, Sofia Sewing #11F, Danni-Elle Townsend #12F, Mari Humberg #13F, Alix Truong #30F, Judit Castillo #32F, Roos Van Reek #36F, Luca Mack #40M, Joseph Wild #41M, Connor Mogle #42M, Harsh Mehta #45M, Alex Emery #46M, Juan Benitez #49M, Tyson McGuffin #50M, Rafa Hewett #52M, Martin Emmrich #53M, Oliver Frank #54M, Tristan Dussault #55M, Patrick Kawka #56M, Mohaned Alhouni #58M, Cailyn Campbell #60F, Hoang Nam Ly #59M.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $725k | 1.0 | 0% | 1.0 | 71.5% | 35.0% | no |
| Anna Bright (#2F) | $520k | 1.0 | 0% | 2.0 | 63.7% | 13.5% | no |
| Parris Todd (#3F) | $431k | 1.0 | 0% | 14.0 | 59.6% | 6.5% | no |
| Jorja Johnson (#4F) | $421k | 2.0 | 0% | 9.0 | 57.6% | 4.0% | no |
| Kate Fahey (#5F) | $404k | 3.0 | 0% | 8.0 | 63.7% | 10.0% | no |
| Jade Kawamoto (#6F) | $397k | 3.0 | 0% | 4.0 | 61.0% | 7.5% | no |
| Ben Johns (#1M) | $420k | 6.0 | 0% | 3.0 | 59.8% | 7.0% | no |
| JW Johnson (#2M) | $387k | 3.0 | 0% | 14.0 | 59.6% | 6.5% | no |
| Hayden Patriquin (#3M) | $380k | 6.0 | 0% | 5.0 | 59.8% | 9.5% | no |
| Gabriel Tardio (#4M) | $371k | 6.0 | 0% | 6.0 | 57.5% | 2.0% | no |
| Christian Alshon (#5M) | $364k | 6.0 | 0% | 7.0 | 56.9% | 3.0% | no |
| Andrei Daescu (#6M) | $304k | 6.0 | 0% | 10.0 | 49.5% | 1.0% | no |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 71 | 64 | 60 | 61 | 60 | 57 | 57 | 64 | 58 | 50 | 51 | 49 | 44 | 60 | 37 | 36 | 33 | 32 | 30 | 29 |
| title% | 35 | 14 | 7 | 8 | 10 | 2 | 3 | 10 | 4 | 1 | 1 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |

## linear draft, owner noise 10% (30 draft(s) x 200 seasons, 47 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender at the $30k floor; teams took 26.9 floor players per draft, leaving 83 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Danni-Elle Townsend (#12F, $306k, 47% of drafts), Tyra Hurricane Black (#10F, $353k, 43% of drafts), Sofia Sewing (#11F, $325k, 40% of drafts), Jackie Kawamoto (#9F, $357k, 37% of drafts), Mari Humberg (#13F, $288k, 37% of drafts), Tina Pisnik (#8F, $376k, 23% of drafts), Rachel Rohrabacher (#7F, $380k, 17% of drafts), Vivian Glozman (#14F, $262k, 17% of drafts), Chao Yi Wang (#18F, $216k, 17% of drafts), Meghan Dizon (#23F, $177k, 17% of drafts), Ewa Radzikowska (#22F, $183k, 13% of drafts), Allyce Jones (#25F, $163k, 13% of drafts), Catherine Parenteau (#15F, $259k, 10% of drafts), Bobbi Oshiro (#16F, $243k, 10% of drafts), Lacy Schneemann (#17F, $224k, 10% of drafts), Alix Truong (#30F, $142k, 10% of drafts), Dekel Bar (#27M, $123k, 10% of drafts), Jillian Braverman (#26F, $158k, 7% of drafts), Layne Sleeth (#27F, $156k, 7% of drafts), Brooke Buckner (#29F, $145k, 7% of drafts), CJ Klinger (#28M, $123k, 7% of drafts), Kate Fahey (#5F, $404k, 3% of drafts), Vivienne David (#19F, $207k, 3% of drafts), Sahra Dennehy (#20F, $205k, 3% of drafts), Etta Tuionetoa (#21F, $192k, 3% of drafts), Leigh Waters (#24F, $165k, 3% of drafts), Yana Newell (#28F, $153k, 3% of drafts), Vanshik Kapadia (#24M, $142k, 3% of drafts), Yuta Funemizu (#26M, $130k, 3% of drafts), Roscoe Bellamy (#29M, $119k, 3% of drafts), Augustus Ge (#30M, $114k, 3% of drafts). All: Tyson McGuffin #50M, Tristan Dussault #55M, Patrick Kawka #56M, Callie Smith #59F, Donald Young #44M, Rafa Hewett #52M, Martin Emmrich #53M, Mohaned Alhouni #58M, Cailyn Campbell #60F, Hien Truong #37M, Liz Truluck #48F, Harsh Mehta #45M, Juan Benitez #49M, Pablo Tellez #38M, Anderson Scarpa #39M, Connor Mogle #42M, Alex Emery #46M, Etienne Blaszkewycz #48M, Joseph Wild #41M, Zoey Weil #50F, Yufei Long #45F, Luca Mack #40M, Max Freeman #43M, Genie Erokhina #56F, Danni-Elle Townsend #12F, Tyra Hurricane Black #10F, Shelby Bates #47F, Grayson Goldin #60M, Sofia Sewing #11F, Hoang Nam Ly #59M, Jackie Kawamoto #9F, Mari Humberg #13F, Thomas Yu #34M, Eugenia Carolina Lopez Ascarate #55F, Isabella Dunlap #38F, George Wall #36M, Tama Shimabukuro #47M, Marianna Petrei #43F, Milan Rane #40F, Christine Maddox #58F, Tina Pisnik #8F, John Lucian Goins #35M, Maggie Brascia #41F, Rachel Rohrabacher #7F, Vivian Glozman #14F, Chao Yi Wang #18F, Meghan Dizon #23F, Oscar Serra #33M, Daria Walczak #44F, Lucy Kovalova #46F, Allie Reichert #51F, Ewa Radzikowska #22F, Allyce Jones #25F, Roos Van Reek #36F, Jessie Irvine #37F, Estee Widdershoven #42F, Matthew Barlow #57M, Catherine Parenteau #15F, Bobbi Oshiro #16F, Lacy Schneemann #17F, Alix Truong #30F, Lea Jansen #31F, Dekel Bar #27M, Genie Bouchard #57F, Jillian Braverman #26F, Layne Sleeth #27F, Brooke Buckner #29F, CJ Klinger #28M, Carlota Trevino #34F, Samantha Parker #35F, Mary Brascia #52F, Oliver Frank #54M, Kate Fahey #5F, Vivienne David #19F, Sahra Dennehy #20F, Etta Tuionetoa #21F, Leigh Waters #24F, Yana Newell #28F, Vanshik Kapadia #24M, Yuta Funemizu #26M, Roscoe Bellamy #29M, Augustus Ge #30M, Jaume Martinez Vich #31M.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $725k | 1.0 | 0% | 1.0 | 69.8% | 32.6% | no |
| Anna Bright (#2F) | $520k | 1.0 | 0% | 2.8 | 61.7% | 13.3% | no |
| Parris Todd (#3F) | $431k | 1.1 | 0% | 13.5 | 56.9% | 6.9% | no |
| Jorja Johnson (#4F) | $421k | 1.3 | 0% | 11.9 | 56.4% | 7.2% | no |
| Kate Fahey (#5F) | $404k | 1.9 | 3% | 11.9 | 56.3% | 7.2% | no |
| Jade Kawamoto (#6F) | $397k | 2.0 | 0% | 11.5 | 56.4% | 6.2% | no |
| Ben Johns (#1M) | $420k | 4.2 | 0% | 6.0 | 57.4% | 6.4% | no |
| JW Johnson (#2M) | $387k | 2.8 | 0% | 6.6 | 56.7% | 5.7% | no |
| Hayden Patriquin (#3M) | $380k | 2.8 | 0% | 10.5 | 57.0% | 6.4% | no |
| Gabriel Tardio (#4M) | $371k | 3.0 | 0% | 10.9 | 56.1% | 5.3% | no |
| Christian Alshon (#5M) | $364k | 3.6 | 0% | 9.7 | 54.7% | 4.2% | no |
| Andrei Daescu (#6M) | $304k | 4.3 | 0% | 10.4 | 50.3% | 2.8% | no |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 70 | 59 | 57 | 57 | 53 | 54 | 52 | 52 | 51 | 49 | 49 | 47 | 47 | 43 | 44 | 41 | 45 | 44 | 46 | 40 |
| title% | 33 | 10 | 6 | 7 | 4 | 5 | 3 | 3 | 3 | 2 | 3 | 2 | 2 | 2 | 2 | 2 | 3 | 3 | 3 | 2 |

## linear draft, owner noise 25% (30 draft(s) x 200 seasons, 47 s)

**Undrafted priced players (info, not a test):** the board is the priced 60+60 plus 60 free agents per gender at the $30k floor; teams took 25.8 floor players per draft, leaving 87 distinct priced players unpicked in at least one draft. Inside the top 30 of their gender: Augustus Ge (#30M, $114k, 30% of drafts), Danni-Elle Townsend (#12F, $306k, 27% of drafts), Sofia Sewing (#11F, $325k, 23% of drafts), Vivian Glozman (#14F, $262k, 23% of drafts), Mari Humberg (#13F, $288k, 20% of drafts), Catherine Parenteau (#15F, $259k, 20% of drafts), Tyra Hurricane Black (#10F, $353k, 17% of drafts), Ewa Radzikowska (#22F, $183k, 17% of drafts), Meghan Dizon (#23F, $177k, 17% of drafts), Allyce Jones (#25F, $163k, 17% of drafts), Bobbi Oshiro (#16F, $243k, 13% of drafts), Chao Yi Wang (#18F, $216k, 13% of drafts), Vivienne David (#19F, $207k, 13% of drafts), Sahra Dennehy (#20F, $205k, 13% of drafts), Etta Tuionetoa (#21F, $192k, 13% of drafts), Yana Newell (#28F, $153k, 13% of drafts), Alix Truong (#30F, $142k, 13% of drafts), CJ Klinger (#28M, $123k, 13% of drafts), Roscoe Bellamy (#29M, $119k, 13% of drafts), Lacy Schneemann (#17F, $224k, 10% of drafts), Jillian Braverman (#26F, $158k, 10% of drafts), Layne Sleeth (#27F, $156k, 10% of drafts), Dekel Bar (#27M, $123k, 10% of drafts), Tina Pisnik (#8F, $376k, 7% of drafts), Leigh Waters (#24F, $165k, 7% of drafts), Brooke Buckner (#29F, $145k, 7% of drafts), Dylan Frazier (#23M, $143k, 7% of drafts), Vanshik Kapadia (#24M, $142k, 7% of drafts), Armaan Bhatia (#25M, $140k, 7% of drafts), Yuta Funemizu (#26M, $130k, 7% of drafts), Kate Fahey (#5F, $404k, 3% of drafts), Jackie Kawamoto (#9F, $357k, 3% of drafts), Noe Khlif (#18M, $184k, 3% of drafts). All: Patrick Kawka #56M, Rafa Hewett #52M, Callie Smith #59F, Pablo Tellez #38M, Donald Young #44M, Tyson McGuffin #50M, Tristan Dussault #55M, Liz Truluck #48F, Juan Benitez #49M, Luca Mack #40M, Martin Emmrich #53M, Thomas Yu #34M, Hien Truong #37M, Alex Emery #46M, Cailyn Campbell #60F, Joseph Wild #41M, Connor Mogle #42M, Etienne Blaszkewycz #48M, Mohaned Alhouni #58M, George Wall #36M, Zoey Weil #50F, Christine Maddox #58F, Hoang Nam Ly #59M, Daria Walczak #44F, Anderson Scarpa #39M, Shelby Bates #47F, Genie Erokhina #56F, Estee Widdershoven #42F, Harsh Mehta #45M, Ting Chieh Wei #33F, Oscar Serra #33M, Max Freeman #43M, Yufei Long #45F, Eugenia Carolina Lopez Ascarate #55F, Tama Shimabukuro #47M, Augustus Ge #30M, Isabella Dunlap #38F, Milan Rane #40F, Allie Reichert #51F, Danni-Elle Townsend #12F, Sofia Sewing #11F, Vivian Glozman #14F, Samantha Parker #35F, Lucy Kovalova #46F, Grayson Goldin #60M, Mari Humberg #13F, Catherine Parenteau #15F, Roos Van Reek #36F, Maggie Brascia #41F, Tyra Hurricane Black #10F, Ewa Radzikowska #22F, Meghan Dizon #23F, Allyce Jones #25F, Bobbi Oshiro #16F, Chao Yi Wang #18F, Vivienne David #19F, Sahra Dennehy #20F, Etta Tuionetoa #21F, Yana Newell #28F, Alix Truong #30F, CJ Klinger #28M, Roscoe Bellamy #29M, Jessie Irvine #37F, Marianna Petrei #43F, Lacy Schneemann #17F, Jillian Braverman #26F, Layne Sleeth #27F, Dekel Bar #27M, Carlota Trevino #34F, Tina Pisnik #8F, Leigh Waters #24F, Brooke Buckner #29F, Dylan Frazier #23M, Vanshik Kapadia #24M, Armaan Bhatia #25M, Yuta Funemizu #26M, John Lucian Goins #35M, Genie Bouchard #57F, Oliver Frank #54M, Matthew Barlow #57M, Kate Fahey #5F, Jackie Kawamoto #9F, Noe Khlif #18M, Judit Castillo #32F, Kiora Kunimoto #49F, Katerina Stewart #53F, Zane Ford #51M.

Stars (top 6 per gender by phi): where they went and how their team did.

| player | price | mean round | undrafted | team slot (mean) | team win% | team title% | stuck? |
|---|---|---|---|---|---|---|---|
| Anna Leigh Waters (#1F) | $725k | 1.0 | 0% | 1.2 | 67.9% | 31.1% | no |
| Anna Bright (#2F) | $520k | 1.0 | 0% | 4.9 | 58.6% | 9.2% | no |
| Parris Todd (#3F) | $431k | 1.2 | 0% | 11.4 | 55.6% | 7.1% | no |
| Jorja Johnson (#4F) | $421k | 1.3 | 0% | 10.6 | 56.1% | 5.8% | no |
| Kate Fahey (#5F) | $404k | 1.5 | 3% | 10.7 | 55.5% | 6.4% | no |
| Jade Kawamoto (#6F) | $397k | 1.7 | 0% | 11.0 | 52.8% | 5.2% | no |
| Ben Johns (#1M) | $420k | 3.1 | 0% | 9.2 | 55.3% | 5.8% | no |
| JW Johnson (#2M) | $387k | 1.5 | 0% | 9.0 | 57.5% | 7.2% | no |
| Hayden Patriquin (#3M) | $380k | 1.8 | 0% | 8.1 | 55.9% | 6.0% | no |
| Gabriel Tardio (#4M) | $371k | 2.3 | 0% | 8.6 | 54.5% | 4.6% | no |
| Christian Alshon (#5M) | $364k | 2.4 | 0% | 9.1 | 54.6% | 5.0% | no |
| Andrei Daescu (#6M) | $304k | 2.9 | 0% | 10.7 | 53.5% | 5.7% | no |

By draft slot (mean over drafts):

| slot | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| win% | 66 | 57 | 56 | 52 | 54 | 53 | 53 | 54 | 50 | 50 | 46 | 48 | 48 | 47 | 48 | 46 | 44 | 46 | 41 | 41 |
| title% | 28 | 9 | 7 | 4 | 6 | 5 | 5 | 5 | 3 | 3 | 2 | 2 | 3 | 2 | 3 | 3 | 2 | 3 | 2 | 2 |
