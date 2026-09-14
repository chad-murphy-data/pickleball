# Market prices -- what the backward induction settles on

Every owner plans whole $1M rosters (exhaustive 3M+3W search), rosters are handed out in a random order, priced players on above-average rosters are marked up and on below-average rosters marked down, unsold priced players fall, prices clipped to [$30k floor, $850k first-buy maximum]; prices are the average over the last third of the rounds. Board `mlp2026`, 20 teams, true values, identical owners unless `noise` is set. The hand-written read is in `auction.md` ("The market limit"). Built by `market_eq.py`.

## Runs

| rule | seed | c | rounds | noise | roster-value sd at the end (free rosters) | rationed (at $850k with excess demand) | priced pool total | priced players at the floor | Waters' team win% / title% | second-best team | runner-up title | teams >= 10% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| demand | 1 | 8 | 150 | 0 | 0.124 | Anna Leigh Waters | $18.38M (list $20.00M) | 38 | 62.0% / 16-24% | 56.8% | 8-10% | 1-2 |
| equalize | 1 | 12 | 200 | 0 | 0.008 | Anna Leigh Waters | $19.98M (list $20.00M) | 27 | 63.4% / 32-37% | 50.6% | 6-8% | 1-1 |
| equalize | 1 | 8 | 200 | 0 | 0.015 | Anna Leigh Waters | $19.93M (list $20.00M) | 27 | 63.0% / 32-38% | 50.6% | 6-8% | 1-1 |
| equalize | 3 | 4 | 200 | 0 | 0.020 | Anna Leigh Waters | $19.84M (list $20.00M) | 28 | 61.7% / 24-31% | 51.1% | 6-7% | 1-1 |

## Market price vs list by tier (list rank within gender)

| run | women #1-5 | #6-15 | #16-30 | #31-60 | at floor | men #1-5 | #6-15 | #16-30 | #31-60 | at floor |
|---|---|---|---|---|---|---|---|---|---|---|
| demand seed 1 c 8 | 103% ($576k vs $560k) | 100% ($357k vs $357k) | 100% ($176k vs $176k) | 57% ($44k vs $77k) | 17 | 100% ($427k vs $427k) | 100% ($253k vs $253k) | 100% ($140k vs $140k) | 62% ($40k vs $64k) | 21 |
| equalize seed 1 c 12 | 117% ($653k vs $560k) | 117% ($418k vs $357k) | 106% ($186k vs $176k) | 55% ($42k vs $77k) | 11 | 113% ($484k vs $427k) | 113% ($285k vs $253k) | 98% ($137k vs $140k) | 61% ($39k vs $64k) | 16 |
| equalize seed 1 c 8 | 117% ($654k vs $560k) | 117% ($419k vs $357k) | 105% ($186k vs $176k) | 53% ($41k vs $77k) | 12 | 114% ($486k vs $427k) | 112% ($283k vs $253k) | 97% ($135k vs $140k) | 61% ($39k vs $64k) | 15 |
| equalize seed 3 c 4 | 117% ($655k vs $560k) | 116% ($415k vs $357k) | 105% ($186k vs $176k) | 53% ($41k vs $77k) | 12 | 112% ($480k vs $427k) | 112% ($283k vs $253k) | 97% ($135k vs $140k) | 60% ($39k vs $64k) | 16 |

## The top 24 by list price (mean over the 3 main runs, range in brackets)

Two benchmarks. Indifference (best) = the price at which the best roster holding the player equals the best roster an owner can build without the rationed players -- a roster only one owner gets. Indifference (average) = the price at which the best roster holding the player equals the AVERAGE roster the leagues below hand out (0.520 tie probability vs the reference team) -- the benchmark the market actually equalises to. with@market / without = the best roster with the player at market prices and the best without the rationed players; with@cap = the best roster holding the player if he cost $850k. The market column is a time average of oscillating prices, and a roster that is affordable in most rounds is not always affordable at the averaged prices, so the two differ by a few percent either way (market above for the top dozen, below for the next); together they bracket the star's price.

| player | list | market | market / list | indifference (average) | (average) / list | indifference (best) | with@market | without | with@cap |
|---|---|---|---|---|---|---|---|---|---|
| Anna Leigh Waters | $769k | $850k | 111% | $850k | 111% | $850k | 0.660 | 0.528 | 0.660 |
| Anna Bright | $613k | $760k [756-763] | 124% | $697k [696-698] | 114% | $680k | 0.492 | 0.529 | 0.386 |
| Parris Todd | $489k | $576k [574-578] | 118% | $547k [544-549] | 112% | $538k | 0.505 | 0.529 | 0.296 |
| Jorja Johnson | $475k | $559k [552-563] | 117% | $538k [532-542] | 113% | $513k | 0.507 | 0.529 | 0.291 |
| Ben Johns | $474k | $545k [539-551] | 115% | $501k [499-504] | 106% | $490k | 0.496 | 0.528 | 0.277 |
| Kate Fahey | $453k | $525k [519-529] | 116% | $501k [500-502] | 111% | $489k | 0.508 | 0.529 | 0.273 |
| Jade Kawamoto | $444k | $521k [521-522] | 118% | $489k [487-491] | 110% | $477k | 0.502 | 0.529 | 0.272 |
| JW Johnson | $430k | $487k [483-493] | 113% | $458k [447-464] | 106% | $442k | 0.501 | 0.528 | 0.246 |
| Hayden Patriquin | $421k | $472k [466-478] | 112% | $442k [439-446] | 105% | $434k | 0.506 | 0.528 | 0.246 |
| Rachel Rohrabacher | $421k | $484k [482-485] | 115% | $472k [464-479] | 112% | $460k | 0.511 | 0.529 | 0.259 |
| Tina Pisnik | $416k | $488k [487-490] | 117% | $462k [461-464] | 111% | $458k | 0.501 | 0.529 | 0.255 |
| Gabriel Tardio | $409k | $458k [451-466] | 112% | $434k [430-438] | 106% | $432k | 0.504 | 0.528 | 0.233 |
| Christian Alshon | $400k | $454k [453-456] | 114% | $429k [425-431] | 107% | $404k | 0.500 | 0.528 | 0.237 |
| Jackie Kawamoto | $391k | $451k [449-455] | 116% | $458k [453-461] | 117% | $425k | 0.517 | 0.529 | 0.243 |
| Tyra Hurricane Black | $386k | $447k [442-453] | 116% | $444k [440-446] | 115% | $424k | 0.518 | 0.529 | 0.238 |
| Sofia Sewing | $350k | $409k [408-410] | 117% | $402k [400-403] | 115% | $391k | 0.519 | 0.529 | 0.224 |
| Danni-Elle Townsend | $325k | $390k [386-394] | 120% | $379k [376-383] | 117% | $365k | 0.513 | 0.529 | 0.208 |
| Andrei Daescu | $323k | $368k [365-369] | 114% | $356k [351-360] | 110% | $341k | 0.512 | 0.528 | 0.192 |
| Federico Staksrud | $315k | $356k [353-360] | 113% | $344k [340-350] | 109% | $335k | 0.516 | 0.528 | 0.194 |
| Eric Oncins | $307k | $350k [347-352] | 114% | $335k [328-341] | 109% | $322k | 0.509 | 0.528 | 0.193 |
| Mari Humberg | $303k | $362k [357-368] | 120% | $353k [349-359] | 117% | $348k | 0.517 | 0.529 | 0.198 |
| Jay Devilliers | $288k | $326k [325-326] | 113% | $319k [313-326] | 111% | $312k | 0.516 | 0.528 | 0.186 |
| Vivian Glozman | $271k | $314k [313-317] | 116% | $327k [326-330] | 121% | $309k | 0.524 | 0.529 | 0.193 |
| Catherine Parenteau | $267k | $307k [306-308] | 115% | $321k [317-328] | 121% | $304k | 0.526 | 0.529 | 0.187 |

## The Waters roster and the best roster without her, at market prices

- seed 1, c 12: best overall 0.665 = Martin Emmrich $30k, Jonathan Truong $30k, Gabriel Joseph $30k, Anna Leigh Waters $850k, Genie Erokhina $30k, Lina Padegimaite $30k; best without the rationed 0.528 = Nicolas Acevedo $215k, Jack Sock $252k, Gabriel Joseph $30k, Vivienne David $229k, Chao Yi Wang $242k, Lina Padegimaite $30k.
- seed 1, c 8: best overall 0.665 = Martin Emmrich $30k, Jonathan Truong $30k, Gabriel Joseph $30k, Anna Leigh Waters $850k, Genie Erokhina $30k, Lina Padegimaite $30k; best without the rationed 0.528 = Nicolas Acevedo $216k, Jack Sock $251k, Gabriel Joseph $30k, Vivienne David $232k, Chao Yi Wang $238k, Lina Padegimaite $30k.
- seed 3, c 4: best overall 0.648 = Martin Emmrich $30k, Jonathan Truong $30k, Gabriel Joseph $30k, Anna Leigh Waters $850k, Alexa Schull $30k, Lina Padegimaite $30k; best without the rationed 0.529 = Phuc Huynh $225k, Robert Slutsky $193k, Gabriel Joseph $30k, Bobbi Oshiro $293k, Vivienne David $227k, Lina Padegimaite $30k.

## Leagues at market prices (five random allocation orders per run, 200 seasons each)

- demand seed 1 c 8 rep 0: win% 62.0 / 56.8 / ... / 18.1, Waters' team 62.0% / title 24%, runner-up 8%, teams >= 10%: 1, spend $294k-$1000k.
- demand seed 1 c 8 rep 1: win% 62.0 / 56.8 / ... / 18.1, Waters' team 62.0% / title 17%, runner-up 8%, teams >= 10%: 1, spend $294k-$1000k.
- demand seed 1 c 8 rep 2: win% 62.0 / 56.8 / ... / 18.1, Waters' team 62.0% / title 20%, runner-up 10%, teams >= 10%: 2, spend $294k-$1000k.
- demand seed 1 c 8 rep 3: win% 62.0 / 56.8 / ... / 18.1, Waters' team 62.0% / title 16%, runner-up 10%, teams >= 10%: 2, spend $294k-$1000k.
- demand seed 1 c 8 rep 4: win% 62.0 / 56.8 / ... / 18.1, Waters' team 62.0% / title 24%, runner-up 8%, teams >= 10%: 1, spend $294k-$1000k.
- equalize seed 1 c 12 rep 0: win% 63.4 / 50.6 / ... / 45.6, Waters' team 63.4% / title 33%, runner-up 6%, teams >= 10%: 1, spend $939k-$1000k.
- equalize seed 1 c 12 rep 1: win% 63.4 / 50.6 / ... / 45.6, Waters' team 63.4% / title 34%, runner-up 6%, teams >= 10%: 1, spend $939k-$1000k.
- equalize seed 1 c 12 rep 2: win% 63.4 / 50.6 / ... / 45.6, Waters' team 63.4% / title 32%, runner-up 8%, teams >= 10%: 1, spend $939k-$1000k.
- equalize seed 1 c 12 rep 3: win% 63.4 / 50.6 / ... / 45.6, Waters' team 63.4% / title 37%, runner-up 6%, teams >= 10%: 1, spend $939k-$1000k.
- equalize seed 1 c 12 rep 4: win% 63.4 / 50.6 / ... / 45.6, Waters' team 63.4% / title 37%, runner-up 6%, teams >= 10%: 1, spend $939k-$1000k.
- equalize seed 1 c 8 rep 0: win% 63.0 / 50.6 / ... / 46.8, Waters' team 63.0% / title 34%, runner-up 6%, teams >= 10%: 1, spend $964k-$1000k.
- equalize seed 1 c 8 rep 1: win% 63.0 / 50.6 / ... / 46.8, Waters' team 63.0% / title 32%, runner-up 6%, teams >= 10%: 1, spend $964k-$1000k.
- equalize seed 1 c 8 rep 2: win% 63.0 / 50.6 / ... / 46.8, Waters' team 63.0% / title 32%, runner-up 7%, teams >= 10%: 1, spend $964k-$1000k.
- equalize seed 1 c 8 rep 3: win% 63.0 / 50.6 / ... / 46.8, Waters' team 63.0% / title 33%, runner-up 8%, teams >= 10%: 1, spend $964k-$1000k.
- equalize seed 1 c 8 rep 4: win% 63.0 / 50.6 / ... / 46.8, Waters' team 63.0% / title 38%, runner-up 6%, teams >= 10%: 1, spend $964k-$1000k.
- equalize seed 3 c 4 rep 0: win% 61.7 / 51.1 / ... / 42.4, Waters' team 61.7% / title 31%, runner-up 6%, teams >= 10%: 1, spend $877k-$1000k.
- equalize seed 3 c 4 rep 1: win% 61.7 / 51.1 / ... / 42.4, Waters' team 61.7% / title 24%, runner-up 7%, teams >= 10%: 1, spend $877k-$1000k.
- equalize seed 3 c 4 rep 2: win% 61.7 / 51.1 / ... / 42.4, Waters' team 61.7% / title 24%, runner-up 7%, teams >= 10%: 1, spend $877k-$1000k.
- equalize seed 3 c 4 rep 3: win% 61.7 / 51.1 / ... / 42.4, Waters' team 61.7% / title 27%, runner-up 7%, teams >= 10%: 1, spend $877k-$1000k.
- equalize seed 3 c 4 rep 4: win% 61.7 / 51.1 / ... / 42.4, Waters' team 61.7% / title 28%, runner-up 6%, teams >= 10%: 1, spend $877k-$1000k.

## Convergence (every 10th round of each run)

- demand seed 1 c 8: r0 free 0.513+-0.108 capped [] sum $19.91M unsold 29 W $850k B $613k J $474k; r30 free 0.528+-0.121 capped [0.691] sum $18.41M unsold 8 W $850k B $613k J $474k; r60 free 0.527+-0.124 capped [0.706] sum $18.38M unsold 8 W $850k B $613k J $474k; r90 free 0.527+-0.124 capped [0.707] sum $18.38M unsold 8 W $850k B $613k J $474k; r120 free 0.527+-0.124 capped [0.707] sum $18.38M unsold 8 W $850k B $613k J $474k; r149 free 0.527+-0.124 capped [0.698] sum $18.38M unsold 8 W $850k B $613k J $474k.
- equalize seed 1 c 12: r0 free 0.513+-0.108 capped [] sum $20.15M unsold 29 W $850k B $637k J $478k; r30 free 0.518+-0.019 capped [0.722] sum $19.92M unsold 13 W $850k B $733k J $548k; r60 free 0.517+-0.016 capped [0.715] sum $19.94M unsold 12 W $850k B $762k J $546k; r90 free 0.518+-0.014 capped [0.705] sum $19.96M unsold 11 W $850k B $751k J $526k; r120 free 0.515+-0.023 capped [0.723] sum $19.98M unsold 12 W $850k B $759k J $550k; r150 free 0.510+-0.044 capped [0.713] sum $19.99M unsold 11 W $850k B $760k J $555k; r180 free 0.518+-0.009 capped [0.724] sum $19.98M unsold 13 W $850k B $749k J $557k.
- equalize seed 1 c 8: r0 free 0.513+-0.108 capped [] sum $20.05M unsold 29 W $850k B $629k J $477k; r30 free 0.518+-0.026 capped [0.722] sum $19.84M unsold 12 W $850k B $752k J $532k; r60 free 0.519+-0.015 capped [0.718] sum $19.87M unsold 12 W $850k B $763k J $544k; r90 free 0.518+-0.020 capped [0.711] sum $19.91M unsold 12 W $850k B $758k J $549k; r120 free 0.520+-0.012 capped [0.721] sum $19.92M unsold 13 W $850k B $764k J $558k; r150 free 0.508+-0.045 capped [0.731] sum $19.94M unsold 13 W $850k B $754k J $552k; r180 free 0.518+-0.013 capped [0.715] sum $19.92M unsold 11 W $850k B $766k J $536k.
- equalize seed 3 c 4: r0 free 0.513+-0.108 capped [] sum $19.93M unsold 29 W $817k B $621k J $475k; r30 free 0.520+-0.033 capped [0.713] sum $19.74M unsold 12 W $850k B $724k J $537k; r60 free 0.520+-0.023 capped [0.712] sum $19.81M unsold 11 W $850k B $757k J $531k; r90 free 0.520+-0.020 capped [0.718] sum $19.84M unsold 13 W $850k B $760k J $540k; r120 free 0.521+-0.016 capped [0.695] sum $19.85M unsold 9 W $850k B $760k J $535k; r150 free 0.518+-0.017 capped [0.714] sum $19.85M unsold 12 W $850k B $761k J $541k; r180 free 0.519+-0.014 capped [0.712] sum $19.83M unsold 12 W $850k B $761k J $550k.
