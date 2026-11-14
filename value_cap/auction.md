# Auction draft -- the same owners and personas, prices set by the room

Shipped tag list as the cheat sheet (Waters listed at $769k), `mlp2026` board, 20 teams, $1M cap, 3M+3W, 20 auctions x 200 seasons per cell, seed 1, every owner at 10% belief noise unless stated. The mechanism is in the docstring of `auction_sim.py`: nominate your snake pick, bid up to your indifference price, pay the second-highest ceiling + $5k, ceilings hard-capped at budget minus the cheapest legal completion ($850k for a first buy on this board). `expect` = the prices owners assume for the players they have not bought yet: the list, or the list inflated by money-left / value-left. Parity = 50% win, 5% title. Built by `auction_sim.py`.

## Headline: auction vs snake with twenty quants

| format | expect | owner noise | Waters paid | sold at sale # | her other five cost (at floor) | her team win% | title% | Bright paid | her team | Johns paid | his team | parity spread | favourite title | runner-up | teams >= 10% | effective contenders | mean spend | unspent per team | top-30 undrafted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| snake draft, quants | -- | noise 10% | $769k (list) | pick 1 | -- | 65.5% | 36.2% | $613k (list) | 51.2% | $474k (list) | 50.9% | 4.4 pts | 36.2% | 7.3% | 1.0 | 6.4 | $975k | $0k | none |
| auction, quants | list | noise 0% | $850k | 1.0 | $150k (5.0 of 5) | 68.2% | 37.0% | $622k | 58.1% | $506k | 53.2% | 7.3 pts | 37.0% | 14.8% | 2.4 | 5.3 | $1,000k | $0k | none |
| auction, quants | list | noise 10% | $850k | 1.0 | $150k (5.0 of 5) | 68.3% | 39.4% | $675k | 55.8% | $540k | 52.7% | 6.3 pts | 39.4% | 11.7% | 1.9 | 5.2 | $999k | $1k | none |
| auction, quants | inflated | noise 0% | $850k | 1.0 | $150k (5.0 of 5) | 67.7% | 33.4% | $610k | 56.9% | $507k | 53.4% | 8.0 pts | 33.4% | 16.0% | 3.0 | 5.8 | $999k | $1k | none |
| auction, quants | inflated | noise 10% | $850k | 1.0 | $150k (5.0 of 5) | 67.5% | 36.9% | $654k | 55.4% | $535k | 52.1% | 6.1 pts | 36.9% | 11.5% | 1.8 | 5.7 | $999k | $1k | none |

## Price discovery: what the room pays vs the list

Mean realized price / list price by pool-rank tier (priced players sold in at least one auction), quant-only auctions.

| expect | owner noise | women #1-5 | #6-15 | #16-30 | #31-60 | men #1-5 | #6-15 | #16-30 | #31-60 | bidders per sale | stranded per auction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| list | noise 0% | 105% | 120% | 105% | 51% | 106% | 126% | 106% | 54% | 15.1 | 0.0 |
| list | noise 10% | 110% | 119% | 96% | 56% | 115% | 121% | 93% | 58% | 14.8 | 0.0 |
| inflated | noise 0% | 101% | 111% | 108% | 64% | 103% | 130% | 99% | 62% | 14.4 | 0.0 |
| inflated | noise 10% | 111% | 112% | 90% | 67% | 115% | 122% | 96% | 63% | 13.9 | 0.0 |

- **list, noise 0%** -- biggest premiums: Armaan Bhatia $213k vs $130k (+65%), Will Howells $249k vs $154k (+62%), Dylan Frazier $209k vs $133k (+58%), Phuc Huynh $312k vs $210k (+49%), Connor Garnett $283k vs $202k (+40%), Bruno Faletto $197k vs $141k (+40%).
  Biggest discounts: Milan Rane $30k vs $83k (-63%), Roos Van Reek $35k vs $96k (-63%), Isabella Dunlap $33k vs $88k (-63%), Maggie Brascia $31k vs $81k (-62%), Estee Widdershoven $30k vs $80k (-62%), Marianna Petrei $30k vs $79k (-62%).

- **list, noise 10%** -- biggest premiums: Noe Khlif $263k vs $178k (+48%), Connor Garnett $267k vs $202k (+32%), Phuc Huynh $263k vs $210k (+25%), Tyra Hurricane Black $481k vs $386k (+25%), Jackie Kawamoto $486k vs $391k (+24%), Riley Newman $312k vs $251k (+24%).
  Biggest discounts: Milan Rane $32k vs $83k (-62%), Estee Widdershoven $31k vs $80k (-61%), Marianna Petrei $31k vs $79k (-60%), George Wall $33k vs $83k (-60%), Maggie Brascia $34k vs $81k (-58%), Yufei Long $33k vs $79k (-58%).

- **inflated, noise 0%** -- biggest premiums: Katerina Stewart $136k vs $65k (+111%), Phuc Huynh $331k vs $210k (+58%), Bruno Faletto $201k vs $141k (+43%), Zane Ford $75k vs $53k (+42%), Connor Garnett $284k vs $202k (+41%), Jackie Kawamoto $547k vs $391k (+40%).
  Biggest discounts: Isabella Dunlap $32k vs $88k (-64%), Milan Rane $31k vs $83k (-63%), Estee Widdershoven $30k vs $80k (-62%), Daria Walczak $30k vs $79k (-62%), Yufei Long $31k vs $79k (-61%), Roos Van Reek $38k vs $96k (-61%).

- **inflated, noise 10%** -- biggest premiums: Katerina Stewart $106k vs $65k (+64%), Kaitlyn Christian $127k vs $87k (+46%), Zane Ford $76k vs $53k (+43%), Seone Mendez $87k vs $64k (+36%), Will Howells $203k vs $154k (+32%), Phuc Huynh $275k vs $210k (+31%).
  Biggest discounts: Yufei Long $32k vs $79k (-59%), Isabella Dunlap $36k vs $88k (-59%), Roos Van Reek $40k vs $96k (-58%), Pablo Tellez $31k vs $71k (-57%), Liz Truluck $30k vs $68k (-56%), Maggie Brascia $36k vs $81k (-56%).

## Personas at auction

Same persona definitions and strengths as `personas.md`; k persona owners at random seats among quants.

| persona | strength | how many of 20 | expect | persona teams: win% | title% | spend | quant teams: win% | title% | parity spread | Waters paid | her team win% | Bright paid | Johns paid | favourite title | runner-up | teams >= 10% | effective contenders | top-30 undrafted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overvalues men | k = 0.5 | 1 | list | 49.9% | 3.2% | $998k | 50.0% | 5.1% | 6.2 pts | $850k | 68.1% | $668k | $544k | 37.0% | 11.9% | 2.0 | 5.6 | none |
| overvalues men | k = 0.5 | 1 | inflated | 50.6% | 4.2% | $999k | 50.0% | 5.0% | 6.2 pts | $850k | 68.0% | $661k | $539k | 38.1% | 11.8% | 1.6 | 5.5 | none |
| overvalues men | k = 0.5 | 5 | list | 49.0% | 2.9% | $1,000k | 50.3% | 5.7% | 6.4 pts | $850k | 68.4% | $672k | $564k | 38.8% | 12.3% | 2.3 | 5.3 | none |
| overvalues men | k = 0.5 | 5 | inflated | 50.7% | 4.5% | $999k | 49.8% | 5.2% | 6.3 pts | $850k | 68.0% | $655k | $557k | 38.5% | 11.5% | 1.8 | 5.4 | none |
| overvalues men | k = 0.5 | 20 | list | 50.0% | 5.0% | $999k | -- | -- | 6.9 pts | $847k | 68.4% | $621k | $595k | 37.2% | 14.5% | 2.5 | 5.4 | none |
| overvalues men | k = 0.5 | 20 | inflated | 50.0% | 5.0% | $999k | -- | -- | 6.5 pts | $849k | 68.2% | $615k | $577k | 38.0% | 14.0% | 2.0 | 5.3 | none |
| overvalues men | k = 1 | 1 | list | 49.7% | 3.5% | $1,000k | 50.0% | 5.1% | 6.6 pts | $850k | 69.6% | $675k | $538k | 41.8% | 12.1% | 1.7 | 4.8 | none |
| overvalues men | k = 1 | 1 | inflated | 49.5% | 4.0% | $997k | 50.0% | 5.1% | 6.4 pts | $850k | 67.3% | $657k | $538k | 36.8% | 13.4% | 2.2 | 5.5 | none |
| overvalues men | k = 1 | 5 | list | 46.9% | 2.5% | $1,000k | 51.0% | 5.8% | 6.4 pts | $850k | 67.7% | $666k | $553k | 35.9% | 14.5% | 2.4 | 5.6 | none |
| overvalues men | k = 1 | 5 | inflated | 48.1% | 2.7% | $999k | 50.6% | 5.8% | 6.1 pts | $850k | 67.2% | $659k | $560k | 37.9% | 11.8% | 2.0 | 5.5 | none |
| overvalues men | k = 1 | 20 | list | 50.0% | 5.0% | $999k | -- | -- | 8.9 pts | $779k | 75.2% | $642k | $647k | 50.2% | 13.4% | 2.1 | 3.5 | none |
| overvalues men | k = 1 | 20 | inflated | 50.0% | 5.0% | $999k | -- | -- | 7.9 pts | $759k | 74.0% | $614k | $626k | 49.5% | 11.8% | 1.9 | 3.6 | none |
| overvalues women | k = 0.5 | 1 | list | 48.7% | 2.7% | $1,000k | 50.1% | 5.1% | 6.4 pts | $850k | 69.1% | $673k | $534k | 40.5% | 11.0% | 1.6 | 5.0 | none |
| overvalues women | k = 0.5 | 1 | inflated | 49.2% | 3.6% | $999k | 50.0% | 5.1% | 6.1 pts | $850k | 67.8% | $658k | $524k | 37.2% | 11.6% | 1.9 | 5.6 | none |
| overvalues women | k = 0.5 | 5 | list | 50.8% | 6.5% | $999k | 49.7% | 4.5% | 6.6 pts | $850k | 69.0% | $678k | $533k | 38.9% | 11.5% | 1.8 | 5.4 | none |
| overvalues women | k = 0.5 | 5 | inflated | 50.4% | 5.6% | $998k | 49.9% | 4.8% | 6.4 pts | $850k | 67.8% | $662k | $528k | 38.7% | 11.2% | 1.9 | 5.4 | none |
| overvalues women | k = 0.5 | 20 | list | 50.0% | 5.0% | $999k | -- | -- | 7.2 pts | $850k | 68.2% | $691k | $460k | 36.1% | 16.0% | 2.6 | 5.4 | none |
| overvalues women | k = 0.5 | 20 | inflated | 50.0% | 5.0% | $998k | -- | -- | 6.6 pts | $850k | 67.7% | $678k | $464k | 37.5% | 15.1% | 2.2 | 5.3 | none |
| overvalues women | k = 1 | 1 | list | 52.7% | 7.1% | $1,000k | 49.9% | 4.9% | 6.3 pts | $850k | 68.4% | $673k | $536k | 37.7% | 11.5% | 1.8 | 5.5 | none |
| overvalues women | k = 1 | 1 | inflated | 53.3% | 8.2% | $1,000k | 49.8% | 4.8% | 6.3 pts | $850k | 67.4% | $661k | $530k | 37.5% | 11.9% | 1.9 | 5.6 | none |
| overvalues women | k = 1 | 5 | list | 51.2% | 6.3% | $999k | 49.6% | 4.6% | 6.3 pts | $850k | 68.7% | $695k | $531k | 39.7% | 11.3% | 1.7 | 5.1 | none |
| overvalues women | k = 1 | 5 | inflated | 50.5% | 4.8% | $999k | 49.8% | 5.1% | 6.3 pts | $850k | 68.2% | $681k | $522k | 38.4% | 11.3% | 1.7 | 5.4 | none |
| overvalues women | k = 1 | 20 | list | 50.0% | 5.0% | $998k | -- | -- | 7.6 pts | $850k | 67.7% | $716k | $515k | 33.3% | 15.9% | 2.8 | 5.9 | none |
| overvalues women | k = 1 | 20 | inflated | 50.0% | 5.0% | $999k | -- | -- | 6.2 pts | $850k | 67.3% | $702k | $507k | 35.9% | 13.9% | 2.0 | 5.7 | none |
| cheapskate $500k | $500k | 1 | list | 27.5% | 0.0% | $500k | 51.2% | 5.3% | 7.9 pts | $850k | 68.8% | $672k | $533k | 38.4% | 12.3% | 1.9 | 5.4 | none |
| cheapskate $500k | $500k | 1 | inflated | 28.9% | 0.0% | $499k | 51.1% | 5.3% | 7.7 pts | $850k | 67.4% | $660k | $520k | 33.8% | 14.1% | 2.0 | 6.1 | none |
| cheapskate $500k | $500k | 5 | list | 30.0% | 0.0% | $500k | 56.7% | 6.7% | 12.7 pts | $850k | 69.9% | $667k | $531k | 32.3% | 14.4% | 3.0 | 6.1 | none |
| cheapskate $500k | $500k | 5 | inflated | 32.4% | 0.0% | $499k | 55.9% | 6.7% | 11.5 pts | $850k | 67.7% | $655k | $514k | 27.8% | 17.4% | 2.9 | 6.8 | none |
| cheapskate $500k | $500k | 20 | list | 50.0% | 5.0% | $500k | -- | -- | 11.7 pts | $350k | 76.3% | $350k | $350k | 43.2% | 22.4% | 2.9 | 3.8 | none |
| cheapskate $500k | $500k | 20 | inflated | 50.0% | 5.0% | $500k | -- | -- | 5.9 pts | $350k | 69.7% | $328k | $260k | 42.9% | 10.0% | 1.4 | 4.7 | none |
| marketing guy (big names) | k = 0.5 | 1 | list | 49.2% | 3.8% | $995k | 50.0% | 5.1% | 6.2 pts | $850k | 68.5% | $668k | $535k | 39.8% | 11.4% | 1.6 | 5.2 | none |
| marketing guy (big names) | k = 0.5 | 1 | inflated | 50.4% | 7.0% | $999k | 50.0% | 4.9% | 6.1 pts | $850k | 67.7% | $661k | $534k | 37.9% | 12.0% | 1.9 | 5.5 | none |
| marketing guy (big names) | k = 0.5 | 5 | list | 49.8% | 5.6% | $1,000k | 50.1% | 4.8% | 6.4 pts | $850k | 68.1% | $671k | $539k | 38.2% | 12.2% | 1.9 | 5.4 | none |
| marketing guy (big names) | k = 0.5 | 5 | inflated | 49.6% | 4.8% | $999k | 50.1% | 5.1% | 6.1 pts | $850k | 67.9% | $651k | $535k | 36.9% | 11.6% | 1.8 | 5.7 | none |
| marketing guy (big names) | k = 0.5 | 20 | list | 50.0% | 5.0% | $999k | -- | -- | 6.7 pts | $850k | 69.0% | $660k | $537k | 39.0% | 13.6% | 2.3 | 5.1 | none |
| marketing guy (big names) | k = 0.5 | 20 | inflated | 50.0% | 5.0% | $999k | -- | -- | 6.4 pts | $850k | 67.5% | $645k | $525k | 36.3% | 12.9% | 2.0 | 5.8 | none |
| marketing guy (big names) | k = 1 | 1 | list | 49.0% | 3.5% | $1,000k | 50.1% | 5.1% | 6.3 pts | $850k | 67.9% | $671k | $538k | 37.8% | 13.4% | 2.1 | 5.4 | none |
| marketing guy (big names) | k = 1 | 1 | inflated | 49.3% | 3.5% | $1,000k | 50.0% | 5.1% | 6.1 pts | $850k | 67.4% | $655k | $524k | 36.6% | 11.7% | 1.9 | 5.8 | none |
| marketing guy (big names) | k = 1 | 5 | list | 50.0% | 5.1% | $1,000k | 50.0% | 5.0% | 6.3 pts | $850k | 68.6% | $677k | $537k | 41.7% | 12.1% | 1.8 | 4.8 | none |
| marketing guy (big names) | k = 1 | 5 | inflated | 49.3% | 5.0% | $999k | 50.2% | 5.0% | 6.4 pts | $850k | 68.1% | $657k | $531k | 37.0% | 11.8% | 2.1 | 5.6 | none |
| marketing guy (big names) | k = 1 | 20 | list | 50.0% | 5.0% | $1,000k | -- | -- | 6.8 pts | $850k | 68.4% | $653k | $536k | 38.6% | 15.5% | 2.4 | 5.1 | none |
| marketing guy (big names) | k = 1 | 20 | inflated | 50.0% | 5.0% | $999k | -- | -- | 6.5 pts | $850k | 68.2% | $636k | $535k | 37.0% | 12.6% | 2.2 | 5.6 | none |
| wants real teams | lam = 0.05 | 1 | list | 50.2% | 3.7% | $1,000k | 50.0% | 5.1% | 6.1 pts | $850k | 68.1% | $676k | $540k | 39.4% | 10.7% | 1.7 | 5.3 | none |
| wants real teams | lam = 0.05 | 1 | inflated | 52.1% | 5.6% | $1,000k | 49.9% | 5.0% | 6.2 pts | $850k | 67.1% | $656k | $535k | 36.0% | 11.8% | 2.3 | 5.8 | none |
| wants real teams | lam = 0.05 | 5 | list | 50.7% | 6.4% | $1,000k | 49.8% | 4.5% | 6.3 pts | $850k | 68.1% | $674k | $541k | 36.9% | 12.7% | 1.9 | 5.6 | none |
| wants real teams | lam = 0.05 | 5 | inflated | 50.4% | 5.5% | $1,000k | 49.9% | 4.8% | 6.1 pts | $850k | 67.6% | $661k | $532k | 37.3% | 12.8% | 2.1 | 5.6 | none |
| wants real teams | lam = 0.05 | 20 | list | 50.0% | 5.0% | $999k | -- | -- | 6.6 pts | $850k | 68.4% | $668k | $541k | 38.7% | 12.8% | 1.9 | 5.3 | none |
| wants real teams | lam = 0.05 | 20 | inflated | 50.0% | 5.0% | $999k | -- | -- | 6.4 pts | $850k | 68.8% | $652k | $536k | 39.2% | 11.6% | 2.0 | 5.2 | none |
| wants real teams | lam = 0.15 | 1 | list | 49.7% | 5.2% | $998k | 50.0% | 5.0% | 6.4 pts | $850k | 68.6% | $664k | $542k | 38.2% | 12.9% | 2.0 | 5.4 | none |
| wants real teams | lam = 0.15 | 1 | inflated | 49.2% | 5.0% | $1,000k | 50.0% | 5.0% | 6.4 pts | $850k | 67.9% | $661k | $526k | 38.9% | 12.0% | 1.9 | 5.4 | none |
| wants real teams | lam = 0.15 | 5 | list | 49.5% | 4.4% | $998k | 50.2% | 5.2% | 6.3 pts | $850k | 68.4% | $674k | $544k | 39.2% | 11.3% | 1.9 | 5.2 | none |
| wants real teams | lam = 0.15 | 5 | inflated | 50.3% | 5.2% | $999k | 49.9% | 4.9% | 6.3 pts | $850k | 68.0% | $662k | $531k | 36.9% | 13.2% | 2.2 | 5.7 | none |
| wants real teams | lam = 0.15 | 20 | list | 50.0% | 5.0% | $999k | -- | -- | 7.0 pts | $850k | 69.6% | $671k | $536k | 40.6% | 13.0% | 2.1 | 4.9 | none |
| wants real teams | lam = 0.15 | 20 | inflated | 50.0% | 5.0% | $998k | -- | -- | 6.3 pts | $850k | 67.9% | $665k | $539k | 38.5% | 13.3% | 2.0 | 5.3 | none |
| wants real teams | lam = 0.5 | 1 | list | 46.6% | 3.3% | $1,000k | 50.2% | 5.1% | 6.6 pts | $850k | 68.5% | $673k | $539k | 37.9% | 12.2% | 2.0 | 5.4 | none |
| wants real teams | lam = 0.5 | 1 | inflated | 49.0% | 4.7% | $1,000k | 50.1% | 5.0% | 6.2 pts | $850k | 67.7% | $660k | $527k | 36.2% | 12.4% | 1.9 | 5.8 | none |
| wants real teams | lam = 0.5 | 5 | list | 46.1% | 2.9% | $1,000k | 51.3% | 5.7% | 7.1 pts | $850k | 68.8% | $673k | $537k | 36.9% | 13.4% | 2.1 | 5.6 | none |
| wants real teams | lam = 0.5 | 5 | inflated | 48.5% | 3.7% | $1,000k | 50.5% | 5.4% | 6.3 pts | $850k | 67.1% | $674k | $541k | 35.8% | 12.5% | 2.0 | 6.0 | none |
| wants real teams | lam = 0.5 | 20 | list | 50.0% | 5.0% | $997k | -- | -- | 9.0 pts | $850k | 67.4% | $705k | $577k | 34.3% | 18.4% | 3.0 | 5.4 | none |
| wants real teams | lam = 0.5 | 20 | inflated | 50.0% | 5.0% | $996k | -- | -- | 8.1 pts | $850k | 68.4% | $739k | $598k | 35.6% | 16.9% | 2.9 | 5.4 | none |
| bargains first | $120k | 1 | list | 35.7% | 0.0% | $637k | 50.8% | 5.3% | 6.8 pts | $850k | 68.1% | $670k | $537k | 38.4% | 11.2% | 1.8 | 5.4 | none |
| bargains first | $120k | 1 | inflated | 34.2% | 0.0% | $652k | 50.8% | 5.3% | 6.9 pts | $850k | 67.8% | $654k | $532k | 36.0% | 13.0% | 2.0 | 5.8 | none |
| bargains first | $120k | 5 | list | 38.7% | 0.3% | $843k | 53.8% | 6.6% | 8.6 pts | $850k | 65.9% | $664k | $527k | 29.8% | 16.0% | 2.4 | 6.7 | none |
| bargains first | $120k | 5 | inflated | 37.3% | 0.3% | $807k | 54.2% | 6.6% | 9.6 pts | $850k | 66.6% | $649k | $528k | 29.8% | 16.8% | 2.9 | 6.5 | none |
| bargains first | $120k | 20 | list | 50.0% | 5.0% | $938k | -- | -- | 9.5 pts | $648k | 75.8% | $623k | $501k | 48.9% | 14.9% | 2.0 | 3.9 | Will Howells #20M, Bruno Faletto #21M, Leigh Waters #24F, Vanshik Kapadia #24M (+6) |
| bargains first | $120k | 20 | inflated | 50.0% | 5.0% | $935k | -- | -- | 11.4 pts | $512k | 79.9% | $533k | $450k | 50.8% | 18.7% | 2.5 | 3.2 | Bruno Faletto #21M, Dylan Frazier #23M, Vanshik Kapadia #24M, Armaan Bhatia #25M (+6) |
| bargains first | $250k | 1 | list | 44.6% | 0.9% | $988k | 50.3% | 5.2% | 6.2 pts | $850k | 68.2% | $676k | $538k | 38.2% | 12.4% | 1.8 | 5.4 | none |
| bargains first | $250k | 1 | inflated | 44.7% | 0.9% | $995k | 50.3% | 5.2% | 6.0 pts | $850k | 67.7% | $659k | $526k | 39.3% | 11.1% | 1.7 | 5.3 | none |
| bargains first | $250k | 5 | list | 42.8% | 0.8% | $992k | 52.4% | 6.4% | 6.9 pts | $850k | 67.8% | $660k | $524k | 36.3% | 12.9% | 2.0 | 5.7 | none |
| bargains first | $250k | 5 | inflated | 40.3% | 0.4% | $992k | 53.2% | 6.5% | 7.7 pts | $850k | 67.2% | $652k | $520k | 34.1% | 12.6% | 2.2 | 6.2 | none |
| bargains first | $250k | 20 | list | 50.0% | 5.0% | $966k | -- | -- | 15.2 pts | $547k | 84.3% | $499k | $396k | 55.9% | 18.2% | 2.6 | 2.8 | none |
| bargains first | $250k | 20 | inflated | 50.0% | 5.0% | $976k | -- | -- | 16.4 pts | $515k | 87.6% | $440k | $283k | 57.4% | 21.1% | 2.6 | 2.6 | none |

## Who each persona buys, and at what premium

k = 1 cells, `list` expectation: players the persona carries more often than the quants in the same room (share of persona rosters minus share of quant rosters), with what the persona paid vs the list.

- **overvalues men**, k = 0.5: Ben Johns (+42pp, paid $545k vs $474k list), Eugenia Carolina Lopez Ascarate (+11pp, paid $30k vs $64k list), Pablo Tellez (+11pp, paid $31k vs $71k list), Jessie Irvine (+11pp, paid $47k vs $89k list), Nicolas Acevedo (+11pp, paid $183k vs $187k list), Chao Yi Wang (+11pp, paid $256k vs $216k list)
- **overvalues men**, k = 1: Christopher Haworth (+26pp, paid $185k vs $95k list), Yufei Long (+16pp, paid $30k vs $79k list), Christian Alshon (+16pp, paid $402k vs $400k list), Gabriel Tardio (+16pp, paid $484k vs $409k list), Seone Mendez (+11pp, paid $51k vs $64k list), Kiora Kunimoto (+11pp, paid $38k vs $68k list)
- **overvalues women**, k = 0.5: Donald Young (+16pp, paid $30k vs $62k list), Joseph Wild (+16pp, paid $30k vs $69k list), Pablo Tellez (+16pp, paid $31k vs $71k list), Grayson Goldin (+11pp, paid $30k vs $44k list), Anderson Scarpa (+11pp, paid $30k vs $70k list), Jessie Irvine (+11pp, paid $51k vs $89k list)
- **overvalues women**, k = 1: Anna Bright (+32pp, paid $674k vs $613k list), Kiora Kunimoto (+21pp, paid $59k vs $68k list), Yuta Funemizu (+16pp, paid $90k vs $119k list), Juan Benitez (+11pp, paid $30k vs $55k list), Matthew Barlow (+11pp, paid $30k vs $48k list), Harsh Mehta (+11pp, paid $30k vs $62k list)
- **cheapskate $500k**, $500k: Roos Van Reek (+26pp, paid $43k vs $96k list), Hunter Johnson (+21pp, paid $99k vs $137k list), Leigh Waters (+21pp, paid $126k vs $157k list), Genie Erokhina (+18pp, paid $30k vs $58k list), Bobbi Oshiro (+16pp, paid $266k vs $247k list), Alexander Crum (+12pp, paid $30k vs $30k list)
- **marketing guy (big names)**, k = 0.5: John Lucian Goins (+11pp, paid $35k vs $83k list), Thomas Yu (+11pp, paid $49k vs $84k list), Jack Munro (+11pp, paid $227k vs $167k list), Meghan Dizon (+11pp, paid $154k vs $171k list), Etta Tuionetoa (+11pp, paid $228k vs $187k list), Nicolas Acevedo (+11pp, paid $160k vs $187k list)
- **marketing guy (big names)**, k = 1: Robert Slutsky (+21pp, paid $182k vs $185k list), Mary Brascia (+16pp, paid $40k vs $65k list), Bruno Faletto (+16pp, paid $153k vs $141k list), Thomas Wilson (+16pp, paid $250k vs $209k list), Bobbi Oshiro (+16pp, paid $307k vs $247k list), Tama Shimabukuro (+11pp, paid $30k vs $58k list)
- **wants real teams**, lam = 0.05: Isabella Dunlap (+16pp, paid $42k vs $88k list), Donald Young (+11pp, paid $30k vs $62k list), Luca Mack (+11pp, paid $30k vs $69k list), Estee Widdershoven (+11pp, paid $52k vs $80k list), Samantha Parker (+11pp, paid $57k vs $99k list), Augustus Ge (+11pp, paid $49k vs $102k list)
- **wants real teams**, lam = 0.15: Zane Ford (+21pp, paid $65k vs $53k list), Catherine Parenteau (+16pp, paid $311k vs $267k list), Eric Oncins (+16pp, paid $342k vs $307k list), Alexander Crum (+11pp, paid $30k vs $30k list), Harsh Mehta (+11pp, paid $30k vs $62k list), Yufei Long (+11pp, paid $51k vs $79k list)
- **wants real teams**, lam = 0.5: Wyatt Stone (+39pp, paid $31k vs $30k list), Jonathan Truong (+19pp, paid $37k vs $30k list), Alexa Schull (+19pp, paid $31k vs $30k list), Milan Rane (+16pp, paid $32k vs $83k list), Camden Chaffin (+13pp, paid $38k vs $30k list), Juan Benitez (+11pp, paid $30k vs $55k list)
- **bargains first**, $120k: Vanshik Kapadia (+47pp, paid $84k vs $132k list), Bruno Faletto (+47pp, paid $111k vs $141k list), Layne Sleeth (+47pp, paid $109k vs $147k list), Yana Newell (+42pp, paid $94k vs $144k list), Leigh Waters (+42pp, paid $119k vs $157k list), Gabriel Joseph (+32pp, paid $98k vs $30k list)
- **bargains first**, $250k: Armaan Bhatia (+32pp, paid $177k vs $130k list), Vivienne David (+32pp, paid $226k vs $204k list), Noe Khlif (+26pp, paid $192k vs $178k list), Sahra Dennehy (+26pp, paid $211k vs $203k list), Thomas Wilson (+26pp, paid $206k vs $209k list), Katerina Stewart (+21pp, paid $162k vs $65k list)

## What this says (hand-written against the seed-1 grid; re-check the numbers above if it is re-run)

- **The auction does not dent the Waters team.** She sells at the cap's maximum
  ($850k = cap minus five floor players) at sale 1 in every quant cell and every
  seed; her buyer is left with $150k for five slots and takes five floor players;
  the team still wins 67-68% of ties with a 33-39% title shot (snake: 66% / 36%).
  The snake gave her buyer floor-priced players too, so nothing is lost. The
  one price-side lever from `dials.md` (charge her more than the list) is what
  the room does on its own, and it is not enough: at $850k she is still the best
  buy in the league. Seeds 2 and 3 agree (67-69% / 32-40%).
- **But the auction makes a chase.** Runner-up title odds 11.5-16% (snake 7.3%),
  1.8-3.0 teams at 10%+ (snake 1.0), effective contenders 5.2-5.8 (snake 6.4,
  i.e. the pack is LESS equal). The 10%+ teams are almost all the same build:
  two players at $390-490k plus four floor players -- Patriquin + Rohrabacher,
  Jorja Johnson + Alshon, Todd + Staksrud, JW Johnson + Humberg, Fahey + Black.
  A man and a woman is the strong version: the tie model puts the two stars
  together in one mixed game (near-lock), and each carries a same-gender game
  with a floor partner at about even odds, so a star plays in three of the four
  games (Patriquin + Rohrabacher vs the field: WD 48%, MD 66%, MXD1 74%, MXD2
  46%, tie 62%; the same roster with Tardio in place of Rohrabacher, 45%; with
  Fahey in place of Patriquin, 59%). A snake cannot build it -- the pick-2 team
  waits until pick 39 and the top 60 are gone before round 4 -- so every snake
  team but Waters' is "one star plus depth", and nineteen of those are equals.
  Money is the only constraint at auction, and the cap rewards concentration.
  Nobody programmed the build; it falls out of owners who value rosters, not
  players (their objective is the projected roster's tie probability).
- **The room re-prices the list's middle.** Stars (#1-5) go at 101-115% of
  list, the #6-15 tier at 111-130% (the second star of a two-star build; the
  biggest single premiums are the $130-210k men -- Bhatia, Howells, Frazier,
  Huynh, Garnett -- at +40-65%), #16-30 at about list, and the #31-60 depth at
  51-67%: the $79-96k role players (Rane, Van Reek, Dunlap, Brascia, Petrei)
  sell for the floor. The room pays for a second star and for fit, and not
  for depth, because the winning build has four floor slots. The DreamBreaker
  specialists the list floors go at 3-6x (Joseph $98k, Haworth $185k on the
  rosters that want them): phi is a context average and cannot see fit; a
  room prices one context. Every cap is spent ($999-1,000k), nothing in the
  top 30 is left unsold, ~14-15 bidders per sale, nobody stranded.
- **Expectations and noise are second order** (the earlier sensitivity was an
  artefact of a nomination rule that let stars come up after the money was
  gone). Owners who anticipate inflation give Waters a slightly smaller title
  share (33-37% vs 37-39%) and the runner-up more (11.5-16%); 10% belief noise
  raises the stars' prices ~8% (Bright $654-675k vs $610-622k) and trims the
  runner-up (11.5-11.7% vs 14.8-16%): noisy owners overpay the stars, which
  squeezes the two-star budgets.
- **Personas: the auction forgives individual mistakes and punishes shared
  ones.** Alone, the $500k cheapskate wins 27-29% (snake 21%), bargains-first
  at $120k 34-36% (snake 24%) -- a bad opening no longer costs the whole top 60,
  because anyone can be bought at any time. Five cheapskates still break the
  pack (spread 11.5-12.7, the quants at 56-57%). Overvaluing a gender, chasing
  names and mild loyalty are free, as in the snake (overvalues-women k=1 alone
  is again mildly AHEAD, 53% / 7-8%: women carry mixed). Strong loyalty
  (lam 0.5) costs 1-3 points alone and, as a league norm, overpays the known
  stars (Bright $705-739k, Johns $577-598k) and gives the widest chase in the
  grid (runner-up 17-18%, 3 teams at 10%+).
- **The only leagues where Waters goes cheaper are the ones where everyone
  shares the same blind spot, and they are worse leagues.** Twenty owners who
  overvalue men (k=1) sell her at $759-779k and her team wins 74-75% with a 50%
  title share; twenty bargain hunters ($250k threshold) spend rounds 1-3 on
  mid-priced players and then let the stars go for half price (Waters
  $515-547k, Bright $440-499k, Johns $283-396k): her team 84-88%, favourite
  56-57%, parity spread 15-16, worse than anything the snake produced. Twenty
  $500k cheapskates hit their own $350k first-buy maximum on Waters (Bright
  $328-350k, Johns $260-350k) and get a 43% favourite with a runner-up at
  10-22%. The cap maximum binds in every mixed league: whatever the room, she
  costs what a team may pay.
- **For the price list**: the room's curve is more convex than phi's -- a
  premium on the second star and on fit, the floor for depth. That is not a
  reason to change the list (the list prices context-averaged value; a room
  prices fit and liquidity on the day) but it is the shape to expect in the
  league-price / surplus column once MLP publishes, and it says the list's
  $60-100k depth tier is where the surplus will look largest.


## The market limit: what the backward induction settles on (2026-09-05; `market_eq.py` -> `market_eq.md`, `market_prices.csv`)

**Why this exists.** The owners above fill one slot at a time against a
cheat sheet (the list, or the list inflated by money left). A planner who
buys whole rosters beats them: at list prices the best $1M roster WITHOUT
Waters is worth 0.597 (tie probability vs the reference team; Staksrud,
Acevedo, Joseph, Oshiro, Tuionetoa, Padegimaite, $996k -- `--at-list`
prints it), while the greedy owners' rosters come out around 0.47, and a
"watch nineteen sales, then buy the #20-26 players" human finished in the
top three of twenty in two of three rooms (both in-session measurements
against `auction_sim.py`; the scripts were not kept -- the 0.597 is
reproducible, the greedy comparison is not). So owner 19 stops chasing the
players that human beats him with, then 18, then 17, and the question is
what the prices look like once nobody overpays.

**Method.** Every owner plans its best $1M roster at the current prices --
exhaustive over every 3M+3W combination of a candidate set (top 20 per
gender by value, plus the top singles values, the cheapest pool players
and the best floor players; brute-force checked by `--selftest`). Rosters
are handed out in a random order each round. A priced player on an
above-average roster is marked up and on a below-average roster marked
down (times exp(eta * c * (roster value - average)), c swept 4/8/12, eta
decaying); unsold priced players fall; prices are clipped to the $30k
floor and the $850k first-buy maximum (cap minus the cheapest legal
completion, the same rule that binds in the auction). A player on a roster
that holds a capped player is priced by the best FREE roster that would
hold him instead (shadow pricing), so the capped player's premium is not
smeared onto floor teammates -- without it Waters' five $30k teammates
drifted to $50-80k and her team read 0.73. Prices are the average over
the last third of 200 rounds. The fixed point: every roster an owner can
buy is worth the same, except for the players the maximum stops from
being priced -- those rationed players are the difference makers. True
values, identical owners (`--noise` perturbs beliefs; at zero noise the
seed only changes the allocation order). `--rule demand` is the textbook
excess-demand version (over-demanded players up, unsold down, nothing
else moves), kept as a check that the rationed set does not depend on the
rule -- with identical owners its demand signal is one roster, so it
never clears (roster sd 0.12) and leaves the middle of the list untouched.

**What settles out (c 4/8/12; ranges across the three runs).**

- **Waters is the only rationed player.** She sits at the $850k maximum
  with excess demand in every run and under both rules; nobody else does.
  Her team at market prices: five $30k teammates (Emmrich, Truong, Joseph,
  Erokhina or Schull, Padegimaite), tie value 0.648-0.665, in-league win%
  61.7-63.4, title 24-38% (auction 67-68% / 33-39%; snake 66 / 36). She
  is worth more than a team may pay; the cap, not the market, sets her
  price.
- **Bright is NOT rationed.** She is bid up to the point where a Bright
  team is an average team: $697k (the price at which the best roster
  holding her equals the average roster the market hands out, 0.520) to
  $760k (the time-averaged market price; 114-124% of her $613k list),
  and stays under the $850k maximum with room to spare. Same for every
  other star: Johns $501-545k (106-115% of $474k), Todd $547-576k
  (112-118%), Jorja Johnson $538-559k, Fahey $501-525k, JW Johnson
  $458-487k, Patriquin $442-472k, Rohrabacher $472-484k, Tardio
  $434-458k. The two numbers bracket the price: the market column is an
  average of oscillating round-to-round prices, and a roster that is
  affordable in most rounds is not always affordable at the averaged
  prices, so for the top dozen the averaged market sits a little above
  the exact average-team price (below it for the #14-24 stars: Jackie
  Kawamoto, Glozman, Parenteau read 117-121% on the average-team
  benchmark vs 115-116% market).
- **The other nineteen teams are equals.** Second-best team 50.6-51.1%
  in-league, the worst 42-47%; free-roster tie values 0.520 +- 0.008-0.020;
  runner-up title share 6-8%, exactly one team at 10%+. The auction's
  two-star chase (runner-up 11.5-16%, 1.8-3.0 teams at 10%+) is a
  greedy-owner artefact: it lives on the second star selling to a room
  that never priced whole rosters. Once everyone plans rosters, the
  second star's roster is worth what everyone else's is.
- **List vs market, by tier (mean market / list within gender):** women
  #1-5 117%, #6-15 116-117%, #16-30 105-106%, #31-60 53-55% (11-12 of
  30 at the floor); men #1-5 112-114%, #6-15 112-113%, #16-30 97-98%,
  #31-60 60-61% (15-16 at the floor). Priced-pool total $19.84-19.98M
  against the list's $20.00M: the money that the list spreads over the
  bottom half of the pool moves to the top thirty. The list is right in
  level and in rank order; the market steepens the shape, in the same
  direction the auction room did but with the premium on the FIRST
  fifteen rather than the second star.
- **Why the stars go UP, not down.** The "don't overpay" benchmark is not
  the 0.597 roster (one owner gets it); it is the average roster the
  market hands out, 0.520 (about 49% in-league), because the mid-priced
  players every best-no-Waters roster leans on (Staksrud, Acevedo,
  Oshiro, Tuionetoa) get bid up until that roster is average too. A star
  priced against an average team is worth more than a star priced
  against the best team you could build without her.
- **"Buy teams" as a solver is this solver.** An owner who tries to buy
  Waters + Johns + Bright + JW Johnson and two floor players finds the
  four cost $2.29M at list and $2.64-2.65M at market; the fallback chain
  (drop the fourth star, then the third, ...) ends at one star plus
  fill-ins, worth what every other roster is worth (about 49%), because
  each star's price already carries the value of the best roster she
  fits in. Only Waters breaks the chain, and only because the maximum
  stops her price short.

**Caveats.** Identical, perfectly informed owners on true values (the
sim's "noise 0"); a heuristic price adjustment rather than an auction
mechanism (there are no bids, nominations or budgets in the loop -- the
fixed point is what any such mechanism has to converge to, not a
prediction of any particular room's path); the fixed point is not exact
(free-roster sd 0.008-0.020, star prices oscillate +-$5-15k round to
round); the candidate set is top-20 per gender by value plus extras
(`--k`), so a roster built from four #21-30 players is not searched;
rosters are allocated sequentially, so an owner late in the order gets
the best of what is left rather than the best roster on the board; and
the indifference (best) column in `market_eq.md` is measured against a
roster only one owner can hold, which is why it reads $10-25k under the
average-team price. None of these change the rationed set (Waters alone,
every run, both rules).

**For the price list.** `market_prices.csv` carries the market price and
the average-team price next to the list price for all 120 pool players;
it is the second column to show once MLP publishes real prices, and the
shape to expect in the surplus column: the top fifteen a little above the
list, #16-30 at the list, the bottom half of the pool at the floor. The
list itself does not change -- it prices context-averaged value, which is
what a fair draft board should say; the market prices fit under a cap
with a first-buy maximum, which is what a room will pay.

**The one thing to carry into the other sims.** `draft_sim.py` and
`auction_sim.py` owners project their own roster greedily (best next
player given what they hold). `market_eq.solve` is an exact roster
planner; swapping it into `Owner`'s projection and re-running the persona
and auction grids is the follow-up that would tell whether the two-star
chase and the persona rankings survive owners who can count.

## Why the chasers buy a man AND a woman (2026-09-05; `two_star.py`, against the market-limit field)

The auction room's 10%+ teams were two-star builds and almost always one
man plus one woman. `two_star.py` plays each two-star shape (two stars +
the $30k fill-ins the market-limit rosters actually use: Truong, Joseph,
Emmrich; Padegimaite, Erokhina, Schull) against the nineteen non-Waters
rosters of a market-limit league (`market_eq` cache, equalize c 8, rep 0)
and reads the tie game by game. Field pair strengths, for reference: WD
+1.76, MD +1.63, MXD1 +1.83, MXD2 +1.57, DreamBreaker singles +1.53 (S =
sum + gamma|gap| on the per-point logit scale).

| build | WD | MD | MXD1 | MXD2 | DB | tie | games won of 4 | cost at market |
|---|---|---|---|---|---|---|---|---|
| M+W Patriquin + Rohrabacher | 47% | 52% | **77%** | 30% | 44% | **49.7%** | 2.06 | $1.08M |
| M+W Tardio + Fahey | 44% | 51% | 75% | 30% | 53% | 51.3% | 2.00 | $1.12M |
| M+M Patriquin + Tardio | **19%** | **86%** | 39% | 56% | 38% | **44.3%** | 2.00 | $1.06M |
| M+M Patriquin + JW Johnson | 19% | 86% | 39% | 57% | 40% | 45.9% | 2.01 | $1.09M |
| M+M Johns + JW Johnson | 19% | 87% | 41% | 57% | 45% | 49.5% | 2.05 | $1.15M |
| F+F Rohrabacher + Fahey | 81% | 26% | 40% | 57% | 52% | 52.2% | 2.03 | $1.13M |
| F+F Rohrabacher + Todd | 84% | 26% | 41% | 60% | 51% | **54.3%** | 2.10 | $1.18M |
| M+W Johns + Bright | 60% | 54% | 88% | 30% | 56% | 65.5% | 2.33 | $1.42M |
| F+F Bright + Todd | 91% | 26% | 53% | 61% | 58% | 65.9% | 2.32 | $1.46M |

(Per-game columns are means over the nineteen opponents; the tie column
is the mean tie probability, which is what the season sim uses.)

**Two men win the same number of games and lose the tie.** Every
two-star build wins about two games of four (2.00-2.10) -- the stars'
value is the same wherever it lands. What differs is where the wins sit.
The tie needs three of four, or 2-2 plus the DreamBreaker, and the
`lineup()` rule decides where the stars play:

1. **A man and a woman play TOGETHER in mixed.** The weakest-link term
   (gamma < 0) rewards equal partners, so the split rule pairs the two
   stars in MXD1 (+2.19 vs the field's +1.83: 77%) and gives each of them
   a floor partner in their own-gender game (about 50%). Two men cannot
   share a mixed game: they play MD together (+2.15 vs +1.63: 86%) and
   then each drags a floor woman through a mixed game (39% / 56%).
   Same expected wins; the M+W shape has one near-lock and three
   coin-flips, the M+M shape has one near-lock, one near-certain loss
   and two coin-flips. Three of four is easier to reach from the first
   shape (P(3+ of 4) about 0.32 vs 0.28).
2. **The punted game is cheaper on the women's side of the sheet than
   it looks -- and dearest in WD.** Every two-star roster fields one
   floor-only pair (+1.31 to +1.33). M+W punts MXD2, the field's weakest
   game (+1.57: 30%); M+M punts WD, the field's strongest (+1.76: 19%).
   Women's values are spread 1.5x wider than men's (priced pool top-5
   mean +1.32 vs +1.07, floor +0.60 vs +0.55), so the field's women's
   pairs are further ahead of a floor pair than its men's pairs are.
   That is also why F+F is the best two-star shape (52-54%): it punts MD
   (26%) and owns WD (81-84%), and the woman + floor man mixed pairs
   still draw even. The room bought M+W more often than F+F only because
   two top women cost more ($1.13-1.18M vs $1.08M here).
3. **Cheap men are singles specialists; cheap women are not.** The floor
   men carry DreamBreaker values of +0.99 / +1.61 / +1.32 (Joseph,
   Emmrich are exactly the DB-only specialists the value model priced in
   Phase 1); the floor women +1.26 / +0.99 / +0.88. A roster with two
   floor women in the DB lineup runs 38-40% in the DreamBreaker; with a
   star woman and Padegimaite it is 44-53%.

So "why not two men" has three answers that all point the same way: two
men cannot share a court in mixed, two men leave women's doubles to the
floor where the gap is widest, and two men leave the DreamBreaker to
floor women. A man and a woman is the shape that spends two stars on
three games.

**The planner's verdict in one line.** Every one of these rosters costs
$1.06-1.18M at market prices ($1.42-1.46M for the Johns/Bright/Todd
versions) -- over the cap -- and plays the market-limit field at
44-54%. The chase exists in the auction room for two reasons, and the
price is the smaller one: the second star sold there for $390-490k
against $440-485k at market, just enough to squeeze under the cap; and
the greedy room's other rosters are weaker (about 0.47 vs the planners'
0.52), so the same Patriquin + Rohrabacher roster reads 62% against
the greedy field and 50% against this one. At market prices it cannot
be assembled, and if it could it would be an average team. What survives is the
shape: if a rule ever lets a two-star roster fit (a higher cap, a
cheaper second star, a min-spend that pushes money to the middle), the
team to build is a man and a woman, or two women, never two men.
