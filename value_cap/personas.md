# Owner personas in the draft

Shipped tag list (alpha 1, joint pool, Waters tagged at $769k), snake draft on the `mlp2026` board, every owner at 10% belief noise, 20 drafts x 200 seasons per cell, seed 1. Each cell puts k persona owners at random draft slots among quants. Persona definitions and strengths are in the docstring of `personas.py` (strengths are swept, not picked). Parity = 50% win, 5% title. Built by `personas.py`.

## The grid

| persona | strength | how many of 20 | persona teams: win% | title% | spend | quant teams: win% | title% | parity spread | Waters' team win% | Bright's team | Johns' team | top-30 undrafted (any draft) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| quant baseline |  | -- | 50.0% | 5.0% | $975k | -- | -- | 4.4 pts | 65.5% | 51.2% | 50.9% | none |
| overvalues men | k = 0.5 | 1 | 50.2% | 4.0% | $985k | 50.0% | 5.1% | 4.5 pts | 65.7% | 51.8% | 50.7% | none |
| overvalues men | k = 0.5 | 5 | 50.2% | 5.2% | $975k | 49.9% | 4.9% | 4.6 pts | 66.1% | 51.9% | 50.4% | none |
| overvalues men | k = 0.5 | 20 | 50.0% | 5.0% | $975k | -- | -- | 4.5 pts | 65.6% | 51.2% | 50.9% | none |
| overvalues men | k = 1 | 1 | 49.1% | 4.5% | $974k | 50.0% | 5.0% | 4.4 pts | 65.6% | 51.7% | 50.5% | none |
| overvalues men | k = 1 | 5 | 47.1% | 2.5% | $956k | 51.0% | 5.8% | 4.8 pts | 65.4% | 51.5% | 49.5% | none |
| overvalues men | k = 1 | 20 | 50.0% | 5.0% | $974k | -- | -- | 5.1 pts | 65.9% | 51.5% | 49.0% | none |
| overvalues women | k = 0.5 | 1 | 49.8% | 4.6% | $976k | 50.0% | 5.0% | 4.5 pts | 65.6% | 51.8% | 50.6% | none |
| overvalues women | k = 0.5 | 5 | 49.9% | 5.0% | $970k | 50.0% | 5.0% | 4.6 pts | 65.6% | 52.0% | 50.1% | none |
| overvalues women | k = 0.5 | 20 | 50.0% | 5.0% | $972k | -- | -- | 4.6 pts | 65.1% | 51.6% | 50.7% | none |
| overvalues women | k = 1 | 1 | 49.5% | 4.6% | $976k | 50.0% | 5.0% | 4.5 pts | 65.6% | 51.7% | 50.7% | none |
| overvalues women | k = 1 | 5 | 49.2% | 4.4% | $968k | 50.3% | 5.2% | 4.5 pts | 65.6% | 51.7% | 50.3% | none |
| overvalues women | k = 1 | 20 | 50.0% | 5.0% | $971k | -- | -- | 4.6 pts | 64.9% | 51.5% | 49.6% | Alix Truong #30F |
| cheapskate $500k | $500k | 1 | 21.3% | 0.0% | $494k | 51.5% | 5.3% | 7.6 pts | 65.6% | 52.0% | 50.4% | Dekel Bar #27M, Brooke Buckner #29F |
| cheapskate $500k | $500k | 5 | 26.7% | 0.0% | $494k | 57.8% | 6.7% | 13.7 pts | 68.5% | 56.7% | 56.6% | Federico Staksrud #7M, Eric Oncins #8M, Tina Pisnik #8F, Jay Devilliers #9M (+39) |
| cheapskate $500k | $500k | 20 | 50.0% | 5.0% | $483k | -- | -- | 4.3 pts | not drafted | not drafted | not drafted | Anna Leigh Waters #1F, Ben Johns #1M, Anna Bright #2F, JW Johnson #2M (+51) |
| marketing guy (big names) | k = 0.5 | 1 | 51.9% | 6.2% | $982k | 49.9% | 4.9% | 4.4 pts | 65.6% | 51.9% | 50.6% | none |
| marketing guy (big names) | k = 0.5 | 5 | 51.0% | 5.6% | $979k | 49.7% | 4.8% | 4.5 pts | 65.9% | 52.1% | 50.5% | none |
| marketing guy (big names) | k = 0.5 | 20 | 50.0% | 5.0% | $975k | -- | -- | 4.4 pts | 65.9% | 51.4% | 50.4% | none |
| marketing guy (big names) | k = 1 | 1 | 51.5% | 5.7% | $977k | 49.9% | 5.0% | 4.4 pts | 65.7% | 51.7% | 50.5% | none |
| marketing guy (big names) | k = 1 | 5 | 51.3% | 5.8% | $981k | 49.6% | 4.7% | 4.5 pts | 65.8% | 52.2% | 50.4% | CJ Klinger #28M |
| marketing guy (big names) | k = 1 | 20 | 50.0% | 5.0% | $975k | -- | -- | 4.3 pts | 65.8% | 51.9% | 50.6% | none |
| wants real teams | lam = 0.05 | 1 | 50.8% | 5.4% | $971k | 50.0% | 5.0% | 4.4 pts | 65.6% | 52.1% | 50.6% | none |
| wants real teams | lam = 0.05 | 5 | 49.9% | 4.8% | $968k | 50.0% | 5.1% | 4.5 pts | 66.0% | 51.3% | 50.7% | none |
| wants real teams | lam = 0.05 | 20 | 50.0% | 5.0% | $972k | -- | -- | 4.5 pts | 65.7% | 51.6% | 50.4% | none |
| wants real teams | lam = 0.15 | 1 | 48.7% | 3.7% | $956k | 50.1% | 5.1% | 4.4 pts | 65.6% | 51.8% | 50.5% | none |
| wants real teams | lam = 0.15 | 5 | 48.4% | 4.1% | $958k | 50.5% | 5.3% | 4.7 pts | 65.8% | 50.9% | 50.2% | none |
| wants real teams | lam = 0.15 | 20 | 50.0% | 5.0% | $968k | -- | -- | 4.5 pts | 64.9% | 51.2% | 49.7% | Yuta Funemizu #26M, Roscoe Bellamy #29M |
| wants real teams | lam = 0.5 | 1 | 42.0% | 2.0% | $888k | 50.4% | 5.2% | 4.8 pts | 65.6% | 50.9% | 49.8% | CJ Klinger #28M |
| wants real teams | lam = 0.5 | 5 | 43.4% | 2.4% | $896k | 52.2% | 5.9% | 6.3 pts | 65.7% | 50.3% | 50.6% | Leigh Waters #24F, Yuta Funemizu #26M, Roscoe Bellamy #29M |
| wants real teams | lam = 0.5 | 20 | 50.0% | 5.0% | $952k | -- | -- | 5.8 pts | 63.2% | 49.3% | 49.1% | Bruno Faletto #21M, Dylan Frazier #23M, Meghan Dizon #23F, Vanshik Kapadia #24M (+6) |
| bargains first | $120k | 1 | 24.1% | 0.0% | $600k | 51.4% | 5.3% | 7.1 pts | 65.5% | 51.6% | 50.6% | CJ Klinger #28M, Augustus Ge #30M |
| bargains first | $120k | 5 | 38.8% | 0.2% | $802k | 53.7% | 6.6% | 7.5 pts | 66.4% | 52.4% | 52.6% | Bruno Faletto #21M, Dylan Frazier #23M, Meghan Dizon #23F, Vanshik Kapadia #24M (+8) |
| bargains first | $120k | 20 | 50.0% | 5.0% | $963k | -- | -- | 4.8 pts | 65.8% | 52.5% | 49.7% | Riley Newman #10M, Jack Sock #11M, Phuc Huynh #12M, Thomas Wilson #13M (+17) |
| bargains first | $250k | 1 | 38.6% | 0.1% | $817k | 50.6% | 5.3% | 4.9 pts | 65.5% | 51.5% | 50.0% | none |
| bargains first | $250k | 5 | 42.2% | 0.5% | $860k | 52.6% | 6.5% | 5.7 pts | 65.1% | 51.8% | 51.4% | Jillian Braverman #26F, Dekel Bar #27M, Yana Newell #28F, CJ Klinger #28M (+2) |
| bargains first | $250k | 20 | 50.0% | 5.0% | $921k | -- | -- | 4.2 pts | not drafted | 56.4% | 52.5% | Anna Leigh Waters #1F, Gabriel Tardio #4M, Andrei Daescu #6M, Federico Staksrud #7M (+7) |

## Who each persona drafts

Players the persona carries more often than the quants in the same league (share of the persona's rosters minus share of the quants' rosters, percentage points), k = 1 cells. The all-20 cells are uninformative here (every drafted player is on exactly one roster per draft).

- **overvalues men**, k = 0.5: Victoria Dimuzio $30k (+21pp), Tyra Hurricane Black $386k (+21pp), Oscar Serra $92k (+11pp), Matthew Barlow $48k (+11pp), Jaume Martinez Vich $100k (+11pp), Hunter Johnson $137k (+11pp), Meghan Dizon $171k (+11pp), Ewa Radzikowska $176k (+11pp)
- **overvalues men**, k = 1: Roos Van Reek $96k (+21pp), Carlota Trevino $101k (+21pp), Christopher Haworth $95k (+16pp), Liz Truluck $68k (+12pp), Keilly Ulery $30k (+12pp), Maggie Brascia $81k (+12pp), Jaume Martinez Vich $100k (+11pp), Jillian Braverman $149k (+11pp)
- **overvalues women**, k = 0.5: Lea Jansen $117k (+21pp), Tyra Hurricane Black $386k (+21pp), Brooke Buckner $134k (+16pp), Anderson Scarpa $70k (+12pp), George Wall $83k (+11pp), Oscar Serra $92k (+11pp), Gabriel Joseph $30k (+11pp), Matthew Barlow $48k (+11pp)
- **overvalues women**, k = 1: Lea Jansen $117k (+21pp), Brooke Buckner $134k (+21pp), Tyra Hurricane Black $386k (+21pp), Jorja Johnson $475k (+16pp), Max Freeman $64k (+13pp), Anderson Scarpa $70k (+12pp), Oscar Serra $92k (+11pp), Rafael Lenhard $30k (+11pp)
- **cheapskate $500k**, $500k: Gabriel Joseph $30k (+42pp), Maggie Brascia $81k (+17pp), Ting Chieh Wei $102k (+17pp), Pablo Tellez $71k (+13pp), Hannah Blatt $30k (+12pp), Carlota Trevino $101k (+12pp), Keilly Ulery $30k (+11pp), Brooke Buckner $134k (+11pp)
- **marketing guy (big names)**, k = 0.5: Jorja Johnson $475k (+21pp), Victoria Dimuzio $30k (+16pp), Alexander Crum $30k (+16pp), Matthew Barlow $48k (+16pp), Tyra Hurricane Black $386k (+16pp), Pablo Tellez $71k (+12pp), Rika Fujiwara $30k (+11pp), Rafael Lenhard $30k (+11pp)
- **marketing guy (big names)**, k = 1: Jorja Johnson $475k (+21pp), Victoria Dimuzio $30k (+16pp), Etta Tuionetoa $187k (+16pp), Rafael Lenhard $30k (+11pp), Alexander Crum $30k (+11pp), Matthew Barlow $48k (+11pp), Armaan Bhatia $130k (+11pp), Alix Truong $132k (+11pp)
- **wants real teams**, lam = 0.05: Victoria Dimuzio $30k (+21pp), Layne Sleeth $147k (+16pp), Camden Chaffin $30k (+12pp), Matthew Barlow $48k (+11pp), Ewa Radzikowska $176k (+11pp), Jay Devilliers $288k (+11pp), Federico Staksrud $315k (+11pp), JW Johnson $430k (+11pp)
- **wants real teams**, lam = 0.15: Camden Chaffin $30k (+17pp), Victoria Dimuzio $30k (+16pp), Jay Devilliers $288k (+16pp), Marcela Aguila Ampon $30k (+15pp), Alexa Schull $30k (+13pp), George Wall $83k (+11pp), Matthew Barlow $48k (+11pp), Layne Sleeth $147k (+11pp)
- **wants real teams**, lam = 0.5: Brooke Buckner $134k (+26pp), Camden Chaffin $30k (+22pp), Alix Truong $132k (+21pp), Ivan Jakovljevic $30k (+20pp), Wyatt Stone $30k (+19pp), Daria Walczak $79k (+18pp), Jessie Irvine $89k (+16pp), Jonathan Truong $30k (+14pp)
- **bargains first**, $120k: Lea Jansen $117k (+74pp), Christopher Haworth $95k (+53pp), Carlota Trevino $101k (+32pp), Roos Van Reek $96k (+32pp), Roscoe Bellamy $107k (+32pp), Oscar Serra $92k (+21pp), Ting Chieh Wei $102k (+21pp), Yuta Funemizu $119k (+21pp)
- **bargains first**, $250k: Jack Sock $226k (+63pp), Phuc Huynh $210k (+32pp), Bobbi Oshiro $247k (+26pp), Rafa Hewett $52k (+23pp), Maggie Brascia $81k (+21pp), Sahra Dennehy $203k (+21pp), Allyce Jones $154k (+16pp), Yufei Long $79k (+11pp)

## Who can actually win: title-odds concentration

Per draft, every team's title odds from the season sim, then averaged over the drafts. 'Effective contenders' = 1 / sum(title odds squared): 20 means twenty equal teams, 1 means one certain champion. Gini is over the twenty teams' title odds (0 = parity). Contenders = teams with at least a 10% (or 5%) title shot. 'Runner-up favourite' = the title odds of the second-best team, i.e. how real the chase is. For scale, a bookmaker's pre-season English Premier League board (favourite ~55%, two challengers at ~20% and ~15%) is about 2.7 effective contenders, 3 teams at 10%+.

| persona | strength | how many | favourite | runner-up favourite | third | gap 1st-2nd | teams >= 10% | teams >= 5% | effective contenders | Gini |
|---|---|---|---|---|---|---|---|---|---|---|
| quant baseline |  | -- | 36.2% | 7.3% | 6.3% | 28.9 pts | 1.0 | 4.5 | 6.4 | 0.51 |
| overvalues men | k = 0.5 | 1 | 36.0% | 8.4% | 7.0% | 27.6 pts | 1.1 | 4.3 | 6.3 | 0.52 |
| overvalues men | k = 0.5 | 5 | 36.1% | 8.1% | 6.9% | 28.0 pts | 1.1 | 4.6 | 6.3 | 0.52 |
| overvalues men | k = 0.5 | 20 | 36.2% | 8.3% | 6.9% | 27.9 pts | 1.1 | 4.7 | 6.3 | 0.53 |
| overvalues men | k = 1 | 1 | 35.6% | 9.1% | 6.9% | 26.5 pts | 1.1 | 4.8 | 6.4 | 0.52 |
| overvalues men | k = 1 | 5 | 35.0% | 8.4% | 6.6% | 26.5 pts | 1.1 | 4.8 | 6.6 | 0.52 |
| overvalues men | k = 1 | 20 | 35.0% | 8.0% | 7.0% | 27.0 pts | 1.1 | 5.4 | 6.5 | 0.53 |
| overvalues women | k = 0.5 | 1 | 35.1% | 8.4% | 6.7% | 26.8 pts | 1.1 | 4.8 | 6.5 | 0.52 |
| overvalues women | k = 0.5 | 5 | 34.7% | 8.4% | 7.1% | 26.3 pts | 1.1 | 5.2 | 6.6 | 0.53 |
| overvalues women | k = 0.5 | 20 | 34.2% | 8.8% | 6.9% | 25.4 pts | 1.1 | 4.7 | 6.8 | 0.52 |
| overvalues women | k = 1 | 1 | 35.3% | 8.4% | 6.9% | 26.9 pts | 1.1 | 5.0 | 6.5 | 0.52 |
| overvalues women | k = 1 | 5 | 36.2% | 8.2% | 6.8% | 28.0 pts | 1.0 | 4.8 | 6.3 | 0.53 |
| overvalues women | k = 1 | 20 | 33.0% | 8.5% | 6.8% | 24.5 pts | 1.1 | 5.2 | 7.2 | 0.50 |
| cheapskate $500k | $500k | 1 | 32.2% | 7.8% | 6.7% | 24.4 pts | 1.0 | 4.6 | 7.4 | 0.48 |
| cheapskate $500k | $500k | 5 | 31.4% | 8.6% | 7.6% | 22.8 pts | 1.0 | 7.0 | 7.3 | 0.54 |
| cheapskate $500k | $500k | 20 | 11.6% | 9.9% | 8.8% | 1.7 pts | 1.1 | 8.9 | 14.2 | 0.36 |
| marketing guy (big names) | k = 0.5 | 1 | 34.7% | 8.0% | 6.8% | 26.7 pts | 1.1 | 4.4 | 6.7 | 0.50 |
| marketing guy (big names) | k = 0.5 | 5 | 36.7% | 7.6% | 6.7% | 29.0 pts | 1.1 | 4.5 | 6.2 | 0.52 |
| marketing guy (big names) | k = 0.5 | 20 | 36.9% | 7.9% | 6.4% | 29.0 pts | 1.1 | 4.3 | 6.1 | 0.52 |
| marketing guy (big names) | k = 1 | 1 | 36.0% | 7.7% | 6.7% | 28.3 pts | 1.1 | 4.5 | 6.4 | 0.51 |
| marketing guy (big names) | k = 1 | 5 | 35.7% | 8.0% | 6.7% | 27.7 pts | 1.1 | 4.5 | 6.5 | 0.51 |
| marketing guy (big names) | k = 1 | 20 | 36.1% | 7.9% | 6.6% | 28.2 pts | 1.1 | 4.8 | 6.4 | 0.51 |
| wants real teams | lam = 0.05 | 1 | 35.2% | 8.2% | 6.8% | 27.0 pts | 1.1 | 4.3 | 6.6 | 0.51 |
| wants real teams | lam = 0.05 | 5 | 36.0% | 8.4% | 7.0% | 27.6 pts | 1.2 | 4.9 | 6.3 | 0.53 |
| wants real teams | lam = 0.05 | 20 | 35.7% | 8.1% | 6.9% | 27.7 pts | 1.1 | 4.7 | 6.4 | 0.52 |
| wants real teams | lam = 0.15 | 1 | 34.9% | 8.3% | 6.9% | 26.6 pts | 1.1 | 4.4 | 6.6 | 0.51 |
| wants real teams | lam = 0.15 | 5 | 35.4% | 8.5% | 7.1% | 26.9 pts | 1.1 | 4.5 | 6.5 | 0.53 |
| wants real teams | lam = 0.15 | 20 | 33.5% | 9.0% | 7.1% | 24.5 pts | 1.3 | 5.2 | 7.0 | 0.51 |
| wants real teams | lam = 0.5 | 1 | 34.8% | 8.3% | 6.9% | 26.5 pts | 1.1 | 4.8 | 6.6 | 0.52 |
| wants real teams | lam = 0.5 | 5 | 33.2% | 8.5% | 7.0% | 24.7 pts | 1.1 | 5.5 | 7.0 | 0.53 |
| wants real teams | lam = 0.5 | 20 | 27.7% | 9.9% | 8.7% | 17.8 pts | 1.4 | 6.8 | 8.3 | 0.53 |
| bargains first | $120k | 1 | 32.7% | 7.3% | 6.3% | 25.4 pts | 1.0 | 5.2 | 7.3 | 0.48 |
| bargains first | $120k | 5 | 34.6% | 8.2% | 6.8% | 26.4 pts | 1.0 | 5.7 | 6.6 | 0.55 |
| bargains first | $120k | 20 | 35.3% | 8.8% | 7.1% | 26.5 pts | 1.2 | 4.8 | 6.6 | 0.53 |
| bargains first | $250k | 1 | 34.4% | 8.4% | 7.0% | 26.0 pts | 1.2 | 5.1 | 6.7 | 0.52 |
| bargains first | $250k | 5 | 33.6% | 8.1% | 6.9% | 25.5 pts | 1.1 | 5.4 | 7.0 | 0.53 |
| bargains first | $250k | 20 | 15.2% | 12.0% | 10.0% | 3.2 pts | 2.5 | 7.7 | 12.1 | 0.43 |

## What this says (hand-written against the seed-1 grid; re-check the numbers above if the grid is re-run)

- **Overvaluing one gender costs nothing.** Stretch the men's or women's gaps to double their real size and the persona's teams still win 47-50% with a normal title shot; the quants around them do not move. The price list already carries the ranking, so a lopsided belief about which gender matters changes a pick or two at the margin, not the roster.
- **The marketing owner comes out slightly AHEAD (51-52% win, ~6% title).** At these prices the big names are fairly priced, so preferring them is free -- and the fame table is built from real doubles rank, so a fame bias is partly a bias toward the truth. Read it as "chasing names at fair prices does not hurt you", not as "fame beats analysis".
- **The $500k cheapskate is the persona that breaks the league.** Alone, the team wins 21% with no title shot. Five of them push the parity spread from 4.4 to 13.7 points and hand the other fifteen a 58% win rate; all twenty, and Waters, Johns, Bright and the top of the list are never drafted (no one can afford them). This is the case for the $500k min-spend rule: it is not about fairness to the cheap team, it is that unspent money makes the whole league worse.
- **Loyalty is cheap in small doses and expensive in large ones.** A light preference for 2026 teammates (lam 0.05) is free; at lam 0.15 it costs ~1.5 points of win rate; at lam 0.5 (a full real six worth half a win of belief) the team drops to 42% / 2% title and leaves $110k unspent because the teammates it wants do not fill a legal roster efficiently.
- **Bargains first is the worst strategy that looks sensible.** Spending rounds 1-3 on <=$120k players wins 24% with no title shot, and only $600k gets spent: in a 20-team snake the whole top 60 is gone before round 4, so the money has nothing left to buy. Raising the threshold to $250k gets 39%. The lesson is that in a draft, waiting is the expensive move -- the stars are gone, not overpriced.
- **Nothing here dents the pick-1 team.** Waters' team wins 63-68% in every cell it is drafted; the only cells that move it are the ones where personas leave the league lopsided (five cheapskates 68.5%).

