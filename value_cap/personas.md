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
| cheapskate $500k | $500k | 20 | 50.0% | 5.0% | $483k | -- | -- | 4.3 pts | nan% | nan% | nan% | Anna Leigh Waters #1F, Ben Johns #1M, JW Johnson #2M, Anna Bright #2F (+51) |
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
| bargains first | $250k | 20 | 50.0% | 5.0% | $921k | -- | -- | 4.2 pts | nan% | 56.4% | 52.5% | Anna Leigh Waters #1F, Gabriel Tardio #4M, Andrei Daescu #6M, Federico Staksrud #7M (+7) |

## Who each persona drafts

Most-drafted players per persona (share of that persona's rosters that carried them), one row per strength at the k = 1 cell (the persona among 19 quants) and the all-20 cell.

- **overvalues men**, k = 0.5, x1: Victoria Dimuzio $30k (25%), Tyra Hurricane Black $386k (25%), Matthew Barlow $48k (15%), Oscar Serra $92k (15%), Jaume Martinez Vich $100k (15%), Hunter Johnson $137k (15%), Meghan Dizon $171k (15%), Ewa Radzikowska $176k (15%)
- **overvalues men**, k = 0.5, x20: Lina Padegimaite $30k (5%), Gabriel Joseph $30k (5%), Irina Tereschenko $30k (5%), Rafael Lenhard $30k (5%), Alexander Crum $30k (5%), Grayson Goldin $44k (5%), Hoang Nam Ly $44k (5%), Matthew Barlow $48k (5%)
- **overvalues men**, k = 1, x1: Roos Van Reek $96k (25%), Carlota Trevino $101k (25%), Christopher Haworth $95k (20%), Keilly Ulery $30k (15%), Liz Truluck $68k (15%), Maggie Brascia $81k (15%), Jaume Martinez Vich $100k (15%), Jillian Braverman $149k (15%)
- **overvalues men**, k = 1, x20: Gabriel Joseph $30k (5%), Lina Padegimaite $30k (5%), Victoria Dimuzio $30k (5%), Rafael Lenhard $30k (5%), Alexander Crum $30k (5%), Grayson Goldin $44k (5%), Hoang Nam Ly $44k (5%), Matthew Barlow $48k (5%)
- **overvalues women**, k = 0.5, x1: Lea Jansen $117k (25%), Tyra Hurricane Black $386k (25%), Brooke Buckner $134k (20%), Gabriel Joseph $30k (15%), Matthew Barlow $48k (15%), Anderson Scarpa $70k (15%), George Wall $83k (15%), Kaitlyn Christian $87k (15%)
- **overvalues women**, k = 0.5, x20: Alexander Crum $30k (5%), Rafael Lenhard $30k (5%), Victoria Dimuzio $30k (5%), Lina Padegimaite $30k (5%), Gabriel Joseph $30k (5%), Yates Johnson $30k (5%), Irina Tereschenko $30k (5%), Grayson Goldin $44k (5%)
- **overvalues women**, k = 1, x1: Lea Jansen $117k (25%), Brooke Buckner $134k (25%), Tyra Hurricane Black $386k (25%), Jorja Johnson $475k (20%), Rafael Lenhard $30k (15%), Oliver Frank $51k (15%), Max Freeman $64k (15%), Katerina Stewart $65k (15%)
- **overvalues women**, k = 1, x20: Alexander Crum $30k (5%), Gabriel Joseph $30k (5%), Lina Padegimaite $30k (5%), Victoria Dimuzio $30k (5%), Rafael Lenhard $30k (5%), Irina Tereschenko $30k (5%), Yates Johnson $30k (5%), Grayson Goldin $44k (5%)
- **cheapskate $500k**, $500k, x1: Gabriel Joseph $30k (45%), Maggie Brascia $81k (20%), Ting Chieh Wei $102k (20%), Keilly Ulery $30k (15%), Victoria Dimuzio $30k (15%), Hannah Blatt $30k (15%), Matthew Barlow $48k (15%), Pablo Tellez $71k (15%)
- **cheapskate $500k**, $500k, x20: Michael Loyd $30k (5%), Naomi Nguyen $30k (5%), Hannah Blatt $30k (5%), Irina Tereschenko $30k (5%), Lina Padegimaite $30k (5%), Camden Chaffin $30k (5%), Ella Yeh $30k (5%), Victoria Dimuzio $30k (5%)
- **marketing guy (big names)**, k = 0.5, x1: Jorja Johnson $475k (25%), Victoria Dimuzio $30k (20%), Alexander Crum $30k (20%), Matthew Barlow $48k (20%), Tyra Hurricane Black $386k (20%), Rafael Lenhard $30k (15%), Rika Fujiwara $30k (15%), Pablo Tellez $71k (15%)
- **marketing guy (big names)**, k = 0.5, x20: Alexander Crum $30k (5%), Lina Padegimaite $30k (5%), Gabriel Joseph $30k (5%), Victoria Dimuzio $30k (5%), Rafael Lenhard $30k (5%), Yates Johnson $30k (5%), Grayson Goldin $44k (5%), Hoang Nam Ly $44k (5%)
- **marketing guy (big names)**, k = 1, x1: Jorja Johnson $475k (25%), Victoria Dimuzio $30k (20%), Etta Tuionetoa $187k (20%), Rafael Lenhard $30k (15%), Alexander Crum $30k (15%), Matthew Barlow $48k (15%), Armaan Bhatia $130k (15%), Alix Truong $132k (15%)
- **marketing guy (big names)**, k = 1, x20: Alexander Crum $30k (5%), Lina Padegimaite $30k (5%), Ben Cawston $30k (5%), Rafael Lenhard $30k (5%), Gabriel Joseph $30k (5%), Irina Tereschenko $30k (5%), Victoria Dimuzio $30k (5%), Hoang Nam Ly $44k (5%)
- **wants real teams**, lam = 0.05, x1: Victoria Dimuzio $30k (25%), Layne Sleeth $147k (20%), Camden Chaffin $30k (15%), Matthew Barlow $48k (15%), Ewa Radzikowska $176k (15%), Jay Devilliers $288k (15%), Federico Staksrud $315k (15%), JW Johnson $430k (15%)
- **wants real teams**, lam = 0.05, x20: Gabriel Joseph $30k (5%), Lina Padegimaite $30k (5%), Rafael Lenhard $30k (5%), Alexander Crum $30k (5%), Grayson Goldin $44k (5%), Matthew Barlow $48k (5%), Oliver Frank $51k (5%), Zane Ford $53k (5%)
- **wants real teams**, lam = 0.15, x1: Camden Chaffin $30k (20%), Victoria Dimuzio $30k (20%), Jay Devilliers $288k (20%), Alexa Schull $30k (15%), Marcela Aguila Ampon $30k (15%), Matthew Barlow $48k (15%), George Wall $83k (15%), Layne Sleeth $147k (15%)
- **wants real teams**, lam = 0.15, x20: Alexander Crum $30k (5%), Blaine Hovenier $30k (5%), Rafael Lenhard $30k (5%), Gabriel Joseph $30k (5%), Grayson Goldin $44k (5%), Matthew Barlow $48k (5%), Seone Mendez $64k (5%), Katerina Stewart $65k (5%)
- **wants real teams**, lam = 0.5, x1: Brooke Buckner $134k (30%), Camden Chaffin $30k (25%), Alix Truong $132k (25%), Wyatt Stone $30k (20%), Ivan Jakovljevic $30k (20%), Daria Walczak $79k (20%), Jessie Irvine $89k (20%), Jonathan Truong $30k (15%)
- **wants real teams**, lam = 0.5, x20: Alexa Schull $30k (5%), Rafael Lenhard $30k (5%), Victoria Dimuzio $30k (5%), Gabriel Joseph $30k (5%), Alexander Crum $30k (5%), Jonathan Truong $30k (5%), Matthew Barlow $48k (5%), Donald Young $62k (5%)
- **bargains first**, $120k, x1: Lea Jansen $117k (75%), Christopher Haworth $95k (55%), Roos Van Reek $96k (35%), Carlota Trevino $101k (35%), Roscoe Bellamy $107k (35%), Oscar Serra $92k (25%), Ting Chieh Wei $102k (25%), Yuta Funemizu $119k (25%)
- **bargains first**, $120k, x20: Ella Yeh $30k (5%), Jalina Ingram $30k (5%), Naomi Nguyen $30k (5%), Hannah Blatt $30k (5%), Elsie Hendershot $30k (5%), Gabriel Joseph $30k (5%), Irina Tereschenko $30k (5%), Tamaryn Emmrich $30k (5%)
- **bargains first**, $250k, x1: Jack Sock $226k (65%), Phuc Huynh $210k (35%), Bobbi Oshiro $247k (30%), Rafa Hewett $52k (25%), Maggie Brascia $81k (25%), Sahra Dennehy $203k (25%), Allyce Jones $154k (20%), Kiora Kunimoto $68k (15%)
- **bargains first**, $250k, x20: Yates Johnson $30k (5%), Gabriel Joseph $30k (5%), Rafael Lenhard $30k (5%), Victoria Dimuzio $30k (5%), Lina Padegimaite $30k (5%), Alexander Crum $30k (5%), Grayson Goldin $44k (5%), Hoang Nam Ly $44k (5%)
