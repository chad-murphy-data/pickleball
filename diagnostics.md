# diagnostics.md — GO/NO-GO gate for the SRM dataset

Dataset: `data/games.csv` — **10003 games** (9984 after excluding 19 forfeit-tainted), **1477 players**, 821 MLP / 9163 PPA. DreamBreakers are in `data/dreambreakers.csv`, never here.

## Data-quality overview

Scoring formats present (modeling rows):

| tour | scoring_format | best_of | games |
|:--|:--|--:|--:|
| MLP | sideout_11 | 1 | 821 |
| PPA | sideout_11 | 3 | 7948 |
| PPA | sideout_11 | 5 | 106 |
| PPA | sideout_15 | 1 | 50 |
| PPA | unknown | 1 | 1059 |

Dropped rows: **311** (full detail in `data/dropped.csv`):

| reason | rows |
|:--|--:|
| no played games | 157 |
| match not completed | 106 |
| bye matchup | 30 |
| missing player uuid | 18 |

Flagged-but-kept rows: **154** (full detail in `data/flags.csv`):

| flag | rows |
|:--|--:|
| winning score 16 with unresolved format | 51 |
| mid-match player swap recorded — listed lineup may not have  | 46 |
| winning score 17 with unresolved format | 17 |
| winning score 18 with unresolved format | 7 |
| winning score 19 with unresolved format | 2 |
| winning score 20 with unresolved format | 2 |
| MLP fixture 01fb2c30-9d6e-4981-9c31-d2292d13bbce has 3 non-D | 1 |
| MLP fixture 2b659118-40b2-49b3-a4c0-c02f8445fafd has 3 non-D | 1 |
| MLP fixture 3851e52a-c9f1-4fa4-a1fb-d76d581dd7b8 has 3 non-D | 1 |
| MLP fixture 3b5221c5-8478-49c8-919e-6b79730afe13 has 3 non-D | 1 |
| MLP fixture 3d336b4d-3b58-4c66-9ca2-4fd904a6c224 has 3 non-D | 1 |
| MLP fixture 43e8a929-4585-4e09-84bf-69ffe4acf893 has 3 non-D | 1 |

## (a) Dyad game-count distribution

Distinct dyads: **2709** across all contexts (a dyad = unordered player pair that appeared on the same side of the net).

| games together | dyads | |
|---:|---:|:---|
|       1 |    66 | █ |
|     2–4 |  1134 | ████████████████████████ |
|     5–9 |   969 | █████████████████████ |
|   10–14 |   297 | ██████ |
|   15–24 |   152 | ███ |
|   25–49 |    71 | ██ |
|     50+ |    20 | █ |

Median games per dyad: **5**. Dyads with ≥10 games: **540**; with ≥15: **243**.

### Focal dyads (games together, all contexts)

| dyad | games | contexts |
|:--|--:|:--|
| Anna Bright + Hayden Patriquin | 138 | mixed |
| Ben Johns + Gabriel Tardio | 113 | mens |
| Anna Leigh Waters + Ben Johns | 111 | mixed |
| Andrei Daescu + Federico Staksrud | 101 | mens |
| Anna Bright + Anna Leigh Waters | 98 | womens |
| JW Johnson + Jorja Johnson | 95 | mixed |
| Christian Alshon + Hayden Patriquin | 95 | mens |
| Jorja Johnson + Tyra Hurricane Black | 83 | womens |
| Catherine Parenteau + Gabriel Tardio | 79 | mixed |
| Federico Staksrud + Kate Fahey | 78 | mixed |
| Noe Khlif + Tina Pisnik | 46 | mixed |
| Noe Khlif + Will Howells | 41 | mens |
| Jackie Kawamoto + Jade Kawamoto | 29 | womens |
| Catherine Parenteau + Jade Kawamoto | 29 | womens |
| Kate Fahey + Parris Todd | 28 | womens |
| Gabriel Tardio + Hayden Patriquin | 25 | mens |
| Anna Bright + Kate Fahey | 25 | womens |
| Anna Leigh Waters + Jorja Johnson | 25 | womens |
| Anna Leigh Waters + Noe Khlif | 25 | mixed |
| Parris Todd + Tyra Hurricane Black | 22 | womens |
| Christian Alshon + Tyra Hurricane Black | 21 | mixed |
| Andrei Daescu + Tyra Hurricane Black | 21 | mixed |
| Jorja Johnson + Will Howells | 21 | mixed |
| Ben Johns + Max Freeman | 20 | mens |
| Ben Johns + Jade Kawamoto | 19 | mixed |
| Gabriel Tardio + Kate Fahey | 19 | mixed |
| Federico Staksrud + Milan Rane | 18 | mixed |
| Noe Khlif + Tyra Hurricane Black | 17 | mixed |
| Andrei Daescu + Gabriel Tardio | 15 | mens |
| Federico Staksrud + Jack Sock | 15 | mens |
| Gabriel Tardio + Rachel Rohrabacher | 14 | mixed |
| Eric Oncins + Tyra Hurricane Black | 14 | mixed |
| CJ Klinger + Federico Staksrud | 13 | mens |
| Kate Fahey + Lacy Schneemann | 11 | womens |
| Brooke Buckner + Kate Fahey | 11 | womens |
| Noe Khlif + Rafa Hewett | 10 | mens |
| Meghan Dizon + Noe Khlif | 9 | mixed |
| Rachel Rohrabacher + Will Howells | 8 | mixed |
| Andrei Daescu + Jade Kawamoto | 8 | mixed |
| Noe Khlif + Pablo Tellez | 8 | mens |

## (b) Distinct partners per focal player

| player | games | partners (all) | mixed | mens | womens |
|:--|--:|--:|--:|--:|--:|
| Ben Johns | 263 | 4 | 2 | 2 | 0 |
| Anna Leigh Waters | 259 | 4 | 2 | 0 | 2 |
| Anna Bright | 261 | 3 | 1 | 0 | 2 |
| Hayden Patriquin | 259 | 4 | 2 | 2 | 0 |
| Gabriel Tardio | 265 | 6 | 3 | 3 | 0 |
| Federico Staksrud | 240 | 8 | 4 | 4 | 0 |
| Jade Kawamoto | 94 | 6 | 4 | 0 | 2 |
| Jorja Johnson | 224 | 4 | 2 | 0 | 2 |
| Will Howells | 87 | 6 | 4 | 2 | 0 |
| Noe Khlif | 190 | 13 | 6 | 7 | 0 |
| Kate Fahey | 186 | 9 | 3 | 0 | 6 |
| Tyra Hurricane Black | 214 | 13 | 10 | 0 | 3 |

**Focal-name resolution notes (human should confirm):**
- ⚠️ "Etta Fahey" not found; last-name fallback picked **Kate Fahey** (`3af01041-5679-400a-87b8-12c01db8cec4`) — verify this is the intended person.
- "Tyra Black" resolved to **Tyra Hurricane Black** (`fec65f7e-f2c8-482f-ab5b-541cf53b1064`) by token match.

## (c) Partnership-graph connectivity, per context

Edges = played together. (Supplementary: components when opponent edges are added too, since margins also link the two sides.)

| context | players | components (partner edges) | component sizes | components (+opponent edges) |
|:--|--:|--:|:--|--:|
| mixed | 935 | 162 | 575, 7, 6, 6, 5, 4, 4, 4… | 1 (935) |
| mens | 899 | 177 | 503, 6, 6, 5, 5, 5, 4, 4… | 1 (899) |
| womens | 421 | 74 | 253, 5, 5, 4, 4, 4, 3, 3… | 1 (421) |

Component membership of focal players (size of their component, partner-edge graph):

| context | Ben Johns | Anna Leigh Waters | Anna Bright | Hayden Patriquin | Gabriel Tardio | Federico Staksrud | Jade Kawamoto | Jorja Johnson | Will Howells | Noe Khlif | Etta Fahey | Tyra Black |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| mixed | 575 | 575 | 575 | 575 | 575 | 575 | 575 | 575 | 575 | 575 | 575 | 575 |
| mens | 503 | — | — | 503 | 503 | 503 | — | — | 503 | 503 | — | — |
| womens | — | 253 | 253 | — | — | — | 253 | 253 | — | — | 253 | 253 |

## (d) Margin distributions

| slice | n games | mean margin | SD | mean \|margin\| |
|:--|--:|--:|--:|--:|
| MLP | 821 | +0.61 | 6.22 | 5.62 |
| PPA | 9163 | +2.64 | 6.03 | 5.93 |
| PPA game 1 | 4614 | +2.66 | 6.21 | 6.05 |
| PPA game 2 | 3505 | +2.99 | 5.81 | 5.93 |
| PPA game 3 | 1028 | +1.42 | 5.87 | 5.44 |
| PPA game 4 | 11 | +3.18 | 4.33 | 4.64 |
| PPA game 5 | 5 | +3.20 | 3.96 | 4.40 |

Margin histogram (absolute margin, modeling rows):


**MLP**

| \|margin\| | games | |
|---:|---:|:---|
|       2 |   143 | ███████████████████ |
|     3–4 |   183 | ████████████████████████ |
|     5–6 |   178 | ███████████████████████ |
|     7–8 |   165 | ██████████████████████ |
|    9–10 |   120 | ████████████████ |
|     11+ |    32 | ████ |

**PPA**

| \|margin\| | games | |
|---:|---:|:---|
|       2 |  1535 | ███████████████████ |
|     3–4 |  1744 | █████████████████████ |
|     5–6 |  1940 | ███████████████████████ |
|     7–8 |  1986 | ████████████████████████ |
|    9–10 |  1427 | █████████████████ |
|     11+ |   531 | ██████ |

## Interpretation & verdict

- **(a)** 36 focal dyads have ≥10 games; 28 are under 10 (heavy shrinkage for those).
- **(b)** every resolved focal player has ≥2 distinct partners — actor/partner effects separable.
- **(c/mixed)** giant component holds 575/935 players (61%); focal players all inside it.
- **(c/mens)** giant component holds 503/899 players (56%); focal players all inside it.
- **(c/womens)** giant component holds 253/421 players (60%); focal players all inside it.
- **(d)** PPA mean |margin|: games 1–2 = 6.00, game 3 = 5.41 → compression confirmed (game 3 exists only after a 1–1 split), consistent with real, correctly-labeled data.
- **(d)** margin SD: MLP 6.22 vs PPA 6.03 (3% relative gap) — comparable across tours, as expected for same-format side-out-to-11 games.
