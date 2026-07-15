# diagnostics.md — GO/NO-GO gate for the SRM dataset

Dataset: `data/games.csv` — **10003 games** (9984 after excluding 19 forfeit-tainted), **1477 players**, 821 MLP / 9163 PPA. DreamBreakers are in `data/dreambreakers.csv`, never here.

## Data-quality overview

Scoring formats present (modeling rows):

| tour | scoring_format | best_of | games |
|:--|:--|--:|--:|
| MLP | sideout_11 | 1 | 821 |
| PPA | sideout_11 | 3 | 7948 |
| PPA | sideout_11 | 5 | 106 |
| PPA | sideout_15 | 1 | 1109 |

Dropped rows: **311** (full detail in `data/dropped.csv`):

| reason | rows |
|:--|--:|
| no played games | 157 |
| match not completed | 106 |
| bye matchup | 30 |
| bracket bye/shell | 18 |

Flagged-but-kept rows: **73** (full detail in `data/flags.csv`):

| flag | rows |
|:--|--:|
| mid-match player swap recorded — listed lineup may not have played the | 46 |
| MLP fixture <uuid> has 3 non-DB games | 27 |

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

Computed on `sideout_11` rows only (8875 games; 1109 games in other formats excluded from this comparison — see format table above).

| slice | n games | mean margin | SD | mean \|margin\| |
|:--|--:|--:|--:|--:|
| MLP | 821 | +0.61 | 6.22 | 5.62 |
| PPA | 8054 | +2.79 | 5.79 | 5.81 |
| PPA game 1 | 3505 | +2.98 | 5.70 | 5.81 |
| PPA game 2 | 3505 | +2.99 | 5.81 | 5.93 |
| PPA game 3 | 1028 | +1.42 | 5.87 | 5.44 |
| PPA game 4 | 11 | +3.18 | 4.33 | 4.64 |
| PPA game 5 | 5 | +3.20 | 3.96 | 4.40 |

Margin histogram (absolute margin, sideout_11 modeling rows):


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
|       2 |  1357 | ██████████████████ |
|     3–4 |  1567 | █████████████████████ |
|     5–6 |  1748 | ███████████████████████ |
|     7–8 |  1801 | ████████████████████████ |
|    9–10 |  1238 | ████████████████ |
|     11+ |   343 | █████ |

## Interpretation & verdict

- **(a)** 36 focal dyads have ≥10 games; 28 are under 10 (heavy shrinkage for those).
- **(b)** every resolved focal player has ≥2 distinct partners — actor/partner effects separable.
- **(b-caveat)** single-partner *within a context*: Anna Bright in mixed (138 games, only partner: Hayden Patriquin). Within that context alone, actor and dyad effects for these players are confounded — separation leans on their play in other contexts (i.e., on the pooled-SRM structure).
- **(c/mixed)** giant component holds 575/935 players (61%); focal players all inside it.
- **(c/mens)** giant component holds 503/899 players (56%); focal players all inside it.
- **(c/womens)** giant component holds 253/421 players (60%); focal players all inside it.
- **(d)** PPA mean |margin|: games 1–2 = 5.87, game 3 = 5.41 → compression confirmed (game 3 exists only after a 1–1 split), consistent with real, correctly-labeled data.
- **(MLP structure)** 27 fixtures have only 3 doubles games: in every case a 3-0 sweep where the dead 4th game (MXD2) was skipped. The handoff's "all four games are always played" does NOT hold in 2026 — but the games that are played are all live, so no garbage-time filtering is needed.
- **(d-note)** PPA mean *signed* margin is +2.79: team one is not random (bracket convention favors the higher seed / qualifier winner), so signed margins are not zero-centered. Not a bug — but don't interpret raw team-one margins as symmetric.
- **(d)** margin SD: MLP 6.22 vs PPA 5.79 (7% relative gap) — comparable across tours, as expected for same-format side-out-to-11 games.
