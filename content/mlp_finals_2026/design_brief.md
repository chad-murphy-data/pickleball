# Design brief — MLP Championship Sunday: The 5s' pick

**Concept:** New Jersey finished pool play on top and gets to CHOOSE their
semifinal opponent — Dallas, Brooklyn, or St. Louis. We priced all three
doors, plus the whole bracket behind each one. The pick is worth 12 points
of championship probability, and it's not the door most fans would argue
about.

Two hero elements: (1) the **three doors** (NJ's title odds under each
pick), (2) the **NJ–STL coin flip** (the final everyone wants, priced game
by game). Everything else is supporting.

All numbers are calibrated v2 model output, priced on the lineups below
(NJ with Khlif in men's — see the lineup note). Baseline "as-run" variant
at the bottom if the 5s surprise us.

---

## HERO 1 — the three doors

*P(New Jersey wins the title) under each possible pick:*

| pick | P(win semi) | P(win final \| reach it) | **P(NJ TITLE)** | normalized¹ |
|---|---|---|---|---|
| **Dallas Flash** | 96% | 61% | **58%** | 100 |
| St. Louis Shock | 56% | 83% | 46% | 80 |
| Brooklyn Pickleball Team | 80% | 57% | 46% | 79 |

¹ *Normalized to the Dallas pick = 100, for bar lengths.*

**Point for the designer:** the surprise is the TIE for second. "Grab the
Shock early" (a 56% semi, then an easy final) and "duck the Shock as long
as possible" (an 80% semi, then probably the Shock anyway) price out
IDENTICALLY — 46.2% vs 45.9%, inside the noise. The pick was never about
St. Louis. It's about who gets the free pass against Dallas: if NJ doesn't
take it, St. Louis does (96.6%, the biggest favorite of any possible
matchup), and NJ has handed its rival the easiest road for nothing.

One-line version: **Take the bye. Don't gift it to the Shock.**

---

## HERO 2 — the final everyone wants: NJ 5s vs STL Shock

> **55.8% New Jersey — the closest big matchup the model can build, and
> it goes to a DreamBreaker 44% of the time.**

Game-by-game (P is NJ's, race to 11):

| game | NJ pair | STL pair | P(NJ) | modal score |
|---|---|---|---|---|
| WD | Waters / Jorja Johnson | Bright / Fahey | **86%** | 11-6 NJ |
| MD | Staksrud / Khlif | Tardio / Patriquin | **17%** | 11-6 STL |
| MXD1 | Waters / Khlif | Bright / Patriquin | **55%** | 11-9 NJ |
| MXD2 | Jorja Johnson / Staksrud | Fahey / Tardio | **41%** | 11-8 STL |
| **DB** (if 2-2) | — | — | **64%** | rally to 21 |

Shape of the story: each side owns one same-gender game outright (Waters'
women's doubles vs Tardio/Patriquin's men's — 86% and 83%, near mirror
images), both mixed games are jump balls, so it lands 2-2 nearly half the
time — and then Waters (2.27) and Staksrud (1.86), two of the best singles
players alive, make NJ a 64% DreamBreaker favorite. NJ's edge in this
matchup IS the DreamBreaker.

---

## The full matchup board (supporting grid)

*Every possible Championship Sunday matchup, favorite's calibrated
probability:*

| matchup | favorite | P | P(reaches DB) |
|---|---|---|---|
| STL vs Dallas | STL | **96.6%** | 9% |
| NJ vs Dallas | NJ | **95.6%** | 17% |
| Brooklyn vs Dallas | BKLYN | **84.0%** | 25% |
| NJ vs Brooklyn | NJ | **80.3%** | 35% |
| STL vs Brooklyn | STL | **79.4%** | 28% |
| NJ vs STL | NJ | **55.8%** | 44% |

Dallas is the underdog in all three of its matchups; NJ is the favorite in
all three of its own. The board has a strict pecking order — NJ > STL >
BKLYN > DAL — and the only close game on it is the top two.

---

## Secondary panel — the Shock's rooting guide

*P(St. Louis wins the title), by what NJ does:*

| NJ picks | STL's semi | STL P(title) |
|---|---|---|
| Brooklyn | vs Dallas (96.6%) | **49.4%** ← STL's dream |
| Dallas | vs Brooklyn (79.4%) | 37.0% |
| St. Louis | vs NJ (44.2%) | 36.4% |

If NJ picks Brooklyn, St. Louis becomes the title favorite — the ONLY
scenario where anyone but NJ is. The Shock bench should be openly rooting
for it.

---

## Lineups priced (the four-team card, if wanted)

Values = current v2 rating (expected edge, per-point logit scale).
Ordered strongest roster to weakest.

**New Jersey 5s** — WD: Anna Leigh Waters (1.80) / Jorja Johnson (1.17)
· MD: Federico Staksrud (0.91) / Noe Khlif (0.79) · MXD1: Waters/Khlif
· MXD2: Jorja Johnson/Staksrud. *Lineup note: in pool play NJ ran Will
Howells (0.78) in men's with Khlif mixed-only; this brief prices Khlif in
men's (both play, Howells sits) as the likelier finals lineup. It's ~2
points of title prob — direction of every conclusion unchanged.*

**St. Louis Shock** — WD: Anna Bright (1.33) / Kate Fahey (1.08) · MD:
Gabriel Tardio (1.07) / Hayden Patriquin (1.08) · MXD1: Bright/Patriquin
· MXD2: Fahey/Tardio.

**Brooklyn Pickleball Team** — WD: Rachel Rohrabacher (1.12) / Jackie
Kawamoto (1.09) · MD: Christian Alshon (1.01) / Riley Newman (0.91) ·
MXD1: Rohrabacher/Alshon · MXD2: Kawamoto/Newman.

**Dallas Flash** — WD: Danni-Elle Townsend (1.02) / Alix Truong (0.79) ·
MD: JW Johnson (1.09) / Augustus Ge (0.74) · MXD1: Brooke Buckner (0.73) /
JW Johnson · MXD2: Townsend/Ge.

---

## Copy blocks (ready to set)

- **Eyebrow:** PICKLES · pro pickleball, with receipts
- **Headline:** Three doors. One of them is worth 12 points of a championship.
- **Alt headline (spicier):** Pick Dallas. It was never about the Shock.
- **Subhead / method line:** New Jersey gets to choose its semifinal
  opponent. We priced all three brackets — every game, every lineup, every
  DreamBreaker — with the model that's called 77% of pro pickleball
  winners since June.
- **The honest twist (caption):** Ducking St. Louis doesn't help. Playing
  them early doesn't either — 46% both ways. The entire value of the pick
  is taking the 96% Dallas semi before the Shock gets it.
- **DreamBreaker kicker (caption for Hero 2):** NJ–STL reaches a
  DreamBreaker 44% of the time — and that's where NJ's real edge lives:
  Waters and Staksrud make them 64% in the singles rally race.
- **Footer / source:** v2 Bayesian rating model · every MLP + PPA pro
  game since 2024 · probabilities calibrated, priced 8/17 pre-pick

---

## Guardrails (please honor)

- **Never display 0% or 100%** — house rule, empirically ~1% of 99%
  favorites lose. Largest displayable number here is 96.6%.
- Round to whole percents on display; the tables above carry a decimal
  only where two numbers would otherwise collide (46.2 vs 45.9 — display
  both as 46% and say "tie").
- These are PRE-LINEUP prices. If official finals lineups publish and
  differ (esp. NJ's Howells/Khlif call or Dallas's Buckner mixed look),
  the numbers move a little — don't present the board as final after
  lineups are out without repricing.
- Don't rank men vs women off the values shown — cross-gender comparisons
  are a model convention, not a finding (house rule).
- "Worth 12 points" = 58% vs 46%, the Dallas pick vs either alternative.
  Keep the comparative framing; don't restate as "58% chance NJ wins it
  all" in isolation without the pick attached.

---

## Numbers appendix (full precision, for any layout that needs them)

Matchup trees, P listed for the first-named team; game order WD, MD,
MXD1, MXD2; "2-2" = P(match reaches DreamBreaker); DB = first team's
DreamBreaker win prob.

```
NJ  vs DAL  : 0.956  games 0.981/0.439/0.921/0.761  2-2 0.172  DB 0.823
NJ  vs BKLYN: 0.803  games 0.920/0.330/0.746/0.574  2-2 0.354  DB 0.727
NJ  vs STL  : 0.558  games 0.861/0.165/0.553/0.412  2-2 0.442  DB 0.643
STL vs DAL  : 0.966  games 0.898/0.803/0.908/0.832  2-2 0.086  DB 0.712
STL vs BKLYN: 0.794  games 0.649/0.716/0.715/0.671  2-2 0.277  DB 0.593
DAL vs BKLYN: 0.160  games 0.191/0.401/0.227/0.311  2-2 0.249  DB 0.374

Title tree (NJ pick -> semi x final = title):
  Dallas  : 0.956 x 0.608 = 0.581
  Brooklyn: 0.803 x 0.571 = 0.459
  St.Louis: 0.558 x 0.828 = 0.462

STL title by NJ pick:  Dallas 0.370 · Brooklyn 0.494 · St.Louis 0.364

As-run variant (Howells in MD, Khlif mixed-only):
  NJ vs DAL 0.950 · NJ vs BKLYN 0.788 · NJ vs STL 0.538
  title: Dallas 0.560 · Brooklyn 0.435 · St.Louis 0.438  (same order)
```

Method: race-to-11 DP on v2 values + weakest-link, display calibration,
DreamBreaker from the singles model (K=0.42, imputation for no-record
players); matchup = P(win ≥3 of 4) + P(2-2)·P(DB). Same stack as the
graded Mid-Season receipts. Reproduce: session sim over data/games.csv +
data/mlp_matchups_2026.csv, rosters latest-appearance-wins with
data/roster_overrides.csv applied.
