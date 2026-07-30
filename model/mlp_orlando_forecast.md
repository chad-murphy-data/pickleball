# Amway MLP Orlando — predicted results (design brief)

Generated 2026-07-30, before any ball is struck. Machine-readable payload:
`data/event_forecast.json`. Regenerate with:

```bash
python scraper/mlp_rosters.py                 # refresh official rosters first
python web/make_event_forecast.py --sims 200000
```

Event runs **Thu 30 July – Sat 1 Aug 2026**. 12 teams, two groups of six.
200,000 simulations. Nothing here is a receipt yet — see "If you want this
graded" at the bottom.

> **Roster note (v2 of this brief).** These numbers use the OFFICIAL team
> rosters scraped from majorleaguepickleball.co's team pages (2026-07-30),
> which lead the played-lineup record by a full trade window. The moves
> that matter: **Staksrud and Rane are now New Jersey 5s** (both formerly
> Orlando), **Hunter Johnson is now St. Louis**, **Navratil/Koller/Nelson/
> Hendershot are Chicago**, **Garnett and Parker are LA**, **Ingram is
> California**. Orlando Squeeze — playing at home — lost its two best
> players and collapses from 3rd in Group B to the 9–12 tier as a result.

---

## 1. Format — verified, with one correction to the brief

Your description was right in every structural respect. Verified against the
BFF's own pool/bracket fields for **all eight completed 2026 MLP events**
(Dallas, Columbus, St. Louis, Austin, St. Petersburg, New York, San Diego,
Chicago):

- Teams are split into **two round-robin groups**; every team plays every
  other team in its group once (5 matchups each here).
- Groups are **ranked**, and the playoff is a **single round of rank-vs-rank
  crossovers**: A1 v B1, A2 v B2, A3 v B3, A4 v B4. All eight events
  reproduce exactly under this rule.
- Winners take places **1, 3, 5, 7**; losers take **2, 4, 6, 8**.

**The one correction:** only the **top four** of each group cross over. The
5th- and 6th-place teams play no playoff matchup — every 2026 event has
exactly four playoff matchups, never five or six. So the weekend produces a
clean 1–8 order and then two unordered tiers: the two group-5th teams share
**9th–10th**, the two group-6th teams share **11th–12th**. That's how the
tables below report them, and it's the one place the infographic should show
a tie rather than a rank.

**Group tiebreak ladder** (reverse-engineered; it is what makes all eight
events' playoff pairings come out right):

1. matchup wins
2. head-to-head, for a two-way tie
3. game differential
4. rally-point differential

Two events pin steps 2 and 3 specifically: at Chicago the Slice finished
above the Bouncers on head-to-head despite a worse game differential, and at
Austin a three-way tie at 2–3 resolved on game differential, then points.

Group labels **A/B are ours** (sorted by the pool's internal UUID) — MLP
doesn't publish group names in this feed. Use whatever labels the broadcast
uses if they differ.

---

## 2. What was assumed

- **Rosters are the league's own** (majorleaguepickleball.co team pages,
  fetched 2026-07-30 by `scraper/mlp_rosters.py`; 121/121 players matched to
  our identity space by name, zero ambiguity).
- **Best possible lineups**, as asked: each team's top 2 women + top 2 men by
  current v2 value from that official roster, with the mixed pairs split to
  maximize weakest-link-adjusted strength. No official match lineups have
  been published yet — all 30 matchups are still
  `SCHEDULED_WAITING_FOR_HOME_TEAM_LINEUP`.
- **Lineup deviation is still the largest error source.** Captains routinely
  deviate from the on-paper best four: on day 2 of MLP Chicago, 9 of 10
  matchups ran pairings that differed from projection. New-roster teams
  (Chicago, Carolina, Orlando) also have pairs with no shared MLP history,
  where the model leans hardest on individual values.
- Pricing is the **graded methodology, unchanged**: per-game win probability
  from v2 values + the weakest-link term, race-to-11 DP, display
  calibration; P(matchup) = P(win ≥3 of 4 games) + P(2–2) × P(DreamBreaker),
  where the DreamBreaker comes from the singles-value model.
- All four round-robin games are simulated (MLP only skips the dead 4th game
  in the playoff), so game and rally differentials — which decide the
  tiebreaks — come out of simulated scores rather than an assumption.
- **No probability is shown as 0% or 100%.** Values displayed as 0.0% are
  rounded-down small numbers, never impossibility. Individual matchups are
  floored/capped at 1.1%/98.9% by the calibration layer.

---

## 3. Projected lineups (what the numbers are priced off)

| Team | WD | MD | MXD1 | MXD2 |
|---|---|---|---|---|
| New Jersey 5s | Anna Leigh Waters / Jorja Johnson | Federico Staksrud / Noe Khlif | Waters / Staksrud | Johnson / Khlif |
| St. Louis Shock | Anna Bright / Kate Fahey | Hayden Patriquin / Gabriel Tardio | Bright / Patriquin | Fahey / Tardio |
| Brooklyn Pickleball Team | Rachel Rohrabacher / Jackie Kawamoto | Christian Alshon / Riley Newman | Rohrabacher / Alshon | Kawamoto / Newman |
| Los Angeles Mad Drops | Jade Kawamoto / Catherine Parenteau | Ben Johns / Connor Garnett | Kawamoto / Johns | Parenteau / Garnett |
| Palm Beach Royals | Tina Pisnik / Sofia Sewing | Dekel Bar / Tyson McGuffin | Pisnik / Bar | Sewing / McGuffin |
| California Black Bears | Sahra Dennehy / Zoey Weil | Dylan Frazier / Pablo Tellez | Dennehy / Frazier | Weil / Tellez |
| Las Vegas Night Owls | Chao Yi Wang / Liz Truluck | Roscoe Bellamy / Clayton Powell | Wang / Bellamy | Truluck / Powell |
| Miami Pickleball Club | Estee Widdershoven / Isabella Dunlap | Anderson Scarpa / James Delgado | Widdershoven / Scarpa | Dunlap / Delgado |
| Phoenix Flames | Daria Walczak / Alexa Schull | Jonathan Truong / Michael Loyd | Walczak / Truong | Schull / Loyd |
| Orlando Squeeze | Lacy Schneemann / Lina Padegimaite | Jack Sock / Gregory Dow | Schneemann / Sock | Padegimaite / Dow |
| Chicago Slice | Ting Chieh Wei / Emma Nelson | John Lucian Goins / Zane Navratil | Wei / Goins | Nelson / Navratil |
| Carolina Hogs | Abbigal Hatton / Nicole Conard | Wyatt Stone / Brandon French | Hatton / French | Conard / Stone |

---

## 4. GROUP A — predicted table

Rows in predicted order. "Wins" = expected matchup wins out of 5.

| # | Team | Exp. wins | Exp. game ± | 1st | 2nd | 3rd | 4th | 5th | 6th | Makes playoff |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | New Jersey 5s | 4.74 | +14.3 | **80.2%** | 18.9% | 0.8% | 0.1% | 0.0% | 0.0% | ~100% |
| 2 | Los Angeles Mad Drops | 3.97 | +9.9 | 18.9% | **67.3%** | 12.7% | 1.1% | 0.0% | 0.0% | ~100% |
| 3 | Palm Beach Royals | 2.87 | +2.6 | 0.8% | 12.1% | **64.5%** | 20.8% | 1.8% | 0.1% | 98.1% |
| 4 | California Black Bears | 2.14 | −2.4 | 0.1% | 1.6% | 20.9% | **65.2%** | 11.5% | 0.7% | 87.8% |
| 5 | Chicago Slice | 1.05 | −9.7 | 0.0% | 0.1% | 1.1% | 12.3% | **70.0%** | 16.5% | 13.5% |
| 6 | Carolina Hogs | 0.22 | −14.7 | 0.0% | 0.0% | 0.1% | 0.6% | 16.6% | **82.7%** | 0.7% |

## 5. GROUP B — predicted table

| # | Team | Exp. wins | Exp. game ± | 1st | 2nd | 3rd | 4th | 5th | 6th | Makes playoff |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | St. Louis Shock | 4.75 | +16.2 | **76.3%** | 23.6% | 0.1% | 0.0% | 0.0% | 0.0% | ~100% |
| 2 | Brooklyn Pickleball Team | 4.20 | +12.4 | 23.7% | **75.1%** | 1.3% | 0.0% | 0.0% | 0.0% | ~100% |
| 3 | Las Vegas Night Owls | 2.06 | −4.5 | 0.0% | 0.8% | **47.3%** | 27.9% | 14.4% | 9.5% | 76.1% |
| 4 | Miami Pickleball Club | 1.41 | −7.8 | 0.0% | 0.2% | 18.9% | **25.7%** | 28.2% | 27.0% | 44.8% |
| 5 | Phoenix Flames | 1.34 | −8.1 | 0.0% | 0.2% | 17.4% | 24.1% | **28.5%** | 29.9% | 41.6% |
| 6 | Orlando Squeeze | 1.24 | −8.3 | 0.0% | 0.2% | 15.0% | 22.3% | 28.9% | **33.6%** | 37.5% |

Group B splits into a top two and a genuine four-team scramble: Las Vegas,
Miami, Phoenix and hometown Orlando are all within 0.8 expected wins of each
other for the 3rd and 4th playoff spots. Three of the weekend's four closest
matchups sit inside that scramble (Phoenix–Orlando 51.6%, Miami–Phoenix
52.0%, Miami–Orlando 55.3%).

---

## 6. Predicted final finish (1–12)

Places 1–8 come from the crossover playoff; 9–10 and 11–12 are shared tiers
(those teams play no playoff matchup).

| Team | Grp | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th | 9–10 | 11–12 | Exp. place |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| New Jersey 5s | A | **49.8%** | 30.4% | 14.1% | 4.8% | 0.8% | 0.0% | 0.1% | 0.0% | 0.0% | 0.0% | 1.77 |
| St. Louis Shock | B | 39.0% | **37.3%** | 17.9% | 5.7% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1.91 |
| Brooklyn Pickleball Team | B | 6.3% | 17.3% | **39.0%** | 36.1% | 1.0% | 0.2% | 0.0% | 0.0% | 0.0% | 0.0% | 3.09 |
| Los Angeles Mad Drops | A | 4.8% | 14.1% | 27.2% | **40.2%** | 12.3% | 0.3% | 1.0% | 0.0% | 0.0% | 0.0% | 3.47 |
| Palm Beach Royals | A | 0.1% | 0.7% | 1.8% | 10.4% | **57.2%** | 7.3% | 18.9% | 1.9% | 1.8% | 0.1% | 5.43 |
| California Black Bears | A | 0.0% | 0.1% | 0.1% | 1.5% | 14.8% | 6.0% | **48.7%** | 16.5% | 11.5% | 0.7% | 7.07 |
| Las Vegas Night Owls | B | 0.0% | 0.0% | 0.0% | 0.8% | 8.1% | **39.2%** | 9.8% | 18.1% | 14.4% | 9.5% | 7.39 |
| Miami Pickleball Club | B | 0.0% | 0.0% | 0.0% | 0.1% | 1.9% | 17.0% | 6.0% | 19.7% | **28.2%** | 27.0% | 8.91 |
| Phoenix Flames | B | 0.0% | 0.0% | 0.0% | 0.2% | 1.8% | 15.6% | 5.7% | 18.4% | 28.5% | **29.9%** | 9.04 |
| Orlando Squeeze | B | 0.0% | 0.0% | 0.0% | 0.2% | 1.5% | 13.5% | 5.0% | 17.3% | 28.9% | **33.6%** | 9.24 |
| Chicago Slice | A | 0.0% | 0.0% | 0.0% | 0.1% | 0.4% | 0.7% | 4.8% | 7.5% | **70.0%** | 16.5% | 9.55 |
| Carolina Hogs | A | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% | 0.1% | 0.6% | 16.6% | **82.7%** | 11.14 |

(The saw-tooth in the middle columns — e.g. Palm Beach 57.2% for 5th but
7.3% for 6th — is real, not noise: odd places go to crossover *winners* and
even places to *losers*, so a team that usually enters the A3vB3 matchup as
the favorite lands on 5th far more often than 6th.)

Headline numbers for the infographic:

- **Title: New Jersey 49.8%, St. Louis 39.0%.** The Staksrud trade flips the
  favorite — between them 88.8%, and the two meet in the title matchup in
  ~61% of simulations.
- **Reaching the title matchup** (winning your group): New Jersey 80.2%,
  St. Louis 76.3%, Brooklyn 23.7%, LA Mad Drops 18.9%, everyone else
  under 1%.
- **Podium (top 3):** New Jersey 94.3%, St. Louis 94.2%, Brooklyn 62.6%,
  LA Mad Drops 46.1%. Nobody else clears 3%.
- **The home-team story:** Orlando lost Staksrud and Rane to New Jersey in
  the trade window and drops from a projected 3rd in Group B to most likely
  missing the playoff round entirely (62.5%).

---

## 7. All 30 round-robin matchups

Times converted to ET (Orlando local).

| Date | ET | Grp | Matchup | Favorite | Win prob |
|---|---|---|---|---|---|
| Thu 7/30 | 11:30a | B | Phoenix v Brooklyn | Brooklyn | 98.9% |
| Thu 7/30 | 12:00p | B | St. Louis v Miami | St. Louis | 98.9% |
| Thu 7/30 | 1:00p | A | California v Palm Beach | Palm Beach | 74.6% |
| Thu 7/30 | 1:30p | A | LA Mad Drops v Chicago | LA Mad Drops | 98.9% |
| Thu 7/30 | 2:30p | B | Phoenix v Las Vegas | Las Vegas | 67.0% |
| Thu 7/30 | 3:00p | A | New Jersey v Carolina | New Jersey | 98.9% |
| Thu 7/30 | 4:00p | B | Miami v Brooklyn | Brooklyn | 98.9% |
| Thu 7/30 | 4:30p | A | Palm Beach v Chicago | Palm Beach | 93.4% |
| Thu 7/30 | 5:30p | A | California v Carolina | California | 97.2% |
| Thu 7/30 | 6:00p | B | Las Vegas v Orlando | Las Vegas | 69.6% |
| Fri 7/31 | 11:30a | B | St. Louis v Phoenix | St. Louis | 98.9% |
| Fri 7/31 | 12:00p | A | LA Mad Drops v Palm Beach | LA Mad Drops | 82.6% |
| Fri 7/31 | 1:00p | B | Brooklyn v Las Vegas | Brooklyn | 98.5% |
| Fri 7/31 | 1:30p | A | Chicago v New Jersey | New Jersey | 98.9% |
| Fri 7/31 | 2:30p | A | California v LA Mad Drops | LA Mad Drops | 94.5% |
| Fri 7/31 | 3:00p | B | Phoenix v Orlando | Phoenix | 51.6% |
| Fri 7/31 | 4:00p | A | Chicago v Carolina | Chicago | 82.2% |
| Fri 7/31 | 4:30p | B | Brooklyn v St. Louis | St. Louis | 76.3% |
| Fri 7/31 | 5:30p | A | New Jersey v California | New Jersey | 98.4% |
| Fri 7/31 | 6:00p | B | Orlando v Miami | Miami | 55.3% |
| Sat 8/1 | 11:30a | B | Miami v Las Vegas | Las Vegas | 67.3% |
| Sat 8/1 | 12:00p | A | Palm Beach v New Jersey | New Jersey | 97.0% |
| Sat 8/1 | 1:00p | A | Carolina v LA Mad Drops | LA Mad Drops | 98.9% |
| Sat 8/1 | 1:30p | B | Brooklyn v Orlando | Brooklyn | 98.9% |
| Sat 8/1 | 2:30p | A | Chicago v California | California | 84.3% |
| Sat 8/1 | 3:00p | B | Las Vegas v St. Louis | St. Louis | 98.9% |
| Sat 8/1 | 4:00p | A | Carolina v Palm Beach | Palm Beach | 98.6% |
| Sat 8/1 | 4:30p | A | New Jersey v LA Mad Drops | New Jersey | 79.4% |
| Sat 8/1 | 5:30p | B | Miami v Phoenix | Miami | 52.0% |
| Sat 8/1 | 7:00p | B | Orlando v St. Louis | St. Louis | 98.9% |

**The matchups that decide the weekend:** Brooklyn v St. Louis (Fri 4:30p)
and New Jersey v LA Mad Drops (Sat 4:30p) for the two group titles, and the
Group B scramble triangle — Phoenix–Orlando (Fri 3:00p), Miami–Orlando
(Fri 6:00p), Miami–Phoenix (Sat 5:30p) — for the last two playoff spots.

---

## 8. Notes a designer should know

- **The 9–10 and 11–12 tiers must render as ties**, not as ranks. Splitting
  them would be inventing a result the format doesn't produce.
- **Never render a 0% or 100%.** House rule; 0.0% cells in the tables above
  are rounded, and several matchup prices are sitting on the 98.9%
  calibration cap rather than being genuinely near-certain.
- Expected place mixes ranks and tiers (a 9–10 tier counts as 9.5), so it's
  a sorting key and a rough summary — not a prediction of a specific rank.
- The saw-tooth across even/odd places (Section 6) is a real feature of the
  crossover format — don't smooth it away.
- Group labels A/B are ours, not MLP's.

## If you want this graded

These numbers are **not** in `model/receipts.json`. To put the weekend in the
permanent ledger before it starts, run `python web/make_forecast.py --commit`,
which freezes the per-matchup prices as pending receipts. The event-level
placement probabilities here would need their own scoring rule — say, Brier
on P(title) and on each team's top-4 flag — decided before Thursday if
they're going to count.
