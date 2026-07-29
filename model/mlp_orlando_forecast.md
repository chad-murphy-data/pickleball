# Amway MLP Orlando — predicted results (design brief)

Generated 2026-07-29, before any ball is struck. Machine-readable payload:
`data/event_forecast.json`. Regenerate with:

```bash
python web/make_event_forecast.py --sims 200000
```

Event runs **Thu 30 July – Sat 1 Aug 2026**. 12 teams, two groups of six.
200,000 simulations. Nothing here is a receipt yet — see "If you want this
graded" at the bottom.

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

- **Best possible lineups**, as asked: each team's top 2 women + top 2 men by
  current v2 value, drawn from that team's 2026 MLP roster
  (latest-appearance-wins, so trades and rotations resolve), with the mixed
  pairs split to maximize weakest-link-adjusted strength. No official lineups
  have been published yet — all 30 matchups are still
  `SCHEDULED_WAITING_FOR_HOME_TEAM_LINEUP`.
- **This is the single largest source of error.** Rosters rotate event to
  event, and captains routinely deviate: on day 2 of MLP Chicago, 9 of 10
  matchups ran pairings that differed from the projection. A team's "best
  lineup" also assumes every listed player actually travels to Orlando,
  which the feed cannot confirm ahead of time.
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
| St. Louis Shock | Anna Bright / Kate Fahey | Hayden Patriquin / Gabriel Tardio | Bright / Patriquin | Fahey / Tardio |
| New Jersey 5s | Anna Leigh Waters / Jorja Johnson | Noe Khlif / Will Howells | Waters / Khlif | Johnson / Howells |
| Brooklyn Pickleball Team | Rachel Rohrabacher / Jackie Kawamoto | Christian Alshon / Riley Newman | Rohrabacher / Alshon | Kawamoto / Newman |
| Los Angeles Mad Drops | Jade Kawamoto / Catherine Parenteau | Ben Johns / Max Freeman | Kawamoto / Johns | Parenteau / Freeman |
| Palm Beach Royals | Tina Pisnik / Sofia Sewing | Dekel Bar / Tyson McGuffin | Pisnik / Bar | Sewing / McGuffin |
| Orlando Squeeze | Lacy Schneemann / Milan Rane | Federico Staksrud / Jack Sock | Schneemann / Staksrud | Rane / Sock |
| California Black Bears | Sahra Dennehy / Zoey Weil | Dylan Frazier / Joseph Wild | Dennehy / Frazier | Weil / Wild |
| Las Vegas Night Owls | Chao Yi Wang / Liz Truluck | Roscoe Bellamy / Clayton Powell | Wang / Bellamy | Truluck / Powell |
| Chicago Slice | Ting Chieh Wei / Jalina Ingram | Hunter Johnson / John Lucian Goins | Wei / Johnson | Ingram / Goins |
| Miami Pickleball Club | Estee Widdershoven / Isabella Dunlap | Anderson Scarpa / James Delgado | Widdershoven / Scarpa | Dunlap / Delgado |
| Phoenix Flames | Daria Walczak / Alexa Schull | Jonathan Truong / Wyatt Stone | Walczak / Truong | Schull / Stone |
| Carolina Hogs | Samantha Parker / Kelly Goodnow | Brandon French / Darrian Young | Parker / French | Goodnow / Young |

---

## 4. GROUP A — predicted table

Rows in predicted order. "Wins" = expected matchup wins out of 5.

| # | Team | Exp. wins | Exp. game ± | 1st | 2nd | 3rd | 4th | 5th | 6th | Makes playoff |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | New Jersey 5s | 4.71 | +13.4 | **80.1%** | 18.3% | 1.4% | 0.1% | 0.0% | 0.0% | ~100% |
| 2 | Los Angeles Mad Drops | 3.81 | +8.4 | 17.7% | **59.7%** | 19.6% | 2.6% | 0.4% | 0.0% | 99.6% |
| 3 | Palm Beach Royals | 2.91 | +2.8 | 1.9% | 18.8% | **57.1%** | 18.3% | 3.6% | 0.3% | 96.1% |
| 4 | California Black Bears | 2.00 | −2.7 | 0.2% | 2.6% | 17.7% | **55.2%** | 21.5% | 2.7% | 75.8% |
| 5 | Chicago Slice | 1.29 | −7.9 | 0.0% | 0.5% | 4.0% | 22.2% | **57.7%** | 15.5% | 26.7% |
| 6 | Carolina Hogs | 0.28 | −14.0 | 0.0% | 0.0% | 0.2% | 1.5% | 16.8% | **81.5%** | 1.7% |

## 5. GROUP B — predicted table

| # | Team | Exp. wins | Exp. game ± | 1st | 2nd | 3rd | 4th | 5th | 6th | Makes playoff |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | St. Louis Shock | 4.73 | +15.6 | **77.1%** | 22.1% | 0.7% | 0.0% | 0.0% | 0.0% | ~100% |
| 2 | Brooklyn Pickleball Team | 4.09 | +11.3 | 22.4% | **69.2%** | 8.2% | 0.2% | 0.0% | 0.0% | ~100% |
| 3 | Orlando Squeeze | 2.79 | +0.8 | 0.4% | 8.3% | **72.9%** | 15.4% | 2.1% | 0.8% | 97.1% |
| 4 | Las Vegas Night Owls | 1.54 | −7.1 | 0.0% | 0.3% | 11.6% | **45.5%** | 27.0% | 15.6% | 57.4% |
| 5 | Miami Pickleball Club | 0.95 | −10.3 | 0.0% | 0.0% | 3.6% | 19.5% | **35.7%** | 41.1% | 23.1% |
| 6 | Phoenix Flames | 0.90 | −10.2 | 0.0% | 0.0% | 3.0% | 19.4% | 35.1% | **42.5%** | 22.4% |

Group B's bottom two are a genuine coin flip — Miami and Phoenix are
separated by 0.05 expected wins, and their head-to-head on Saturday night is
the closest matchup of the weekend at 51.7%.

---

## 6. Predicted final finish (1–12)

Places 1–8 come from the crossover playoff; 9–10 and 11–12 are shared tiers
(those teams play no playoff matchup).

| Team | Grp | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th | 9–10 | 11–12 | Exp. place |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| St. Louis Shock | B | **47.1%** | 30.0% | 18.4% | 3.8% | 0.7% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 1.81 |
| New Jersey 5s | A | 41.5% | **38.6%** | 12.5% | 5.8% | 1.3% | 0.1% | 0.1% | 0.0% | 0.0% | 0.0% | 1.88 |
| Brooklyn Pickleball Team | B | 8.0% | 14.4% | **43.5%** | 25.7% | 6.8% | 1.4% | 0.2% | 0.0% | 0.0% | 0.0% | 3.14 |
| Los Angeles Mad Drops | A | 3.2% | 14.5% | 20.2% | **39.5%** | 15.1% | 4.5% | 2.4% | 0.2% | 0.4% | 0.0% | 3.75 |
| Palm Beach Royals | A | 0.1% | 1.8% | 3.1% | 15.7% | **32.5%** | 24.6% | 15.4% | 2.9% | 3.6% | 0.3% | 5.54 |
| Orlando Squeeze | B | 0.1% | 0.4% | 2.2% | 6.2% | 34.6% | **38.4%** | 10.9% | 4.5% | 2.1% | 0.8% | 5.77 |
| California Black Bears | A | 0.0% | 0.2% | 0.1% | 2.5% | 5.7% | 12.0% | **34.7%** | 20.5% | 21.5% | 2.7% | 7.54 |
| Las Vegas Night Owls | B | 0.0% | 0.0% | 0.0% | 0.3% | 2.0% | 9.6% | 16.6% | **28.9%** | 27.0% | 15.6% | 8.52 |
| Chicago Slice | A | 0.0% | 0.0% | 0.0% | 0.5% | 0.7% | 3.3% | 9.9% | 12.3% | **57.7%** | 15.5% | 9.20 |
| Miami Pickleball Club | B | 0.0% | 0.0% | 0.0% | 0.0% | 0.3% | 3.2% | 4.8% | 14.8% | 35.7% | **41.1%** | 9.85 |
| Phoenix Flames | B | 0.0% | 0.0% | 0.0% | 0.0% | 0.3% | 2.7% | 4.9% | 14.5% | 35.1% | **42.5%** | 9.90 |
| Carolina Hogs | A | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.2% | 0.3% | 1.3% | 16.8% | **81.5%** | 11.10 |

Headline numbers for the infographic:

- **Title: St. Louis 47.1%, New Jersey 41.5%.** Between them, 88.6% — this
  is a two-team weekend on paper.
- **Reaching the title matchup** (winning your group): St. Louis 77.1%,
  New Jersey 80.1%, Brooklyn 22.4%, LA Mad Drops 17.7%, everyone else
  under 2%.
- **Podium (top 3):** St. Louis 95.5%, New Jersey 92.6%, Brooklyn 65.9%,
  LA Mad Drops 37.9%. Nobody else clears 5%.
- **Most likely single outcome overall:** St. Louis and New Jersey meet for
  the title (61.7% of simulations produce that final).

---

## 7. All 30 round-robin matchups

Times converted to ET (Orlando local).

| Date | ET | Grp | Matchup | Favorite | Win prob |
|---|---|---|---|---|---|
| Thu 7/30 | 11:30a | B | Phoenix v Brooklyn | Brooklyn | 98.9% |
| Thu 7/30 | 12:00p | B | St. Louis v Miami | St. Louis | 98.9% |
| Thu 7/30 | 1:00p | A | California v Palm Beach | Palm Beach | 75.9% |
| Thu 7/30 | 1:30p | A | LA Mad Drops v Chicago | LA Mad Drops | 96.1% |
| Thu 7/30 | 2:30p | B | Phoenix v Las Vegas | Las Vegas | 66.7% |
| Thu 7/30 | 3:00p | A | New Jersey v Carolina | New Jersey | 98.9% |
| Thu 7/30 | 4:00p | B | Miami v Brooklyn | Brooklyn | 98.9% |
| Thu 7/30 | 4:30p | A | Palm Beach v Chicago | Palm Beach | 86.4% |
| Thu 7/30 | 5:30p | A | California v Carolina | California | 93.1% |
| Thu 7/30 | 6:00p | B | Las Vegas v Orlando | Orlando | 82.2% |
| Fri 7/31 | 11:30a | B | St. Louis v Phoenix | St. Louis | 98.9% |
| Fri 7/31 | 12:00p | A | LA Mad Drops v Palm Beach | LA Mad Drops | 74.2% |
| Fri 7/31 | 1:00p | B | Brooklyn v Las Vegas | Brooklyn | 98.5% |
| Fri 7/31 | 1:30p | A | Chicago v New Jersey | New Jersey | 98.7% |
| Fri 7/31 | 2:30p | A | California v LA Mad Drops | LA Mad Drops | 90.7% |
| Fri 7/31 | 3:00p | B | Phoenix v Orlando | Orlando | 92.0% |
| Fri 7/31 | 4:00p | A | Chicago v Carolina | Chicago | 81.2% |
| Fri 7/31 | 4:30p | B | Brooklyn v St. Louis | St. Louis | 76.3% |
| Fri 7/31 | 5:30p | A | New Jersey v California | New Jersey | 97.5% |
| Fri 7/31 | 6:00p | B | Orlando v Miami | Orlando | 90.4% |
| Sat 8/1 | 11:30a | B | Miami v Las Vegas | Las Vegas | 67.3% |
| Sat 8/1 | 12:00p | A | Palm Beach v New Jersey | New Jersey | 94.8% |
| Sat 8/1 | 1:00p | A | Carolina v LA Mad Drops | LA Mad Drops | 98.9% |
| Sat 8/1 | 1:30p | B | Brooklyn v Orlando | Brooklyn | 88.5% |
| Sat 8/1 | 2:30p | A | Chicago v California | California | 71.2% |
| Sat 8/1 | 3:00p | B | Las Vegas v St. Louis | St. Louis | 98.9% |
| Sat 8/1 | 4:00p | A | Carolina v Palm Beach | Palm Beach | 97.7% |
| Sat 8/1 | 4:30p | A | New Jersey v LA Mad Drops | New Jersey | 79.6% |
| Sat 8/1 | 5:30p | B | Miami v Phoenix | Miami | 51.7% |
| Sat 8/1 | 7:00p | B | Orlando v St. Louis | St. Louis | 96.8% |

**The three that decide the weekend:** Brooklyn v St. Louis (Fri 4:30p, the
Group B title), New Jersey v LA Mad Drops (Sat 4:30p, the Group A title), and
Miami v Phoenix (Sat 5:30p) for last place in Group B.

---

## 8. Notes a designer should know

- **The 9–10 and 11–12 tiers must render as ties**, not as ranks. Splitting
  them would be inventing a result the format doesn't produce.
- **Never render a 0% or 100%.** House rule; 0.0% cells in the tables above
  are rounded, and several matchup prices are sitting on the 98.9%
  calibration cap rather than being genuinely near-certain.
- Expected place mixes ranks and tiers (a 9–10 tier counts as 9.5), so it's
  a sorting key and a rough summary — not a prediction of a specific rank.
- Group labels A/B are ours, not MLP's.

## If you want this graded

These numbers are **not** in `model/receipts.json`. To put the weekend in the
permanent ledger before it starts, run `python web/make_forecast.py --commit`,
which freezes the per-matchup prices as pending receipts. The event-level
placement probabilities here would need their own scoring rule — say, Brier
on P(title) and on each team's top-4 flag — decided before Thursday if
they're going to count.
