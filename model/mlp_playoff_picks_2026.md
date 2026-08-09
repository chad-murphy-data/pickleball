# MLP 2026 quarterfinals — who should the top 4 pick?

Written 2026-08-09, hours after the Dallas first round ended and before MLP
announces the quarterfinal pairings. Newport Beach, Aug 14–16.

## The format (majorleaguepickleball.co)

Top 12 make the playoffs; seeds 1–4 get byes. Seeds 5–12 played best-of-three
matchup series in Dallas (Aug 7–9); the four winners are **re-seeded 5–8 by
regular-season standings points**. At the quarterfinals the **No. 1 seed
selects its opponent from the 5–8 pool, then No. 2, then No. 3; No. 4 takes
the leftover.** The picker is also home team, which in MLP means *responding*
to the other side's Mixed Doubles and DreamBreaker lineups. Each quarterfinal
is itself a best-of-three matchup series. Semifinals (NYC, Aug 28–30) repeat
the selection: highest remaining seed picks from the two lowest remaining.

## The field

All four higher seeds swept their series 2–0, so the re-seed is the identity
map and the pool is exactly seeds 5–8:

| | first round (Dallas) |
|---|---|
| #5 Brooklyn Pickleball Team | d. #10 SoCal Hard Eights 3–2, 3–1 |
| #6 Dallas Flash | d. #11 Las Vegas Night Owls 3–1, 3–0 |
| #7 Palm Beach Royals | d. #12 Chicago Slice 3–1, 3–1 |
| #8 Texas Ranchers | d. #9 Atlanta Bouncers 3–2, 3–2 |

## Rosters actually being fielded

v2 current-form values (per-point logit). Two teams' inferred best lineup is
**not** who they are playing, so both reads are carried through:

| | women | men | note |
|---|---|---|---|
| #1 New Jersey 5s | Waters +1.80, Jorja Johnson +1.17 | Staksrud +0.91, Khlif +0.79 | Howells +0.78 rotates |
| #2 St. Louis Shock | Bright +1.33, Fahey +1.08 | Patriquin +1.08, Tardio +1.07 | Hunter Johnson +0.72 traded in, no MLP game yet |
| #3 Los Angeles Mad Drops | Jade Kawamoto +1.16, Parenteau +0.90 | Ben Johns +1.11, Garnett +0.81 | Parker/Freeman rotate |
| #4 Columbus Sliders | Todd +1.14, Black +1.09 | Daescu +0.98, Klinger +0.76 | idle since 7/19 |
| #5 Brooklyn | Rohrabacher +1.12, J. Kawamoto +1.09 | Alshon +1.01, Newman +0.91 | deepest bench in the field |
| #6 Dallas Flash | Townsend +1.02, Truong +0.79 | JW Johnson +1.09, Ge +0.74 | Buckner +0.73 rotates |
| #7 Palm Beach Royals | Pisnik +1.13, Sewing +1.02 | Bar +0.76, **Diamond +0.50** | **McGuffin +0.65 sat out all of Dallas** |
| #8 Texas Ranchers | Jansen +0.70, **Christian +0.63** | Oncins +0.95, Acevedo +0.84 | **Sleeth +0.79 has not played since 6/20** |

Roster sum orders the field **NJ 4.67 > STL 4.56 > Brooklyn 4.12 > LA 3.98 ≈
Columbus 3.96 > Dallas 3.64 > Palm Beach 3.40 > Texas 3.12** — the No. 5 seed
is the third-strongest team in the bracket, which is the whole story of the
draft.

## Preference lists

Availability-corrected lineups; production pricing (v2 + weakest link →
race-to-11 DP → display calibration; DreamBreaker from the singles model).
"series" = best-of-three, matchups assumed independent.

| picker | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| **#1 New Jersey** | Palm Beach 97.8 / 98.9 | Texas 95.7 / 98.6 | Dallas 94.2 / 98.3 | Brooklyn 79.5 / 88.9 |
| **#2 St. Louis** | Texas 97.6 / 98.9 | Palm Beach 97.1 / 98.8 | Dallas 95.1 / 98.5 | Brooklyn 78.8 / 88.2 |
| **#3 Los Angeles** | Texas 90.9 / 97.1 | Palm Beach 89.4 / 96.4 | Dallas 80.3 / 89.6 | Brooklyn 45.9 / 43.9 |
| **#4 Columbus** | Texas 86.3 / 94.4 | Palm Beach 86.1 / 94.3 | Dallas 73.9 / 83.0 | Brooklyn 38.2 / 32.7 |

*(single matchup % / best-of-three series %)*

**Predicted draft: New Jersey → Palm Beach, St. Louis → Texas, Los Angeles →
Dallas, Columbus → Brooklyn (forced).**

### Why No. 1 wants a different team than everyone else

New Jersey's soft slot is men's doubles: Staksrud/Khlif are *underdogs* against
Brooklyn (33%), Texas (43%) and Dallas (44%), but 85% against Bar/Diamond. St.
Louis's soft slot is women's: Bright/Fahey are only 71% against Pisnik/Sewing
but 98% against Jansen/Christian. So the No. 1 seed hunts the weakest men and
the No. 2 seed hunts the weakest women — and those are different teams, so both
get their first choice without conflict.

### Robustness

4,000 resamples of every player's v2 value from its posterior N(mean, sd),
repricing all 16 quarterfinals each draw — P(this team is the pick):

| | Texas | Palm Beach | Dallas | Brooklyn |
|---|---|---|---|---|
| New Jersey | 12.7% | **83.8%** | 3.6% | 0.0% |
| St. Louis | **57.4%** | 37.3% | 5.3% | 0.0% |
| Los Angeles | **59.1%** | 37.0% | 3.9% | 0.0% |
| Columbus | **50.4%** | 46.1% | 3.6% | 0.0% |

Two things survive the noise and one does not. **Avoiding Brooklyn is
unanimous** (≥99% last choice for all four seeds) and **Dallas is a clear
third** for everyone. **Texas vs Palm Beach is a coin flip** for seeds 2–4 —
do not read the 0.1–0.7pp gaps in the table above as a real preference. New
Jersey's preference for Palm Beach is the one first-choice call with real
support (84%), and it comes from the men's-doubles mismatch.

Lookahead (backward induction over the whole bracket, each seed maximising its
own title equity knowing later seeds pick optimally) moves the recommended
picks by 0.1–0.5pp of title probability. That is far inside the model's own
error; the myopic quarterfinal-win-probability pick is the right one.

## Is a 95%+ matchup number believable?

Tested, because the compounded numbers look extreme. All 286 completed 2026 MLP
matchups, re-priced from **actual** lineups with month-of-game v2 values
(n = 197 after dropping matchups with untracked players):

| model says | n | observed |
|---|---|---|
| 50–60% | 20 | 30.0% |
| 60–70% | 17 | 52.9% |
| 70–80% | 22 | 81.8% |
| 80–90% | 45 | 93.3% |
| 90–95% | 33 | 93.9% |
| 95%+ | 60 | 98.3% (pred 97.9) |

Brier 0.097, accuracy 0.838. Logistic recalibration of the matchup logit gives
slope **1.19, 95% CI [0.92, 1.68]** — if anything the matchup-level number is
*under*confident, not over. Four-game matchups between mismatched MLP rosters
really are that lopsided. (Caveat: actual lineups remove projection error, and
month-of-game values come from a fit that saw those games, so this is a best
case. The thin 50–60% bin under-performing is n = 20.)

## What this does not model

- **v2 values are frozen at the 2026-07-25 fit** — they do not see the Aug 1–2
  event or the Dallas playoffs.
- **Series independence is assumed and untested.** If a weekend carries a
  common form component, the best-of-three column overstates favourites. The
  ordering is unaffected (the series map is monotone in the matchup
  probability).
- **Home-team advantage is not priced.** Responding to the opponent's mixed and
  DreamBreaker lineups is worth something real, and it accrues to the picker in
  every branch, so it cannot change a preference ordering.
- **McGuffin may return for Newport Beach.** Priced both ways: with him, Palm
  Beach's men go +0.65 instead of +0.50 and New Jersey's edge drops 97.8 → 96.0,
  which keeps Palm Beach as New Jersey's top choice and leaves every other
  ordering intact.
- Cross-gender comparisons remain a prior convention, per the house rule.

## Reproduce

Rosters and pricing come from `web/make_forecast.py`
(`mlp_rosters`, `best_lineup`, `price_game`, `matchup_tree`) against the open
BFF; the availability correction restricts each team's pool to players who
appeared in its most recent event. The matchup calibration table is a re-price
of `data/mlp_matchups_2026.csv` joined to `data/games.csv` with
`data/v2_trajectories.csv`.
