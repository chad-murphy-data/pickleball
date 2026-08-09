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
| #5 Brooklyn | Rohrabacher +1.12, J. Kawamoto +1.09 | Alshon +1.01, Newman +0.91 | **weakest** bench in the field, and the most dependent on it — see below |
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

## Roster verification (2026-08-09)

MLP's own team pages render their rosters in JS, so the HTML carries none of
it — same wall as the rest of this project. Checked instead against the
team-run sites and The Dink's transaction tracker. Every inferred roster
holds up:

| team | independent check |
|---|---|
| Palm Beach | palmbeachroyals.com/team lists Sewing, McGuffin, Pisnik, Diamond, Bar, Goldin, Emmrich — an exact 7/7 match. Tracker: Diamond added and **Goldin to IR** on 6/23, matching Goldin's last appearance (6/18) |
| Texas | ranchers.com/our-players lists Oncins, Acevedo, Hewett, Christian, Sleeth, Jansen. Barlow waived 6/23 (tracker) matches his 6/21 last appearance |
| New Jersey | Staksrud + Milan Rane acquired from Orlando at the deadline for Emmrich/Padegimaite — matches his first NJ appearance on 7/30. Rane (+0.72) does not displace Jorja Johnson |
| St. Louis | Hunter Johnson from Chicago 6/24 — the existing `roster_overrides.csv` entry. Also received Angie Walker (+0.55), who has still not played; she is the one player the inference misses and she does not crack the four |
| Los Angeles | Freeman (1/30) and Garnett from Utah (6/10) both confirmed |
| Columbus | Castillo in / Truong out (5/28) and Tyra Black in / Townsend out (6/15) both confirmed, and latest-appearance-wins moved Truong and Townsend correctly |
| Dallas | Buckner (5/16), Townsend (6/15), Truong in / Huang out (6/22) all confirmed |
| Brooklyn | Alshon from Texas (3/3) confirmed. The tracker also shows Haworth going to California on 3/4, but he played for Brooklyn through 8/2; bench either way |

**The Sleeth correction is right, and the reason is injury, not a trade.**
She was *in a sling at MLP Austin*, and Kaitlyn Christian was acquired on 6/24
explicitly as the contingency to partner Jansen. Texas still rosters Sleeth, so
a healthy return before Newport Beach is live — it would lift Texas's women
from +0.63 to +0.79 and make them a slightly less attractive pick.

Two cosmetic mismatches, neither load-bearing: MLP actually pairs Staksrud with
**Howells** in men's doubles, not Khlif (a 0.009 gap in v2 — the two are
interchangeable), and the tracker's "Marcela Hones" is the BFF's "Marcela
Aguila Ampon".

## Is there ever a reason not to take the easiest opponent?

Tested three mechanisms exhaustively. **No — not in this bracket.**

**Non-transitivity: none.** Zero intransitive triples across all 336 ordered
triples. The eight teams form a clean total order, NJ > STL > Brooklyn > LA >
Columbus > Dallas > Palm Beach > Texas. There is one *shape* effect that stops
short of a cycle: Palm Beach beats Texas head-to-head (53%), yet New Jersey
would rather play Palm Beach — the No. 1 seed prefers the objectively stronger
team, because its own soft slot is men's doubles and Palm Beach's men are the
weakest in the pool. That is the closest thing to a genuine oddity here.

**The semi-pool effect: real mechanism, doesn't pay.** In the semis the top
surviving seed picks from the two *lowest* remaining, so the second surviving
seed is never selectable and inherits the *stronger* of that pair. That gives
the No. 2 seed a real reason to want Brooklyn gone, and its only lever is
taking Brooklyn itself. It doesn't come close: eliminating Brooklyn yourself
costs a **certain** 89% series instead of a 99% one, while leaving Brooklyn
alive costs a **probabilistic** one — and St. Louis beats them 89% in the semis
anyway. Title equity 34.8% for taking Brooklyn vs 39.5% for the myopic pick.
The general rule: ducking early only pays when the team you would duck is
near-certain to survive *and* far more dangerous to you than any alternative
semifinal opponent. Brooklyn is 66% to get through Columbus, and St. Louis is
nearly as strong against them as against anyone.

**Kingmaking by No. 3: no lever.** The No. 3 seed's pick decides what No. 4
gets, so it can choose whether Brooklyn or a fellow top seed reaches the semis.
But No. 3 sits in the bottom-two pool itself, so its semifinal is against
No. 1 or No. 2 either way — and LA prices at 21.3% vs New Jersey and 20.0% vs
St. Louis. Nothing to engineer when the two teams you might steer toward
yourself are interchangeable.

Shrinking every probability toward 50% on the logit scale — what correlated
series outcomes or overstated rating gaps would look like — the myopic pick
stays optimal at every level from k = 0.9 down to k = 0.1. The lone flip is at
k = 1.0 exactly, where St. Louis's Dallas-over-Texas edge is 0.02pp of title
equity: numerical noise, and it vanishes the moment the model is shaded at all.

Two live reasons to deviate that are *not* about bracket structure: McGuffin
returning would harden Palm Beach, and a healthy Sleeth would harden Texas.
Both are information problems, not strategy problems.

## The six-player rosters (confirmed 2026-08-09)

Every MLP roster is exactly **3 men + 3 women** — four starters plus a
one-man, one-woman bench. All eight playoff teams validate, and the six-player
rosters are committed to `data/mlp_rosters_2026.csv`. Bench strength (sum of
the two reserves' v2 values), weakest first:

| team | starters | bench | bench players |
|---|---|---|---|
| #5 Brooklyn | 4.125 | **0.809** | Haworth +0.62, Blatt +0.19 |
| #4 Columbus | 3.959 | 0.864 | Castillo +0.70, Crum +0.16 |
| #7 Palm Beach | 3.405 | 1.097 | **McGuffin +0.65**, Emmrich +0.45 |
| #6 Dallas | 3.643 | 1.242 | Buckner +0.73, Jakovljevic +0.52 |
| #2 St. Louis | 4.559 | 1.275 | Hunter Johnson +0.72, Angie Walker +0.55 |
| #3 Los Angeles | 3.974 | 1.420 | Parker +0.73, Freeman +0.69 |
| #8 Texas | 3.120 | 1.449 | **Sleeth +0.79**, Hewett +0.66 |
| #1 New Jersey | 4.669 | 1.504 | Howells +0.78, Rane +0.72 |

Exactly two teams have a reserve who would crack their own best four —
**Palm Beach's McGuffin and Texas's Sleeth**, the two availability cases
already handled above. Nobody else's bench changes a lineup, so the pick
analysis is untouched by any of this.

Two consequences that make earlier reads *stronger*:

- **Palm Beach's weak men are forced, not chosen.** Goldin is on IR, so the
  roster carries exactly three men: Bar, McGuffin, Diamond. If McGuffin can't
  go, Bar/Diamond is the only legal men's pair — the hole New Jersey is hunting
  has no escape hatch.
- **Blatt is Brooklyn's only bench woman.** Any game Rohrabacher sits, Blatt
  *must* take. The 12-of-13 pattern below is not a selection among options.

## Brooklyn's real women's doubles pair (correction)

The tables above put Rachel Rohrabacher in Brooklyn's women's doubles. **She
mostly isn't there.** Hannah Blatt has started WD in **12 of Brooklyn's last 13
matchups** (every one since 7/23 except 8/8), and in 6 of those she played a
mixed as well. Rohrabacher has been reduced to one mixed game in most matchups.
In Dallas it was 1 of 2 — full strength on 8/8, Blatt in WD on 8/9.

This is not a small sub. Blatt is +0.190 (254 games, sd 0.097) against
Rohrabacher's +1.117 (910, sd 0.086) — a 0.927 logit hole in one of four games.
The 2026 MLP record agrees with the model: Brooklyn games with Blatt on court
go **36.7% won, −2.10 average margin** (n=30); with Rohrabacher, **77.1% and
+3.63** (n=35).

Top seed's best-of-three series win probability against Brooklyn:

| | full strength | Blatt in WD | Blatt in a mixed | Blatt in WD + mixed | playoff blend |
|---|---|---|---|---|---|
| #1 New Jersey | 88.5% | 91.2% | 92.4% | 94.5% | 89.8% |
| #2 St. Louis | 87.8% | 95.8% | 93.2% | 98.0% | 91.8% |
| #3 Los Angeles | 43.9% | 77.9% | 58.1% | 91.0% | 60.9% |
| #4 Columbus | 32.7% | 56.0% | 52.5% | 80.1% | 44.3% |

*(playoff blend = the observed 1-of-2 Dallas rate)*

**The preference ordering survives — Brooklyn is still everyone's last choice**
in every scenario but one. The exception: if Blatt plays *two* games, Los
Angeles would rather have Brooklyn (91.0%) than Dallas (89.6%).

**The magnitude does not survive, and this corrects the framing above.**
Columbus is not doomed. Against a Blatt-in-WD Brooklyn they are a 56% favourite,
not a 33% underdog; on the playoff blend, 44%. The claim that Brooklyn is the
third-strongest team in the bracket holds only at full strength — with Blatt
starting WD as she has all summer, Brooklyn is roughly a coin flip with
Columbus and clearly behind Los Angeles.

The asymmetry is worth noting: the sub costs Brooklyn **least against New
Jersey** (+2.7pp), because Waters/Johnson were already winning that women's
game 92% anyway, and most against St. Louis (+8.0pp) and Los Angeles (+34pp),
whose women's pairs were in a real fight with Rohrabacher.

Where should Brooklyn hide her? Women's doubles, in all four matchups — and by
a wide margin against New Jersey (8.7% Brooklyn vs 2.0% if she goes in a
mixed). The reason is the home-team rule: Brooklyn is the lower seed in every
quarterfinal, so it posts mixed first and the top seed responds, which lets the
top seed aim its strongest mixed pair straight at Blatt. Brooklyn's actual
practice matches the model's advice.

## Reproduce

Rosters and pricing come from `web/make_forecast.py`
(`mlp_rosters`, `best_lineup`, `price_game`, `matchup_tree`) against the open
BFF; the availability correction restricts each team's pool to players who
appeared in its most recent event. The matchup calibration table is a re-price
of `data/mlp_matchups_2026.csv` joined to `data/games.csv` with
`data/v2_trajectories.csv`.
