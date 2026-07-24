# DreamBreaker split-role scenarios on real MLP rosters (2026-07-22)

**The rule simulated** (Anna Bright's proposal): Team 1 sets the four
same-gender singles **matchups**; Team 2 then sets the **order** (which
matchup occupies which rotation slot). Run over the real current rosters of
all 20 MLP franchises, every ordered pair (380 matchups per scenario), with
each team appearing as Team 1 and as Team 2.

Everything below was adversarially verified before reporting: 8 independent
auditors (spec compliance, rules implementation, independent summary
recomputation, policy optimality, roster provenance, correction-fit
reproduction, hand-replay of raw referee logs, independent margin DP) plus
3 re-audits after fixes. The two audit failures found (a stale same-day
roster tie-break; S1 "maximin" semantics) are fixed and documented below.

## 1. The rules (verified, not assumed)

From the official 2025 MLP Rules Guide sec. 7, the 2026 Competition
Updates, and this project's own referee-log archive (74+ reconstructed
DreamBreakers):

- Race to **21, win by 2, no cap**; **rally scoring** with **the freeze**:
  *"a team must win while serving"* — a rally won by the receiving team
  that would be the game-winning point scores **nothing** and only
  transfers the serve. Log-verified: zero receiver-won game-winners in 74
  reconstructed DBs; all no-point side-outs cluster at would-win states.
- The **winner of each rally serves the next** (99.3% of log transitions).
- **Four players a side, 4-point rotations**, cycling P1→P4→P1; the
  rotation counter advances on POINTS (frozen rallies consume no slot).
- Composition must be **2 men + 2 women**; the **order is unconstrained**.
- Under today's actual rules the home team reveals its order first and the
  away team counters; lineups are set right after game 4 (2026 change).
  The Team-1-pairs / Team-2-orders rule simulated here is the hypothetical
  variant under discussion.

## 2. Data

- **Rosters** (`model/build_db_rosters.py` → `data/db_rosters.csv`):
  every franchise's most recent completed matchup (MLP San Diego 7/16–19,
  else the Edward Jones Mid-Season 7/8–12), ordered by actual start
  timestamp; the WD pair = the team's two women, the MD pair = its two
  men. UUID-keyed throughout. All 80 franchise player rows complete,
  gender-consistent, valued.
- **PICKLE singles scores**: `fit_singles` values for players with >= 10
  pro singles games (64 of 80 players).
- **Non-singles correction** (`model/db_impute.py`, `model/db_impute.md`):
  for the 16 players without a real singles record, singles is imputed
  from doubles and **shrunk by 0.35** — measured from 3,101 same-gender
  DreamBreaker rallies (explicit on-court tracking from substitution log
  rows; 88/94 logged DBs reconstruct the official final EXACTLY; the rest
  are excluded). Controls for opposition and own doubles rating via the
  rally logistic: penalty −0.183 ± 0.081 (z = −2.25), value-scale shrink
  0.36, cluster-bootstrap 95% CI [+0.02, +0.74], P(no effect) = 1.9%;
  threshold sensitivities 0.42/0.43 both clear zero.
- **Rally model**: P(win rally) = sigmoid(k·(v1−v2)), **k = 0.510** — the
  rally-level coefficient fit at exactly this grain (0.42 team-level
  sensitivity reported below).
- One real-world quirk: **Yuta Funemizu appears in both Miami's (7/10) and
  Utah's (7/18) latest lineups** (cross-team substitute usage is legal in
  2026). The 2 Miami↔Utah pairs per scenario are flagged
  `shared_player=1`; aggregates are unchanged at reported precision with
  them excluded.

## 3. Engine

Exact dynamic programming over (score, score, slot, points-into-slot,
server) — freeze and serve modelled; no simulation noise. Win probability
from a backward DP; margins from an independent forward DP over final
scores; the two agree to 1e-9 on every configuration, and an independent
Monte Carlo implementation agrees within sampling noise (worst 2.2 sigma at
500k sims x 20 configs). First server (coin toss) averaged 50/50.

## 4. Scenarios

- **S1** — Team 1 maximins; Team 2 sorts the matchups by its own edge,
  biggest edge in slot 1. ("Maximin" = best response to Team 2's specified
  edge-sort policy, which is deterministic, so the two coincide.)
- **S2** — Team 1 instead builds the most **unbalanced** same-gender
  matchups it can (per gender, maximize |p1 − p2|); Team 2 still edge-sorts.
- **S3** — Team 1 maximins; Team 2 puts its **men in slots 1–2**, higher
  PICKLE rating first (women 3–4 by rating — the spec pins only the men).
- **S1adv** (robustness, not requested): Team 2 orders truly adversarially
  (min over all 24 permutations). Included because, under the freeze,
  edge-sort is close to but not exactly Team 2's optimum: it concedes a
  mean 0.56 pp / max 2.6 pp of Team-1 win probability.

## 5. Headline results (k = 0.510)

| scenario | T1 win % (mean) | T1 favored | E[margin T1] | MOV if T1 wins | MOV if T2 wins | % women's matches in slots 1–2 | % women's match first |
|---|---:|---:|---:|---:|---:|---:|---:|
| **S1** | **47.5%** | 45.3% | −0.43 | 4.86 | 5.20 | **52.2%** | **47.9%** |
| **S2** | **45.8%** | 42.4% | −0.63 | 4.81 | 5.22 | 50.0% | 38.7% |
| **S3** | **50.7%** | 51.8% | −0.16 | 4.79 | 5.27 | **0.0%** | **0.0%** |
| S1adv | 46.9% | 44.2% | −0.31 | 5.04 | 5.03 | 50.1% | 52.4% |

(Margins in points; "MOV" = expected margin of victory conditional on that
side winning. k = 0.42 sensitivity: S1 47.8/S2 46.3/S3 50.6 — same story,
slightly compressed. Excluding the 2 shared-player rows changes no cell at
this precision.)

Reading it:

- **Holding the order beats holding the matchups.** In S1 the
  matchup-picker averages 47.5%, so the order-picker averages 52.5% — the
  same team gains **+5.05 pp** on average by holding position rather than
  pairs (per-pair role swap). The margin columns agree (E[margin] −0.43 for
  the picker).
- **Unbalancing your own pairings is a pure mistake**: S2 costs Team 1 a
  further **1.69 pp on average (max 8.1 pp)** and is strictly worse in
  **379 of 380** matchups. Balanced (rank-matched) pairings are the
  defensive optimum: the order-picker punishes any lopsided pairing by
  featuring your worst matchup in the busy slot 1.
- **Men-first ordering throws the order advantage away — and more.** In S3
  Team 2 gives back its entire +5 pp edge (T1 mean rises to 50.7%,
  +3.26 pp vs S1, better for T1 in 351/380, max +15.2 pp). A team that
  wins the order and then plays men-first is, on average, no better off
  than the team that lost it.
- **Women and the busy slots**: with both teams playing the specified S1
  strategies, women's matchups take **52.2% of slots 1–2** and lead off
  **47.9%** of DreamBreakers. Under S3 ordering: 0% by construction —
  which is not a hypothetical: in the **176 real announced DB orders on
  record, slot 1 was a man 176/176 times** (men fill 96% of slots 1–2).
  S3 is, to first order, the league's actual revealed strategy today.

## 6. Per-team view (S1)

| team | as matchup-picker (T1, S1) | as order-picker (T2, S1) | advantage from being T2 | cost of men-first (S3, as T2) |
|---|---:|---:|---:|---:|
| New Jersey 5s | 78.3% | 83.1% | +4.8 pp | -3.13 pp |
| St. Louis Shock | 72.3% | 77.1% | +4.8 pp | -4.96 pp |
| Los Angeles Mad Drops | 69.1% | 72.6% | +3.5 pp | -2.95 pp |
| Texas Ranchers | 63.1% | 68.5% | +5.4 pp | -6.13 pp |
| Orlando Squeeze | 55.0% | 63.5% | +8.5 pp | -0.18 pp |
| Columbus Sliders | 56.0% | 62.1% | +6.1 pp | -6.24 pp |
| Palm Beach Royals | 52.1% | 56.5% | +4.4 pp | -4.39 pp |
| Atlanta Bouncers | 48.9% | 55.8% | +6.9 pp | -0.97 pp |
| Brooklyn Pickleball Team | 47.6% | 54.0% | +6.4 pp | -4.39 pp |
| Las Vegas Night Owls | 46.3% | 51.4% | +5.1 pp | -4.85 pp |
| California Black Bears | 47.0% | 51.3% | +4.3 pp | -3.70 pp |
| Utah Black Diamonds | 46.8% | 50.9% | +4.1 pp | -3.35 pp |
| Miami Pickleball Club | 46.8% | 50.7% | +3.9 pp | -2.93 pp |
| Dallas Flash | 39.4% | 44.4% | +5.0 pp | -1.44 pp |
| Chicago Slice | 37.0% | 42.9% | +5.9 pp | -0.67 pp |
| Phoenix Flames | 36.4% | 40.0% | +3.6 pp | -2.73 pp |
| Bay Area Breakers | 31.9% | 37.4% | +5.5 pp | -3.72 pp |
| SoCal Hard Eights | 31.4% | 36.7% | +5.3 pp | -5.73 pp |
| Florida Smash | 23.9% | 27.8% | +3.9 pp | -0.91 pp |
| Carolina Hogs | 20.3% | 23.8% | +3.5 pp | -1.79 pp |


The four teams that lose the most by men-first ordering (Columbus −6.2,
Texas −6.1, SoCal −5.7, St. Louis −5.0) are exactly the teams whose women
carry their biggest edges.

## 7. Honest limits

- Rally probabilities are iid within a matchup given the values; serve
  enters only through the freeze (the rally-level fit absorbs average
  serve effects). k = 0.510 ± 0.078 — win probabilities move ~0.3–0.5 pp
  across that range (0.42 sensitivity shown above).
- The 0.35 non-singles correction is significant but wide
  (CI [+0.02, +0.74]); 16 of 80 players carry it.
- Rosters are lineups from each team's most recent event — teams can and
  do change DreamBreaker fours (2026 allows any 2M+2W from the roster).
- Momentum, fatigue, and matchup-specific tactics are outside the model;
  values are current form (monthly random walk), not season averages.

## 8. Why slot 1 matters: empirical point distribution

A DreamBreaker averages **36.6 total points** (median 37, range 26–46 across
88 logged DBs). The 4-point rotation cycles P1→P4→P1, advancing on scored
points only (frozen side-outs consume no slot). At ~37 total points you get
2.5 full cycles, so slot 1's pair plays **three passes** (points 1–4, 17–20,
33–36) while slot 4's pair only gets **two** (13–16, 29–32):

| slot | avg points | share | median | range |
|---|---:|---:|---:|---:|
| Slot 1 | 10.9 | 30.1% | 12 | 6–12 |
| Slot 2 | 9.6 | 26.4% | 9 | 7–12 |
| Slot 3 | 8.1 | 22.4% | 8 | 6–12 |
| Slot 4 | 7.7 | 21.2% | 8 | 4–10 |

**Slot 1 sees 1.42× more points than slot 4.** The slot-1 median is 12 —
the empirical ceiling — because most games end mid-way through slot 1's
third pass. Slot 4 often never starts a third pass at all. This is the
mechanical reason edge-sorting is worth ~5 pp on average: you're not just
choosing *which* matchup plays, you're choosing which one plays a third
time.

## 9. Reproduce

```bash
python model/build_db_rosters.py            # rosters (cached API)
python model/db_impute.py                   # parser + correction refit
python model/db_scenarios.py --adversarial  # all scenarios + S1adv
python model/db_scenarios.py --mc 500000    # Monte Carlo self-check
```

Full per-matchup detail (orders, margins, flags):
`data/db_scenarios_matchups.csv`.

## 10. Appendix: every matchup, Team-1 win probability by scenario

`*` = shared-player pair (Funemizu on both rosters).

| Team 1 (picks matchups) | Team 2 (picks order) | S1 T1 win | S2 T1 win | S3 T1 win | S1adv T1 win | S1 E[margin] |
|---|---|---:|---:|---:|---:|---:|
| Atlanta Bouncers | Bay Area Breakers | 64.6% | 64.1% | 71.4% | 63.2% | +1.75 |
| Atlanta Bouncers | Brooklyn Pickleball Team | 46.0% | 44.1% | 55.2% | 45.8% | -0.37 |
| Atlanta Bouncers | California Black Bears | 49.1% | 48.4% | 57.7% | 48.2% | -0.21 |
| Atlanta Bouncers | Carolina Hogs | 78.6% | 77.9% | 82.1% | 78.4% | +4.16 |
| Atlanta Bouncers | Chicago Slice | 61.3% | 59.7% | 63.1% | 61.1% | +1.51 |
| Atlanta Bouncers | Columbus Sliders | 37.6% | 37.0% | 49.5% | 36.7% | -1.76 |
| Atlanta Bouncers | Dallas Flash | 59.7% | 58.3% | 62.4% | 59.4% | +1.26 |
| Atlanta Bouncers | Florida Smash | 75.5% | 74.1% | 77.7% | 75.3% | +3.61 |
| Atlanta Bouncers | Las Vegas Night Owls | 48.5% | 47.6% | 58.6% | 47.9% | -0.16 |
| Atlanta Bouncers | Los Angeles Mad Drops | 27.5% | 26.9% | 34.7% | 27.2% | -3.10 |
| Atlanta Bouncers | Miami Pickleball Club | 50.3% | 49.6% | 57.5% | 49.9% | +0.03 |
| Atlanta Bouncers | New Jersey 5s | 17.6% | 17.3% | 24.1% | 16.4% | -4.91 |
| Atlanta Bouncers | Orlando Squeeze | 41.0% | 39.9% | 41.4% | 40.9% | -1.16 |
| Atlanta Bouncers | Palm Beach Royals | 43.6% | 43.0% | 53.2% | 43.4% | -0.83 |
| Atlanta Bouncers | Phoenix Flames | 61.4% | 60.7% | 67.9% | 60.9% | +1.50 |
| Atlanta Bouncers | SoCal Hard Eights | 63.1% | 62.3% | 73.8% | 62.9% | +1.89 |
| Atlanta Bouncers | St. Louis Shock | 22.5% | 22.4% | 32.3% | 22.1% | -3.88 |
| Atlanta Bouncers | Texas Ranchers | 30.8% | 30.2% | 42.6% | 30.6% | -2.57 |
| Atlanta Bouncers | Utah Black Diamonds | 50.0% | 49.3% | 57.7% | 49.8% | +0.04 |
| Bay Area Breakers | Atlanta Bouncers | 27.3% | 26.9% | 29.1% | 26.9% | -3.08 |
| Bay Area Breakers | Brooklyn Pickleball Team | 30.2% | 27.7% | 34.5% | 29.1% | -2.83 |
| Bay Area Breakers | California Black Bears | 33.4% | 30.4% | 35.9% | 33.4% | -2.23 |
| Bay Area Breakers | Carolina Hogs | 61.0% | 60.6% | 64.1% | 60.5% | +1.54 |
| Bay Area Breakers | Chicago Slice | 40.5% | 37.7% | 41.3% | 39.6% | -1.36 |
| Bay Area Breakers | Columbus Sliders | 24.8% | 23.3% | 28.3% | 24.3% | -3.61 |
| Bay Area Breakers | Dallas Flash | 38.3% | 37.6% | 40.5% | 37.6% | -1.58 |
| Bay Area Breakers | Florida Smash | 56.7% | 54.2% | 58.1% | 55.8% | +0.81 |
| Bay Area Breakers | Las Vegas Night Owls | 32.3% | 30.3% | 37.3% | 31.9% | -2.42 |
| Bay Area Breakers | Los Angeles Mad Drops | 14.8% | 13.8% | 17.2% | 14.7% | -5.23 |
| Bay Area Breakers | Miami Pickleball Club | 32.6% | 30.8% | 35.8% | 32.3% | -2.34 |
| Bay Area Breakers | New Jersey 5s | 9.3% | 7.7% | 10.0% | 9.1% | -6.61 |
| Bay Area Breakers | Orlando Squeeze | 21.3% | 20.0% | 22.1% | 20.9% | -4.05 |
| Bay Area Breakers | Palm Beach Royals | 28.0% | 26.8% | 31.8% | 27.7% | -3.03 |
| Bay Area Breakers | Phoenix Flames | 43.5% | 41.1% | 46.5% | 43.3% | -0.82 |
| Bay Area Breakers | SoCal Hard Eights | 47.1% | 45.9% | 53.9% | 46.9% | -0.38 |
| Bay Area Breakers | St. Louis Shock | 13.2% | 12.4% | 15.3% | 12.7% | -5.72 |
| Bay Area Breakers | Texas Ranchers | 19.2% | 18.4% | 22.9% | 18.5% | -4.58 |
| Bay Area Breakers | Utah Black Diamonds | 32.1% | 31.2% | 36.0% | 31.9% | -2.38 |
| Brooklyn Pickleball Team | Atlanta Bouncers | 45.4% | 44.1% | 47.4% | 43.5% | -0.87 |
| Brooklyn Pickleball Team | Bay Area Breakers | 61.1% | 58.5% | 68.0% | 60.9% | +1.60 |
| Brooklyn Pickleball Team | California Black Bears | 47.8% | 43.3% | 53.9% | 47.4% | -0.22 |
| Brooklyn Pickleball Team | Carolina Hogs | 76.8% | 74.2% | 79.5% | 76.1% | +3.72 |
| Brooklyn Pickleball Team | Chicago Slice | 59.1% | 55.0% | 60.2% | 57.5% | +1.01 |
| Brooklyn Pickleball Team | Columbus Sliders | 36.2% | 33.6% | 45.3% | 36.0% | -1.67 |
| Brooklyn Pickleball Team | Dallas Flash | 56.4% | 53.9% | 59.3% | 55.0% | +0.67 |
| Brooklyn Pickleball Team | Florida Smash | 73.9% | 69.9% | 75.0% | 72.7% | +3.14 |
| Brooklyn Pickleball Team | Las Vegas Night Owls | 50.5% | 42.4% | 54.8% | 50.3% | +0.11 |
| Brooklyn Pickleball Team | Los Angeles Mad Drops | 26.6% | 23.5% | 31.7% | 26.2% | -3.23 |
| Brooklyn Pickleball Team | Miami Pickleball Club | 49.2% | 44.7% | 54.0% | 48.6% | -0.12 |
| Brooklyn Pickleball Team | New Jersey 5s | 15.1% | 14.1% | 20.9% | 15.0% | -5.04 |
| Brooklyn Pickleball Team | Orlando Squeeze | 38.9% | 35.9% | 39.0% | 36.3% | -1.88 |
| Brooklyn Pickleball Team | Palm Beach Royals | 42.4% | 39.1% | 49.5% | 42.2% | -0.92 |
| Brooklyn Pickleball Team | Phoenix Flames | 59.8% | 55.8% | 64.5% | 59.2% | +1.30 |
| Brooklyn Pickleball Team | SoCal Hard Eights | 64.9% | 58.3% | 70.3% | 64.4% | +2.02 |
| Brooklyn Pickleball Team | St. Louis Shock | 21.0% | 20.2% | 28.8% | 20.8% | -3.94 |
| Brooklyn Pickleball Team | Texas Ranchers | 30.0% | 27.7% | 38.7% | 29.7% | -2.62 |
| Brooklyn Pickleball Team | Utah Black Diamonds | 48.8% | 44.9% | 54.2% | 48.4% | -0.16 |
| California Black Bears | Atlanta Bouncers | 42.3% | 41.9% | 43.4% | 41.4% | -1.12 |
| California Black Bears | Bay Area Breakers | 63.5% | 60.0% | 65.1% | 63.4% | +1.82 |
| California Black Bears | Brooklyn Pickleball Team | 46.0% | 41.9% | 49.1% | 44.8% | -0.68 |
| California Black Bears | Carolina Hogs | 75.3% | 75.1% | 77.1% | 75.2% | +3.65 |
| California Black Bears | Chicago Slice | 56.4% | 53.9% | 56.4% | 55.4% | +0.65 |
| California Black Bears | Columbus Sliders | 37.7% | 36.0% | 42.3% | 37.6% | -1.84 |
| California Black Bears | Dallas Flash | 54.5% | 53.8% | 55.6% | 53.6% | +0.45 |
| California Black Bears | Florida Smash | 71.8% | 69.7% | 72.1% | 71.0% | +2.81 |
| California Black Bears | Las Vegas Night Owls | 48.4% | 44.5% | 52.1% | 48.0% | -0.23 |
| California Black Bears | Los Angeles Mad Drops | 26.8% | 24.7% | 28.5% | 26.7% | -3.21 |
| California Black Bears | Miami Pickleball Club | 49.0% | 46.3% | 50.7% | 48.7% | -0.13 |
| California Black Bears | New Jersey 5s | 16.6% | 14.9% | 18.4% | 16.2% | -5.00 |
| California Black Bears | Orlando Squeeze | 34.9% | 33.3% | 34.9% | 34.0% | -2.17 |
| California Black Bears | Palm Beach Royals | 43.8% | 41.0% | 46.2% | 43.5% | -0.84 |
| California Black Bears | Phoenix Flames | 60.0% | 57.3% | 61.5% | 60.0% | +1.39 |
| California Black Bears | SoCal Hard Eights | 63.7% | 60.8% | 68.2% | 63.4% | +1.86 |
| California Black Bears | St. Louis Shock | 22.6% | 21.9% | 25.9% | 22.3% | -3.87 |
| California Black Bears | Texas Ranchers | 32.0% | 29.6% | 35.7% | 31.3% | -2.58 |
| California Black Bears | Utah Black Diamonds | 48.5% | 46.7% | 50.9% | 48.3% | -0.16 |
| Carolina Hogs | Atlanta Bouncers | 18.6% | 18.6% | 18.8% | 18.1% | -4.71 |
| Carolina Hogs | Bay Area Breakers | 32.6% | 31.4% | 37.0% | 31.4% | -2.51 |
| Carolina Hogs | Brooklyn Pickleball Team | 18.1% | 16.3% | 22.8% | 18.1% | -4.54 |
| Carolina Hogs | California Black Bears | 20.6% | 18.8% | 24.2% | 20.0% | -4.32 |
| Carolina Hogs | Chicago Slice | 27.9% | 27.1% | 28.8% | 27.8% | -2.97 |
| Carolina Hogs | Columbus Sliders | 12.9% | 12.5% | 18.3% | 12.5% | -5.66 |
| Carolina Hogs | Dallas Flash | 27.5% | 26.1% | 28.1% | 27.2% | -3.16 |
| Carolina Hogs | Florida Smash | 43.5% | 41.9% | 44.6% | 43.5% | -0.81 |
| Carolina Hogs | Las Vegas Night Owls | 20.4% | 18.0% | 25.1% | 20.0% | -4.23 |
| Carolina Hogs | Los Angeles Mad Drops | 8.2% | 7.3% | 10.0% | 8.1% | -6.85 |
| Carolina Hogs | Miami Pickleball Club | 21.6% | 19.3% | 24.0% | 21.3% | -4.11 |
| Carolina Hogs | New Jersey 5s | 4.0% | 3.8% | 5.6% | 3.7% | -8.34 |
| Carolina Hogs | Orlando Squeeze | 13.4% | 13.4% | 13.4% | 12.8% | -5.65 |
| Carolina Hogs | Palm Beach Royals | 16.9% | 15.8% | 20.8% | 16.8% | -4.87 |
| Carolina Hogs | Phoenix Flames | 30.5% | 28.0% | 33.3% | 30.1% | -2.73 |
| Carolina Hogs | SoCal Hard Eights | 32.6% | 30.6% | 40.0% | 32.6% | -2.28 |
| Carolina Hogs | St. Louis Shock | 5.8% | 5.8% | 8.8% | 5.6% | -7.49 |
| Carolina Hogs | Texas Ranchers | 9.6% | 9.1% | 14.0% | 9.6% | -6.36 |
| Carolina Hogs | Utah Black Diamonds | 21.4% | 19.5% | 24.2% | 21.3% | -4.10 |
| Chicago Slice | Atlanta Bouncers | 34.9% | 33.6% | 38.0% | 34.8% | -2.02 |
| Chicago Slice | Bay Area Breakers | 52.1% | 50.8% | 59.8% | 50.1% | -0.03 |
| Chicago Slice | Brooklyn Pickleball Team | 34.4% | 30.1% | 42.7% | 34.1% | -1.99 |
| Chicago Slice | California Black Bears | 37.2% | 35.1% | 45.2% | 35.9% | -1.96 |
| Chicago Slice | Carolina Hogs | 68.6% | 65.2% | 72.7% | 67.7% | +2.42 |
| Chicago Slice | Columbus Sliders | 26.4% | 25.7% | 37.1% | 25.1% | -3.43 |
| Chicago Slice | Dallas Flash | 46.4% | 43.4% | 50.1% | 45.8% | -0.57 |
| Chicago Slice | Florida Smash | 64.8% | 60.5% | 67.2% | 64.5% | +1.96 |
| Chicago Slice | Las Vegas Night Owls | 36.8% | 33.3% | 46.0% | 35.6% | -1.90 |
| Chicago Slice | Los Angeles Mad Drops | 18.4% | 16.7% | 24.0% | 17.6% | -4.74 |
| Chicago Slice | Miami Pickleball Club | 38.3% | 35.2% | 45.0% | 37.2% | -1.75 |
| Chicago Slice | New Jersey 5s | 10.8% | 10.4% | 15.3% | 9.6% | -6.33 |
| Chicago Slice | Orlando Squeeze | 29.0% | 26.0% | 29.8% | 28.7% | -2.93 |
| Chicago Slice | Palm Beach Royals | 32.1% | 30.7% | 40.8% | 31.1% | -2.58 |
| Chicago Slice | Phoenix Flames | 49.3% | 46.4% | 55.9% | 48.2% | -0.30 |
| Chicago Slice | SoCal Hard Eights | 51.4% | 49.2% | 62.6% | 50.4% | +0.12 |
| Chicago Slice | St. Louis Shock | 14.1% | 14.1% | 22.0% | 13.3% | -5.45 |
| Chicago Slice | Texas Ranchers | 20.8% | 19.9% | 30.9% | 20.1% | -4.22 |
| Chicago Slice | Utah Black Diamonds | 38.0% | 35.3% | 45.2% | 37.0% | -1.75 |
| Columbus Sliders | Atlanta Bouncers | 50.1% | 49.6% | 51.2% | 49.6% | +0.00 |
| Columbus Sliders | Bay Area Breakers | 72.2% | 68.8% | 72.3% | 72.0% | +3.05 |
| Columbus Sliders | Brooklyn Pickleball Team | 56.4% | 52.8% | 57.4% | 54.4% | +0.56 |
| Columbus Sliders | California Black Bears | 58.8% | 55.2% | 58.9% | 58.0% | +1.01 |
| Columbus Sliders | Carolina Hogs | 81.6% | 81.6% | 82.6% | 81.0% | +4.55 |
| Columbus Sliders | Chicago Slice | 64.0% | 62.1% | 64.0% | 63.1% | +1.76 |
| Columbus Sliders | Dallas Flash | 62.2% | 61.8% | 63.3% | 61.5% | +1.57 |
| Columbus Sliders | Florida Smash | 78.0% | 76.9% | 78.2% | 77.3% | +3.89 |
| Columbus Sliders | Las Vegas Night Owls | 58.7% | 55.8% | 60.3% | 57.5% | +1.00 |
| Columbus Sliders | Los Angeles Mad Drops | 34.8% | 33.5% | 35.9% | 33.9% | -2.20 |
| Columbus Sliders | Miami Pickleball Club | 57.9% | 56.4% | 58.7% | 56.7% | +0.88 |
| Columbus Sliders | New Jersey 5s | 23.6% | 21.0% | 24.2% | 23.5% | -3.68 |
| Columbus Sliders | Orlando Squeeze | 42.4% | 40.6% | 42.4% | 41.8% | -1.07 |
| Columbus Sliders | Palm Beach Royals | 52.7% | 51.6% | 54.4% | 52.3% | +0.37 |
| Columbus Sliders | Phoenix Flames | 68.3% | 66.9% | 69.0% | 67.5% | +2.37 |
| Columbus Sliders | SoCal Hard Eights | 72.4% | 71.6% | 75.2% | 71.9% | +3.08 |
| Columbus Sliders | St. Louis Shock | 31.8% | 30.0% | 32.9% | 31.7% | -2.50 |
| Columbus Sliders | Texas Ranchers | 41.0% | 40.0% | 43.6% | 40.9% | -1.17 |
| Columbus Sliders | Utah Black Diamonds | 57.5% | 56.9% | 58.9% | 56.5% | +0.86 |
| Dallas Flash | Atlanta Bouncers | 38.2% | 37.4% | 38.5% | 37.9% | -1.72 |
| Dallas Flash | Bay Area Breakers | 55.0% | 54.1% | 60.4% | 53.4% | +0.45 |
| Dallas Flash | Brooklyn Pickleball Team | 36.3% | 33.8% | 43.5% | 36.3% | -1.70 |
| Dallas Flash | California Black Bears | 39.8% | 38.3% | 45.8% | 38.9% | -1.50 |
| Dallas Flash | Carolina Hogs | 71.1% | 69.0% | 73.2% | 70.9% | +2.92 |
| Dallas Flash | Chicago Slice | 50.3% | 48.4% | 51.4% | 50.3% | +0.10 |
| Dallas Flash | Columbus Sliders | 28.7% | 28.0% | 37.7% | 27.8% | -3.02 |
| Dallas Flash | Florida Smash | 66.7% | 64.3% | 67.8% | 66.6% | +2.29 |
| Dallas Flash | Las Vegas Night Owls | 39.2% | 36.6% | 46.8% | 38.6% | -1.43 |
| Dallas Flash | Los Angeles Mad Drops | 20.1% | 18.9% | 24.4% | 19.9% | -4.33 |
| Dallas Flash | Miami Pickleball Club | 40.8% | 38.6% | 45.6% | 40.5% | -1.26 |
| Dallas Flash | New Jersey 5s | 11.8% | 11.6% | 15.7% | 10.9% | -6.06 |
| Dallas Flash | Orlando Squeeze | 30.3% | 29.5% | 30.3% | 29.6% | -2.80 |
| Dallas Flash | Palm Beach Royals | 34.5% | 33.4% | 41.3% | 34.3% | -2.12 |
| Dallas Flash | Phoenix Flames | 52.0% | 50.0% | 56.6% | 51.6% | +0.20 |
| Dallas Flash | SoCal Hard Eights | 54.1% | 52.6% | 63.3% | 53.9% | +0.62 |
| Dallas Flash | St. Louis Shock | 15.7% | 15.6% | 22.3% | 15.3% | -5.08 |
| Dallas Flash | Texas Ranchers | 23.0% | 22.1% | 31.3% | 22.8% | -3.79 |
| Dallas Flash | Utah Black Diamonds | 40.6% | 38.7% | 45.8% | 40.4% | -1.26 |
| Florida Smash | Atlanta Bouncers | 22.0% | 21.1% | 23.2% | 21.8% | -4.01 |
| Florida Smash | Bay Area Breakers | 36.9% | 35.6% | 43.0% | 35.4% | -2.00 |
| Florida Smash | Brooklyn Pickleball Team | 21.7% | 18.6% | 27.5% | 21.6% | -3.90 |
| Florida Smash | California Black Bears | 24.0% | 22.1% | 29.3% | 23.1% | -3.84 |
| Florida Smash | Carolina Hogs | 54.0% | 50.5% | 57.1% | 53.4% | +0.45 |
| Florida Smash | Chicago Slice | 33.4% | 30.1% | 34.3% | 33.4% | -2.22 |
| Florida Smash | Columbus Sliders | 15.6% | 15.1% | 22.7% | 14.8% | -5.20 |
| Florida Smash | Dallas Flash | 31.5% | 29.2% | 33.6% | 31.4% | -2.55 |
| Florida Smash | Las Vegas Night Owls | 23.6% | 20.8% | 30.2% | 23.0% | -3.78 |
| Florida Smash | Los Angeles Mad Drops | 10.0% | 8.8% | 13.0% | 9.7% | -6.44 |
| Florida Smash | Miami Pickleball Club | 24.9% | 22.2% | 29.1% | 24.2% | -3.65 |
| Florida Smash | New Jersey 5s | 5.2% | 4.9% | 7.5% | 4.6% | -7.89 |
| Florida Smash | Orlando Squeeze | 17.0% | 15.4% | 17.0% | 16.4% | -5.01 |
| Florida Smash | Palm Beach Royals | 19.9% | 18.6% | 25.5% | 19.4% | -4.42 |
| Florida Smash | Phoenix Flames | 34.5% | 31.7% | 39.2% | 33.8% | -2.25 |
| Florida Smash | SoCal Hard Eights | 36.5% | 34.4% | 46.1% | 35.9% | -1.81 |
| Florida Smash | St. Louis Shock | 7.3% | 7.2% | 11.6% | 7.0% | -7.06 |
| Florida Smash | Texas Ranchers | 11.7% | 11.1% | 17.8% | 11.3% | -5.93 |
| Florida Smash | Utah Black Diamonds | 24.6% | 22.3% | 29.3% | 24.1% | -3.65 |
| Las Vegas Night Owls | Atlanta Bouncers | 42.9% | 42.5% | 43.5% | 40.9% | -1.22 |
| Las Vegas Night Owls | Bay Area Breakers | 60.6% | 58.1% | 64.7% | 60.3% | +1.45 |
| Las Vegas Night Owls | Brooklyn Pickleball Team | 46.8% | 40.1% | 48.9% | 46.1% | -0.52 |
| Las Vegas Night Owls | California Black Bears | 47.1% | 43.2% | 50.5% | 46.7% | -0.39 |
| Las Vegas Night Owls | Carolina Hogs | 75.5% | 74.1% | 76.9% | 74.7% | +3.52 |
| Las Vegas Night Owls | Chicago Slice | 56.1% | 53.5% | 56.5% | 54.2% | +0.46 |
| Las Vegas Night Owls | Columbus Sliders | 35.6% | 33.2% | 41.9% | 35.5% | -1.83 |
| Las Vegas Night Owls | Dallas Flash | 54.2% | 53.2% | 55.6% | 52.9% | +0.24 |
| Las Vegas Night Owls | Florida Smash | 71.1% | 69.2% | 72.0% | 70.2% | +2.80 |
| Las Vegas Night Owls | Los Angeles Mad Drops | 26.3% | 23.3% | 28.5% | 25.8% | -3.38 |
| Las Vegas Night Owls | Miami Pickleball Club | 48.7% | 44.5% | 50.5% | 48.0% | -0.27 |
| Las Vegas Night Owls | New Jersey 5s | 14.7% | 13.8% | 18.3% | 14.5% | -5.22 |
| Las Vegas Night Owls | Orlando Squeeze | 35.2% | 34.1% | 35.2% | 33.1% | -2.27 |
| Las Vegas Night Owls | Palm Beach Royals | 42.1% | 39.0% | 46.0% | 41.8% | -1.06 |
| Las Vegas Night Owls | Phoenix Flames | 59.5% | 55.7% | 61.2% | 58.7% | +1.17 |
| Las Vegas Night Owls | SoCal Hard Eights | 64.5% | 58.5% | 67.6% | 64.1% | +1.92 |
| Las Vegas Night Owls | St. Louis Shock | 20.8% | 19.9% | 25.7% | 20.5% | -4.09 |
| Las Vegas Night Owls | Texas Ranchers | 29.5% | 27.5% | 35.4% | 29.4% | -2.73 |
| Las Vegas Night Owls | Utah Black Diamonds | 48.1% | 44.8% | 50.7% | 47.6% | -0.31 |
| Los Angeles Mad Drops | Atlanta Bouncers | 66.3% | 66.0% | 66.5% | 65.3% | +2.05 |
| Los Angeles Mad Drops | Bay Area Breakers | 81.6% | 80.3% | 83.6% | 81.3% | +4.61 |
| Los Angeles Mad Drops | Brooklyn Pickleball Team | 68.2% | 64.2% | 71.1% | 67.5% | +2.44 |
| Los Angeles Mad Drops | California Black Bears | 71.2% | 67.7% | 73.0% | 70.9% | +2.90 |
| Los Angeles Mad Drops | Carolina Hogs | 90.3% | 89.8% | 90.8% | 90.3% | +6.53 |
| Los Angeles Mad Drops | Chicago Slice | 77.2% | 76.0% | 77.4% | 76.5% | +3.68 |
| Los Angeles Mad Drops | Columbus Sliders | 60.2% | 58.5% | 65.5% | 59.3% | +1.26 |
| Los Angeles Mad Drops | Dallas Flash | 76.6% | 75.9% | 76.8% | 75.9% | +3.56 |
| Los Angeles Mad Drops | Florida Smash | 87.7% | 86.9% | 88.0% | 87.5% | +5.67 |
| Los Angeles Mad Drops | Las Vegas Night Owls | 70.8% | 66.8% | 73.8% | 70.7% | +2.93 |
| Los Angeles Mad Drops | Miami Pickleball Club | 72.1% | 68.8% | 72.8% | 72.0% | +3.07 |
| Los Angeles Mad Drops | New Jersey 5s | 34.1% | 32.6% | 38.0% | 33.1% | -2.27 |
| Los Angeles Mad Drops | Orlando Squeeze | 58.0% | 57.3% | 58.0% | 56.8% | +0.91 |
| Los Angeles Mad Drops | Palm Beach Royals | 66.8% | 63.7% | 69.1% | 66.7% | +2.30 |
| Los Angeles Mad Drops | Phoenix Flames | 80.6% | 78.0% | 81.1% | 80.5% | +4.45 |
| Los Angeles Mad Drops | SoCal Hard Eights | 82.9% | 80.1% | 85.4% | 82.7% | +4.90 |
| Los Angeles Mad Drops | St. Louis Shock | 42.3% | 42.1% | 47.9% | 42.0% | -1.05 |
| Los Angeles Mad Drops | Texas Ranchers | 53.4% | 51.5% | 58.9% | 53.1% | +0.44 |
| Los Angeles Mad Drops | Utah Black Diamonds | 72.0% | 69.0% | 73.0% | 72.0% | +3.07 |
| Miami Pickleball Club | Atlanta Bouncers | 43.6% | 43.2% | 43.9% | 42.4% | -1.03 |
| Miami Pickleball Club | Bay Area Breakers | 61.9% | 60.1% | 65.5% | 61.3% | +1.56 |
| Miami Pickleball Club | Brooklyn Pickleball Team | 45.1% | 40.8% | 49.2% | 44.5% | -0.68 |
| Miami Pickleball Club | California Black Bears | 48.3% | 44.4% | 51.2% | 47.8% | -0.27 |
| Miami Pickleball Club | Carolina Hogs | 76.3% | 75.0% | 77.5% | 76.2% | +3.76 |
| Miami Pickleball Club | Chicago Slice | 56.3% | 54.6% | 56.9% | 55.6% | +0.64 |
| Miami Pickleball Club | Columbus Sliders | 36.5% | 35.0% | 42.7% | 35.5% | -1.94 |
| Miami Pickleball Club | Dallas Flash | 55.1% | 54.3% | 56.1% | 54.7% | +0.47 |
| Miami Pickleball Club | Florida Smash | 71.8% | 70.2% | 72.5% | 71.6% | +3.04 |
| Miami Pickleball Club | Las Vegas Night Owls | 48.1% | 43.4% | 52.3% | 48.0% | -0.19 |
| Miami Pickleball Club | Los Angeles Mad Drops | 27.1% | 23.9% | 28.9% | 27.1% | -3.19 |
| Miami Pickleball Club | New Jersey 5s | 15.7% | 14.8% | 18.8% | 15.0% | -5.19 |
| Miami Pickleball Club | Orlando Squeeze | 35.3% | 34.6% | 35.3% | 34.1% | -2.13 |
| Miami Pickleball Club | Palm Beach Royals | 43.3% | 40.0% | 46.6% | 42.9% | -0.96 |
| Miami Pickleball Club | Phoenix Flames | 60.5% | 56.8% | 61.9% | 60.4% | +1.42 |
| Miami Pickleball Club | SoCal Hard Eights | 63.5% | 59.7% | 68.3% | 63.2% | +1.83 |
| Miami Pickleball Club | St. Louis Shock | 21.4% | 21.2% | 26.3% | 21.0% | -4.11 |
| Miami Pickleball Club | Texas Ranchers | 30.1% | 28.7% | 36.1% | 29.7% | -2.74 |
| Miami Pickleball Club* | Utah Black Diamonds | 49.6% | 45.7% | 51.2% | 49.4% | -0.07 |
| New Jersey 5s | Atlanta Bouncers | 74.3% | 73.8% | 76.4% | 74.0% | +3.50 |
| New Jersey 5s | Bay Area Breakers | 89.8% | 86.8% | 90.2% | 89.8% | +6.41 |
| New Jersey 5s | Brooklyn Pickleball Team | 77.8% | 74.7% | 80.9% | 76.7% | +3.82 |
| New Jersey 5s | California Black Bears | 80.4% | 77.4% | 82.3% | 80.2% | +4.48 |
| New Jersey 5s | Carolina Hogs | 93.8% | 93.7% | 94.8% | 93.6% | +7.61 |
| New Jersey 5s | Chicago Slice | 84.6% | 82.6% | 85.3% | 84.0% | +5.17 |
| New Jersey 5s | Columbus Sliders | 73.7% | 70.5% | 76.3% | 73.1% | +3.23 |
| New Jersey 5s | Dallas Flash | 83.2% | 82.8% | 84.8% | 82.8% | +4.98 |
| New Jersey 5s | Florida Smash | 92.3% | 91.3% | 92.9% | 92.0% | +7.02 |
| New Jersey 5s | Las Vegas Night Owls | 79.6% | 77.3% | 83.0% | 79.2% | +4.27 |
| New Jersey 5s | Los Angeles Mad Drops | 59.3% | 57.4% | 63.3% | 59.0% | +1.32 |
| New Jersey 5s | Miami Pickleball Club | 79.6% | 77.9% | 82.1% | 79.2% | +4.34 |
| New Jersey 5s | Orlando Squeeze | 67.7% | 65.5% | 69.0% | 67.2% | +2.43 |
| New Jersey 5s | Palm Beach Royals | 75.6% | 74.4% | 79.1% | 75.5% | +3.71 |
| New Jersey 5s | Phoenix Flames | 86.7% | 85.2% | 88.3% | 86.5% | +5.72 |
| New Jersey 5s | SoCal Hard Eights | 88.5% | 88.1% | 91.5% | 88.4% | +6.14 |
| New Jersey 5s | St. Louis Shock | 56.6% | 54.7% | 60.4% | 55.9% | +0.83 |
| New Jersey 5s | Texas Ranchers | 65.7% | 64.8% | 70.7% | 65.0% | +2.08 |
| New Jersey 5s | Utah Black Diamonds | 79.2% | 78.3% | 82.2% | 78.9% | +4.30 |
| Orlando Squeeze | Atlanta Bouncers | 55.9% | 54.4% | 59.4% | 55.3% | +0.71 |
| Orlando Squeeze | Bay Area Breakers | 70.0% | 69.5% | 78.5% | 68.3% | +2.44 |
| Orlando Squeeze | Brooklyn Pickleball Team | 52.4% | 49.9% | 63.8% | 52.0% | +0.31 |
| Orlando Squeeze | California Black Bears | 55.4% | 54.4% | 66.2% | 54.2% | +0.52 |
| Orlando Squeeze | Carolina Hogs | 82.7% | 82.0% | 87.3% | 82.2% | +4.80 |
| Orlando Squeeze | Chicago Slice | 68.0% | 65.5% | 71.3% | 67.9% | +2.49 |
| Orlando Squeeze | Columbus Sliders | 43.6% | 42.8% | 58.4% | 42.3% | -1.03 |
| Orlando Squeeze | Dallas Flash | 65.5% | 64.2% | 70.7% | 64.7% | +1.97 |
| Orlando Squeeze | Florida Smash | 80.3% | 78.7% | 83.8% | 79.9% | +4.23 |
| Orlando Squeeze | Las Vegas Night Owls | 54.6% | 53.4% | 67.0% | 53.6% | +0.56 |
| Orlando Squeeze | Los Angeles Mad Drops | 33.0% | 32.1% | 43.5% | 32.1% | -2.38 |
| Orlando Squeeze | Miami Pickleball Club | 56.4% | 55.4% | 66.0% | 55.5% | +0.75 |
| Orlando Squeeze | New Jersey 5s | 22.2% | 21.8% | 31.6% | 20.4% | -4.17 |
| Orlando Squeeze | Palm Beach Royals | 49.8% | 48.8% | 62.1% | 48.9% | -0.11 |
| Orlando Squeeze | Phoenix Flames | 67.1% | 66.2% | 75.5% | 66.2% | +2.20 |
| Orlando Squeeze | SoCal Hard Eights | 68.7% | 67.5% | 80.5% | 67.9% | +2.57 |
| Orlando Squeeze | St. Louis Shock | 27.4% | 27.3% | 40.9% | 26.4% | -3.18 |
| Orlando Squeeze | Texas Ranchers | 36.5% | 35.6% | 51.7% | 35.7% | -1.86 |
| Orlando Squeeze | Utah Black Diamonds | 56.2% | 55.1% | 66.3% | 55.3% | +0.76 |
| Palm Beach Royals | Atlanta Bouncers | 47.9% | 47.6% | 47.9% | 47.0% | -0.42 |
| Palm Beach Royals | Bay Area Breakers | 67.2% | 65.8% | 69.2% | 66.8% | +2.33 |
| Palm Beach Royals | Brooklyn Pickleball Team | 50.7% | 47.6% | 53.6% | 49.7% | -0.03 |
| Palm Beach Royals | California Black Bears | 54.0% | 51.1% | 55.4% | 53.7% | +0.52 |
| Palm Beach Royals | Carolina Hogs | 80.0% | 80.0% | 80.5% | 79.5% | +4.13 |
| Palm Beach Royals | Chicago Slice | 60.8% | 60.4% | 60.8% | 59.8% | +1.29 |
| Palm Beach Royals | Columbus Sliders | 42.9% | 41.1% | 46.8% | 42.2% | -1.06 |
| Palm Beach Royals | Dallas Flash | 60.1% | 59.7% | 60.1% | 59.2% | +1.15 |
| Palm Beach Royals | Florida Smash | 75.8% | 75.2% | 75.8% | 75.0% | +3.45 |
| Palm Beach Royals | Las Vegas Night Owls | 53.6% | 50.3% | 56.6% | 53.2% | +0.46 |
| Palm Beach Royals | Los Angeles Mad Drops | 31.7% | 29.5% | 32.6% | 31.5% | -2.52 |
| Palm Beach Royals | Miami Pickleball Club | 54.4% | 52.3% | 55.2% | 54.1% | +0.57 |
| Palm Beach Royals | New Jersey 5s | 19.4% | 18.6% | 21.7% | 18.8% | -4.50 |
| Palm Beach Royals | Orlando Squeeze | 39.2% | 38.8% | 39.2% | 38.2% | -1.54 |
| Palm Beach Royals | Phoenix Flames | 65.0% | 63.4% | 65.8% | 65.0% | +2.08 |
| Palm Beach Royals | SoCal Hard Eights | 68.9% | 66.2% | 72.1% | 68.9% | +2.65 |
| Palm Beach Royals | St. Louis Shock | 26.5% | 26.1% | 29.7% | 26.2% | -3.25 |
| Palm Beach Royals | Texas Ranchers | 36.4% | 34.5% | 40.0% | 36.1% | -1.86 |
| Palm Beach Royals | Utah Black Diamonds | 54.4% | 52.6% | 55.4% | 54.2% | +0.57 |
| Phoenix Flames | Atlanta Bouncers | 32.7% | 32.4% | 33.1% | 31.9% | -2.48 |
| Phoenix Flames | Bay Area Breakers | 51.6% | 49.7% | 54.7% | 50.9% | +0.14 |
| Phoenix Flames | Brooklyn Pickleball Team | 34.2% | 30.9% | 38.4% | 33.5% | -2.18 |
| Phoenix Flames | California Black Bears | 37.6% | 34.2% | 40.1% | 37.3% | -1.72 |
| Phoenix Flames | Carolina Hogs | 66.8% | 65.8% | 68.2% | 66.7% | +2.33 |
| Phoenix Flames | Chicago Slice | 45.6% | 43.5% | 45.7% | 44.6% | -0.77 |
| Phoenix Flames | Columbus Sliders | 27.0% | 25.9% | 32.1% | 26.1% | -3.33 |
| Phoenix Flames | Dallas Flash | 44.2% | 43.2% | 44.9% | 43.5% | -0.96 |
| Phoenix Flames | Florida Smash | 62.0% | 60.1% | 62.4% | 61.5% | +1.40 |
| Phoenix Flames | Las Vegas Night Owls | 36.9% | 33.3% | 41.3% | 36.7% | -1.71 |
| Phoenix Flames | Los Angeles Mad Drops | 18.7% | 16.4% | 20.1% | 18.6% | -4.57 |
| Phoenix Flames | Miami Pickleball Club | 38.6% | 35.1% | 39.9% | 38.5% | -1.55 |
| Phoenix Flames | New Jersey 5s | 10.2% | 9.6% | 12.2% | 9.7% | -6.39 |
| Phoenix Flames | Orlando Squeeze | 25.5% | 24.8% | 25.5% | 24.7% | -3.52 |
| Phoenix Flames | Palm Beach Royals | 33.1% | 30.1% | 35.7% | 32.6% | -2.40 |
| Phoenix Flames | SoCal Hard Eights | 52.6% | 49.1% | 57.9% | 52.1% | +0.36 |
| Phoenix Flames | St. Louis Shock | 14.5% | 14.4% | 18.0% | 14.1% | -5.40 |
| Phoenix Flames | Texas Ranchers | 21.5% | 20.4% | 26.2% | 21.1% | -4.11 |
| Phoenix Flames | Utah Black Diamonds | 38.1% | 35.4% | 40.1% | 38.1% | -1.59 |
| SoCal Hard Eights | Atlanta Bouncers | 27.7% | 27.7% | 27.7% | 26.2% | -3.29 |
| SoCal Hard Eights | Bay Area Breakers | 44.6% | 44.1% | 48.0% | 44.5% | -0.68 |
| SoCal Hard Eights | Brooklyn Pickleball Team | 30.4% | 27.5% | 32.9% | 29.7% | -2.75 |
| SoCal Hard Eights | California Black Bears | 31.7% | 29.9% | 34.0% | 31.5% | -2.49 |
| SoCal Hard Eights | Carolina Hogs | 61.5% | 61.5% | 62.3% | 60.5% | +1.22 |
| SoCal Hard Eights | Chicago Slice | 39.5% | 39.4% | 39.6% | 37.5% | -1.65 |
| SoCal Hard Eights | Columbus Sliders | 22.5% | 21.2% | 26.4% | 22.3% | -3.86 |
| SoCal Hard Eights | Dallas Flash | 38.7% | 38.7% | 38.7% | 36.8% | -1.80 |
| SoCal Hard Eights | Florida Smash | 55.8% | 55.7% | 56.2% | 54.2% | +0.48 |
| SoCal Hard Eights | Las Vegas Night Owls | 33.2% | 29.4% | 35.3% | 33.1% | -2.27 |
| SoCal Hard Eights | Los Angeles Mad Drops | 15.1% | 13.9% | 16.0% | 14.8% | -5.29 |
| SoCal Hard Eights | Miami Pickleball Club | 32.6% | 31.1% | 34.0% | 32.1% | -2.40 |
| SoCal Hard Eights | New Jersey 5s | 7.5% | 7.2% | 9.2% | 7.4% | -6.94 |
| SoCal Hard Eights | Orlando Squeeze | 21.0% | 20.9% | 21.0% | 19.6% | -4.26 |
| SoCal Hard Eights | Palm Beach Royals | 28.3% | 26.2% | 29.9% | 27.9% | -3.06 |
| SoCal Hard Eights | Phoenix Flames | 43.4% | 41.6% | 44.4% | 42.6% | -0.96 |
| SoCal Hard Eights | St. Louis Shock | 11.4% | 11.4% | 14.0% | 11.3% | -5.93 |
| SoCal Hard Eights | Texas Ranchers | 18.3% | 17.0% | 21.2% | 18.2% | -4.59 |
| SoCal Hard Eights | Utah Black Diamonds | 32.9% | 31.4% | 34.2% | 32.3% | -2.40 |
| St. Louis Shock | Atlanta Bouncers | 67.6% | 67.2% | 68.1% | 67.3% | +2.42 |
| St. Louis Shock | Bay Area Breakers | 84.7% | 83.3% | 84.9% | 84.6% | +5.30 |
| St. Louis Shock | Brooklyn Pickleball Team | 72.2% | 70.3% | 73.3% | 70.4% | +2.79 |
| St. Louis Shock | California Black Bears | 74.7% | 72.7% | 74.8% | 74.1% | +3.41 |
| St. Louis Shock | Carolina Hogs | 91.2% | 91.2% | 91.6% | 90.9% | +6.71 |
| St. Louis Shock | Chicago Slice | 78.6% | 77.7% | 78.6% | 78.0% | +4.09 |
| St. Louis Shock | Columbus Sliders | 66.1% | 64.7% | 67.5% | 66.0% | +2.20 |
| St. Louis Shock | Dallas Flash | 77.8% | 77.5% | 78.2% | 77.3% | +3.96 |
| St. Louis Shock | Florida Smash | 88.8% | 88.3% | 88.8% | 88.4% | +6.09 |
| St. Louis Shock | Las Vegas Night Owls | 74.5% | 73.0% | 75.8% | 73.3% | +3.27 |
| St. Louis Shock | Los Angeles Mad Drops | 52.8% | 51.7% | 53.1% | 51.8% | +0.24 |
| St. Louis Shock | Miami Pickleball Club | 74.4% | 73.6% | 74.6% | 73.5% | +3.29 |
| St. Louis Shock | New Jersey 5s | 38.7% | 37.0% | 39.9% | 38.3% | -1.50 |
| St. Louis Shock | Orlando Squeeze | 59.6% | 58.5% | 59.6% | 59.2% | +1.29 |
| St. Louis Shock | Palm Beach Royals | 70.4% | 69.5% | 71.0% | 69.7% | +2.74 |
| St. Louis Shock | Phoenix Flames | 82.5% | 81.8% | 82.5% | 81.9% | +4.70 |
| St. Louis Shock | SoCal Hard Eights | 85.0% | 84.9% | 86.9% | 84.6% | +5.24 |
| St. Louis Shock | Texas Ranchers | 58.9% | 58.6% | 61.1% | 58.9% | +1.21 |
| St. Louis Shock | Utah Black Diamonds | 74.2% | 74.0% | 74.8% | 73.3% | +3.28 |
| Texas Ranchers | Atlanta Bouncers | 58.3% | 58.2% | 58.3% | 57.6% | +1.02 |
| Texas Ranchers | Bay Area Breakers | 76.9% | 76.2% | 77.8% | 76.8% | +3.92 |
| Texas Ranchers | Brooklyn Pickleball Team | 62.6% | 61.1% | 64.0% | 61.1% | +1.49 |
| Texas Ranchers | California Black Bears | 65.0% | 63.9% | 65.6% | 64.9% | +2.10 |
| Texas Ranchers | Carolina Hogs | 86.8% | 86.8% | 86.8% | 86.2% | +5.48 |
| Texas Ranchers | Chicago Slice | 70.3% | 70.2% | 70.3% | 69.4% | +2.69 |
| Texas Ranchers | Columbus Sliders | 55.8% | 54.1% | 57.5% | 55.4% | +0.73 |
| Texas Ranchers | Dallas Flash | 69.8% | 69.7% | 69.8% | 69.2% | +2.58 |
| Texas Ranchers | Florida Smash | 83.1% | 83.0% | 83.1% | 82.5% | +4.79 |
| Texas Ranchers | Las Vegas Night Owls | 65.5% | 63.6% | 66.8% | 64.6% | +1.99 |
| Texas Ranchers | Los Angeles Mad Drops | 42.2% | 41.9% | 42.7% | 41.5% | -1.24 |
| Texas Ranchers | Miami Pickleball Club | 65.3% | 65.0% | 65.5% | 64.4% | +1.86 |
| Texas Ranchers | New Jersey 5s | 28.3% | 27.6% | 30.1% | 27.9% | -2.98 |
| Texas Ranchers | Orlando Squeeze | 49.4% | 49.2% | 49.4% | 48.6% | -0.13 |
| Texas Ranchers | Palm Beach Royals | 60.7% | 60.2% | 61.4% | 60.5% | +1.44 |
| Texas Ranchers | Phoenix Flames | 74.9% | 74.6% | 74.9% | 74.2% | +3.33 |
| Texas Ranchers | SoCal Hard Eights | 79.3% | 77.7% | 80.4% | 78.8% | +4.15 |
| Texas Ranchers | St. Louis Shock | 38.1% | 37.8% | 39.5% | 38.0% | -1.60 |
| Texas Ranchers | Utah Black Diamonds | 65.7% | 65.5% | 65.7% | 64.6% | +1.89 |
| Utah Black Diamonds | Atlanta Bouncers | 43.7% | 43.4% | 43.7% | 42.5% | -1.03 |
| Utah Black Diamonds | Bay Area Breakers | 61.9% | 60.4% | 65.3% | 61.2% | +1.52 |
| Utah Black Diamonds | Brooklyn Pickleball Team | 44.9% | 41.2% | 49.0% | 44.4% | -0.70 |
| Utah Black Diamonds | California Black Bears | 48.3% | 44.7% | 50.9% | 47.7% | -0.30 |
| Utah Black Diamonds | Carolina Hogs | 76.5% | 75.3% | 77.3% | 76.4% | +3.77 |
| Utah Black Diamonds | Chicago Slice | 55.8% | 55.2% | 56.6% | 55.4% | +0.60 |
| Utah Black Diamonds | Columbus Sliders | 36.4% | 35.3% | 42.5% | 35.6% | -1.91 |
| Utah Black Diamonds | Dallas Flash | 55.3% | 54.9% | 55.9% | 54.7% | +0.49 |
| Utah Black Diamonds | Florida Smash | 71.5% | 70.8% | 72.3% | 71.4% | +3.03 |
| Utah Black Diamonds | Las Vegas Night Owls | 47.9% | 43.8% | 52.1% | 47.8% | -0.24 |
| Utah Black Diamonds | Los Angeles Mad Drops | 27.3% | 24.2% | 28.7% | 27.1% | -3.19 |
| Utah Black Diamonds* | Miami Pickleball Club | 49.6% | 45.9% | 50.8% | 49.5% | -0.05 |
| Utah Black Diamonds | New Jersey 5s | 15.7% | 14.9% | 18.7% | 15.0% | -5.20 |
| Utah Black Diamonds | Orlando Squeeze | 35.1% | 34.9% | 35.1% | 33.9% | -2.14 |
| Utah Black Diamonds | Palm Beach Royals | 43.3% | 40.4% | 46.4% | 43.1% | -0.92 |
| Utah Black Diamonds | Phoenix Flames | 60.3% | 57.2% | 61.7% | 60.2% | +1.38 |
| Utah Black Diamonds | SoCal Hard Eights | 63.5% | 60.1% | 68.1% | 63.4% | +1.87 |
| Utah Black Diamonds | St. Louis Shock | 21.4% | 21.3% | 26.1% | 21.0% | -4.08 |
| Utah Black Diamonds | Texas Ranchers | 30.1% | 28.9% | 35.9% | 29.9% | -2.71 |
