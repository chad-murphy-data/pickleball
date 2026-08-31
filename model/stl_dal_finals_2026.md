# STL Shock vs Dallas Flash — 2026 MLP Finals pricing (baseline vs Dallas hot streak)

*2026-08-27. Reproduce with `python model/stl_dal_finals_2026.py` (offline,
stdlib, ~5 s). Companion to the standing `data/forecasts.json` numbers; the
baseline here reproduces them exactly.*

The Finals in NYC are a **best-of-3 matchup series** (three STL–DAL meetings
on the 8/28–8/29 schedule; series ends when one team takes two). Lineups
below are the production best-lineup projections — STL: Bright/Fahey,
Patriquin/Tardio, Bright+Patriquin MX1, Fahey+Tardio MX2 (Fahey at her
current value, 1.081); Dallas: Townsend/Truong, JW Johnson/Ge, Townsend+JW
MX1, Truong+Ge MX2.

**A fact that frames everything: current v2 values were last refit
2026-08-03 — before the playoffs.** Every playoff game (Dallas 8/7–8/8,
Newport Beach 8/14–8/17) is out-of-sample for the ratings, so "the hot
streak" is measured exactly on games the model has never seen.

## Scenario 1 — baseline, current ratings

| game | STL pair | DAL pair | STL win | modal |
|---|---|---|---|---|
| WD | Bright/Fahey | Townsend/Truong | **89.8%** | 11-5 |
| MD | Patriquin/Tardio | JW/Ge | **80.3%** | 11-6 |
| MX1 | Bright/Patriquin | Townsend/JW | **72.7%** | 11-7 |
| MX2 | Fahey/Tardio | Truong/Ge | **91.0%** | 11-5 |

Matchup: **STL 96.1%** (4-0 47.7 / 3-1 39.7 / 2-2 11.2 with STL 76.5% in the
DreamBreaker / 1-3 1.3 / 0-4 0.0). **Best-of-3 series: STL 99.55%, Dallas
0.45%** (well inside the house eps floor territory — never display as 100%).

## Scenario 2 — Dallas playoff hot streak taken at face value

Measure: per game, team logit surplus = logit(observed point share) − v2
predicted eta; each player gets half their mean team surplus as a rating
bump. Window = the playoff run (since 8/7), where Dallas went **12-7 in
games** while beating rating expectations:

| player | n | team surplus (logit) | se | bump |
|---|---|---|---|---|
| JW Johnson | 10 | +0.646 | 0.282 | +0.323 |
| Augustus Ge | 9 | +0.495 | 0.314 | +0.248 |
| Danni-Elle Townsend | 9 | +0.353 | 0.266 | +0.177 |
| Alix Truong | 5 | +0.162 | 0.251 | +0.081 |

(For scale: 0.3 logit is a big move — the whole Bright↔Truong gap is 0.53.)

| game | STL win | modal |
|---|---|---|
| WD | **78.1%** | 11-7 |
| MD | **37.0%** | 8-11 |
| MX1 | **34.3%** | 8-11 |
| MX2 | **75.4%** | 11-7 |

Matchup: **STL 70.8%** (4-0 7.5 / 3-1 31.6 / 2-2 41.5, DB unchanged at STL
76.5% — the streak is doubles evidence, singles values untouched / 1-3 17.3
/ 0-4 2.2). **Best-of-3: STL 79.35%, Dallas 20.65%.**

So the streak, believed literally, moves Dallas from ~1-in-220 to ~1-in-5 in
the series, flips both men-involved games (MD and MX1 become Dallas
favorites), and makes a 2-2-into-DreamBreaker the modal matchup path (41.5%).

## Why face value is too generous — three checks

1. **STL ran hotter.** Same measure, same window: STL went **6-0** with mean
   team surplus bigger than Dallas's (Bright +0.30, Fahey +0.46, Patriquin
   +0.56, Tardio +0.98 — the last two on 2-and-4-game samples). Apply the
   identical logic to all 8 players and STL goes *up*: matchup 98.3%,
   series 99.92%. A one-sided "their streak is real, ours isn't" is the
   most Dallas-favorable framing available.
2. **Noise.** Only JW's surplus clears ~2 se; Truong's is 0.6 se. A 19-game
   team sample of point-share residuals (sd ≈ 0.35 logit/game, the known
   match overdispersion) supports bumps of this size about as well as it
   supports half of them.
3. **Window sensitivity.** Since 7/23 (43 games, 31-12) the surpluses
   shrink roughly by half: matchup STL 85.5%, series 94.27%. The face-value
   number is the *most recent, smallest window* — the maximum-streak read.

Honest summary band: **STL series probability ~79% (pure Dallas-hot, playoff
window) to ~99.6% (baseline)**, with the defensible center — shrink the
streak, or credit both teams — sitting in the low-to-mid 90s.

## Lineup variant — Dallas's actual playoff mixed

Dallas never ran the projected mixed in the playoffs: they played
**Townsend+Ge** and **Buckner+JW** in every matchup. Priced with Buckner
(value 0.726, singles 1.76, playoff surplus +0.874 on n=5 — the hottest and
noisiest number on the team):

- Baseline values: matchup STL 96.6%, series 99.66% (MX1 91.8 / MX2 81.4;
  DB drops to STL 71.5% because Buckner's singles value enters the roster
  mean).
- Dallas hot: **matchup STL 61.8%, series 67.41%** — Buckner+JW become a
  78% favorite over Fahey/Tardio in MX2. That entire swing rides on 5
  games of Buckner mixed; se-wise it is the least trustworthy row here,
  but it is also the lineup Dallas actually used to get to New York.

forecast.html reprices client-side the moment official lineups publish, so
the projected-vs-actual gap resolves itself at announcement time.

## Notes

- Method is session-defined (2026-08-27): the earlier NJ-vs-Dallas hot-streak
  session (2026-08-26) left no PR/branch/file, so nothing here inherits from
  it. If those numbers differed, this file is the reproducible reference.
- Everything uses production conventions: race-to-11 DP, weakest-link
  γ = −0.1829, display calibration (`web/calibration.json`), DreamBreaker =
  race-to-21 at K_DB_SINGLES × mean-roster singles gap, matchup tree with the
  eps floor, series = p²(3−2p).
- Not frozen into `model/receipts.json`; this is analysis, not a committed
  forecast. The standing receipts path is `web/make_forecast.py --commit`.
