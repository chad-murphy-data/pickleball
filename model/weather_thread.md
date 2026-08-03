# The weather thread — full session record (2026-07-28)

One afternoon, one question: **does weather affect pro pickleball, and is
there a good side / bad side?** Answer: no, at every grain the archive
reaches. This file is the narrative record — what was built, what was
tested, what the reasoning turns were, and what's left. The per-analysis
numbers live in `model/weather_report.md`, `model/end_effects.md`,
`model/favorites_wind.md`, `model/wind_skill.md`; this is the thing to
re-read first when picking the thread back up.

Branch: `claude/weather-condition-effects-7updhz` · PR #37 (open, unmerged
as of this writing) · 16 commits.

---

## 1. What got built (the durable part)

| Artifact | What it is |
|---|---|
| `scraper/weather.py` | Resolves all 110 events' venue lat/lon/tz from the open BFF (PPA: `getTournamentsOnDate` carries `Latitude`/`Longitude`; MLP: `getTeamLeaguesResultsOnDate` → `location` object with coords + IANA tz), then pulls Open-Meteo ERA5. Writes `data/event_geo.csv`, `event_weather.csv` (daily), `event_weather_hourly.csv` (venue-local hourly). Cached under `raw/weather/`. |
| `scraper/extract_match_times.py` | Sweeps `localDateMatch*` start times + per-game UTC end stamps → `data/match_times.csv` (20,405 matches; 13,248 with referee-clocked starts, rest planned). Runs through the `raw/` cache so it's ~instant where raw/ is warm (droplet), ~35 min cold. **This is what makes the hour-level join possible.** |
| `model/weather_report.py` | Day- AND hour-level cuts: serve-point rate vs wind, favorites vs wind, favorites vs heat, each outdoor vs indoor. |
| `model/end_effects.py` | The good-side/bad-side machinery. Four designs (A, A2, B, C) + simulated null + the dummy-contrast weather test. |
| `model/favorites_wind.py` | Three regressions testing whether wind compresses the favorite's edge. Data-referenced nulls only. |
| `model/wind_skill.py` | The F1-rain hypothesis: split-half reliability, standout/tail tests, pre-specified player tests. |
| `data/decider_splits.csv` | Point counts per side per half (pre/post the mid-game switch at 6) — 5,061 games (PPA deciders + all MLP games). From `pb_rally`. |
| `data/decider_serve_splits.csv` | Same games at rally grain: serve rallies + serve wins per side per half. |
| `web/insights/wind/index.html` | The published write-up, "The sport shrugged" (unlisted, noindex, ships to `site/insights/wind/`). |

Everything re-runs in under a minute except the match-time sweep.

---

## 2. The six hypotheses and their verdicts

**All null.** Ordered as tested.

### H1 — Wind changes the serve/return balance
Serve-point rate vs match-hour wind, 7,905 outdoor matches:
slope **+0.003 per +10 mph [−0.001, +0.007]**. Indoor equally flat.
Rate is 0.4483 calm → 0.4511 at 14–20 mph.

### H2 — There is a good side / bad side
Ends are never recorded (`server_side` = team, not end), but the switch
schedule leaks the information. Four designs:

| Design | Grain | Result |
|---|---|---|
| A — game1 vs game2 margins | match | Var(swing) flat across weather (38.0–39.6 pts²); **underpowered** — only sees end-adv sd ≳1 pt/game |
| A2 — game1 vs game2 point shares | match (13,926) | z² 1.81–1.90 vs sim null 1.84–1.88. Flat. |
| B — pre/post the mid-game switch at 6 | game (5,061) | z² 1.67 indoor / 1.72 calm / 1.76 breezy / 1.95 windy vs nulls 1.76–1.79 |
| C — same switch, serve-rally win rate | rally (7,743 team-halves) | z² 1.10 / 1.15 / 1.09 / 1.34 vs nulls 1.15–1.20 |

Weather test (dummy contrasts vs outdoor-calm reference, no sim):
- Design B: indoor −0.056 [−0.177, +0.064]; windy 14+ **+0.224 [−0.160, +0.585]**; continuous slope +0.165 z²/10 mph [−0.053, +0.364]
- Design C: indoor −0.052; windy **+0.190 [−0.067, +0.544]**; continuous slope **+0.038 [−0.093, +0.166] ≈ flat**

Verdict: no confirmed end effect. The 14+ mph bin runs hot in every
design but never with dose-response, and rests on 111 games (162
team-halves).

### H3 — Momentum (bonus finding, not the original question)
The observed swing overdispersion (z² ≈ 1.7–1.9 vs binomial 1.0) is
**fully reproduced by a no-momentum simulation** that has only side-out
serve mechanics. Within- and between-games. Serve clustering explains
the entire lumpiness of a pickleball game. Matches the spec shootout's
null momentum challenger, now extended inside games.

### H4 — Wind causes upsets (favorite compression)
**This one initially looked real and was wrong.** Binned obs−pred showed
favorites at −4.0pp calm → −6.0pp at 14–20 mph outdoors (1,151 games).
Killed by two things:
- Continuous interaction `share ~ skill + wind + skill×wind`, 24,819
  outdoor games: **d = +0.002 [−0.060, +0.064]**. Zero. (Also: skill
  slope b = 1.040 — v2 shares are near-perfectly calibrated on the share
  scale; the −4pp overconfidence lives in the race-DP tail transform,
  not the ratings.)
- Rally-level binomial logit (P(server wins rally) ~ adv×wind, 98k
  outdoor rallies): d = −0.017 [−0.098, +0.058].
- **The falsification arm fails**: indoor shows d = −0.080 (game level)
  and −0.060 (rally level) — *more* apparent effect where wind cannot
  reach. So the binned drift is composition/label noise, not wind.

### H5 — A wind-skill dimension (F1 rain)
Per-player wind slopes, 552 players with ≥40 outdoor games. Split-half
reliability **r = +0.061 vs permutation null [−0.065, +0.124]**. Not
established. (Same design certified clutch at 0.15 and durability at
0.13.)

### H6 — A single wind standout (the Verstappen case)
Split-half tests a *dimension* and washes out a lone genius, so tested
separately:
- max |z| across 552 players = 3.27, **permutation p = 0.58**
- players with |z| > 2: 29 observed vs null median 27 [19, 36]
- Pre-specified: **Anna Leigh Waters** −0.020 share/10 mph (726 games,
  p = 0.95, i.e. wind-*negative* edge of her band); **Tyra "Hurricane"
  Black** +0.017 (585 games, p = 0.18, right direction, not significant)
- Pre-named defensive-grinder cohort (Tellez, Tardio, Patriquin, Hewett,
  Staksrud, Frazier, C. Smith, Parenteau, J. Johnson, Irvine): pooled
  **−0.012 /10 mph, p = 0.975** — grinders trend *worse* in wind.
  Caveat: cohort members share games/partners, so the permutation band
  is too narrow; read as "clearly not positive."

---

## 3. Methodological turns (the reusable part)

These were user course-corrections and each one changed a result. Keep
them; they generalize past weather.

1. **Indoor is a CONTROL, not a placebo.** Never assume indoor is
   effect-free — drafts, HVAC, lighting exist. What indoor rules out is
   *outside wind*. It gets its own estimate in every design. (It came
   back clean at 2,383 games in Design B — measured, not assumed.)
2. **Difference of means can't work here; the paired difference can.**
   Nobody records which end is good, so the per-game swing is
   sign-symmetric and its *mean is identically zero* whether or not an
   effect exists. But the paired difference cancels skill exactly (same
   teams both halves) — unlike correlation, which was the original
   framing and is uninterpretable in levels. **The signal is the
   swing's variance, not its mean.**
3. **Weight each game by its own information.** Raw excess-variance
   averaging treats a 6–0 half like a 3–2 half. Standardizing each game
   by its binomial noise (z², null = 1.00) is the correct weighting.
4. **Don't restrict to deciders.** Game scores *are* point counts
   (11–6 = shares of 17), so the between-games version needs no rally
   logs → 13,926 matches instead of 3,076. And **every MLP game switches
   at 6** (user rule), and MLP matches are single games, so all 1,985
   logged MLP games qualify for the mid-game design (DBs excluded).
5. **The sim null answers a different question than the weather test.**
   The simulation answers "is there anything beyond serve mechanics?"
   (the momentum question) — needed because "is 1.72 big?" has no
   within-data referent. The *weather* question is a dummy regression
   with a reference group; it needs no simulation and cancels all shared
   contamination. Conflating the two was a real presentation error.
6. **Prefer data-referenced nulls.** Where a within-data contrast exists,
   use it. Reserve simulation for the narrow case where none does, and
   say so explicitly.
7. **Rate vs 0/1 series**: with covariates constant within a match-side,
   the Bernoulli rally series collapses losslessly to (wins, attempts) —
   binomial logit *is* the rally-level model. But it fixed a real flaw:
   the per-game rate version weighted a 20-rally game like a 50-rally
   one.
8. **Split-half detects a dimension, not a standout.** Retroactively
   applies to the clutch finding too: r ≈ 0.15 established a broad faint
   trait, and never by itself ruled standout individuals in or out.
   Tail/max-|z| tests with a panel permutation null are the standout
   instrument.
9. **In-sample parameter leakage was checked, not hand-waved.** k = 0.43
   and the etas come from data including the tested games. Sensitivity:
   null z² moves ~0.027 per 0.01 of k; k's se over 216k rallies is
   ~0.001; and the leaked statistic (a *mean*) is orthogonal to the
   tested statistic (a *variance*) — end effects and momentum don't move
   mean serve rate. Bounded leakage error ≈ 0.003 z², ~50× smaller than
   the contrast being adjudicated. The real limitation is
   misspecification (constant k, odds-split), which moves all bins
   together and can't fake a bin-specific excess.

---

## 4. Open threads / what would change the answer

- **The 14+ mph bin.** Hot in every design, never with dose-response,
  111 games. Accumulates on its own. Re-run `end_effects.py` after a
  stormy stretch. This is the only live crack.
- **Venue labels are the weakest link.** `data/venue_overrides.csv`
  (`event_id,setting`) is the curation hook and is currently empty of
  corrections — labels are tour-default + venue keywords. Several Life
  Time clubs have both indoor and outdoor courts. Mislabelling blurs the
  indoor/outdoor contrast *toward* the null, and could plausibly explain
  the indoor arm's odd behaviour in the binned favorites cut. **Highest
  value / lowest effort improvement available**: a dozen broadcast
  checks.
- **Split-sample k** (if the windy tail ever makes a call close): fit k
  and the odds-split on PPA games 1–2 only (no mid-game switch), test on
  deciders + MLP. Then parameters and test set share no games. Overkill
  today.
- **Heat** was never chased down. Hour-level shows a mild monotone
  gradient outdoors (−3.9 → −4.9pp obs−pred from <70°F to 92°F+) with
  the indoor arm clean-ish. It got dropped when the favorites-wind
  interaction died; the same continuous-interaction treatment would
  settle it. **The most likely place a real weather effect still hides.**
- **Two forward predictions on record** (in the insights page endnotes,
  found by rummaging so they need out-of-sample grading): (a) finesse-
  first players underperform in high wind relative to power players;
  (b) the 14+ mph bins stay slightly hot as they fill. Grade against
  games played after 2026-07-28 only.

---

## 5. Publication status

- **`web/insights/wind/index.html`** — "The sport shrugged: six ways to
  look for a wind effect in pro pickleball. Six nulls." Written in the
  PICKLES design language ported from `insights/unsolved-meta/`.
  Unlisted + noindex per house convention. Builds into
  `site/insights/wind/` via `build_site.py`.
- Every figure in the page was cross-checked mechanically against the
  committed `model/*.md` reports.
- Deliberately **not** published: the wind leaderboards (every name sits
  at minimum sample size — noise-ranked).
- Editorial notes if revisiting the piece: the ten-name grinder cohort is
  an outsider's labelling and is the thing most likely to draw a quibble
  from someone who watches more pickleball than the author of the
  analysis. No social/export images exist yet (unsolved-meta has
  `exports/*.png`; this piece has none).

---

## 6. Bottom line

No serve effect. No bad side. No momentum. No upset multiplier. No wind
dimension. No wind standout. ~37,000 games, ~193,000 rallies, six
hypotheses, a dozen designs.

What can't be claimed is that the effects are *zero* — only that they're
small. Resolution floors: per-player tests can't see an individual edge
under ~3pp share per 10 mph; Design A can't see an end advantage under
~1 pt/game. A real, tiny bad side could be hiding under all of it.

The likely reason: **the rules already handle it.** Ends switch between
games and again mid-game specifically so nobody eats a bad side for
long, and everyone on tour was filtered through years of outdoor play
before turning pro.

---

## 7. Postscript (2026-08-03): the rare-trait re-test

After the clutch thread (finding 10) showed that split-half r and
population-variance tests DILUTE a trait carried by a minority — and that
clutch itself only appeared once minority-aimed instruments ran — the
obvious objection to §5's wind-skill null was "wrong instrument." So the
identical battery from `clutch_rare.py` was pointed at the identical
wind panel (`model/wind_rare.py`; same 552 players / 24.8k outdoor games
as wind_skill.py, so any verdict change is the instrument, not the data):

- **Spike-and-slab LR** against pi=0: observed 5.8 vs permutation null
  1.6±1.8 (max 7.2) — inside the null envelope. (Clutch on this test:
  72.2 vs 1.0.)
- **Tail counts** both directions: wind-strong 3 at z>2.5 vs null median
  3; wind-fragile 5 vs 3 (p=0.27). Nothing.
- **Select-then-verify** (top-K on 2024-25, measured on 2026, both
  directions): every K, both tails, p ≥ 0.17; the wind-strong selection
  actually anti-persists. (Clutch: K=40 z=3.77 vs null 0.29.)
- **Injection power** (`model/wind_rare_power.py`): planting the trait in
  13% of players (clutch's fitted fraction, sign persistent across eras)
  fires the battery 75% of the time at ±0.02 share per 10 mph and 100%
  at ±0.03; false-positive rate 0/20 at s=0. Observed data fired nothing.

Verdict: wind skill fails the SAME instruments that established clutch,
with a measured floor. **No minority wind trait at ≥0.02 share per
10 mph (~½ point per game in a 10 mph breeze) exists in this archive;
anything smaller is below the telescope.** The §5 null stands, now for
the rare-trait shape too.
