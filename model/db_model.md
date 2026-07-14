# DreamBreaker model — first pass (2026-07-13)

Question: our matchup forecasts priced every DreamBreaker 50/50 ("singles
is outside the doubles model"). Is that actually right?

Data: all **101 DreamBreakers** ever played (2024–2026, dreambreakers.csv),
rosters recovered from the matchup records (raw/matchup_data, fetched via
the cached client). Team strength = mean of the four roster players'
monthly doubles values (per-point logit).

## Fits

- **Winner-level logistic** (DB win ~ β·gap): β = 2.7, 95% CI [0.2, 5.5].
  Barely excludes zero on 101 binary outcomes.
- **Rally-level binomial** (rallies won ~ Binomial(total, σ(k·gap))):
  **k = 0.55, 95% CI [0.15, 0.95]** — cleanly excludes zero. This is the
  estimate the forecasts use.
- Picking the stronger-doubles-roster team wins **57.4%** of DBs.

## Interpretation

Doubles skill transfers to DreamBreaker rallies at roughly **half
strength** (k ≈ 0.55 vs 1.0 by construction in doubles). That is exactly
the shape you'd expect if DB (rally-scoring, singles-style, rotating every
4 rallies) rewards a correlated-but-different skill. 50/50 was not crazy —
for evenly-matched rosters it still is — but for gapped rosters the DB is
a real, mild edge, worth a few points of matchup probability.

## Wired into forecasts

web/make_forecast.py: p(DB win) = race-to-21 DP at per-rally
σ(0.55 · mean-roster-value gap), clamped by the display-calibration floor
(never 0/100%). The forecast page states the DB winner probability
alongside the P(reaches DB) path.

## Honest limits

- 101 observations; the CI on k is wide (0.15–0.95). Refit as DBs accrue.
- Player-level DB (singles) effects are hopeless at this n (~8 players per
  DB, no player has more than a couple dozen appearances). Rally-resolution
  data (Tier-2 SSE or scorebug OCR) would unlock per-player singles form —
  including Waters, whose singles reputation was the original reason to
  distrust 50/50.
- DB segments are same-gender (W vs W, M vs M), so none of this touches
  the cross-gender identification problem.
