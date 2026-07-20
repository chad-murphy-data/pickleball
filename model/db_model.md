> **Audit note (2026-07): the numbers below are PROVISIONAL and partly
> pseudoreplicated.** The rally-level CIs (k=0.55 [0.15,0.95] doubles; the
> k=0.42 singles figure once wired into make_forecast) treat correlated
> within-DreamBreaker rallies as independent, so they are far too narrow. On
> the honest unit of analysis — one DreamBreaker, n≈101 — the effect is only
> marginal (stronger-singles roster wins ~61/101, exact p≈0.05, and that is
> the better of two proxies). `model/db_model.py` now fits this correctly
> (per-DB logistic + nonparametric bootstrap over DreamBreakers, with a
> synthetic self-test); it needs the roster1/roster2 columns that
> `scraper/parse.py` now emits into dreambreakers.csv. Re-run parse.py against
> the raw cache, then `python model/db_model.py`, and if the per-DB CI
> includes zero, drop the "not 50/50" claim. Do not publish these first-pass
> numbers as established.

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

## v2 (same day): singles values replace the doubles proxy

With the full singles corpus harvested (26,048 PPA pro singles games,
2024–26) and model/fit_singles.py fitted:

- **Singles-gap model: k = 0.42, 95% CI [0.20, 0.65]** — beats the
  doubles-gap proxy by 3.1 nll units on the same 101 DBs. Picking the
  stronger-singles roster calls 60.4% of DBs (doubles proxy: 57.4%).
- Players without a singles history (many MLP-only rosters) are imputed
  via the fitted cross-skill regression singles ≈ 0.28 + 1.14·doubles
  (r = 0.74, n = 543). Selection caveat: never-plays-singles players are
  plausibly below their imputation.
- Cross-check that fell out for free: the biggest singles OVERperformers
  vs their doubles-implied value (Haworth, Crum, Z. Ford, G. Joseph,
  Bouchard) are exactly the players whose singles skill outruns their
  doubles rating — that gap was singles ability, not noise.
- Worked example (the Gold final rosters): NJ's DreamBreaker roster
  (Waters +2.27 singles, Khlif +1.48, Howells +1.32, Johnson +1.20)
  prices at 57.2% over STL-with-Fahey — milder than the 65% the pre-match
  memo hypothesized, because Fahey (+1.80, the #2 women's singles player)
  drags the gap back. Swap her for a Jade-Kawamoto-caliber woman
  (imputed +1.60) and NJ's DB edge grows to 60.0%.

make_forecast.py now uses the singles-based model (K_DB_SINGLES = 0.42).

## v3 (same day): roster economics — pricing the DreamBreaker specialist

Question (user): Kate Fahey is a good-not-elite doubles player and an
elite singles player. How many doubles games does she cost you vs how
many DreamBreakers does she win you — especially against Waters teams?

Confirmed premise: Fahey is the **#2 women's singles player** (+1.80, 340
games; Waters +2.27). Her singles residual vs doubles-implied is +0.29 —
the quantified definition of a singles specialist.

The exchange rate (STL vs NJ context, the graded Gold final matchup;
baseline 60.8% ≈ the committed 60.7%):

- Fahey −0.10 doubles logit → −5.6 pp matchup (she plays 2 of 4 games).
- +5 pp of DB win prob → +2.2 pp matchup (DB reached 44.5%, and one
  player is only 1/4 of DB rallies). Generic DB skill trades ~4:1
  against you: **doubles is the main course.**

The Fahey-vs-Jade-Kawamoto swap, real numbers (per 100 matchups vs NJ):
doubles channel −3.4 (Fahey 1.081 vs Jade 1.156, −0.075 logit); DB
channel +1.2 (DB 42.8% with Fahey vs 40.0% with Jade-imputed-singles
+1.60). **Net −2.2: Jade wins the swap** — but the verdict hinges on
Jade's unobservable singles (zero pro singles games; imputation is
selection-biased upward). If her true singles is ~+1.0, the swap goes to
near break-even.

The Waters asymmetry survives in milder form: NJ's DB roster prices at
57.2% over STL-with-Fahey (not the 65% the pre-match memo feared —
Fahey personally drags it back). Against DB-strong opponents the
specialist's value is defensive and roughly doubles; against average
opponents it shrinks. Slots directly into the Phase 4 lineup optimizer.
