# The singles suite (2026-08-16)

MLP said on broadcast they want singles stats; the user's design brief:
fitted ratings for everyone with a real singles record, a Bayesian
doubles-informed prior for everyone else, DreamBreaker points counting as
singles evidence, a DB record book, and public surfacing (rankings page +
existing player pages). All shipped in this change. The pre-build audit is
`model/singles_audit.md`; this file is the record of what was built, what
the gate said, and the two hard-won lessons.

## What ships

- **`model/fit_singles.py`** (rewritten): three stages, pure python, ~30 s.
  1. *Pure fit* — unchanged first-pass model (per-point Binomial race MAP,
     N(0, 0.6) prior, 12-month recency half-life). Ships in the
     `singles_value_pure` column; stays doubles-blind for the finding-12
     surplus work and the singles~doubles diagnostic.
  2. *Hyperpriors* — per-gender regression of pure values on v2 doubles
     values over the 60+-game population (M: 0.49 + 1.03·d, τ 0.28, r 0.75,
     n 159; F: 0.30 + 1.06·d, τ 0.32, r 0.72, n 76). Per gender because the
     two models' cross-gender prior conventions differ ~0.2 logit; 60+ only
     because the selection penalty persists into the 10–59 range.
  3. *Suite fit* — per-player prior N(a + b·d − c·fade(games), τ² +
     (b·sd_d)² + (0.17·fade)²) with c = 0.35 (db_impute, refreshed
     2026-08-16: 0.348, CI [0.05, 0.71] on 114 validated DBs) and fade
     **calibrated per evidence bin** (below); DreamBreaker rallies enter
     the likelihood as Bernoulli(σ(0.49·gap)) — same-gender only, K = the
     measured rally slope (0.488 ± 0.067). Posterior sd from the diagonal
     Laplace at the MAP.
- **`data/singles_players.csv`** (new schema): every player with singles
  evidence AND every v2 player — 4,307 rows, columns
  `player_id, full_name, gender, singles_games, db_rallies, db_rally_wins,
  singles_value, singles_sd, tier, singles_value_pure`. Tiers: `fitted`
  (≥10 games, 894), `blended` (1–9 games and/or DB rallies, 1,480),
  `imputed` (prior only, 1,933).
- **`model/singles_model.json`**: all fitted constants, incl. the
  per-gender zero-evidence closed form (`impute`: value = a − c + b·d) that
  consumers use as a fallback for unknown UUIDs.
- **Consumers rewired**: `make_forecast.db_win_prob` and the live page just
  look up suite values for everyone (no thresholds, no local imputation
  formulas). This kills the drift bug where livepage shipped the
  pre-correction intercept (0.28) while make_forecast used −0.07 — live DB
  repricing and graded receipts now share one number by construction.
- **Site**: `singles.html` (gender-separate rated tables with ◐ blend
  badges + uncertainty intervals, projected-only table for the 2026 MLP
  pool, DreamBreaker record book ≥30 rallies) and a Singles block on every
  player page (value ± 90%, evidence basis, singles W–L, DB rally/team
  record). Display scale = expected margin vs the average 2026-active
  fitted singles player of the same gender, race to 11 (the singles twin
  of `value_points`).
- **Nightly**: `db_impute.py --parse-only` (new flag) + `fit_singles.py`
  added to site.yml between parse and forecast; `model/singles_model.json`
  added to the nightly data commit.

## Holdout gate (pre-June train, post-June eval; `model/singles_holdout.py`)

Doubles prior from the frozen `v2_players_train.csv`, so no post-cutoff
information enters. "Status quo" = pure fit + the production imputation
rule (what shipped before this change). Suite scored two ways: point
values, and integrated over posterior sd (`suite+unc` — how the site's
race convention actually prices).

| arm | warm games (n=1,067) acc / Brier / nll | cold (n=601) Brier / nll |
|---|---|---|
| status quo | .718 / .1926 / .5881 | .2463 / .7571 |
| pure       | .718 / .1944 / .6009 | .2597 / .8317 |
| suite (point) | .722 / .1913 / .5898 | .2422 / .8000 |
| **suite+unc** | **.722 / .1880 / .5653** | **.2250 / .6625** |

DreamBreakers (n=50 team outcomes, 1,829 rallies): status quo nll 0.6718,
suite 0.6719 — **a tie, not the win the audit's gate asked for**. Read
honestly: most DB rosters are all-ranked players where the arms barely
differ; the sub-ranked-roster subset (n=26) is also a wash. OOS rally
calibration (db_impute-style logistic on post-cutoff rallies, train
values): sub-ranked indicator b2 = +0.14 ± 0.14 for the suite (z = 1.0,
no detectable miscalibration; power limits the CI to roughly ±0.4 of
value-scale shrink). Ship decision: wins singles games decisively, ties
DBs, calibration clean, and it brings what status quo cannot — honest
uncertainty, unified tiers, thin-record evidence, DB rallies as evidence.
Re-run the gate at season end when the DB sample has grown.

## Lesson 1: the location mode (do not re-learn this)

The singles likelihood only constrains within-gender gaps; the overall
location of each gender's connected component is set by the prior system.
Give thin-tier players optimistic doubles-implied priors and the whole
component inflates away from the imputation line (+0.15 here): fitted
players' wins over now-better-rated thin opponents lift everyone
connected. Three "fixes" that fail, tried and measured:

- *Refit the line to the posteriors and iterate*: the location is a
  nearly-free mode (contraction ~0.92/round) — "converges" to an
  arbitrary inflated scale (+1.2 M / +1.5 F here; Waters +3.13).
- *Rigid gauge translation back*: exact only at the fixed point, and even
  then it crushes every player whose prior did NOT ride the drift — the
  748 non-v2 qualifiers took up to −1.2 (a 9-game qualifier scored −1.47
  while beating his rating's implied odds tenfold).
- *Anything where sub-threshold priors chase the posterior location*:
  diverges by construction; the prior-vs-posterior mismatch is
  gauge-invariant.

The actual fix is Lesson 2: remove the drift at its source. Residual
drift after calibration is ~+0.2 on the 60+ tier, accepted and published
as-is: every consumer uses within-gender gaps (game pricing; DB rosters
are always 2M+2W so even cross-gender display locations cancel;
rankings; margin-vs-average display), and there was no public singles
scale to stay continuous with. The published scale therefore sits ~+0.2
above the pure column — do not "fix" one to match the other.

## Lesson 2: the selection penalty fades far slower than assumed

A linear fade of c to zero at 10 games is wrong: players with 1–9 games
sat 0.11–0.17 below their priors, and db_impute's threshold sensitivities
(shrink 0.33/0.35/0.39 at ranked ≥1/≥10/≥30) always said the penalty
outlives the ranked line. The fade is now CALIBRATED: bins (1-4, 5-9,
10-29, 30-59 games) adjust until each bin's mean posterior−prior residual
is ~0 (empirical Bayes at the fade level; f(0)=1 fixed — that is
db_impute's directly measured point, and the 0-games-with-DB-rallies bin
verified it at +0.03). Converged fade: **1.80 / 0.70 / 0.11 / 0.00**.
The raw values are location-entangled (they absorb the +0.2 drift), so
don't read them as pure penalties; the meaningful facts are (i) bin
residuals ≈ 0, (ii) the 1–4-game bin needs a penalty well beyond c — the
players who dabble and stop are an even more negatively selected group
than the ones who never start, which db_impute's ≥1 sensitivity
independently supports; (iii) 30-59 clamps at 0 with a +0.08 residual
(mild out-performance; prior weight there is <15%, bias ≤0.01 — accepted).

## Refresh obligations

Nightly is automated. At season end: re-run `model/db_impute.py` (full)
to re-measure c and the rally slope on the grown DB corpus, update
`C_NONRANKED`/`K_DB` if they move, and re-run `model/singles_holdout.py`
— especially the DB arm, which is the undecided one. If v2's scale or
`value_now_sd` conventions ever change, the hyperprior regression and
every prior inherits it silently — recheck stage-2 prints.
