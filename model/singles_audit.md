# Singles ratings — audit of existing assets (2026-08-16)

Trigger: MLP said on broadcast they want singles stats. Goal product: a
full public suite of singles ratings — fitted values for everyone with a
real singles record, a doubles-informed Bayesian imputation for everyone
without one, honest uncertainty and labeling throughout.

Verdict up front: **~80% of this already exists and is validated; none of
it is rendered publicly.** The main build is (1) a prior upgrade to
`fit_singles.py` that is exactly the "Bayesian prior on whatever singles
points we have" idea, (2) uncertainty output, (3) surfacing. Two hygiene
defects found along the way, one of them a live-page pricing inconsistency.

## 1. What exists (inventory)

**Data — fresh.** `data/singles_games.csv`: 27,294 PPA pro singles games
(26,881 non-forfeit), 2024-01-09 → 2026-08-15, men 18,644 / women 8,650,
~7.6k games in 2026. Harvested + parsed **nightly** (`parse_singles.py` is
in site.yml). Rally-level singles serve/return already lives in the
Supabase warehouse (PPA singles serve-rally win rate 0.538 vs 0.43–0.44
doubles, n≈3.5k — HANDOFF 2026-08-09).

**Model — built, stale.** `model/fit_singles.py`: per-point Binomial race
MAP (same likelihood family as v2), prior v ~ N(0, 0.6), exponential
recency half-life 365 d. Output `data/singles_players.csv`: 2,229 players
(everyone who ever appears), point estimates only. Games-count mix:
836 players ≥10 g (228 of them ≥60 g), 445 at 5–9, 948 at 1–4.
**Last fitted 2026-07-31 on 25,678 games — 1,203 games behind the data**,
because `fit_singles.py` is *not* in the nightly workflow.

**Imputation — built AND bias-corrected.** For <10 singles games,
`web/make_forecast.py` maps doubles → singles via
`SINGLES_IMPUTE = (−0.07, 1.14)`:
- regression on ranked players: singles ≈ 0.28 + 1.14·d, r = 0.74
  (n = 543 then; refit today on n = 792: **0.22 + 1.16·d, r = 0.74,
  residual sd τ = 0.35** — τ is the prior width the Bayesian design needs);
- selection correction **−0.35**: never-plays-singles players underperform
  their doubles-implied value in real DreamBreaker rallies
  (`model/db_impute.md`: rally-level logistic on 3,189 validated DB
  rallies, shrink 0.36, 95% CI [0.02, 0.74], P(≤0) = 1.9%; threshold
  sensitivities 0.42/0.43 agree).

**DreamBreaker assets (the MLP-native singles data).**
`data/dreambreakers.csv` 133 DBs; `data/db_rallies.csv` 3,189
player-attributed rallies from the validated referee-log parser (88/94
exact-score validation); `data/db_orders.csv` 176 announced rotations
(slot 1 is a man 176/176). Top personal DB rally counts: McGuffin 148,
Freeman 145, Hunter Johnson 139, Khlif 125, Ignatowich 123, JW Johnson
119. DB model: p(rally) = σ(0.42·mean-roster-singles-value gap), beats
the doubles proxy by 3.1 nll on 101 DBs (`db_model.md`); measured
player-level rally slope on singles values ≈ 0.51 ± 0.08 (`db_impute.md`).

**Second axis.** `data/singles_surplus.csv` (199 players): singles-minus-
doubles-predicted surplus, split-half r = 0.56, cross-era 0.36 — a
validated "how much of their game survives without a partner" stat for
player pages (finding 12). Not a doubles-model feature; display-only.

## 2. What is public today

- **No rendered singles ratings anywhere.** `web/build_site.py` contains
  zero singles content (no rankings page, no player-page line); git
  history confirms there never was one — the memory of it being public is
  the *prose* rankings in the unsolved-meta insights article ("Haworth #1
  / Alshon #4 in the world; Waters #1 by a margin nobody else enjoys"),
  frozen at a June-26 fit.
- Raw values ≥10 g DO ship in `site/data/live_values.json` (unrendered).
- DB win probabilities on forecast/live pages are singles-model output
  ("rough by design" note on the live page).

## 3. Coverage of the 2026 MLP pool (the product population)

151 players appeared in 2026 MLP doubles games:

| tier | total | men | women | treatment today |
|---|---|---|---|---|
| ranked, ≥10 singles g (median 118 g) | **119** | 65 | 54 | real fitted value |
| thin, 1–9 g | 14 | 5 | 9 | **singles data discarded**; imputed from doubles |
| zero singles | 18 | 8 | 10 | imputed from doubles |

Every zero/thin player has a v2 doubles value — nobody is unpriceable.
Faces of the imputed tier: Jade & Jackie Kawamoto, Pisnik, Tyra Black,
Daescu, Riley Newman, Tuionetoa, Bar, Klinger; thin tier: Rohrabacher
(7 g), Dizon (7 g), Sewing (2 g), Acevedo (2 g).

## 4. Defects found

1. **Live-page imputation drift (real bug).** `web/sitelib/livepage.py`
   ships `SINGLES_IMPUTE = (0.28, 1.14)` — the pre-correction intercept —
   while `make_forecast.py` uses (−0.07, 1.14). Client-side DB repricing
   on the live page therefore disagrees with the graded pre-match number
   for any roster with imputed players, violating the "live curves agree
   with receipts at rally zero" invariant. One-line fix (or better: move
   the constant to one shared place).
2. **fit_singles.py not in the nightly** → ratings go stale silently
   (currently 16 days / 1,203 games). It's a 10 s pure-python step; add it
   to site.yml right after parse_singles.py and commit
   singles_players.csv with the nightly data.
3. **db_rallies/db_orders stale**: parsed when 94 DBs were logged; 133
   DBs now exist. Re-running `db_impute.py` also tightens the −0.35
   correction (its CI [0.02, 0.74] is wide).
4. No uncertainty anywhere in singles outputs (MAP point estimates), and
   the hard 10-game threshold throws away real singles evidence for the
   1–9 g tier — both fixed by the model upgrade below.

## 5. Build plan (recommended)

**Phase 0 — hygiene** (do regardless): fix the livepage intercept; add
fit_singles to nightly; re-run db_impute on the 133-DB corpus.

**Phase 1 — the Bayesian suite** (the user's design, confirmed right):
replace the zero-centered prior in fit_singles with a doubles-informed
per-player prior
`v_p ~ N(a + b·d_p − c, τ² + b²·sd_d(p)²)`
with (a, b, τ) refit from ranked players (today: 0.22 + 1.16·d, τ = 0.35),
c = the DB-measured selection correction (~0.35) for players without a
real singles record, and sd_d from v2_players.value_now_sd. Add a
diagonal-Laplace posterior sd (the fit's preconditioner is already the
Fisher diagonal — nearly free). Output columns: value, **sd**, games,
evidence tier (`fitted` ≥10 g / `blended` 1–9 g / `imputed` 0 g). Ranked
players barely move (likelihood swamps the prior); thin players finally
use their actual points; zero-singles players get prior mean ± ~0.36 with
an honest label. Keep the current zero-centered variant behind an env
flag — the surplus axis (finding 12) and the singles~doubles r diagnostic
must keep using a doubles-blind fit to avoid circularity.

**Phase 2 — optional, MLP-native evidence**: add the validated DB rallies
as Bernoulli likelihood terms with the measured attenuation slope (~0.51),
so MLP players' actual DreamBreaker play informs their singles value
(McGuffin has 148 attributed rallies). Gate on out-of-sample DB nll +
a singles holdout before shipping.

**Validation gate to define** (none exists for singles today, unlike v2's
holdout): winner accuracy/Brier on post-2026-06-01 singles games (1,668
available) + DB-outcome nll on the 133 DBs; new prior must not lose to
the current fit.

**Phase 3 — surfacing**: gender-separate singles rankings page (house
rule: men/women never meet in PPA singles, the cross-scale offset is a
prior convention — never a combined list); player-page singles block
(value ± sd, evidence badge, DB rally record from db_rallies, surplus
axis); display scale via the existing race DP ("expected margin vs an
average [tour] singles player, game to 11" — singles twin of
race.py:value_points). Values floored/labeled per house rules; imputed
tier always visibly badged.

## 6. Decisions needed (user)

1. Scope: ratings suite only, or also a **DreamBreaker record book**
   (per-player DB rally W-L, rotation-order history — pure accounting,
   data already parsed, very on-theme for the TV ask)?
2. Surfacing: new rankings-singles page + player-page block, or player
   pages only first?
3. Phase 2 (DB rallies as evidence) — build now behind the gate, or park?

Notes that bind any public version: cross-gender singles comparisons stay
unpublishable as fact (DB cross-gender rallies are 2.7% of the corpus and
identify nothing); UUIDs are identity; no 0%/100% anywhere.
