# CLAUDE.md — orientation for future sessions

Pro pickleball analytics: scrape every MLP + PPA pro doubles game, fit
Bayesian rating models, publish predictions with receipts. Hobby project,
honesty-first. Read EXPLAINER.md for the plain-language story, analysis.md
for full technical results, design_handoff.md for the content/social plan.

**If active live-listener/receipts work is in flight, HANDOFF.md is the dated
status snapshot + next-thread to-do (check its date; stale ones defer to here).**

## Working rules (user-set)

- **Given an easy way and a hard way, pick the best way.** Difficulty is
  never a blocker and never a virtue — choose on merit, then do it.
  (Example of the standard: when asked "is the model underconfident?",
  the answer was to fit the recalibration out-of-sample on the frozen
  _train values, not to eyeball the old v1 curve.)
- **No probability is ever displayed as 0% or 100%.** Empirical basis:
  ~1% of ≥99% favorites lose (44/4,248 across all games). The calibration
  layer (web/calibration.json, refit via web/fit_calibration.py) encodes
  this as a mixture floor eps ≈ 0.021.

## Pipeline (run in this order; everything is idempotent & cached)

```bash
python scraper/harvest.py                    # sweep season → raw/ (cached, ~1 req/s)
python scraper/enrich_formats.py             # resolve ambiguous score formats
python scraper/parse.py                      # raw/ → data/games.csv etc.
python scraper/build_model_data.py           # games → model tables (env-configurable)
python model/fit_v2.py                       # THE model (dynamic + race likelihood)
python model/report.py                       # regenerate analysis.md (v1 sections)
python scraper/parse_singles.py              # raw/ → data/singles_games.csv (26k games)
python model/fit_singles.py                  # singles MAP ratings (pure python, ~10 s)
python scraper/extract_ratings.py            # raw/ → per-match + latest DUPR (merges)
python scraper/live_poller.py                # live score JSONL during event days
python web/make_forecast.py [--commit]       # price scheduled MLP matchups (network);
                                             #   --commit freezes into receipts.json
python web/build_site.py                     # data/*.csv → site/ static website (~4 s)
```

Deploy: .github/workflows/site.yml → GitHub Pages on push to main +
nightly data refresh (raw/ cached in Actions; guard restores committed
CSVs if a partial parse shrinks games.csv). Setup once: Settings → Pages
→ Source "GitHub Actions".

`harvest.py` accepts `--start/--end`; re-runs only fetch new/recent dates
(last 3 days are "volatile" and refetched). Data source: pickleball.com's
UNAUTHENTICATED same-origin BFF (`/api/v1|v2/results/*`) — discovered by
grepping the JS bundle for `fetch("` (see recon.md). No token, no browser.

## Models

- **v2 = the real model** (`model/fit_v2.py`): joint 2024–2026, per-point
  Binomial race likelihood (handles to-11 AND to-15 games), monthly
  random-walk skill for ≥60-game players, weakest-link gamma, dyad chemistry,
  ramp-up coefficient. Values are on a PER-POINT LOGIT scale (~0.38 sd).
  Outputs: `data/v2_players.csv` (current form), `data/v2_trajectories.csv`
  (monthly curves), `data/v2_dyads.csv`, `model/v2_draws.npz` (posterior,
  gitignored — refit with SRM2_SAVE_DRAWS=1 to regenerate).
- **v1 = Gaussian margin model** (`model/fit_srm.py`): per-season, values in
  POINTS PER GAME (Waters ≈ +7.7, median regular ≈ +2). Still used for the
  human-readable scale and analysis.md tables. Many env knobs
  (SRM_SUFFIX, SRM_SD_D_PRIOR, SRM_PLAYER_TOUR, SRM_NEWNESS, SRM_MINMAX,
  SRM_MIXTEST, SRM_SAVE_DRAWS). Suffix convention: `_2026`, `_2026core`,
  `_train`, `_2026mm`, etc. map to data/model_*{suffix}.csv inputs.
- Validation: v2 = 77.4% winner accuracy / 0.165 Brier on 884 post-June-1
  holdout games (v1 75.2%/0.178; DUPR 64.7%/0.229). Gate any model change
  on beating this (`model/v2_holdout.py`; needs a `_train`-suffixed fit
  with SRM2_DATE_BEFORE=2026-06-01).

## Established findings (don't re-derive; analysis.md has details)

1. Weakest-link: team = sum + γ|gap|, γ = −0.47 pts (Gaussian) / −0.18
   logit (race model — the truncation-free estimate). Gender-blind in mixed.
2. Chemistry is small: sd ≈ 0.3–0.5 pts, prior-insensitive; no single pair
   certifiable (need ~1,000 games; max on record 138).
3. Skill transfers across contexts (sd_w ≈ 0.13) and tours (no MLP
   sandbagging; slope test + player-tour effects both null).
4. Johns never declined in absolute terms (dynamic model); the field rose.
   Tardio's rise is smooth and real.
5. New pairings OVERperform first ~6 games (beta_new > 0) — window-edge
   caveat only partially resolved; treat gently.
6. DreamBreakers are NOT 50/50: mean roster SINGLES value predicts them
   (k = 0.42, CI [0.20, 0.65], n = 101; beats the doubles proxy by 3.1
   nll; stronger-singles roster wins 60%; model/db_model.md). Singles
   ratings: 26k PPA singles games, fit_singles.py; singles~doubles
   r = 0.74; imputation for never-plays-singles rosters ≈ 0.28+1.14·d.
   Waters +2.27 / Fahey +1.80 are the top two women's singles values.
   Wired into make_forecast (K_DB_SINGLES).
7. Cross-gender offset: the γ|gap| term is the ONLY identification channel
   and it's stable in-form (c* ≈ +0.08 logit, scales ~1:1) but the nominal
   precision is fake (values held fixed; form-borne). House rule stands —
   never publish as fact. A single 2W-vs-2M exhibition game carries ~se
   0.24 logit of DIRECT offset info; a weekend of them beats 14k mixed
   games. (Session analysis 2026-07-13; not on the site.)

## House rules (hard-won; violating these produces silently wrong results)

- **UUIDs are identity, never names.** API mixes upper/lowercase UUIDs —
  always lowercase. Three Kawamotos exist; two are twins.
- **Cross-gender comparisons are likelihood-flat**: every game has equal
  women per side, so the M/W offset is a prior convention. Never publish
  a cross-gender ranking as fact. (The γ nonlinearity technically breaks
  the flatness — do not rely on it.)
- **Actor vs partner effects are NOT identifiable** from team margins —
  only total value + dyad deviations + the weakest-link structure.
- A player's dyad effects see-saw around their own average (mechanically
  anticorrelated) — a "bad chemistry" pairing implies mirrored "good" ones.
  Only within-player contrasts are identified, and for Bright/Patriquin
  they're collinear with tour (bad partner = PPA-only, good = MLP-only).
- **Score formats are data, not assumptions**: PPA Challengers hide
  single-game-to-15 rounds (side-out, NOT rally); MLP DreamBreakers are
  rally-to-21 singles and NEVER enter models (isTieBreaker flag). 2026 MLP
  skips the dead 4th game at 3-0. matchCompletedType 5 = normal; 6 =
  walkover/cancelled; treat others as forfeits.
- MLP is identified by organizationSlug == "major-league-pickleball"
  (titles lie: Grand Rapids = "Edward Jones Mid-Season Tournament";
  exclude "Junior" titles). PPA filter: ppatour.com contact emails or
  \bPPA\b in title, minus Australia/Asia/College.
- 403 endpoints (getMatchInfos, getTeamLeaguesMatchupsOnDivision) are
  permanent — the "Short" variants + detail endpoints cover everything.
- **Browsers cannot reach the network from this environment** (egress
  gateway TLS-fingerprints and resets Chromium; curl/httpx fine). Don't
  waste time on Playwright; recon.md documents the diagnosis.
- The embedded per-match "rating" IS the player's synced DUPR doubles
  rating (verified: singles is a separate ledger; scale 2–8; compresses
  hard at the top and has data glitches — see analysis.md benchmark).
  Known artifacts, re-verified 2026-07-13 against a fresh refetch:
  Jackie Kawamoto = 3.50021 since 2026-06-04 (was 6.13 in Feb; 3.5 is
  DUPR's reset default — the platform still serves it, treat as glitch,
  site nulls it via data.finalize_dupr); tour-wide recalibration dropped
  everyone ~0.3–0.7 on 2026-05-22 (Truong 5.83→5.137 and Jade Kawamoto
  6.1→5.819 are CORRECT post-recal values, confirmed in fresh records).
- Be polite: ~1 req/s harvest, ≥15 s live-poll interval.

## Live win probability (in progress)

- `scraper/live_poller.py` = Tier 1 (poll BFF during events → live/*.jsonl).
  Needs a persistent machine, not an ephemeral session. `deploy/` has the
  VPS kit: user-level systemd timer (daily 09:15 PT, no-op on quiet days),
  wrapper pins TZ and pushes the day's JSONL.
- Tier 2 protocol DECODED (2026-07-15, recon.md): SSE at
  rte.pbgql.co/live-scoring, PB-RTE-TOKEN = base64(JSON{ua,origin,fingerprint})
  (non-secret; server takes any well-formed token), subscribe via base64
  X-Request-Matches/-Matchups headers. `scraper/sse_probe.py` captures it
  (auto-discovers today's live UUIDs; --with-logs = per-rally feed). Handshake
  verified from here; real event SHAPES still need a live-match capture, then
  fold the parser into live_poller as Tier 2.
- Win-prob math: per-point p from v2 → exact race-to-T DP. Serve-aware
  version: point-share pins the skill gap exactly (p = σ(2d)); a league
  serve-rally baseline k (~0.35–0.45, currently assumed) sets streakiness;
  4 serve states per score; the no-score side-out cycle must be solved
  algebraically (see session history — naive recursion loops forever).

## Scheduled obligations

- **September 2026**: score `model/registered_predictions.md` (frozen
  2026-07-12) against games dated AFTER 2026-07-12 only, using the method
  written in that file; update the pending entry in `model/receipts.json`.
- Season end (~Sept): full re-harvest + v2 refit + refresh analysis.md,
  trajectories, leaderboards; rebuild the site.
- ~~Grade the Gold final~~ DONE — graded independently twice with identical
  calls (2026-07-13 from the API matchup record → `model/receipts.json`,
  which the site renders; 2026-07-16 narrative → `model/receipts.md`):
  STL won 3-0 (WD Bright/Fahey 11-6, MD Tardio/Patriquin 11-3, MXD1
  Bright/Patriquin 11-8; MXD2 skipped, no DB — Waters did lose twice).
  Overall 61% STL HIT, headline WD 88% MISS. Graded tables in
  model/prediction_midseason_final.md.

## Website (Phase 2 MVP — BUILT; see ROADMAP Phase 2)

`python web/build_site.py` regenerates `site/` (gitignored, ~511 pages) from
data/*.csv + model/receipts.json in ~4 s, stdlib-only (no pandas). Pages:
PICKLES landing page (index.html — live doorway teasers + conditional
"tonight" band off data/forecasts.json), power rankings (rankings.html),
499 player pages (trajectory + game-log-vs-expectation SVGs), client-side
matchup simulator (race DP + weakest link + uncertainty in embedded JS,
shareable permalinks), receipts ledger + calibration, record book,
DUPR×model, methods, 404. The look is the PICKLES design handoff: master
stylesheet = CSS string in web/sitelib/style.py (design port verbatim +
landing additions; light AND dark). Conventions: values are displayed as
"expected margin vs an average pairing" via web/sitelib/race.py:value_points;
the race DP there mirrors model/v2_holdout.py AND the JS inside
build_simulator — keep all three in sync. Rankings rank 2026-active players
only; men/women always separate. `model/receipts.json` is the receipts
source of truth — commit predictions there BEFORE matches, grade after.

## Open threads (specced, unbuilt)

Website extras: live win-prob charts (needs Tier 1/2 listener on a VPS);
social prediction-card renders (design bundle `Prediction Cards.dc.html`,
port later). Deploy is .github/workflows/site.yml (build + Pages deploy on
push to main, nightly data refresh); one-time setup = repo Settings →
Pages → Source "GitHub Actions". Scorebug OCR of YouTube broadcasts could
backfill point-by-point history (Tier 0 of the vision pipeline;
championship-court sample bias noted).
