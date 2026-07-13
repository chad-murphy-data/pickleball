# CLAUDE.md — orientation for future sessions

Pro pickleball analytics: scrape every MLP + PPA pro doubles game, fit
Bayesian rating models, publish predictions with receipts. Hobby project,
honesty-first. Read EXPLAINER.md for the plain-language story, analysis.md
for full technical results, design_handoff.md for the content/social plan.

## Pipeline (run in this order; everything is idempotent & cached)

```bash
python scraper/harvest.py                    # sweep season → raw/ (cached, ~1 req/s)
python scraper/enrich_formats.py             # resolve ambiguous score formats
python scraper/parse.py                      # raw/ → data/games.csv etc.
python scraper/build_model_data.py           # games → model tables (env-configurable)
python model/fit_v2.py                       # THE model (dynamic + race likelihood)
python model/report.py                       # regenerate analysis.md (v1 sections)
python scraper/live_poller.py                # live score JSONL during event days
python web/build_site.py                     # data/*.csv → site/ static website (~4 s)
```

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
- Be polite: ~1 req/s harvest, ≥15 s live-poll interval.

## Live win probability (in progress)

- `scraper/live_poller.py` = Tier 1 (poll BFF during events → live/*.jsonl).
  Needs a persistent machine, not an ephemeral session.
- Tier 2 undiscovered: SSE stream at rte.pbgql.co (client subscribes with
  Accept: text/event-stream; match records carry rteMetaUuid). Discovery
  requires dumping the stream DURING a live match.
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
- ~~Grade the Gold final~~ DONE 2026-07-13, verified from the API matchup
  record: STL won 3-0 (WD Bright/Fahey 11-6, MD Tardio/Patriquin 11-3,
  MXD1 Bright/Patriquin 11-8; MXD2 skipped, no DB — Waters did lose twice).
  Overall 61% STL HIT, headline WD 88% MISS. Graded table appended to
  model/prediction_midseason_final.md; ledger entries in model/receipts.json.

## Website (Phase 2 MVP — BUILT; see ROADMAP Phase 2)

`python web/build_site.py` regenerates `site/` (gitignored, ~506 pages) from
data/*.csv + model/receipts.json in ~4 s, stdlib-only (no pandas). Pages:
power rankings, 499 player pages (trajectory + game-log-vs-expectation
SVGs), client-side matchup simulator (race DP + weakest link + uncertainty
in embedded JS, shareable permalinks), receipts ledger + calibration,
record book, DUPR×model, methods. Conventions: values are displayed as
"expected margin vs an average pairing" via web/sitelib/race.py:value_points;
the race DP there mirrors model/v2_holdout.py AND the JS inside
build_simulator — keep all three in sync. Rankings rank 2026-active players
only; men/women always separate. `model/receipts.json` is the receipts
source of truth — commit predictions there BEFORE matches, grade after.

## Open threads (specced, unbuilt)

Website extras: open-CSV downloads page, deploy target + nightly rebuild
(Pages action or VPS cron), live win-prob charts (needs Tier 1/2 listener
on a VPS). Scorebug OCR of YouTube broadcasts could backfill point-by-point
history (Tier 0 of the vision pipeline; championship-court sample bias
noted).
