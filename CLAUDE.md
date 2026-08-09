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
python scraper/live_poller.py                # live score JSONL during event days
python web/make_forecast.py [--commit]       # price scheduled MLP matchups (network);
                                             #   --commit freezes into receipts.json
python scraper/tournament_state.py           # live-event state (MLP standings/slate,
                                             #   PPA seeded draws) → data/tournament_state.json
python web/build_site.py                     # data/*.csv → site/ static website (~4 s)
```

Deploy: .github/workflows/site.yml → GitHub Pages on push to main +
nightly data refresh. The nightly harvests --start 2024-01-01 (raw/ is
cached in Actions; parse.py rebuilds games.csv from raw/ ALONE, so the
sweep must cover the full history or the shrink guard fires), then
commits the refreshed data/ back to the branch so push builds never
regress to stale CSVs. Guard: restore committed data/ if games.csv drops
under 30k rows. Setup once: Settings → Pages → Source "GitHub Actions".

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
  holdout games (v1 75.2%/0.178). Gate any model change
  on beating this (`model/v2_holdout.py`; needs a `_train`-suffixed fit
  with SRM2_DATE_BEFORE=2026-06-01).

## Established findings (don't re-derive; analysis.md has details)

1. Weakest-link: team = sum + γ|gap|, γ = −0.47 pts (Gaussian) / −0.18
   logit (race model — the truncation-free estimate). Gender-blind in mixed.
   Equivalent to weighting better/worse partner 0.41/0.59. PER-DIVISION
   (2026-08-03, joint refit `model/fit_v2_gamma_div.py` →
   `gamma_division_refit.md`): NOT uniform — mens −0.28 (0.36/0.64),
   womens −0.20 (0.40/0.60), mixed −0.09 (0.45/0.55, CI touches 0); only
   mens−mixed is credible (Δ=−0.19, P=0.008). The pooled −0.18 is a
   mixed-heavy average. CAUTION: a conditional profile on frozen pooled-γ
   values (`gamma_division.py`) gave the REVERSED ordering — that
   circularity is real, always use the joint fit for γ questions. Mixed γ
   is entangled with the cross-gender offset channel (finding 8: γ|gap| is
   the offset's only identification). Production v2 keeps pooled γ until a
   per-division variant beats the holdout gate.
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
7. **Specification shootout (2026-07-18; model/spec_shootout.md +
   spec_shootout.py, big_points.py)**: v2 survived ~35 challengers on the
   frozen June+ holdout (n=926). Min-only/max-only/men-only/chem-boost/
   momentum/experience/seed-order: worse or null. Level ladder:
   match < game < margin ≈ points. Rally-level serve/return split TIES
   plain points pre-match (live-only asset); winprob.py's odds-split
   assumption VALIDATED on 216k harvested rallies (slope 1.03). Women
   carry mixed predictively (74% women-only vs 65% men-only — equal
   loadings, women's spread 1.5×; offset-safe). ~~Clutch & durability
   exist but are faint beyond skill (split-half r ≈ 0.15/0.13)~~ — the
   clutch half is SUPERSEDED, see finding 10 (that split-half was mostly
   a side-out artifact; the real effect is smaller AND needs a
   team-outcome estimator). Durability untouched by that work; biggest
   rally = 9-10 on #2 server, 0.47 swing, still stands.
   Lone P≥0.95 challenger: equal-weight v2+margin+Elo ensemble
   (0.1638 vs 0.1665 Brier) — shadow-ledger it before ever adopting.
   Rally logs cache: raw/match_logs/ (gitignored; harvest_logs.py).
8. Cross-gender offset: the γ|gap| term is the ONLY identification channel
   and it's stable in-form (c* ≈ +0.08 logit, scales ~1:1) but the nominal
   precision is fake (values held fixed; form-borne). House rule stands —
   never publish as fact. A single 2W-vs-2M exhibition game carries ~se
   0.24 logit of DIRECT offset info; a weekend of them beats 14k mixed
   games. (Session analysis 2026-07-13; not on the site.)
9. **MLP 2026 awards are FROZEN** (2026-08-01; model/mlp_awards_2026.md,
   reproduce with model/mlp_awards.py — do NOT re-derive definitions or
   re-pick thresholds in a fresh session). Most Improved = 2025→2026 v1
   season delta, ≥20 games/season (Hendershot +2.0 / Shimabukuro +1.6).
   MVP = matchup WPA — pure outcome accounting, MLP-2026-only, matchups
   reconstructed from games.csv (no matchup id there: event/date/stage +
   roster union-find, time-ordered); roster WPA sums to net
   matchups-decided-in-games; men's +5.62 tie broken Johns > Tardio on
   clinchers 13-10 (Waters +7.12 / Johns). Under the Radar = GWAE vs
   month-of-game v2 values, ≥25 games (Sewing +4.6 / Braham +4.0);
   ALL-TOUR expectations are a deliberate design call (MLP-only ratings
   would grade small-sample players against their own graded games —
   considered 2026-08-01 and rejected; the clean upgrade, if ever wanted,
   is frozen pre-May ratings, not MLP-only). PAR was set aside (career
   skill × MLP usage, not MLP-earned); total points & POE set aside too.

10. **CLUTCH IS REAL BUT SMALL — and the naive measure is ~2/3 artifact**
   (2026-08-03; `model/clutch_notes.md` = what was FOUND, read that first;
   `model/clutch_leverage.md` = the technical record. Read both before
   re-opening this). Reusable beyond clutch: a per-rally LEVERAGE scale
   (max 0.457 = down 9-10 receiving vs the #2 server), measured k = 0.4383
   doubles / 0.5254 singles, and the match-vs-game VARIANCE SPLIT for PPA
   best-of-3s (sd_match 0.15 + sd_game 0.35 — use this in any future
   match simulation, the nominal 0.352 per-match is wrong). Two independent errors had to be fixed, and
   they pull opposite ways.
   (a) **Side-out scoring FAKES clutch and the fake scales with skill**: a
   service run is a string of wins ending in exactly ONE loss, at the run's
   highest score and so its highest leverage, which manufactures the
   covariance at a true effect of zero; better servers have longer runs, so
   the artifact correlates +0.87 with own serve rate and +0.65 with v2
   rating. That is what `model/clutch.md` measured — its existence claims
   and named 7-player FDR list are RETRACTED.
   (b) **NEVER attribute a doubles rally to its server or receiver.** Four
   players contest it. Injecting TEAM-level clutch of tau=0.010 and reading
   it through server/receiver attribution recovers 0.0042 and names ZERO
   players (`clutch_circularity.py`). Use `model/clutch_team.py`: the cell
   is the whole game, the outcome is the SIDE's, both partners get the same
   number, individuals identified by partner rotation (same channel as v2;
   same reason actor/partner aren't separable within a pairing).
   Result vs a real-schedule no-clutch null (`clutch_null.py --model`,
   60 replayed seasons): doubles tau = 0.0050 [0.0032, 0.0066], singles
   0.0069 [0.0043, 0.0088], var(z) = 1.56 vs chance 1.00 — and var(z)
   RISES to 2.10 among 300+ game players (noise would fall). Ben Johns
   z = 5.28, Anna Leigh Waters z = 5.00 in a 336-player field whose null
   max is ~3.2 — BUT see the partner decomposition before naming both:
   Johns and Waters are mixed partners in 500 games (44%/46% of their
   doubles records), and `model/clutch_partner.py` shows Johns survives
   with her removed (z 3.27 on 648 games) while Waters does NOT survive
   with him removed (z 1.14 on 577 games — a real null, not thin data;
   Tardio is intermediate at 2.25). Defensible claim = "Ben Johns and the
   teams he plays on", not two independent players. ALWAYS run the
   partner split before publishing a doubles name — the team estimator
   gives both partners the same per-game number, so one dominant pairing
   can carry a whole career total. Doubles cross-era r = +0.171 (floor
   +0.037); singles serve-vs-return r = +0.207 (floor +0.068), +0.160
   after removing skill. **Clutch does NOT transfer doubles↔singles
   (r ≈ −0.10) — pooling the two DILUTES; never pool.** All taus are ~12%
   LOW: the null's abilities are fitted on clutch-contaminated data, which
   costs 12% of the effect (measured, `clutch_circularity.py`).
   **It is a MINORITY trait — do NOT read tau as "everyone has 0.005"**
   (`model/clutch_rare.py`, the right instrument once you accept clutch
   may be rare; var(z)/tau are DILUTION-prone and a Gaussian EB prior
   over-shrinks true outliers). Spike-and-slab b~(1−π)δ₀+πN(μ,σ²):
   doubles LR = 72.2 against π=0 vs a null LR of 1.0±1.3 (singles 18.2 vs
   1.1±1.9); fitted shape ≈ 13% of players carrying sd 0.014, rest zero.
   π alone is NOT quotable — π and σ trade off and null fits also return
   large π; the LR is the test. Tail counts doubles z>2.5: 6 obs vs null
   median 2; z>3: 3 vs 0 — and the excess sits in the 200-500 and 500+
   game bins (11 vs 4, 5 vs 1), NOT the thin 60-200 bin (9 vs 7), which
   is the direction that rules out se miscalibration. SELECT-THEN-VERIFY
   (the only test that establishes individuals): name top-K on 2024-25
   alone, measure that fixed roster on 2026 alone — K=40 gives z = 3.77
   vs null 0.29±1.00, K=5 gives 3.31 vs 0.02±0.97, and 0 of 30 no-clutch
   seasons ever reached it. Tail is TWO-SIDED (several z<−3); house
   position on not naming chokers stands. Size: tau 0.005 ≈ half a point
   of rally-win probability per sd of leverage; Johns ≈ one extra game
   per 100. Real, repeatable, concentrated in a minority.
   BETWEEN-GAME CLUTCH IS DEAD (`model/clutch_decider.py`, 13,767 PPA
   best-of-3s, 3,873 deciders): D = mean point-share residual in a
   player's game-3 deciders minus their other games. var(z) = 0.827 vs
   chance 1.000, tau = 0 [0, 0.0043], 2 players past |z|>1.96 where 8 are
   expected. Nobody elevates for the big match. NOTE THE TRAP — deciders
   are SELECTED (reaching 1-1 is evidence the favourite is having a bad
   day), the artifact correlates −0.68 with skill, and the race model's
   nominal SD_MATCH=0.352 OVERSTATES it ~13x: it reproduces neither the
   decider rate (0.219 sim vs 0.281 real) nor the selection structure
   (−0.213 vs −0.016), and subtracting it manufactured a confident false
   positive (var(z) 1.519, tau 0.017, 17 "significant" players). The null
   must be CALIBRATED to reproduce both observed quantities first —
   sd_match 0.15 + sd_game 0.35 does (0.281/0.281, −0.016/−0.016). A
   per-match effect lowers the decider rate and drives the bias; a
   per-game effect raises it and biases nothing.
   TWO DELIVERABLES, don't conflate: the TRAIT question (a forecast —
   needs replication + shrinkage; only Johns/Waters survive) vs the
   LEDGER (`model/clutch_ledger.py` → data/clutch_ledger.csv). CWPA =
   Clutch Win Probability Added, denominated in GAMES; +1.0 = one game's
   worth of win prob banked purely from WHEN the rally wins landed. Not
   shrunk (a record isn't an estimate) but still baseline-corrected —
   subtracting the side-out artifact sets where zero is, it is not
   shrinkage. Same family as finding 9's MVP outcome accounting. Career
   doubles: Johns +26.3, Waters +21.0, Tardio +15.4, Alshon +10.6,
   Erokhina +10.0 (best rate, +2.30/100). 2026: Johns +12.2. Singles:
   Haworth +8.4 (and near the BOTTOM in doubles — the non-transfer in one
   person). Publish CWPA next to its noise sd (±4.9 games per 1,000-game
   career); most of the list sits inside its own error bar.
   GOTCHA: the null's U must be rescaled to each player's OWN SSL before
   averaging — Waters' games run short, so an unscaled baseline charged
   her for leverage she never had and dropped her 2nd → 14th.
   Traps that ate this session: (i) under the WEAKER flat-rate null the
   cross-era correlation is +0.246 with calib slope 1.00 and looks like
   proof — skill is stable across eras so any residual skill-linked bias
   fakes it; residualising on skill kills it (−0.002). (ii) The attributed
   estimator + pooling gave a confident "no clutch at any size"; it was an
   estimator failure, not a finding.

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
- **Every MLP roster is exactly 3 men + 3 women** — four starters (2M+2W)
  plus a one-man, one-woman bench. Verified against all 8 playoff teams
  2026-08-09; `data/mlp_rosters_2026.csv` holds those six-player rosters.
  Use it as a SHAPE CHECK: `mlp_rosters()` infers membership from who has
  appeared, which structurally cannot see an arrival who has not played yet
  (Milan Rane, Angie Walker) and cannot drop a departure or an IR move
  (Grayson Goldin, Stefan Auvergne — whose +0.72 came off 17 games and who
  was never really on the team). So the inferred pool is a SUPERSET, and
  ranking it by value picks phantom players. Two real misfires this way:
  Layne Sleeth anchored Texas's projected women's pair while seven weeks
  into a shoulder injury, and Tyson McGuffin was projected into Palm
  Beach's MD after sitting out an entire playoff weekend. Membership is
  not availability — a player on the roster may still be unavailable, so
  cross-check against the team's most recent event before trusting a
  projected lineup. `make_forecast.roster_shape_warnings()` reports both
  failure modes; `data/roster_overrides.csv` fixes team assignment.
- **Score formats are data, not assumptions**: PPA Challengers hide
  single-game-to-15 rounds (side-out, NOT rally); MLP DreamBreakers are
  rally-to-21 singles and NEVER enter models (isTieBreaker flag). 2026 MLP
  dead 4th game at 3-0: generally PLAYED in round robin, recorded as a
  walkover (skipped) in bracket play — matchup structure lives in
  data/mlp_matchups_2026.csv (scraper/mlp_matchups.py, from the open BFF). matchCompletedType 5 = normal; 6 =
  walkover/cancelled; treat others as forfeits.
- MLP is identified by organizationSlug == "major-league-pickleball"
  (titles lie: Grand Rapids = "Edward Jones Mid-Season Tournament";
  exclude "Junior" titles). PPA filter: ppatour.com contact emails or
  \bPPA\b in title, minus Australia/Asia/College.
- 403 endpoints (getMatchInfos, getTeamLeaguesMatchupsOnDivision) are
  permanent — the "Short" variants + detail endpoints cover everything.
- **A matchup detail cached while SCHEDULED must never satisfy a reader
  who believes the matchup is over.** Lookahead fetches (tournament_state,
  forecasts) cache pre-completion snapshots; before 2026-07-26 these froze
  forever once the short status flipped to COMPLETED, silently dropping
  every played matchup (no MLP game 7/17–7/26 reached games.csv, and the
  title-race sim erased live results). pb_api.matchup_data now self-heals
  (refetch once when the cached payload's own matchupStatus isn't final) —
  keep that property if the caching layer is ever touched.
- **Browsers cannot reach the network from this environment** (egress
  gateway TLS-fingerprints and resets Chromium; curl/httpx fine). Don't
  waste time on Playwright; recon.md documents the diagnosis.
- **DUPR was deliberately removed from this project** (2026-07, user call:
  no interest in a "we beat DUPR" scoreboard). The embedded per-match
  "rating" field in the raw payloads IS the player's synced DUPR doubles
  rating and is still present in raw/, but we no longer extract, store,
  display, or benchmark against it — don't re-add it without asking. It
  was never a model input; nothing in v1/v2/singles depends on it.
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
- **Serve-aware DP BUILT 2026-07-16** (`web/sitelib/winprob.py`): exact
  4-state cell algebra, k measured = 0.43 (doubles). Each game's eta is
  ANCHORED so the DP's start-of-game prob equals the calibrated race
  prob — points cluster on serve possessions, so raw etas make the DP
  underprice favorites vs the validated pre-match numbers; anchoring
  keeps receipts consistent and k shapes only within-game dynamics.
  `web/replay_winprob.py --matchup <uuid>` renders any completed
  matchup's rally logs into charts (matchup track + per-game + DB
  rally-race panel). Smoke calibration (52 matches, 2.9k rally-states):
  all deciles track observed within noise, top decile ~88% obs vs ~95%
  pred — refit in-game calibration after the log backfill. Live mode =
  same engine fed by Tier-1/Tier-2 on the droplet (unbuilt).
- **LIVE PAGE SHIPPED 2026-07-16** (`site/live.html`, Pillar 5): today's
  MLP matchups + PPA pro doubles with rally-by-rally win prob. Engine =
  `web/sitelib/live_engine.js`, the validated JS twin of race.py+winprob.py
  (`node web/test_live_engine.mjs` cross-checks vs Python at 1e-9 — run it
  after touching either side). Data path: browser → Supabase Edge Functions
  `live`/`logs` (source `supabase/functions/`; same code as the alternate
  `netlify/functions/` backend — BFF/SSE send no CORS so a proxy is
  mandatory; responses memo-coalesced so upstream sees ≤1 sweep/15 s
  regardless of viewers). Config via env LIVE_API_BASE / LIVE_API_KEY at
  build. Mid-match joins backfill from getListLogs; no-log courts fall back
  to ~20 s scoreboard snapshots (localStorage). Pre-match numbers anchor to
  the calibrated race DP, so live curves agree with graded receipts at
  rally zero; DB panel uses the singles model.
- **Lineups (2026-07-17; ladder reworked 2026-07-26)**: make_forecast
  projects via a 3-tier ladder — (1) the matchup's own PUBLISHED lineups,
  (2) BEST LINEUP: top 2W+2M by v2 value from the team's season roster
  (roster = latest-MLP-appearance-wins per player, walked from
  SEASON_START; mixed split maximizes weakest-link pair strength), (3)
  last-completed-matchup fallback. Tier 2 exists because MLP Chicago
  priced Brooklyn at <1% off a stale July-10 B-squad lineup (Navratil
  called it out; best-lineup repriced them 83% pre-event and they lost
  the title matchup 3-2). tournament_state's pair matrix uses the same
  ladder. forecast.html still reprices client-side the moment actual
  lineups publish (projected vs LINEUPS OFFICIAL, same engine + a
  differs-from-projection flag; day-2 evidence: 9/10 matchups ran pairings
  that differed from projection). `scraper/lineup_freeze.py` (droplet
  timer 09:15 PT via deploy/run_freeze.sh) auto-freezes at-announcement
  forecasts to live/lineup_freezes-*.jsonl — timestamped, strictly
  pre-match, the "official lineups" tier to compare against the
  projected-lineup ledger. Kept deliberately light: the machine records,
  humans grade.
- **Rally-level history is backfillable** (found 2026-07-16):
  `/api/v1/results/getListLogs?id=<match_uuid>` (open BFF) returns the
  full referee log for completed matches — per-rally server/receiver
  UUIDs, points, side-outs, timestamps. 15/15 coverage in a 2024–26
  MLP+PPA sample. Estimate k and serve/return splits from the archive
  (build scraper/harvest_logs.py); live SSE is only needed for real-time.
  Schemas + log_type enum: recon.md. No shot-level data exists anywhere
  in this stack (ceiling: vision pipeline on broadcasts).

- **Serve/return lives in Supabase — query it, don't re-harvest**
  (2026-07-21). Rally logs are gitignored + droplet-only, and the committed
  per-player-year CSV is too coarse to slice, so serve/return questions used
  to force a fresh harvest every session. Fixed: the droplet upserts a
  per-match-per-player table to Supabase nightly (`scraper/upload_supabase.py`,
  wired into `deploy/run_logs_backfill.sh`, guarded on SUPABASE_URL /
  SUPABASE_SERVICE_KEY). Then ANY serve/return/points cut (player, season,
  tour, opponent set) is one SQL query — no logs touched.
    - Front door: `model/rally_stats.py` (`serve_return(...)`, `freshness()`);
      queries the store when SUPABASE_URL is set, else falls back to the
      committed `data/player_serve_rallies.csv`. `model/serve_return_report.py`
      does the population regression (who beats the field on return).
    - **Data catalog** (Supabase project `nwgxyytowbluuykbdcfc`, public read via
      anon key; writes = service role only):
      · `pb_match_player_serve` — grain: 1 row / player / match
        (side, serve_rallies, serve_wins). The base; slice it however.
      · `pb_player_serve_return` (view) — per (player, tour, year) serve AND
        return; return = opposing side's serve losses, reconstructed in SQL
        (team-attributed in doubles — a per-player RATE, never summed).
      · `pb_rally` — THE finest grain: 1 row / rally (server, receiver,
        server_side, server_number, outcome point|sideout|second, won, and
        running server/receiver score at rally start). Denormalized with
        tour/date so rally mining needs no joins. Unlocks score-state
        (win% at 9-9), serve-runs/streaks (order by rally_number), receiver
        splits, 1st/2nd-server effects. Built by H.rally_events (mirrors
        tally with correction handling; H.test_rally_events asserts it
        reproduces tally's serve counts exactly, wins to ~0.02% — rare
        multi-point rewinds; exact serve/return still comes from the
        tally-based pb_match_player_serve).
      · `pb_player` — dimension (player_uuid → full_name, gender) so queries
        read in names. Join on player_uuid.
      · `pb_meta` — freshness stamp (serve_rows, rally_rows, max_match_date).
    - Raw logs stay the source of truth: a tally-logic fix means re-derive
      from raw + re-upsert, NOT re-fetch. DB is a queryable cache.
    - Ceiling: no shot-level data exists anywhere (only referee logs); that
      needs a broadcast vision pipeline. Everything the logs contain is now
      in the warehouse.

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
"tonight" band off data/forecasts.json), live board (live.html — see the
live-page bullet above), power rankings (rankings.html),
499 player pages (trajectory + game-log-vs-expectation SVGs), client-side
matchup simulator (race DP + weakest link + uncertainty in embedded JS,
shareable permalinks), receipts ledger + calibration, record book,
methods, 404. The look is the PICKLES design handoff: master
stylesheet = CSS string in web/sitelib/style.py (design port verbatim +
landing additions; light AND dark). Conventions: values are displayed as
"expected margin vs an average pairing" via web/sitelib/race.py:value_points;
the race DP there mirrors model/v2_holdout.py AND the JS inside
build_simulator — keep all three in sync. Rankings rank 2026-active players
only; men/women always separate. `model/receipts.json` is the receipts
source of truth — commit predictions there BEFORE matches, grade after.
Unlisted pages: `web/insights/` ships verbatim to `site/insights/` —
public by URL, never linked from nav, noindex meta (user call 2026-07-26;
a linked "insights" section may come later). First page:
`insights/unsolved-meta/` (Anna Bright DreamBreaker post, design handoff
ported verbatim).

## Open threads (specced, unbuilt)

Weather (2026-07-28) — **full session narrative + open threads in
`model/weather_thread.md`; read that first if picking this back up.**
Summary: `scraper/weather.py` resolves every
event's venue lat/lon/tz from the BFF (PPA: getTournamentsOnDate by
TournamentID; MLP: getTeamLeaguesResultsOnDate location object) and pulls
the Open-Meteo archive → data/event_geo.csv + event_weather{,_hourly}.csv;
`model/weather_report.py` runs the day-level cuts (outdoor vs indoor
placebo). Findings so far: NO wind effect on serve-point rate outdoors
(+0.002 per +10 mph, CI spans 0; indoor placebo equally null); favorites'
obs−pred edge drifts a bit more negative in wind and in 92°F+ heat
outdoors, but tail bins are thin and the indoor CONTROL arm moves too at
20+ mph — so the wind-upset signal can't be cleanly attributed (label
noise or storm-day confound). HOUR-level join DONE same day:
`scraper/extract_match_times.py` sweeps localDateMatch* start times from
the open BFF (instant where raw/ is warm, e.g. droplet) →
data/match_times.csv; both reports auto-upgrade to wind/temp at match
hour. Hour-level: serve rate still null; favorites −6.0pp
[−8.0,−4.1] at 14–20 mph outdoors vs −4.0 calm (indoor arm still dips —
keep skeptical); heat gradient monotone but mild. Court-END (good/bad
side) effects: physically unobservable in logs (server_side = team, not
end) BUT inferred from the switch schedule in `model/end_effects.py` —
Design A consecutive-game residual cov (ends swap between games),
Design B pre/post point-share around the mid-game switch at 6
(data/decider_splits.csv from pb_rally) — PPA deciders PLUS every MLP
game (MLP switches at 6 in ALL games, user rule 2026-07-28; MLP matches
are single games; DBs excluded). PRIMARY test = paired-swing
VARIANCE vs noise (user call 2026-07-28: mean-of-swing is identically 0
by end-assignment symmetry; the paired difference cancels skill exactly,
so the signal is excess variance — correlation kept as secondary only).
Verdict: Design A flat but low-powered (game noise ~38 pts²; can only
see end-adv sd ≳1 pt/game). Design B excess is monotone in match-hour
wind (+42 indoor ≈ +42 calm < +49 mod < +55 windy ×10⁻³ share²) BUT the
SIMULATED NULL (winprob.py serve-state model, k=0.43, match etas, no
momentum, no ends — sim_game in end_effects.py) reproduces z² ≈ 1.8
everywhere: observed 1.67–1.95 vs null 1.76–1.79 (Design B, n=4,786
with MLP; only windy 14+ sits above its null, +0.16, CI spans it) and
the DUMMY-CONTRAST regression (the weather test proper, no sim needed:
Δ mean z² vs outdoor-calm reference, cluster-bootstrapped) gives
indoor −0.06 [−0.18,+0.06], windy 14+ +0.22 [−0.16,+0.59], continuous
slope +0.17 z² per +10 mph [−0.05,+0.36] — wind-positive direction,
nothing significant; sim null is only for the momentum question.
Design C = RALLY level (serve-rally win rate per side per half,
data/decider_serve_splits.csv; serve rallies ≈ iid Bernoulli so the
mechanical null falls to ~1.15 and team B adds independent info): same
verdict — indoor/calm/moderate AT their nulls, windy 14+ 1.34
[1.10,1.66] vs null 1.20, Δ vs calm +0.19 [−0.07,+0.54], continuous
slope ≈ 0 — the windy bump is tail-bin only, no dose-response.
WIND SKILL (orthogonal F1-rain dimension, `model/wind_skill.py`):
split-half reliability of per-player wind slopes r = +0.06 vs
permutation null [−0.07,+0.12] over 552 players / 24.8k outdoor
games — NOT ESTABLISHED (clutch/durability cleared 0.13–0.15 on the
same design; wind skill doesn't). Standout (Verstappen) tests too:
max|z| 3.27 across 552 players, permutation p=0.58; |z|>2 count 29 vs
null median 27 [19,36]; pre-specified Anna Bright slope −0.014/10mph,
p=0.84 (per-player resolution ~±3pp share per 10 mph — only large
individual effects detectable). Leaderboard is noise, never publish
names off it. RARE-TRAIT SHAPE ALSO NULL (2026-08-03,
`model/wind_rare.py` + `wind_rare_power.py`, weather_thread.md §7):
the full clutch battery (spike-slab LR 5.8 vs null max 7.2; tails
null both directions; select-then-verify p≥0.17 both tails, strong
side anti-persists) on the IDENTICAL panel, with an injection floor —
a 13% minority at ±0.02 share/10mph fires the battery 75%, ±0.03
100%, 0/20 false positives. So: no minority wind trait ≥0.02
share/10mph; smaller is below the telescope. Don't re-open without a
bigger archive or a sharper wind measure (court-level anemometer). FAVORITES×WIND KILLED (2026-07-28, `model/favorites_wind.py`,
data-referenced nulls only): continuous interaction share~skill+wind+
skill×wind gives d = +0.002 [−0.060,+0.064] OUTDOOR (24.8k games — no
compression at all; b≈1.04 so v2 shares are near-perfectly calibrated)
while INDOOR shows d = −0.080 [−0.150,+0.020]; rally-level fav−dog
serve-rate gap slope is negative in BOTH arms and MORE indoor (−0.031
sig) than outdoor (−0.022) — the falsification arm fails, so the old
binned "favorites −6pp at 14–20 mph" was composition/label noise, NOT
wind. Rally-level binomial logit (P(server wins rally) ~ adv×wind,
193k rallies — the exact 0/1-series likelihood; per-match covariates
make it collapse to (wins,attempts)) agrees: d outdoor −0.017
[−0.098,+0.058], indoor −0.060 [−0.129,+0.014], indoor again more
negative. Weather verdict, complete: no serve-rate effect, no end effect,
no momentum, no wind skill, no favorite compression. Wind does nothing
detectable to pro pickleball outcomes in this archive — and
1.81–1.90 vs 1.84–1.88 (Design A2 = game1-vs-game2 shares, all 14k
matches, no ratings needed). ⇒ (a) WITHIN/BETWEEN-GAME MOMENTUM IS
~ZERO — serve-streak clustering explains the entire swing
overdispersion (matches spec-shootout's null momentum challenger);
(b) the wind hint shrinks to +0.13 z² over its own null floor, CI
spans 0. End effects: still no confirmed signal anywhere. HOUSE STANCE
(user, 2026-07-28): indoor is a CONTROL, never assumed end-effect-free
— more controlled, not fully. Indoor/outdoor = tour default + venue
keywords; curate data/venue_overrides.csv (event_id,setting) as
broadcasts confirm.
Website extras: live win-prob charts (needs Tier 1/2 listener on a VPS);
social prediction-card renders (design bundle `Prediction Cards.dc.html`,
port later). Deploy is .github/workflows/site.yml (build + Pages deploy on
push to main, nightly data refresh); one-time setup = repo Settings →
Pages → Source "GitHub Actions". Scorebug OCR of YouTube broadcasts could
backfill point-by-point history (Tier 0 of the vision pipeline;
championship-court sample bias noted).
