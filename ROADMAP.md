# Roadmap — everything we've specced, in build order

Effort keys: **S** = an evening · **M** = a weekend-to-week · **L** = a month
of evenings · **XL** = a season-long side quest. Status: ✅ built, 🔜 next,
⬜ specced only.

## Phase 0 — Foundation (done, this is the platform everything sits on)

- ✅ Full-history scraper: 36k games, 2024–2026, cached/idempotent/re-runnable
- ✅ v1 season SRM (points scale) + diagnostics + analysis writeups
- ✅ v2 model: dynamic monthly skill, race-to-T likelihood, weakest link —
  validated at 77.4% holdout vs DUPR's 64.7%
- ✅ Findings library: weakest-link γ, chemistry-is-small, gender-blind
  targeting, Johns-never-declined, trajectory curves
- ✅ Forecast machinery: any matchup → odds, score distributions, outcome
  trees, hinge index (worked example: Mid-Season Gold final, committed
  pre-match)
- ✅ Serve-aware win-prob math (side-out cycle DP; baseline k assumed)
- ✅ Tier-1 live poller (`scraper/live_poller.py`, awaiting first live event)
- ✅ Design handoff for Threads system; EXPLAINER; CLAUDE.md

## Phase 1 — Receipts operations (now → September; S each; no new infra)

- ✅ **Grade the Gold final forecast** — graded 2026-07-13 from the API
  matchup record: STL won 3-0; overall 61% call HIT, headline WD 88% call
  MISS (full table in model/prediction_midseason_final.md). First entries
  live in `model/receipts.json`, the machine-readable public ledger.
- 🔜 **Run the poller for real** during MLP San Diego (Jul 16–19) — shakes
  out Tier 1 and starts accumulating live JSONL
- 🔜 **SSE discovery session** (Tier 2): protocol decoded + handshake
  verified (recon.md); `scraper/sse_probe.py` ready. Remaining: run it
  during a live match this weekend to capture real event shapes, then fold
  the parser into live_poller. Unlocks rally-resolution + serve/return est.
- ⬜ **September obligations** (scheduled in CLAUDE.md): score the
  preregistered chemistry predictions on post-July-12 games; season-end
  re-harvest + v2 refit + refreshed leaderboards/trajectories
- ⬜ **Threads launch content**: forecast card + receipts scorecard for one
  event weekend, using design_handoff.md §E fields (S per card once the
  design system exists)

## Phase 2 — Website MVP (M–L total; static site, no backend)

All pages regenerate from existing CSVs in ~4 s: `python web/build_site.py`
→ `site/` (gitignored build artifact; 506 pages). Host anywhere static.

- ✅ ★ PILLAR 1 — Power rankings page (men/women, uncertainty bars, form arrows)
- ✅ ★ PILLAR 2 — Player pages: trajectory curve + band, W/L splits, clutch stats
  (deciding-game/overtime records), DUPR overlay, partner network,
  **game log vs expectation** chart (499 pages, one per tracked player)
- ✅ ★ PILLAR 3 — **Matchup simulator** — values embedded, race DP + weakest
  link + uncertainty integration in client-side JS; shareable permalinks
- ✅ ★ PILLAR 4 — Receipts ledger + calibration curve, generated from
  `model/receipts.json` (Gold final graded; chemistry preregistrations pending)
- ✅ Methods page (EXPLAINER rendered + technical appendix), record book
  (streaks/upsets/marathons mined from games.csv)
- ✅ DUPR-vs-model scoreboard page (scatter + biggest-disagreement tables)
- ✅ Open CSV downloads page (site/data/ + schema notes)
- ✅ Deploy + nightly rebuild: .github/workflows/site.yml (GitHub Pages;
  nightly run refreshes games/ratings/forecasts from the API with a cached
  raw/). One-time setup: Settings → Pages → Source = "GitHub Actions"
- ✅ **Forecast page** (pre-match, receipts culture automated):
  web/make_forecast.py prices scheduled MLP matchups with projected
  lineups → forecast.html; `--commit` freezes them into receipts.json
- ✅ **Recent-results page**: last 14 days, every game with the winner's
  pre-game price + upset chips
- ⬜ Custom domain

## Phase 3 — Live layer (M; needs a persistent $5 VPS)

- 🔜 Deploy Tier-1 poller as a systemd service, schedule-aware — kit ready
  in `deploy/` (install.sh + user units), just needs a box
- ⬜ ★ PILLAR 5 — **Live win-probability charts**: pre-match lookup tables (score × 4
  serve states) shipped as JSON; browser indexes as events arrive; step
  chart with annotations + uncertainty ribbon
- ⬜ Layer Tier-2 SSE feed on top when discovered (rally resolution)
- ⬜ Auto-generated post-match win-prob chart images → biggest-swing posts

## Phase 4 — Tools only this model can power (M each)

- ⬜ **MLP lineup optimizer**: assignment problem over pairings
  (values + weakest-link + chemistry) vs a specific opponent
- ⬜ **Season simulator**: Monte Carlo remaining schedule → title odds,
  refreshed weekly
- ⬜ Trade machine (swap MLP rosters → odds shift)
- ⬜ Partner-choice calculator (the weakest-link finding as an interactive)
- ⬜ Beat-the-model game: users lock picks pre-match, humans-vs-model
  leaderboard (needs light backend or forms hack)

## Phase 5 — Model v3 (after more data / rally feeds exist)

- ⬜ Expanding-window cross-validation (event-by-event refits; hours of
  compute, rigor upgrade for any model comparison)
- ⬜ Serve/return split estimation from accumulated point sequences →
  replaces the assumed k baseline; serve-aware model layer
- ⬜ Per-player senior/junior role effects (phase 2 of the weakest-link
  finding: WHO carries weaker partners best — identified via role-switching
  mid-tier players)
- ⬜ Clean ramp-up estimation (pair histories now span 2024+, but window
  edges still need care)
- ⬜ Posterior rank intervals on leaderboards; selection-aware game-3
  handling (both small rigor upgrades)
- ⬜ 2023 harvest if ever wanted (format archaeology required; diminishing
  returns)

## Phase 6 — Vision / rally data (the season-long side quest)

- ⬜ **Tier 0: scorebug OCR** of archived YouTube broadcasts → retroactive
  point-by-point + server sequences for championship-court matches (M;
  best effort-to-payoff in the whole vision stack; sample bias noted)
- ⬜ Vision MVP: who served/returned via rules-engine + one positional
  check per rally (identity bootstrapped by hand per match) (L)
- ⬜ Last-shot attribution (audio rally-end + pose in final seconds) (L,
  ~85–90% accuracy ceiling)
- ⬜ Speedup detection (fuzzy ground truth; label-hungry) (L)
- ⬜ Full shot-level pipeline (ball tracking, shot taxonomy) (XL, research)

## Dependencies at a glance

```
poller (✅) ──► live charts (P3) ──► biggest-swing content (P3)
SSE discovery (P1) ──► rally data ──► serve/return model (P5)
scorebug OCR (P6) ──────────────────► serve/return model (P5)
website MVP (P2) ──► tools (P4) ──► beat-the-model (P4)
receipts grading (P1) ──► ledger page (P2)   [everything else: no deps]
```

The through-line: every phase makes the receipts pile taller. Nothing here
requires abandoning the honesty brand to be interesting — the model being
publicly wrong 23% of the time IS the product.
