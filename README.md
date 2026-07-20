# Pro Pickleball Match Scraper — MLP + PPA, 2026 season

Game-level doubles results for a Social Relations Model on point margins:
both player identities on both sides of every game, plus the game score.

- **`recon.md`** — how the data is obtained (endpoint discovery, no browser,
  no token; selection rules and gotchas).
- **`diagnostics.md`** — the go/no-go gate: dyad counts, partner counts,
  partnership-graph connectivity per context, margin distributions.
- **`data/games.csv`** — one row per game (the modeling unit).
- **`data/players.csv`** — canonical players keyed by source UUID
  (+ `data/name_variants.json` audit).
- **`data/dreambreakers.csv`** — MLP DreamBreakers, kept out of the modeling set.
- **`data/dropped.csv` / `data/flags.csv`** — every exclusion with a reason;
  rows kept but worth review.

## Pipeline

```bash
pip install httpx
python scraper/harvest.py          # season sweep → raw/ cache (~25 min first run)
python scraper/enrich_formats.py   # resolve ambiguous score formats → raw/
python scraper/parse.py            # raw/ → data/*.csv
python scraper/diagnostics.py      # data/ → diagnostics.md
```

Idempotent and resumable: `raw/` (gitignored) caches every response; re-runs
only fetch new/recent dates. The season is in progress — re-run the pipeline
to pick up new events.

`scraper/discover.py` is the network-interception probe kept for reference;
the discovered BFF endpoints made it unnecessary (see recon.md).

## Website

A fully static site (power rankings, player pages with skill trajectories,
a client-side matchup simulator, the public receipts ledger, record book,
methods) regenerates from the CSVs in seconds:

```bash
python web/build_site.py           # data/*.csv + model/receipts.json → site/
python -m http.server -d site      # preview at http://localhost:8000
```

`site/` is gitignored — it is a build artifact. `.github/workflows/site.yml`
builds and deploys it to GitHub Pages on every push to main and nightly
(the nightly run also refreshes games/ratings/forecasts from the API).
One-time setup: repo Settings → Pages → Source = "GitHub Actions".
`model/receipts.json` is the source of truth for the receipts page: every
prediction is committed there before the match and graded after.

### Forecasts (receipts culture, automated)

```bash
python web/make_forecast.py            # price next 7 days of scheduled MLP
                                       # matchups → data/forecasts.json
python web/make_forecast.py --commit   # ALSO freeze them into the ledger
                                       # (do this BEFORE the matches, then
                                       # git commit + push)
```

Lineups are projected from each team's most recent completed matchup and
labeled as such; per-game probabilities use current v2 values + weakest
link + calibration; the DreamBreaker is 50/50 by stated convention.

## Live poller — event-weekend runbook (e.g. MLP San Diego, Jul 16–19)

The Tier-1 live poller needs a machine that stays awake during play — a
laptop that won't sleep, a Raspberry Pi, or any $5 VPS. Not CI.

```bash
git clone <this repo> && cd pickleball && pip install httpx
python scraper/live_poller.py --once        # smoke test (any day)
nohup python scraper/live_poller.py >> live/poller.log 2>&1 &   # event days
```

Output accumulates in `live/events-YYYYMMDD.jsonl` (one line per observed
score change, ≥15 s polling interval — keep it polite). Afterwards, commit
the JSONL: it is the ground truth that unlocks live win-probability charts
(ROADMAP Phase 3) and serve/return estimation (Phase 5). If a match is
live, also try dumping the Tier-2 SSE stream (`rte.pbgql.co`, see ROADMAP)
from the same machine — discovery only works during play.
