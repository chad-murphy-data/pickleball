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
