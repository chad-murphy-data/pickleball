# recon.md — how the data is actually obtained

**TL;DR: no browser needed, no token needed.** pickleball.com is a Next.js app whose
client bundle calls **same-origin BFF routes** (`pickleball.com/api/v1|v2/results/*`).
Those routes proxy the locked `api.pickleball.com` backend server-side — the Next.js
server injects the `PB-API-TOKEN`, and the BFF itself requires **no authentication**.
Plain `httpx` GETs work. The entire harvest is plain HTTP at ~1 req/s.

## How the endpoints were found

The handoff's recon was right that the token isn't in the client bundle and that the
pages are client-rendered — but the conclusion "this is a browser automation job"
turned out to be avoidable:

1. **Browser interception was attempted first and is dead in this environment** —
   not because of the site, but because this sandbox's egress gateway resets
   Chromium's TLS ClientHello mid-handshake (`ECONNRESET` after a successful
   proxy `CONNECT`; verified via Chromium net-log). curl/httpx handshakes pass.
   TLS-fingerprint filtering, effectively. Flags (`--disable-http2`, no-ECH,
   `--ignore-certificate-errors`) don't help. A local mitmproxy bridge would work
   if a real browser is ever needed. It wasn't:
2. **The prior recon grepped the bundle for `api.pickleball.com` and tokens — the
   right grep is for relative `/api/` paths.** Downloading the ~44 webpack chunks
   from `/results` and grepping for `fetch("` reveals the whole BFF surface in
   `chunks/7733-*.js` (react-query fetchers, one per endpoint).

## The BFF surface (all unauthenticated GET)

### PPA (tournament) chain
| step | endpoint |
|---|---|
| 1. tournaments active on a date | `/api/v1/results/getTournamentsOnDate?date=YYYY-MM-DD` |
| 2. event groups for a tournament+date | `/api/v1/results/getListActiveEventsFlatGroup?tournamentId={id}&date={d}` → find group `"Pro Events"` (`bracket_level_id=2`) |
| 3. pro events active that date | `/api/v1/results/getTournamentEventsShort?tournamentId={id}&formatId=0&playerGroupId=0&bracketLevelId=2&date={d}` → e.g. `"Mixed Doubles Pro Main Draw"` |
| 4. all matches for those events that date | `/api/v1/results/getMatchInfosShort?eventIds={uuid,uuid}&date={d}` |
| 5. (enrichment) full single-match record | `/api/v1/results/getResultMatchInfos?id={match_uuid}` |

Step 4 returns, per match: **all four player UUIDs + names**, per-game scores
(up to 5 games), `score_format_game_best_out_of`, `match_status`,
`match_completed_type`, `round_text`, event/tournament uuids. Step 5 adds the
exact score format (`"1 game to 15 win by 2"`, `is_rally_scoring`, per-game
point targets) — needed because **PPA Challengers play early rounds as a single
side-out game to 15**, which the short payload doesn't distinguish from Bo3-to-11.

### MLP (team-league) chain
| step | endpoint |
|---|---|
| 1. leagues active on a date | `/api/v2/results/getTeamLeaguesResultsOnDate?date={d}` |
| 2. matchups (team fixtures) | `/api/v2/results/getTeamLeaguesMatchupsShortOnDivision?teamLeagueId=&organizationId=&divisionId=&seasonId=&districtId=&date=&matchupGroupUuid=` (all ids from step 1's `divisions[]`) |
| 3. fixture detail | `/api/v2/results/getResultsMatchupData?matchupId={uuid}` → `matches[]` |

Step 3's `matches[]` is the full authenticated-API schema from the handoff
(`teamOnePlayerOneUuid`, `teamOneGameOneScore`, `scoreFormatTitle`,
`isRallyScoring`, `isTieBreaker`, …).

Not every route is open: `getMatchInfos` (non-short) and
`getTeamLeaguesMatchupsOnDivision` (non-short) return
`403 Method Forbidden` — the "Short" variants + the single-match/matchup detail
routes cover everything we need.

### Sample raw payload (MLP matchup detail, abridged)
```json
{
  "matchUuid": "65a1681d-da9a-43c5-8d5e-d824bc95cb4f",
  "teamOnePlayerOneName": "Callie Smith", "teamOnePlayerTwoName": "Liz Truluck",
  "teamTwoPlayerOneName": "Alix Truong",  "teamTwoPlayerTwoName": "Daria Walczak",
  "teamOnePlayerOneUuid": "0c1bb94d-3af2-42f4-b938-d712c707ace4",
  "teamOneGameOneScore": 9, "teamTwoGameOneScore": 11,
  "matchStatus": 4, "matchCompletedType": 5, "winner": 2,
  "roundText": "RR Womens Doubles", "matchAbbreviation": "WD",
  "scoreFormatTitle": "1 game to 11 win by 2", "isRallyScoring": false,
  "isTieBreaker": null,
  "moduleSubTitle": "MLP St. Louis - 2026 Regular Season - 2026 Season"
}
```
DreamBreakers appear as a 5th match with `isTieBreaker: true`,
`matchAbbreviation: "DB"`, `isRallyScoring: true`, format `"1 game to 21 win by 2"` —
trivially separable, and they carry only one named player per side (the rotation
gimmick), which independently keeps them out of any dyad table.

### Validation against known results
The handoff's sanity example is reproduced exactly by the pipeline:
2026-01-18, Carvana PPA Masters, Mixed Doubles Pro **Finals** (best-of-5):
Ben Johns / Anna Leigh Waters def. Anna Bright / Hayden Patriquin
**11-7, 7-11, 9-11, 11-7, 11-4** ✓

## Selection & scope decisions
- **MLP**: one `teamLeagueId` persists all season; the *title* and
  `matchupGroupUuid` change per city stop ("MLP Dallas" → "MLP St. Louis" → …
  → "Edward Jones Mid-Season Tournament" for the Grand Rapids stop, which is
  why the filter must key on `organizationSlug == "major-league-pickleball"`
  rather than the title). "Junior MLP …" events share the org and are excluded
  by title; MLP Australia is a different org (`ppa-tour-australia`). `event_id`
  := `matchupGroupUuid` (the city stop), `event_name` := the matchup group
  title.
- **PPA**: a tournament is PPA-linked if any contact email ends `@ppatour.com`
  or the title contains the word "PPA" (main stops are titled "PPA Tour: …" but
  the email rule also catches oddballs). **Excluded** by regex: `PPA TOUR
  AUSTRALIA`, `PPA Asia`, College Pickleball tour — different player populations
  (they'd show up as disconnected graph components anyway). Amateur-only events
  (Powerball state champs etc.) are excluded structurally: they have no
  `"Pro Events"` bracket group, so nothing is ever fetched for them.
  **PPA Challengers are included** (they have real Pro doubles draws and add
  graph connectivity); `event_name` carries "Challenger" so they're one
  `WHERE` clause away from exclusion.
- Singles never enter: only events whose title contains "Doubles" are fetched.
  Senior Pro / Junior bracket groups are skipped at the group level.
- 2026 only: the date sweep starts 2026-01-01 (PPA season opener was Jan 9).

## Caching / re-runs
Every response is cached under `raw/` (gitignored) before parsing; the parser
never touches the network. Dates within the last 3 days are treated as volatile
and refetched (live events change); MLP matchup details are refetched until
`matchupStatus == 4` (completed). Full season sweep ≈ 25 min at ~1 req/s;
incremental re-runs only fetch new days.

```
python scraper/harvest.py          # sweep season → raw/
python scraper/enrich_formats.py   # resolve ambiguous score formats → raw/
python scraper/parse.py            # raw/ → data/*.csv
python scraper/diagnostics.py      # data/ → diagnostics.md
```

## Gotchas encountered (so nobody re-trips on them)
- **UUID case is inconsistent** in the API (`FC05556E-…` vs `fc05556e-…`).
  Everything is lowercased at parse time; a case-split identity would silently
  disconnect the partner graph.
- Player names come with trailing spaces and embedded nicknames
  ("Tyra Hurricane Black "). UUIDs are the identity; names are display only.
- The handoff listed "Federico Tardio" — the actual player is **Gabriel
  Tardio** (Federico Staksrud is a different player). Both are tracked as focal.
- `match_completed_type == 5` is normal completion; anything else is treated as
  forfeit/retirement/walkover (`is_forfeit=True`, kept in games.csv but excluded
  from all diagnostics and intended to be excluded from the model).
- A few "completed" matches have all-zero scores (walkover shells) — dropped
  with reason `no played games`.
