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

## Tier-2 live SSE feed (rte.pbgql.co) — DECODED 2026-07-15

The live win-probability plan's "Tier 2" is no longer undiscovered. The
client's SSE machinery lives in webpack chunk `1444-*.js` (the `O` class,
an XHR-based EventSource polyfill) and its React wiring in `7742-*.js`
(`RTEProvider`). Decoded from a static bundle pull — no live match needed
for the protocol, though real event *shapes* still need a live capture.

**Endpoint & environments** (from the bundle's config object):
- prod `https://rte.pbgql.co`, dev `rte-dev.pbgql.co`, uat `rte-train.pbgql.co`,
  local `:6969`. Path is `/live-scoring`; query `?opts=slice,karma`
  (prepend `withLogs` only when subscribing to a single match).

**Auth is not a credential.** Header `PB-RTE-TOKEN` =
`base64(JSON{ua, origin, fingerprint})`. The browser computes `fingerprint`
via FingerprintJS, but the server accepts any well-formed token — a random
32-hex fingerprint returns `200 text/event-stream`. Same unauthenticated-
public-feed posture as the BFF; we synthesize the token like the client does.

**Subscription headers** (base64 of comma-joined UUIDs):
`X-Request-Matches`, `X-Request-Matchups`, `X-Request-Tiebreaker-Matches`.
Events are SSE blocks named by the match UUID (score updates) or
`matchup_<uuid>` (matchup/DreamBreaker state); payload is JSON in `data:`.
With `withLogs`, per-rally events carrying `log_index` also arrive — the
rally-resolution feed the Tier-1 poller structurally cannot see.

**Verified from this environment** (2026-07-15, a no-event day): handshake
returns `200 text/event-stream` and holds the connection open; subscribing
to a *completed* match yields zero events (correct — nothing live). Real
event payloads must be captured during a live match. Tool: `scraper/sse_probe.py`
(`--matches`, or auto-discovers today's live UUIDs via the Tier-1 discovery;
`--duration` bounds the run; `--with-logs` for single-match rally logs).
~~Next step: run it during an MLP San Diego / PPA Macon match this weekend~~
DONE 2026-07-16 — real payloads captured during live play; see next section.

## Tier-2 event shapes — CAPTURED LIVE 2026-07-16 (MLP San Diego)

Two probe windows (~8 min total, `live/sse-20260716.jsonl`, 48 events)
during MXD2 Todd/Daescu vs Rane/Staksrud on the championship court
(`courtTitle: "CC"`). Both subscription modes verified end-to-end from
this environment via httpx.

**Match-state events** (named by bare match UUID; also what the broad
no-`withLogs` subscription delivers): the payload is a full camelCase
match object — the same shape as the BFF's MLP match records. Notable
fields: complete serve state (`server`, `serverFromTeam`,
`currentServingNumber`), per-game scores + `game*Status`, `matchStatus` /
`winner` / `matchCompletedType`, all four player UUIDs + names, court
UUID/title, `isConsumedByRefereeApp: true`, and timestamps. CAUTION: the
`localDateMatch*` fields carry a `Z` suffix but are venue-LOCAL times
(observed `localDateMatchStart: ...T10:51:13Z` for a match that started
10:51 PDT). Events usually arrive in pairs ~0.2–0.4 s apart (referee-app
score entry + serve-state update); occasional unchanged re-pushes occur —
dedupe downstream, same rule as Tier 1.

**Rally log events** (`reflog_<match_uuid>`; only with `withLogs`, which
the client only uses on single-match subscriptions): one structured
referee-log entry per action, 1:1 interleaved with state pushes:

```
{id, referee_uuid, match_uuid, server_uuid, receiver_uuid, server_index,
 date_created: {seconds}, log_type, LogData: {<TypedLog>} | {},
 game_number, group_uuid,
 start_score_current_game_string, end_score_current_game_string,  # "5-4-2"
 log_index}                                # dense, sequential per match
```

`PointLog` payload: `{time_started/time_ended: {seconds}, team_uuid
(the MLP FRANCHISE uuid — logs carry team identity), start_score,
end_score}`. Score strings are serving-team-first: `"5-4-2"` = server
team 5, receiver team 4, server #2. `log_type` enum observed so far
(45 logs across two matches):

| type | meaning (observed)                             | LogData            |
|------|------------------------------------------------|--------------------|
| 12   | rally underway (start marker; score unchanged) | {}                 |
| 14   | serving team wins the rally → point            | PointLog           |
| 16   | side-out — serve crosses to the other team     | {} (string flips perspective) |
| 17   | court-side switch (at 6 in a game to 11)       | SwitchCourtSideLog |
| 23   | serve passes to the second server (partner)    | {}                 |
| 37   | video challenge                                | VideoChallengeLog {team_uuid, challenge_referee_call} |

Unseen so far (expect on volume): game/match end, timeouts, faults,
injury/medical, DreamBreaker-specific types. The 16/23 rows appear to
carry the POST-transition server's perspective — verify on a full game
before hard-coding. No `matchup_<uuid>` events arrived during two
windows of mid-game play despite subscribing — they presumably fire
only on matchup-level transitions (match completed, DreamBreaker
created, lineup lock); still uncaptured, as is the
`X-Request-Tiebreaker-Matches` DreamBreaker feed and any PPA match
(test during Macon, Fri 7/17+).

Other state-event fields worth knowing: `localDateMatchPlannedStart` vs
`localDateMatchStart` (this match ran 39 min AHEAD of schedule —
delay/schedule analytics, "when to tune in"), `matchupUuid` (in-stream
join key), player countries + photo URLs, seeds (0 in MLP; presumably
bracket seeds in PPA), `logCreatedAt` (server-side ns timestamp),
lifecycle timestamps (confirm/dispute/auto-complete), and
`tieBreakerIsDirectFinalScore`.

**Early empirics — serve-rally win rate** (the DP's assumed k):
across the 20 fully-logged rallies, the serving side won 5
(k̂ ≈ 0.25, Wilson 95% ≈ 0.11–0.47, vs the ASSUMED 0.35–0.45).
Far too little data to conclude — but it leans low. (Superseded within
the hour: estimate k from the HISTORICAL logs below instead.)

## getListLogs — the archive has rally-level history (found 2026-07-16)

**`GET /api/v1/results/getListLogs?id=<match_uuid>`** (open BFF, no auth)
returns the COMPLETE referee log for a completed match — same log stream
the SSE feed delivers live, so rally-level data is BACKFILLABLE without
having listened. Found by pulling `/results` chunks and grepping
`"/api/` (the enum strings PointLog etc. are NOT in the /results chunks;
the endpoint was). Works for doubles AND singles match uuids.

**Coverage is EVENT-dependent, not universal** (the first 15/15 random
sample was misleading): a 92-match validation batch found PPA Gold Coast
Open 2024 with 0/30 logged and a Challenger with partial coverage —
events/courts without digital refereeing simply have no logs. Empty
responses are cached and flagged; the full harvest measures true
coverage. Where logs exist they are rich (59–379 rows/match) and
**~97% score-reconcile exactly against games.csv** via
`scraper/harvest_logs.py --summarize` (the rest are flagged `mismatch`
— observed causes: log ends before the final point was entered, or the
platform score was edited after the fact).

REST shape differs slightly from SSE: typed payloads are inline
snake_case keys (`point_log`, `match_over_log`, …) instead of the
`LogData` wrapper, timestamps are ISO strings instead of proto seconds,
and pre-rally admin rows appear that SSE mid-game never showed. Decoded
from one full game (WD Todd/Black 11-2, 59 rows, arithmetic validates —
13 points + 4 side-outs + 3 second-server passes = 20 rally markers):

| type | meaning | payload |
|---|---|---|
| 38 | match setup (log_index 1) | — |
| 46 | pre-start, unknown | — |
| 10 | player court arrival (timestamped) | court_arrival_log {player_uuid} |
| 15, 3, 31 | pre/mid, unknown, score-neutral | — |
| 32, 41 | DreamBreaker-specific (32 ≈ server rotation, 16× in a 35-rally DB; DB score strings have NO server number — "14-21" not "14-21-1") | — |
| 20 | warm-up begins | warming_up_log |
| 22 | scoring starts | start_scoring_log {left/right_of_umpire_team_uuid, serving_team_uuid} |
| 2 | challenge (referee call) | challenge_log {team_uuid, referee_call} |
| 45 | line review | line_review_log {team_uuid, line_review_success_status} |
| 18 | timeout (start + start/end pair) | timeout_log {team_uuid, time_started/ended} |
| 35 | additional timeout | additional_timeout_log {team_uuid} |
| 4 | game over | game_over_log {team_uuid} |
| 6 | match over | match_over_log {team_uuid} |

**Log-quality quirks the tally logic must handle** (all observed in the
wild; `harvest_logs.py:tally` implements these rules):
- Corrections are type-14 rows with NEGATIVE payload deltas (successful
  challenge → point removed); on rewind rows the score STRINGS are
  garbled (mixed perspectives) while the payload stays clean.
- Phantom double-entries: payload repeats `+1` (start None) but the
  string doesn't move. Real points with a merely STALE string also
  exist — disambiguate by whether the next row carries the score
  forward incremented (side-outs flip the string perspective).
- When string and payload deltas disagree, the smaller-|delta| one has
  been correct in every observed case.
- Scoring sometimes opens LATE: the first point row's string starts at
  0-0 while its payload starts higher — the gap is real score with no
  logged rallies (credit to the side, exclude from k).
- The payload team_uuid is a FRANCHISE/side uuid: learn team→side from
  the log's own normal points (scoring team = serving team in side-out),
  then apply to corrections.

(12/14/16/17/23/37 as in the SSE table.) Games open at `0-0-2` — the
STANDARD pickleball first-server exception (opening team serves with
its second server only), not an app artifact; the strings encode the
rule as expected, and the opening turn side-outs directly (type 16, no
type 23 first). The four type-45 rows before scoring start look like
challenge-budget init, all stamped OVERRULED_CHALLENGE_SUCCESS_STATUS —
don't count them as real challenges without checking timestamps.

**Implications**: empirical k (serve-rally win rate) per tour/gender/
format from ~a million archived rallies; per-player serve/return splits
historically; rally timestamps for pacing; court-side assignments;
challenge analytics; player arrival times. Live SSE remains necessary
ONLY for real-time win-prob. Backfill politeness math: ~36k doubles
games (fewer unique matches) + 26k singles at ~1 req/s ≈ a long
weekend of nightly chunks on the droplet; completed-match logs are
immutable → cache forever, harvest incrementally.

Shot-level data does NOT exist anywhere in this stack — the referee app
logs outcomes and administration, not strokes. Rally-end shot/reason is
likewise absent (no fault-type rows observed in a full game; the rally
either ends in a point or a serve change). Ceiling confirmed: for shot
detail, the only route is the Phase-6 vision pipeline on broadcasts.

**This confirms side-out scoring in 2026 MLP pro games** (score moves only
on `PointLog`, serve rotates 1→2→side-out) — matching the win-prob DP's
4-serve-state design with the algebraic no-score cycle.

**What rally logs unlock** (none of this is visible to Tier-1 polling):
- Empirical serve-rally win rate k — the DP's streakiness input, currently
  ASSUMED 0.35–0.45. Estimator: `#(type 14) / #(types 14+16+23)`. The tiny
  first sample gave 1/5; a weekend of matches gives thousands of rallies,
  splittable per league / per player.
- Per-PLAYER serve/return splits (`server_uuid`/`receiver_uuid` on every
  rally) — a new modeling axis no scraped box score contains.
- Per-rally timestamps (`date_created.seconds`) → rally cadence, momentum,
  broadcast-sync for annotated win-prob charts.
- Full point-by-point reconstruction of covered matches — scorebug OCR
  (ROADMAP Tier 0) becomes unnecessary wherever RTE coverage exists.
