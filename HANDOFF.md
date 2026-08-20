# HANDOFF — live-listener launch & receipts

## ► 2026-08-20 — COURT COVERAGE (PR #58, branch `claude/court-coverage-model-8rg94l`) — NEWEST, read before the 08-11 entry

Court occupancy per player from broadcast video, on the SOLVED layers
only (pose tracks + court homography + referee-log identity; no ball, no
contacts, no training). This does NOT re-open the vision program stopped
on 08-11 — it deliberately avoids the layer that failed. Canonical
technical record: `vision/coverage_spec.md` (read its last four sections
first, they are the corrections). Full narrative: PR #58 description.

**STATUS: code done, selftests green, waiting on THREE user
verifications. Numbers are provisional until those close.**

### Run it
```
bash vision/coverage_pipeline.sh <vod.mp4> <match_uuid> <vod_id>
python3 vision/coverage_heatmap.py --cache <npz> --out <svg>   # instant re-render
python3 vision/coverage_heatmap.py --cache <npz> --stack-report
```
Every module ships `--selftest`; all pass (heat map alone runs 41 checks).

### The one real run — 2026-01-25 PPA Indoor mixed final (`c4eb30d0`)
Bright/Patriquin vs Black/Alshon. 113/141 rallies score-verified ->
**63 by geometry, 90 after anchor-free identity**. End-map 63/63.
Outputs: `data/coverage_players.csv`, `coverage_dominance{,_rallies}.csv`,
`coverage_events.csv`, `data/vision/coverage_heatmap_ppa0125.svg`.

WHAT SURVIVES (space, never intent): width share — Alshon 0.552,
Patriquin 0.561, Bright 0.440, Black 0.448, the first DIRECT observation
of finding 11's coverage dial; off-court fraction Black 4.7% / Alshon
3.5% / Bright 3.0% / Patriquin 1.5%; observed 90% occupancy areas
Alshon 261 > Black 232 > Patriquin 204 > Bright 188 ft^2; and the depth
split — Patriquin owns the kitchen (28.8% of frames in the band, median
10.7 ft from net) while Alshon plays deep (14.8%, median 15.5 ft, 53.4%
beyond 14 ft). Depth results do not depend on any mirroring choice.

### THREE THINGS WERE RETRACTED OR WITHDRAWN — do not re-quote them
1. **"Deep poach"** is retracted entirely as a poaching measure. It
   counts CROSSINGS. At each player's deepest crossing the PARTNER is
   already wide (Black 8.9 ft, 86% of hers) — she held the middle while
   he went wide. The pre-registered initiation test FAILED to separate.
   Intent needs the ball; the ball is closed.
2. **"Alshon crosses 22.8%, double everyone else"** is withdrawn. That
   anchor mistook STACK UNWIND for crossing; robust anchor gives 12.8%
   vs Patriquin 11.0%.
3. **"Black 6.8% off court vs Alshon 3.3%"** was a stale 63-rally figure;
   committed data says 4.73% vs 3.51%.
Pattern worth internalising: all three came from the USER looking at
overlays or questioning a definition, not from the code.

### REPRODUCIBILITY HAZARD — read before touching anything
The identity ledgers (`identity_{swaps,track_map,anchorfree}_*.csv`) are
committed but keyed on POSE TRACK IDS, which live only in gitignored
`data/vision/pose_*/`. Re-extracting renumbers them. A mismatch now
fails loudly (guards in `carry_names` + `run()`), but the shipped numbers
need the exact extraction `pose_ppa0125c` (rtmpose-balanced, 10 fps, 113
rallies, ~2 h CPU) which is a SCRATCH dir that does not survive a fresh
container. Re-deriving it invalidates all three ledgers and moves the
numbers. Real fix (unbuilt): fingerprint each ledger against its
extraction.

### OPEN GATES — user actions, nothing to code
1. Check B first half: watch `overlay_anchorfree_sm.mp4` rallies
   3,5,11,19,21,23,25,26,27,28,40,41,48,52 — gates 90 vs 63 rallies.
2. Check A: `coverage_overlay_ppa0125_v4_sm.mp4` — gates the 63; game 3
   is the risky part (kit change, Gate A 46.7%).
3. Match watch: grade `poach_watchlist_af.csv` as CROSSINGS, not poaches.

### Traps a fresh session will otherwise re-learn
* Yield is capped by SOURCE VIDEO, not compute/model/labels. Failed
  rallies average 0.48 main-camera fraction at the serve vs 0.67 covered;
  the VOD is 49% main camera. 30 fps null, appearance rescued 1 of 16.
* The appearance descriptor is a TEAM-COLOUR detector. Cross-team pairs
  separate 97-100%, PARTNERS 59-88%. Partners wear matching kit by
  design. MLP will be worse (identical numbered uniforms) — jersey-number
  OCR is the generalizable channel, specced and unbuilt.
* Identity comes from aggregating across TIME (~75%/crop -> 96-100%/rally),
  not from a better same-instant classifier.
* End switches must be FITTED, not assumed per game (MLP switches at 6 in
  every game; PPA only in a decider).
* `coverage.run(write=False)` for any collect-only caller — diagnostic
  scripts with a placeholder `--vod` once appended 13 junk rows to the
  committed CSVs.
* Two crashes this session came from SELFTEST FIXTURES not matching
  production shape (a tuple game key read as int; a ledger naming an
  absent track). A synthetic input that does not match production shape
  is not a test of production.


## ► 2026-08-11 — VISION MVP BUILT, MEASURED, AND STOPPED (read this first)

**STOPPED.** Shot-level vision is capped — see the top of
`vision/mvp_findings.md`. Short version: the metrics worth having are
SEQUENCES (speed-up lost after one shot, who forced the error), which
recover as p^k. At the best measured per-frame rate (TrackNet BGR, 46%) a
3-shot chain lands 10% of the time, and a missed middle shot CORRUPTS the
chain rather than shortening it. 80% of 3-shot chains needs p ~ 0.93.
Nothing here is close, and the cheap fine-tune path is blocked because the
free auto-labels are biased toward slow shots (42% in the kitchen band vs a
14% base rate) — training on them would teach the model to miss exactly the
speed-ups the project cares about. Restarting needs a real hand-labelling
budget, not more tuning.

KEEP: the lineup state machine — and only that. It is a fact about referee
logs, not vision (all four players' court positions at every serve, no
camera needed), so it works with this whole directory switched off. The
court homography is a camera calibration with no consumer while vision is
capped; `court.py` being validated just means it would not need redoing.

**Canonical doc: `vision/mvp_findings.md`.** The full tracking system now
exists and every stage has a number on it:

| stage | module | status |
|---|---|---|
| court geometry | `vision/court.py` | **solved** — 0.06 ft median residual |
| player identity | `vision/lineup.py` | **solved, free** — 99.25% over 45,689 rallies |
| player detection | `vision/track_match.py` | adequate; broadcast crops deep players |
| ball detection | `vision/track_match.py` | **THE WALL** — ~1.1 contacts/rally vs ~12 played |
| contacts + attribution | `vision/shots.py` | built, physically motivated, starved of input |

The identity problem is gone: side-out doubles is a state machine, so the
referee log's two names (server, receiver) yield all four players' court
halves at every serve. That replaces the 57% appearance-based attribution.

The blocker is ball detection, and it is measured, not guessed: the
label-free side-alternation test (consecutive contacts must land on
opposite sides; chance 50%) returns 9%, and no sweep of area, strength,
track speed or reversal strictness moves it. Selecting tracks by
net-crossing lifts it to 33–37% but leaves ~1.1 contacts/rally. **The old
"50–65% recall" was never measured; the real figure is nearer 10%.**

~~NEXT: the GPU weekend is now the whole job~~ — SUPERSEDED by the STOPPED
block above, later the same day. A TrackNet-class detector was actually
tested on CPU (`vision/tracknet_probe.py`): it does roughly double the
colour detector (46% of frames vs ~21%, after fixing a BGR/RGB bug worth
2.7x on its own), and that is still nowhere near the p ~ 0.93 that sequence
metrics need. Do not buy GPU time on the strength of the doubling.
**Do not run the tale-of-two analysis on any detector built here.**

The target matchup is **four** meetings, not two — Dallas 5/25 (11-4),
Columbus 5/31 (11-3), Mid-Season final 7/12 (6-11, the 88% miss), Orlando
8/2 (11-5). All four have informative referee logs; timelines and lineups
are committed under `data/vision/`. VODs must be downloaded locally —
`*.googlevideo.com` is blocked by this environment's egress policy.

## ► 2026-08-10 — VISION THREAD (superseded by the block above)

Branch `claude/pickleball-vision-match-analysis-dow5ds` / PR #52. Canonical
doc for the whole vision effort = **`vision/README.md`** (verdict table,
derived-measure specs, run instructions). Read that first.

State: the POC is DONE and positive (ball trackable at 360p with zero ML;
scorebug flips give frame-exact log sync; audio demoted to timing polish —
full story in the README). `vision/track_full_vod.py` is smoke-tested and
frozen: one decode pass over the 80-min Chicago VOD harvesting seven
streams (ball candidates, density, motion, scorebug flips + 1/s crops,
player blobs, audio loudness).

**2026-08-11 UPDATE: the full-VOD run is DONE and the core question is
ANSWERED** — see `vision/interval_results.md` (fast mode at 0.15–0.25 s,
validated by duration-correlation +0.70, burstiness, and the blind gender
split). Crowd analysis in `vision/crowd_leverage.md` (leverage AND
allegiance both null). NEXT: attribution (side-of-net → player via the
log's server/receiver anchors), then the freeze-out final VOD (New Jersey
5s v STL 2026-07-12, vetted start-marked) for Waters ball-share.

When the zip arrives, the derivation stack (each layer validated against
the referee log before the next builds on it):
  1. tracks -> contacts -> the interval histogram (dinks vs speed-ups — THE
     core question)
  2. rally-end taxonomy (net/out/winner), audited against the log's rally
     winners
  3. four speed-up roles (offerer/initiator/finisher/victim) + the outcome
     ledger (punished-selection quadrant) — all specced in vision/README,
     all views over the contact sequence, no re-collection needed
  4. fun layer: crowd-roar vs leverage, rally->timestamp clip index,
     side heatmaps

Match context: Chicago Slice v Utah Black Diamonds 2026-07-25, matchup
timeline committed at data/vision/rally_timeline_matchup_20260725_*.csv
(193 rallies, all four games start-marked). Also merged into this branch:
findings 11 & 12 (gap exploitation null; singles surplus) + the t1-ordering
audit — those are complete, do not re-open.

---

# (stale below — 2026-07-16)

## ► ADDENDUM 2026-07-16 late PT — live page built; check it Friday morning

`site/live.html` (Pillar 5) ships on the `claude/live-match-progress-page-*`
branch: rally-by-rally win prob for MLP + PPA, backed by Supabase Edge
Functions (`live`/`logs` on project nwgxyytowbluuykbdcfc — deployed and
verified against day-1 data). First thing on Friday (MLP San Diego day 2 +
PPA Macon day 1, first ball ~10:00 PT): open the page during play and check
(a) LIVE cards update every ~20 s, (b) whether getListLogs serves rows
MID-match (verified only for completed matches so far — if not, charts use
the snapshot fallback until courts finish), (c) PPA Macon payloads carry
player uuids + fmt (single-15 Challenger vs Bo3-11) so PPA rows price.
Engine sync check: `node web/test_live_engine.mjs`.

*Written 2026-07-15 (Pacific) for the next thread. Canonical docs remain
CLAUDE.md (house rules), ROADMAP.md (build order), recon.md (data + SSE protocol).
This file is a dated status snapshot — if it's more than a week old, trust the
canonical docs over it.*

## Where we are (end of 2026-07-15 PT)

- **Tier-1 live poller is DEPLOYED and ARMED** on a DigitalOcean VPS (droplet
  `ubuntu-s-1vcpu-1gb-sfo2`, sfo2 region, root, repo at `~/pickleball` on `main`,
  venv `.venv`). User-level systemd timer `pickleball-live.timer` fires daily at
  **09:15 America/Los_Angeles (16:15 UTC)** → `deploy/run_poller.sh` → polls the BFF
  all day → commits & pushes `live/events-YYYYMMDD.jsonl` to `main`, then self-exits
  ~30 min after the last match. Verified end-to-end 2026-07-15: TZ pin works,
  discovery finds San Diego, push to `main` works (commit `1b9e614`), linger enabled
  so it survives logout.
- **Gold final graded** → `model/receipts.md` entry 1 (STL swept NJ 3–0; overall STL
  call HIT, match Brier v2 0.154; the 88%-NJ women's-doubles call MISSED — Waters lost
  both her lines). PR #4.
- **Tier-2 SSE decoded** → `scraper/sse_probe.py` ready, handshake verified. Real event
  payloads NOT yet captured (needs a live match — that's the weekend job below).

## ► FIRST thing to check tomorrow (Thu 2026-07-16 = MLP San Diego day 1)

Confirm the unattended poller actually captured real live data:

1. Look on `main` for a new **`live: 2026-07-16 events`** commit (pushed evening PT after
   play ends). This session can read it straight from the repo/GitHub — no VPS access needed.
2. Open `live/events-20260716.jsonl`. Once games start (~10:00 AM PDT / 17:00 UTC) it should
   have lines with **real player names and progressing scores**. NOTE: the file already holds
   **25 pre-game `[None, None] [[0,0]]` rows** from a setup test — harmless placeholders; real
   rows append after them.
3. If nothing new landed, something didn't fire — have the user SSH to the droplet and run:
   ```
   systemctl --user list-timers pickleball-live.timer     # did it fire / when next?
   journalctl --user -u pickleball-live -e                 # what happened
   loginctl show-user root --property=Linger               # must be Linger=yes
   ```
   Likely culprits: linger off, the fine-grained PAT expired (push auth), or the box slept/rebooted.
   Watch live any time: `journalctl --user -u pickleball-live -f`.

## ► Weekend task — Tier-2 SSE capture (Sat 7/18 or Sun 7/19, ATTENDED)

**UPDATE 2026-07-16: discovery is DONE.** Real payloads were captured live
during San Diego day 1 (session probe, ~8 min, `live/sse-20260716.jsonl`):
match-state events are full BFF-shaped match objects (serve state included);
`withLogs` adds `reflog_<uuid>` referee-log events — per-rally server/receiver
UUIDs, timestamps, typed logs. **2026 MLP pro games are side-out scoring**,
confirming the DP's 4-serve-state design. Full schemas + log_type enum:
recon.md "Tier-2 event shapes". The weekend job is now VOLUME, not discovery:
```
python scraper/sse_probe.py --duration 14400                 # broad: all matches, state stream
python scraper/sse_probe.py --with-logs --matches <uuid>     # rally logs, one court at a time
```
**SECOND UPDATE, same day: k does NOT need live capture.**
`/api/v1/results/getListLogs?id=<match_uuid>` (open BFF) serves the full
referee log for COMPLETED matches — coverage is event-dependent (recon.md).
Rally history is backfillable; `scraper/harvest_logs.py` is BUILT and
validated (56/58 logged matches score-reconcile exactly; early k:
MLP doubles 0.430, PPA doubles 0.439, PPA singles 0.538 — n≈3.5k
rallies). Run the ~9.5 h backfill on the droplet per deploy/README.md
"Backfill", then `--summarize` → commit data/match_rally_summary.csv +
data/player_serve_rallies.csv. The weekend LIVE capture is still worth an
attended hour for what the archive can't give: `matchup_<uuid>` event
shapes (fire on transitions only), the `X-Request-Tiebreaker-Matches`
DreamBreaker feed, PPA coverage check (Macon), and a live win-prob chart
rehearsal against real-time reflogs. Note `--with-logs` takes ONE match —
re-run per championship-court match, or test whether the server honors
multi-match withLogs despite the client never asking for it.

Weekend schedule (from the BFF): **Sat 7/18** MLP San Diego (10 matchups) **+** PPA Macon
Challenger (49 matches) — double-header; **Sun 7/19** PPA Macon finals (32). Next windows:
MLP Chicago 7/23–25, MLP Orlando 7/30–8/1.

## Loose ends (optional, non-blocking)

- **Set the repo default branch to `main`** on GitHub (Settings → Branches). It's currently an
  old `claude/*` branch — that's why the VPS clone first landed without `deploy/`. Fixing it
  makes future clones sane.
- The poller commits data straight to `main`. Fine for a hobby repo; if that's noisy, point the
  wrapper at a `live-data` branch (it pushes whatever branch is checked out).
- Harmless cruft on `main`: ~10 stray `.log` files + the 25 pre-game test rows in
  `events-20260716.jsonl`. Clean up anytime, or ignore.

## Don't re-derive

CLAUDE.md "Established findings" + "House rules" are load-bearing (UUIDs = identity, cross-gender
comparisons are likelihood-flat, score formats are data not assumptions, v2 is the real model).
Read them before touching the model or the data.
