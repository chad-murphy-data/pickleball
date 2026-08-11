# HANDOFF — live-listener launch & receipts

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

KEEP: the court homography and the lineup state machine (the latter is a
fact about referee logs, not vision — all four players' court positions at
every serve, no camera needed).

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

NEXT: the GPU weekend is now the whole job, not a polish step — a
TrackNet-class detector emitting one ball position per frame removes both
failure modes (75 candidate tracks per rally, 0.23 s fragments) and
everything downstream is already written. **Do not run the tale-of-two
analysis on the current detector**: at 1.1 contacts/rally the result would
be dominated by which rallies happened to track.

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
