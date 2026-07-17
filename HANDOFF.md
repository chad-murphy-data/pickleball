# HANDOFF — live-listener launch & receipts

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
