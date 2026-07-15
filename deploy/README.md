# Deploying the Tier-1 live poller on a VPS

Unattended live-score capture during MLP/PPA event days. The timer fires
every morning at 09:15 Pacific; on non-event days the poller exits within
seconds (two API calls), so it's safe to leave running year-round.

## Setup (~15 min on any $5 box: Hetzner/DO/Linode/Vultr, Ubuntu 24.04+)

```bash
# 1. Get the repo onto the box. If it's private, use a fine-grained PAT
#    (contents: read/write, this repo only) or an SSH deploy key with
#    write access — the poller pushes its JSONL back at end of day.
git clone https://github.com/chad-murphy-data/pickleball.git
cd pickleball

# 2. Give the box a git identity for the nightly data commits.
git config user.name  "live-poller"
git config user.email "you@example.com"

# 3. Install venv + httpx + the systemd user units, enable the timer.
./deploy/install.sh

# 4. Smoke test (on a non-event day this prints "0 targets ... exiting").
.venv/bin/python scraper/live_poller.py --once
```

## What runs, when

- `pickleball-live.timer` → daily 09:15 America/Los_Angeles (45 min before
  the usual 10:00 first ball). `Persistent=true` catches up after reboots.
- `pickleball-live.service` → `deploy/run_poller.sh`: runs the poller until
  it self-exits (~30 min after the day's last match), then commits and
  pushes `live/events-YYYYMMDD.jsonl`.
- Crash mid-day → systemd restarts it 60 s later; the poller re-discovers
  and re-emits current state once (downstream dedupes on match_uuid+scores).
- The wrapper pins `TZ=America/Los_Angeles` for the poller process, so the
  box's own timezone doesn't matter (UTC default is fine).

## Knobs (systemd drop-in or environment)

- `POLLER_INTERVAL` — seconds between sweeps, default 25. Never below 15
  (the poller refuses; politeness rule).
- `POLLER_TZ` — event-local timezone for "today", default America/Los_Angeles.

## Ops

```bash
systemctl --user list-timers pickleball-live.timer   # next firing
journalctl --user -u pickleball-live -e              # today's log
systemctl --user start pickleball-live.service       # manual start now
```

Upcoming windows this summer (see ROADMAP Phase 1): MLP San Diego
Jul 16–19, PPA Macon Challenger Jul 17–19, MLP Chicago Jul 23–25,
Amway MLP Orlando Jul 30–Aug 1.
