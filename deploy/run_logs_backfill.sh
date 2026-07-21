#!/usr/bin/env bash
# Nightly referee-log harvest window. Started by pickleball-logs.timer
# (20:00 Pacific); harvests missing match logs politely until done or the
# runtime cap, then refreshes the committed summary tables and pushes.
#
# Self-limiting by design: the first night works through the archive
# (~30k matches ≈ 9.5 h); every night after that only new completed
# matches remain (a few minutes), so the cap is a safety rail, not a
# schedule. Interrupting is always safe — the cache resumes for free.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export TZ="${POLLER_TZ:-America/Los_Angeles}"

PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

# Cap ends the run well before the 09:15 live-poller timer and first ball.
"$PY" scraper/harvest_logs.py --max-hours "${LOGS_MAX_HOURS:-11}" \
      --interval "${LOGS_INTERVAL:-1.1}"

"$PY" scraper/harvest_logs.py --summarize

# Upsert per-match-per-player serve tallies into Supabase so serve/return
# questions are a SQL query, never a re-harvest. No-op unless SUPABASE_URL /
# SUPABASE_SERVICE_KEY are set in the droplet environment. Never blocks the
# git refresh below if the network hiccups.
"$PY" scraper/upload_supabase.py || echo "supabase upload skipped/failed (non-fatal)"

if [[ -n "$(git status --porcelain -- data/match_rally_summary.csv data/player_serve_rallies.csv)" ]]; then
    git add data/match_rally_summary.csv data/player_serve_rallies.csv
    git commit -m "rally logs: nightly summary refresh ($(date +%F))"
fi
if [[ -n "$(git log --oneline '@{u}..HEAD' 2>/dev/null)" ]]; then
    git pull --rebase --autostash
    git push origin HEAD
fi
