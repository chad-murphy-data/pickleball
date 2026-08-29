#!/bin/bash
# Court-coverage pipeline: VOD file + match uuid -> committed coverage CSVs
# + local verification overlay.  Idempotent: finished stages are skipped,
# so re-running after an interruption continues where it stopped.
#
#   bash vision/coverage_pipeline.sh <vod.mp4> <match_uuid> <vod_id> \
#       [<event name>] [<date YYYY-MM-DD>] [<workdir>]
#
# e.g. the PPA Indoor Nationals mixed final (YouTube SQg2mHBPHC0):
#   bash vision/coverage_pipeline.sh ppa_mixed_final_0125.mp4 \
#       c4eb30d0-cfc5-490a-8d23-57ef7072730e ppa_indoor_mixed_final_2026 \
#       "PPA Indoor National Championships mixed final" 2026-01-25
#
# Stages: referee timeline + lineup (network) -> court fit -> scorebug
# scan -> flip-aligned windows -> main-camera mask -> rtmpose extraction
# at 10 fps (the long pole, ~2 h CPU for a 3-game match) -> coverage
# CSVs -> sampled verification overlay + spotcheck template.
# Intermediates land in <workdir> (default /tmp/coverage_<vod_id>);
# the overlay mp4 is broadcast imagery: LOCAL ONLY, never commit it.
set -u
VOD=$(readlink -f "${1:?vod file}")
MATCH=${2:?match uuid}
VODID=${3:?vod id}
EVENT=${4:-}
DATE=${5:-}
WORK=${6:-/tmp/coverage_$VODID}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ID8=${MATCH:0:8}
mkdir -p "$WORK"
cd "$REPO/vision"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

[ -f "$VOD" ] || { echo "VOD missing: $VOD"; exit 1; }

if [ ! -f "$REPO/data/vision/rally_timeline_$ID8.csv" ]; then
  log "stage 0a: referee timeline (BFF)"
  python3 rally_timeline.py --match "$MATCH" || exit 1
fi
if [ ! -f "$REPO/data/vision/lineup_$ID8.csv" ]; then
  log "stage 0b: lineup state machine"
  python3 lineup.py --match "$MATCH" || exit 1
fi

if [ ! -f "$REPO/data/vision/court_$VODID.json" ]; then
  log "stage 1: court homography"
  python3 court.py --video "$VOD" --out "$REPO/data/vision/court_$VODID.json" \
    --overlay "$WORK/court_overlay.png" --t0 300 --t1 3600 || exit 1
fi

if [ ! -f "$WORK/sb_diff.csv" ]; then
  log "stage 2: scorebug flip scan"
  python3 scorebug_windows.py --scan "$VOD" --diff "$WORK/sb_diff.csv" || exit 1
fi

if [ ! -f "$REPO/data/vision/coverage_windows_$VODID.csv" ]; then
  log "stage 3: rally windows (flip train x timeline)"
  python3 coverage_windows.py --diff "$WORK/sb_diff.csv" \
    --timeline "$REPO/data/vision/rally_timeline_$ID8.csv" \
    --out "$REPO/data/vision/coverage_windows_$VODID.csv" || exit 1
fi

if [ ! -f "$WORK/cam.csv" ]; then
  log "stage 4: main-camera mask"
  python3 coverage.py --scan-camera "$VOD" --cam-out "$WORK/cam.csv" || exit 1
fi

log "stage 5: pose extraction (rtmpose balanced, 10 fps, court pre-filter)"
python3 coverage_extract.py --video "$VOD" \
  --windows "$REPO/data/vision/coverage_windows_$VODID.csv" \
  --court "$REPO/data/vision/court_$VODID.json" \
  --backend rtmpose --fps 10 \
  --out-dir "$REPO/data/vision/pose_$VODID" || exit 1

log "stage 6: coverage metrics"
python3 coverage.py --run --pose-dir "$REPO/data/vision/pose_$VODID" \
  --court "$REPO/data/vision/court_$VODID.json" \
  --windows "$REPO/data/vision/coverage_windows_$VODID.csv" \
  --lineup "$REPO/data/vision/lineup_$ID8.csv" \
  --cam "$WORK/cam.csv" \
  --spotcheck "$REPO/data/vision/coverage_spotcheck_$VODID.csv" \
  --vod "$VODID" --event "$EVENT" --date "$DATE" --match-id "$MATCH" || exit 1

log "stage 7: verification overlay (watch this FIRST; local only)"
python3 coverage_overlay.py --video "$VOD" \
  --pose-dir "$REPO/data/vision/pose_$VODID" \
  --court "$REPO/data/vision/court_$VODID.json" \
  --windows "$REPO/data/vision/coverage_windows_$VODID.csv" \
  --lineup "$REPO/data/vision/lineup_$ID8.csv" \
  --cam "$WORK/cam.csv" --vod "$VODID" \
  --sample 10 --out "$WORK/overlay_${VODID}_sample.mp4" || exit 1

log "PIPELINE COMPLETE"
echo "  watch:   $WORK/overlay_${VODID}_sample.mp4"
echo "  fill:    data/vision/coverage_spotcheck_$VODID.csv (then re-run stage 6)"
echo "  commit:  data/coverage_players.csv data/coverage_events.csv"
echo "           data/vision/{coverage_windows,court}_$VODID.* rally_timeline/lineup CSVs"
