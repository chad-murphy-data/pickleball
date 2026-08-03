-- Per-rally timing on pb_rally.
--
-- Populated by scraper/harvest_logs.py:rally_events(), which computes the gap
-- from the type-12 "rally underway" marker to the resolving entry INSIDE the
-- correction-aware walk -- so timing carries the same rally_number as the rest
-- of the row. Validated against the explicit point_log time_started/time_ended
-- payload: identical multiset on every point of a test match.
--
-- Nullable by design: rallies logged without a start marker keep the row and
-- carry NULL timing rather than being dropped (dropping them desynchronises
-- rally_number).
--
-- Run once, then the existing nightly deploy/run_logs_backfill.sh ->
-- scraper/upload_supabase.py path fills it with no further changes.

alter table public.pb_rally
  add column if not exists start_utc timestamptz,
  add column if not exists dur_s     integer;

comment on column public.pb_rally.start_utc is
  'Rally start (type-12 referee marker), UTC. NULL when no start marker was logged.';
comment on column public.pb_rally.dur_s is
  'Rally duration in seconds, start marker -> resolution. Referee tap resolution (~1s). Exclude rallies near a timeout/line review/penalty and clamp to a sane range (2-90s) before analysis.';

-- Useful for the score-state duration cuts in model/rally_duration.md
create index if not exists pb_rally_dur_idx
  on public.pb_rally (discipline, dur_s)
  where dur_s is not null;
