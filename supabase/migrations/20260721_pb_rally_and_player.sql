-- Per-rally fact table + player dimension (2026-07-21) — the finest grain in
-- this stack, so the warehouse is self-sufficient for ad-hoc rally mining.
-- Loaded by scraper/upload_supabase.py (H.rally_events, validated to reproduce
-- tally()'s serve counts exactly and its win attribution to within ~0.02%).
--
-- Applied to project nwgxyytowbluuykbdcfc; recorded here for reproducibility.

create table if not exists public.pb_rally (
  match_id       text not null,
  discipline     text not null,
  tour           text not null,
  match_date     date,
  game_number    int  not null,
  rally_number   int  not null,          -- sequential within the game
  server_uuid    text,
  receiver_uuid  text,
  server_side    int,                    -- 0/1, null if server off-roster
  server_number  int,                    -- 1 or 2 (side-out), from the log
  outcome        text not null,          -- point | sideout | second
  won            int  not null,          -- 1 if the serving team won the rally
  server_score   int,                    -- running score at START of rally
  receiver_score int,
  primary key (match_id, game_number, rally_number)
);
create index if not exists pb_rally_server_idx   on public.pb_rally (server_uuid);
create index if not exists pb_rally_receiver_idx on public.pb_rally (receiver_uuid);
create index if not exists pb_rally_score_idx    on public.pb_rally (server_score, receiver_score);
create index if not exists pb_rally_date_idx     on public.pb_rally (match_date);
comment on table public.pb_rally is
  'One row per rally from referee logs. Score-state, streaks, receiver splits. Win attribution matches tally() to within ~0.02% (rare multi-point rewinds).';
alter table public.pb_rally enable row level security;
drop policy if exists pb_rally_read on public.pb_rally;
create policy pb_rally_read on public.pb_rally for select using (true);

-- Player dimension so queries read in names, not UUIDs (join on player_uuid).
create table if not exists public.pb_player (
  player_uuid text primary key,
  full_name   text,
  gender      text
);
alter table public.pb_player enable row level security;
drop policy if exists pb_player_read on public.pb_player;
create policy pb_player_read on public.pb_player for select using (true);
