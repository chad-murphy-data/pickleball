-- Pickleball serve/return store (2026-07-21).
--
-- Grain: one row per player per match. Keeps rally-log-derived serve tallies
-- queryable from any session so serve/return/points questions never trigger a
-- re-harvest. Loaded by scraper/upload_supabase.py from the droplet (which holds
-- every raw referee log); return is reconstructed in the view below, not stored.
--
-- Applied to project nwgxyytowbluuykbdcfc. Recorded here for reproducibility;
-- this repo does not run a Supabase migration tool, so apply by hand / MCP.

create table if not exists public.pb_match_player_serve (
  match_id      text not null,
  discipline    text not null,
  tour          text not null,
  match_date    date,
  player_uuid   text not null,
  side          int  not null,          -- 0/1, the two doubles sides
  serve_rallies int  not null,
  serve_wins    int  not null,
  primary key (match_id, player_uuid)
);
create index if not exists pb_mps_player_idx on public.pb_match_player_serve (player_uuid);
create index if not exists pb_mps_date_idx   on public.pb_match_player_serve (match_date);
comment on table public.pb_match_player_serve is
  'Pickleball serve rallies per player per match (rally referee logs). Return = opposing side serve losses, derived in SQL. Loaded by the droplet log collector.';

alter table public.pb_match_player_serve enable row level security;
drop policy if exists pb_mps_read on public.pb_match_player_serve;
create policy pb_mps_read on public.pb_match_player_serve for select using (true);

-- Per-player serve AND return at (player, tour, year) grain. Return rallies are
-- the opposing side's serve rallies, won on every side-out (team-attributed in
-- doubles: a per-player RATE, never summed across a team).
create or replace view public.pb_player_serve_return as
with side_tot as (
  select match_id, side, tour, match_date,
         sum(serve_rallies) sr, sum(serve_wins) sw
  from public.pb_match_player_serve
  group by match_id, side, tour, match_date
),
per as (
  select p.player_uuid, p.tour,
         extract(year from p.match_date)::int as yr,
         p.serve_rallies, p.serve_wins,
         o.sr as opp_sr, (o.sr - o.sw) as opp_ret_wins
  from public.pb_match_player_serve p
  join side_tot o
    on o.match_id = p.match_id and o.side <> p.side
)
select player_uuid, tour, yr,
       sum(serve_rallies)::int as serve_rallies,
       sum(serve_wins)::int    as serve_wins,
       sum(opp_sr)::int        as return_rallies,
       sum(opp_ret_wins)::int  as return_wins
from per
group by player_uuid, tour, yr;

-- Freshness/coverage stamp (serve_rows, max_match_date) set by the uploader.
create table if not exists public.pb_meta (
  key         text primary key,
  value       text,
  updated_at  timestamptz default now()
);
alter table public.pb_meta enable row level security;
drop policy if exists pb_meta_read on public.pb_meta;
create policy pb_meta_read on public.pb_meta for select using (true);
