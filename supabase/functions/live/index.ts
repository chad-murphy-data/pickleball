// GET /functions/v1/live — compact live-state snapshot of today's MLP
// matchups and PPA pro doubles AND pro singles matches from pickleball.com's
// open BFF (singles rows carry sg: true; one player per side).
//
// Deno twin of netlify/functions/live.mjs (the alternate backend) — keep the
// two in sync; the page only needs ONE of them deployed. Full protocol notes
// live in the .mjs header and recon.md. Politeness: a 15 s in-memory memo +
// in-flight coalescing per isolate — BUT isolates here rarely survive between
// polls (function_logs 2026-09-04: a boot and a shutdown around nearly every
// request), so every module-level cache is a bonus, never a guarantee: a
// sweep must come out complete and cheap from a cold start. The page polls
// every 20 s.
//
// CORS is deliberate: the page lives on GitHub Pages (different origin).
// Auth: standard Supabase anon JWT (public by design, baked into the page).
import "jsr:@supabase/functions-js/edge-runtime.d.ts";

const BASE = "https://pickleball.com";
const UA = "Mozilla/5.0 (compatible; pickles-live/1.0; +https://chad-murphy-data.github.io/pickleball/methods.html)";
const TZ = "America/Los_Angeles";

const ORD = ["One", "Two", "Three", "Four", "Five"];
const ORD_SNAKE = ["one", "two", "three", "four", "five"];

// deno-lint-ignore no-explicit-any
type J = any;

const disco: J = { date: null, ts: 0, mlp: [], ppa: [], nextDates: [] };
const doneMatchups = new Map<string, { ts: number; data: J }>();
const fmtCache = new Map<string, J>();        // match uuid -> fmt (misses are not cached)
const fmtGroupCache = new Map<string, J>();   // fmtKey -> fmt (a bracket round's shared format)
let sweepCache: { key: string | null; ts: number; body: J } = { key: null, ts: 0, body: null };
let inflight: Promise<J> | null = null;

// Upstream-block circuit breaker (2026-08-28: pickleball.com's CloudFront
// WAF briefly 403'd our UA). After BREAKER_TRIP consecutive sweeps whose
// only outcome is upstream 403s, stop touching upstream for BREAKER_HOLD
// and serve a "paused" payload instead — retry-spamming a WAF that has
// made up its mind is pointless and impolite. State is in-memory, so an
// isolate recycle re-probes naturally (a fresh isolate makes at most
// BREAKER_TRIP attempts before re-tripping), which doubles as recovery
// detection well before the full hold elapses.
const BREAKER = { fails: 0, until: 0 };
const BREAKER_TRIP = 5;
const BREAKER_HOLD = 24 * 3600e3;

function upstreamBlocked(out: J, err: unknown): boolean {
  if (err) return /\b403\b/.test(String((err as Error)?.message || err));
  const errs: string[] = out?.errors || [];
  return errs.length > 0 && out.mlp.length + out.ppa.length === 0 &&
    errs.every((s) => /\b403\b/.test(s));
}

function breakerNote(blocked: boolean) {
  if (!blocked) { BREAKER.fails = 0; return; }
  if (++BREAKER.fails >= BREAKER_TRIP) {
    BREAKER.until = Date.now() + BREAKER_HOLD;
    BREAKER.fails = 0;
  }
}

const pausedBody = (date: string) => ({
  paused: true,
  until: new Date(BREAKER.until).toISOString(),
  reason: "data source is declining automated requests (HTTP 403); polling paused",
  date, mlp: [], ppa: [], next: [], errors: [],
});

async function bff(path: string): Promise<J> {
  const r = await fetch(BASE + path, { headers: { "User-Agent": UA, Accept: "application/json" } });
  if (!r.ok) throw new Error(`${r.status} for ${path}`);
  return r.json();
}

const localDate = (d = new Date()) =>
  new Intl.DateTimeFormat("en-CA", { timeZone: TZ }).format(d);

const isMlpLeague = (tl: J) =>
  tl.organizationSlug === "major-league-pickleball" && !/junior/i.test(tl.title || "");

function isPpaTournament(t: J) {
  const title = t.Title || t.title || "";
  const email = (t.RegistrationContactEmail || t.registrationContactEmail || "").toLowerCase();
  if (/australia|asia|college/i.test(title)) return false;
  return email.includes("ppatour.com") || /\bPPA\b/.test(title);
}

async function discover(date: string) {
  if (disco.date === date && Date.now() - disco.ts < 10 * 60e3) return disco;
  const mlp: J[] = [];
  const tls = (await bff(`/api/v2/results/getTeamLeaguesResultsOnDate?date=${date}`)).data || [];
  for (const tl of tls) {
    if (!isMlpLeague(tl)) continue;
    for (const div of tl.divisions || []) mlp.push({ tl, div });
  }
  const ppa: J[] = [];
  const ts = (await bff(`/api/v1/results/getTournamentsOnDate?date=${date}`)).data || [];
  for (const t of ts) {
    if (!isPpaTournament(t)) continue;
    const tid = t.TournamentID;
    let groups = await bff(
      `/api/v1/results/getListActiveEventsFlatGroup?tournamentId=${tid}&date=${date}`);
    groups = Array.isArray(groups) ? groups : groups.data || [];
    const pro = groups.filter((g: J) =>
      /pro/i.test(g.group_title) && !/senior|junior/i.test(g.group_title));
    if (!pro.length) continue;
    const ev = (await bff(
      "/api/v1/results/getTournamentEventsShort" +
      `?tournamentId=${tid}&formatId=${pro[0].format_id}` +
      `&playerGroupId=${pro[0].player_group_id}` +
      `&bracketLevelId=${pro[0].bracket_level_id}&date=${date}`)).data || [];
    // pro doubles + pro singles; the singles flag rides on the EVENT uuid
    // (event titles on the short match payload are the same strings, but
    // discovery is the one place that has the list, so decide it here)
    const events = ev.filter((e: J) => /doubles|singles/i.test(e.title)).map((e: J) => e.uuid);
    const singles = ev.filter((e: J) => /singles/i.test(e.title)).map((e: J) => lc(e.uuid));
    if (events.length) ppa.push({ tid, title: t.Title, events, singles });
  }
  let nextDates: string[] = [];
  if (!mlp.length && !ppa.length) {
    for (let i = 1; i <= 3 && !nextDates.length; i++) {
      const dd = localDate(new Date(Date.now() + i * 864e5));
      const t2 = (await bff(`/api/v2/results/getTeamLeaguesResultsOnDate?date=${dd}`)).data || [];
      if (t2.some(isMlpLeague)) nextDates.push(dd);
    }
  }
  Object.assign(disco, { date, ts: Date.now(), mlp, ppa, nextDates });
  return disco;
}

const lc = (u: string | null | undefined) => (u || "").toLowerCase();

function playerPair(m: J, side: string, camel: boolean) {
  const out: J[] = [];
  for (const pn of ["One", "Two"]) {
    const id = camel
      ? lc(m[`team${side}Player${pn}Uuid`])
      : lc(m[`team_${side.toLowerCase()}_player_${pn.toLowerCase()}_uuid`]);
    const name = camel
      ? m[`team${side}Player${pn}Name`] ||
        [m[`team${side}Player${pn}FirstName`], m[`team${side}Player${pn}LastName`]]
          .filter(Boolean).join(" ")
      : m[`team_${side.toLowerCase()}_player_${pn.toLowerCase()}_name`] ||
        [m[`team_${side.toLowerCase()}_player_${pn.toLowerCase()}_first_name`],
         m[`team_${side.toLowerCase()}_player_${pn.toLowerCase()}_last_name`]]
          .filter(Boolean).join(" ");
    if (id || name) out.push({ id, n: name || "?" });
  }
  return out;
}

function gameScores(m: J, camel: boolean, currentGame?: number) {
  const g: number[][] = [];
  for (let i = 0; i < 5; i++) {
    const s1 = camel ? m[`teamOneGame${ORD[i]}Score`] : m[`team_one_game_${ORD_SNAKE[i]}_score`];
    const s2 = camel ? m[`teamTwoGame${ORD[i]}Score`] : m[`team_two_game_${ORD_SNAKE[i]}_score`];
    g.push([s1 || 0, s2 || 0]);
  }
  let last = 0;
  g.forEach(([a, b], i) => { if (a || b) last = i; });
  if (currentGame) last = Math.max(last, currentGame - 1);
  return g.slice(0, last + 1);
}

const compactMlpMatch = (m: J) => ({
  uuid: lc(m.matchUuid),
  ab: m.matchAbbreviation || "",
  st: m.matchStatus,
  win: m.winner || 0,
  tb: !!m.isTieBreaker,
  ct: m.matchCompletedType ?? null,
  cg: m.currentGame || 1,
  g: gameScores(m, true, m.currentGame),
  svT: m.serverFromTeam || 0,
  svN: m.currentServingNumber || 0,
  t1: playerPair(m, "One", true),
  t2: playerPair(m, "Two", true),
});

const compactPpaMatch = (m: J, fmt: J, sg: boolean) => ({
  uuid: lc(m.match_uuid),
  ev: m.event_title || "",
  sg,                                  // singles: one player per side
  rd: m.round_title || m.round_text || "",
  st: m.match_status,
  win: m.winner || 0,
  ct: m.match_completed_type ?? null,
  g: gameScores(m, false),
  svT: m.server_from_team || 0,
  svN: m.current_serving_number || 0,
  t1: playerPair(m, "One", false),
  t2: playerPair(m, "Two", false),
  start: m.planned_start_date || m.match_planned_start || null,
  fmt: fmt || null,
});

async function matchupDetail(uuid: string, completed: boolean) {
  if (completed) {
    const hit = doneMatchups.get(uuid);
    if (hit && Date.now() - hit.ts < 10 * 60e3) return hit.data;
  }
  const data = (await bff(`/api/v2/results/getResultsMatchupData?matchupId=${uuid}`)).data || {};
  if (completed) doneMatchups.set(uuid, { ts: Date.now(), data });
  return data;
}

// ---- PPA score formats --------------------------------------------------
// getMatchInfosShort carries the best-of count but not the points target or
// the rally flag; those take getResultMatchInfos per match. Formats are set
// per bracket ROUND, though — every match in the same (event, bracket side,
// round) shares one — so one lookup per group covers the day, and a match's
// own score_format_game_best_out_of is the check that the group's format
// really applies to it (a mismatch gets its own lookup). This replaced a
// 6-per-sweep budget that, with isolates recycling per poll, only ever
// priced the first six matches of the day (the missing-PRE bug, 2026-09-04).
const FMT_MAX_LOOKUPS = 20;   // per sweep; a heavy Challenger day is ~15 groups
const FMT_CONCURRENCY = 4;

const fmtKey = (m: J) => m.event_uuid
  ? `${lc(m.event_uuid)}|${m.in_bracket_type || ""}|${lc(m.pool_id)}|${m.round_number ?? m.round_text ?? ""}`
  : `match:${lc(m.match_uuid)}`;

const fmtFits = (m: J, fmt: J) =>
  !m.score_format_game_best_out_of || !fmt.bestOf || m.score_format_game_best_out_of === fmt.bestOf;

async function matchFormat(uuid: string) {
  if (fmtCache.has(uuid)) return fmtCache.get(uuid);
  let fmt: J = null;
  try {
    const body = await bff(`/api/v1/results/getResultMatchInfos?id=${uuid}`);
    const d = body.data, m = Array.isArray(d) ? d[0] : d;
    if (m && typeof m === "object") {
      const max = ORD_SNAKE.map((o) => m[`score_format_game_${o}_max`] || 0);
      fmt = {
        rally: !!m.is_rally_scoring,
        max,
        winBy: m.score_format_game_one_win_by || 2,
        title: m.score_format_title || "",
        bestOf: m.score_format_game_best_out_of || max.filter((x: number) => x > 0).length,
      };
    }
  } catch { fmt = null; }
  if (fmt) fmtCache.set(uuid, fmt);   // a failed lookup is retried next sweep
  return fmt;
}

async function pooled<T>(items: T[], n: number, fn: (x: T) => Promise<unknown>) {
  let i = 0;
  const worker = async () => { while (i < items.length) await fn(items[i++]); };
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, worker));
}

// match uuid -> fmt | null, for every match in one short payload
async function resolveFormats(ms: J[]) {
  const out = new Map<string, J>();
  const groups = new Map<string, J[]>();
  for (const m of ms) {
    const u = lc(m.match_uuid), k = fmtKey(m);
    const own = fmtCache.get(u), grp = fmtGroupCache.get(k);
    if (own) out.set(u, own);
    else if (grp && fmtFits(m, grp)) out.set(u, grp);
    else { if (!groups.has(k)) groups.set(k, []); groups.get(k)!.push(m); }
  }
  let budget = FMT_MAX_LOOKUPS;
  const reps = [...groups.values()].map((g) => g[0]).slice(0, budget);
  budget -= reps.length;
  await pooled(reps, FMT_CONCURRENCY, (m) => matchFormat(lc(m.match_uuid)));
  const strays: J[] = [];
  for (const [k, members] of groups) {
    const fmt = fmtCache.get(lc(members[0].match_uuid)) || null;
    if (fmt) fmtGroupCache.set(k, fmt);
    for (const m of members) {
      const u = lc(m.match_uuid);
      if (!fmt) out.set(u, null);
      else if (m === members[0] || fmtFits(m, fmt)) out.set(u, fmt);
      else strays.push(m);
    }
  }
  await pooled(strays.slice(0, budget), FMT_CONCURRENCY, (m) => matchFormat(lc(m.match_uuid)));
  for (const m of strays) out.set(lc(m.match_uuid), fmtCache.get(lc(m.match_uuid)) || null);
  return out;
}

async function sweep(date: string) {
  const d = await discover(date);
  const out: J = {
    generated: new Date().toISOString(),
    date, tz: TZ, mlp: [], ppa: [], next: d.nextDates, errors: [],
  };
  for (const { tl, div } of d.mlp) {
    try {
      let q =
        `teamLeagueId=${tl.uuid}&organizationId=${tl.organizationUuid}` +
        `&divisionId=${div.divisionUuid}&seasonId=${div.seasonUuid}` +
        `&districtId=${div.districtUuid}&date=${date}`;
      if (div.matchupGroupUuid) q += `&matchupGroupUuid=${div.matchupGroupUuid}`;
      const mus = (await bff(`/api/v2/results/getTeamLeaguesMatchupsShortOnDivision?${q}`)).data || [];
      for (const mu of mus) {
        const status = mu.matchupStatus || "";
        if (status.startsWith("BYE")) continue;
        const completed = status === "COMPLETED_MATCHUP_STATUS";
        const detail = await matchupDetail(mu.uuid, completed);
        out.mlp.push({
          uuid: lc(mu.uuid),
          event: tl.title,
          t1: mu.teamOneTitle, t2: mu.teamTwoTitle,
          s1: mu.teamOneScore ?? 0, s2: mu.teamTwoScore ?? 0,
          status,
          start: mu.plannedStartDate || null,
          matches: (detail.matches || []).map(compactMlpMatch),
        });
      }
    } catch (e) { out.errors.push(`mlp: ${(e as Error).message}`); }
  }
  for (const { tid, title, events, singles } of d.ppa) {
    try {
      const ms = (await bff(
        `/api/v1/results/getMatchInfosShort?eventIds=${events.join(",")}&date=${date}`)).data || [];
      const fmts = await resolveFormats(ms);
      const isSg = (m: J) => singles.includes(lc(m.event_uuid)) || /singles/i.test(m.event_title || "");
      out.ppa.push({ tid, title, matches: ms.map((m: J) => compactPpaMatch(m, fmts.get(lc(m.match_uuid)), isSg(m))) });
    } catch (e) { out.errors.push(`ppa: ${(e as Error).message}`); }
  }
  return out;
}

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Access-Control-Allow-Headers": "authorization, apikey, content-type",
};

function json(body: J, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      ...CORS,
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "public, max-age=5",
    },
  });
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });
  const url = new URL(req.url);
  let date = url.searchParams.get("date") || localDate();
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) date = localDate();

  if (Date.now() < BREAKER.until) return json(pausedBody(date));
  if (sweepCache.key === date && Date.now() - sweepCache.ts < 15e3) {
    return json(sweepCache.body);
  }
  if (!inflight) {
    inflight = sweep(date)
      .then((body) => {
        breakerNote(upstreamBlocked(body, null));
        sweepCache = { key: date, ts: Date.now(), body };
        return body;
      })
      .catch((e) => { breakerNote(upstreamBlocked(null, e)); throw e; })
      .finally(() => { inflight = null; });
  }
  try {
    return json(await inflight);
  } catch (e) {
    if (Date.now() < BREAKER.until) return json(pausedBody(date));
    return json({ error: String((e as Error).message || e) }, 502);
  }
});
