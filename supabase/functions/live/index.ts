// GET /functions/v1/live — compact live-state snapshot of today's MLP
// matchups and PPA pro-doubles matches from pickleball.com's open BFF.
//
// Deno twin of netlify/functions/live.mjs (the alternate backend) — keep the
// two in sync; the page only needs ONE of them deployed. Full protocol notes
// live in the .mjs header and recon.md. Politeness: a 15 s in-memory memo +
// in-flight coalescing per isolate caps the upstream sweep rate no matter how
// many viewers poll; the page itself polls every 20 s.
//
// CORS is deliberate: the page lives on GitHub Pages (different origin).
// Auth: standard Supabase anon JWT (public by design, baked into the page).
import "jsr:@supabase/functions-js/edge-runtime.d.ts";

const BASE = "https://pickleball.com";
const UA = "pickles-live/1.0 (unofficial fan analytics; hobby project)";
const TZ = "America/Los_Angeles";

const ORD = ["One", "Two", "Three", "Four", "Five"];
const ORD_SNAKE = ["one", "two", "three", "four", "five"];

// deno-lint-ignore no-explicit-any
type J = any;

const disco: J = { date: null, ts: 0, mlp: [], ppa: [], nextDates: [] };
const doneMatchups = new Map<string, { ts: number; data: J }>();
const fmtCache = new Map<string, J>();
let sweepCache: { key: string | null; ts: number; body: J } = { key: null, ts: 0, body: null };
let inflight: Promise<J> | null = null;

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
    const doubles = ev.filter((e: J) => /doubles/i.test(e.title)).map((e: J) => e.uuid);
    if (doubles.length) ppa.push({ tid, title: t.Title, doubles });
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

const compactPpaMatch = (m: J, fmt: J) => ({
  uuid: lc(m.match_uuid),
  ev: m.event_title || "",
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

async function matchFormat(uuid: string) {
  if (fmtCache.has(uuid)) return fmtCache.get(uuid);
  let fmt: J = null;
  try {
    const body = await bff(`/api/v1/results/getResultMatchInfos?id=${uuid}`);
    const d = body.data, m = Array.isArray(d) ? d[0] : d;
    if (m && typeof m === "object") {
      fmt = {
        rally: !!m.is_rally_scoring,
        max: ORD_SNAKE.map((o) => m[`score_format_game_${o}_max`] || 0),
        winBy: m.score_format_game_one_win_by || 2,
        title: m.score_format_title || "",
      };
    }
  } catch { fmt = null; }
  fmtCache.set(uuid, fmt);
  return fmt;
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
  for (const { tid, title, doubles } of d.ppa) {
    try {
      const ms = (await bff(
        `/api/v1/results/getMatchInfosShort?eventIds=${doubles.join(",")}&date=${date}`)).data || [];
      let fmtBudget = 6;
      const rows: J[] = [];
      for (const m of ms) {
        let fmt = fmtCache.get(lc(m.match_uuid));
        if (fmt === undefined && fmtBudget > 0) { fmt = await matchFormat(lc(m.match_uuid)); fmtBudget--; }
        rows.push(compactPpaMatch(m, fmt));
      }
      out.ppa.push({ tid, title, matches: rows });
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

  if (sweepCache.key === date && Date.now() - sweepCache.ts < 15e3) {
    return json(sweepCache.body);
  }
  if (!inflight) {
    inflight = sweep(date)
      .then((body) => { sweepCache = { key: date, ts: Date.now(), body }; return body; })
      .finally(() => { inflight = null; });
  }
  try {
    return json(await inflight);
  } catch (e) {
    return json({ error: String((e as Error).message || e) }, 502);
  }
});
