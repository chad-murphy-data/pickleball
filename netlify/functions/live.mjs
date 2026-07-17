// GET /api/live — compact live-state snapshot of today's MLP matchups and
// PPA pro-doubles matches, straight from pickleball.com's open BFF.
//
// This is the browser-facing proxy for the PICKLES live page: the BFF and
// the rte SSE feed send no CORS headers, so a static page can't read them
// directly. This function does one polite sweep and the CDN fans it out —
// with s-maxage=15 the upstream sees at most ~4 sweeps/minute NO MATTER how
// many viewers are on the page (same posture as scraper/live_poller.py,
// which stays the archival ground truth on the droplet).
//
// In-memory (warm-isolate) caches keep a sweep down to a handful of calls:
//   discovery (leagues/tournaments/events for the day)  — 10 min
//   completed-matchup details                            — 10 min
//   per-match score formats (immutable)                  — forever
//   whole response                                       — 12 s + in-flight coalescing
//
// Query params: ?date=YYYY-MM-DD (optional, tour-local override for replay
// and testing; default = today in America/Los_Angeles, matching the poller).

const BASE = "https://pickleball.com";
const UA = "pickles-live/1.0 (unofficial fan analytics; hobby project)";
const TZ = "America/Los_Angeles";

const ORD = ["One", "Two", "Three", "Four", "Five"];
const ORD_SNAKE = ["one", "two", "three", "four", "five"];

// ---- module-level caches (persist across warm invocations) --------------
const disco = { date: null, ts: 0, mlp: [], ppa: [], nextDates: [] };
const doneMatchups = new Map();   // uuid -> {ts, data}
const fmtCache = new Map();       // match uuid -> fmt object|null
let sweepCache = { key: null, ts: 0, body: null };
let inflight = null;

async function bff(path) {
  const r = await fetch(BASE + path, {
    headers: { "User-Agent": UA, Accept: "application/json" },
  });
  if (!r.ok) throw new Error(`${r.status} for ${path}`);
  return r.json();
}

const localDate = (d = new Date()) =>
  new Intl.DateTimeFormat("en-CA", { timeZone: TZ }).format(d);

function isMlpLeague(tl) {
  return (
    tl.organizationSlug === "major-league-pickleball" &&
    !/junior/i.test(tl.title || "")
  );
}

// PPA filter, trimmed port of harvest.is_ppa_tournament: ppatour.com contact
// email or a standalone PPA in the title, minus the non-US franchises.
function isPpaTournament(t) {
  const title = t.Title || t.title || "";
  const email = (t.RegistrationContactEmail || t.registrationContactEmail || "").toLowerCase();
  if (/australia|asia|college/i.test(title)) return false;
  return email.includes("ppatour.com") || /\bPPA\b/.test(title);
}

async function discover(date) {
  if (disco.date === date && Date.now() - disco.ts < 10 * 60e3) return disco;
  const mlp = [];
  const tls = (await bff(`/api/v2/results/getTeamLeaguesResultsOnDate?date=${date}`)).data || [];
  for (const tl of tls) {
    if (!isMlpLeague(tl)) continue;
    for (const div of tl.divisions || []) mlp.push({ tl, div });
  }
  const ppa = [];
  const ts = (await bff(`/api/v1/results/getTournamentsOnDate?date=${date}`)).data || [];
  for (const t of ts) {
    if (!isPpaTournament(t)) continue;
    const tid = t.TournamentID;
    let groups = await bff(
      `/api/v1/results/getListActiveEventsFlatGroup?tournamentId=${tid}&date=${date}`);
    groups = Array.isArray(groups) ? groups : groups.data || [];
    const pro = groups.filter(
      (g) => /pro/i.test(g.group_title) && !/senior|junior/i.test(g.group_title));
    if (!pro.length) continue;
    const ev = (await bff(
      "/api/v1/results/getTournamentEventsShort" +
      `?tournamentId=${tid}&formatId=${pro[0].format_id}` +
      `&playerGroupId=${pro[0].player_group_id}` +
      `&bracketLevelId=${pro[0].bracket_level_id}&date=${date}`)).data || [];
    const doubles = ev.filter((e) => /doubles/i.test(e.title)).map((e) => e.uuid);
    if (doubles.length) ppa.push({ tid, title: t.Title, doubles });
  }
  // when the day is empty, peek ahead so the page can say when play resumes
  let nextDates = [];
  if (!mlp.length && !ppa.length) {
    for (let i = 1; i <= 3 && !nextDates.length; i++) {
      const d = new Date(Date.now() + i * 864e5);
      const dd = localDate(d);
      const t2 = (await bff(`/api/v2/results/getTeamLeaguesResultsOnDate?date=${dd}`)).data || [];
      if (t2.some(isMlpLeague)) nextDates.push(dd);
    }
  }
  Object.assign(disco, { date, ts: Date.now(), mlp, ppa, nextDates });
  return disco;
}

const lc = (u) => (u || "").toLowerCase();
const playerPair = (m, side, camel) => {
  const out = [];
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
};

function gameScores(m, camel, currentGame) {
  const g = [];
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

function compactMlpMatch(m) {
  return {
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
  };
}

function compactPpaMatch(m, fmt) {
  return {
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
  };
}

async function matchupDetail(uuid, completed) {
  if (completed) {
    const hit = doneMatchups.get(uuid);
    if (hit && Date.now() - hit.ts < 10 * 60e3) return hit.data;
  }
  const data = (await bff(`/api/v2/results/getResultsMatchupData?matchupId=${uuid}`)).data || {};
  if (completed) doneMatchups.set(uuid, { ts: Date.now(), data });
  return data;
}

async function matchFormat(uuid) {
  if (fmtCache.has(uuid)) return fmtCache.get(uuid);
  let fmt = null;
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
      };
    }
  } catch { fmt = null; }
  fmtCache.set(uuid, fmt);
  return fmt;
}

async function sweep(date) {
  const d = await discover(date);
  const out = {
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
    } catch (e) { out.errors.push(`mlp: ${e.message}`); }
  }

  for (const { tid, title, doubles } of d.ppa) {
    try {
      const ms = (await bff(
        `/api/v1/results/getMatchInfosShort?eventIds=${doubles.join(",")}&date=${date}`)).data || [];
      let fmtBudget = 6;    // spread format lookups across sweeps
      const rows = [];
      for (const m of ms) {
        let fmt = fmtCache.get(lc(m.match_uuid));
        if (fmt === undefined && fmtBudget > 0) { fmt = await matchFormat(lc(m.match_uuid)); fmtBudget--; }
        rows.push(compactPpaMatch(m, fmt));
      }
      out.ppa.push({ tid, title, matches: rows });
    } catch (e) { out.errors.push(`ppa: ${e.message}`); }
  }
  return out;
}

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

export default async function handler(req) {
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });
  const url = new URL(req.url);
  let date = url.searchParams.get("date") || localDate();
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) date = localDate();

  if (sweepCache.key === date && Date.now() - sweepCache.ts < 12e3) {
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
    return json({ error: String(e && e.message || e) }, 502);
  }
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      ...CORS,
      "Content-Type": "application/json; charset=utf-8",
      // one upstream sweep per ~15 s across ALL viewers; stale served while
      // revalidating so the page never blocks on the BFF
      "Cache-Control": "public, max-age=5",
      "Netlify-CDN-Cache-Control": "public, s-maxage=15, stale-while-revalidate=90",
    },
  });
}

export const config = { path: "/api/live" };
