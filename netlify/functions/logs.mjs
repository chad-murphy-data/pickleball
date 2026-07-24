// GET /api/logs?match=<uuid> — compact referee-log proxy (getListLogs).
//
// Gives the live page its rally-by-rally backfill: every logged rally's
// start score + serve state, so a viewer joining mid-match still sees the
// full win-probability curve (same rows web/replay_winprob.py replays).
// Coverage is event-dependent — some courts aren't digitally refereed; the
// page falls back to poll snapshots when this returns nothing.
//
// Row compaction (~90% smaller than upstream):
//   i=log_index t=log_type g=game_number s/e=start/end "srv-rcv-num" score
//   sv=server_uuid rcv=receiver_uuid n=server_index ts=epoch seconds
//   team=timeout/challenge team

const BASE = "https://pickleball.com";
const UA = "Mozilla/5.0 (compatible; pickles-bot/1.0; +https://chad-murphy-data.github.io/pickleball/methods.html)";
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

const TEAM_KEYS = [
  "timeout_log", "additional_timeout_log", "challenge_log",
  "video_challenge_log", "line_review_log",
];

function compact(row) {
  const out = {
    i: row.log_index ?? 0,
    t: row.log_type ?? 0,
    g: row.game_number ?? 1,
    s: row.start_score_current_game_string || "",
    e: row.end_score_current_game_string || "",
    sv: (row.server_uuid || "").toLowerCase(),
    rcv: (row.receiver_uuid || "").toLowerCase(),
    n: row.server_index ?? 0,
    ts: row.date_created == null ? null
      : typeof row.date_created === "object" ? row.date_created.seconds ?? null
      : typeof row.date_created === "number" ? row.date_created
      : Math.floor(Date.parse(row.date_created) / 1000) || null,
  };
  // team attribution for timeout/challenge marks; archive rows carry
  // snake_case sub-objects at the top level, SSE rows nest under LogData
  for (const k of TEAM_KEYS) {
    if (row[k] && row[k].team_uuid) { out.team = row[k].team_uuid.toLowerCase(); return out; }
  }
  if (row.LogData && typeof row.LogData === "object") {
    const v = Object.values(row.LogData).find((x) => x && typeof x === "object" && x.team_uuid);
    if (v) out.team = (v.team_uuid || "").toLowerCase();
  }
  return out;
}

export default async function handler(req) {
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });
  const url = new URL(req.url);
  const match = (url.searchParams.get("match") || "").toLowerCase();
  if (!UUID_RE.test(match)) {
    return json({ error: "match must be a uuid" }, 400);
  }
  try {
    const r = await fetch(`${BASE}/api/v1/results/getListLogs?id=${match}`, {
      headers: { "User-Agent": UA, Accept: "application/json" },
    });
    if (r.status === 404) return json({ match, rows: null, note: "no log for this match" });
    if (!r.ok) return json({ error: `upstream ${r.status}` }, 502);
    const body = await r.json();
    const data = Array.isArray(body) ? body : body.data;
    const rows = Array.isArray(data) ? data.map(compact) : null;
    return json({ match, rows });
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
      "Cache-Control": "public, max-age=5",
      "Netlify-CDN-Cache-Control": "public, s-maxage=15, stale-while-revalidate=60",
    },
  });
}

export const config = { path: "/api/logs" };
