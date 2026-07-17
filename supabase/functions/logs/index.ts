// GET /functions/v1/logs?match=<uuid> — compact referee-log proxy
// (getListLogs). Deno twin of netlify/functions/logs.mjs — keep in sync.
// Gives the live page rally-by-rally backfill so mid-match joins still show
// the full win-probability curve.
import "jsr:@supabase/functions-js/edge-runtime.d.ts";

const BASE = "https://pickleball.com";
const UA = "pickles-live/1.0 (unofficial fan analytics; hobby project)";
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Access-Control-Allow-Headers": "authorization, apikey, content-type",
};

const TEAM_KEYS = [
  "timeout_log", "additional_timeout_log", "challenge_log",
  "video_challenge_log", "line_review_log",
];

// deno-lint-ignore no-explicit-any
type J = any;

function compact(row: J) {
  const out: J = {
    i: row.log_index ?? 0,
    t: row.log_type ?? 0,
    g: row.game_number ?? 1,
    s: row.start_score_current_game_string || "",
    e: row.end_score_current_game_string || "",
    sv: (row.server_uuid || "").toLowerCase(),
    n: row.server_index ?? 0,
    ts: row.date_created == null ? null
      : typeof row.date_created === "object" ? row.date_created.seconds ?? null
      : typeof row.date_created === "number" ? row.date_created
      : Math.floor(Date.parse(row.date_created) / 1000) || null,
  };
  for (const k of TEAM_KEYS) {
    if (row[k] && row[k].team_uuid) { out.team = row[k].team_uuid.toLowerCase(); return out; }
  }
  if (row.LogData && typeof row.LogData === "object") {
    const v = Object.values(row.LogData).find((x: J) => x && typeof x === "object" && x.team_uuid) as J;
    if (v) out.team = (v.team_uuid || "").toLowerCase();
  }
  return out;
}

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
  const match = (url.searchParams.get("match") || "").toLowerCase();
  if (!UUID_RE.test(match)) return json({ error: "match must be a uuid" }, 400);
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
    return json({ error: String((e as Error).message || e) }, 502);
  }
});
