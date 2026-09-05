// Offline check of the live proxy's PPA score-format resolution
// (netlify/functions/live.mjs; the Deno twin in supabase/functions/live/
// index.ts carries the same logic — keep them in sync). Run:
//   node web/test_live_proxy.mjs
// fetch() is mocked with a synthetic tournament day, so no network.
//
// What it pins down (the 2026-09-04 missing-PRE bug): a COLD sweep must
// resolve a format for EVERY match with one getResultMatchInfos call per
// bracket round; a best-of mismatch inside a round gets its own lookup; a
// failed lookup is retried on the next sweep instead of being cached as
// null; and the per-sweep ceiling carries the remainder over.
import assert from "node:assert/strict";

const FMT_MAX_LOOKUPS = 20;   // mirrors the constant in live.mjs

const T = { TournamentID: "t-1", Title: "PPA Tour: Test Open", RegistrationContactEmail: "ops@ppatour.com" };
const EVENTS = [
  { uuid: "e-md", title: "Mens Doubles Pro Main Draw" },
  { uuid: "e-wd", title: "Womens Doubles Pro Main Draw" },
  { uuid: "e-ms", title: "Mens Singles Pro Main Draw" },   // singles: swept too, flagged sg
];
// score formats as getResultMatchInfos reports them
const FORMATS = {
  bo3_11: { title: "2 out of 3, All games to 11 win by 2", max: [11, 11, 11, 0, 0], bestOf: 3 },
  bo1_15: { title: "1 game to 15 win by 2", max: [15, 0, 0, 0, 0], bestOf: 1 },
  bo5_11: { title: "3 out of 5, All games to 11 win by 2", max: [11, 11, 11, 11, 11], bestOf: 5 },
};

let seq = 0;
function mk(ev, round, bracket, fmtId, n) {
  const rows = [];
  for (let i = 0; i < n; i++) {
    seq += 1;
    rows.push({
      match_uuid: `M-${seq}`,                 // the API mixes UUID case; the proxy lowercases
      event_uuid: ev, event_title: EVENTS.find((e) => e.uuid === ev).title,
      round_number: round, round_text: `Round ${round}`, in_bracket_type: bracket, pool_id: `pool-${ev}`,
      score_format_game_best_out_of: FORMATS[fmtId].bestOf,
      match_status: 1, winner: 0,
      team_one_player_one_uuid: `p${seq}a`, team_one_player_one_name: "A One",
      team_two_player_one_uuid: `p${seq}c`, team_two_player_one_name: "B One",
      _fmt: fmtId,                            // mock-side truth, ignored by the proxy
    });
    if (ev !== "e-ms") {                      // singles carries one player per side
      Object.assign(rows[rows.length - 1], {
        team_one_player_two_uuid: `p${seq}b`, team_one_player_two_name: "A Two",
        team_two_player_two_uuid: `p${seq}d`, team_two_player_two_name: "B Two",
      });
    }
  }
  return rows;
}

let DAY = [];
const lookups = new Map();      // match uuid (lower) -> getResultMatchInfos call count
const failOnce = new Set();     // uuids whose next lookup returns HTTP 500
globalThis.fetch = async (url) => {
  const u = new URL(url), p = u.pathname, q = u.searchParams;
  const ok = (body) => new Response(JSON.stringify(body), { status: 200 });
  if (p.endsWith("getTeamLeaguesResultsOnDate")) return ok({ data: [] });
  if (p.endsWith("getTournamentsOnDate")) return ok({ data: [T] });
  if (p.endsWith("getListActiveEventsFlatGroup")) {
    return ok([{ group_title: "Pro Events", format_id: 0, player_group_id: 0, bracket_level_id: 2 }]);
  }
  if (p.endsWith("getTournamentEventsShort")) return ok({ data: EVENTS });
  if (p.endsWith("getMatchInfosShort")) {
    assert.equal(q.get("eventIds"), "e-md,e-wd,e-ms", "doubles AND singles pro events are swept");
    return ok({ data: DAY });
  }
  if (p.endsWith("getResultMatchInfos")) {
    const id = q.get("id");
    assert.equal(id, id.toLowerCase(), "lookups use lowercased uuids");
    lookups.set(id, (lookups.get(id) || 0) + 1);
    if (failOnce.has(id)) { failOnce.delete(id); return new Response("boom", { status: 500 }); }
    const m = DAY.find((r) => r.match_uuid.toLowerCase() === id);
    assert.ok(m, `lookup for a match not on the board: ${id}`);
    const f = FORMATS[m._fmt];
    const rec = {
      match_uuid: m.match_uuid, score_format_title: f.title, score_format_game_one_win_by: 2,
      is_rally_scoring: false, score_format_game_best_out_of: f.bestOf,
    };
    ["one", "two", "three", "four", "five"].forEach((o, i) => { rec[`score_format_game_${o}_max`] = f.max[i]; });
    return ok({ data: [rec] });
  }
  throw new Error(`unexpected fetch ${url}`);
};

const { default: handler } = await import("../netlify/functions/live.mjs");

// each sweep uses a fresh date: the response memo and discovery are per date,
// the format caches are not — exactly the warm-instance case under test, so
// every scenario below uses bracket rounds no earlier scenario has cached
let day = 0;
async function sweep() {
  day += 1;
  const date = `2026-10-${String(day).padStart(2, "0")}`;
  const res = await handler(new Request(`https://x.test/api/live?date=${date}`));
  assert.equal(res.status, 200);
  const body = await res.json();
  assert.deepEqual(body.errors, [], "a failed format lookup must not fail the sweep");
  assert.equal(body.ppa.length, 1);
  return body.ppa[0].matches;
}
const withFmt = (rows) => rows.filter((r) => r.fmt).length;
const totalLookups = () => [...lookups.values()].reduce((a, b) => a + b, 0);

// ---- 1. a full cold day: one lookup per bracket round, strays on their own
DAY = [
  ...mk("e-md", 3, "W", "bo3_11", 8),          // men's R16
  ...mk("e-md", 4, "W", "bo3_11", 4),          // men's QF (same format, separate round -> its own lookup)
  ...mk("e-wd", 3, "W", "bo3_11", 8),          // women's R16 ...
  ...mk("e-wd", 3, "W", "bo5_11", 1),          // ... with one odd best-of-5 inside it (the stray)
  ...mk("e-wd", 3, "L", "bo1_15", 4),          // women's back draw: single game to 15
  ...mk("e-wd", 6, "W", "bo5_11", 1),          // gold medal match
  ...mk("e-ms", 2, "W", "bo3_11", 4),          // men's SINGLES R32 (one player per side)
];
let rows = await sweep();
assert.equal(rows.length, 30);
assert.equal(withFmt(rows), 30, "every match on a cold sweep carries a format");
assert.equal(lookups.size, 7, "6 bracket-round groups + 1 stray = 7 lookups");
assert.equal(totalLookups(), 7);
const sgRows = rows.filter((r) => r.sg);
assert.equal(sgRows.length, 4, "singles rows are flagged sg");
assert.ok(sgRows.every((r) => r.t1.length === 1 && r.t2.length === 1), "singles: one player per side");
assert.ok(rows.filter((r) => !r.sg).every((r) => r.t1.length === 2 && r.t2.length === 2));
for (const r of rows) {
  const truth = FORMATS[DAY.find((m) => m.match_uuid.toLowerCase() === r.uuid)._fmt];
  assert.deepEqual(r.fmt.max, truth.max, `format for ${r.uuid} (${r.rd})`);
  assert.equal(r.fmt.title, truth.title);
  assert.equal(r.fmt.bestOf, truth.bestOf);
  assert.equal(r.fmt.rally, false);
  assert.equal(r.fmt.winBy, 2);
}
console.log("ok  cold day: 30 matches (4 singles) priced with 7 lookups");

// ---- 2. a failed lookup leaves its round unpriced for ONE sweep, then heals
lookups.clear();
DAY = [...mk("e-md", 5, "W", "bo3_11", 4), ...mk("e-wd", 5, "W", "bo3_11", 2)];
const mdRep = DAY[0].match_uuid.toLowerCase(), wdRep = DAY[4].match_uuid.toLowerCase();
failOnce.add(mdRep);
rows = await sweep();
assert.equal(withFmt(rows), 2, "the round whose lookup failed is unpriced, the other is fine");
assert.ok(rows.slice(0, 4).every((r) => r.fmt === null));
assert.equal(lookups.get(mdRep), 1);
rows = await sweep();                          // warm caches, next poll
assert.equal(withFmt(rows), 6, "the failed lookup is retried, not cached as null");
assert.equal(lookups.get(mdRep), 2, "exactly one retry");
assert.equal(lookups.get(wdRep), 1, "a cached round is not looked up again");
assert.equal(totalLookups(), 3);
console.log("ok  failure: retried next sweep, cached rounds untouched");

// ---- 3. the per-sweep ceiling carries the remainder to the next sweep
lookups.clear();
DAY = [];
for (let r = 1; r <= FMT_MAX_LOOKUPS + 3; r++) DAY.push(...mk("e-md", r, "L", "bo3_11", 1));
rows = await sweep();
assert.equal(withFmt(rows), FMT_MAX_LOOKUPS, "ceiling respected on a cold sweep");
assert.equal(totalLookups(), FMT_MAX_LOOKUPS);
rows = await sweep();
assert.equal(withFmt(rows), FMT_MAX_LOOKUPS + 3, "the remainder fills in on the next sweep");
assert.equal(totalLookups(), FMT_MAX_LOOKUPS + 3, "nothing is looked up twice");
console.log("ok  ceiling: 20 per sweep, remainder next sweep");

console.log("live proxy format resolution: all checks passed");
