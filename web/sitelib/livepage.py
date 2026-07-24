"""live.html — live win-probability page (ROADMAP Pillar 5).

Static shell generated with the rest of the site; all live behavior is
client-side JS talking to the pickles-live Netlify functions (the BFF and
the SSE feed send no CORS headers, so a proxy is unavoidable — the CDN
caches every response ~15 s, so upstream load is viewer-independent).

Engine: web/sitelib/live_engine.js (inlined) — the validated JS twin of
race.py + winprob.py. Page numbers reproduce web/replay_winprob.py exactly:
per-game curves from the serve-aware DP anchored to the calibrated
pre-match probability, matchup track composed via matchup_prob, DreamBreaker
via the singles model of make_forecast.py.

Data: site/data/live_values.json (built here) — every v2 player's current
value/sd + singles value where fitted, keyed by uuid.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from . import style
from .race import GAMMA

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# The data proxy. Primary: Supabase Edge Functions (supabase/functions/,
# deployed via MCP). Alternate: the same logic as Netlify functions
# (netlify/functions/, base would be https://pickles-live.netlify.app/api
# with LIVE_API_KEY=""). Endpoints are ${base}/live and ${base}/logs.
API_BASE = os.environ.get(
    "LIVE_API_BASE", "https://nwgxyytowbluuykbdcfc.supabase.co/functions/v1")
# Supabase anon key — public by design (standard client-side pattern).
API_KEY = os.environ.get("LIVE_API_KEY", (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im53Z3h5"
    "eXRvd2JsdXV5a2JkY2ZjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE1MzE1ODYsImV4cCI6MjA4"
    "NzEwNzU4Nn0.ktyO_FYxFP5xwQB0TXucnPMjMQi0HAVKGSdC0miDi4w"))

K_DOUBLES = 0.43          # winprob.py — measured serve-rally win rate
K_DB_SINGLES = 0.42       # make_forecast.py DreamBreaker model
SINGLES_IMPUTE = (0.28, 1.14)
SINGLES_MIN_GAMES = 10
POLL_MS = 20_000          # ≥15 s floor, same politeness rule as the poller


def _load_singles():
    path = DATA / "singles_players.csv"
    if not path.exists():
        return {}
    out = {}
    for r in csv.DictReader(path.open()):
        if int(r["singles_games"]) >= SINGLES_MIN_GAMES:
            out[r["player_id"].lower()] = round(float(r["singles_value"]), 4)
    return out


def build_values_json(players, cal, updated, site_dir):
    """site/data/live_values.json: pid -> [name, gender, value, sd, singles, has_page]."""
    singles = _load_singles()
    recs = {}
    for p in players.values():
        recs[p.pid] = [p.name, p.gender or "?", round(p.value, 4), round(p.sd, 4),
                       singles.get(p.pid), 1 if (p.dynamic and p.stats) else 0]
    body = {
        "meta": {
            "gamma": GAMMA, "k": K_DOUBLES, "kDbSingles": K_DB_SINGLES,
            "singlesImpute": list(SINGLES_IMPUTE),
            "cal": {"a": cal["a"], "b": cal["b"], "eps": cal["eps"]},
            "updated": updated,
        },
        "players": recs,
    }
    out = site_dir / "data"
    out.mkdir(parents=True, exist_ok=True)
    (out / "live_values.json").write_text(json.dumps(body, separators=(",", ":")))
    return len(recs)


# ---------------------------------------------------------------- page JS
# Plain string (NOT an f-string): placeholder __CFG__ is replaced at build.
LIVE_JS = r"""
'use strict';
const CFG = __CFG__;
PKL.configure({ gamma: CFG.gamma, kDoubles: CFG.k, kDbSingles: CFG.kDbSingles,
                singlesImpute: CFG.singlesImpute, cal: CFG.cal, epsFloor: CFG.cal.eps });

const $app = document.getElementById('live-app');
const $asof = document.getElementById('live-asof');
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fpF = p => PKL.fp(PKL.displayFloor(p));          // floored display %
const RALLY = 12, MARK_TYPES = {18:'T',35:'T+',2:'C',37:'C',45:'C'};

let VALUES = null;                 // pid -> [name, gender, v, sd, singles, hasPage]
let nameIndex = null;              // lower name -> pid (unambiguous only)
let logsCache = new Map();         // match uuid -> {rows, done, fetchedAt}
let snapStore = {};                // match uuid -> [[ts,a,b,state]]  (no-log fallback)
let pollTimer = null, errBackoff = 0;

function val(pid) { const r = pid && VALUES[pid]; return r ? { n: r[0], g: r[1], v: r[2], s: r[3], sv: r[4], pg: r[5] } : null; }
function resolve(p) {              // {id, n} from the API -> value record
  if (p.id && VALUES[p.id]) return { pid: p.id, ...val(p.id), matched: 'id' };
  const hit = p.n && nameIndex[p.n.toLowerCase()];
  if (hit) return { pid: hit, ...val(hit), matched: 'name' };
  return { pid: null, n: p.n || '?', v: null, s: null, sv: null, pg: 0, matched: null };
}

// ---- pre-match pricing (mirrors make_forecast.price_game) -------------
function priceGame(pair1, pair2, T) {
  if (pair1.length !== 2 || pair2.length !== 2) return null;
  const a = pair1.map(resolve), b = pair2.map(resolve);
  if ([...a, ...b].some(p => p.v === null)) return null;
  const eta = PKL.teamEta(a[0].v, a[1].v, b[0].v, b[1].v);
  const p0 = PKL.calibrate(PKL.raceDist(PKL.sig(eta), T).pw);
  return { eta, p0, nameMatched: [...a, ...b].some(p => p.matched === 'name') };
}

function dbProb(mu) {
  const side = i => {
    const ids = new Set();
    for (const m of mu.matches) for (const p of (i === 1 ? m.t1 : m.t2)) {
      const r = resolve(p); if (r.pid) ids.add(r.pid);
    }
    return [...ids].map(pid => { const r = val(pid); return { sv: r.sv, v: r.v }; });
  };
  const r1 = side(1), r2 = side(2);
  if (!r1.length || !r2.length) return 0.5;
  return PKL.dbWinProb(r1, r2);
}

// ---- rally series (mirrors replay_winprob.match_series) ---------------
function skipped(m) { return m.st === 4 && !m.win && (m.ct === 6 || m.ct === 14); }
function sideSet(mu, team) {
  const ids = new Set();
  for (const m of mu.matches) for (const p of (team === 1 ? m.t1 : m.t2)) {
    const r = resolve(p); if (r.pid) ids.add(r.pid);
  }
  return ids;
}

function seriesFromLogs(rows, side1, dp) {
  const pts = [], marks = []; let n = 0;
  const sorted = [...rows].sort((x, y) => (x.i || 0) - (y.i || 0));
  for (const r of sorted) {
    if (MARK_TYPES[r.t]) { if (pts.length) marks.push([n, MARK_TYPES[r.t]]); continue; }
    if (r.t !== RALLY) continue;
    const parts = (r.s || '').split('-').map(Number);
    if (parts.length !== 3 || parts.some(isNaN)) continue;
    const [sSrv, sRcv, num] = parts;
    let a, b, state;
    if (side1.has(r.sv)) { a = sSrv; b = sRcv; state = num === 1 ? PKL.A1 : PKL.A2; }
    else { a = sRcv; b = sSrv; state = num === 1 ? PKL.B1 : PKL.B2; }
    n += 1;
    pts.push({ x: n, p: dp.p(a, b, state), a, b, state });
  }
  return { pts, marks };
}

function dbSeriesFromLogs(rows, side1, pRally) {
  const table = PKL.rallyRaceTable(pRally, 21);
  const pts = []; let n = 0;
  for (const r of [...rows].sort((x, y) => (x.i || 0) - (y.i || 0))) {
    if (r.t !== RALLY) continue;
    const parts = (r.s || '').split('-').map(Number);
    if (parts.length < 2 || isNaN(parts[0]) || isNaN(parts[1])) continue;
    const [sSrv, sRcv] = parts;
    const [a, b] = side1.has(r.sv) ? [sSrv, sRcv] : [sRcv, sSrv];
    n += 1;
    pts.push({ x: n, p: table.p(a, b), a, b });
  }
  return { pts, marks: [] };
}

function snapshotSeries(uuid, dp) {
  const snaps = snapStore[uuid] || [];
  return { pts: snaps.map((s, i) => ({ x: i + 1, p: dp.p(s[1], s[2], s[3]), a: s[1], b: s[2] })), marks: [], sampled: true };
}

function recordSnapshot(uuid, a, b, state) {
  const arr = snapStore[uuid] = snapStore[uuid] || [];
  const last = arr[arr.length - 1];
  if (last && last[1] === a && last[2] === b && last[3] === state) return;
  arr.push([Math.floor(Date.now() / 1000), a, b, state]);
  if (arr.length > 2500) arr.splice(0, arr.length - 2500);
  try { localStorage.setItem(snapKey(uuid), JSON.stringify(arr)); } catch (e) {}
}
const snapKey = uuid => `pklSnap:${CFG.today || ''}:${uuid}`;
function loadSnaps() {
  try {
    const keep = `pklSnap:${CFG.today || ''}:`, drop = [];
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (!k || !k.startsWith('pklSnap:')) continue;
      if (k.startsWith(keep)) snapStore[k.split(':')[2]] = JSON.parse(localStorage.getItem(k)) || [];
      else drop.push(k);
    }
    drop.forEach(k => localStorage.removeItem(k));
  } catch (e) {}
}

async function ensureLogs(uuid, inProgress) {
  const hit = logsCache.get(uuid);
  if (hit && (hit.done || Date.now() - hit.fetchedAt < CFG.poll - 2000)) return hit.rows;
  try {
    const r = await fetch(`${CFG.api}/logs?match=${uuid}`, { headers: CFG.headers });
    const b = await r.json();
    const rows = Array.isArray(b.rows) ? b.rows : null;
    logsCache.set(uuid, { rows, done: !inProgress && rows !== null, fetchedAt: Date.now() });
    return rows;
  } catch (e) {
    logsCache.set(uuid, { rows: null, done: false, fetchedAt: Date.now() });
    return null;
  }
}

// ---- charts (replay_winprob.svg_step, redesigned for legibility) -------
// Position is the message: the line lives between two labeled poles
// (top = team 1 wins, bottom = team 2 wins), the halves are tinted in the
// two side colors (CVD-validated green/blue pair), a badge at the line's
// end names the current favorite, and a crosshair readout gives the exact
// score + number at any rally. Color never carries the side alone.
let chartReg = new Map(), chartSeq = 0;
function svgStep(series, opts) {
  const { pts, marks } = series;
  opts = opts || {};
  const W = 760, H = opts.h || 150;
  if (!pts.length) return '<p class="note lp-nolog">no rally log on this court yet</p>';
  const id = `wp${++chartSeq}`;
  const L = 45, R = W - 15, TOP = 12, BOT = H - 20;
  const n = Math.max(pts[pts.length - 1].x, 1);
  const X = i => L + (R - L) * i / n, Y = p => TOP + (BOT - TOP) * (1 - p);
  const midY = Y(0.5);
  const disp = pts.map(q => [q.x, PKL.displayFloor(q.p)]);
  let d = `M ${X(disp[0][0]).toFixed(1)} ${Y(disp[0][1]).toFixed(1)} `;
  for (let i = 1; i < disp.length; i++) d += `H ${X(disp[i][0]).toFixed(1)} V ${Y(disp[i][1]).toFixed(1)} `;
  // side fills: the step area closed along the 50/50 line, clipped to each half
  const area = d + `L ${X(disp[disp.length - 1][0]).toFixed(1)} ${midY.toFixed(1)} L ${X(disp[0][0]).toFixed(1)} ${midY.toFixed(1)} Z`;
  const fills =
    `<clipPath id="${id}t"><rect x="0" y="0" width="${W}" height="${midY}"/></clipPath>` +
    `<clipPath id="${id}b"><rect x="0" y="${midY}" width="${W}" height="${H - midY}"/></clipPath>` +
    `<path d="${area}" fill="var(--wp1)" opacity=".16" clip-path="url(#${id}t)"/>` +
    `<path d="${area}" fill="var(--wp2)" opacity=".16" clip-path="url(#${id}b)"/>`;
  const grid = [0.1, 0.9].map(v =>
    `<line x1="${L}" y1="${Y(v)}" x2="${R}" y2="${Y(v)}" stroke="var(--grid)" stroke-width="0.8"/>` +
    `<text x="${L - 5}" y="${Y(v) + 3}" font-size="9" fill="var(--muted)" text-anchor="end">${v * 100}%</text>`).join('') +
    `<line x1="${L}" y1="${midY}" x2="${R}" y2="${midY}" stroke="var(--baseline)" stroke-width="1.6"/>` +
    `<text x="${L - 5}" y="${midY + 3}" font-size="9" fill="var(--muted)" text-anchor="end">50/50</text>`;
  // game dividers on the composed track (label every segment, rule between)
  const bounds = (opts.bounds || []).filter(b => b.x > 0).map(b =>
    `<line x1="${X(b.x).toFixed(1)}" y1="${TOP - 2}" x2="${X(b.x).toFixed(1)}" y2="${BOT}" stroke="var(--baseline)" stroke-width="1" stroke-dasharray="3 3"/>` +
    `<text x="${(X(b.x) + 4).toFixed(1)}" y="${BOT - 4}" font-size="8.5" fill="var(--muted)" font-family="Space Mono,monospace">${b.label}</text>`).join('');
  const ticks = (marks || []).map(([i, lab]) =>
    `<line x1="${X(i).toFixed(1)}" y1="${BOT + 2}" x2="${X(i).toFixed(1)}" y2="${BOT + 9}" stroke="var(--muted)" stroke-width="1.5"/>` +
    `<text x="${X(i).toFixed(1)}" y="${H - 2}" font-size="8" fill="var(--muted)" text-anchor="middle">${lab}</text>`).join('');
  // favorite badge pinned to the line's end (computed first so the pole
  // labels can dodge it)
  const t = opts.teams || null;
  const lastP = disp[disp.length - 1][1], fav1 = lastP >= 0.5;
  const favTxt = t ? `${esc(t[fav1 ? 0 : 1]).toUpperCase()} ${PKL.fp(fav1 ? lastP : 1 - lastP)}%` : `${PKL.fp(lastP)}%`;
  const ex = X(disp[disp.length - 1][0]), ey = Y(lastP);
  const bw = 8 + favTxt.length * 6.6;
  const bx = Math.min(ex + 6, R - bw), by = Math.max(TOP + 2, Math.min(ey - 8, BOT - 18));
  const badge =
    `<rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${bw.toFixed(0)}" height="15" fill="var(--surface)" stroke="var(--wp${fav1 ? 1 : 2})" stroke-width="1.5"/>` +
    `<text x="${(bx + bw / 2).toFixed(1)}" y="${(by + 11).toFixed(1)}" font-size="10" font-weight="700" fill="var(--ink)" text-anchor="middle" font-family="Space Mono,monospace">${favTxt}</text>`;
  // labeled poles: the answer to "who is winning" is the line's position.
  // Opaque backing, and each label flips to the empty end of the chart when
  // the curve (or the badge) sits on top of its default spot.
  const curveIn = (x0, x1, y0, y1) =>
    disp.some(([qx, qp]) => { const px = X(qx), py = Y(qp); return px >= x0 && px <= x1 && py >= y0 && py <= y1; });
  const rectsHit = (ax, aw, ay) => ax < bx + bw && ax + aw > bx && ay < by + 15 && ay + 14 > by;
  const poleChip = (y, cls, label) => {
    const w = 24 + label.length * 6.2;
    let x = L + 4;
    const busyLeft = curveIn(x - 2, x + w + 8, y - 2, y + 16) || rectsHit(x, w, y);
    if (busyLeft) {
      const xr = R - w - 4;
      if (!curveIn(xr - 8, R, y - 2, y + 16) && !rectsHit(xr, w, y)) x = xr;
    }
    return `<rect x="${x.toFixed(1)}" y="${y}" width="${w.toFixed(0)}" height="14" fill="var(--surface)"/>` +
      `<rect x="${(x + 4).toFixed(1)}" y="${y + 3.5}" width="7" height="7" fill="var(--${cls})"/>` +
      `<text x="${(x + 15).toFixed(1)}" y="${y + 10.5}" font-size="9.5" font-weight="700" fill="var(--ink2)" font-family="Space Mono,monospace">${label}</text>`;
  };
  const poles = t
    ? poleChip(TOP + 2, 'wp1', `▲ ${esc(t[0]).toUpperCase()} WINS`) +
      poleChip(BOT - 16, 'wp2', `▼ ${esc(t[1]).toUpperCase()} WINS`)
    : '';
  const live = opts.live
    ? `<circle cx="${ex.toFixed(1)}" cy="${ey.toFixed(1)}" r="3.4" fill="var(--ink)"><animate attributeName="opacity" values="1;.25;1" dur="1.6s" repeatCount="indefinite"/></circle>` : '';
  chartReg.set(id, { pts, teams: t, L, R, TOP, BOT, n, W, H });
  const hover =
    `<line data-cross="1" x1="0" y1="${TOP}" x2="0" y2="${BOT}" stroke="var(--ink2)" stroke-width="1" visibility="hidden"/>` +
    `<rect data-catch="1" x="${L}" y="${TOP}" width="${R - L}" height="${BOT - TOP}" fill="transparent"/>`;
  return `<div class="lp-chart" data-wp="${id}"><svg viewBox="0 0 ${W} ${H}" width="100%" role="img">${fills}${grid}${bounds}` +
    `<path d="${d}" fill="none" stroke="var(--ink)" stroke-width="2"/>${ticks}${poles}${badge}${live}${hover}</svg>` +
    `<div class="lp-wptip" hidden></div></div>` +
    (series.sampled ? '<p class="note lp-sampled">no rally log on this court — sampled every ~20 s from the scoreboard</p>' : '');
}

// crosshair + readout: nearest rally under the pointer
function wireCharts(root) {
  for (const box of root.querySelectorAll('[data-wp]')) {
    const reg = chartReg.get(box.dataset.wp);
    if (!reg || box.dataset.wired) continue;
    box.dataset.wired = '1';
    const svg = box.querySelector('svg'), tip = box.querySelector('.lp-wptip');
    const cross = svg.querySelector('[data-cross]');
    const show = (ev) => {
      const r = svg.getBoundingClientRect();
      const fx = (ev.clientX - r.left) / r.width * reg.W;
      const xi = Math.round((fx - reg.L) / (reg.R - reg.L) * reg.n);
      let best = reg.pts[0];
      for (const q of reg.pts) if (Math.abs(q.x - xi) < Math.abs(best.x - xi)) best = q;
      const px = reg.L + (reg.R - reg.L) * best.x / reg.n;
      cross.setAttribute('x1', px); cross.setAttribute('x2', px);
      cross.setAttribute('visibility', 'visible');
      const p = PKL.displayFloor(best.p), fav1 = p >= 0.5;
      const who = reg.teams ? `${esc(reg.teams[fav1 ? 0 : 1]).toUpperCase()} ${PKL.fp(fav1 ? p : 1 - p)}%` : `${PKL.fp(p)}%`;
      const score = best.a !== undefined ? ` · ${best.a}–${best.b}` : '';
      tip.innerHTML = `rally ${best.x}${score} · <strong>${who}</strong>`;
      tip.hidden = false;
      const tx = (px / reg.W) * r.width;
      tip.style.left = `${Math.min(Math.max(tx - 60, 0), r.width - 130)}px`;
    };
    box.addEventListener('pointermove', show);
    box.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); tip.hidden = true; });
  }
}

// ---- MLP matchup card ---------------------------------------------------
const SLOT_LABEL = m => esc(m.ab || '?');
const stateFromSnap = m => m.svT === 1 ? (m.svN === 1 ? PKL.A1 : PKL.A2) : (m.svN === 1 ? PKL.B1 : PKL.B2);
function playerLink(p) {
  const r = resolve(p);
  const name = esc(r.n || p.n || '?');
  return r.pid && r.pg ? `<a href="players/${r.pid}.html">${name}</a>` : name;
}
function pairNames(pair, side) {
  if (!pair.length) return '<span class="note">lineup TBD</span>';
  return pair.map(playerLink).join(' / ');
}
// current two players on court in a live DreamBreaker, from the latest rally
// log (server + receiver), ordered [team1 player, team2 player] to match the
// pairing cell. null if the log lacks a usable server/receiver.
function dbCurrentPair(rows, side1) {
  let last = null;
  for (const r of rows) {
    if (r.t === RALLY && r.sv && r.rcv && (!last || (r.i || 0) > (last.i || 0))) last = r;
  }
  if (!last) return null;
  const srv = { id: last.sv }, rcv = { id: last.rcv };
  const cur = side1.has(last.sv) ? [srv, rcv] : [rcv, srv];
  // only trust it if both UUIDs resolve to real names (else keep the static pair)
  if (cur.some(p => { const n = resolve(p).n; return !n || n === '?'; })) return null;
  return cur;
}

function gameRow(m, info, liveP, preText, dbKey) {
  const dead = m.tb && skipped(m);
  const score = dead ? '<span class="note">not played (dead game)</span>'
    : m.g.map(([a, b]) => `<span class="lp-score">${a}–${b}</span>`).join(' ');
  const serve = m.st === 2 && m.svT ? `<span class="lp-serve" title="serving, server #${m.svN}">${m.svT === 1 ? '◀' : '▶'} #${m.svN}</span>` : '';
  let prob = '';
  if (m.st === 2 && liveP !== null) prob = `<strong>${fpF(liveP)}%</strong>`;
  else if (m.st === 4 && m.win) prob = m.win === 1 ? '✓' : '✗';
  const pre = preText !== undefined ? preText
    : info ? `<span class="note">${fpF(info.p0)}%</span>`
    : (m.t1.length && !dead ? '<span class="note">unrated</span>' : '');
  const pairs = dead ? '<span class="note">—</span>'
    : `${pairNames(m.t1, 1)}<br>${pairNames(m.t2, 2)}`;
  const pairAttr = dbKey ? ` data-dbpair="${dbKey}"` : '';
  return `<tr class="${m.st === 2 ? 'lp-liverow' : ''}"><td class="lp-slot">${SLOT_LABEL(m)}</td>` +
    `<td${pairAttr}>${pairs}</td>` +
    `<td class="num">${score} ${serve}</td><td class="num">${prob}</td><td class="num">${pre}</td></tr>`;
}

function matchupCard(mu) {
  const games = mu.matches.filter(m => !m.tb);
  const db = mu.matches.find(m => m.tb && !skipped(m));
  const infos = games.map(m => priceGame(m.t1, m.t2, 11));
  const pDb = dbProb(mu);

  let w1 = 0, w2 = 0;
  for (const m of games) { if (m.st === 4 && m.win === 1) w1++; else if (m.st === 4 && m.win === 2) w2++; }
  const dbDecided = db && db.st === 4 && db.win;
  const isLive = mu.matches.some(m => m.st === 2);
  const isDone = mu.status === 'COMPLETED_MATCHUP_STATUS';

  // headline: P(team1 wins matchup) right now
  let head = null, headNote = '';
  const cur = mu.matches.find(m => m.st === 2 && !skipped(m));
  const futureP0 = (fromIdx) => {
    const fut = [];
    for (let i = fromIdx; i < games.length; i++) {
      const m = games[i];
      if (m.st === 4 || skipped(m)) continue;
      if (m === cur) continue;
      if (!infos[i]) return null;
      fut.push(infos[i].p0);
    }
    return fut;
  };
  if (isDone) head = mu.s1 > mu.s2 ? 1 : 0;
  else if (cur && !cur.tb) {
    const gi = games.indexOf(cur), info = infos[gi];
    const fut = futureP0(0);
    if (info && fut !== null) {
      const dp = PKL.ServeDP(PKL.etaAnchor(info.p0, CFG.k, 11), CFG.k, 11);
      const [a, b] = cur.g[0];
      const pCur = dp.p(a, b, stateFromSnap(cur));
      recordSnapshot(cur.uuid, a, b, stateFromSnap(cur));
      head = pCur * PKL.matchupProb(w1 + 1, w2, fut, pDb) + (1 - pCur) * PKL.matchupProb(w1, w2 + 1, fut, pDb);
    }
  } else if (cur && cur.tb) {
    const pRally = PKL.rallyPForTarget(pDb, 21);
    const [a, b] = cur.g[0];
    head = PKL.rallyRaceTable(pRally, 21).p(a, b);
  } else {
    const fut = futureP0(0);
    if (fut !== null) head = PKL.matchupProb(w1, w2, fut, pDb);
    else headNote = 'lineups pending';
  }

  const chip = isLive ? '<span class="lp-chip lp-live">LIVE</span>'
    : isDone ? '<span class="lp-chip">FINAL</span>'
    : `<span class="lp-chip lp-sched">${mu.start ? new Date(mu.start).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : 'scheduled'}</span>`;
  const headline = head === null
    ? `<span class="note">${headNote || 'unpriced'}</span>`
    : isDone
      ? `<strong>${mu.s1 > mu.s2 ? esc(mu.t1) : esc(mu.t2)} won ${Math.max(mu.s1, mu.s2)}–${Math.min(mu.s1, mu.s2)}</strong>`
      : `<strong class="lp-big"><span class="lp-sw ${head >= 0.5 ? 'lp-sw1' : 'lp-sw2'}"></span>${esc(teamShort(head >= 0.5 ? mu.t1 : mu.t2))} ${fpF(head >= 0.5 ? head : 1 - head)}%</strong>`;

  const dbPre = `<span class="note">${fpF(pDb)}%</span>`;
  const rows = games.map((m, i) => gameRow(m, infos[i], liveGameP(m, infos[i]))).join('') +
    (db ? gameRow(db, null, db.st === 2 ? PKL.rallyRaceTable(PKL.rallyPForTarget(pDb, 21), 21).p(db.g[0][0], db.g[0][1]) : null, dbPre, db.st === 2 ? db.uuid : null)
        : (mu.matches.some(m => m.tb) ? gameRow(mu.matches.find(m => m.tb), null, null, '') : ''));

  return `<div class="card lp-card" data-mu="${mu.uuid}">
    <div class="lp-head">${chip}<span class="lp-teams"><span class="lp-sw lp-sw1"></span>${esc(mu.t1)} <b>${mu.s1}</b>–<b>${mu.s2}</b> <span class="lp-sw lp-sw2"></span>${esc(mu.t2)}</span>${headline}</div>
    <div class="lp-track" data-track="${mu.uuid}"></div>
    <table class="lp-games"><tr><th></th><th>pairing</th><th>score</th><th>live</th><th>pre</th></tr>${rows}</table>
    <p class="note lp-dbnote">DreamBreaker rated ${fpF(pDb)}% for ${esc(teamShort(mu.t1))} (singles model — rough by design).</p>
  </div>`;
}

function liveGameP(m, info) {
  if (m.st !== 2 || m.tb || !info) return null;
  const dp = PKL.ServeDP(PKL.etaAnchor(info.p0, CFG.k, 11), CFG.k, 11);
  const [a, b] = m.g[0];
  return dp.p(a, b, stateFromSnap(m));
}

function teamShort(t) { return (t || '').split(' ').pop(); }

// async: fill the matchup track chart from rally logs (exact replay logic)
async function fillTrack(mu) {
  const el = document.querySelector(`[data-track="${mu.uuid}"]`);
  if (!el) return;
  const games = mu.matches.filter(m => !m.tb);
  const db = mu.matches.find(m => m.tb && !skipped(m));
  const infos = games.map(m => priceGame(m.t1, m.t2, 11));
  if (infos.some((x, i) => x === null && (games[i].st >= 2))) { el.innerHTML = ''; return; }
  const pDb = dbProb(mu);
  const side1 = sideSet(mu, 1);
  const seq = [];
  for (let i = 0; i < games.length; i++) {
    const m = games[i];
    if (skipped(m) || m.st < 2) continue;
    const rows = await ensureLogs(m.uuid, m.st === 2);
    const dp = PKL.ServeDP(PKL.etaAnchor(infos[i].p0, CFG.k, 11), CFG.k, 11);
    let s = rows && rows.some(r => r.t === RALLY) ? seriesFromLogs(rows, side1, dp) : snapshotSeries(m.uuid, dp);
    if (m.st === 2) {   // append the freshest scoreboard point
      const [a, b] = m.g[0], st = stateFromSnap(m);
      const last = s.pts[s.pts.length - 1];
      if (!last || last.a !== a || last.b !== b) s.pts.push({ x: (last ? last.x + 1 : 1), p: dp.p(a, b, st), a, b });
    }
    if (m.st === 4 && m.win) s.pts.push({ x: (s.pts.length ? s.pts[s.pts.length - 1].x + 1 : 1), p: m.win === 1 ? 1 : 0 });
    seq.push({ m, i, s, sampled: s.sampled });
  }
  if (db && db.st >= 2) {
    const pRally = PKL.rallyPForTarget(pDb, 21);
    const tbl = PKL.rallyRaceTable(pRally, 21);
    const rows = await ensureLogs(db.uuid, db.st === 2);
    if (db.st === 2 && rows) {   // update the pairing cell to who's on court now
      const cur = dbCurrentPair(rows, side1);
      const cell = document.querySelector(`[data-dbpair="${db.uuid}"]`);
      if (cur && cell) cell.innerHTML = `${playerLink(cur[0])}<br>${playerLink(cur[1])}`;
    }
    let s = rows && rows.some(r => r.t === RALLY)
      ? dbSeriesFromLogs(rows, side1, pRally)
      : snapshotSeries(db.uuid, { p: (a, b) => tbl.p(a, b) });
    if (db.st === 2) { const [a, b] = db.g[0]; const last = s.pts[s.pts.length - 1]; if (!last || last.a !== a || last.b !== b) s.pts.push({ x: (last ? last.x + 1 : 1), p: tbl.p(a, b), a, b }); }
    if (db.st === 4 && db.win) s.pts.push({ x: (s.pts.length ? s.pts[s.pts.length - 1].x + 1 : 1), p: db.win === 1 ? 1 : 0 });
    seq.push({ m: db, i: games.length, s, db: true });
  }
  if (!seq.length) { el.innerHTML = ''; return; }
  // compose the matchup-level track
  let w1 = 0, w2 = 0, x = 0;
  const track = { pts: [], marks: [] };
  const bounds = [];
  const teams = [teamShort(mu.t1), teamShort(mu.t2)];
  let sampledAny = false;
  for (const item of seq) {
    // the future AS OF this game: every later non-skipped game at its
    // pre-match price, regardless of whether it has since been decided
    // (replay_winprob semantics — w1/w2 carry the sequential outcomes)
    const fut = [];
    for (let j = item.i + 1; j < games.length; j++) {
      if (skipped(games[j])) continue;
      fut.push(infos[j] ? infos[j].p0 : 0.5);
    }
    bounds.push({ x, label: item.db ? 'DB' : SLOT_LABEL(item.m) });
    for (const q of item.s.pts) {
      const live = item.db ? q.p
        : q.p * PKL.matchupProb(w1 + 1, w2, fut, pDb) + (1 - q.p) * PKL.matchupProb(w1, w2 + 1, fut, pDb);
      track.pts.push({ x: x + q.x, p: live, a: q.a, b: q.b });
    }
    for (const [mi, lab] of item.s.marks || []) track.marks.push([x + mi, lab]);
    x += item.s.pts.length ? item.s.pts[item.s.pts.length - 1].x : 0;
    if (item.m.st === 4 && item.m.win === 1) w1++;
    else if (item.m.st === 4 && item.m.win === 2) w2++;
    sampledAny = sampledAny || item.sampled;
  }
  track.sampled = sampledAny;
  const anyLive = mu.matches.some(m => m.st === 2);
  let html = `<p class="note lp-tracklab">Matchup win probability, rally by rally — line toward a team = that team winning</p>` +
    svgStep(track, { h: 175, teams, live: anyLive, bounds });
  // per-game mini charts for games with real rally logs
  const minis = [];
  for (const item of seq) {
    if (!item.s.pts.length || item.s.sampled) continue;
    const label = item.db ? 'DB' : SLOT_LABEL(item.m);
    minis.push(`<div class="lp-mini"><span class="note">${label}</span>` +
      svgStep(item.s, { h: 96, teams, live: item.m.st === 2 }) + '</div>');
  }
  if (minis.length) html += `<details class="lp-minis"><summary class="note">per-game curves (${minis.length})</summary>${minis.join('')}</details>`;
  el.innerHTML = html;
  wireCharts(el);
}

// ---- PPA cards ----------------------------------------------------------
function ppaDecided(gRow, T, winBy) {
  const [a, b] = gRow;
  if (a >= T && a - b >= winBy) return 1;
  if (b >= T && b - a >= winBy) return 2;
  return 0;
}

function ppaCard(t) {
  const byEvent = new Map();
  for (const m of t.matches) {
    const k = m.ev || 'Pro doubles';
    if (!byEvent.has(k)) byEvent.set(k, []);
    byEvent.get(k).push(m);
  }
  let html = `<div class="card lp-card"><div class="lp-head"><span class="lp-chip lp-ppa">PPA</span><span class="lp-teams">${esc(t.title)}</span></div>`;
  for (const [ev, ms] of byEvent) {
    html += `<h3 class="lp-ev">${esc(ev)}</h3><table class="lp-games"><tr><th>round</th><th>pairing</th><th>score</th><th>live</th><th>pre</th></tr>`;
    for (const m of ms) html += ppaRow(m);
    html += '</table>';
  }
  return html + '</div>';
}

function ppaRow(m) {
  const fmt = m.fmt, playable = fmt && !fmt.rally;
  const Ts = fmt ? fmt.max.filter(x => x > 0) : [];
  const winBy = fmt ? (fmt.winBy || 2) : 2;
  const need = Ts.length ? Math.ceil(Ts.length / 2) : null;
  let w1 = 0, w2 = 0, curIdx = -1;
  if (playable && Ts.length) {
    for (let i = 0; i < m.g.length && i < Ts.length; i++) {
      const d = ppaDecided(m.g[i], Ts[i], winBy);
      if (d === 1) w1++; else if (d === 2) w2++;
      else if (curIdx < 0 && m.st === 2) curIdx = i;
    }
    if (m.st === 2 && curIdx < 0) curIdx = Math.min(m.g.length - 1, Ts.length - 1);
  }
  let live = null, pre = null;
  if (playable && Ts.length) {
    const info = priceGame(m.t1, m.t2, Ts[0]);
    if (info) {
      const perGame = Ts.map(T => priceGame(m.t1, m.t2, T).p0);
      pre = seqProb(0, 0, need, perGame);
      if (m.st === 2 && curIdx >= 0) {
        const T = Ts[curIdx];
        const dp = PKL.ServeDP(PKL.etaAnchor(perGame[curIdx], CFG.k, T), CFG.k, T);
        const [a, b] = m.g[curIdx];
        const pCur = dp.p(a, b, stateFromSnap(m));
        recordSnapshot(m.uuid, a, b, stateFromSnap(m));
        const fut = perGame.slice(curIdx + 1);
        live = pCur * seqProb(w1 + 1, w2, need, fut) + (1 - pCur) * seqProb(w1, w2 + 1, need, fut);
      } else if (m.st === 2) {
        live = seqProb(w1, w2, need, perGame.slice(w1 + w2));
      }
    }
  }
  const score = m.g.map(([a, b]) => `<span class="lp-score">${a}–${b}</span>`).join(' ');
  const serve = m.st === 2 && m.svT ? `<span class="lp-serve">${m.svT === 1 ? '◀' : '▶'} #${m.svN}</span>` : '';
  const liveCell = m.st === 2
    ? (live !== null ? `<strong>${fpF(live)}%</strong>` : '<span class="note">' + (fmt ? (fmt.rally ? 'rally format' : 'unrated') : 'format?') + '</span>')
    : m.st === 4 ? (m.win === 1 ? '✓' : m.win === 2 ? '✗' : '') : '';
  const preCell = pre !== null ? `<span class="note">${fpF(pre)}%</span>` : '';
  const status = m.st === 2 ? ' class="lp-liverow"' : '';
  return `<tr${status}><td class="lp-slot">${esc(m.rd || '')}</td>` +
    `<td>${pairNames(m.t1, 1)}<br>${pairNames(m.t2, 2)}</td>` +
    `<td class="num">${score} ${serve}</td><td class="num">${liveCell}</td><td class="num">${preCell}</td></tr>`;
}

function seqProb(w1, w2, need, fut) {
  if (w1 >= need) return 1;
  if (w2 >= need) return 0;
  if (!fut.length) return 0.5;
  const p = fut[0], rest = fut.slice(1);
  return p * seqProb(w1 + 1, w2, need, rest) + (1 - p) * seqProb(w1, w2 + 1, need, rest);
}

// ---- top-level ----------------------------------------------------------
async function fetchLive() {
  const r = await fetch(`${CFG.api}/live`, { headers: CFG.headers });
  if (!r.ok) throw new Error(`live api ${r.status}`);
  return r.json();
}

let snapsLoaded = false;
function render(state) {
  CFG.today = state.date;
  chartReg = new Map(); chartSeq = 0;
  if (!snapsLoaded) { snapsLoaded = true; loadSnaps(); }
  const anyLive = state.mlp.some(mu => mu.matches.some(m => m.st === 2)) ||
    state.ppa.some(t => t.matches.some(m => m.st === 2));
  $asof.innerHTML = (anyLive ? '<span class="lp-dot"></span> LIVE — ' : '') +
    `board as of ${new Date(state.generated).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' })}` +
    (state.errors && state.errors.length ? ' <span class="note">(partial: upstream hiccup)</span>' : '');
  let html = '';
  const order = mu => ({ IN_PROGRESS_MATCHUP_STATUS: 0 }[mu.status] ?? (mu.status === 'COMPLETED_MATCHUP_STATUS' ? 2 : 1));
  for (const mu of [...state.mlp].sort((a, b2) => order(a) - order(b2))) html += matchupCard(mu);
  for (const t of state.ppa) html += ppaCard(t);
  if (!html) {
    html = `<div class="card"><p>No MLP or PPA pro play today${state.next && state.next.length ? ` — next event day: <strong>${esc(state.next[0])}</strong>` : ''}.
      See the <a href="forecast.html">forecast page</a> for what's coming.</p></div>`;
  }
  $app.innerHTML = html;
  for (const mu of state.mlp) fillTrack(mu);
}

async function tick() {
  try {
    const state = await fetchLive();
    errBackoff = 0;
    render(state);
  } catch (e) {
    errBackoff = Math.min((errBackoff || 1) * 2, 8);
    $asof.innerHTML = `<span class="note">connection hiccup — retrying (${esc(e.message)})</span>`;
  }
  const delay = CFG.poll * Math.max(1, errBackoff);
  pollTimer = setTimeout(() => { if (!document.hidden) tick(); else pollTimer = null; }, delay);
}

document.addEventListener('visibilitychange', () => {
  if (!document.hidden && !pollTimer) tick();
});

(async function boot() {
  try {
    const vr = await fetch(CFG.values);
    const vb = await vr.json();
    VALUES = vb.players;
    nameIndex = {};
    const counts = {};
    for (const [pid, rec] of Object.entries(VALUES)) {
      const k = rec[0].toLowerCase();
      counts[k] = (counts[k] || 0) + 1;
      nameIndex[k] = pid;
    }
    for (const k of Object.keys(nameIndex)) if (counts[k] > 1) delete nameIndex[k];
    await tick();
  } catch (e) {
    $app.innerHTML = `<div class="card"><p>Couldn't load the live board (${esc(e.message)}). Refresh to retry.</p></div>`;
  }
})();
"""

# ---------------------------------------------------------------- forecast
# Client-side "lineups official" repricing for forecast.html (Tier 1 of the
# lineup work): when the BFF publishes actual lineups for a forecasted
# matchup, the card reprices in the browser with the same engine the live
# board uses, and shows the projected-vs-official delta. Plain string,
# __CFG__ substituted at build.
FORECAST_JS = r"""
'use strict';
const CFG = __CFG__;
PKL.configure({ gamma: CFG.gamma, kDoubles: CFG.k, kDbSingles: CFG.kDbSingles,
                singlesImpute: CFG.singlesImpute, cal: CFG.cal, epsFloor: CFG.cal.eps });
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const cards = [...document.querySelectorAll('.fxcard')].map(el => ({
  el, fx: JSON.parse(el.dataset.fx), out: el.querySelector('.fx-official') }));
let VALUES = null, timer = null;
const fpF = p => PKL.fp(PKL.displayFloor(p));
function slotOf(ab) { ab = (ab || '').toUpperCase();
  for (const s of ['MXD1', 'MXD2', 'WD', 'MD']) if (ab.startsWith(s)) return s;
  return null; }
function priceIds(ids) {
  if (ids.length !== 4 || ids.some(id => !VALUES[id])) return null;
  const v = ids.map(id => VALUES[id][2]);
  return PKL.calibrate(PKL.raceDist(PKL.sig(PKL.teamEta(v[0], v[1], v[2], v[3])), 11).pw);
}
function schedule(ms) { clearTimeout(timer); timer = setTimeout(tick, ms); }
async function tick() {
  if (!cards.length || !VALUES) return;
  let pending = false;
  try {
    const r = await fetch(`${CFG.api}/live`, { headers: CFG.headers });
    const st = await r.json();
    for (const c of cards) {
      if (c.fx.d !== st.date) continue;
      const mu = (st.mlp || []).find(m => m.t1 === c.fx.t1 && m.t2 === c.fx.t2);
      if (!mu) { pending = true; continue; }
      const status = mu.status.replace('_MATCHUP_STATUS', '');
      const info = {};
      for (const m of mu.matches) {
        if (m.tb) continue;
        const s = slotOf(m.ab); if (!s) continue;
        const i1 = m.t1.map(p => p.id).filter(Boolean), i2 = m.t2.map(p => p.id).filter(Boolean);
        if (i1.length === 2 && i2.length === 2)
          info[s] = { ids: [...i1, ...i2], n1: m.t1.map(p => p.n), n2: m.t2.map(p => p.n) };
      }
      const nIn = Object.keys(info).length;
      if (!nIn) { if (status.startsWith('SCHEDULED')) pending = true; continue; }
      const proj = {}; for (const g of c.fx.games) proj[g.slot] = g;
      const diffs = [], ps = [];
      let allRated = true;
      for (const slot of ['WD', 'MD', 'MXD1', 'MXD2']) {
        const a = info[slot];
        if (!a) { allRated = false; continue; }
        const p = priceIds(a.ids);
        if (p === null) allRated = false; else ps.push(p);
        const pj = proj[slot];
        if (pj) {
          const norm = x => x.map(s => s.toLowerCase()).sort().join('|');
          if (norm(a.n1) !== norm(pj.t1) || norm(a.n2) !== norm(pj.t2)) diffs.push(slot);
        }
      }
      let html = '';
      if (nIn === 4 && allRated && ps.length === 4) {
        const roster = i => { const s = new Set();
          for (const slot of Object.keys(info))
            info[slot].ids.slice(i === 1 ? 0 : 2, i === 1 ? 2 : 4).forEach(id => s.add(id));
          return [...s].map(id => ({ sv: VALUES[id][4], v: VALUES[id][2] })); };
        const pDb = PKL.dbWinProb(roster(1), roster(2));
        const off = PKL.matchupProb(0, 0, ps, pDb);
        const fav1 = off >= 0.5;
        let was = '';
        if (c.fx.p != null) {
          const pf1 = c.fx.p >= 0.5;
          was = ` <span class="note">— projected had ${esc((pf1 ? c.fx.t1 : c.fx.t2).split(' ').pop())} ${fpF(pf1 ? c.fx.p : 1 - c.fx.p)}%</span>`;
        }
        html = `<strong>LINEUPS OFFICIAL:</strong> ${esc(fav1 ? c.fx.t1 : c.fx.t2)} <strong>${fpF(fav1 ? off : 1 - off)}%</strong>${was}`;
        if (diffs.length) html += `<br><span class="note">pairings differ from projection: ${diffs.join(', ')}</span>`;
        if (status === 'IN_PROGRESS') html += ' · <a href="live.html"><strong>LIVE now →</strong></a>';
        else if (status === 'COMPLETED') html += ` · <span class="note">final ${mu.s1}–${mu.s2}</span> · <a href="live.html">charts →</a>`;
        else pending = true;   // lineups can still change until first serve
      } else {
        html = `<span class="note">lineups announced for ${nIn}/4 games — full repricing when all four are in</span>`;
        pending = true;
      }
      c.out.innerHTML = html;
      c.out.hidden = false;
    }
  } catch (e) { pending = true; }
  if (pending && !document.hidden) schedule(60000);
}
document.addEventListener('visibilitychange', () => { if (!document.hidden) schedule(500); });
(async () => {
  try {
    const vr = await fetch(CFG.values);
    VALUES = (await vr.json()).players;
    tick();
  } catch (e) {}
})();
"""


def forecast_script():
    """The <script> block build_forecast appends: engine + repricer."""
    engine = (Path(__file__).parent / "live_engine.js").read_text()
    cal = json.loads((ROOT / "web" / "calibration.json").read_text())
    cfg = json.dumps({
        "api": API_BASE, "values": "data/live_values.json",
        "headers": ({"Authorization": f"Bearer {API_KEY}", "apikey": API_KEY}
                    if API_KEY else {}),
        "gamma": GAMMA, "k": K_DOUBLES, "kDbSingles": K_DB_SINGLES,
        "singlesImpute": list(SINGLES_IMPUTE),
        "cal": {"a": cal["a"], "b": cal["b"], "eps": cal["eps"]},
    })
    return f"<script>\n{engine}\n{FORECAST_JS.replace('__CFG__', cfg)}\n</script>"


LIVE_CSS = """
/* chart side colors — CVD-validated green/blue pair (light + dark steps
   checked with the palette validator against each surface) */
:root { --wp1: #1e7a3c; --wp2: #1d6fa5; }
@media (prefers-color-scheme: dark) { :root { --wp1: #65a82c; --wp2: #3a8fd4; } }
:root[data-theme="light"] { --wp1: #1e7a3c; --wp2: #1d6fa5; }
:root[data-theme="dark"] { --wp1: #65a82c; --wp2: #3a8fd4; }
.lp-sw { display:inline-block; width:9px; height:9px; margin:0 5px 0 2px; vertical-align:baseline; }
.lp-sw1 { background: var(--wp1); }
.lp-sw2 { background: var(--wp2); margin-left:5px; }
.lp-chart { position:relative; }
.lp-chart [data-catch] { cursor: crosshair; }
.lp-wptip { position:absolute; top:2px; pointer-events:none;
  background: var(--surface); border:1.5px solid var(--baseline);
  padding:2px 8px; font-family:"Space Mono",ui-monospace,monospace;
  font-size:11.5px; white-space:nowrap; }
.lp-head { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
.lp-teams { font-weight:600; font-size:17px; }
.lp-big { font-size:22px; }
.lp-chip { font-family:"Space Mono",ui-monospace,monospace; font-size:11px; padding:1px 7px;
  border:1px solid var(--border); color:var(--ink2); letter-spacing:.04em; }
.lp-chip.lp-live { background:var(--hl); color:var(--hl-ink); border-color:transparent; font-weight:700; }
.lp-chip.lp-ppa { background:var(--wash); }
.lp-dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--loss);
  animation:lp-pulse 1.4s ease-in-out infinite; vertical-align:baseline; }
@keyframes lp-pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
.lp-games { width:100%; border-collapse:collapse; margin:8px 0 2px; }
.lp-games th { text-align:left; font-size:11px; color:var(--muted); font-weight:500;
  font-family:"Space Mono",ui-monospace,monospace; }
.lp-games td { padding:4px 8px 4px 0; border-top:1px solid var(--grid); vertical-align:top; font-size:13.5px; }
.lp-games td.num, .lp-games th:nth-child(n+3) { text-align:right; white-space:nowrap; }
.lp-slot { font-family:"Space Mono",ui-monospace,monospace; font-size:12px; color:var(--ink2); }
.lp-liverow { background:var(--band); }
.lp-score { font-family:"Space Mono",ui-monospace,monospace; padding:0 2px; }
.lp-serve { color:var(--s1); font-size:11px; font-family:"Space Mono",ui-monospace,monospace; }
.lp-track { margin-top:6px; }
.lp-tracklab { margin:2px 0; }
.lp-mini { margin:4px 0; }
.lp-minis summary { cursor:pointer; }
.lp-ev { font-size:13px; margin:10px 0 2px; color:var(--ink2); }
.lp-dbnote { margin:4px 0 0; }
.lp-sampled, .lp-nolog { font-size:11.5px; }
#live-asof { font-family:"Space Mono",ui-monospace,monospace; font-size:12.5px; color:var(--ink2); }
"""


def build_live(players, cal, updated, site_dir, write):
    n = build_values_json(players, cal, updated, site_dir)
    engine = (Path(__file__).parent / "live_engine.js").read_text()
    cfg = json.dumps({
        "api": API_BASE, "values": "data/live_values.json", "poll": POLL_MS,
        "headers": ({"Authorization": f"Bearer {API_KEY}", "apikey": API_KEY}
                    if API_KEY else {}),
        "gamma": GAMMA, "k": K_DOUBLES, "kDbSingles": K_DB_SINGLES,
        "singlesImpute": list(SINGLES_IMPUTE),
        "cal": {"a": cal["a"], "b": cal["b"], "eps": cal["eps"]},
    })
    body = f"""
<style>{LIVE_CSS}</style>
<h1>Live</h1>
<p class="sub">Rally-by-rally win probability for today's MLP matchups and PPA
pro doubles, from the same validated model the receipts ledger grades — the
serve-aware engine is anchored so its pre-match numbers match the frozen
forecasts exactly. Board refreshes every ~20 seconds.</p>
<p id="live-asof">connecting…</p>
<div id="live-app"><div class="card"><p class="note">Loading the live board…</p></div></div>
<noscript><p>This page needs JavaScript — it polls live scores and computes
win probabilities in your browser.</p></noscript>
<p class="note">Fine print: per-point strength from the v2 Bayesian model
(36k games); serve-aware side-out DP with league serve-rally rate
k&nbsp;=&nbsp;{K_DOUBLES} measured from referee logs; each game anchored to the
calibrated pre-match probability, so live curves and the
<a href="receipts.html">receipts ledger</a> agree at rally zero. In-game
calibration is provisional pending the full rally-log backfill — treat
mid-game numbers as honest estimates, not gospel. DreamBreakers use the
singles model (rough by design). Probabilities are floored — nothing is ever
0% or 100%. Rally-resolution curves appear on digitally-refereed courts;
otherwise the page samples the scoreboard every ~20&nbsp;s. Chart ticks:
T&nbsp;=&nbsp;timeout, C&nbsp;=&nbsp;challenge/review.</p>
<script>
{engine}
{LIVE_JS.replace("__CFG__", cfg)}
</script>
"""
    write("live.html", style.page("Live — PICKLES", body, "live.html", "", updated))
    return n
