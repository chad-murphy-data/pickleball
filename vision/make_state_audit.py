"""State audit v2 — guided, one-player-per-pass labeling for the
hitter-episode pilot (v1 same day; reworked to the user's UX spec:
"Rally 1: tracking Etta Tuionetoa (highlight her in a green box)…"
with named states instead of raw episode taps).

THE FLOW (build 2026-08-29b): each rally is 4 PASSES, one per player.
The tool tells you who you're tracking; if pose data was embedded
(--pose-dir at generation time) you click that player's box once at
the start of the pass and a green box follows her for the whole pass.
Then you just answer, whenever it changes, "what is SHE doing?":

  (default)  BALL-WATCHING — ready/tracking the ball. Costs ZERO
             clicks: every pass starts here and every other state
             returns here. The default-state trick is what keeps this
             boundary coding rather than per-frame coding.
  C          CLEARING — moving out of the way for her partner
             (user-named state; the freeze-out/coverage geometry
             signal, and a hard negative for the hitter decoder).
             Press C again (or W) when she's back to watching.
  B          HITTING EPISODE begins ("beginning to hit the ball") —
             then I on the exact impact frame, E when the
             follow-through is done (auto-returns to watching).
             X toggles NO-CONTACT on the episode (fake /
             both-went-for-it / aborted — never struck).

Rally-level marks (kept from v1): R = service routine starts,
D = point dead. N = pass done -> next player / next rally.

Frame keys: arrows = ±1 frame, ,/. = ±10, space = play/pause.
The gray ticks are your OLD contact taps (red = fast) — navigation
aid only; find the impact frame yourself, never snap to the tick
(measuring your own tap jitter is half the pilot).

Pilot rallies pre-specified (train only, r9/r10 quarantined): 1-8,
13, 14 — 124 contacts. Export every sitting as
data/vision/state_labels_chicago0725.csv. CSV schema
(rally_cum,player,episode,kind,t_s) is v1's plus three run kinds:
clear_start/clear_end, and track_assign rows (t_s holds the tid) —
your click-to-identify answers are themselves ground truth the
touch-attribution thread can grade against.

Generate WITH boxes (on the Mac, pose_rtm/ present):
    python3 vision/make_state_audit.py --pose-dir pose_rtm
Without --pose-dir the tool still works; passes just show the name
banner with no box.

GATE NOTE unchanged: new label types — before any temporal-model code
exists, temporal_gate.md needs the dated amendment (draft in
swing_explore_notes.md 2026-08-29, awaiting the user's explicit
freeze).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "vision"
LABELS = DATA / "contact_labels_chicago0725.csv"
OUT_HTML = DATA / "state_audit_chicago0725.html"
PILOT = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14]
PRE_PAD_S = 8.0   # covers the service routine (log lead ~6 s)
POST_PAD_S = 4.0  # covers point-dead + follow-through
BOX_FPS = 10      # embedded box sampling rate
KPT_CONF = 0.3
POSE_W, POSE_H = 1280, 720   # pose working frame (paddle_probe convention)


def build_rallies(labels_path=LABELS, pilot=PILOT):
    rows = list(csv.DictReader(open(labels_path)))
    out = []
    for cum in pilot:
        rs = [r for r in rows if int(r["rally_cum"]) == cum]
        if not rs:
            raise SystemExit(f"pilot rally {cum} has no labels")
        game = rs[0]["game"]
        players = sorted({r["hitter_name"] for r in rows
                          if r["game"] == game})
        if len(players) != 4:
            raise SystemExit(f"game {game}: expected 4 players, "
                             f"got {players}")
        contacts = []
        for r in rs:
            t = float(r["t_refined_s"] or r["t_tap_s"])
            contacts.append({"t": round(t, 3),
                             "hitter": r["hitter_name"],
                             "type": r["shot_type"],
                             "whiff": r.get("contact", "1") == "0"})
        contacts.sort(key=lambda c: c["t"])
        out.append({"rally_cum": cum,
                    "players": players,
                    "t0": round(contacts[0]["t"] - PRE_PAD_S, 3),
                    "t1": round(contacts[-1]["t"] + POST_PAD_S, 3),
                    "contacts": contacts,
                    "tracks": {}})
    return out


def embed_tracks(rallies, pose_dir):
    """Per rally: {tid: [[t, x, y, w, h], ...]} at ~BOX_FPS, box from
    confident-keypoint extents in the pose working frame (1280x720) —
    the same derivation paddle_probe uses for rtmlib boxes, so nothing
    new is being trusted."""
    import numpy as np
    n_done = 0
    for r in rallies:
        p = Path(pose_dir) / f"r{r['rally_cum']:04d}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        t = np.asarray(z["t"], dtype=float)
        trk = np.asarray(z["track"])
        kpt = np.asarray(z["kpt"], dtype=float)
        kpc = np.asarray(z["kpc"], dtype=float)
        keep = (t >= r["t0"]) & (t <= r["t1"])
        tracks = {}
        for tid in np.unique(trk[keep]):
            m = keep & (trk == tid)
            if m.sum() < 8:
                continue
            ts, ks, cs = t[m], kpt[m], kpc[m]
            o = np.argsort(ts)
            ts, ks, cs = ts[o], ks[o], cs[o]
            rows, last = [], -1e9
            for i in range(len(ts)):
                if ts[i] - last < 1.0 / BOX_FPS:
                    continue
                conf = cs[i] >= KPT_CONF
                if conf.sum() < 3:
                    continue
                xs, ys = ks[i][conf, 0], ks[i][conf, 1]
                x0, x1 = xs.min() - 14, xs.max() + 14
                y0, y1 = ys.min() - 10, ys.max() + 10
                rows.append([round(float(ts[i]), 2), int(x0), int(y0),
                             int(x1 - x0), int(y1 - y0)])
                last = ts[i]
            if rows:
                tracks[int(tid)] = rows
        r["tracks"] = tracks
        if tracks:
            n_done += 1
    return n_done


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>state audit — hitter episodes (pilot v2)</title>
<style>
 body{font:14px system-ui;margin:0;background:#111;color:#ddd}
 #wrap{max-width:1040px;margin:0 auto;padding:12px}
 #vbox{position:relative}
 video{width:100%;display:block;background:#000;border-radius:6px}
 #ov{position:absolute;left:0;top:0;pointer-events:auto}
 #drop{border:2px dashed #555;border-radius:8px;padding:18px;text-align:center;
       cursor:pointer;margin:8px 0}
 #banner{font-size:17px;margin:8px 0;padding:8px 12px;background:#1d2b1d;
         border:1px solid #4a7;border-radius:6px}
 #banner b{color:#8f8}
 #banner.assign{background:#2b241d;border-color:#a74}
 .bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:6px 0}
 button{background:#2a2a2a;color:#ddd;border:1px solid #555;border-radius:5px;
        padding:4px 10px;cursor:pointer}
 kbd{background:#333;border-radius:4px;padding:1px 6px;border:1px solid #555}
 .lane{position:relative;height:26px;background:#1a1a1a;border-radius:4px;
       margin:3px 0}
 .lane.dim{opacity:.45;height:14px}
 .lane .nm{position:absolute;left:6px;top:4px;font-size:12px;color:#999;
           pointer-events:none;z-index:2}
 .run{position:absolute;top:3px;height:20px;border-radius:3px;cursor:pointer}
 .run.hit{background:#2d4d2d;border:1px solid #4a7}
 .run.hit.nc{background:#4d3d2d;border-color:#a74}
 .run.hit.open{border-style:dashed}
 .run.clear{background:#28405c;border:1px solid #58a}
 .imp{position:absolute;top:0;width:2px;height:26px;background:#8f8;z-index:1}
 .tick{position:absolute;top:20px;width:1px;height:6px;background:#666}
 .tick.f{background:#c66}
 .cursor{position:absolute;top:0;width:1px;height:100%;background:#8cf}
 .rmark{position:absolute;top:0;width:2px;height:100%;background:#d93}
 .dmark{position:absolute;top:0;width:2px;height:100%;background:#d44}
 #note{color:#999;font-size:12px;line-height:1.6;margin-top:8px}
 select,input{background:#222;color:#ddd;border:1px solid #555;border-radius:4px}
 #statebadge{padding:2px 10px;border-radius:10px;background:#333}
 #statebadge.hit{background:#2d5d2d}#statebadge.clear{background:#28507c}
</style></head><body><div id="wrap">
<div id="drop">🎬 <b>Load the match video</b> (full_match.mp4.webm) — click or drag
<input type="file" id="fpick" accept="video/*,.webm,.mp4,.mkv" hidden></div>
<div id="vbox"><video id="v" preload="auto"></video><canvas id="ov"></canvas></div>
<div id="banner">—</div>
<div class="bar">
 <label>rally <select id="rsel"></select></label>
 <label>pass <select id="psel"></select></label>
 <label>fps <input id="fps" type="number" value="30" step="0.01" style="width:60px"></label>
 <span>state: <span id="statebadge">watching</span></span>
 <span style="flex:1"></span>
 <span id="status">—</span>
 <button id="bnext">N next pass ▸</button>
 <button id="bexp">⬇ export</button>
 <button id="bimp">⬆ import</button>
 <input type="file" id="csvpick" accept=".csv" hidden>
</div>
<div id="lanes"></div>
<div id="note">
<b>Per pass, answer "what is she doing?" whenever it changes.</b>
Default = <b>ball-watching</b> (zero clicks — every state returns
here). <kbd>C</kbd> clearing for partner (again/<kbd>W</kbd> = back to
watching) · <kbd>B</kbd> beginning to hit · <kbd>I</kbd> exact impact
frame · <kbd>E</kbd> follow-through done · <kbd>X</kbd> no-contact
episode (fake / never struck) · <kbd>R</kbd> service routine starts ·
<kbd>D</kbd> point dead · <kbd>N</kbd> pass done ·
<kbd>←</kbd>/<kbd>→</kbd> ±1 frame · <kbd>,</kbd>/<kbd>.</kbd> ±10 ·
<kbd>space</kbd> play/pause · <kbd>⌫</kbd> delete this player's
nearest mark. Gray ticks = your old taps (red = fast): navigate by
them, <b>never snap to them</b>. If boxes are embedded, click your
player once when asked (orange banner) — the green box then follows
her; <kbd>A</kbd> re-assigns. Export every sitting as
<code>state_labels_chicago0725.csv</code> into data/vision/.
</div>
</div><script>
const RALLIES = __RALLIES__;
const LSK = "state_audit_chicago0725_v2";
let store = JSON.parse(localStorage.getItem(LSK) || "{}");
// store[cum] = {players:{name:{hits:[{s,i,e,nc}],clears:[{s,e}],tid}},
//               _rally:{routine,dead}}
const save = () => localStorage.setItem(LSK, JSON.stringify(store));
const V = document.getElementById("v"), OV = document.getElementById("ov");
const fpsEl = document.getElementById("fps");
const frame = () => 1 / (+fpsEl.value || 30);
let R = RALLIES[0], passI = 0, assigning = false;

function D(){ const d = store[R.rally_cum] = store[R.rally_cum] ||
  {players: {}, _rally: {routine: null, dead: null}};
  d.players = d.players || {}; d._rally = d._rally || {routine:null,dead:null};
  return d; }
function P(name){ const d = D();
  return d.players[name] = d.players[name] ||
    {hits: [], clears: [], tid: null}; }
const curName = () => R.players[passI];

drop.onclick = () => fpick.click();
drop.ondragover = e => e.preventDefault();
drop.ondrop = e => {e.preventDefault(); loadf(e.dataTransfer.files[0]);};
fpick.onchange = () => loadf(fpick.files[0]);
function loadf(f){ if(f){ V.src = URL.createObjectURL(f); goPass(0); } }

const rsel = document.getElementById("rsel"), psel = document.getElementById("psel");
RALLIES.forEach((r, i) => { const o = document.createElement("option");
  o.value = i; o.textContent = "rally " + r.rally_cum; rsel.appendChild(o); });
rsel.onchange = () => { R = RALLIES[+rsel.value]; goPass(0); };
function fillPsel(){ psel.innerHTML = "";
  R.players.forEach((p, i) => { const o = document.createElement("option");
    o.value = i; o.textContent = (i + 1) + ": " + p; psel.appendChild(o); }); }
psel.onchange = () => goPass(+psel.value);

function goPass(i){
  passI = i; fillPsel(); psel.value = i;
  const has = Object.keys(R.tracks || {}).length > 0;
  assigning = has && P(curName()).tid == null;
  if (V.src) { V.pause();
    V.currentTime = assigning ? R.contacts[0].t : R.t0; }
  render();
}
bnext.onclick = next;
function next(){
  if (passI < 3) goPass(passI + 1);
  else { const ri = RALLIES.indexOf(R);
    if (ri < RALLIES.length - 1) { R = RALLIES[ri + 1];
      rsel.value = ri + 1; goPass(0); } }
}

function boxAt(tid, t){
  const rows = (R.tracks || {})[tid]; if (!rows) return null;
  let best = null, bd = 0.35;
  for (const r of rows){ const d = Math.abs(r[0] - t);
    if (d < bd) { bd = d; best = r; } }
  return best;
}
function drawOverlay(){
  OV.width = V.clientWidth; OV.height = V.clientHeight;
  const ctx = OV.getContext("2d");
  ctx.clearRect(0, 0, OV.width, OV.height);
  if (!V.videoWidth) return;
  const sx = OV.width / 1280, sy = OV.height / 720, t = V.currentTime;
  if (assigning){
    ctx.font = "13px system-ui";
    for (const tid of Object.keys(R.tracks || {})){
      const b = boxAt(+tid, t); if (!b) continue;
      ctx.strokeStyle = "#d93"; ctx.lineWidth = 2;
      ctx.strokeRect(b[1] * sx, b[2] * sy, b[3] * sx, b[4] * sy);
    }
  } else {
    const tid = P(curName()).tid;
    if (tid != null){ const b = boxAt(tid, t);
      if (b){ ctx.strokeStyle = "#4caf50"; ctx.lineWidth = 3;
        ctx.strokeRect(b[1] * sx, b[2] * sy, b[3] * sx, b[4] * sy);
        ctx.fillStyle = "#4caf50"; ctx.font = "bold 13px system-ui";
        ctx.fillText(curName(), b[1] * sx, Math.max(12, b[2] * sy - 5)); } }
  }
}
OV.onclick = e => {
  if (assigning){
    const sx = OV.width / 1280, sy = OV.height / 720, t = V.currentTime;
    for (const tid of Object.keys(R.tracks || {})){
      const b = boxAt(+tid, t); if (!b) continue;
      if (e.offsetX >= b[1] * sx && e.offsetX <= (b[1] + b[3]) * sx &&
          e.offsetY >= b[2] * sy && e.offsetY <= (b[2] + b[4]) * sy){
        P(curName()).tid = +tid; assigning = false;
        if (V.src) V.currentTime = R.t0;
        save(); render(); return;
      }
    }
  } else { if (V.paused) V.play(); else V.pause(); }
};

function stateAt(name, t){
  const p = P(name);
  for (const h of p.hits)
    if (h.s != null && t >= h.s && (h.e == null || t <= h.e)) return "hit";
  for (const c of p.clears)
    if (c.s != null && t >= c.s && (c.e == null || t <= c.e)) return "clear";
  return "watching";
}

function render(){
  const b = document.getElementById("banner");
  if (assigning){
    b.className = "assign";
    b.innerHTML = `Rally ${R.rally_cum} — pass ${passI + 1}/4: ` +
      `<b>click the box on ${curName()}</b> (orange boxes; paused at the serve)`;
  } else {
    b.className = "";
    b.innerHTML = `Rally ${R.rally_cum} — pass ${passI + 1}/4: tracking ` +
      `<b>${curName()}</b>. What is she doing?`;
  }
  const st = stateAt(curName(), V.currentTime || 0);
  const sb = document.getElementById("statebadge");
  sb.textContent = {watching: "ball-watching", clear: "clearing",
                    hit: "hitting"}[st];
  sb.className = st === "watching" ? "" : st;

  const lanes = document.getElementById("lanes"); lanes.innerHTML = "";
  const span = R.t1 - R.t0, X = t => (100 * (t - R.t0) / span) + "%";
  R.players.forEach((p, i) => {
    const lane = document.createElement("div");
    lane.className = "lane" + (i === passI ? "" : " dim");
    lane.innerHTML = `<span class="nm">${i + 1} ${p}</span>`;
    const pd = P(p);
    pd.clears.forEach(c => { if (c.s == null) return;
      const d = document.createElement("div"); d.className = "run clear";
      d.style.left = X(c.s);
      d.style.width = Math.max(0.4, 100 * ((c.e ?? c.s) - c.s) / span) + "%";
      d.title = `clearing ${c.s.toFixed(2)}–${c.e ? c.e.toFixed(2) : "…"}`;
      lane.appendChild(d); });
    pd.hits.forEach((h, j) => { if (h.s == null && h.i == null) return;
      const s = h.s ?? h.i, e = h.e ?? h.i ?? h.s;
      const d = document.createElement("div");
      d.className = "run hit" + (h.nc ? " nc" : "") +
        (h.e == null ? " open" : "");
      d.style.left = X(s);
      d.style.width = Math.max(0.4, 100 * (e - s) / span) + "%";
      d.title = `hit ep${j + 1} ${h.s?.toFixed(2)}–${h.e?.toFixed(2)}` +
        (h.i ? ` impact ${h.i.toFixed(2)}` : "") + (h.nc ? " NO-CONTACT" : "");
      d.onclick = ev => { ev.stopPropagation();
        V.currentTime = h.i ?? s; };
      if (h.i != null){ const m = document.createElement("div");
        m.className = "imp"; m.style.left = X(h.i); lane.appendChild(m); }
      lane.appendChild(d); });
    R.contacts.filter(c => c.hitter === p).forEach(c => {
      const t = document.createElement("div");
      t.className = "tick" + (["smash","speed-up","drive","counter","fast"]
        .includes(c.type) ? " f" : "");
      t.style.left = X(c.t); t.title = c.type + " @" + c.t;
      lane.appendChild(t); });
    const rm = D()._rally;
    if (rm.routine != null){ const m = document.createElement("div");
      m.className = "rmark"; m.style.left = X(rm.routine); lane.appendChild(m); }
    if (rm.dead != null){ const m = document.createElement("div");
      m.className = "dmark"; m.style.left = X(rm.dead); lane.appendChild(m); }
    const cu = document.createElement("div"); cu.className = "cursor";
    cu.style.left = X(Math.min(Math.max(V.currentTime || R.t0, R.t0), R.t1));
    lane.appendChild(cu);
    lane.onclick = ev => { const r = lane.getBoundingClientRect();
      V.currentTime = R.t0 + span * (ev.clientX - r.left) / r.width; };
    lanes.appendChild(lane);
  });
  const marked = RALLIES.filter(r => { const d = store[r.rally_cum];
    return d && Object.values(d.players || {}).some(
      p => p.hits.length || p.clears.length); }).length;
  status.textContent = `t=${(V.currentTime || 0).toFixed(2)}s · ` +
    `rallies touched ${marked}/${RALLIES.length}`;
  drawOverlay();
}
V.ontimeupdate = render;
new ResizeObserver(drawOverlay).observe(V);

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const k = e.key.toLowerCase(), t = V.currentTime, p = P(curName());
  const openHit = p.hits.length && p.hits[p.hits.length - 1].e == null
    ? p.hits[p.hits.length - 1] : null;
  const openClear = p.clears.length && p.clears[p.clears.length - 1].e == null
    ? p.clears[p.clears.length - 1] : null;
  if (k === "arrowright") V.currentTime = t + frame();
  else if (k === "arrowleft") V.currentTime = Math.max(0, t - frame());
  else if (k === ".") V.currentTime = t + 10 * frame();
  else if (k === ",") V.currentTime = Math.max(0, t - 10 * frame());
  else if (k === " ") { if (V.paused) V.play(); else V.pause(); }
  else if (k === "c") {
    if (openClear) openClear.e = +t.toFixed(3);
    else p.clears.push({s: +t.toFixed(3), e: null});
  }
  else if (k === "w") { if (openClear) openClear.e = +t.toFixed(3); }
  else if (k === "b") {
    if (openClear) openClear.e = +t.toFixed(3);
    p.hits.push({s: +t.toFixed(3), i: null, e: null, nc: false});
  }
  else if (k === "i" && openHit) openHit.i = +t.toFixed(3);
  else if (k === "e" && openHit) openHit.e = +t.toFixed(3);
  else if (k === "x" && p.hits.length)
    p.hits[p.hits.length - 1].nc = !p.hits[p.hits.length - 1].nc;
  else if (k === "r") { const m = D()._rally;
    m.routine = m.routine == null ? +t.toFixed(3) : null; }
  else if (k === "d") { const m = D()._rally;
    m.dead = m.dead == null ? +t.toFixed(3) : null; }
  else if (k === "n") { next(); e.preventDefault(); return; }
  else if (k === "a") { if (Object.keys(R.tracks || {}).length){
    p.tid = null; assigning = true;
    V.currentTime = R.contacts[0].t; } }
  else if (k === "backspace") {
    let best = null, bd = 1e9;
    p.hits.forEach(h => ["s","i","e"].forEach(f => {
      if (h[f] != null && Math.abs(h[f] - t) < bd)
        { bd = Math.abs(h[f] - t); best = ["hits", h, f]; } }));
    p.clears.forEach(c => ["s","e"].forEach(f => {
      if (c[f] != null && Math.abs(c[f] - t) < bd)
        { bd = Math.abs(c[f] - t); best = ["clears", c, f]; } }));
    if (best) { best[1][best[2]] = null;
      const L = p[best[0]];
      if (Object.values(best[1]).every(v => v == null || v === false))
        L.splice(L.indexOf(best[1]), 1); }
  } else return;
  save(); render(); e.preventDefault();
});

bexp.onclick = () => {
  let out = "rally_cum,player,episode,kind,t_s\n";
  RALLIES.forEach(r => {
    const d = store[r.rally_cum]; if (!d) return;
    const rm = d._rally || {};
    if (rm.routine != null) out += `${r.rally_cum},,0,routine_start,${rm.routine}\n`;
    if (rm.dead != null) out += `${r.rally_cum},,0,point_dead,${rm.dead}\n`;
    r.players.forEach(p => { const pd = (d.players || {})[p]; if (!pd) return;
      if (pd.tid != null) out += `${r.rally_cum},${p},0,track_assign,${pd.tid}\n`;
      pd.hits.forEach((h, j) => {
        if (h.s != null) out += `${r.rally_cum},${p},${j + 1},start,${h.s}\n`;
        if (h.i != null) out += `${r.rally_cum},${p},${j + 1},impact,${h.i}\n`;
        if (h.e != null) out += `${r.rally_cum},${p},${j + 1},end,${h.e}\n`;
        if (h.nc) out += `${r.rally_cum},${p},${j + 1},no_contact,\n`; });
      pd.clears.forEach((c, j) => {
        if (c.s != null) out += `${r.rally_cum},${p},${j + 1},clear_start,${c.s}\n`;
        if (c.e != null) out += `${r.rally_cum},${p},${j + 1},clear_end,${c.e}\n`; });
    });
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([out], {type: "text/csv"}));
  a.download = "state_labels_chicago0725.csv"; a.click();
};
bimp.onclick = () => csvpick.click();
csvpick.onchange = () => {
  const rd = new FileReader();
  rd.onload = () => {
    store = {};
    rd.result.trim().split("\n").slice(1).forEach(line => {
      const [cum, pn, epn, kind, ts] = line.split(",");
      const r = RALLIES.find(x => x.rally_cum === +cum); if (!r) return;
      const d = store[+cum] = store[+cum] ||
        {players: {}, _rally: {routine: null, dead: null}};
      if (kind === "routine_start") { d._rally.routine = +ts; return; }
      if (kind === "point_dead") { d._rally.dead = +ts; return; }
      const p = d.players[pn] = d.players[pn] ||
        {hits: [], clears: [], tid: null};
      if (kind === "track_assign") { p.tid = +ts; return; }
      const isClear = kind.startsWith("clear_");
      const L = isClear ? p.clears : p.hits;
      while (L.length < +epn)
        L.push(isClear ? {s: null, e: null}
                       : {s: null, i: null, e: null, nc: false});
      const ep = L[+epn - 1];
      if (kind === "no_contact") ep.nc = true;
      else ep[{start: "s", impact: "i", end: "e", clear_start: "s",
               clear_end: "e"}[kind]] = +ts;
    });
    save(); render();
  };
  rd.readAsText(csvpick.files[0]);
};
fillPsel(); render();
</script></body></html>
"""


def selftest():
    rallies = build_rallies()
    assert [r["rally_cum"] for r in rallies] == PILOT
    for r in rallies:
        assert len(r["players"]) == 4
        assert r["t0"] < r["contacts"][0]["t"] < r["contacts"][-1]["t"] < r["t1"]
        assert r["tracks"] == {}
    n = sum(len(r["contacts"]) for r in rallies)
    assert n == 124, n
    assert 9 not in PILOT and 10 not in PILOT
    # synthetic pose npz -> box embedding
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        r0 = rallies[0]
        ts = np.arange(r0["t0"], r0["t1"], 1 / 30)
        nfr = len(ts)
        kpt = np.zeros((nfr * 2, 17, 2))
        kpt[:nfr, :, 0], kpt[:nfr, :, 1] = 300, 400
        kpt[nfr:, :, 0], kpt[nfr:, :, 1] = 900, 200
        np.savez(Path(td) / f"r{r0['rally_cum']:04d}.npz",
                 t=np.concatenate([ts, ts]),
                 track=np.concatenate([np.zeros(nfr), np.ones(nfr)]),
                 kpt=kpt, kpc=np.full((nfr * 2, 17), 0.9),
                 side=np.zeros(nfr * 2), fps=30.0, hw=720)
        done = embed_tracks(rallies, td)
        assert done == 1 and set(r0["tracks"]) == {0, 1}
        row = r0["tracks"][0][0]
        assert abs(row[0] - r0["t0"]) < 0.2 and row[1] == 300 - 14
        step = r0["tracks"][0][1][0] - r0["tracks"][0][0][0]
        assert step >= 1 / BOX_FPS - 1e-6
    html = HTML.replace("__RALLIES__", json.dumps(rallies))
    assert "__RALLIES__" not in html and '"rally_cum": 13' in html
    print(f"selftest OK — {len(rallies)} pilot rallies, {n} contacts, "
          f"box embedding verified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--pose-dir", default=None,
                    help="pose_rtm/ dir with r*.npz — embeds per-track "
                         "boxes so the tool can highlight the tracked "
                         "player (omit: tool works, no boxes)")
    ap.add_argument("--out", default=str(OUT_HTML))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    rallies = build_rallies(a.labels)
    n_boxed = embed_tracks(rallies, a.pose_dir) if a.pose_dir else 0
    Path(a.out).write_text(HTML.replace("__RALLIES__", json.dumps(rallies)))
    n = sum(len(r["contacts"]) for r in rallies)
    print(f"wrote {a.out} — {len(rallies)} rallies, {n} contacts, "
          f"boxes embedded for {n_boxed} rallies"
          + ("" if a.pose_dir else " (no --pose-dir: name banner only)"))


if __name__ == "__main__":
    main()
