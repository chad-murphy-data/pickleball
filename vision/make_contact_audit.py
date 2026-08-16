"""Generate the CONTACT-TIME labeling instrument (Gate C, pre-registered
in vision/contact_gate.md — read that first; bars are frozen there).

Successor to vision/make_shot_audit.py with one deliberate inversion:
that tool asked for NO timestamps because the audio pop train was going
to carry the timing, and audio's premise failed (POSTMORTEM.md §Gate B).
Here the human carries the timing — each tap records the video clock, so
the export supervises a frame-level detector instead of only grading one.

Two-pass design (agreed 2026-08-15):

  PASS 1 — play at 0.25-1x and STAMP each contact as it happens.
    Core-16 rallies (rally_cum 1-16, the frozen ordinal label set) are
    prefilled with their known hitter+type sequence, so one key (Enter)
    stamps the next expected shot; keys 1-4 stamp an explicit hitter
    (divergence from the prefill is flagged, not silently accepted).
    Fresh rallies use keys 1-4; shots 1-2 auto-type serve/return (rule,
    not guess). W arms a WHIFF (swing-and-miss: a swing event but NOT a
    contact — its own class, contact=0, never consumes the prefill).
    Tap jitter is measured against the 16 hand-pinned serves of
    rally_windows_chicago0725_v4.csv and shown live; the recommended
    training guard band (max(0.4 s, 3x p95)) comes from that measurement.

  PASS 2 (optional) — per-tap refine: jump to the tap, nudge with 5-frame
    / 1-frame steps, re-stamp. Only worth running if the ceiling test or
    the pilot says label sharpness is binding; the tap pass alone is
    designed to train (guard bands absorb tap noise on the negative side,
    window models + jitter augmentation absorb it on the positive side).

Inputs (all committed):
    data/vision/rally_timeline_matchup_20260725_c4e686d1.csv  rally spine
    data/vision/rally_timeline_matchup_20260725_c4e686d1_meta.json
    data/vision/rally_windows_chicago0725_v4.csv   video windows (16 pinned)
    data/vision/shot_labels_chicago0725.csv        ordinal labels -> prefill
    data/players.csv, data/games.csv               names + rosters

Output: data/vision/contact_audit_chicago0725.html (self-contained; the
user loads their local VOD into it — nothing is uploaded), which exports
data/vision/contact_labels_chicago0725.csv, consumed by
vision/pose_extract.py and vision/contact_ceiling.py.

    python vision/make_contact_audit.py --data-dir .
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from make_shot_audit import MATCHUP, VIDEO_NOTE, SHOT_TYPES, load, cumulative

CORE = set(range(1, 17))     # the frozen 16-rally label set (womens game)
PILOT_PER_GAME = 12          # suggested pilot corpus: longest per game

TYPES = SHOT_TYPES + [
    ("w", "whiff", "swing-and-miss or full swing at a fake — a SWING but "
                   "not a contact (stamped, contact=0, own class)"),
]


def load_windows(path: Path):
    win = {}
    for r in csv.DictReader(open(path)):
        win[int(r["rally_cum"])] = {
            "t0": float(r["t0s"]), "t1": float(r["t1s"]),
            "approx": r["approx"] == "1",
        }
    return win


def load_prefill(path: Path):
    """rally_cum -> {shots: [{h,t}], note, pin} from the ordinal labels.
    Restricted to the frozen CORE set: rallies 17/18/43/49 carry 2-row
    serve/return template stubs from the abandoned core-20 plan (and one
    stale early-tool mark — see serve_pin_windows.py), not labels."""
    pf = {}
    for r in csv.DictReader(open(path)):
        cum = int(r["rally_cum"])
        if cum not in CORE:
            continue
        d = pf.setdefault(cum, {"shots": [], "note": "", "pin": None})
        idx = int(r["shot_index"])
        while len(d["shots"]) < idx:
            d["shots"].append({"h": None, "t": None})
        d["shots"][idx - 1] = {"h": r["hitter_uuid"].lower() or None,
                               "t": r["shot_type"] or None}
        if idx == 1:
            if r.get("rally_note"):
                d["note"] = r["rally_note"]
            if r.get("serve_time_s"):
                d["pin"] = float(r["serve_time_s"])
    return pf


def pick_pilot(rallies, windows):
    """Suggested pilot corpus: the longest non-core, non-approx rallies,
    PILOT_PER_GAME from each game — long rallies are where the speed-up
    exchanges live, and all four games keeps the corpus off a
    one-division diet."""
    pilot = set()
    by_slot = {}
    for r in rallies:
        w = windows.get(r["cum"])
        if w and not w["approx"] and r["cum"] not in CORE:
            by_slot.setdefault(r["slot"], []).append(r)
    for rs in by_slot.values():
        rs.sort(key=lambda x: (-x["dur"], x["rally"]))
        pilot |= {r["cum"] for r in rs[:PILOT_PER_GAME]}
    return pilot


def build_payload(games, names, teams, rallies, windows, prefill, pilot):
    out = {"video": VIDEO_NOTE,
           "types": [[k, lab] for k, lab, _ in TYPES],
           "typedefs": {lab: d for _, lab, d in TYPES},
           "games": {}, "rallies": []}
    for slot, g in sorted(games.items()):
        t1, t2 = teams[g["match_id"]]
        out["games"][slot] = {
            "division": g["context"], "match_id": g["match_id"],
            "teams": [[{"uuid": u, "name": names.get(u, u[:8])} for u in t1],
                      [{"uuid": u, "name": names.get(u, u[:8])} for u in t2]],
        }
    for r in rallies:
        w = windows.get(r["cum"])
        if w is None:          # dropped in v4: location unresolved
            continue
        pf = prefill.get(r["cum"], {})
        out["rallies"].append({
            "cum": r["cum"], "slot": r["slot"], "rally": r["rally"],
            "t0s": round(w["t0"], 1), "t1s": round(w["t1"], 1),
            "dur": r["dur"], "approx": w["approx"],
            "score": r["score"], "outcome": r["outcome"],
            "server": names.get(r["server"], "?"), "server_uuid": r["server"],
            "receiver": names.get(r["receiver"], "?"),
            "receiver_uuid": r["receiver"],
            "core": r["cum"] in CORE, "pilot": r["cum"] in pilot,
            "pf": pf.get("shots", []), "pfnote": pf.get("note", ""),
            "pin": pf.get("pin"),
        })
    return out


HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Contact audit — Chicago 2026-07-25</title>
<style>
 :root{--bg:#101418;--panel:#1a2129;--ink:#e8edf2;--dim:#8b98a5;--acc:#4cc38a;
       --t1:#5ba8f5;--t2:#f0a35e;--warn:#e5c07b;--bad:#e06c75;--line:#2a3441}
 *{box-sizing:border-box;margin:0}
 body{background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif;
      display:grid;grid-template-columns:300px 1fr;height:100vh}
 #side{border-right:1px solid var(--line);overflow-y:auto;padding:10px}
 #right{display:flex;flex-direction:column;height:100vh;min-width:0}
 #vidwrap{padding:10px 14px 6px;border-bottom:1px solid var(--line);background:#0b0e12}
 #vid{width:100%;max-height:40vh;background:#000;border-radius:8px;display:none}
 #drop{border:2px dashed var(--line);border-radius:10px;padding:26px;text-align:center;
       color:var(--dim);cursor:pointer}
 #drop:hover{border-color:var(--acc);color:var(--ink)}
 #vbar{display:none;gap:6px;align-items:center;flex-wrap:wrap;padding-top:7px}
 #panel{overflow-y:auto;padding:14px 22px;flex:1}
 h1{font-size:17px;margin:4px 0 10px}
 .dim{color:var(--dim)} .small{font-size:12px}
 button{background:var(--panel);color:var(--ink);border:1px solid var(--line);
        border-radius:7px;padding:6px 10px;cursor:pointer;font:inherit}
 button:hover{border-color:var(--acc)}
 button.on{background:var(--acc);color:#08130d;border-color:var(--acc);font-weight:600}
 select{background:var(--panel);color:var(--ink);border:1px solid var(--line);
        border-radius:6px;padding:3px 4px;font:13px system-ui}
 .rrow{display:flex;gap:7px;align-items:center;padding:5px 7px;border-radius:7px;
       cursor:pointer;font-size:13px}
 .rrow:hover{background:var(--panel)} .rrow.sel{background:var(--panel);outline:1px solid var(--acc)}
 .badge{font-size:11px;padding:1px 6px;border-radius:9px;background:var(--line)}
 .badge.done{background:var(--acc);color:#08130d}
 .badge.core{background:var(--warn);color:#1a1408}
 .badge.pilot{background:#3b5bd0;color:#eef}
 .badge.div{background:var(--bad);color:#fff}
 .badge.whiff{background:#8b5cf6;color:#fff}
 #next{font-size:19px;font-weight:700;padding:8px 12px;border:1px solid var(--line);
       border-radius:9px;background:var(--panel)}
 #next.armed{outline:2px solid #8b5cf6}
 .shotrow{display:flex;gap:8px;align-items:center;margin:3px 0;font-size:13px;
          padding:3px 6px;border-radius:6px}
 .shotrow:nth-child(odd){background:rgba(255,255,255,.025)}
 .shotrow .idx{width:24px;text-align:right;color:var(--dim)}
 .shotrow .tt{width:74px;font-variant-numeric:tabular-nums;cursor:pointer}
 .shotrow .tt:hover{color:var(--acc)}
 .hitter.t1{color:var(--t1)} .hitter.t2{color:var(--t2)}
 .hitter{width:110px;font-weight:600}
 .stepb{padding:2px 6px;font-size:12px}
 .del{color:var(--dim);border:none;background:none;font-size:14px;cursor:pointer}
 .help{background:var(--panel);border:1px solid var(--line);border-radius:9px;
       padding:12px 14px;margin:12px 0;font-size:13px}
 kbd{background:var(--line);border-radius:4px;padding:0 5px;font-size:12px}
 input[type=text]{background:var(--panel);border:1px solid var(--line);color:var(--ink);
       border-radius:7px;padding:6px 9px;width:100%;font:inherit}
 input[type=number]{background:var(--panel);border:1px solid var(--line);color:var(--ink);
       border-radius:7px;padding:4px 6px;width:64px;font:inherit}
 .bar{display:flex;gap:8px;margin:10px 0;flex-wrap:wrap;align-items:center}
 .approx{color:var(--warn)}
 .vsep{width:1px;height:22px;background:var(--line)}
 #toast{position:fixed;bottom:18px;right:18px;background:var(--acc);color:#08130d;
        padding:9px 14px;border-radius:9px;font-weight:600;display:none;z-index:9}
 #jit{font-size:12px;color:var(--dim);margin-top:8px;border-top:1px solid var(--line);
      padding-top:8px}
</style>
<div id="side"></div>
<div id="right">
  <div id="vidwrap">
    <div id="drop">🎬 <b>Load the match video</b> — click here or drag the file in<br>
      <span class="small">__VIDEO_NOTE__<br>
      stays on your machine; nothing is uploaded</span></div>
    <input type="file" id="fpick" accept="video/*,.webm,.mp4,.mkv" hidden>
    <video id="vid" controls preload="metadata"></video>
    <div id="vbar">
      <button id="breplay" title="R">⟲ rally</button>
      <button id="bplay" title="space">⏯</button>
      <button id="bm2" title="←">−2s</button>
      <button id="bp2" title="→">+2s</button>
      <span class="vsep"></span>
      <button class="stepb" id="fm5" title="shift+,">−5f</button>
      <button class="stepb" id="fm1" title=",">−1f</button>
      <button class="stepb" id="fp1" title=".">+1f</button>
      <button class="stepb" id="fp5" title="shift+.">+5f</button>
      <span class="small dim">fps</span>
      <input type="number" id="fps" step="0.01" value="30">
      <span class="vsep"></span>
      <span class="small dim">speed</span>
      <button class="spd" data-r="0.25">.25×</button>
      <button class="spd" data-r="0.5">.5×</button>
      <button class="spd" data-r="0.75">.75×</button>
      <button class="spd on" data-r="1">1×</button>
      <span class="vsep"></span>
      <button id="bpause" class="on" title="pause automatically at the end of the current rally">auto-pause ✓</button>
      <span class="small dim">offset</span>
      <input type="number" id="voff" step="0.5" value="0" title="if every rally starts consistently early/late in YOUR file, correct it here (seconds)">
      <button id="bswap" class="small">↺ file</button>
    </div>
  </div>
  <div id="panel"></div>
</div>
<div id="toast"></div>
<script>
const DATA = __PAYLOAD__;
const LSK = "contact_audit_chicago0725";
let store = JSON.parse(localStorage.getItem(LSK) || "{}");
let prefs = JSON.parse(localStorage.getItem(LSK + "_prefs") || "{}");
let tab = "core", cur = null, whiffArmed = false;
let autoPause = prefs.autoPause !== false;

const save = () => localStorage.setItem(LSK, JSON.stringify(store));
const savePrefs = () => localStorage.setItem(LSK + "_prefs", JSON.stringify(
  {rate: V.playbackRate, voff: +el("voff").value, autoPause, fps: +el("fps").value}));
const el = id => document.getElementById(id);
const V = el("vid");
const fps = () => +el("fps").value || 30;
const voff = () => +el("voff").value || 0;
const loaded = () => !!V.src;
const rget = c => DATA.rallies.find(r => r.cum === c);
const rec = c => (store[c] = store[c] || {taps: [], note: rget(c).pfnote || ""});
const fmts = s => `${Math.floor(s/60)}:${(s%60).toFixed(2).padStart(5,"0")}`;
/* effective prefill: the old ordinal sequences carry mid-rally errors
   (pop-counting era — see contact_gate.md 2026-08-16 note), so a rally's
   prefill can be dropped once it diverges from the screen */
const pfOf = c => (store[c] && store[c].nopf) ? [] : (rget(c).pf || []);

function rallies(){
  return DATA.rallies.filter(r =>
    tab === "core" ? r.core : tab === "pilot" ? r.pilot : true);
}
function players(r){
  const g = DATA.games[r.slot];
  return g.teams[0].map(p => ({...p, team: 1}))
    .concat(g.teams[1].map(p => ({...p, team: 2})));
}
/* taps sorted by time; alignment to the prefill queue is DERIVED (k-th
   non-whiff tap <-> k-th prefilled shot) so inserts/deletes/undo never
   corrupt it. A tap's effective type = explicit user choice, else the
   aligned prefill type, else serve/return by rule on shots 1-2.
   READ-ONLY accessor: render paths must not create store entries
   (side() sweeps all 188 rallies via done()). */
function sortedTaps(c){
  const s = store[c];
  return s ? s.taps.slice().sort((a, b) => a.t - b.t) : [];
}
function rows(c){
  const out = [], pfe = pfOf(c);
  let k = 0;
  for (const tp of sortedTaps(c)){
    if (tp.w){ out.push({tp, ty: "whiff", pf: null, div: false}); continue; }
    const pf = pfe[k] || null;
    const ty = tp.ty || (pf && pf.t) || (k === 0 ? "serve" : k === 1 ? "return" : "");
    out.push({tp, ty, pf, div: !!(pf && tp.h && pf.h && tp.h !== pf.h), k});
    k++;
  }
  return out;
}
function expected(c){
  const k = sortedTaps(c).filter(t => !t.w).length;
  return pfOf(c)[k] || null;
}
function done(c){
  const rs = rows(c), pfe = pfOf(c);
  const n = rs.filter(x => !x.tp.w).length;
  const need = pfe.length ? pfe.length : 2;
  return n >= need && rs.every(x => x.ty && x.tp.h);
}
function dropPrefill(){
  if (cur === null || !rget(cur).pf.length) return;
  const s = rec(cur);
  if (s.nopf){ delete s.nopf; }
  else {
    /* materialize the types already showing, so stamped rows keep them */
    rows(cur).forEach(x => { if (!x.tp.w && !x.tp.ty && x.ty) x.tp.ty = x.ty; });
    s.nopf = 1;
  }
  save(); panel(); side();
}

/* ---------------- video ---------------- */
function loadFile(f){
  if (!f) return;
  V.src = URL.createObjectURL(f);
  V.style.display = "block"; el("drop").style.display = "none";
  el("vbar").style.display = "flex";
  V.playbackRate = prefs.rate || 1;
  markSpeed();
  if (cur !== null) openSeek(rget(cur));
}
function seekTo(t, play){
  if (!loaded()) return;
  const go = () => { V.currentTime = Math.max(0, t + voff());
                     if (play) V.play().catch(() => {}); else V.pause(); };
  V.readyState >= 1 ? go() : V.addEventListener("loadedmetadata", go, {once: true});
}
function serveStamp(c){
  const rs = rows(c).filter(x => !x.tp.w);
  return rs.length ? (rs[0].tp.tr ?? rs[0].tp.t) : null;
}
function openSeek(r){
  const sv = serveStamp(r.cum);
  if (sv != null) seekTo(sv - 2, true);
  else if (r.pin != null) seekTo(r.pin - 2, true);       // v4 hand pin
  else seekTo(r.t0s - 1, !r.approx);
}
function step(nf){ if (!loaded()) return; V.pause();
  V.currentTime = Math.max(0, V.currentTime + nf / fps()); }
function markSpeed(){
  document.querySelectorAll(".spd").forEach(b =>
    b.classList.toggle("on", +b.dataset.r === V.playbackRate));
}
function toast(msg, ms){
  const t = el("toast"); t.textContent = msg; t.style.display = "block";
  clearTimeout(t._h); t._h = setTimeout(() => t.style.display = "none", ms || 1200);
}

/* ---------------- stamping ---------------- */
function stamp(uuid, viaEnter){
  if (!loaded() || cur === null) return;
  const r = rget(cur), t = V.currentTime;
  if (whiffArmed){
    if (!uuid){ toast("whiff needs a hitter — press 1-4"); return; }
    rec(cur).taps.push({t, h: uuid, w: 1, ty: "whiff", tr: null,
                        rate: V.playbackRate});
    whiffArmed = false;
  } else {
    const exp = expected(cur);
    if (viaEnter && !exp){ toast("no prefill left — use keys 1-4"); return; }
    const h = uuid || (exp && exp.h);
    rec(cur).taps.push({t, h, w: 0, ty: null, tr: null, rate: V.playbackRate});
  }
  save();
  /* locate the row of the tap just pushed (sorted order != push order
     when a missed shot is stamped later at an earlier time) */
  const pushed = rec(cur).taps[rec(cur).taps.length - 1];
  const rs = rows(cur);
  const mine = rs.find(x => x.tp === pushed);
  const nContacts = rs.filter(x => !x.tp.w).length;
  if (!pushed.w && nContacts === 1 && r.pin != null)
    toast(`serve Δ vs pin ${(t - r.pin >= 0 ? "+" : "")}${(t - r.pin).toFixed(2)}s`, 2000);
  else if (mine && mine.div)
    toast("⚠ differs from prefilled hitter — flagged", 1600);
  panel(); side();
}
function undo(){
  if (cur === null || !rec(cur).taps.length) return;
  rec(cur).taps.pop(); save(); panel(); side(); toast("undone");
}
function refine(i, setNow){
  const tp = sortedTaps(cur)[i];
  const real = rec(cur).taps.indexOf(tp);
  if (setNow){ rec(cur).taps[real].tr = V.currentTime; save(); panel(); }
  else { seekTo((tp.tr ?? tp.t) - 0.8, true); }   // replay into the contact;
                                                  // pause + steppers to freeze it
}
function clearRefine(i){
  const tp = sortedTaps(cur)[i];
  rec(cur).taps[rec(cur).taps.indexOf(tp)].tr = null; save(); panel();
}
function setType(i, ty){
  const tp = sortedTaps(cur)[i];
  rec(cur).taps[rec(cur).taps.indexOf(tp)].ty = ty || null; save(); panel(); side();
}
function setHitter(i, h){
  const tp = sortedTaps(cur)[i];
  rec(cur).taps[rec(cur).taps.indexOf(tp)].h = h; save(); panel(); side();
}
function delTap(i){
  const tp = sortedTaps(cur)[i];
  rec(cur).taps.splice(rec(cur).taps.indexOf(tp), 1); save(); panel(); side();
}

/* ---------------- jitter vs the 16 hand-pinned serves ---------------- */
function jitter(){
  const ds = [];
  for (const r of DATA.rallies){
    if (r.pin == null || !store[r.cum]) continue;
    const sv = rows(r.cum).filter(x => !x.tp.w)[0];
    if (sv) ds.push(sv.tp.t - r.pin);        // RAW tap: measures tap noise
  }
  if (!ds.length) return null;
  const a = ds.map(Math.abs).sort((x, y) => x - y);
  const q = p => a[Math.min(a.length - 1, Math.floor(p * a.length))];
  return {n: ds.length, med: q(0.5), p95: q(0.95),
          guard: Math.max(0.4, 3 * q(0.95))};
}

/* ---------------- panels ---------------- */
function side(){
  let h = `<h1>Contact audit</h1>
  <div class="small dim">MLP Chicago 2026-07-25<br>Chicago Slice v Utah Black Diamonds</div>
  <div class="bar">
    <button class="${tab === "core" ? "on" : ""}" onclick="tab='core';side()">core 16</button>
    <button class="${tab === "pilot" ? "on" : ""}" onclick="tab='pilot';side()">pilot</button>
    <button class="${tab === "all" ? "on" : ""}" onclick="tab='all';side()">all</button>
  </div>`;
  let slot = 0;
  for (const r of rallies()){
    if (r.slot !== slot){ slot = r.slot;
      h += `<div class="small dim" style="margin-top:9px">GAME ${slot} — ${DATA.games[slot].division}</div>`; }
    const n = store[r.cum] ? rows(r.cum).length : 0;
    h += `<div class="rrow ${cur === r.cum ? "sel" : ""}" onclick="open_(${r.cum})">
      <span style="width:60px">${n ? n + " taps" : "—"}</span>
      <span>R${r.rally}</span><span class="dim small">#${r.cum}</span>
      ${r.core ? '<span class="badge core">core</span>' : ""}
      ${r.pilot ? '<span class="badge pilot">pilot</span>' : ""}
      ${done(r.cum) ? '<span class="badge done">✓</span>' : ""}</div>`;
  }
  const nd = DATA.rallies.filter(r => done(r.cum)).length;
  const ns = DATA.rallies.reduce((s, r) => s + (store[r.cum] ? rows(r.cum).length : 0), 0);
  h += `<div class="bar"><button onclick="dl()">⬇ labels CSV</button>
        <button onclick="dlMeta()">meta</button>
        <button onclick="el('csvpick').click()">⬆ import</button></div>
        <input type="file" id="csvpick" accept=".csv,text/csv" hidden>
        <div class="small dim">${nd} rallies done · ${ns} taps${lastImport}</div>`;
  const j = jitter();
  if (j) h += `<div id="jit"><b>tap jitter</b> vs ${j.n} pinned serves:<br>
    median |Δ| ${j.med.toFixed(2)}s · p95 ${j.p95.toFixed(2)}s<br>
    → training guard band ${j.guard.toFixed(2)}s</div>`;
  el("side").innerHTML = h;
  el("csvpick").onchange = e => importCSV(e.target.files[0]);
}

function open_(c){ cur = c; whiffArmed = false; side(); panel();
  openSeek(rget(c)); }

function panel(){
  const p = el("panel");
  if (cur === null){ p.innerHTML = intro(); return; }
  const r = rget(cur), ps = players(r), rs = rows(cur);
  const exp = expected(cur);
  const expName = exp ? ps.find(x => x.uuid === exp.h) : null;
  let h = `<div class="bar">
    <span class="badge">G${r.slot} R${r.rally} · #${r.cum}</span>
    <span class="badge">${DATA.games[r.slot].division}</span>
    ${r.core ? '<span class="badge core">core — prefilled</span>' : ""}
    ${r.pilot ? '<span class="badge pilot">pilot</span>' : ""}
    <span class="dim small">score ${r.score} · serve <b>${r.server}</b> → ${r.receiver}
      · ~${r.dur.toFixed(0)}s logged
      ${r.approx ? ' · <span class="approx">window approximate — trust the scorebug</span>' : ""}</span>
  </div>
  <div class="bar" style="font-size:14px">
    <span>🔎 <b>scorebug check</b>: at the serve the bug must read
    <b style="color:var(--warn)">${r.score}</b> — if it doesn't, scrub until
    it does. The score IS the rally's identity; where the prefill disagrees
    with the screen, <b>the screen wins</b> (keys 1–4).</span></div>
  <div class="bar"><span id="next" class="${whiffArmed ? "armed" : ""}">${
    whiffArmed ? "next stamp = WHIFF (press 1-4)" :
    exp ? `NEXT ⏎ #${(rs.filter(x=>!x.tp.w).length)+1}: <span class="hitter t${expName ? expName.team : 1}">${expName ? expName.name : "?"}</span> — ${exp.t || "?"}` :
    pfOf(cur).length ? "prefill complete — keys 1-4 for extras" :
    "keys 1-4 stamp the hitter"}</span>
    <button onclick="whiffArmed=!whiffArmed;panel()" title="W">${whiffArmed ? "cancel whiff" : "＋whiff (W)"}</button>
    <button onclick="undo()" title="backspace">↶ undo</button>
    ${r.pf.length ? `<button onclick="dropPrefill()">${(store[cur]&&store[cur].nopf) ? "↩ restore prefill" : "✕ prefill (wrong for this rally)"}</button>` : ""}</div>
  <div class="bar small">`;
  ps.forEach((pl, i) => {
    h += `<button class="stepb" onclick="stamp('${pl.uuid}')">
      <kbd>${i + 1}</kbd> <span class="hitter t${pl.team}">${pl.name.split(" ").slice(-1)[0]}</span></button>`;
  });
  h += `</div><div class="help small">${helprow(r)}</div>`;
  rs.forEach((x, i) => {
    const nm = ps.find(pp => pp.uuid === x.tp.h);
    h += `<div class="shotrow"><span class="idx">${i + 1}</span>
      <span class="tt" title="click to replay from just before this tap"
            onclick="refine(${i},false)">${fmts(x.tp.tr ?? x.tp.t)}${x.tp.tr != null ? "*" : ""}</span>
      <span class="hitter t${nm ? nm.team : 0}">${nm ? nm.name.split(" ").slice(-1)[0] : "?"}</span>
      <select onchange="setType(${i},this.value)">
        <option value=""></option>`;
    for (const [k2, lab] of DATA.types)
      h += `<option ${x.ty === lab ? "selected" : ""}>${lab}</option>`;
    h += `</select>
      ${x.tp.w ? '<span class="badge whiff">whiff</span>' : ""}
      ${x.div ? `<span class="badge div" title="prefill says ${(ps.find(pp=>pp.uuid===x.pf.h)||{}).name}">≠ prefill</span>
        <button class="stepb" onclick="setHitter(${i},'${x.pf.h}')">accept prefill</button>` : ""}
      <button class="stepb" onclick="refine(${i},true)" title="pause on the contact frame (use −5f/−1f/+1f/+5f), then click">set=now</button>
      ${x.tp.tr != null ? `<button class="stepb" onclick="clearRefine(${i})">✕ refined</button>` : ""}
      <button class="del" onclick="delTap(${i})">✕</button></div>`;
  });
  h += `<div class="bar"><input type="text" placeholder="rally note (optional)"
      value="${(rec(cur).note || "").replace(/"/g, "&quot;")}"
      onchange="rec(cur).note=this.value;save()"></div>
    <div class="bar">
      <button onclick="nav(-1)">← prev rally</button>
      <button onclick="nav(1)">next rally →</button></div>`;
  p.innerHTML = h;
}

function helprow(r){
  return `<b>Stamp every paddle strike AS IT HAPPENS</b> at 0.25–0.5×.
  The prefill is a CONVENIENCE, not an authority — it was coded in an era
  whose counting aid (audio pops) later proved to be noise, so mid-rally
  divergence from the screen is expected sometimes: trust your eyes,
  stamp with <kbd>1</kbd>–<kbd>4</kbd>, and the flag keeps the record.
  ${pfOf(r.cum).length ? "This rally is prefilled: <kbd>⏎</kbd> stamps the next expected shot; keys <kbd>1</kbd>–<kbd>4</kbd> stamp an explicit hitter (mismatch is flagged, not lost). At the first real divergence, hit <b>✕ prefill</b> and go keys-only."
                         : "Keys <kbd>1</kbd>–<kbd>4</kbd> stamp hitter + time; shots 1–2 auto-type serve/return, fill the rest in the table."}
  <kbd>W</kbd> arms a whiff (swing-and-miss — stamped, but contact=0, never consumes the prefill).
  <kbd>⌫</kbd> undo · <kbd>space</kbd> play/pause · <kbd>R</kbd> replay ·
  <kbd>←</kbd><kbd>→</kbd> ±2s · <kbd>,</kbd><kbd>.</kbd> ±1 frame
  (<kbd>shift</kbd> = ±5) · <kbd>[</kbd><kbd>]</kbd> speed.
  You can stamp while paused — frame-step onto the contact, then press the key.
  Refine later (pass 2): click a time to replay it, freeze the contact frame
  with the steppers, hit <b>set=now</b>.
  <span class="dim">Players: <span style="color:var(--t1)">${DATA.games[r.slot].teams[0].map(p=>p.name).join(" / ")}</span> vs
  <span style="color:var(--t2)">${DATA.games[r.slot].teams[1].map(p=>p.name).join(" / ")}</span></span>`;
}

function intro(){
  return `<h1>How this works</h1>
  <div class="help"><b>Why this exists</b>: the ordinal labels graded the old
  probe; TIMESTAMPED contacts can train a new one. Gate C (pre-registered,
  vision/contact_gate.md) needs the <b>core 16</b> re-tapped with times —
  about an evening — before anything else gets built.<br><br>
  <b>1.</b> Load the VOD above (click the box or drag the file in).<br>
  <b>2.</b> Work the <b>core 16</b> tab first. Each rally is prefilled with
  its known shot sequence: play at 0.25–0.5× and hit <kbd>⏎</kbd> on each
  contact. Your first tap of each rally is checked live against the
  hand-pinned serve, so the tool measures your own timing noise as you go.<br>
  <b>3.</b> <b>pilot</b> tab = the suggested next ~48 rallies (longest per
  game), for AFTER the ceiling test passes. Fresh rallies: keys
  <kbd>1</kbd>–<kbd>4</kbd> stamp hitter + time.<br>
  <b>4.</b> Export ⬇ and commit as
  <b>data/vision/contact_labels_chicago0725.csv</b>, then run
  <code>vision/pose_extract.py</code> and <code>vision/contact_ceiling.py</code>.<br><br>
  <b>Saving</b>: everything autosaves in this browser; the CSV is the
  deliverable AND the backup (⬆ import restores it exactly). Export at the
  end of each sitting.</div>`;
}

function nav(d){ const rs = rallies(); const i = rs.findIndex(r => r.cum === cur);
  const n = rs[i + d]; if (n) open_(n.cum); }

/* ---------------- keyboard + video wiring ---------------- */
function wireVideo(){
  el("drop").onclick = () => el("fpick").click();
  el("bswap").onclick = () => el("fpick").click();
  el("fpick").onchange = e => loadFile(e.target.files[0]);
  document.addEventListener("dragover", e => e.preventDefault());
  document.addEventListener("drop", e => { e.preventDefault();
    loadFile(e.dataTransfer.files[0]); });
  el("bplay").onclick = () => V.paused ? V.play().catch(() => {}) : V.pause();
  el("breplay").onclick = () => cur !== null && openSeek(rget(cur));
  el("bm2").onclick = () => { if (loaded()) V.currentTime -= 2; };
  el("bp2").onclick = () => { if (loaded()) V.currentTime += 2; };
  el("fm5").onclick = () => step(-5); el("fm1").onclick = () => step(-1);
  el("fp1").onclick = () => step(1);  el("fp5").onclick = () => step(5);
  document.querySelectorAll(".spd").forEach(b => b.onclick = () => {
    V.playbackRate = +b.dataset.r; markSpeed(); savePrefs(); });
  el("bpause").onclick = () => { autoPause = !autoPause;
    el("bpause").classList.toggle("on", autoPause);
    el("bpause").textContent = autoPause ? "auto-pause ✓" : "auto-pause ✗";
    savePrefs(); };
  el("voff").value = prefs.voff || 0;
  el("voff").onchange = savePrefs;
  el("fps").value = prefs.fps || 30;
  el("fps").onchange = savePrefs;
  V.addEventListener("timeupdate", () => {
    if (!autoPause || cur === null || V.paused) return;
    const r = rget(cur);
    const sv = serveStamp(cur) ?? r.pin;
    let end = sv != null ? sv + r.dur + 2 : r.t1s + 1;     // v4 window fallback
    /* the log duration includes pre-serve lead the condensed video cuts,
       so pin+dur can overrun the NEXT rally's serve — clamp to it */
    const i = DATA.rallies.findIndex(x => x.cum === r.cum);
    const nxt = DATA.rallies[i + 1];
    if (nxt){
      const nsv = serveStamp(nxt.cum) ?? nxt.pin ?? nxt.t0s;
      if (nsv != null) end = Math.min(end, nsv - 0.5);
    }
    if (V.currentTime > end) V.pause();
  });
  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || !loaded()) return;
    if (e.code === "Space"){ e.preventDefault(); el("bplay").onclick(); }
    else if (e.key === "Enter"){ e.preventDefault(); stamp(null, true); }
    else if (e.key >= "1" && e.key <= "4" && cur !== null){
      const ps = players(rget(cur));
      const pl = ps[+e.key - 1];
      if (pl) stamp(pl.uuid);
    }
    else if (e.key === "w" || e.key === "W"){ whiffArmed = !whiffArmed; panel(); }
    else if (e.key === "Backspace" || e.key === "z"){ e.preventDefault(); undo(); }
    else if (e.key === "r" || e.key === "R") el("breplay").onclick();
    else if (e.key === "ArrowLeft") el("bm2").onclick();
    else if (e.key === "ArrowRight") el("bp2").onclick();
    else if (e.key === ",") step(-1);
    else if (e.key === ".") step(1);
    else if (e.key === "<") step(-5);
    else if (e.key === ">") step(5);
    else if (e.key === "[" || e.key === "]"){
      const steps = [0.25, 0.5, 0.75, 1];
      let i = steps.indexOf(V.playbackRate); if (i < 0) i = 3;
      V.playbackRate = steps[Math.min(3, Math.max(0, i + (e.key === "]" ? 1 : -1)))];
      markSpeed(); savePrefs();
    }
  });
}

/* ---------------- export / import ---------------- */
function csv(){
  const uuid2name = {}; for (const s in DATA.games)
    for (const t of DATA.games[s].teams) for (const p of t) uuid2name[p.uuid] = p.name;
  let out = [["game","division","rally_in_game","rally_cum","shot_index",
              "hitter_name","hitter_uuid","shot_type","contact",
              "t_tap_s","t_refined_s","tap_rate","source",
              "rally_note","pin_ref_s"]];
  for (const r of DATA.rallies){
    if (!store[r.cum] || !rec(r.cum).taps.length) continue;
    rows(r.cum).forEach((x, i) => {
      const src = x.div ? "divergent" : x.tp.ty ? "manual"
                  : (x.pf ? "prefill" : "manual");
      out.push([r.slot, DATA.games[r.slot].division, r.rally, r.cum, i + 1,
                uuid2name[x.tp.h] || "", x.tp.h || "", x.ty || "",
                x.tp.w ? 0 : 1,
                x.tp.t.toFixed(3), x.tp.tr != null ? x.tp.tr.toFixed(3) : "",
                x.tp.rate || "", src,
                i === 0 ? (rec(r.cum).note || "") : "",
                i === 0 && r.pin != null ? r.pin.toFixed(2) : ""]);
    });
  }
  return out.map(r => r.map(c => `"${String(c).replace(/"/g, '""')}"`).join(",")).join("\n");
}
function dl(){ const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv()], {type: "text/csv"}));
  a.download = "contact_labels_chicago0725.csv"; a.click(); }
function dlMeta(){
  const j = jitter();
  const nd = DATA.rallies.filter(r => done(r.cum)).length;
  const meta = {tool: "contact_audit v1", exported_at: new Date().toISOString(),
                fps: fps(), rallies_done: nd, jitter: j};
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(meta, null, 1)],
                                        {type: "application/json"}));
  a.download = "contact_labels_chicago0725_meta.json"; a.click(); }

let lastImport = "";
function parseCSV(text){
  const rows2 = [[]]; let cell = "", q = false;
  for (let i = 0; i < text.length; i++){
    const c = text[i];
    if (q){
      if (c === '"' && text[i + 1] === '"'){ cell += '"'; i++; }
      else if (c === '"') q = false;
      else cell += c;
    } else if (c === '"') q = true;
    else if (c === ","){ rows2[rows2.length - 1].push(cell); cell = ""; }
    else if (c === "\n" || c === "\r"){
      if (cell !== "" || rows2[rows2.length - 1].length){
        rows2[rows2.length - 1].push(cell); cell = ""; rows2.push([]); }
    } else cell += c;
  }
  if (cell !== "" || rows2[rows2.length - 1].length) rows2[rows2.length - 1].push(cell);
  return rows2.filter(r => r.length > 1);
}
function importCSV(f){
  if (!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    const rows2 = parseCSV(rd.result);
    const head = rows2.shift().map(s => s.trim());
    const col = n => head.indexOf(n);
    if (col("rally_cum") < 0 || col("t_tap_s") < 0){
      lastImport = " — import failed: not a contact-labels CSV"; side(); return; }
    const byR = {};
    for (const r of rows2){
      const cum = +r[col("rally_cum")];
      if (!rget(cum)) continue;
      const d = byR[cum] = byR[cum] || {taps: [], note: ""};
      d.taps.push({t: +r[col("t_tap_s")], h: r[col("hitter_uuid")] || null,
                   w: +r[col("contact")] === 0 ? 1 : 0,
                   ty: r[col("shot_type")] || null,
                   tr: r[col("t_refined_s")] ? +r[col("t_refined_s")] : null,
                   rate: r[col("tap_rate")] ? +r[col("tap_rate")] : null});
      if (r[col("rally_note")]) d.note = r[col("rally_note")];
    }
    let nR = 0, nT = 0;
    for (const cum in byR){ store[cum] = byR[cum]; nR++; nT += byR[cum].taps.length; }
    save();
    lastImport = ` — imported ${nR} rallies / ${nT} taps`;
    side(); panel();
  };
  rd.readAsText(f);
}

wireVideo(); side(); panel();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", type=Path)
    ap.add_argument("--out-dir", default=None, type=Path)
    a = ap.parse_args()
    out = a.out_dir or (a.data_dir / "data/vision")
    out.mkdir(parents=True, exist_ok=True)

    games, names, teams, _cheer, rallies = load(a.data_dir)
    rallies = cumulative(games, rallies)
    windows = load_windows(a.data_dir / "data/vision/rally_windows_chicago0725_v4.csv")
    prefill = load_prefill(a.data_dir / "data/vision/shot_labels_chicago0725.csv")
    pilot = pick_pilot(rallies, windows)

    payload = build_payload(games, names, teams, rallies, windows, prefill, pilot)
    n_pf = sum(1 for r in payload["rallies"] if r["pf"])
    n_pin = sum(1 for r in payload["rallies"] if r["pin"] is not None)
    assert n_pf == len(CORE) and n_pin == len(CORE), \
        f"expected {len(CORE)} prefilled+pinned core rallies, got {n_pf}/{n_pin}"

    html = (HTML.replace("__PAYLOAD__", json.dumps(payload))
                .replace("__VIDEO_NOTE__", VIDEO_NOTE))
    p = out / "contact_audit_chicago0725.html"
    p.write_text(html)
    print(f"wrote {p} ({len(payload['rallies'])} rallies, "
          f"{n_pf} prefilled core, {len(pilot)} pilot)")


if __name__ == "__main__":
    main()
