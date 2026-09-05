"""Bounce coder — tap the bounce in every ball flight (2026-09-05).

WHY. Nobody had ever tapped a bounce in this project. The "35 human
bounces" the bound oracle grades against are the trajectory fitter's
OWN calls on the owner's clicked ball path, so both that answer key and
the 25-of-35 ceiling are model outputs. This tool records the first
independent bounce truth: for every FLIGHT (the ball's trip from one
contact tap to the next) the owner says bounce / volley / can't tell,
and for a bounce taps WHEN (the frame) and roughly WHERE (a click on
the landing spot). About ten taps a rally.

Flights come from the contact taps in contact_labels_chicago0725.csv
(manual taps where a rally has them, prefill otherwise; rally 1 keeps
its frozen state-label impacts). The last contact opens a TERMINAL
flight (to point-dead where labeled, else +2.5 s) that can hold up to
two bounces.

Rules: only YOUR clicks are ever drawn (the T trail is your own
ball-path clicks from ball_path_r{N}.csv); the fitter's bounces are
never shown, so the label cannot be led. Holdout rallies (label_split
22+) are left out unless --include-holdout; labeling them is fine
(a label is truth, not a read) — the flag just keeps the default list
on the train side.

    python3 vision/make_bounce_audit.py                  # -> data/vision/bounce_audit_chicago0725.html
    python3 vision/make_bounce_audit.py --score data/vision/bounce_labels_chicago0725.csv
    python3 vision/make_bounce_audit.py --selftest

--score compares the taps with the fitter's human-path bounces (time
within 0.30 s; landing spot in court feet through the floor
homography) and, where the shipped tracked fit is cached, with the
tracked bounces; per flight it crosses your call with the fitter's
segment kind.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "vision"
CONTACTS = DATA / "contact_labels_chicago0725.csv"
STATE = DATA / "state_labels_chicago0725.csv"
SPLIT = DATA / "label_split.csv"
OUT_HTML = DATA / "bounce_audit_chicago0725.html"
BALLSEARCH = HERE / "ballsearch"
VIDEO = "full_match.mp4.webm"
VIDEO_DUR = 4820.0

LEAD_S = 0.15      # the flight replay starts this long before contact a
TAIL_S = 0.15      # ...and auto-pauses this long after contact b
TERM_S = 2.5       # terminal flight length when no point_dead is labeled
TRAIL_S = 0.5      # T trail: your ball clicks within +/- this of the frame
MATCH_S = 0.30     # tap <-> fitter bounce time match (= br.BOUNCE_MATCH_S)
CALLS = ("bounce", "volley", "unsure", "nobounce")


# ------------------------------------------------------------- inputs

def load_split():
    return {int(r["rally_cum"]): r["split"] for r in csv.DictReader(open(SPLIT))}


def point_dead(rally):
    for r in csv.DictReader(open(STATE)):
        if int(r["rally_cum"]) == rally and r["kind"] == "point_dead":
            return float(r["t_s"])
    return None


def load_contacts():
    """{rally: [(t, hitter, contact_bool)]} — same source rule as
    make_ball_audit.load_impacts: rally 1 = frozen state-label impacts;
    otherwise manual/divergent taps when the rally has any, else prefill."""
    rows = defaultdict(list)
    for r in csv.DictReader(open(CONTACTS)):
        rows[int(r["rally_cum"])].append(r)
    out = {}
    for rally, rs in rows.items():
        if rally == 1:
            continue
        man = [r for r in rs if r["source"] in ("manual", "divergent")]
        use = man or [r for r in rs if r["source"] == "prefill"]
        cs = [(float(r["t_refined_s"] or r["t_tap_s"]),
               r["hitter_name"].replace(",", " ").strip() or "?",
               r.get("contact", "1") != "0") for r in use]
        out[rally] = (sorted(cs), "manual" if man else "prefill")
    imps = []
    for r in csv.DictReader(open(STATE)):
        if int(r["rally_cum"]) == 1 and r["kind"] == "impact":
            imps.append((float(r["t_s"]), (r.get("player") or "?").replace(",", " "), True))
    if imps:
        out[1] = (sorted(imps), "state")
    return out


def flights(contacts, dead):
    """Consecutive real contacts -> flights; whiffs (contact=0) are
    skipped (the ball flies through them); a terminal flight follows the
    last contact."""
    real = [c for c in contacts if c[2]]
    fl = []
    for a, b in zip(real, real[1:]):
        fl.append(dict(t0=round(a[0], 3), t1=round(b[0], 3), frm=a[1], to=b[1], term=False))
    if real:
        a = real[-1]
        t1 = dead if dead and dead > a[0] else a[0] + TERM_S
        fl.append(dict(t0=round(a[0], 3), t1=round(t1, 3), frm=a[1], to="", term=True))
    return fl


def load_trail(rally):
    p = DATA / f"ball_path_r{rally}.csv"
    if not p.exists():
        return []
    return [[round(float(r["t_s"]), 3), int(float(r["x"])), int(float(r["y"])), r["vis"]]
            for r in csv.DictReader(open(p)) if r["vis"] != "N" and r["x"]]


def build_cfg(include_holdout=False, only=None):
    split = load_split()
    cons = load_contacts()
    rallies = []
    for rally, (cs, src) in cons.items():
        if only and rally not in only:
            continue
        if split.get(rally, "train") != "train" and not include_holdout:
            continue
        fl = flights(cs, point_dead(rally))
        if not fl:
            continue
        rallies.append(dict(
            r=rally, split=split.get(rally, "train"), src=src,
            key=(BALLSEARCH / f"c3_cache_r{rally}.pkl").exists(),
            clicked=(DATA / f"ball_path_r{rally}.csv").exists(),
            n_contacts=sum(1 for c in cs if c[2]),
            flights=fl, trail=load_trail(rally)))
    # clicked rallies first (the ones the bounce question is about), then the rest
    rallies.sort(key=lambda d: (not d["clicked"], d["split"] != "train", d["r"]))
    return dict(rallies=rallies, lead=LEAD_S, tail=TAIL_S, trail_s=TRAIL_S,
                video=VIDEO, video_dur=VIDEO_DUR)


# --------------------------------------------------------------- score

def read_labels(path):
    """{rally: {flight_index0: {'call': str, 'b': [(t, x, y)]}}}"""
    out = defaultdict(lambda: defaultdict(lambda: {"call": "", "b": []}))
    for r in csv.DictReader(open(path)):
        e = out[int(r["rally_cum"])][int(r["flight"]) - 1]
        e["call"] = r["call"]
        if r["t_bounce_s"]:
            e["b"].append((float(r["t_bounce_s"]),
                           float(r["x_px"]) if r["x_px"] else None,
                           float(r["y_px"]) if r["y_px"] else None))
    return out


def _match(taps, ref, tol=MATCH_S):
    """Greedy nearest-time matching; returns [(tap_i, ref_j)]."""
    pairs = sorted((abs(t[0] - r[0]), i, j) for i, t in enumerate(taps)
                   for j, r in enumerate(ref) if abs(t[0] - r[0]) <= tol)
    used_i, used_j, out = set(), set(), []
    for _, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i); used_j.add(j); out.append((i, j))
    return out


def score_rows(labels, rallies=None, quiet=False):
    sys.path.insert(0, str(BALLSEARCH)); sys.path.insert(0, str(HERE))
    import numpy as np
    import ball_replicate as br                      # noqa: E402
    from claim_lab import load as c3load             # noqa: E402
    tot = defaultdict(int)
    errs, conf = [], defaultdict(int)
    say = (lambda *a: None) if quiet else print
    for rally in sorted(labels):
        if rallies and rally not in rallies:
            continue
        fl = labels[rally]
        taps = [b for e in fl.values() for b in e["b"]]
        n_calls = sum(1 for e in fl.values() if e["call"])
        cp = BALLSEARCH / f"c3_cache_r{rally}.pkl"
        if not cp.exists():
            say(f"r{rally}: {n_calls} flights coded, {len(taps)} bounce taps "
                f"— no fitter run for this rally, nothing to compare")
            tot["taps_uncompared"] += len(taps)
            continue
        c = c3load(rally)
        H = br.floor_homography(c["P"])
        fit = [(float(s["ts"]), s["bounce_xy"]) for s in c["h_segs"]
               if s and s.get("ok") and s["kind"] == "bounce"]
        pairs = _match(taps, fit)
        ft = []
        for i, j in pairs:
            t, x, y = taps[i]
            if x is None:
                continue
            v = H @ np.array([x, y, 1.0])
            X, Y = v[0] / v[2], v[1] / v[2]
            ft.append(math.hypot(X - fit[j][1][0], Y - fit[j][1][1]))
        errs += ft
        # per-flight: your call vs the fitter's segment kind on the human path
        segs = c["h_segs"]
        same_index = len(segs) == max(fl.keys(), default=-1) + 1 or len(segs) >= len(fl)
        rc = defaultdict(int)
        if same_index:
            for j, e in fl.items():
                if not e["call"] or j >= len(segs):
                    continue
                s = segs[j]
                kind = "none" if not s else ("bounce" if s.get("ok") and s["kind"] == "bounce"
                                             else ("arc" if s.get("ok") else "notok"))
                conf[(e["call"], kind)] += 1
                rc[(e["call"], kind)] += 1
        # tracked side, where cached
        tr_txt = ""
        ap = BALLSEARCH / f"autopsy_track_r{rally}.pkl"
        if ap.exists():
            import pickle
            with open(ap, "rb") as f:
                d = pickle.load(f)
            segs_t = d[1] if isinstance(d, tuple) else d
            trk = [(float(s["ts"]), None) for s in segs_t
                   if s and s.get("ok") and s["kind"] == "bounce"]
            pt = _match(taps, trk)
            tr_txt = (f"; tracked fit: {len(trk)} bounces, {len(pt)} match your taps")
            tot["trk"] += len(trk); tot["trk_match"] += len(pt)
            tot["trk_rallies"] += 1
        say(f"r{rally}: {n_calls}/{len(c['h_segs'])} flights coded, "
            f"{len(taps)} bounce taps vs {len(fit)} fitter bounces on your "
            f"path: {len(pairs)} match (±{MATCH_S}s), {len(taps)-len(pairs)} "
            f"tap-only, {len(fit)-len(pairs)} fitter-only"
            + (f"; landing error median {np.median(ft):.1f} ft (n={len(ft)})"
               if ft else "") + tr_txt)
        if rc:
            say("     call×fitter: " + ", ".join(f"{a}/{b} {n}" for (a, b), n in sorted(rc.items())))
        tot["taps"] += len(taps); tot["fit"] += len(fit); tot["match"] += len(pairs)
        tot["rallies"] += 1
    say(f"\nTOTAL over {tot['rallies']} fitter rallies: {tot['taps']} taps, "
        f"{tot['fit']} fitter bounces, {tot['match']} matched"
        + (f"; landing error median {np.median(errs):.1f} ft, "
           f"90th pct {np.percentile(errs, 90):.1f} ft (n={len(errs)})" if errs else "")
        + (f"; tracked: {tot['trk_match']}/{tot['trk']} tracked bounces match a tap "
           f"over {tot['trk_rallies']} rallies" if tot["trk"] else ""))
    if conf:
        say("call × fitter-kind (flights): "
            + ", ".join(f"{a}/{b} {n}" for (a, b), n in sorted(conf.items())))
    return dict(tot), errs, dict(conf)


# ---------------------------------------------------------------- HTML

HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>bounce coder — Chicago 07/25 game 1</title>
<style>
 body{font:14px system-ui;margin:0;background:#111;color:#ddd}
 #wrap{max-width:1040px;margin:0 auto;padding:12px}
 #vbox{position:relative}
 video{width:100%;display:block;background:#000;border-radius:6px}
 #ov{position:absolute;left:0;top:0;cursor:crosshair}
 #tl{width:100%;height:28px;display:block;border-radius:4px;cursor:pointer;margin:6px 0}
 #drop{border:2px dashed #555;border-radius:8px;padding:14px;text-align:center;
       cursor:pointer;margin:8px 0}
 .bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:6px 0}
 button{background:#2a2a2a;color:#ddd;border:1px solid #555;border-radius:5px;
        padding:4px 10px;cursor:pointer}
 kbd{background:#333;border-radius:4px;padding:1px 6px;border:1px solid #555}
 #note{color:#999;font-size:12px;line-height:1.6;margin-top:8px}
 #prog{height:10px;background:#222;border-radius:5px;overflow:hidden;flex:1}
 #progfill{height:100%;background:#4a7;width:0}
 input,select{background:#222;color:#ddd;border:1px solid #555;border-radius:4px}
 #finfo{font-size:15px;margin:4px 0}
 #call{font-weight:bold;color:#ff5}
 .fl{display:inline-block;min-width:26px;text-align:center;padding:2px 4px;margin:1px;
     border:1px solid #444;border-radius:4px;cursor:pointer;font-size:12px}
 .fl.on{border-color:#ff5;color:#ff5}
 #flash{position:fixed;top:10px;right:14px;background:#333;color:#ff5;padding:6px 12px;
        border-radius:6px;opacity:0;transition:opacity .3s;pointer-events:none}
 #vinfo{color:#888;font-size:12px}
</style></head><body><div id="wrap">
<h3>Bounce coder — tap where and when the ball bounces</h3>
<div id="drop">🎬 <b>Load the match video</b> (full_match.mp4.webm) — click or drag
<input type="file" id="fpick" accept="video/*,.webm,.mp4,.mkv" hidden></div>
<div class="bar">
 <label>rally <select id="rsel"></select></label>
 <span id="vinfo">no video</span>
 <span style="flex:1"></span>
 <label>speed <span id="rate">0.5×</span></label>
 <label>fps <input id="fps" type="number" value="30" step="0.01" style="width:60px"></label>
 <label>offset s <input id="off" type="number" value="0" step="0.01" style="width:70px"
   title="add to every label time when seeking — leave 0 for the original VOD"></label>
 <button id="bexp">⬇ export</button>
 <button id="bimp">⬆ import</button>
 <input type="file" id="csvpick" accept=".csv" hidden>
</div>
<div id="vbox"><video id="v" preload="auto"></video><canvas id="ov"></canvas></div>
<canvas id="tl"></canvas>
<div id="finfo">—</div>
<div class="bar"><span>this flight: <span id="call">·</span></span>
 <span id="time"></span><span style="flex:1"></span>
 <button id="bprev">◀ prev (P)</button><button id="bjump">↻ replay (⏎)</button>
 <button id="bnext">next ▶ (N)</button></div>
<div id="flist"></div>
<div class="bar"><span id="prog2">—</span><div id="prog"><div id="progfill"></div></div></div>
<div id="note">
<b>Each flight replays at half speed from just before one contact to just after the next
(⏎ replays it). Say what the ball did in between:</b><br>
🟡 <b>It bounced → pause on the bounce frame and click the landing spot.</b> That records
the time and the place and moves on. <kbd>B</kbd> records the time without a spot if you
can't see where it landed. A terminal flight (after the last contact) can take two taps —
come back with <kbd>P</kbd> and tap again.<br>
⚪ <b>No bounce before the next contact → <kbd>V</kbd></b> (volley).<br>
❓ <b>Can't tell → <kbd>U</kbd>.</b> An honest U is data too.<br>
✕ <b>Last flight, no bounce at all (net, caught, out of frame) → <kbd>X</kbd>.</b><br>
<b>Frame control:</b> <kbd>space</kbd> play/pause · <kbd>←</kbd>/<kbd>→</kbd> ±1 frame ·
<kbd>,</kbd>/<kbd>.</kbd> ±5 · <kbd>[</kbd>/<kbd>]</kbd> slower/faster · click the strip
to scrub · <kbd>T</kbd> shows your own ball clicks (±0.5 s) when you lose the ball ·
<kbd>⌫</kbd> clears this flight · <kbd>N</kbd>/<kbd>P</kbd> next/previous flight.<br>
<b>Accuracy bar:</b> the bounce frame ±1 and the spot within a foot or so — don't agonize.
Rallies are listed clicked-first (★ = a fitter run exists to compare with, ✎ = you clicked
its ball path); prefill rallies (r2–r5) have approximate contact times, so a flight may start
a little early or late — the bounce tap is still exact.<br>
<b>Stopping:</b> anytime — progress autosaves in this browser. At the end of every sitting
hit <b>⬇ export</b> and save as <code>bounce_labels_chicago0725.csv</code> into data/vision/
(⬆ import restores it). Score with
<code>python3 vision/make_bounce_audit.py --score data/vision/bounce_labels_chicago0725.csv</code>
</div>
<div id="flash"></div>
</div><script>
const CFG = __CFG__;
const LSK = "bounce_audit_chicago0725";
let store = {};
try { store = JSON.parse(localStorage.getItem(LSK) || "{}"); } catch (e) {}
const save = () => localStorage.setItem(LSK, JSON.stringify(store));
const $ = id => document.getElementById(id);
const V = $("v"), OV = $("ov"), TL = $("tl");
const RATES = [0.125, 0.25, 0.5, 1, 2];
let ri = Math.min(+(localStorage.getItem(LSK + "_ri") || 0), CFG.rallies.length - 1);
let fi = +(localStorage.getItem(LSK + "_fi") || 0);
let rate = +(localStorage.getItem(LSK + "_rate") || 0.5);
let trail = localStorage.getItem(LSK + "_trail") === "1";
let stopAt = null;
const fps = () => (+$("fps").value || 30);
const off = () => (+$("off").value || 0);
const R = () => CFG.rallies[ri];
const F = () => R().flights[Math.min(fi, R().flights.length - 1)];
const cur = () => V.currentTime - off();
const BADGE = {bounce: "●", volley: "○", unsure: "?", nobounce: "✕"};

function ent(create){
  const k = R().r;
  if (!store[k]) { if (!create) return null; store[k] = {}; }
  if (!store[k][fi]) { if (!create) return null; store[k][fi] = {call: "", b: []}; }
  return store[k][fi];
}
function flash(msg){
  const el = $("flash"); el.textContent = msg; el.style.opacity = 1;
  clearTimeout(el._t); el._t = setTimeout(() => el.style.opacity = 0, 1400);
}
function rallyLabel(r){
  const sr = store[r.r] || {};
  const done = r.flights.filter((f, j) => sr[j] && sr[j].call).length;
  return `r${r.r}${r.key ? " ★" : ""}${r.clicked ? " ✎" : ""}${r.split !== "train" ? " (holdout)" : ""}` +
    ` — ${r.n_contacts} contacts, ${done}/${r.flights.length} flights`;
}
function fillSelect(){
  const s = $("rsel");
  s.innerHTML = CFG.rallies.map((r, i) => `<option value="${i}">${rallyLabel(r)}</option>`).join("");
  s.value = ri;
}
$("rsel").onchange = () => goto(+$("rsel").value, 0, true);
drop.onclick = () => fpick.click();
drop.ondragover = e => e.preventDefault();
drop.ondrop = e => { e.preventDefault(); loadf(e.dataTransfer.files[0]); };
fpick.onchange = () => loadf(fpick.files[0]);
function loadf(f){
  if (!f) return;
  V.src = URL.createObjectURL(f);
  V.onloadedmetadata = () => {
    const d = V.duration;
    $("vinfo").textContent = `${f.name} · ${d.toFixed(0)} s` +
      (Math.abs(d - CFG.video_dur) > 2 ? ` ⚠ expected ${CFG.video_dur} s — wrong file, or set the offset` : " ✓");
    jump();
  };
}
function seek(t){ V.pause(); stopAt = null; V.currentTime = Math.max(0, t + off()); }
function jump(){
  if (!V.src) { render(); return; }
  const f = F();
  stopAt = f.t1 + CFG.tail;
  V.currentTime = Math.max(0, f.t0 - CFG.lead + off());
  V.playbackRate = rate; V.play();
}
function step(n){ V.pause(); stopAt = null; V.currentTime = Math.max(0, V.currentTime + n / fps()); }
function goto(r, f, play){
  ri = Math.max(0, Math.min(CFG.rallies.length - 1, r));
  fi = Math.max(0, Math.min(CFG.rallies[ri].flights.length - 1, f));
  localStorage.setItem(LSK + "_ri", ri); localStorage.setItem(LSK + "_fi", fi);
  $("rsel").value = ri;
  render();
  if (play) jump();
}
function next(){
  if (fi + 1 < R().flights.length) goto(ri, fi + 1, true);
  else if (ri + 1 < CFG.rallies.length) { goto(ri + 1, 0, true); flash("rally " + R().r); }
  else { V.pause(); flash("last flight of the list"); }
}
function prev(){
  if (fi > 0) goto(ri, fi - 1, true);
  else if (ri > 0) goto(ri - 1, CFG.rallies[ri - 1].flights.length - 1, true);
}
function record(x, y){
  if (!V.videoWidth) { flash("load the video first"); return; }
  V.pause(); stopAt = null;
  const e = ent(true), t = +cur().toFixed(3), tap = {t, x, y};
  if (e.b.length >= 2) e.b[1] = tap; else e.b.push(tap);
  e.b.sort((a, b) => a.t - b.t);
  e.call = "bounce"; save();
  flash(`bounce @ ${t.toFixed(2)} s` + (x == null ? " (no spot)" : "") +
        (e.b.length > 1 ? ` — ${e.b.length} in this flight` : ""));
  render(); next();
}
function setCall(c){
  if (!V.videoWidth) { flash("load the video first"); return; }
  const e = ent(true); e.call = c; e.b = []; save(); flash(c); render(); next();
}
function clearF(){
  const k = R().r;
  if (store[k]) { delete store[k][fi]; if (!Object.keys(store[k]).length) delete store[k]; }
  save(); flash("cleared"); render();
}
function callText(e){
  if (!e || !e.call) return "· (uncoded)";
  if (e.call === "bounce") return "● bounce " + e.b.map(b => b.t.toFixed(2) + " s" + (b.x == null ? " (no spot)" : "")).join(", ");
  return {volley: "○ volley", unsure: "? unsure", nobounce: "✕ no bounce"}[e.call] || e.call;
}
function render(){
  const r = R(), f = F(), e = ent(false);
  $("finfo").innerHTML = `rally <b>${r.r}</b>${r.key ? " ★" : ""}${r.clicked ? " ✎" : ""}` +
    ` — flight <b>${fi + 1}</b>/${r.flights.length}: ${f.frm} → ${f.term ? "point dead" : f.to}` +
    ` &nbsp; ${f.t0.toFixed(2)}–${f.t1.toFixed(2)} s (${(f.t1 - f.t0).toFixed(2)} s)` +
    (f.term ? " — <i>terminal: up to two bounces</i>" : "");
  $("call").textContent = callText(e);
  $("rate").textContent = rate + "×";
  const sr = store[r.r] || {};
  $("flist").innerHTML = r.flights.map((g, j) => {
    const ee = sr[j];
    return `<span class="fl${j === fi ? " on" : ""}" data-j="${j}">${j + 1}${ee && ee.call ? BADGE[ee.call] || "" : ""}</span>`;
  }).join("");
  let done = 0, tot = 0;
  CFG.rallies.forEach(rr => rr.flights.forEach((g, j) => { tot++; if ((store[rr.r] || {})[j]?.call) done++; }));
  $("prog2").textContent = `${done}/${tot} flights coded`;
  $("progfill").style.width = (100 * done / tot) + "%";
  fillSelect();
  draw(); drawTL();
}
$("flist").onclick = e => { const j = e.target.dataset.j; if (j != null) goto(ri, +j, true); };
function draw(){
  OV.width = V.clientWidth; OV.height = V.clientHeight;
  const c = OV.getContext("2d"); c.clearRect(0, 0, OV.width, OV.height);
  if (!V.videoWidth) return;
  const sx = OV.width / V.videoWidth, sy = OV.height / V.videoHeight, t = cur();
  if (trail) for (const p of R().trail) {
    const d = Math.abs(p[0] - t); if (d > CFG.trail_s) continue;
    c.beginPath(); c.arc(p[1] * sx, p[2] * sy, d < 0.02 ? 5 : 2.5, 0, 7);
    c.fillStyle = ({V: "#4caf50", S: "#55aacc", I: "#cc7722"}[p[3]] || "#888") + "aa"; c.fill();
  }
  const e = ent(false);
  if (e) for (const b of e.b) {
    if (b.x == null) continue;
    c.beginPath(); c.arc(b.x * sx, b.y * sy, 9, 0, 7);
    c.strokeStyle = "#ff5"; c.lineWidth = 2; c.stroke();
  }
}
function drawTL(){
  const f = F(), lo = f.t0 - CFG.lead, hi = f.t1 + CFG.tail;
  TL.width = TL.clientWidth || 800; TL.height = 28;
  const c = TL.getContext("2d"), X = t => (t - lo) / (hi - lo) * TL.width;
  c.fillStyle = "#222"; c.fillRect(0, 0, TL.width, TL.height);
  c.fillStyle = "#2b3a4a"; c.fillRect(X(f.t0), 0, X(f.t1) - X(f.t0), TL.height);
  c.fillStyle = "#8cf"; c.fillRect(X(f.t0) - 1, 0, 2, TL.height);
  if (!f.term) c.fillRect(X(f.t1) - 1, 0, 2, TL.height);
  const e = ent(false);
  if (e) { c.fillStyle = "#ff5"; for (const b of e.b) c.fillRect(X(b.t) - 2, 4, 4, TL.height - 8); }
  if (V.videoWidth) {
    const x = Math.max(0, Math.min(TL.width, X(cur())));
    c.fillStyle = "#f66"; c.fillRect(x - 1, 0, 2, TL.height);
  }
  $("time").textContent = V.videoWidth ? cur().toFixed(3) + " s" : "";
}
TL.onclick = e => {
  const f = F(), lo = f.t0 - CFG.lead, hi = f.t1 + CFG.tail;
  seek(lo + e.offsetX / TL.width * (hi - lo));
};
function tick(){
  if (!V.paused && stopAt !== null && cur() >= stopAt) { V.pause(); stopAt = null; }
  if (V.videoWidth) { draw(); drawTL(); }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
V.onseeked = () => { draw(); drawTL(); };
new ResizeObserver(draw).observe(V);
OV.onclick = e => {
  if (!V.videoWidth) { flash("load the video first"); return; }
  const x = e.offsetX * V.videoWidth / OV.width, y = e.offsetY * V.videoHeight / OV.height;
  record(Math.round(x), Math.round(y));
};
function setRate(d){
  const i = Math.max(0, Math.min(RATES.length - 1, RATES.indexOf(rate) + d));
  rate = RATES[i]; localStorage.setItem(LSK + "_rate", rate);
  V.playbackRate = rate; $("rate").textContent = rate + "×";
}
$("bprev").onclick = prev; $("bnext").onclick = next; $("bjump").onclick = jump;
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const k = e.key.toLowerCase();
  if (k === " ") { if (V.paused) { stopAt = null; V.playbackRate = rate; V.play(); } else V.pause(); }
  else if (k === "arrowright") step(1);
  else if (k === "arrowleft") step(-1);
  else if (k === ".") step(5);
  else if (k === ",") step(-5);
  else if (k === "[") setRate(-1);
  else if (k === "]") setRate(1);
  else if (k === "enter" || k === "j" || k === "home") jump();
  else if (k === "b") record(null, null);
  else if (k === "v") setCall("volley");
  else if (k === "u") setCall("unsure");
  else if (k === "x") setCall("nobounce");
  else if (k === "n") next();
  else if (k === "p") prev();
  else if (k === "t") { trail = !trail; localStorage.setItem(LSK + "_trail", trail ? "1" : "0"); draw(); flash(trail ? "trail on" : "trail off"); }
  else if (k === "backspace") clearF();
  else return;
  e.preventDefault();
});
const HEAD = "rally_cum,flight,bounce_index,t_from_s,t_to_s,hitter_from,hitter_to,call,t_bounce_s,x_px,y_px,video_name";
$("bexp").onclick = () => {
  const rows = [HEAD];
  for (const r of CFG.rallies) {
    const sr = store[r.r]; if (!sr) continue;
    r.flights.forEach((f, j) => {
      const e = sr[j]; if (!e || !e.call) return;
      const base = [r.r, j + 1], mid = [f.t0.toFixed(3), f.t1.toFixed(3), f.frm, f.term ? "" : f.to, e.call];
      if (e.b.length) e.b.forEach((b, i) => rows.push([...base, i + 1, ...mid, b.t.toFixed(3), b.x ?? "", b.y ?? "", CFG.video].join(",")));
      else rows.push([...base, 0, ...mid, "", "", "", CFG.video].join(","));
    });
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.join("\n") + "\n"], {type: "text/csv"}));
  a.download = "bounce_labels_chicago0725.csv"; a.click();
  flash(`exported ${rows.length - 1} rows`);
};
$("bimp").onclick = () => csvpick.click();
csvpick.onchange = () => {
  const rd = new FileReader();
  rd.onload = () => {
    const seen = new Set(); let n = 0;
    for (const line of rd.result.trim().split(/\r?\n/).slice(1)) {
      const c = line.split(","); if (c.length < 12) continue;
      const r = +c[0], j = +c[1] - 1, key = r + ":" + j;
      if (!store[r]) store[r] = {};
      if (!seen.has(key)) { store[r][j] = {call: c[7], b: []}; seen.add(key); }
      if (c[8]) store[r][j].b.push({t: +c[8], x: c[9] === "" ? null : +c[9], y: c[10] === "" ? null : +c[10]});
      n++;
    }
    save(); render(); flash(`imported ${n} rows`);
  };
  rd.readAsText(csvpick.files[0]);
};
fillSelect(); render();
</script></body></html>
"""


def write_html(cfg, out):
    html = HTML.replace("__CFG__", json.dumps(cfg, separators=(",", ":")))
    assert "__CFG__" not in html
    Path(out).write_text(html)
    return html


def js_check(html):
    """node --check on the page script (skipped when node is absent)."""
    js = html.split("<script>", 1)[1].split("</script>", 1)[0]
    tmp = Path("/tmp") / "bounce_audit_check.js"
    tmp.write_text(js)
    try:
        r = subprocess.run(["node", "--check", str(tmp)], capture_output=True, text=True)
    except FileNotFoundError:
        return "node not found (skipped)"
    return "js ok" if r.returncode == 0 else r.stderr


# ------------------------------------------------------------- selftest

def selftest():
    sys.path.insert(0, str(HERE))
    from make_ball_audit import load_impacts
    cfg = build_cfg(False)
    rs = [d["r"] for d in cfg["rallies"]]
    assert all(r <= 21 for r in rs), rs
    assert rs[:11] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17], rs      # clicked first
    r7 = next(d for d in cfg["rallies"] if d["r"] == 7)
    imps, dead = load_impacts(rally=7)
    assert [f["t0"] for f in r7["flights"]] == [round(t, 3) for t in imps], "r7 flights != contacts"
    assert r7["flights"][-1]["term"] and r7["flights"][-1]["t1"] == round(imps[-1] + TERM_S, 3)
    r1 = next(d for d in cfg["rallies"] if d["r"] == 1)
    assert r1["n_contacts"] == 25 and r1["flights"][-1]["t1"] == 32.066, r1["flights"][-1]
    assert r7["key"] and r7["clicked"] and len(r7["trail"]) > 100
    full = build_cfg(True)
    assert any(d["r"] >= 22 for d in full["rallies"]) and full["rallies"][-1]["split"] == "holdout"
    html = HTML.replace("__CFG__", json.dumps(cfg))
    assert "__CFG__" not in html
    jsr = js_check(html)
    assert jsr in ("js ok",) or "skipped" in jsr, jsr
    # score round-trip: the fitter's own r7 bounces, projected to pixels,
    # must come back matched with ~0 landing error
    sys.path.insert(0, str(BALLSEARCH))
    import numpy as np
    from claim_lab import load as c3load
    c = c3load(7)
    P = c["P"]
    fl = flights(load_contacts()[7][0], None)
    labels = {7: {}}
    for j, s in enumerate(c["h_segs"]):
        if s and s.get("ok") and s["kind"] == "bounce":
            bx, by = s["bounce_xy"]
            p = P @ np.array([bx, by, 0.0, 1.0])
            labels[7][j] = {"call": "bounce", "b": [(float(s["ts"]), p[0] / p[2], p[1] / p[2])]}
        elif s and s.get("ok"):
            labels[7][j] = {"call": "volley", "b": []}
    assert len(fl) == len(c["h_segs"]), (len(fl), len(c["h_segs"]))
    tot, errs, conf = score_rows(labels, quiet=True)
    assert tot["match"] == tot["fit"] == tot["taps"] == 3, tot
    assert max(errs) < 0.01, errs
    assert conf.get(("bounce", "bounce")) == 3 and ("volley", "arc") in conf, conf
    print(f"selftest OK — {len(rs)} train rallies, "
          f"{sum(len(d['flights']) for d in cfg['rallies'])} flights; {jsr}; "
          f"r7 round-trip 3/3 matched, max landing error {max(errs):.4f} ft")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", metavar="BOUNCE_CSV")
    ap.add_argument("--out", default=str(OUT_HTML))
    ap.add_argument("--include-holdout", action="store_true")
    ap.add_argument("--rallies", help="comma-separated subset")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.score:
        score_rows(read_labels(a.score))
        return
    only = [int(x) for x in a.rallies.split(",")] if a.rallies else None
    cfg = build_cfg(a.include_holdout, only)
    html = write_html(cfg, a.out)
    nf = sum(len(d["flights"]) for d in cfg["rallies"])
    print(f"wrote {a.out} — {len(cfg['rallies'])} rallies, {nf} flights "
          f"(★ fitter-comparable: {sum(d['key'] for d in cfg['rallies'])}, "
          f"✎ ball-clicked: {sum(d['clicked'] for d in cfg['rallies'])}); {js_check(html)}")


if __name__ == "__main__":
    main()
