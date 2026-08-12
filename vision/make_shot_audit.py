"""Generate the hand-coding instrument for the swing-proxy gate (Gate B).

Produces a single self-contained HTML page with an EMBEDDED VIDEO PLAYER —
the user loads their local VOD file into it (nothing is uploaded; the
browser streams the file from disk), and the page auto-seeks to each
rally, plays it at 0.25-1x, auto-pauses at the rally's end, and takes
who-hit + shot-type taps. One row of output per shot.  No timestamps are
asked of the human — order is enough, because the audio pop train carries
the timing and the label sequence aligns to it by order.  These labels
are EVALUATION-side only (model/vision_adjudication.md §Gate B): the
probe is scored against them, never tuned on them, and the page embeds no
tracker output whatsoever, so the blind rule of vision/recall_audit.md is
preserved by construction.

Inputs (all committed; --data-dir must point at a checkout that has them —
data/vision/ lives on the vision branch, PR #52):
    data/vision/rally_timeline_matchup_20260725_c4e686d1.csv   rally spine
    data/vision/rally_timeline_matchup_20260725_c4e686d1_meta.json
    data/vision/chicago0725_cheer_rally_join.json              video times
    data/players.csv                                           uuid -> name
    data/games.csv                                             team rosters

Video time per rally = cheer_video_t - duration - 2 s (the cheer marks the
rally END and peaks ~2 s before the referee's button press).  This mapping
is VALIDATED against the ten hand-scrubbed anchors in
vision/recall_audit.md before anything is written; the script refuses to
emit if the median absolute error exceeds MAX_MEDIAN_ERR_S.  Rallies the
cheer alignment could not match (13 of 193) get a time carried forward
from the previous matched rally and are flagged "~approx" — usable for
scrubbing, excluded from the core set.  A global offset field in the page
covers any residual drift of the user's local file.

Core set = the ten original blind-audit rallies (kept comparable) plus the
longest remaining matched rallies per game, five per game, twenty total —
long rallies are where the speed-up exchanges live, which is the stratum
the gate's kill threshold reads.

    python vision/make_shot_audit.py --data-dir <checkout-with-data/vision>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MATCHUP = "rally_timeline_matchup_20260725_c4e686d1"
VIDEO_NOTE = "full_match.mp4 / full_match.mp4.webm (MLP Chicago 2026-07-25, Chicago Slice v Utah Black Diamonds)"
CHEER_LEAD_S = 2.0          # cheer peaks ~2 s before the ref's end press
MAX_MEDIAN_ERR_S = 6.0      # refuse to ship a mapping worse than this
CORE_PER_GAME = 5

# The ten pre-registered blind-audit rallies (vision/recall_audit.md),
# cumulative numbering, with the hand-scrubbed start times that validate
# the cheer->video mapping.
ORIG_AUDIT = {2: "0:14", 8: "3:37", 18: "8:07", 40: "16:44", 43: "19:00",
              79: "32:34", 81: "33:45", 140: "59:00", 173: "70:22",
              186: "76:02"}

SHOT_TYPES = [  # (key, label, definition shown in the help panel)
    ("s", "serve", "the serve (prefilled on shot 1)"),
    ("r", "return", "return of serve (prefilled on shot 2)"),
    ("v", "drive", "flat/hard groundstroke or volley hit at pace from mid/deep court"),
    ("p", "drop", "soft third-shot-style drop into the kitchen from mid/deep court"),
    ("d", "dink", "soft kitchen-line exchange ball"),
    ("u", "speed-up", "the ATTACK: a dink-height ball deliberately flicked/rolled at pace"),
    ("c", "counter", "fast reflex reply during a hands battle (incl. blocks/resets under fire)"),
    ("l", "lob", "lifted over the opponents"),
    ("m", "smash", "overhead putaway"),
    ("o", "other", "anything else / can't tell (use the note)"),
]


def mmss(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 60}:{t % 60:02d}"


def parse_mmss(s: str) -> int:
    m, ss = s.split(":")
    return int(m) * 60 + int(ss)


def wall_s(iso: str) -> float:
    """Seconds since midnight UTC — fine for one afternoon of matches."""
    hh, mm, ss = iso.split("T")[1].split("+")[0].split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def load(data_dir: Path):
    meta = json.load(open(data_dir / "data/vision" / f"{MATCHUP}_meta.json"))
    games = {int(g["slot"]): g for g in meta["games"]}

    names = {}
    for r in csv.DictReader(open(data_dir / "data/players.csv")):
        names[r["player_id"].lower()] = r["full_name"]

    teams = {}  # match_id -> ([t1 uuids], [t2 uuids])
    for r in csv.DictReader(open(data_dir / "data/games.csv")):
        if r["match_id"] in {g["match_id"] for g in games.values()}:
            teams[r["match_id"]] = ([r["t1_p1"].lower(), r["t1_p2"].lower()],
                                    [r["t2_p1"].lower(), r["t2_p2"].lower()])

    cheer = {}
    for e in json.load(open(data_dir / "data/vision/chicago0725_cheer_rally_join.json")):
        cheer[(int(e["slot"]), int(e["rally"]))] = float(e["cheer_video_t"])

    rallies = []
    for r in csv.DictReader(open(data_dir / "data/vision" / f"{MATCHUP}.csv")):
        rallies.append({
            "slot": int(r["slot"]), "rally": int(r["rally"]),
            "match_id": r["match_id"],
            "dur": float(r["duration_s"]),
            "wall_t0": wall_s(r["t_start"]), "wall_t1": wall_s(r["t_end"]),
            "score": r["start_score"], "outcome": r["outcome"],
            "server": r["server_uuid"].lower(),
            "receiver": r["receiver_uuid"].lower(),
        })
    rallies.sort(key=lambda x: (x["slot"], x["rally"]))
    return games, names, teams, cheer, rallies


def video_times(rallies, cheer):
    """start/end in video seconds; carry an offset forward over unmatched."""
    last = {}  # slot -> (wall_t1, video_end) of last MATCHED rally
    for r in rallies:
        key = (r["slot"], r["rally"])
        if key in cheer:
            end = cheer[key] - CHEER_LEAD_S
            r["v0"], r["v1"], r["approx"] = end - r["dur"], end, False
            last[r["slot"]] = (r["wall_t1"], end)
        elif r["slot"] in last:
            w1, v1 = last[r["slot"]]
            end = v1 + (r["wall_t1"] - w1)      # drifts across cuts: flagged
            r["v0"], r["v1"], r["approx"] = end - r["dur"], end, True
        else:
            r["v0"], r["v1"], r["approx"] = None, None, True
    return rallies


def cumulative(games, rallies):
    """Both committed files number rallies CUMULATIVELY across the matchup
    (1-193); derive the within-game number from the slot offsets."""
    off, cum = {}, 0
    for slot in sorted(games):
        off[slot] = cum
        cum += games[slot]["n_rallies"]
    for r in rallies:
        r["cum"] = r["rally"]
        r["rally"] = r["cum"] - off[r["slot"]]
        assert 1 <= r["rally"] <= games[r["slot"]]["n_rallies"]
    return rallies


def validate(rallies):
    print("cheer->video mapping vs the 10 hand-scrubbed anchors:")
    errs = []
    for r in rallies:
        if r["cum"] in ORIG_AUDIT and r["v0"] is not None:
            want = parse_mmss(ORIG_AUDIT[r["cum"]])
            err = r["v0"] - want
            errs.append(abs(err))
            print(f"  #{r['cum']:>3} (G{r['slot']} R{r['rally']:>2})  "
                  f"derived {mmss(r['v0'])}  audit {ORIG_AUDIT[r['cum']]}  "
                  f"err {err:+.0f}s")
    med = sorted(errs)[len(errs) // 2]
    print(f"  median |err| = {med:.0f}s over {len(errs)} anchors")
    if len(errs) < len(ORIG_AUDIT) or med > MAX_MEDIAN_ERR_S:
        raise SystemExit("mapping failed validation — not writing outputs")


def pick_core(rallies):
    core = {r["cum"] for r in rallies if r["cum"] in ORIG_AUDIT}
    by_slot = {}
    for r in rallies:
        by_slot.setdefault(r["slot"], []).append(r)
    for slot, rs in by_slot.items():
        have = sum(1 for r in rs if r["cum"] in core)
        pool = [r for r in rs if r["cum"] not in core and not r["approx"]]
        pool.sort(key=lambda x: (-x["dur"], x["rally"]))
        core |= {r["cum"] for r in pool[:max(0, CORE_PER_GAME - have)]}
    return core


def build_payload(games, names, teams, rallies, core):
    out = {"video": VIDEO_NOTE, "types": [[k, lab] for k, lab, _ in SHOT_TYPES],
           "typedefs": {lab: d for _, lab, d in SHOT_TYPES}, "games": {}, "rallies": []}
    for slot, g in sorted(games.items()):
        t1, t2 = teams[g["match_id"]]
        out["games"][slot] = {
            "division": g["context"], "match_id": g["match_id"],
            "teams": [[{ "uuid": u, "name": names.get(u, u[:8])} for u in t1],
                      [{"uuid": u, "name": names.get(u, u[:8])} for u in t2]],
        }
    for r in rallies:
        if r["v0"] is None:
            continue
        out["rallies"].append({
            "cum": r["cum"], "slot": r["slot"], "rally": r["rally"],
            "t0": mmss(r["v0"]), "t1": mmss(r["v1"]),
            "t0s": round(r["v0"], 1), "t1s": round(r["v1"], 1),
            "dur": r["dur"],
            "approx": r["approx"], "score": r["score"], "outcome": r["outcome"],
            "server": names.get(r["server"], "?"), "server_uuid": r["server"],
            "receiver": names.get(r["receiver"], "?"),
            "receiver_uuid": r["receiver"],
            "core": r["cum"] in core, "orig": r["cum"] in ORIG_AUDIT,
        })
    return out


def write_template(payload, path: Path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game", "division", "rally_in_game", "rally_cum",
                    "scrub_to", "shot_index", "hitter_name", "shot_type",
                    "note"])
        for r in payload["rallies"]:
            if not r["core"]:
                continue
            div = payload["games"][r["slot"]]["division"]
            w.writerow([r["slot"], div, r["rally"], r["cum"], r["t0"],
                        1, r["server"], "serve", ""])
            w.writerow([r["slot"], div, r["rally"], r["cum"], r["t0"],
                        2, r["receiver"], "return", ""])
            for i in range(3, 17):
                w.writerow([r["slot"], div, r["rally"], r["cum"], r["t0"],
                            i, "", "", ""])
    print(f"wrote {path}")


# The <video> element lives OUTSIDE the re-rendered panel on purpose:
# every label tap re-renders the shot panel, and a video inside it would
# be torn down and lose the loaded file + playback position on each tap.
HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>Shot audit — Chicago 2026-07-25</title>
<style>
 :root{--bg:#101418;--panel:#1a2129;--ink:#e8edf2;--dim:#8b98a5;--acc:#4cc38a;
       --t1:#5ba8f5;--t2:#f0a35e;--warn:#e5c07b;--line:#2a3441}
 *{box-sizing:border-box;margin:0}
 body{background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif;
      display:grid;grid-template-columns:290px 1fr;height:100vh}
 #side{border-right:1px solid var(--line);overflow-y:auto;padding:10px}
 #right{display:flex;flex-direction:column;height:100vh;min-width:0}
 #vidwrap{padding:10px 14px 6px;border-bottom:1px solid var(--line);background:#0b0e12}
 #vid{width:100%;max-height:42vh;background:#000;border-radius:8px;display:none}
 #drop{border:2px dashed var(--line);border-radius:10px;padding:26px;text-align:center;
       color:var(--dim);cursor:pointer}
 #drop:hover{border-color:var(--acc);color:var(--ink)}
 #vbar{display:flex;gap:6px;align-items:center;flex-wrap:wrap;padding-top:7px;display:none}
 #panel{overflow-y:auto;padding:14px 22px;flex:1}
 h1{font-size:17px;margin:4px 0 10px}
 .dim{color:var(--dim)} .small{font-size:12px}
 button{background:var(--panel);color:var(--ink);border:1px solid var(--line);
        border-radius:7px;padding:6px 10px;cursor:pointer;font:inherit}
 button:hover{border-color:var(--acc)}
 button.on{background:var(--acc);color:#08130d;border-color:var(--acc);font-weight:600}
 .rrow{display:flex;gap:8px;align-items:center;padding:5px 7px;border-radius:7px;
       cursor:pointer;font-size:13px}
 .rrow:hover{background:var(--panel)} .rrow.sel{background:var(--panel);outline:1px solid var(--acc)}
 .badge{font-size:11px;padding:1px 6px;border-radius:9px;background:var(--line)}
 .badge.done{background:var(--acc);color:#08130d}
 .badge.orig{background:var(--warn);color:#1a1408}
 #scrub{font-size:26px;font-weight:700;letter-spacing:1px;cursor:pointer}
 #scrub:hover{color:var(--acc)}
 .shot{display:flex;gap:6px;align-items:center;margin:5px 0;flex-wrap:wrap}
 .shot .idx{width:26px;text-align:right;color:var(--dim)}
 .pbtn{border-width:2px}
 .pbtn.t1{border-color:var(--t1)} .pbtn.t2{border-color:var(--t2)}
 .pbtn.t1.on{background:var(--t1);border-color:var(--t1);color:#06121f}
 .pbtn.t2.on{background:var(--t2);border-color:var(--t2);color:#1f1206}
 .tbtn{font-size:12px;padding:5px 7px}
 .del{color:var(--dim);border:none;background:none;font-size:15px}
 .help{background:var(--panel);border:1px solid var(--line);border-radius:9px;
       padding:12px 14px;margin:12px 0;font-size:13px}
 kbd{background:var(--line);border-radius:4px;padding:0 5px;font-size:12px}
 input[type=text]{background:var(--panel);border:1px solid var(--line);color:var(--ink);
       border-radius:7px;padding:6px 9px;width:100%;font:inherit}
 input[type=number]{background:var(--panel);border:1px solid var(--line);color:var(--ink);
       border-radius:7px;padding:4px 6px;width:64px;font:inherit}
 .bar{display:flex;gap:8px;margin:10px 0;flex-wrap:wrap;align-items:center}
 a{color:var(--acc)}
 .approx{color:var(--warn)}
 .vsep{width:1px;height:22px;background:var(--line)}
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
<script>
const DATA = __PAYLOAD__;
const LSK = "shot_audit_chicago0725";
let store = JSON.parse(localStorage.getItem(LSK) || "{}");
let prefs = JSON.parse(localStorage.getItem(LSK + "_prefs") || "{}");
let coreOnly = true, cur = null, autoPause = prefs.autoPause !== false;

const save = () => localStorage.setItem(LSK, JSON.stringify(store));
const savePrefs = () => localStorage.setItem(LSK + "_prefs",
  JSON.stringify({rate: V.playbackRate, voff: +el("voff").value, autoPause}));
const el = id => document.getElementById(id);
const V = el("vid");
const rallies = () => DATA.rallies.filter(r => !coreOnly || r.core);
const rget = c => DATA.rallies.find(r => r.cum === c);
const shots = c => (store[c] = store[c] || {shots: prefill(c), note: ""}).shots;
const voff = () => +el("voff").value || 0;
const loaded = () => !!V.src;

function prefill(c){
  const r = rget(c);
  return [{h: r.server_uuid, t: "serve"}, {h: r.receiver_uuid, t: "return"}];
}
function players(r){
  const g = DATA.games[r.slot];
  return g.teams[0].map(p => ({...p, team: 1})).concat(g.teams[1].map(p => ({...p, team: 2})));
}
function done(c){ const s = store[c]; return s && s.shots.length > 2 &&
  s.shots.every(x => x.h && x.t); }

/* ---------------- video ---------------- */
function loadFile(f){
  if (!f) return;
  V.src = URL.createObjectURL(f);
  V.style.display = "block"; el("drop").style.display = "none";
  el("vbar").style.display = "flex";
  V.playbackRate = prefs.rate || 1;
  markSpeed();
  if (cur !== null) seekRally(rget(cur), false);
}
function seekTo(t, play){
  if (!loaded()) return;
  const go = () => { V.currentTime = Math.max(0, t + voff());
                     if (play) V.play().catch(() => {}); };
  V.readyState >= 1 ? go() : V.addEventListener("loadedmetadata", go, {once: true});
}
function seekRally(r, play){ seekTo(r.t0s - 1.0, play); }
function markSpeed(){
  document.querySelectorAll(".spd").forEach(b =>
    b.classList.toggle("on", +b.dataset.r === V.playbackRate));
}
function wireVideo(){
  el("drop").onclick = () => el("fpick").click();
  el("bswap").onclick = () => el("fpick").click();
  el("fpick").onchange = e => loadFile(e.target.files[0]);
  document.addEventListener("dragover", e => e.preventDefault());
  document.addEventListener("drop", e => { e.preventDefault();
    loadFile(e.dataTransfer.files[0]); });
  el("bplay").onclick = () => V.paused ? V.play().catch(() => {}) : V.pause();
  el("breplay").onclick = () => cur !== null && seekRally(rget(cur), true);
  el("bm2").onclick = () => { if (loaded()) V.currentTime -= 2; };
  el("bp2").onclick = () => { if (loaded()) V.currentTime += 2; };
  document.querySelectorAll(".spd").forEach(b => b.onclick = () => {
    V.playbackRate = +b.dataset.r; markSpeed(); savePrefs(); });
  el("bpause").onclick = () => { autoPause = !autoPause;
    el("bpause").classList.toggle("on", autoPause);
    el("bpause").textContent = autoPause ? "auto-pause ✓" : "auto-pause ✗";
    savePrefs(); };
  el("voff").value = prefs.voff || 0;
  el("voff").onchange = savePrefs;
  V.addEventListener("timeupdate", () => {
    if (autoPause && cur !== null && !V.paused &&
        V.currentTime > rget(cur).t1s + voff() + 1.5) V.pause();
  });
  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || !loaded()) return;
    if (e.code === "Space"){ e.preventDefault(); el("bplay").onclick(); }
    else if (e.key === "r" || e.key === "R") el("breplay").onclick();
    else if (e.key === "ArrowLeft") el("bm2").onclick();
    else if (e.key === "ArrowRight") el("bp2").onclick();
    else if (e.key === "[" || e.key === "]"){
      const steps = [0.25, 0.5, 0.75, 1];
      let i = steps.indexOf(V.playbackRate); if (i < 0) i = 3;
      V.playbackRate = steps[Math.min(3, Math.max(0, i + (e.key === "]" ? 1 : -1)))];
      markSpeed(); savePrefs();
    }
  });
}

/* ---------------- panels ---------------- */
function side(){
  let h = `<h1>Shot audit</h1>
  <div class="small dim">MLP Chicago 2026-07-25<br>Chicago Slice v Utah Black Diamonds</div>
  <div class="bar">
    <button class="${coreOnly ? "on" : ""}" onclick="coreOnly=true;side()">core 20</button>
    <button class="${coreOnly ? "" : "on"}" onclick="coreOnly=false;side()">all rallies</button>
  </div>`;
  let slot = 0;
  for (const r of rallies()){
    if (r.slot !== slot){ slot = r.slot;
      h += `<div class="small dim" style="margin-top:9px">GAME ${slot} — ${DATA.games[slot].division}</div>`; }
    h += `<div class="rrow ${cur === r.cum ? "sel" : ""}" onclick="open_(${r.cum})">
      <span style="width:74px">${r.approx ? "≈" : ""}${r.t0}</span>
      <span>R${r.rally}</span><span class="dim small">#${r.cum}</span>
      ${r.orig ? '<span class="badge orig">blind10</span>' : ""}
      ${done(r.cum) ? '<span class="badge done">✓</span>' : ""}</div>`;
  }
  const n = DATA.rallies.filter(r => done(r.cum)).length;
  h += `<div class="bar"><button onclick="dl()">⬇ download CSV</button>
        <button onclick="cp()">copy CSV</button></div>
        <div class="small dim">${n} rallies coded</div>`;
  el("side").innerHTML = h;
}

function open_(c){
  cur = c; side(); panel();
  seekRally(rget(c), true);
}

function panel(){
  const p = el("panel");
  if (cur === null){ p.innerHTML = intro(); return; }
  const r = rget(cur), ps = players(r), sh = shots(cur);
  let h = `<div class="bar"><span id="scrub" onclick="seekRally(rget(cur),true)"
      title="click to replay this rally">${r.approx ? "≈" : ""}${r.t0}</span>
    <span class="dim">→ ends ~${r.t1} (${r.dur.toFixed(0)}s)</span>
    <span class="badge">G${r.slot} R${r.rally} · #${r.cum}</span>
    <span class="badge">${DATA.games[r.slot].division}</span>
    ${r.orig ? '<span class="badge orig">original blind-10</span>' : ""}</div>
    <div class="small dim">score ${r.score} · serve: <b>${r.server}</b> → ${r.receiver}
    ${r.approx ? ' · <span class="approx">time approximated across a broadcast cut — trust the scorebug</span>' : ""}</div>
    <div class="help small">${helprow(r)}</div>`;
  sh.forEach((s, i) => {
    h += `<div class="shot"><span class="idx">${i + 1}</span>`;
    for (const p2 of ps)
      h += `<button class="pbtn t${p2.team} ${s.h === p2.uuid ? "on" : ""}"
        onclick="setH(${i},'${p2.uuid}')">${p2.name.split(" ").slice(-1)[0]}</button>`;
    h += `<span style="width:8px"></span>`;
    for (const [k, lab] of DATA.types)
      h += `<button class="tbtn ${s.t === lab ? "on" : ""}"
        onclick="setT(${i},'${lab}')">${lab}</button>`;
    h += `<button class="del" onclick="delS(${i})">✕</button></div>`;
  });
  h += `<div class="bar"><button onclick="addS()">+ shot</button>
    <span class="dim small">first two prefilled from the referee log — fix them if the video disagrees</span></div>
    <div class="bar"><input type="text" placeholder="rally note (optional — e.g. 'fake at shot 5', 'both lunged')"
      value="${(store[cur].note || "").replace(/"/g, "&quot;")}"
      onchange="store[cur].note=this.value;save()"></div>
    <div class="bar">
      <button onclick="nav(-1)">← prev rally</button>
      <button onclick="nav(1)">next rally →</button></div>`;
  p.innerHTML = h;
}

function helprow(r){
  const t = Object.entries(DATA.typedefs).map(([k, v]) => `<b>${k}</b> — ${v}`).join("<br>");
  return `<b>Count every PADDLE STRIKE in order</b> (serve included; bounces are not shots;
  a fake or swing-and-miss is NOT a shot — note it instead). If the rally ends with a ball
  nobody touches, the last shot is the previous strike.
  Keys: <kbd>space</kbd> play/pause · <kbd>R</kbd> replay rally · <kbd>←</kbd><kbd>→</kbd> ±2s ·
  <kbd>[</kbd><kbd>]</kbd> slower/faster. Tip: 0.5× with sound on makes the pops easy to count.<br><br>${t}<br><br>
  <span class="dim">Blind rule: don't look at any tracker output for these rallies.
  Players: <span style="color:var(--t1)">${DATA.games[r.slot].teams[0].map(p=>p.name).join(" / ")}</span> vs
  <span style="color:var(--t2)">${DATA.games[r.slot].teams[1].map(p=>p.name).join(" / ")}</span></span>`;
}

function intro(){
  return `<h1>How this works</h1>
  <div class="help"><b>1.</b> Load the VOD above (click the box or drag the file in).
  It plays inside this page — one window, no juggling.<br>
  <b>2.</b> Pick a rally on the left (core 20 first — the ten marked
  <span class="badge orig">blind10</span> are the pre-registered audit set).
  The video jumps there and plays; it auto-pauses when the rally ends.<br>
  <b>3.</b> Tap who hit + what it was for each shot in order. Shots 1–2 are
  prefilled from the referee log. <kbd>R</kbd> replays the rally; drop to
  0.5× or 0.25× for the hands battles. Everything autosaves locally.<br>
  <b>4.</b> Download the CSV when done (or partway — partial coverage is fine)
  and drop it in the repo as <b>data/vision/shot_labels_chicago0725.csv</b>.<br><br>
  No timestamps needed from you — the order is enough; audio carries the timing.<br>
  <span class="small dim">If every rally starts consistently early/late in your
  file, set the offset (seconds) in the bar above once.</span></div>`;
}

function setH(i, u){ shots(cur)[i].h = u; save(); panel(); side(); }
function setT(i, t){ shots(cur)[i].t = t; save(); panel(); side(); }
function addS(){ shots(cur).push({h: null, t: null}); save(); panel(); }
function delS(i){ shots(cur).splice(i, 1); save(); panel(); side(); }
function nav(d){ const rs = rallies(); const i = rs.findIndex(r => r.cum === cur);
  const n = rs[i + d]; if (n) open_(n.cum); }

function csv(){
  const uuid2name = {}; for (const s in DATA.games)
    for (const t of DATA.games[s].teams) for (const p of t) uuid2name[p.uuid] = p.name;
  let rows = [["game","division","rally_in_game","rally_cum","shot_index",
               "hitter_name","hitter_uuid","shot_type","rally_note"]];
  for (const r of DATA.rallies){
    const s = store[r.cum]; if (!s) continue;
    s.shots.forEach((x, i) => { if (x.h || x.t)
      rows.push([r.slot, DATA.games[r.slot].division, r.rally, r.cum, i + 1,
                 uuid2name[x.h] || "", x.h || "", x.t || "",
                 i === 0 ? (s.note || "") : ""]); });
  }
  return rows.map(r => r.map(c => `"${String(c).replace(/"/g, '""')}"`).join(",")).join("\n");
}
function dl(){ const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv()], {type: "text/csv"}));
  a.download = "shot_labels_chicago0725.csv"; a.click(); }
function cp(){ navigator.clipboard.writeText(csv()); }

wireVideo(); side(); panel();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", type=Path,
                    help="checkout containing data/vision (the vision branch)")
    ap.add_argument("--out-dir", default=None, type=Path,
                    help="where to write outputs (default <data-dir>/data/vision)")
    a = ap.parse_args()
    out = a.out_dir or (a.data_dir / "data/vision")
    out.mkdir(parents=True, exist_ok=True)

    games, names, teams, cheer, rallies = load(a.data_dir)
    assert max(r["rally"] for r in rallies) == sum(
        g["n_rallies"] for g in games.values()), "expected cumulative numbering"

    rallies = cumulative(games, video_times(rallies, cheer))
    validate(rallies)
    core = pick_core(rallies)
    payload = build_payload(games, names, teams, rallies, core)

    html = (HTML.replace("__PAYLOAD__", json.dumps(payload))
                .replace("__VIDEO_NOTE__", VIDEO_NOTE))
    p = out / "shot_audit_chicago0725.html"
    p.write_text(html)
    print(f"wrote {p} ({len(payload['rallies'])} rallies, "
          f"{sum(1 for r in payload['rallies'] if r['core'])} core)")
    write_template(payload, out / "shot_audit_template.csv")


if __name__ == "__main__":
    main()
