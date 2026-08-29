"""State audit — the gold-subset labeling tool for the hitter-episode
pilot (user go, 2026-08-29; design in swing_explore_notes.md same day).

The user's "fanciest model" reads contacts off HANDOFFS of a per-player
"looks like a hitter" state. The pilot measures the shape of that state
on ~10 TRAIN rallies before anything trains on it: when the hitter-look
starts relative to impact, when the follow-through ends, the user's own
tap jitter (frame-exact impact vs the existing tap), and how often a
player looks like a hitter WITHOUT striking (the hard-negative class a
handoff decoder must survive).

Coding scheme — step every frame, tap only BOUNDARIES (the interval
inherits, so the derived per-frame state set is complete at ~10x fewer
judgments than literal per-frame coding):

  pick the player (1-4), then per episode
    B = episode start   (first frame they read as "about to hit")
    I = impact frame    (frame-exact; the existing contact tap is shown
                         as a gray tick ~nearby, deliberately NOT
                         auto-snapped — find the frame yourself)
    E = episode end     (follow-through done, back to neutral)
    X = toggle NO-CONTACT on the episode (looked like a hitter, never
        struck — fakes, both-went-for-it, aborted swings). An episode
        without I and without X is treated as unfinished, not negative.

  plus two RALLY-level marks (user addition, 2026-08-29):
    R = service routine starts (server begins the pre-serve ritual —
        NOT the serve wind-up; the server's own B still marks that).
        The decoder's opening anchor state, and the video-truth side
        of the referee log's ~6 s pre-serve lead.
    D = point dead (ball down / rally over). Video-truth rally end,
        against the log end times the temporal model is registered to
        use; bounds every episode.

Pilot rallies (pre-specified, train only, quarantine r9/r10 excluded,
fast-heavy r1/r3 included): 1-8, 13, 14 — 124 contacts.

Export lands as data/vision/state_labels_chicago0725.csv
(rally_cum,player,episode,kind,t_s). Same hygiene as every audit tool:
export at the end of every sitting; localStorage is the working copy.

GATE NOTE: these are new label types. Before any temporal-model code
exists, a dated amendment on temporal_gate.md must record them as
permitted T inputs (draft in swing_explore_notes.md 2026-08-29,
awaiting the user's explicit freeze).
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
                    "contacts": contacts})
    return out


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>state audit — hitter episodes (pilot)</title>
<style>
 body{font:14px system-ui;margin:0;background:#111;color:#ddd}
 #wrap{max-width:1020px;margin:0 auto;padding:12px}
 video{width:100%;background:#000;border-radius:6px}
 #drop{border:2px dashed #555;border-radius:8px;padding:20px;text-align:center;
       cursor:pointer;margin:8px 0}
 .bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:6px 0}
 button{background:#2a2a2a;color:#ddd;border:1px solid #555;border-radius:5px;
        padding:4px 10px;cursor:pointer}
 button.sel{background:#365;border-color:#7c7}
 kbd{background:#333;border-radius:4px;padding:1px 6px;border:1px solid #555}
 .lane{position:relative;height:26px;background:#1a1a1a;border-radius:4px;
       margin:3px 0}
 .lane .nm{position:absolute;left:6px;top:4px;font-size:12px;color:#888;
           pointer-events:none}
 .ep{position:absolute;top:3px;height:20px;background:#2d4d2d;
     border:1px solid #4a7;border-radius:3px;cursor:pointer}
 .ep.nc{background:#4d3d2d;border-color:#a74}
 .ep.open{border-style:dashed}
 .imp{position:absolute;top:0;width:2px;height:26px;background:#8f8}
 .tick{position:absolute;top:20px;width:1px;height:6px;background:#666}
 .tick.f{background:#c66}
 .cursor{position:absolute;top:0;width:1px;height:26px;background:#8cf}
 .rmark{position:absolute;top:0;width:2px;height:26px;background:#d93}
 .dmark{position:absolute;top:0;width:2px;height:26px;background:#d44}
 #note{color:#999;font-size:12px;line-height:1.5;margin-top:8px}
 select,input{background:#222;color:#ddd;border:1px solid #555;border-radius:4px}
</style></head><body><div id="wrap">
<h3>State audit — hitter episodes (pilot, 10 train rallies)</h3>
<div id="drop">🎬 <b>Load the match video</b> (full_match.mp4.webm) — click or drag
<input type="file" id="fpick" accept="video/*,.webm,.mp4,.mkv" hidden></div>
<video id="v" controls preload="auto"></video>
<div class="bar">
 <label>rally <select id="rsel"></select></label>
 <label>fps <input id="fps" type="number" value="30" step="0.01" style="width:64px"></label>
 <span id="pbtns"></span>
 <span style="flex:1"></span>
 <span id="status">—</span>
 <button id="bexp">⬇ export CSV</button>
 <button id="bimp">⬆ import</button>
 <input type="file" id="csvpick" accept=".csv" hidden>
</div>
<div id="lanes"></div>
<div id="note">
<kbd>←</kbd>/<kbd>→</kbd> ±1 frame · <kbd>,</kbd>/<kbd>.</kbd> ±10 ·
<kbd>space</kbd> play/pause · <kbd>1</kbd>–<kbd>4</kbd> pick player ·
<kbd>B</kbd> episode start · <kbd>I</kbd> impact frame ·
<kbd>E</kbd> episode end · <kbd>X</kbd> toggle no-contact ·
<kbd>R</kbd> service routine starts · <kbd>D</kbd> point dead
(rally-level, orange/red lines) ·
<kbd>⌫</kbd> delete selected player's nearest mark.
Gray ticks under each lane = your existing contact taps (red = fast) —
navigation aid only; find the exact impact frame yourself, don't snap
to the tick. An episode needs B and E; I inside it if a strike
happened, X if it never did. Work rallies in order; export every
sitting as <code>state_labels_chicago0725.csv</code> into data/vision/.
</div>
</div><script>
const RALLIES = __RALLIES__;
const LSK = "state_audit_chicago0725";
let store = JSON.parse(localStorage.getItem(LSK) || "{}");
// store: {rally_cum: {player: [{s,i,e,nc}]}}  times in video seconds
const save = () => localStorage.setItem(LSK, JSON.stringify(store));
const V = document.getElementById("v");
let R = RALLIES[0], selP = 0;
const fpsEl = document.getElementById("fps");
const frame = () => 1 / (+fpsEl.value || 30);

drop.onclick = () => fpick.click();
drop.ondragover = e => e.preventDefault();
drop.ondrop = e => {e.preventDefault(); loadf(e.dataTransfer.files[0]);};
fpick.onchange = () => loadf(fpick.files[0]);
function loadf(f){ if(f){ V.src = URL.createObjectURL(f); go(R); } }

const rsel = document.getElementById("rsel");
RALLIES.forEach((r, i) => {
  const o = document.createElement("option");
  o.value = i; o.textContent = "rally " + r.rally_cum +
    " (" + r.contacts.length + " contacts)";
  rsel.appendChild(o);
});
rsel.onchange = () => go(RALLIES[+rsel.value]);
function go(r){ R = r; if (V.src) V.currentTime = r.t0; render(); }

function eps(p){
  const d = store[R.rally_cum] = store[R.rally_cum] || {};
  return d[p] = d[p] || [];
}
function rmarks(){
  const d = store[R.rally_cum] = store[R.rally_cum] || {};
  return d._rally = d._rally || {routine: null, dead: null};
}
function pbtns(){
  const el = document.getElementById("pbtns"); el.innerHTML = "";
  R.players.forEach((p, i) => {
    const b = document.createElement("button");
    b.textContent = (i + 1) + " " + p.split(" ").pop();
    b.className = i === selP ? "sel" : "";
    b.onclick = () => { selP = i; render(); };
    el.appendChild(b);
  });
}
function render(){
  pbtns();
  const lanes = document.getElementById("lanes"); lanes.innerHTML = "";
  const span = R.t1 - R.t0, X = t => (100 * (t - R.t0) / span) + "%";
  R.players.forEach((p, i) => {
    const lane = document.createElement("div"); lane.className = "lane";
    lane.innerHTML = `<span class="nm">${i + 1} ${p}</span>`;
    eps(p).forEach((ep, j) => {
      const s = ep.s ?? ep.i ?? ep.e, e = ep.e ?? ep.i ?? ep.s;
      if (s == null) return;
      const d = document.createElement("div");
      d.className = "ep" + (ep.nc ? " nc" : "") + (ep.e == null ? " open" : "");
      d.style.left = X(s);
      d.style.width = Math.max(0.4, 100 * (e - s) / span) + "%";
      d.title = `ep${j + 1} ${ep.s?.toFixed(2)}–${ep.e?.toFixed(2)}` +
        (ep.i ? ` impact ${ep.i.toFixed(2)}` : "") + (ep.nc ? " NO-CONTACT" : "");
      d.onclick = ev => { ev.stopPropagation(); selP = i;
        V.currentTime = ep.i ?? s; render(); };
      if (ep.i != null){
        const m = document.createElement("div"); m.className = "imp";
        m.style.left = X(ep.i); lane.appendChild(m);
      }
      lane.appendChild(d);
    });
    R.contacts.filter(c => c.hitter === p).forEach(c => {
      const t = document.createElement("div");
      t.className = "tick" + (["smash","speed-up","drive","counter","fast"]
        .includes(c.type) ? " f" : "");
      t.style.left = X(c.t); t.title = c.type + " @" + c.t;
      lane.appendChild(t);
    });
    const rm = rmarks();
    if (rm.routine != null){
      const m = document.createElement("div"); m.className = "rmark";
      m.style.left = X(rm.routine); m.title = "routine " + rm.routine.toFixed(2);
      lane.appendChild(m);
    }
    if (rm.dead != null){
      const m = document.createElement("div"); m.className = "dmark";
      m.style.left = X(rm.dead); m.title = "dead " + rm.dead.toFixed(2);
      lane.appendChild(m);
    }
    const cu = document.createElement("div"); cu.className = "cursor";
    cu.style.left = X(Math.min(Math.max(V.currentTime || R.t0, R.t0), R.t1));
    lane.appendChild(cu);
    lane.onclick = ev => {
      const r = lane.getBoundingClientRect();
      V.currentTime = R.t0 + span * (ev.clientX - r.left) / r.width;
      selP = i; render();
    };
    lanes.appendChild(lane);
  });
  const done = RALLIES.filter(r => {
    const d = store[r.rally_cum] || {};
    return Object.values(d).some(l => l.length);
  }).length;
  status.textContent = `player: ${R.players[selP]} · t=` +
    `${(V.currentTime || 0).toFixed(2)}s · rallies touched ${done}/${RALLIES.length}`;
}
V.ontimeupdate = render;

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const k = e.key.toLowerCase(), p = R.players[selP], t = V.currentTime;
  const L = eps(p), open = L.length && L[L.length - 1].e == null
    ? L[L.length - 1] : null;
  if (k === "arrowright") V.currentTime = t + frame();
  else if (k === "arrowleft") V.currentTime = Math.max(0, t - frame());
  else if (k === ".") V.currentTime = t + 10 * frame();
  else if (k === ",") V.currentTime = Math.max(0, t - 10 * frame());
  else if (k === " ") { if (V.paused) { V.playbackRate = 1; V.play(); } else V.pause(); }
  else if (["1","2","3","4"].includes(k)) selP = +k - 1;
  else if (k === "b") L.push({s: +t.toFixed(3), i: null, e: null, nc: false});
  else if (k === "i" && open) open.i = +t.toFixed(3);
  else if (k === "e" && open) open.e = +t.toFixed(3);
  else if (k === "x" && L.length) L[L.length - 1].nc = !L[L.length - 1].nc;
  else if (k === "r") { const m = rmarks();
    m.routine = m.routine == null ? +t.toFixed(3) : null; }
  else if (k === "d") { const m = rmarks();
    m.dead = m.dead == null ? +t.toFixed(3) : null; }
  else if (k === "backspace") {
    let best = null, bd = 1e9;
    L.forEach(ep => ["s","i","e"].forEach(f => {
      if (ep[f] != null && Math.abs(ep[f] - t) < bd)
        { bd = Math.abs(ep[f] - t); best = [ep, f]; }
    }));
    if (best) {
      best[0][best[1]] = null;
      if (best[0].s == null && best[0].i == null && best[0].e == null)
        L.splice(L.indexOf(best[0]), 1);
    }
  } else return;
  save(); render(); e.preventDefault();
});

bexp.onclick = () => {
  let out = "rally_cum,player,episode,kind,t_s\n";
  RALLIES.forEach(r => {
    const d = store[r.rally_cum] || {};
    const rm = d._rally || {};
    if (rm.routine != null)
      out += `${r.rally_cum},,0,routine_start,${rm.routine}\n`;
    if (rm.dead != null)
      out += `${r.rally_cum},,0,point_dead,${rm.dead}\n`;
    r.players.forEach(p => (d[p] || []).forEach((ep, j) => {
      if (ep.s != null) out += `${r.rally_cum},${p},${j + 1},start,${ep.s}\n`;
      if (ep.i != null) out += `${r.rally_cum},${p},${j + 1},impact,${ep.i}\n`;
      if (ep.e != null) out += `${r.rally_cum},${p},${j + 1},end,${ep.e}\n`;
      if (ep.nc) out += `${r.rally_cum},${p},${j + 1},no_contact,\n`;
    }));
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
      const [cum, p, epn, kind, ts] = line.split(",");
      const r = RALLIES.find(x => x.rally_cum === +cum); if (!r) return;
      const d = store[+cum] = store[+cum] || {};
      if (kind === "routine_start" || kind === "point_dead") {
        const rm = d._rally = d._rally || {routine: null, dead: null};
        rm[kind === "routine_start" ? "routine" : "dead"] = +ts;
        return;
      }
      const L = d[p] = (d[p] || []);
      while (L.length < +epn) L.push({s:null,i:null,e:null,nc:false});
      const ep = L[+epn - 1];
      if (kind === "no_contact") ep.nc = true;
      else ep[{start:"s",impact:"i",end:"e"}[kind]] = +ts;
    });
    save(); render();
  };
  rd.readAsText(csvpick.files[0]);
};
render();
</script></body></html>
"""


def selftest():
    rallies = build_rallies()
    assert [r["rally_cum"] for r in rallies] == PILOT
    for r in rallies:
        assert len(r["players"]) == 4
        assert r["t0"] < r["contacts"][0]["t"] < r["contacts"][-1]["t"] < r["t1"]
    n = sum(len(r["contacts"]) for r in rallies)
    assert n == 124, n
    assert 9 not in PILOT and 10 not in PILOT      # quarantine
    html = HTML.replace("__RALLIES__", json.dumps(rallies))
    assert "__RALLIES__" not in html and '"rally_cum": 13' in html
    print(f"selftest OK — {len(rallies)} pilot rallies, {n} contacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--out", default=str(OUT_HTML))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    rallies = build_rallies(a.labels)
    Path(a.out).write_text(HTML.replace("__RALLIES__", json.dumps(rallies)))
    n = sum(len(r["contacts"]) for r in rallies)
    print(f"wrote {a.out} — {len(rallies)} rallies, {n} contacts. "
          f"Open it, load the VOD, mark B/I/E episodes per player.")


if __name__ == "__main__":
    main()
