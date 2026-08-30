"""Ball audit — frame-by-frame human ball positions for ONE rally
(rally 1), with the oracle-test pre-registration frozen BEFORE any
label exists.

PROVENANCE. The ball thread is CLOSED (2026-08-15, physical wall).
The 2026-08-19 re-examination left exactly one honest instrument on
the table: the ORACLE TEST — "label the ball in EVERY frame of
contiguous inter-contact intervals and ask whether trajectory fitting
recovers the known contact times from human-quality positions. That
is decisive in the negative and needs no detector." Re-opening on a
reframing is legitimate per the process note but needs an explicit
user call and a fresh pre-registration. The user made the call
2026-08-30 ("I kind of want to do the ball in this rally next to
complete it"). This file IS the fresh pre-registration; it is
committed before the first ball label exists.

## FROZEN BAR (before any label; grading constants may not move)

Primary: run `--score` (the detector below, committed with this bar)
on the exported positions. Direction-change events on VISIBLE-only
positions must recover >= 80% of rally 1's 25 frame-exact impacts
(data/vision/state_labels_chicago0725.csv) within +/-0.15 s, AND
clear the 95th percentile of a 1000-draw circular-shift null (the
same event times rotated by a random phase within the rally span —
the lesson of score_localization.py: never grade placement on raw
recall).

Consequences, pre-committed:
- BAR CLEARS -> human ball positions are a real channel on this
  footage; extending ball passes to more rallies and feeding
  positions to the temporal model becomes licensed (train rallies
  only, via the pending temporal_gate amendment). The DETECTOR
  question stays closed — this licenses HUMAN labels, not a tracker.
- BAR FAILS on human-quality positions -> decisive in the negative,
  exactly as specced: the ball thread is closed FOR GOOD on this
  footage class, including the human-label variant. No knob-turning;
  a different bar needs different footage.

Secondary (reported, never decisive): V/I/N findability rates over
ALL frames — the zero-selection contiguous-frame visibility numbers
the 2026-08-19 thread wanted (the 270-cell instrument remains valid
for the match-wide draw; this is the one-rally dense version).

## The labeling protocol (tool below)

Every frame from 1 s before the serve impact to point-dead, ~30 fps:
  CLICK on the ball  = VISIBLE position (V) — a clean ball
  S then click       = SMEAR (a streak/blur that IS the ball; click
                       the streak's center). Ball in the pixels —
                       detector-reachable in principle.
  I then click       = INFERRED (occluded behind a body/paddle; you
                       know where it must be but it is NOT in the
                       pixels — detector-unreachable, the seen/
                       inferred trap class).
  N                  = not visible, position unknown
AMENDMENT 2026-08-30 (before any label existed): S/I split added at
the user's suggestion, subdividing the non-clean-visible space; V's
meaning and the frozen V-only primary bar are untouched. V+S recall
becomes a pre-declared SECONDARY (reported, never decisive).
  arrows navigate, backspace clears the frame, auto-advance on answer.
A dotted trail of your recent positions is drawn — contiguity is the
treatment; the trail is the lookup the isolated-frame test lacked.
Export -> data/vision/ball_path_r1.csv (frame,t_s,x,y,vis; x,y in
native video pixels; N rows have empty x,y).

Usage:
    python3 vision/make_ball_audit.py                    # -> HTML tool
    python3 vision/make_ball_audit.py --score data/vision/ball_path_r1.csv
    python3 vision/make_ball_audit.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "vision"
STATE = DATA / "state_labels_chicago0725.csv"
OUT_HTML = DATA / "ball_audit_r1.html"
RALLY = 1
FPS = 30.0
PRE_S = 1.0            # lead before the serve impact
TOL_S = 0.15           # frozen matching tolerance
RECALL_BAR = 0.80      # frozen primary bar
N_NULL = 1000
TURN_DEG = 25.0        # direction-change threshold, frozen
MIN_SEP_S = 0.25       # min separation between detected events, frozen


def load_impacts(state_path=STATE, rally=RALLY):
    imps, dead = [], None
    for r in csv.DictReader(open(state_path)):
        if int(r["rally_cum"]) != rally:
            continue
        if r["kind"] == "impact":
            imps.append(float(r["t_s"]))
        elif r["kind"] == "point_dead":
            dead = float(r["t_s"])
    if not imps:
        raise SystemExit(f"no impacts for rally {rally} in {state_path}")
    imps.sort()
    return imps, (dead if dead is not None else imps[-1] + 2.0)


def detect_events(points, turn_deg=TURN_DEG, min_sep=MIN_SEP_S):
    """Direction-change times from (t, x, y) VISIBLE points. For each
    interior point, the turn angle between the incoming and outgoing
    segments (nearest visible neighbors); peaks above turn_deg,
    greedily separated by min_sep, largest first."""
    pts = sorted(points)
    cands = []
    for k in range(1, len(pts) - 1):
        t0, x0, y0 = pts[k - 1]
        t1, x1, y1 = pts[k]
        t2, x2, y2 = pts[k + 1]
        if t1 - t0 > 0.35 or t2 - t1 > 0.35:
            continue                      # across a blind gap: no claim
        ax, ay = x1 - x0, y1 - y0
        bx, by = x2 - x1, y2 - y1
        na, nb = math.hypot(ax, ay), math.hypot(bx, by)
        if na < 2 or nb < 2:
            continue                      # ball ~static: angle is noise
        cosv = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
        ang = math.degrees(math.acos(cosv))
        if ang >= turn_deg:
            cands.append((ang, t1))
    events = []
    for ang, t in sorted(cands, reverse=True):
        if all(abs(t - e) >= min_sep for e in events):
            events.append(t)
    return sorted(events)


def score_events(events, impacts, span, tol=TOL_S, n_null=N_NULL, seed=1):
    def recall(evs):
        hit, used = 0, set()
        for t0 in impacts:
            m = [(abs(t0 - e), i) for i, e in enumerate(evs)
                 if i not in used and abs(t0 - e) <= tol]
            if m:
                used.add(min(m)[1]); hit += 1
        return hit / len(impacts)
    obs = recall(events)
    lo, hi = span
    rng = random.Random(seed)
    null = []
    for _ in range(n_null):
        ph = rng.uniform(0, hi - lo)
        null.append(recall(sorted((e - lo + ph) % (hi - lo) + lo
                                  for e in events)))
    null.sort()
    p95 = null[int(0.95 * n_null)]
    pct = sum(1 for x in null if x < obs) / n_null
    return obs, p95, pct, null[n_null // 2]


def run_score(path, state_path=STATE):
    impacts, dead = load_impacts(state_path)
    rows = list(csv.DictReader(open(path)))
    def pts(classes):
        return [(float(r["t_s"]), float(r["x"]), float(r["y"]))
                for r in rows if r["vis"] in classes and r["x"]]
    vis = pts("V")
    counts = {k: sum(1 for r in rows if r["vis"] == k) for k in "VSIN"}
    n = len(rows)
    print(f"frames answered: {n}  " + "  ".join(
        f"{k} {counts[k]} ({100*counts[k]/n:.0f}%)" for k in "VSIN"))
    events = detect_events(vis)
    span = (impacts[0] - PRE_S, dead)
    obs, p95, pct, med = score_events(events, impacts, span)
    print(f"direction-change events: {len(events)} over {len(impacts)} impacts")
    print(f"recall @ +/-{TOL_S}s: {100*obs:.1f}%  "
          f"(shift-null median {100*med:.1f}%, 95th {100*p95:.1f}%, "
          f"observed at {100*pct:.0f}th pct)")
    ev2 = detect_events(pts("VS"))
    obs2, p952, *_ = score_events(ev2, impacts, span, seed=2)
    print(f"secondary (never decisive) V+S recall: {100*obs2:.1f}% "
          f"(null 95th {100*p952:.1f}%) — what smear positions add")
    passed = obs >= RECALL_BAR and obs > p95
    print(f"FROZEN BAR (>= {100*RECALL_BAR:.0f}% AND clears null 95th): "
          f"{'PASS' if passed else 'FAIL'}")
    print("consequence per the registration: "
          + ("human ball positions are a licensed channel (train-only, "
             "gate amendment pending)." if passed else
             "ball thread closed for good on this footage class, "
             "human-label variant included."))


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>ball audit — rally 1 oracle test</title>
<style>
 body{font:14px system-ui;margin:0;background:#111;color:#ddd}
 #wrap{max-width:1040px;margin:0 auto;padding:12px}
 #vbox{position:relative}
 video{width:100%;display:block;background:#000;border-radius:6px}
 #ov{position:absolute;left:0;top:0;cursor:crosshair}
 #drop{border:2px dashed #555;border-radius:8px;padding:18px;text-align:center;
       cursor:pointer;margin:8px 0}
 .bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:6px 0}
 button{background:#2a2a2a;color:#ddd;border:1px solid #555;border-radius:5px;
        padding:4px 10px;cursor:pointer}
 kbd{background:#333;border-radius:4px;padding:1px 6px;border:1px solid #555}
 #note{color:#999;font-size:12px;line-height:1.6;margin-top:8px}
 #imode{color:#cc7;font-weight:bold;visibility:hidden}
 #prog{height:10px;background:#222;border-radius:5px;overflow:hidden}
 #progfill{height:100%;background:#4a7;width:0}
 input{background:#222;color:#ddd;border:1px solid #555;border-radius:4px}
</style></head><body><div id="wrap">
<h3>Ball audit — rally 1, every frame (the oracle test)</h3>
<div id="drop">🎬 <b>Load the match video</b> — click or drag
<input type="file" id="fpick" accept="video/*,.webm,.mp4,.mkv" hidden></div>
<div id="vbox"><video id="v" preload="auto"></video><canvas id="ov"></canvas></div>
<div class="bar">
 <span id="status">—</span>
 <span id="imode">—</span>
 <span style="flex:1"></span>
 <label>fps <input id="fps" type="number" value="30" step="0.01" style="width:60px"></label>
 <button id="bexp">⬇ export</button>
 <button id="bimp">⬆ import</button>
 <input type="file" id="csvpick" accept=".csv" hidden>
</div>
<div id="prog"><div id="progfill"></div></div>
<div id="note">
<b>One answer per frame — four kinds:</b><br>
🟢 <b>See a clean ball → just click it.</b> Records visible (V) and
auto-advances. When the ball moves cleanly you'll settle into
click-click-click rhythm.<br>
🔵 <b>See a streak/blur that IS the ball → press <kbd>S</kbd>, then
click the streak's center.</b> "I know where it is… kinda" — the ball
is in the pixels, just smeared.<br>
🟠 <b>Ball hidden behind a body or paddle → press <kbd>I</kbd>, then
click where it must be.</b> "I know it's right behind her" — the ball
is NOT in the pixels; your trajectory sense supplies the position.<br>
⚫ <b>Genuinely can't place it → press <kbd>N</kbd>.</b> No click, no
guess — an honest N is data too.<br><br>
After <kbd>S</kbd> or <kbd>I</kbd> a colored banner shows the armed
mode; it resets to plain-click after each answer. Dots are HIDDEN by
default so they never sit on the ball — lost it? press <kbd>T</kbd> to
show your last second of positions, follow the trail, <kbd>T</kbd>
again to hide.
<b>Accuracy bar:</b> center-of-blob is fine, don't agonize; be
consistent on streaks (always the center).<br><br>
<b>Navigation:</b> <kbd>←</kbd>/<kbd>→</kbd> step ±1 frame without
answering · <kbd>,</kbd>/<kbd>.</kbd> jump ±10 · <kbd>⌫</kbd> clear
this frame (then re-answer).<br>
<b>Stopping:</b> anytime — progress autosaves in this browser. At the
end of every sitting hit <b>⬇ export</b> and save as
<code>ball_path_r1.csv</code> into data/vision/ (partial is fine;
⬆ import restores it exactly). Score with:
<code>python3 vision/make_ball_audit.py --score data/vision/ball_path_r1.csv</code>
</div>
</div><script>
const CFG = __CFG__;
const LSK = "ball_audit_r1";
let store = JSON.parse(localStorage.getItem(LSK) || "{}");
const save = () => localStorage.setItem(LSK, JSON.stringify(store));
const V = document.getElementById("v"), OV = document.getElementById("ov");
const fpsEl = document.getElementById("fps");
let k = +(localStorage.getItem(LSK + "_k") || 0), mode = "V";
let trail = localStorage.getItem(LSK + "_trail") === "1";
const fps = () => (+fpsEl.value || 30);
const NF = Math.round((CFG.t1 - CFG.t0) * 30);
const tOf = j => CFG.t0 + j / fps();

drop.onclick = () => fpick.click();
drop.ondragover = e => e.preventDefault();
drop.ondrop = e => {e.preventDefault(); loadf(e.dataTransfer.files[0]);};
fpick.onchange = () => loadf(fpick.files[0]);
function loadf(f){ if(f){ V.src = URL.createObjectURL(f); go(k); } }

function go(j){
  k = Math.max(0, Math.min(NF - 1, j));
  if (V.src) { V.pause(); V.currentTime = tOf(k); }
  localStorage.setItem(LSK + "_k", k);
  render();
}
function render(){
  const done = Object.keys(store).length;
  const a = store[k];
  status.innerHTML = `frame <b>${k + 1}</b>/${NF} (t=${tOf(k).toFixed(2)}s)` +
    ` — this frame: <b>${a ? a.vis : "·"}</b> — answered ${done}/${NF}`;
  progfill.style.width = (100 * done / NF) + "%";
  imodeEl.style.visibility = mode === "V" ? "hidden" : "visible";
  imodeEl.textContent = mode === "I"
    ? "INFERRED mode — click where it must be (behind the body)"
    : "SMEAR mode — click the streak's center";
  imodeEl.style.color = mode === "I" ? "#cc7" : "#7ac";
  draw();
}
const imodeEl = document.getElementById("imode");
function draw(){
  OV.width = V.clientWidth; OV.height = V.clientHeight;
  const c = OV.getContext("2d");
  c.clearRect(0, 0, OV.width, OV.height);
  if (!V.videoWidth || !trail) return;
  const sx = OV.width / V.videoWidth, sy = OV.height / V.videoHeight;
  for (let j = k - 30; j <= k; j++){
    const a = store[j];
    if (!a || a.vis === "N") continue;
    const r = j === k ? 6 : 2.5;
    c.beginPath();
    c.arc(a.x * sx, a.y * sy, r, 0, 7);
    c.fillStyle = ({V: "#4caf50", S: "#55aacc", I: "#cc7722"}[a.vis]
                   || "#888") + (j === k ? "" : "88");
    c.fill();
  }
}
V.onloadedmetadata = render;
V.onseeked = render;
new ResizeObserver(draw).observe(V);

OV.onclick = e => {
  if (!V.videoWidth) return;
  const x = e.offsetX * V.videoWidth / OV.width;
  const y = e.offsetY * V.videoHeight / OV.height;
  store[k] = {vis: mode, x: Math.round(x), y: Math.round(y)};
  mode = "V"; save(); go(k + 1);
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const key = e.key.toLowerCase();
  if (key === "arrowright") go(k + 1);
  else if (key === "arrowleft") go(k - 1);
  else if (key === ".") go(k + 10);
  else if (key === ",") go(k - 10);
  else if (key === "i") { mode = mode === "I" ? "V" : "I"; render(); }
  else if (key === "s") { mode = mode === "S" ? "V" : "S"; render(); }
  else if (key === "t") { trail = !trail;
    localStorage.setItem(LSK + "_trail", trail ? "1" : "0"); render(); }
  else if (key === "n") { store[k] = {vis: "N"}; mode = "V"; save(); go(k + 1); }
  else if (key === "backspace") { delete store[k]; save(); render(); }
  else return;
  e.preventDefault();
});
bexp.onclick = () => {
  let out = "frame,t_s,x,y,vis\n";
  for (let j = 0; j < NF; j++){
    const a = store[j]; if (!a) continue;
    out += `${j},${tOf(j).toFixed(3)},${a.x ?? ""},${a.y ?? ""},${a.vis}\n`;
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([out], {type: "text/csv"}));
  a.download = "ball_path_r1.csv"; a.click();
};
bimp.onclick = () => csvpick.click();
csvpick.onchange = () => {
  const rd = new FileReader();
  rd.onload = () => {
    rd.result.trim().split("\n").slice(1).forEach(line => {
      const [j, , x, y, vis] = line.split(",");
      store[+j] = vis === "N" ? {vis: "N"} : {vis, x: +x, y: +y};
    });
    save(); render();
  };
  rd.readAsText(csvpick.files[0]);
};
render();
</script></body></html>
"""


def selftest():
    # synthetic zig-zag path: ball flies straight between impacts,
    # turns at each — detector must find the turns, shift null must sit
    # far below
    rng = random.Random(7)
    impacts = [10.0 + 1.1 * i for i in range(12)]
    pts, x, y, vx, vy = [], 100.0, 300.0, 260.0, -35.0
    t, i = impacts[0] - 0.9, 0
    while t < impacts[-1] + 0.5:
        if i < len(impacts) and t >= impacts[i]:
            vx, vy = -vx * rng.uniform(0.8, 1.2), rng.uniform(-90, 90)
            i += 1
        x += vx / 30; y += vy / 30
        pts.append((round(t, 3), x + rng.gauss(0, 1), y + rng.gauss(0, 1)))
        t += 1 / 30
    evs = detect_events(pts)
    obs, p95, pct, med = score_events(evs, impacts,
                                      (impacts[0] - 1, impacts[-1] + 1),
                                      n_null=300)
    assert obs >= 0.8, (obs, evs[:5])
    assert obs > p95, (obs, p95)
    # occlusion gaps around impacts must not fabricate events
    gappy = [p for p in pts if all(abs(p[0] - t0) > 0.2 for t0 in impacts)]
    evs2 = detect_events(gappy)
    obs2, p952, *_ = score_events(evs2, impacts,
                                  (impacts[0] - 1, impacts[-1] + 1),
                                  n_null=300)
    assert obs2 < obs, "blind-at-impact path should score lower"
    impacts_r, dead = load_impacts()
    assert len(impacts_r) == 25 and dead > impacts_r[-1]
    cfg = {"t0": round(impacts_r[0] - PRE_S, 3), "t1": round(dead, 3)}
    html = HTML.replace("__CFG__", json.dumps(cfg))
    assert "__CFG__" not in html
    print(f"selftest OK — synthetic recall {obs:.0%} vs null95 {p95:.0%}; "
          f"gappy path drops to {obs2:.0%}; rally-1 cfg {cfg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(STATE))
    ap.add_argument("--score", metavar="BALL_CSV")
    ap.add_argument("--out", default=str(OUT_HTML))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.score:
        run_score(a.score, a.state)
        return
    impacts, dead = load_impacts(a.state)
    cfg = {"t0": round(impacts[0] - PRE_S, 3), "t1": round(dead, 3)}
    n = int((cfg["t1"] - cfg["t0"]) * FPS)
    Path(a.out).write_text(HTML.replace("__CFG__", json.dumps(cfg)))
    print(f"wrote {a.out} — {n} frames ({cfg['t0']}s to {cfg['t1']}s), "
          f"{len(impacts)} known impacts as the sealed answer key.")


if __name__ == "__main__":
    main()
