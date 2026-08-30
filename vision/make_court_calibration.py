"""Court calibration clicks — the vertical references that upgrade the
ground homography to a FULL CAMERA, for the 3D court prototype
(user go 2026-08-30: "Would love that!").

WHY. The homography (court.py, 0.06 ft) maps image <-> floor, which is
enough for feet-on-the-ground but not for rays in the air: lifting the
labeled 2D ball path (data/vision/ball_path_r1.csv, oracle-PASSED
channel) into 3D needs the camera's full projection matrix. That is
solvable by DLT from >= 6 non-coplanar point correspondences: the
court's painted intersections give the plane, and the NET gives the
verticals — posts 36 in, center tape 34 in. One sitting of ~11 clicks
on a single clean frame, once per camera setup.

COORDINATE FRAME (court.py convention, exactly): x in [0,20] ft from
the LEFT sideline, y in [0,44] ft with y=0 at the FAR baseline (top of
screen) and y=44 NEAR (bottom), net at y=22, kitchen lines y=15 (far)
and y=29 (near), z up in feet. Net posts are assumed 1 ft outside the
sidelines (22 ft net span, the standard) — recorded here so the solver
knows it is an assumption it may relax.

THE CLICKS, in order (K skips one that is occluded/out of frame —
plane points are redundant, the two POST TOPS are the payload):

    far-left corner        (0, 0, 0)
    far-right corner       (20, 0, 0)
    near-left corner       (0, 44, 0)
    near-right corner      (20, 44, 0)
    far-kitchen x left SL  (0, 15, 0)
    far-kitchen x right SL (20, 15, 0)
    near-kitchen x left SL (0, 29, 0)
    near-kitchen x right SL(20, 29, 0)
    LEFT net post TOP      (-1, 22, 3.0)
    RIGHT net post TOP     (21, 22, 3.0)
    net center tape TOP    (10, 22, 2.833)

A 4x magnifier follows the cursor; after a click, arrow keys nudge the
point by 1 px. Pick a clean, full-court frame first (the pre-serve
pause of rally 1, ~t=9.5 s, is ideal) — the time box + arrows seek.
Export -> data/vision/court_landmarks_chicago0725.csv
(name,X_ft,Y_ft,Z_ft,px_x,px_y,t_s). The camera is static on this
broadcast's main angle, so one frame calibrates the whole VOD (camera
cuts were always excluded anyway).

Usage:
    python3 vision/make_court_calibration.py          # -> HTML tool
    python3 vision/make_court_calibration.py --selftest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "vision"
OUT_HTML = DATA / "court_calibration.html"
DEFAULT_T = 9.5   # rally 1 pre-serve pause

LANDMARKS = [
    ("far_left_corner",   0.0,  0.0, 0.0, "FAR baseline × LEFT sideline (top-left court corner)"),
    ("far_right_corner", 20.0,  0.0, 0.0, "FAR baseline × RIGHT sideline"),
    ("near_left_corner",  0.0, 44.0, 0.0, "NEAR baseline × LEFT sideline (bottom-left)"),
    ("near_right_corner", 20.0, 44.0, 0.0, "NEAR baseline × RIGHT sideline"),
    ("far_kitchen_left",  0.0, 15.0, 0.0, "FAR kitchen line × LEFT sideline"),
    ("far_kitchen_right", 20.0, 15.0, 0.0, "FAR kitchen line × RIGHT sideline"),
    ("near_kitchen_left", 0.0, 29.0, 0.0, "NEAR kitchen line × LEFT sideline"),
    ("near_kitchen_right", 20.0, 29.0, 0.0, "NEAR kitchen line × RIGHT sideline"),
    ("left_post_top",    -1.0, 22.0, 3.0, "TOP of the LEFT net post (36 in)"),
    ("right_post_top",   21.0, 22.0, 3.0, "TOP of the RIGHT net post (36 in)"),
    ("net_center_top",   10.0, 22.0, 2.833, "TOP of the net's white tape at CENTER (34 in)"),
]

HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>court calibration — 11 clicks</title>
<style>
 body{font:14px system-ui;margin:0;background:#111;color:#ddd}
 #wrap{max-width:1060px;margin:0 auto;padding:12px}
 #vbox{position:relative}
 video{width:100%;display:block;background:#000;border-radius:6px}
 #ov{position:absolute;left:0;top:0;cursor:crosshair}
 #mag{position:absolute;pointer-events:none;border:2px solid #8cf;
      border-radius:6px;display:none;background:#000}
 #drop{border:2px dashed #555;border-radius:8px;padding:18px;text-align:center;
       cursor:pointer;margin:8px 0}
 #task{font-size:17px;margin:8px 0;padding:8px 12px;background:#1d2b1d;
       border:1px solid #4a7;border-radius:6px}
 #task b{color:#8f8}
 .bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:6px 0}
 button{background:#2a2a2a;color:#ddd;border:1px solid #555;border-radius:5px;
        padding:4px 10px;cursor:pointer}
 kbd{background:#333;border-radius:4px;padding:1px 6px;border:1px solid #555}
 #note{color:#999;font-size:12px;line-height:1.6;margin-top:8px}
 input{background:#222;color:#ddd;border:1px solid #555;border-radius:4px}
</style></head><body><div id="wrap">
<h3>Court calibration — the 11 clicks that make the camera 3D</h3>
<div id="drop">🎬 <b>Load the match video</b> — click or drag
<input type="file" id="fpick" accept="video/*,.webm,.mp4,.mkv" hidden></div>
<div id="vbox"><video id="v" preload="auto"></video><canvas id="ov"></canvas>
<canvas id="mag" width="180" height="180"></canvas></div>
<div id="task">—</div>
<div class="bar">
 <label>frame t <input id="tsel" type="number" step="0.1" value="__T0__"
   style="width:80px"> s</label>
 <button id="bseek">seek</button>
 <span style="flex:1"></span>
 <span id="status">—</span>
 <button id="bexp">⬇ export</button>
 <button id="bimp">⬆ import</button>
 <input type="file" id="csvpick" accept=".csv" hidden>
</div>
<div id="note">
<b>First</b> pick a clean full-court frame (default t=__T0__ s, rally 1's
pre-serve pause, is usually perfect) — type a time and seek, or nudge
with <kbd>,</kbd>/<kbd>.</kbd> (±1 frame). <b>Then click each named
landmark</b> — the magnifier follows your cursor for precision. After a
click, <kbd>←→↑↓</kbd> nudge that point by 1 px. <kbd>K</kbd> skips a
landmark you can't see (plane points are redundant; the two <b>net post
tops</b> matter most). <kbd>U</kbd> steps back to redo the previous
one. When all 11 are done (or skipped), <b>⬇ export</b> as
<code>court_landmarks_chicago0725.csv</code> into data/vision/ and drop
it in the thread — that's everything the 3D solver needs.
</div>
</div><script>
const LMS = __LMS__;
const LSK = "court_calib_chicago0725";
let store = JSON.parse(localStorage.getItem(LSK) || "{}");
const save = () => localStorage.setItem(LSK, JSON.stringify(store));
const V = document.getElementById("v"), OV = document.getElementById("ov");
const MAG = document.getElementById("mag"), tsel = document.getElementById("tsel");
let cur = 0;
while (cur < LMS.length && store[LMS[cur][0]]) cur++;

drop.onclick = () => fpick.click();
drop.ondragover = e => e.preventDefault();
drop.ondrop = e => {e.preventDefault(); loadf(e.dataTransfer.files[0]);};
fpick.onchange = () => loadf(fpick.files[0]);
function loadf(f){ if(f){ V.src = URL.createObjectURL(f);
  V.onloadedmetadata = () => { V.currentTime = +tsel.value; render(); }; } }
bseek.onclick = () => { V.pause(); V.currentTime = +tsel.value; };

function render(){
  const done = Object.keys(store).filter(k => store[k] !== "skip").length;
  const lm = LMS[cur];
  task.innerHTML = lm
    ? `Click <b>${lm[4]}</b> <span style="color:#888">(court ` +
      `${lm[1]},${lm[2]},${lm[3]} ft) — ${cur + 1}/${LMS.length}</span>`
    : `<b>All landmarks done</b> — export below.`;
  status.textContent = `placed ${done}/${LMS.length}`;
  draw();
}
function draw(){
  OV.width = V.clientWidth; OV.height = V.clientHeight;
  const c = OV.getContext("2d");
  c.clearRect(0, 0, OV.width, OV.height);
  if (!V.videoWidth) return;
  const sx = OV.width / V.videoWidth, sy = OV.height / V.videoHeight;
  c.font = "11px system-ui";
  LMS.forEach((lm, i) => {
    const a = store[lm[0]];
    if (!a || a === "skip") return;
    c.strokeStyle = i === cur - 1 ? "#8cf" : "#4caf50"; c.lineWidth = 1.5;
    const x = a[0] * sx, y = a[1] * sy;
    c.beginPath(); c.moveTo(x - 7, y); c.lineTo(x + 7, y);
    c.moveTo(x, y - 7); c.lineTo(x, y + 7); c.stroke();
    c.fillStyle = "#8f8"; c.fillText(lm[0], x + 8, y - 4);
  });
}
V.onseeked = render;
new ResizeObserver(draw).observe(V);

OV.onmousemove = e => {
  if (!V.videoWidth) return;
  const m = MAG.getContext("2d"), Z = 4, R = MAG.width / (2 * Z);
  const vx = e.offsetX * V.videoWidth / OV.width;
  const vy = e.offsetY * V.videoHeight / OV.height;
  m.imageSmoothingEnabled = false;
  m.clearRect(0, 0, MAG.width, MAG.height);
  m.drawImage(V, vx - R, vy - R, 2 * R, 2 * R, 0, 0, MAG.width, MAG.height);
  m.strokeStyle = "#f55";
  m.beginPath(); m.moveTo(MAG.width/2 - 8, MAG.height/2);
  m.lineTo(MAG.width/2 + 8, MAG.height/2);
  m.moveTo(MAG.width/2, MAG.height/2 - 8);
  m.lineTo(MAG.width/2, MAG.height/2 + 8); m.stroke();
  MAG.style.display = "block";
  MAG.style.left = Math.min(e.offsetX + 20, OV.width - 190) + "px";
  MAG.style.top = Math.max(e.offsetY - 200, 0) + "px";
};
OV.onmouseleave = () => { MAG.style.display = "none"; };
OV.onclick = e => {
  if (!V.videoWidth || cur >= LMS.length) return;
  const x = e.offsetX * V.videoWidth / OV.width;
  const y = e.offsetY * V.videoHeight / OV.height;
  store[LMS[cur][0]] = [Math.round(x * 10) / 10, Math.round(y * 10) / 10,
                        +V.currentTime.toFixed(3)];
  cur++; save(); render();
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const k = e.key.toLowerCase(), fr = 1 / 30;
  const last = cur > 0 ? store[LMS[cur - 1][0]] : null;
  if (k === ",") V.currentTime = Math.max(0, V.currentTime - fr);
  else if (k === ".") V.currentTime += fr;
  else if (k === "k" && cur < LMS.length) { store[LMS[cur][0]] = "skip"; cur++; }
  else if (k === "u" && cur > 0) { cur--; delete store[LMS[cur][0]]; }
  else if (k.startsWith("arrow") && last && last !== "skip") {
    if (k === "arrowleft") last[0] -= 1;
    else if (k === "arrowright") last[0] += 1;
    else if (k === "arrowup") last[1] -= 1;
    else if (k === "arrowdown") last[1] += 1;
  } else return;
  save(); render(); e.preventDefault();
});
bexp.onclick = () => {
  let out = "name,X_ft,Y_ft,Z_ft,px_x,px_y,t_s\n";
  LMS.forEach(lm => {
    const a = store[lm[0]];
    if (!a || a === "skip") return;
    out += `${lm[0]},${lm[1]},${lm[2]},${lm[3]},${a[0]},${a[1]},${a[2]}\n`;
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([out], {type: "text/csv"}));
  a.download = "court_landmarks_chicago0725.csv"; a.click();
};
bimp.onclick = () => csvpick.click();
csvpick.onchange = () => {
  const rd = new FileReader();
  rd.onload = () => {
    rd.result.trim().split("\n").slice(1).forEach(line => {
      const [name, , , , x, y, t] = line.split(",");
      store[name] = [+x, +y, +t];
    });
    cur = 0; while (cur < LMS.length && store[LMS[cur][0]]) cur++;
    save(); render();
  };
  rd.readAsText(csvpick.files[0]);
};
render();
</script></body></html>
"""


def selftest():
    names = [l[0] for l in LANDMARKS]
    assert len(names) == len(set(names)) == 11
    plane = [l for l in LANDMARKS if l[3] == 0.0]
    vert = [l for l in LANDMARKS if l[3] > 0]
    assert len(plane) == 8 and len(vert) == 3
    # court.py frame: x in [0,20] except posts, y in [0,44], net y=22
    for n, x, y, z, _ in LANDMARKS:
        assert -1 <= x <= 21 and 0 <= y <= 44
        if "post" in n or "net" in n:
            assert y == 22.0
    assert dict((l[0], l[3]) for l in LANDMARKS)["net_center_top"] == 2.833
    html = (HTML.replace("__LMS__", json.dumps(LANDMARKS))
                .replace("__T0__", str(DEFAULT_T)))
    assert "__LMS__" not in html and "__T0__" not in html
    print("selftest OK — 8 plane + 3 vertical landmarks, court.py frame")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_HTML))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    html = (HTML.replace("__LMS__", json.dumps(LANDMARKS))
                .replace("__T0__", str(DEFAULT_T)))
    Path(a.out).write_text(html)
    print(f"wrote {a.out} — 11 landmarks (8 plane + 3 vertical). "
          f"Open it, load the VOD, click them, export.")


if __name__ == "__main__":
    main()
