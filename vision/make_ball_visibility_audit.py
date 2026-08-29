"""Contiguous-frame ball findability — the unbiased test the 2026-08-19
thread left on the table (swing_explore_notes.md, "BALL FINDABILITY BY
VLM" -> "THE 64% RE-EXAMINED").

The ball thread's closure rests on 63.5% [57, 70] in-play per-frame
findability, measured on ISOLATED frames judged alone (a search). The
user's counter — finding the ball in frame k+1 having seen it in frame
k is a LOOKUP — ran ~92% on contiguous grids, but on windows I had
selected after tracing their trajectories, so the gap is confounded.
The clean instrument already exists: the 30 localization windows
(vlm_loc_key_20260819.csv) were drawn at RANDOM (seeded, dead time
included), 9 contiguous frames each at 0.15 s — 270 cells. Marking
visible / not-visible over them, IN SEQUENCE, is a contiguous-frame
findability estimate directly comparable to the 64%, with no selection
by anyone.

This script generates the marking page (default) and scores the export
(--score). Three calls per cell:

  V = visible (you can see the ball in this frame)
  I = inferred only (you know where it is — occluded behind a body,
      pure streak-smear, etc. — but cannot SEE it)
  N = not visible, position unknown

PRIMARY stat = V / (V+I+N) over IN-PLAY cells (cell time inside its
rally's serve..last-contact span). I counts as NOT visible there —
that is what keeps the number comparable to the loupe test, where
"can't find" included occlusion. The I share is reported separately:
it is the occlusion decomposition the closure never had, and the
population a seen/inferred-flagged tracker would interpolate through.

DECISION SEMANTICS (recorded before the data exists): this measures;
it does not reopen. If contiguous unbiased findability lands at the
isolated 64%, the closure's number stands as stated. If it lands near
the biased ~92%, isolation was load-bearing in the closure and
re-opening the ball thread becomes a legitimate user call needing a
fresh pre-registration (the process note of 2026-08-19 stands either
way).

Usage:
    python3 vision/make_ball_visibility_audit.py            # -> HTML tool
    python3 vision/make_ball_visibility_audit.py --score data/vision/ball_visibility_contiguous_calls.csv
    python3 vision/make_ball_visibility_audit.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "vision"
KEY = DATA / "vlm_loc_key_20260819.csv"
LABELS = DATA / "contact_labels_chicago0725.csv"
OUT_HTML = DATA / "ball_visibility_audit.html"
STEP_S = 0.15          # vlm_localize_sample.py cell spacing — must match
N_CELLS = 9
ISOLATED = (129, 203)  # the closure's in-play read, for the comparison line


def load_cells(key_path=KEY):
    """[{window, rally_cum, cell, t_s}] — 9 contiguous cells per window."""
    cells = []
    for r in csv.DictReader(open(key_path)):
        t0 = float(r["t0_s"])
        for k in range(N_CELLS):
            cells.append({"window": r["window"].replace(".png", ""),
                          "rally_cum": int(r["rally_cum"]),
                          "cell": k + 1,
                          "t_s": round(t0 + k * STEP_S, 3)})
    return cells


def rally_spans(labels_path=LABELS):
    """rally_cum -> (serve_t, last_contact_t) in video seconds."""
    spans = {}
    for r in csv.DictReader(open(labels_path)):
        t = float(r["t_refined_s"] or r["t_tap_s"])
        cum = int(r["rally_cum"])
        lo, hi = spans.get(cum, (t, t))
        spans[cum] = (min(lo, t), max(hi, t))
    return spans


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def score(calls_rows, cells, spans):
    """calls_rows: [{window, cell, call}]. Returns the report string."""
    by_key = {(c["window"], c["cell"]): c for c in cells}
    marked = {}
    for r in calls_rows:
        key = (r["window"], int(r["cell"]))
        if key not in by_key:
            raise SystemExit(f"call for unknown cell {key}")
        call = r["call"].strip().upper()
        if call not in ("V", "I", "N"):
            raise SystemExit(f"bad call {r['call']!r} at {key}")
        marked[key] = call

    def in_play(c):
        lo, hi = spans.get(c["rally_cum"], (None, None))
        return lo is not None and lo <= c["t_s"] <= hi

    lines = []
    total = [c for c in cells if by_key[(c["window"], c["cell"])] and
             (c["window"], c["cell"]) in marked]
    play = [c for c in total if in_play(c)]
    lines.append(f"marked {len(total)}/{len(cells)} cells "
                 f"({len(play)} in-play by rally span)")
    for tag, pop in (("ALL", total), ("IN-PLAY", play)):
        n = len(pop)
        v = sum(1 for c in pop if marked[(c["window"], c["cell"])] == "V")
        i = sum(1 for c in pop if marked[(c["window"], c["cell"])] == "I")
        p, lo, hi = wilson(v, n)
        lines.append(f"  {tag:8s} visible {v}/{n} = {100 * p:.1f}% "
                     f"[{100 * lo:.0f}, {100 * hi:.0f}]"
                     f"   inferred-only {i}/{n}"
                     f" ({100 * i / n if n else 0:.0f}%)")
    k0, n0 = ISOLATED
    p0, lo0, hi0 = wilson(k0, n0)
    lines.append(f"  isolated-frame benchmark (closure): {k0}/{n0} = "
                 f"{100 * p0:.1f}% [{100 * lo0:.0f}, {100 * hi0:.0f}]")
    if play:
        v = sum(1 for c in play if marked[(c["window"], c["cell"])] == "V")
        _, lo, _ = wilson(v, len(play))
        if lo > hi0:
            lines.append("  -> contiguous > isolated, CIs disjoint: "
                         "isolation was load-bearing in the closure. "
                         "Re-opening the ball thread is now a legitimate "
                         "user call + fresh pre-registration.")
        elif wilson(v, len(play))[2] < lo0:
            lines.append("  -> contiguous < isolated: the closure's "
                         "number stands, if anything generous.")
        else:
            lines.append("  -> CIs overlap: no verdict; the closure's "
                         "number stands as stated.")
    per_w = {}
    for c in play:
        per_w.setdefault(c["window"], []).append(
            marked[(c["window"], c["cell"])])
    row = "  per-window (in-play V/I/N): " + "  ".join(
        f"{w}:{''.join(cs)}" for w, cs in sorted(per_w.items()))
    lines.append(row)
    return "\n".join(lines)


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>ball visibility — contiguous unbiased test</title>
<style>
 body{font:14px system-ui;margin:0;background:#111;color:#ddd}
 #wrap{max-width:980px;margin:0 auto;padding:12px}
 video{width:100%;background:#000;border-radius:6px}
 #drop{border:2px dashed #555;border-radius:8px;padding:24px;text-align:center;
       cursor:pointer;margin:10px 0}
 .bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:8px 0}
 #cellinfo{font-size:18px}
 kbd{background:#333;border-radius:4px;padding:1px 6px;border:1px solid #555}
 button{background:#2a2a2a;color:#ddd;border:1px solid #555;border-radius:5px;
        padding:4px 10px;cursor:pointer}
 #strip{display:grid;grid-template-columns:repeat(30,1fr);gap:2px;margin:8px 0}
 .w{height:16px;border-radius:2px;background:#333;font-size:8px;text-align:center;
    cursor:pointer;line-height:16px;color:#999}
 .w.done{background:#265b26}.w.part{background:#5b5426}.w.cur{outline:2px solid #8cf}
 .call-V{color:#7c7}.call-I{color:#cc7}.call-N{color:#c77}
 #note{color:#999;font-size:12px;line-height:1.5}
</style></head><body><div id="wrap">
<h3>Contiguous-frame ball findability — 270 cells, unbiased draw</h3>
<div id="drop">🎬 <b>Load the match video</b> (full_match.mp4.webm) — click or drag it in
<input type="file" id="fpick" accept="video/*,.webm,.mp4,.mkv" hidden></div>
<video id="v" controls preload="auto"></video>
<div class="bar">
 <span id="cellinfo">—</span>
 <span style="flex:1"></span>
 <label>offset s <input type="number" id="voff" step="0.05" value="0"
   style="width:70px;background:#222;color:#ddd;border:1px solid #555"></label>
 <button id="bexp">⬇ export CSV</button>
 <button id="bimp">⬆ import</button>
 <input type="file" id="csvpick" accept=".csv" hidden>
</div>
<div id="strip"></div>
<div id="note">
<b>Protocol (from the 2026-08-19 thread):</b> work each window's 9 cells
<b>in order, 1→9</b> — seeing cell k before judging k+1 IS the treatment
being measured. Per cell: <kbd>V</kbd> ball visible · <kbd>I</kbd> can't
see it but you know where it is (occluded / pure smear) · <kbd>N</kbd>
not visible, position unknown. <kbd>←</kbd>/<kbd>→</kbd> move without
calling; <kbd>Backspace</kbd> clears the call; a call auto-advances.
Judge honestly per frame — I is not a failure, it's the occlusion
decomposition the closure never had. Export at the end of every sitting
and drop the CSV in the thread. Score with:
<code>python3 vision/make_ball_visibility_audit.py --score ball_visibility_contiguous_calls.csv</code>
</div>
</div><script>
const CELLS = __CELLS__;
const LSK = "ballvis_contig_20260819";
let store = JSON.parse(localStorage.getItem(LSK) || "{}");
const save = () => localStorage.setItem(LSK, JSON.stringify(store));
let idx = +(localStorage.getItem(LSK + "_idx") || 0);
const V = document.getElementById("v"), voff = document.getElementById("voff");
const key = c => c.window + ":" + c.cell;
document.getElementById("drop").onclick = () => fpick.click();
document.getElementById("drop").ondragover = e => e.preventDefault();
document.getElementById("drop").ondrop = e => {e.preventDefault(); loadf(e.dataTransfer.files[0]);};
fpick.onchange = () => loadf(fpick.files[0]);
function loadf(f){ if(f){ V.src = URL.createObjectURL(f); seek(); } }
function seek(){
  const c = CELLS[idx];
  if (V.src) { V.currentTime = c.t_s + (+voff.value || 0); V.pause(); }
  const cur = store[key(c)] || "·";
  cellinfo.innerHTML = `<b>${c.window}</b> cell <b>${c.cell}</b>/9 ` +
    `(rally ${c.rally_cum}, t=${c.t_s.toFixed(2)}s) — call: ` +
    `<b class="call-${cur}">${cur}</b> · ` +
    `${Object.keys(store).length}/${CELLS.length} marked`;
  strip();
  localStorage.setItem(LSK + "_idx", idx);
}
function strip(){
  const el = document.getElementById("strip"); el.innerHTML = "";
  const wins = [...new Set(CELLS.map(c => c.window))];
  wins.forEach(w => {
    const cs = CELLS.filter(c => c.window === w);
    const n = cs.filter(c => store[key(c)]).length;
    const d = document.createElement("div");
    d.className = "w" + (n === 9 ? " done" : n ? " part" : "") +
      (CELLS[idx].window === w ? " cur" : "");
    d.textContent = w.replace("w", ""); d.title = `${w}: ${n}/9`;
    d.onclick = () => { idx = CELLS.indexOf(cs[0]); seek(); };
    el.appendChild(d);
  });
}
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const k = e.key.toLowerCase();
  if (k === "arrowright") { idx = Math.min(idx + 1, CELLS.length - 1); seek(); }
  else if (k === "arrowleft") { idx = Math.max(idx - 1, 0); seek(); }
  else if (k === "v" || k === "i" || k === "n") {
    store[key(CELLS[idx])] = k.toUpperCase(); save();
    idx = Math.min(idx + 1, CELLS.length - 1); seek();
  } else if (k === "backspace") {
    delete store[key(CELLS[idx])]; save(); seek();
  } else return;
  e.preventDefault();
});
bexp.onclick = () => {
  let out = "window,cell,t_s,call\n";
  CELLS.forEach(c => { if (store[key(c)])
    out += `${c.window},${c.cell},${c.t_s},${store[key(c)]}\n`; });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([out], {type: "text/csv"}));
  a.download = "ball_visibility_contiguous_calls.csv"; a.click();
};
bimp.onclick = () => csvpick.click();
csvpick.onchange = () => {
  const r = new FileReader();
  r.onload = () => {
    r.result.trim().split("\n").slice(1).forEach(line => {
      const [w, c, , call] = line.split(",");
      if (call) store[w + ":" + (+c)] = call.trim().toUpperCase();
    });
    save(); seek();
  };
  r.readAsText(csvpick.files[0]);
};
seek();
</script></body></html>
"""


def selftest():
    cells = [{"window": "w01", "rally_cum": 1, "cell": k + 1,
              "t_s": 10.0 + k * STEP_S} for k in range(9)]
    cells += [{"window": "w02", "rally_cum": 1, "cell": k + 1,
               "t_s": 100.0 + k * STEP_S} for k in range(9)]  # out of play
    spans = {1: (9.5, 12.0)}
    calls = [{"window": "w01", "cell": k + 1, "call": c}
             for k, c in enumerate("VVVINVVNV")]
    calls += [{"window": "w02", "cell": 1, "call": "N"}]
    out = score(calls, cells, spans)
    # w01 cells 1..9 run 10.0..11.2, all inside (9.5,12.0); w02 outside
    assert "10 in-play" not in out and "(9 in-play" in out, out
    assert "visible 6/9 = 66.7%" in out, out          # in-play V count
    assert "inferred-only 1/9" in out, out
    assert "63.5%" in out, out                        # benchmark line
    assert "CIs overlap" in out, out
    p, lo, hi = wilson(129, 203)
    assert abs(p - 0.635) < 0.001 and lo > 0.56 and hi < 0.71
    # generator wiring against the real key, when present
    if KEY.exists():
        cs = load_cells()
        assert len(cs) == 270 and cs[0]["cell"] == 1
        assert abs(cs[1]["t_s"] - cs[0]["t_s"] - STEP_S) < 1e-9
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", metavar="CALLS_CSV")
    ap.add_argument("--key", default=str(KEY))
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--out", default=str(OUT_HTML))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    cells = load_cells(a.key)
    if a.score:
        rows = list(csv.DictReader(open(a.score)))
        print(score(rows, cells, rally_spans(a.labels)))
        return
    html = HTML.replace("__CELLS__", json.dumps(cells))
    Path(a.out).write_text(html)
    print(f"wrote {a.out}: {len(cells)} cells over "
          f"{len({c['window'] for c in cells})} windows. "
          f"Open it, load full_match.mp4.webm, mark V/I/N in order.")


if __name__ == "__main__":
    main()
