"""Generate the hand-coding instrument for the swing-proxy gate (Gate B).

Produces a single self-contained HTML page the user opens NEXT TO their
local video player, plus a spreadsheet template for anyone who prefers
Excel.  One row of output per shot: WHO hit it and WHAT it was
(serve/return/drive/drop/dink/speed-up/counter/lob/smash/other).  No
timestamps are asked of the human — order is enough, because the audio
pop train carries the timing and the label sequence aligns to it by
order.  These labels are EVALUATION-side only (model/vision_adjudication.md
§Gate B): the probe is scored against them, never tuned on them, and the
page embeds no tracker output whatsoever, so the blind rule of
vision/recall_audit.md is preserved by construction.

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
scrubbing, excluded from the core set.

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

    teams = {}  # match_id -> (frozenset t1 uuids, frozenset t2 uuids)
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
            "t0": mmss(r["v0"]), "t1": mmss(r["v1"]), "dur": r["dur"],
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
 #main{overflow-y:auto;padding:18px 22px}
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
 #scrub{font-size:34px;font-weight:700;letter-spacing:1px}
 .shot{display:flex;gap:6px;align-items:center;margin:5px 0;flex-wrap:wrap}
 .shot .idx{width:26px;text-align:right;color:var(--dim)}
 .pbtn{border-width:2px}
 .pbtn.t1{border-color:var(--t1)} .pbtn.t2{border-color:var(--t2)}
 .pbtn.t1.on{background:var(--t1);border-color:var(--t1);color:#06121f}
 .pbtn.t2.on{background:var(--t2);border-color:var(--t2);color:#1f1206}
 .tbtn{font-size:12px;padding:5px 7px}
 .del{color:var(--dim);border:none;background:none;font-size:15px}
 #help{background:var(--panel);border:1px solid var(--line);border-radius:9px;
       padding:12px 14px;margin:12px 0;font-size:13px}
 kbd{background:var(--line);border-radius:4px;padding:0 5px;font-size:12px}
 input[type=text]{background:var(--panel);border:1px solid var(--line);color:var(--ink);
       border-radius:7px;padding:6px 9px;width:100%;font:inherit}
 .bar{display:flex;gap:8px;margin:10px 0;flex-wrap:wrap;align-items:center}
 a{color:var(--acc)}
 .approx{color:var(--warn)}
</style>
<div id="side"></div>
<div id="main"></div>
<script>
const DATA = __PAYLOAD__;
const LSK = "shot_audit_chicago0725";
let store = JSON.parse(localStorage.getItem(LSK) || "{}");
let coreOnly = true, cur = null;

const save = () => localStorage.setItem(LSK, JSON.stringify(store));
const rallies = () => DATA.rallies.filter(r => !coreOnly || r.core);
const rget = c => DATA.rallies.find(r => r.cum === c);
const shots = c => (store[c] = store[c] || {shots: prefill(c), note: ""}).shots;

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

function side(){
  const el = document.getElementById("side");
  let h = `<h1>Shot audit</h1>
  <div class="small dim">MLP Chicago 2026-07-25<br>scrub in: ${DATA.video}</div>
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
  el.innerHTML = h;
}

function open_(c){ cur = c; side(); mainv(); }

function mainv(){
  const el = document.getElementById("main");
  if (cur === null){ el.innerHTML = intro(); return; }
  const r = rget(cur), ps = players(r), sh = shots(cur);
  let h = `<div class="bar"><span id="scrub">${r.approx ? "≈" : ""}${r.t0}</span>
    <span class="dim">→ ends ~${r.t1} (${r.dur.toFixed(0)}s)</span>
    <span class="badge">G${r.slot} R${r.rally} · #${r.cum}</span>
    <span class="badge">${DATA.games[r.slot].division}</span>
    ${r.orig ? '<span class="badge orig">original blind-10</span>' : ""}</div>
    <div class="small dim">score ${r.score} · serve: <b>${r.server}</b> → ${r.receiver}
    ${r.approx ? ' · <span class="approx">time approximated across a broadcast cut — trust the scorebug</span>' : ""}</div>
    <div id="help">${helprow(r)}</div>`;
  sh.forEach((s, i) => {
    h += `<div class="shot"><span class="idx">${i + 1}</span>`;
    for (const p of ps)
      h += `<button class="pbtn t${p.team} ${s.h === p.uuid ? "on" : ""}"
        onclick="setH(${i},'${p.uuid}')">${p.name.split(" ").slice(-1)[0]}</button>`;
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
  el.innerHTML = h;
}

function helprow(r){
  const t = Object.entries(DATA.typedefs).map(([k, v]) => `<b>${k}</b> — ${v}`).join("<br>");
  return `<b>Count every PADDLE STRIKE in order</b> (serve included; bounces are not shots;
  a fake or swing-and-miss is NOT a shot — note it instead). If the rally ends with a ball
  nobody touches, the last shot is the previous strike.<br><br>${t}<br><br>
  <span class="dim">Blind rule: don't look at any tracker output for these rallies.
  Players: <span style="color:var(--t1)">${DATA.games[r.slot].teams[0].map(p=>p.name).join(" / ")}</span> vs
  <span style="color:var(--t2)">${DATA.games[r.slot].teams[1].map(p=>p.name).join(" / ")}</span></span>`;
}

function intro(){
  return `<h1>How this works</h1>
  <div id="help"><b>1.</b> Open your local copy of the VOD (${DATA.video}).<br>
  <b>2.</b> Pick a rally on the left (core 20 first — the ten marked
  <span class="badge orig">blind10</span> are the pre-registered audit set).<br>
  <b>3.</b> Scrub the video to the big timestamp, watch the rally, and tap
  who hit + what it was for each shot in order. Shots 1–2 are prefilled from
  the referee log. Everything autosaves locally.<br>
  <b>4.</b> Download the CSV when done (or partway — partial coverage is fine)
  and drop it in the repo as <b>data/vision/shot_labels_chicago0725.csv</b>.<br><br>
  No timestamps needed from you — the order is enough; audio carries the timing.</div>`;
}

function setH(i, u){ shots(cur)[i].h = u; save(); mainv(); side(); }
function setT(i, t){ shots(cur)[i].t = t; save(); mainv(); side(); }
function addS(){ shots(cur).push({h: null, t: null}); save(); mainv(); }
function delS(i){ shots(cur).splice(i, 1); save(); mainv(); side(); }
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

side(); mainv();
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

    html = HTML.replace("__PAYLOAD__", json.dumps(payload))
    p = out / "shot_audit_chicago0725.html"
    p.write_text(html)
    print(f"wrote {p} ({len(payload['rallies'])} rallies, "
          f"{sum(1 for r in payload['rallies'] if r['core'])} core)")
    write_template(payload, out / "shot_audit_template.csv")


if __name__ == "__main__":
    main()
