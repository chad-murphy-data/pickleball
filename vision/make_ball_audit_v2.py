"""Generate a video-level, sampling-aware ball labeling page.

This is a fresh labeling path.  It does not change the original ball-audit
pages or their CSVs.

The v1 audit asked for a judgment on every nominal 30 fps frame.  V2 spends
labels where they are most useful: native-rate frames around known contacts
and a configurable sparse sample elsewhere.  It also records the media time
the browser actually displayed, so seek drift is measurable rather than
assumed away.

Typical use (run locally, where the uncommitted VOD lives):

    python vision/make_ball_audit_v2.py \
        --video full_match.mp4.webm --rally 17

The generated page is written to data/vision/ball_audit_v2_r17.html.  Open it
in a browser, load the same video, label, and export ball_labels_v2_r17.csv.

The page can optionally import a model proposal CSV.  Supported proposal
columns are either ``source_frame,x,y`` or ``frame,x,y``.  A deterministic
blind subset never shows proposals before the human judgment; this gives us
an estimate of proposal anchoring while retaining model-assisted labeling on
the remaining frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "vision"
CONTACTS = DATA / "contact_labels_chicago0725.csv"
SCHEMA_VERSION = "ball-label-v2.0"


def probe_video(path: Path) -> dict:
    """Return native video metadata using ffprobe.

    avg_frame_rate is retained as a rational string in the generated config
    and converted to float only for frame/time arithmetic.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,width,height:format=duration",
        "-of", "json", str(path),
    ]
    try:
        raw = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise SystemExit("ffprobe is required and was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffprobe could not read {path}: {exc.stderr}") from exc
    obj = json.loads(raw.stdout)
    if not obj.get("streams"):
        raise SystemExit(f"no video stream found in {path}")
    stream = obj["streams"][0]
    rate = stream.get("avg_frame_rate", "0/1")
    fps = float(Fraction(rate)) if rate not in ("0/0", "0") else 0.0
    if fps <= 0:
        raise SystemExit(f"invalid frame rate reported for {path}: {rate}")
    return {
        "name": path.name,
        "duration_s": float(obj.get("format", {}).get("duration", 0.0)),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": fps,
        "fps_rational": rate,
    }


def load_contacts(path: Path, rally: int) -> list[float]:
    out = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["rally_cum"]) != rally:
                continue
            if row.get("contact", "1") == "0":
                continue
            if row.get("source") not in ("manual", "divergent", "prefill"):
                continue
            value = row.get("t_refined_s") or row.get("t_tap_s")
            if value:
                out.append(float(value))
    if not out:
        raise SystemExit(f"no contact timestamps found for rally {rally} in {path}")
    return sorted(out)


def build_samples(
    start_s: float,
    end_s: float,
    fps: float,
    contacts: list[float],
    flight_stride: int = 3,
    contact_radius_s: float = 0.20,
    blind_fraction: float = 0.20,
    seed: int = 20260908,
) -> list[dict]:
    """Build sampled source frames with reasons and a deterministic blind arm."""
    if not 0 <= blind_fraction <= 1:
        raise ValueError("blind_fraction must be between 0 and 1")
    if flight_stride < 1:
        raise ValueError("flight_stride must be >= 1")
    first = math.ceil(start_s * fps)
    last = math.floor(end_s * fps)
    samples = []
    for frame in range(first, last + 1):
        t = frame / fps
        near_contact = any(abs(t - c) <= contact_radius_s for c in contacts)
        if not near_contact and (frame - first) % flight_stride:
            continue
        samples.append({
            "source_frame": frame,
            "nominal_t_s": round(t, 6),
            "reason": "contact_dense" if near_contact else "flight_sparse",
            "blind": False,
        })
    rng = random.Random(seed + 1009 * int(round(start_s * 1000)))
    order = list(range(len(samples)))
    rng.shuffle(order)
    n_blind = round(blind_fraction * len(samples))
    for i in order[:n_blind]:
        samples[i]["blind"] = True
    return samples


def render_html(config: dict) -> str:
    cfg = json.dumps(config, separators=(",", ":"))
    return HTML.replace("__CONFIG__", cfg)


HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>Ball labels v2</title>
<style>
body{font:14px system-ui;margin:0;background:#101114;color:#e4e6eb}
#wrap{max-width:1120px;margin:auto;padding:12px} h2{margin:4px 0 10px}
#drop{border:2px dashed #59606b;border-radius:8px;padding:14px;text-align:center;cursor:pointer;margin:8px 0}
#vbox{position:relative;background:#000;border-radius:7px;overflow:hidden}
video{width:100%;display:block} canvas{position:absolute;inset:0;cursor:crosshair}
.bar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:8px 0}
button{background:#292d34;color:#eee;border:1px solid #59606b;border-radius:5px;padding:5px 10px;cursor:pointer}
button:hover{background:#353b45}.spacer{flex:1}.warn{color:#ffb86b}.ok{color:#71db88}
kbd{background:#2b3038;border:1px solid #59606b;border-radius:4px;padding:1px 6px}
#mode{font-weight:700;color:#72b7ff}#proposalState{color:#c5a7ff}
#progress{height:10px;background:#252932;border-radius:5px;overflow:hidden}
#fill{height:100%;background:#43a568;width:0}.card{background:#181b20;border-radius:7px;padding:10px;margin-top:9px;line-height:1.55}
#reason{padding:2px 6px;border-radius:4px;background:#303641} code{color:#9fd3ff}
</style></head><body><div id="wrap">
<h2 id="title">Ball labels v2</h2>
<div id="drop">Load the locally stored source video<input id="videoPick" type="file" accept="video/*,.webm,.mp4,.mkv" hidden></div>
<div id="vbox"><video id="video" preload="auto"></video><canvas id="overlay"></canvas></div>
<div class="bar"><span id="status">—</span><span id="reason">—</span><span id="mode"></span><span class="spacer"></span><button id="proposalButton">Import proposals</button><input id="proposalPick" type="file" accept=".csv" hidden><button id="importButton">Import labels</button><input id="labelPick" type="file" accept=".csv" hidden><button id="exportButton">Export labels</button></div>
<div class="bar"><span id="videoCheck">No video loaded</span><span id="proposalState"></span></div>
<div id="progress"><div id="fill"></div></div>
<div class="card">
Click a clean ball for <b>V</b>. Press <kbd>S</kbd>, then click a visible smear; <kbd>I</kbd>, then click an occluded/inferred position; or <kbd>N</kbd> when its position is unknown. <kbd>A</kbd> accepts a visible model proposal. Arrows move between sampled frames; <kbd>,</kbd>/<kbd>.</kbd> jump ten samples; <kbd>Backspace</kbd> clears. Press <kbd>T</kbd> to show/hide the recent human trail and <kbd>P</kbd> to show/hide proposals. Proposal-blind frames remain blind until answered. Export after every session—the browser cache is a convenience, not a backup.
</div>
</div><script>
const CFG=__CONFIG__;
const KEY=`${CFG.schema}:${CFG.video.name}:r${CFG.rally}`;
const video=document.getElementById('video'),overlay=document.getElementById('overlay');
const drop=document.getElementById('drop'),videoPick=document.getElementById('videoPick');
const status=document.getElementById('status'),reason=document.getElementById('reason');
const modeEl=document.getElementById('mode'),fill=document.getElementById('fill');
const videoCheck=document.getElementById('videoCheck'),proposalState=document.getElementById('proposalState');
const proposalButton=document.getElementById('proposalButton'),proposalPick=document.getElementById('proposalPick');
const importButton=document.getElementById('importButton'),labelPick=document.getElementById('labelPick');
const exportButton=document.getElementById('exportButton');
let labels=JSON.parse(localStorage.getItem(KEY)||'{}');
let proposals={},idx=+(localStorage.getItem(KEY+':idx')||0),mode='V';
let showTrail=localStorage.getItem(KEY+':trail')!=='0';
let showProposals=localStorage.getItem(KEY+':proposals')!=='0';
let displayedTime=null,proposalWasVisible=false;
title.textContent=`Ball labels v2 — rally ${CFG.rally}`;
const save=()=>localStorage.setItem(KEY,JSON.stringify(labels));
const sample=()=>CFG.samples[idx];
const csvEscape=v=>`"${String(v??'').replaceAll('"','""')}"`;

drop.onclick=()=>videoPick.click(); drop.ondragover=e=>e.preventDefault();
drop.ondrop=e=>{e.preventDefault();loadVideo(e.dataTransfer.files[0])};
videoPick.onchange=()=>loadVideo(videoPick.files[0]);
function loadVideo(file){if(!file)return;video.src=URL.createObjectURL(file);video.dataset.filename=file.name;video.onloadedmetadata=()=>{const nameOK=file.name===CFG.video.name;const durOK=!CFG.video.duration_s||Math.abs(video.duration-CFG.video.duration_s)<=2;videoCheck.className=nameOK&&durOK?'ok':'warn';videoCheck.textContent=`${file.name} · ${video.duration.toFixed(2)}s · expected ${CFG.video.name}${durOK?'':' (duration mismatch)'}`;go(idx)}}
if(video.requestVideoFrameCallback){const watch=(now,meta)=>{displayedTime=meta.mediaTime;video.requestVideoFrameCallback(watch)};video.requestVideoFrameCallback(watch)}

function go(n){idx=Math.max(0,Math.min(CFG.samples.length-1,n));localStorage.setItem(KEY+':idx',idx);displayedTime=null;if(video.src){video.pause();video.currentTime=sample().nominal_t_s}render()}
function proposalAllowed(s){return showProposals&&proposals[s.source_frame]&&(!s.blind||labels[idx])}
function render(){const s=sample(),a=labels[idx];const done=Object.keys(labels).length;const drift=displayedTime==null?'':` · shown ${displayedTime.toFixed(4)}s · drift ${((displayedTime-s.nominal_t_s)*CFG.video.fps).toFixed(2)}f`;status.innerHTML=`sample <b>${idx+1}</b>/${CFG.samples.length} · source frame ${s.source_frame} · ${s.nominal_t_s.toFixed(4)}s${drift} · label <b>${a?a.visibility:'—'}</b>`;reason.textContent=s.reason+(s.blind?' · blind':'');modeEl.textContent=mode==='V'?'':`${mode} armed`;fill.style.width=`${100*done/CFG.samples.length}%`;proposalWasVisible=proposalAllowed(s)&&!a;proposalState.textContent=Object.keys(proposals).length?`${Object.keys(proposals).length} proposals loaded${s.blind&&!a?' · current frame blinded':''}`:'';draw()}
function draw(){overlay.width=video.clientWidth;overlay.height=video.clientHeight;const c=overlay.getContext('2d');c.clearRect(0,0,overlay.width,overlay.height);if(!video.videoWidth)return;const sx=overlay.width/video.videoWidth,sy=overlay.height/video.videoHeight,s=sample();if(showTrail){for(let j=Math.max(0,idx-15);j<=idx;j++){const a=labels[j];if(!a||a.visibility==='N')continue;c.beginPath();c.arc(a.x*sx,a.y*sy,j===idx?6:2.5,0,7);c.fillStyle=({V:'#48d36f',S:'#55b7ee',I:'#e29a43'}[a.visibility]||'#aaa')+(j===idx?'':'88');c.fill()}}if(proposalAllowed(s)){const p=proposals[s.source_frame];c.beginPath();c.arc(p.x*sx,p.y*sy,9,0,7);c.strokeStyle='#c982ff';c.lineWidth=2;c.stroke()}}
new ResizeObserver(draw).observe(video);video.onseeked=()=>{if(displayedTime==null)displayedTime=video.currentTime;render()};
function record(visibility,x=null,y=null,method='click'){labels[idx]={visibility,x,y,displayed_t_s:displayedTime??video.currentTime,input_method:method,proposal_visible:proposalWasVisible};mode='V';save();go(idx+1)}
overlay.onclick=e=>{if(!video.videoWidth)return;record(mode,Math.round(e.offsetX*video.videoWidth/overlay.width),Math.round(e.offsetY*video.videoHeight/overlay.height))};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;const k=e.key.toLowerCase();if(k==='arrowright')go(idx+1);else if(k==='arrowleft')go(idx-1);else if(k==='.')go(idx+10);else if(k===',')go(idx-10);else if(k==='s'){mode=mode==='S'?'V':'S';render()}else if(k==='i'){mode=mode==='I'?'V':'I';render()}else if(k==='n')record('N',null,null,'key');else if(k==='a'){const p=proposals[sample().source_frame];if(proposalAllowed(sample()))record('V',p.x,p.y,'accepted_proposal');else return}else if(k==='t'){showTrail=!showTrail;localStorage.setItem(KEY+':trail',showTrail?'1':'0');render()}else if(k==='p'){showProposals=!showProposals;localStorage.setItem(KEY+':proposals',showProposals?'1':'0');render()}else if(k==='backspace'){delete labels[idx];save();render()}else return;e.preventDefault()});

function parseCSV(text){const lines=text.trim().split(/\r?\n/),head=lines[0].split(',').map(x=>x.replaceAll('"','').trim());return lines.slice(1).map(line=>{const vals=line.split(',').map(x=>x.replace(/^"|"$/g,'').replaceAll('""','"'));return Object.fromEntries(head.map((h,i)=>[h,vals[i]??'']))})}
proposalButton.onclick=()=>proposalPick.click();proposalPick.onchange=()=>{const rd=new FileReader();rd.onload=()=>{for(const r of parseCSV(rd.result)){let f=r.source_frame!==undefined&&r.source_frame!==''?+r.source_frame:(r.t_s!==undefined&&r.t_s!==''?Math.round(+r.t_s*CFG.video.fps):+r.frame);if(Number.isFinite(f)&&r.x!==''&&r.y!=='')proposals[f]={x:+r.x,y:+r.y}}render()};rd.readAsText(proposalPick.files[0])};
importButton.onclick=()=>labelPick.click();labelPick.onchange=()=>{const rd=new FileReader();rd.onload=()=>{for(const r of parseCSV(rd.result)){const i=CFG.samples.findIndex(s=>s.source_frame===+r.source_frame);if(i<0)continue;labels[i]={visibility:r.visibility,x:r.x===''?null:+r.x,y:r.y===''?null:+r.y,displayed_t_s:r.displayed_t_s===''?null:+r.displayed_t_s,input_method:r.input_method||'import',proposal_visible:r.proposal_visible==='1'}}save();render()};rd.readAsText(labelPick.files[0])};
exportButton.onclick=()=>{const cols=['schema','video_name','video_duration_s','fps','fps_rational','rally','sample_index','source_frame','nominal_t_s','displayed_t_s','seek_drift_frames','x','y','visibility','sampling_reason','blind','input_method','proposal_visible'];let out=cols.join(',')+'\n';for(let i=0;i<CFG.samples.length;i++){const a=labels[i];if(!a)continue;const s=CFG.samples[i],drift=a.displayed_t_s==null?'':(a.displayed_t_s-s.nominal_t_s)*CFG.video.fps;const row=[CFG.schema,CFG.video.name,CFG.video.duration_s,CFG.video.fps,CFG.video.fps_rational,CFG.rally,i,s.source_frame,s.nominal_t_s,a.displayed_t_s,drift,a.x,a.y,a.visibility,s.reason,s.blind?1:0,a.input_method,a.proposal_visible?1:0];out+=row.map(csvEscape).join(',')+'\n'}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([out],{type:'text/csv'}));a.download=`ball_labels_v2_r${CFG.rally}.csv`;a.click()};
render();
</script></body></html>'''


def selftest() -> None:
    fps = 30000 / 1001
    contacts = [3.0, 7.0]
    a = build_samples(1.0, 9.0, fps, contacts, flight_stride=3,
                      contact_radius_s=0.20, blind_fraction=0.20, seed=4)
    assert a == build_samples(1.0, 9.0, fps, contacts, flight_stride=3,
                              contact_radius_s=0.20, blind_fraction=0.20,
                              seed=4)
    frames = {r["source_frame"] for r in a}
    first, last = math.ceil(fps), math.floor(9 * fps)
    for frame in range(first, last + 1):
        t = frame / fps
        dense = any(abs(t - c) <= 0.20 for c in contacts)
        if dense:
            assert frame in frames
        elif (frame - first) % 3:
            assert frame not in frames
    assert sum(x["blind"] for x in a) == round(0.20 * len(a))
    html = render_html({"schema": SCHEMA_VERSION, "rally": 1,
                        "video": {"name": "v.mp4", "duration_s": 10,
                                  "fps": fps, "fps_rational": "30000/1001"},
                        "samples": a})
    assert "__CONFIG__" not in html and "requestVideoFrameCallback" in html
    print(f"selftest OK — {len(a)} samples, {sum(x['blind'] for x in a)} blind")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path)
    ap.add_argument("--rally", type=int)
    ap.add_argument("--contacts", type=Path, default=CONTACTS)
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--pre", type=float, default=1.0)
    ap.add_argument("--post", type=float, default=1.0)
    ap.add_argument("--flight-stride", type=int, default=3)
    ap.add_argument("--contact-radius", type=float, default=0.20)
    ap.add_argument("--blind-fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.video or args.rally is None:
        ap.error("--video and --rally are required")
    meta = probe_video(args.video)
    contacts = load_contacts(args.contacts, args.rally)
    start = args.start if args.start is not None else contacts[0] - args.pre
    end = args.end if args.end is not None else contacts[-1] + args.post
    samples = build_samples(start, end, meta["fps"], contacts,
                            args.flight_stride, args.contact_radius,
                            args.blind_fraction, args.seed)
    cfg = {"schema": SCHEMA_VERSION, "rally": args.rally, "video": meta,
           "window": {"start_s": start, "end_s": end},
           "sampling": {"flight_stride": args.flight_stride,
                        "contact_radius_s": args.contact_radius,
                        "blind_fraction": args.blind_fraction,
                        "seed": args.seed},
           "contacts_s": contacts, "samples": samples}
    out = args.out or DATA / f"ball_audit_v2_r{args.rally}.html"
    out.write_text(render_html(cfg), encoding="utf-8")
    dense = sum(x["reason"] == "contact_dense" for x in samples)
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    print(f"  video: {meta['width']}x{meta['height']} @ {meta['fps']:.6f} fps "
          f"({meta['fps_rational']}), {meta['duration_s']:.2f}s")
    print(f"  rally {args.rally}: {len(contacts)} contacts; {len(samples)} labels "
          f"({dense} contact-dense, {len(samples)-dense} flight-sparse)")


if __name__ == "__main__":
    main()
