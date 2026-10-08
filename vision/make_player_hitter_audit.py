"""Build three short player/hitter reviews from existing pose files (no inference).

python3 vision/make_player_hitter_audit.py
Then open the generated HTML and select full_match.mp4.webm.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from player_hitter_review import build

WINDOWS = [(6, 145.8, 151.2), (7, 170.5, 173.6), (10, 295.3, 297.4)]


def find_pose(root, rally):
    matches = sorted(root.rglob(f'r{rally:04d}.npz'))
    if not matches:
        raise ValueError(f'Cannot find r{rally:04d}.npz under {root}. Use --pose-dir with its folder.')
    hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in matches}
    if len(hashes) > 1:
        raise ValueError(f'Different copies of r{rally:04d}.npz found. Use --pose-dir to choose the intended folder.')
    return matches[0]


def assemble(root, state, contacts):
    windows = []
    for rally, start, end in WINDOWS:
        pose = find_pose(root, rally)
        z, names, report = build(pose, state, contacts, rally)
        if z['hw'].tolist() != [720, 1280] or abs(float(z['fps'][0]) - 60) > .01:
            raise ValueError('This audit expects the existing 1280x720, 60fps pose extraction')
        if z['t'][0] > start or z['t'][-1] < end:
            raise ValueError(f'Rally {rally}: poses do not cover the review window')
        frames = []
        for t in np.unique(z['t'][(z['t'] >= start - .03) & (z['t'] <= end + .03)]):
            people = []
            for i in np.flatnonzero(z['t'] == t):
                tid = int(z['track'][i])
                points = [[j, *np.round(z['kpt'][i, j], 1).tolist()]
                          for j in (9, 10, 15, 16) if z['kpc'][i, j] > .3]
                people.append(dict(track=tid, name=names.get(tid, 'Unassigned'),
                                   box=np.round(z['box'][i], 1).tolist(), points=points))
            frames.append(dict(t=float(t), people=people))
        windows.append(dict(rally=rally, start=start, end=end, frames=frames,
                            names=sorted(set(names.values())), provenance=report['provenance'],
                            events=[e for e in report['contacts'] if start <= e['t_s'] <= end]))
    config = dict(schema='player-hitter-audit-v1', windows=windows,
                  caveat='Development review at human contact times, with existing human track names. Not blinded or held-out.')
    config['audit_id'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    return config


HTML = r'''<!doctype html><meta charset="utf-8"><title>Player and hitter review</title>
<style>body{font:16px system-ui;background:#15191f;color:#eee;max-width:1150px;margin:24px auto;padding:0 16px}button,select,input{font:inherit;margin:5px;padding:8px}canvas{width:100%;background:#000}textarea{width:95%;height:60px}fieldset{margin:16px 0}#message{color:#ffd479}#v{display:none}.event{border-top:1px solid #667;padding:12px 0}small{color:#bec7d1}</style>
<h1>Player and hitter review</h1>
<p>Three short windows. First check whether each name stays with the correct player. Then review the hitter at each contact. Existing labels and arm-reach suggestions are shown for diagnosis; either can be wrong.</p>
<p>Select <b>full_match.mp4.webm</b> from your computer. The video stays on your computer.</p>
<input id="file" type="file" accept="video/*,.webm"><p id="message"></p>
<div id="tabs"></div><video id="v" muted playsinline></video><canvas id="canvas" width="1280" height="720"></canvas>
<div><button id="play">Play / pause</button><button id="back">Back ~1 frame</button><button id="forward">Forward ~1 frame</button><button id="restart">Restart window</button><select id="rate"><option value="0.25">Quarter speed</option><option value="0.5">Half speed</option><option value="1">Normal speed</option></select><label><input id="overlay" type="checkbox" checked>Show tracks</label></div>
<p id="clock"></p><small>Hollow circles: confident wrist/ankle estimates. Grey: unassigned track. Pose timestamps use an offset grid; overlays use the nearest sample within half a frame. Frame buttons are approximate browser seeks.</small>
<div id="questions"></div><button id="export">Export review</button>
<script>
const CFG=__CONFIG__, $=id=>document.getElementById(id), v=$('v'), ctx=$('canvas').getContext('2d');
const key='player-hitter:'+CFG.audit_id; let idx=0, ready=false, url=null, answers={}, shownTime=null;
try{answers=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
function persist(){try{localStorage.setItem(key,JSON.stringify(answers))}catch(e){$('message').textContent='Could not save browser progress. Export before closing.'}}
function answer(){const k=String(CFG.windows[idx].rally);return answers[k]||(answers[k]={identity:'',note:'',contacts:{}})}
function select(options,value,change){const s=document.createElement('select');for(const [val,label] of options){const o=document.createElement('option');o.value=val;o.textContent=label;s.append(o)}s.value=value;s.onchange=()=>{change(s.value);persist()};return s}
function seek(t){if(!ready)return;v.pause();shownTime=null;v.currentTime=Math.max(CFG.windows[idx].start,Math.min(CFG.windows[idx].end-.001,t))}
function renderQuestions(){const w=CFG.windows[idx],a=answer(),q=$('questions');q.replaceChildren();const h=document.createElement('h2');h.textContent=`Rally ${w.rally}: ${w.start}–${w.end} seconds`;q.append(h);
const p=document.createElement('p');p.textContent='Do the displayed names stay attached to the correct players throughout this window?';q.append(p,select([['','Choose…'],['correct','Yes, throughout'],['wrong','Wrong name or identity switch'],['missing','A player is missing / unassigned'],['uncertain','Cannot tell']],a.identity,x=>a.identity=x));
const note=document.createElement('textarea');note.placeholder='Identity notes: player, time, missing feet, or other problems';note.value=a.note;note.oninput=()=>{a.note=note.value;persist()};q.append(note);
for(const e of w.events){const k=String(e.shot),c=a.contacts[k]||(a.contacts[k]={hitter:'',note:''});const box=document.createElement('div');box.className='event';const b=document.createElement('button');b.textContent=`Contact ${e.shot} · ${e.t_s.toFixed(3)}s`;b.onclick=()=>seek(e.t_s);const replay=document.createElement('button');replay.textContent='Replay approach';replay.onclick=()=>{seek(e.t_s-.5)};const text=document.createElement('p');text.textContent=`Existing label: ${e.labeled_hitter}. Arm-reach suggestion: ${e.proposed_hitter||'Unknown'}${e.agrees?' (agree)':' (DISAGREE)'}. Who actually hits?`;
box.append(b,replay,text,select([['','Choose…'],...w.names.map(n=>[n,n]),['no_contact','No contact here'],['unknown','Cannot tell']],c.hitter,x=>c.hitter=x));const n=document.createElement('textarea');n.placeholder='Optional: wrong timing, decoy swing, occlusion, identity issue…';n.value=c.note;n.oninput=()=>{c.note=n.value;persist()};box.append(n);q.append(box)}}
function draw(t){shownTime=t;ctx.drawImage(v,0,0,1280,720);const w=CFG.windows[idx];let nearest=null;for(const f of w.frames)if(!nearest||Math.abs(f.t-t)<Math.abs(nearest.t-t))nearest=f;
if($('overlay').checked&&nearest&&Math.abs(nearest.t-t)<=.51/60){for(const p of nearest.people){const color=['#ffe060','#65baff','#80ed99','#f998ed'][w.names.indexOf(p.name)]||'#aaa';ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;const [x,y,x1,y1]=p.box;ctx.strokeRect(x,y,x1-x,y1-y);ctx.font='17px system-ui';ctx.fillText(`${p.track}: ${p.name}`,Math.max(0,x),Math.max(20,y-7));for(const [j,px,py] of p.points){ctx.beginPath();ctx.arc(px,py,5,0,Math.PI*2);ctx.stroke()}}}
$('clock').textContent=`Source time ${t.toFixed(3)}s`;}
function choose(i){v.pause();idx=i;renderQuestions();seek(CFG.windows[i].start)}
CFG.windows.forEach((w,i)=>{const b=document.createElement('button');b.textContent=`Window ${i+1}: rally ${w.rally}`;b.onclick=()=>choose(i);$('tabs').append(b)});
if(v.requestVideoFrameCallback){const cb=(now,meta)=>{if(ready){draw(meta.mediaTime);if(meta.mediaTime>=CFG.windows[idx].end)v.pause()}v.requestVideoFrameCallback(cb)};v.requestVideoFrameCallback(cb)}else{$('message').textContent='Use Chrome or Safari with video-frame callbacks for synchronized overlays.'}
$('file').onchange=()=>{ready=false;v.pause();if(url)URL.revokeObjectURL(url);const f=$('file').files[0];if(!f)return;url=URL.createObjectURL(f);v.src=url;v.load()};
v.onloadedmetadata=()=>{if(v.videoWidth!==1280||v.videoHeight!==720||Math.abs(v.duration-4820)>2){$('message').textContent='Video does not match expected 1280×720, approximately 4820-second source.';return}ready=true;v.playbackRate=Number($('rate').value);$('message').textContent='Video loaded. Watch each window, then answer below.';seek(CFG.windows[idx].start)};
v.onerror=()=>{$('message').textContent='Could not decode video. Try Chrome with the original full_match.mp4.webm.'};
$('play').onclick=()=>{if(!ready)return;if(v.paused){if(v.currentTime>=CFG.windows[idx].end)v.currentTime=CFG.windows[idx].start;v.play().catch(e=>$('message').textContent=e.message)}else v.pause()};
$('back').onclick=()=>seek((shownTime??v.currentTime)-1/60);$('forward').onclick=()=>seek((shownTime??v.currentTime)+1/60);$('restart').onclick=()=>seek(CFG.windows[idx].start);$('rate').onchange=()=>v.playbackRate=Number($('rate').value);$('overlay').onchange=()=>{if(ready&&shownTime!==null)draw(shownTime)};
$('export').onclick=()=>{persist();const reviews=CFG.windows.map(w=>({rally:w.rally,start:w.start,end:w.end,...(answers[String(w.rally)]||{})}));const complete=CFG.windows.every(w=>{const a=answers[String(w.rally)];return a&&a.identity&&w.events.every(e=>a.contacts[String(e.shot)]?.hitter)});const blob=new Blob([JSON.stringify({schema:CFG.schema,audit_id:CFG.audit_id,complete,reviews},null,2)],{type:'application/json'});const u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download='player_hitter_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);$('message').textContent=complete?'Complete review exported.':'Partial review exported; some answers are still missing.'};
renderQuestions();
</script>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pose-dir', type=Path, default=Path('.'))
    p.add_argument('--state', type=Path, default=Path('data/vision/state_labels_chicago0725.csv'))
    p.add_argument('--contacts', type=Path, default=Path('data/vision/contact_labels_chicago0725.csv'))
    p.add_argument('--out', type=Path, default=Path('data/vision/player_hitter_audit'))
    a = p.parse_args()
    cfg = assemble(a.pose_dir, a.state, a.contacts)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'review.html').write_text(HTML.replace('__CONFIG__', json.dumps(cfg).replace('<', '\\u003c')))
    manifest = {**cfg, 'windows': [{k:v for k,v in w.items() if k!='frames'} for w in cfg['windows']]}
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Wrote {a.out}/review.html. Select full_match.mp4.webm in the page.')


if __name__ == '__main__':
    main()
