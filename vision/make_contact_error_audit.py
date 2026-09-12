"""Generate a 12-window diagnostic review using existing predictions and poses.

No video decoding or model inference. Open HTML and select the original video.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from make_player_hitter_audit import find_pose
from ball_hitter_baseline import read_ball


def select_items(report):
    items=[]
    for w in report['rallies']:
        r=w['rally'];s=w['score']
        misses=[e for e in s['missed_labels'] if e['shot']!=1]
        if misses:
            e=misses[len(misses)//2]
            items.append(dict(rally=r,kind='missed',t_s=e['t_s'],shot=e['shot']))
        extras=s['extra_events']
        if extras:
            e=extras[len(extras)//2]
            items.append(dict(rally=r,kind='extra',t_s=e['t_s']))
        if r in (7,10):
            excluded={e['t_s'] for e in extras}
            good=[e for e in w['events'] if e['t_s'] not in excluded]
            if good:items.append(dict(rally=r,kind='matched_comparison',t_s=good[len(good)//2]['t_s']))
        if r==6:
            serve=next((e for e in s['missed_labels'] if e['shot']==1),None)
            # If the serve was detected, use its matched event instead.
            if serve is None:
                serve=min(w['events'],key=lambda e:e['t_s'])
            items.append(dict(rally=r,kind='serve_camera_context',t_s=serve['t_s']))
    for i,x in enumerate(items):x['id']=f'item{i+1}_r{x["rally"]}'
    return items


def generate(a):
    report=json.loads(a.report.read_text());items=select_items(report)
    with a.state.open(newline='') as f:state=list(csv.DictReader(f))
    cache={}
    for w in report['rallies']:
        r=w['rally'];pose=find_pose(a.pose_dir,r);pred=a.predictions/f'r{r}'/'predictions.csv'
        # File paths may differ across machines; content must match this run.
        expected=set(w['provenance'].values())
        for path in (pose,pred,a.state):
            if hashlib.sha256(path.read_bytes()).hexdigest() not in expected:
                raise ValueError(f'{path} differs from the scored inputs')
        with np.load(pose,allow_pickle=False) as z:pose_data={k:z[k] for k in z.files}
        names={int(float(x['t_s'])):x['player'] for x in state if int(x['rally_cum'])==r and x['kind']=='track_assign'}
        cache[r]=(pose_data,read_ball(pred),names,w)
    for x in items:
        z,ball,names,w=cache[x['rally']];start=max(float(z['t'][0]),x['t_s']-.75);end=min(float(z['t'][-1]),x['t_s']+.75)
        x.update(start=start,end=end,names=sorted(set(names.values())),frames=[],
                 ball=ball[(ball[:,0]>=start-.2)&(ball[:,0]<=end+.03)].tolist(),
                 detected=[e['t_s'] for e in w['events'] if start<=e['t_s']<=end])
        for t in np.unique(z['t'][(z['t']>=start-.03)&(z['t']<=end+.03)]):
            people=[]
            for j in np.flatnonzero(z['t']==t):
                k,c=z['kpt'][j],z['kpc'][j];tid=int(z['track'][j]);paddles=[]
                for wrist,elbow in ((9,7),(10,8)):
                    if min(c[wrist],c[elbow])>.3:
                        paddles.append([*k[wrist].tolist(),*(k[wrist]+.5*(k[wrist]-k[elbow])).tolist()])
                people.append(dict(name=names.get(tid,'Unassigned'),box=z['box'][j].tolist(),paddles=paddles))
            x['frames'].append(dict(t=float(t),people=people))
    cfg=dict(schema='contact-error-audit-v1',items=items,
             source_report_sha256=hashlib.sha256(a.report.read_bytes()).hexdigest(),
             selection='One middle non-serve miss per rally, one middle extra where available, matched comparisons on 7/10, serve context on 6. Diagnostic selection, not representative rates.')
    cfg['audit_id']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:16]
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'review.html').write_text(HTML.replace('__CONFIG__',json.dumps(cfg).replace('<','\\u003c')))
    manifest={**cfg,'items':[{k:v for k,v in x.items() if k not in ('frames','ball')} for x in items]}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Wrote {a.out}/review.html: {len(items)} windows. Select full_match.mp4.webm.')


HTML=r'''<!doctype html><meta charset="utf-8"><title>Contact error review</title>
<style>body{font:16px system-ui;background:#15191f;color:#eee;max-width:1150px;margin:24px auto;padding:0 14px}button,select,input{font:inherit;padding:8px;margin:5px}canvas{width:100%}textarea{width:95%;height:90px}#msg{color:#ffd479}video{display:none}</style>
<h1>Contact error review</h1><p>Select your original full_match.mp4.webm. It stays on your computer. Judge the highlighted time, using the surrounding motion for context.</p>
<input id="file" type="file" accept="video/*,.webm"><p id="msg"></p><h2 id="title"></h2>
<button id="prev">Previous item</button><button id="next">Next item</button><video id="v" muted playsinline></video><canvas id="c" width="1280" height="720"></canvas>
<button id="play">Play / pause</button><button id="restart">Replay window</button><button id="center">Go to event</button><button id="back">Back ~1 frame</button><button id="forward">Forward ~1 frame</button>
<select id="rate"><option value="0.25">Quarter speed</option><option value="0.5">Half speed</option><option value="1">Normal speed</option></select><label><input id="over" type="checkbox" checked>Overlays</label><p id="clock"></p>
<p>Magenta hollow ring: predicted ball. Magenta trail: preceding 0.15 seconds of predictions (gaps and large jumps are disconnected). Coloured lines: wrist to estimated paddle tip, not actual paddle detections. Names come from earlier human assignments.</p>
<p>At the highlighted time, is there a real paddle contact?</p><select id="contact"><option value="">Choose…</option><option value="yes">Yes</option><option value="nearby">A contact nearby, but timing is off</option><option value="no">No contact</option><option value="unknown">Cannot tell</option></select>
<p>What does the ball prediction do around this event?</p><select id="tracking"><option value="">Choose…</option><option value="correct">Follows the ball</option><option value="jump">Jumps to another object</option><option value="hidden">Ball is hidden / too faint to judge</option><option value="missing">Prediction or overlay missing</option><option value="mixed">Mixed / other</option></select>
<p>Notes: actual hitter, contact time if shifted, camera cut, bounce, decoy swing, wrong paddle estimate, or anything else you notice.</p><textarea id="note"></textarea><p><button id="export">Export review</button></p>
<script>const CFG=__CONFIG__,$=id=>document.getElementById(id),v=$('v'),ctx=$('c').getContext('2d'),key='contact-errors:'+CFG.audit_id;let idx=0,ready=false,url=null,time=null,answers={};try{answers=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
function save(){const id=CFG.items[idx].id;answers[id]={contact:$('contact').value,tracking:$('tracking').value,note:$('note').value};try{localStorage.setItem(key,JSON.stringify(answers))}catch(e){$('msg').textContent='Browser progress could not be saved. Export before closing.'}}
function seek(t){if(!ready)return;v.pause();time=null;v.currentTime=Math.max(CFG.items[idx].start,Math.min(CFG.items[idx].end-.001,t))}
function show(){const x=CFG.items[idx],a=answers[x.id]||{};$('title').textContent=`${idx+1}/${CFG.items.length} · Rally ${x.rally} · ${x.kind.replaceAll('_',' ')} · event ${x.t_s.toFixed(3)}s`;for(const k of ['contact','tracking','note'])$(k).value=a[k]||'';seek(x.start)}
function nearest(a,t,field){let best=null;for(const x of a)if(best===null||Math.abs(field(x)-t)<Math.abs(field(best)-t))best=x;return best}
function draw(t){time=t;ctx.drawImage(v,0,0,1280,720);const x=CFG.items[idx];if($('over').checked){const f=nearest(x.frames,t,p=>p.t);if(f&&Math.abs(f.t-t)<=.51/60)for(const p of f.people){const color=['#ffe060','#65baff','#80ed99','#f998ed'][x.names.indexOf(p.name)]||'#aaa';ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1;const [a,b,c,d]=p.box;ctx.strokeRect(a,b,c-a,d-b);ctx.font='17px system-ui';ctx.fillText(p.name,Math.max(0,a),Math.max(20,b-6));for(const [wx,wy,px,py] of p.paddles){ctx.beginPath();ctx.moveTo(wx,wy);ctx.lineTo(px,py);ctx.stroke();ctx.beginPath();ctx.arc(px,py,7,0,7);ctx.stroke()}}
ctx.strokeStyle='#ff55ee';ctx.lineWidth=2;const trail=x.ball.filter(p=>p[0]<=t&&p[0]>=t-.15);ctx.beginPath();for(let i=0;i<trail.length;i++){const p=trail[i],prev=trail[i-1];if(!prev||p[0]-prev[0]>.025||Math.hypot(p[1]-prev[1],p[2]-prev[2])>100)ctx.moveTo(p[1],p[2]);else ctx.lineTo(p[1],p[2])}ctx.stroke();const b=nearest(x.ball,t,p=>p[0]);if(b&&Math.abs(b[0]-t)<.011){ctx.beginPath();ctx.arc(b[1],b[2],12,0,7);ctx.stroke()}}
$('clock').textContent=`Source ${t.toFixed(3)}s · event offset ${(t-x.t_s).toFixed(3)}s`+(x.detected.some(dt=>Math.abs(dt-t)<.05)?' · DETECTOR CONTACT':'');}
if(v.requestVideoFrameCallback){const cb=(n,m)=>{if(ready){draw(m.mediaTime);if(m.mediaTime>=CFG.items[idx].end)v.pause()}v.requestVideoFrameCallback(cb)};v.requestVideoFrameCallback(cb)}else $('msg').textContent='Use a browser supporting video-frame callbacks, such as current Chrome.';
$('file').onchange=()=>{ready=false;v.pause();if(url)URL.revokeObjectURL(url);const f=$('file').files[0];if(!f)return;url=URL.createObjectURL(f);v.src=url;v.load()};v.onloadedmetadata=()=>{if(v.videoWidth!==1280||v.videoHeight!==720||Math.abs(v.duration-4820)>2){$('msg').textContent='Video differs from the expected 1280×720, approximately 4820-second original.';return}ready=true;v.playbackRate=Number($('rate').value);$('msg').textContent='Ready. Overlay timestamps use the nearest saved sample; frame buttons are approximate browser seeks.';seek(CFG.items[idx].start)};v.onerror=()=>{$('msg').textContent='Video could not be decoded. Try Chrome and the original video.'};
$('prev').onclick=()=>{save();idx=Math.max(0,idx-1);show()};$('next').onclick=()=>{save();idx=Math.min(CFG.items.length-1,idx+1);show()};for(const k of ['contact','tracking','note'])$(k).oninput=save;
function play(){if(ready)v.play().catch(e=>$('msg').textContent=e.message)}$('play').onclick=()=>{if(v.paused){if(v.currentTime>=CFG.items[idx].end)seek(CFG.items[idx].start);play()}else v.pause()};$('restart').onclick=()=>{seek(CFG.items[idx].start);play()};$('center').onclick=()=>seek(CFG.items[idx].t_s);$('back').onclick=()=>seek((time??v.currentTime)-1/60);$('forward').onclick=()=>seek((time??v.currentTime)+1/60);$('rate').onchange=()=>v.playbackRate=Number($('rate').value);$('over').onchange=()=>{if(ready&&time!==null)draw(time)};
$('export').onclick=()=>{save();const reviews=CFG.items.map(x=>({id:x.id,rally:x.rally,t_s:x.t_s,...(answers[x.id]||{})})),complete=reviews.every(x=>x.contact&&x.tracking);const u=URL.createObjectURL(new Blob([JSON.stringify({schema:CFG.schema,audit_id:CFG.audit_id,complete,reviews},null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='contact_error_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);$('msg').textContent=complete?'Complete review exported.':'Partial review exported; some answers are missing.'};show();</script>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pose-dir',type=Path,default=Path('.'))
    p.add_argument('--predictions',type=Path,default=Path('data/vision/ball_hitter_predictions'))
    p.add_argument('--report',type=Path,default=Path('data/vision/auto_contact_run1/report.json'))
    p.add_argument('--state',type=Path,default=Path('data/vision/state_labels_chicago0725.csv'))
    p.add_argument('--out',type=Path,default=Path('data/vision/contact_error_audit'))
    generate(p.parse_args())


if __name__=='__main__':main()
