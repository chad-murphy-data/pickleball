"""Generate a prediction-blind, native-versus-model-resolution condition audit."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

from ball_eval_batch import load_evaluation, read_predictions


def select_items(labels, predictions, cohorts, per_rally=6):
    selected=[]
    for rally in ('25','27'):
        candidates=[]
        for (r,f),a in sorted(labels.items()):
            if r!=rally or a['visibility'] not in ('V','S'):
                continue
            p=predictions[r][f]
            e=math.hypot(p['x']-float(a['x']),p['y']-float(a['y']))
            candidates.append(dict(rally=int(r),frame=f,x=float(a['x']),y=float(a['y']),error_px=e,
                                   original_visibility=a['visibility'],cohort=cohorts[r][str(f)]))
        misses=[c for c in candidates if c['error_px']>20]
        controls=[c for c in candidates if c['error_px']<=10]
        n=min(per_rally,len(misses),len(controls))
        if not n:
            raise ValueError(f'No paired misses/controls for rally {rally}')
        # Spread misses through time, not just the largest errors.
        cases=[misses[math.floor((i+.5)*len(misses)/n)] for i in range(n)]
        for case in cases:
            control=min(controls,key=lambda c:(c['cohort']!=case['cohort'],abs(c['frame']-case['frame']),c['frame']))
            controls.remove(control)
            pair=f'r{rally}_f{case["frame"]}'
            selected.extend([dict(case,arm='miss',pair=pair),dict(control,arm='control',pair=pair)])
    return sorted(selected,key=lambda c:(c['rally'],c['frame']))


def crop_box(x,y,width,height,size=256):
    if width<size or height<size or width%2 or height%2:
        raise ValueError('Need even source dimensions at least 256 pixels')
    left=int(max(0,min(width-size,x-size/2)))//2*2
    top=int(max(0,min(height-size,y-size/2)))//2*2
    return left,top,left+size,top+size


def generate(args):
    import cv2
    plan=json.loads(Path(args.plan).read_text())
    labels=load_evaluation(args.labels,plan)
    predictions={r:read_predictions(Path(args.predictions)/f'r{r}'/'predictions.csv') for r in ('25','27')}
    for r in predictions:
        p=Path(args.predictions)/f'r{r}'/'predictions.csv'
        if hashlib.sha256(p.read_bytes()).hexdigest()!=plan['selection_predictions_sha256'][r]:
            raise ValueError('Predictions differ from the evaluated model')
    items=select_items(labels,predictions,plan['cohorts'])
    meta=plan['video']; width,height=meta['width'],meta['height']
    if (width,height)!=(1280,720) or Path(args.video).name!=meta['name']:
        raise ValueError('This comparison expects the labeled 1280x720 source video')
    cap=cv2.VideoCapture(str(args.video))
    if not cap.isOpened(): raise ValueError('Cannot open video')
    public=[]; needs={}
    for i,item in enumerate(items):
        box=crop_box(item['x'],item['y'],width,height)
        public.append(dict(id=f'r{item["rally"]}_f{item["frame"]}',rally=item['rally'],frame=item['frame'],
                           t_s=item['frame']/meta['fps'],native=[None]*5,model=[None]*5))
        for j,offset in enumerate(range(-2,3)):
            needs.setdefault(item['frame']+offset,[]).append((i,j,box))
    def png(bgr):
        ok,buf=cv2.imencode('.png',bgr)
        if not ok: raise ValueError('PNG encoding failed')
        return 'data:image/png;base64,'+base64.b64encode(buf).decode()
    try:
        actual=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if actual!=(width,height) or abs(cap.get(cv2.CAP_PROP_FPS)-meta['fps'])>.01:
            raise ValueError('Video geometry/FPS differs from labels')
        for frame in range(max(needs)+1):
            ok,bgr=cap.read()
            if not ok: raise ValueError('Video ended before required audit context')
            if frame%6000==0: print(f'Decoded {frame}/{max(needs)}',flush=True)
            if frame not in needs: continue
            # Resize the WHOLE frame just as training does, then crop it.
            small=cv2.resize(bgr,(640,360),interpolation=cv2.INTER_LINEAR)
            for i,j,(x0,y0,x1,y1) in needs[frame]:
                public[i]['native'][j]=png(bgr[y0:y1,x0:x1])
                public[i]['model'][j]=png(small[y0//2:y1//2,x0//2:x1//2])
                if j==2:
                    context=small.copy(); item=items[i]
                    cv2.circle(context,(round(item['x']/2),round(item['y']/2)),10,(255,255,0),1,cv2.LINE_AA)
                    public[i]['context']=png(context)
    finally:
        cap.release()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    audit_id=hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()[:16]
    # Error, arm and pair metadata are deliberately not embedded in the page.
    config=dict(audit_id=audit_id,items=public)
    encoded=json.dumps(config).replace('<','\\u003c')
    (out/'review.html').write_text(HTML.replace('__CONFIG__',encoded))
    (out/'manifest.json').write_text(json.dumps(dict(audit_id=audit_id,items=items,
        source_plan_sha256=hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
        labels_sha256=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.labels],
        method='Six time-spread >20px misses and six <=10px controls per rally; controls matched by cohort when possible, then time.',
        caveat='Outcome-balanced diagnostic selection; not representative condition-specific error rates. No training labels modified.'),indent=2))
    print(f'Wrote {out}/review.html: {len(items)} reviews, no ball clicking required.')


HTML=r'''<!doctype html><html><head><meta charset="utf-8"><title>Ball visibility review</title>
<style>body{font:16px system-ui;background:#14171c;color:#eee;margin:24px auto;max-width:1080px;padding:0 12px}button,select{font:inherit;padding:8px;margin:5px;background:#29313c;color:white;border:1px solid #718096;border-radius:5px}fieldset{margin:12px 0;border:1px solid #596574}label{display:inline-block;margin:8px}#pair{display:flex;gap:20px;flex-wrap:wrap}#pair img{width:384px;height:384px;image-rendering:pixelated}#context{max-width:640px;width:100%}small{color:#bac5d3}#msg{color:#ffd079}</style></head><body>
<h1>Ball visibility review</h1><p>Judge the <b>center frame</b>. Use neighboring frames for context. The cyan ring in the full image marks your earlier human label, not a model prediction. Crops have no marks.</p>
<div id="status"></div><button id="prev">Previous item</button><button id="next">Next item</button><button id="export">Export review</button>
<p id="msg"></p><img id="context" alt="Full center frame with human label ring">
<div id="pair"><div><h3>Native detail</h3><img id="native" alt="256 pixel native crop"></div><div><h3>Model input detail</h3><img id="model" alt="128 pixel crop after full-frame resize"></div></div>
<p><small>Same area at the same display size. Both enlarged with nearest-neighbor display; no artificial sharpening. Model panel is one of its five input frames.</small></p>
<div id="frames"></div><form id="form">
<fieldset><legend>In the center frame, what overlaps the ball? Select all that apply.</legend>
<label><input type="checkbox" name="condition" value="net_tape">Net tape</label><label><input type="checkbox" name="condition" value="net_mesh">Net mesh</label><label><input type="checkbox" name="condition" value="player">Player</label><label><input type="checkbox" name="condition" value="paddle">Paddle</label><label><input type="checkbox" name="condition" value="open">Open flight</label><label><input type="checkbox" name="condition" value="uncertain">Uncertain</label></fieldset>
<label>Native visibility <select id="visibility" required><option value="">Choose…</option><option>clear</option><option>smear</option><option>partly_visible</option><option>hidden</option><option>off_screen</option><option>uncertain</option></select></label>
<label>After resizing, can you still distinguish the ball? <select id="detail" required><option value="">Choose…</option value="yes">Yes</option><option value="harder">Yes, but harder</option><option value="no">No</option><option value="uncertain">Uncertain</option><option value="not_visible_natively">Not visible natively</option></select></label>
<p><label>Optional note <input id="note" style="width:400px" maxlength="1000"></label></p><button type="submit">Save and next</button></form>
<script>const CFG=__CONFIG__;const key='ball_condition:'+CFG.audit_id;let answers={};try{answers=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}let idx=CFG.items.findIndex(x=>!answers[x.id]);if(idx<0)idx=0;let offset=2;
const el=id=>document.getElementById(id);function persist(){try{localStorage.setItem(key,JSON.stringify(answers));return true}catch(e){el('msg').textContent='Browser could not save progress. Export before closing.';return false}}
function saveDraft(){const it=CFG.items[idx];if(!it)return;const conditions=[...document.querySelectorAll('[name=condition]:checked')].map(x=>x.value);const a={conditions,visibility:el('visibility').value,detail:el('detail').value,note:el('note').value};if(answers[it.id]||conditions.length||a.visibility||a.detail||a.note){answers[it.id]=a;persist()}}
function valid(a){return a&&a.conditions.length&&a.visibility&&a.detail&&!((a.conditions.includes('open')||a.conditions.includes('uncertain'))&&a.conditions.length>1)}
function images(){const it=CFG.items[idx];el('native').src=it.native[offset];el('model').src=it.model[offset];el('frames').replaceChildren();[-2,-1,0,1,2].forEach((o,j)=>{const b=document.createElement('button');b.textContent=o===0?'CENTER FRAME':`Frame ${o>0?'+':''}${o}`;b.style.borderColor=j===offset?'#ffe066':'#718096';b.onclick=()=>{offset=j;images()};el('frames').append(b)})}
function render(){const it=CFG.items[idx],a=answers[it.id]||{conditions:[],visibility:'',detail:'',note:''};offset=2;el('status').textContent=`Item ${idx+1}/${CFG.items.length} · Rally ${it.rally} · source frame ${it.frame} · ${it.t_s.toFixed(3)}s · ${Object.values(answers).filter(valid).length} complete`;el('context').src=it.context;document.querySelectorAll('[name=condition]').forEach(x=>x.checked=a.conditions.includes(x.value));for(const k of ['visibility','detail','note'])el(k).value=a[k];images()}
el('prev').onclick=()=>{saveDraft();idx=Math.max(0,idx-1);render()};el('next').onclick=()=>{saveDraft();idx=Math.min(CFG.items.length-1,idx+1);render()};
el('form').onsubmit=e=>{e.preventDefault();const conditions=[...document.querySelectorAll('[name=condition]:checked')].map(x=>x.value);const a={conditions,visibility:el('visibility').value,detail:el('detail').value,note:el('note').value};if(!valid(a)){el('msg').textContent='Choose the condition(s). Open flight and Uncertain must each stand alone.';return}answers[CFG.items[idx].id]=a;el('msg').textContent='';persist();if(idx<CFG.items.length-1)idx++;else el('msg').textContent='Last item saved. Export your review.';render()};
el('export').onclick=()=>{saveDraft();const rows=CFG.items.filter(it=>valid(answers[it.id])).map(it=>({id:it.id,rally:it.rally,frame:it.frame,...answers[it.id]}));const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify({audit_id:CFG.audit_id,expected_items:CFG.items.length,reviews:rows},null,2)],{type:'application/json'}));a.href=url;a.download='ball_condition_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};render();</script></body></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--plan',default='data/vision/ball_eval1/plan.json')
    p.add_argument('--labels',nargs='+',default=['data/vision/ball_eval1_r25.csv','data/vision/ball_eval1_r27.csv'])
    p.add_argument('--predictions',default='data/vision/temporal_batch1_review')
    p.add_argument('--out',default='data/vision/ball_condition_audit')
    generate(p.parse_args())


if __name__=='__main__':main()
