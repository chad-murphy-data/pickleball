"""Generate prediction-blind evaluation pages and score two sets of raw predictions."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from ball_label_batch import batch_html
from make_ball_audit_v2 import load_contacts, probe_video

ROOT=Path(__file__).resolve().parent.parent
SCHEMA='ball-eval-v1'


def read_predictions(path):
    with Path(path).open(newline='') as f:
        rows=list(csv.DictReader(f))
    result={}
    for r in rows:
        frame=int(r['frame'])
        if frame in result:
            raise ValueError('Duplicate prediction frame')
        x,y=float(r['predicted_x']),float(r['predicted_y'])
        if not all(math.isfinite(v) for v in (x,y)):
            raise ValueError('Nonfinite prediction')
        if r.get('context_overlaps_training')!='0' or r.get('exposure')!='unseen_center_same_video':
            raise ValueError('Evaluation predictions must be outside training context')
        result[frame]=dict(x=x,y=y)
    return result


def choose_frames(first,last,predictions):
    if last-first<80:
        raise ValueError('Interval too short for distinct evaluation frames')
    if any(f not in predictions for f in range(first,last+1)):
        raise ValueError('Predictions do not cover the contact interval')
    # Forty equal-width temporal strata, one midpoint from each (not a random sample).
    typical={first+math.floor((i+.5)*(last-first+1)/40) for i in range(40)}
    scored=[]
    for f in range(first+1,last+1):
        a,b=predictions[f-1],predictions[f]
        scored.append((math.hypot(a['x']-b['x'],a['y']-b['y']),f))
    challenge=set()
    # Ten temporal bins: seek one jump in each and label both sides of it.
    # Bins avoid spending the entire diagnostic budget on one burst of flicker.
    for i in range(10):
        lo=first+math.floor(i*(last-first+1)/10)
        hi=first+math.floor((i+1)*(last-first+1)/10)
        for score,f in sorted((p for p in scored if lo<=p[1]<hi),reverse=True):
            pair={f-1,f}
            if score>100 and not pair & (typical|challenge):
                challenge.update(pair)
                break
    for score,f in sorted(scored,reverse=True):
        if len(challenge)>=20:
            break
        if f not in typical|challenge:
            challenge.add(f)
    if len(typical)!=40 or len(challenge)!=20 or typical & challenge:
        raise ValueError('Cannot form disjoint 40/20 evaluation groups')
    return {f:'systematic' for f in typical}|{f:'challenge' for f in challenge}


def evaluation_html(config):
    html=batch_html(config)
    # Force blinding regardless of cache, imported proposals, or the P key.
    start=html.index('function proposalAllowed(')
    end=html.index('\nfunction render()',start)
    html=html[:start]+'function proposalAllowed(s){return false}'+html[end:]
    html=html.replace('id="proposalButton"','id="proposalButton" hidden')
    html=html.replace('id="proposalState"','id="proposalState" hidden')
    return html


def generate(args):
    meta=probe_video(args.video)
    if meta['name']!='full_match.mp4.webm':
        raise ValueError('This evaluation is anchored to full_match.mp4.webm')
    configs=[]; selections={}; hashes={}
    for rally in (25,27):
        path=Path(args.predictions)/f'r{rally}'/'predictions.csv'
        predictions=read_predictions(path)
        contacts=load_contacts(Path(args.contacts),rally)
        first,last=math.ceil(min(contacts)*meta['fps']),math.floor(max(contacts)*meta['fps'])
        cohorts=choose_frames(first,last,predictions)
        selections[str(rally)]={str(f):group for f,group in sorted(cohorts.items())}
        hashes[str(rally)]=hashlib.sha256(path.read_bytes()).hexdigest()
        configs.append(dict(schema=SCHEMA,session_id=f'ball_eval1_r{rally}',export_name=f'ball_eval1_r{rally}.csv',
            rally=rally,video=meta,contacts_s=[],samples=[dict(source_frame=f,nominal_t_s=round(f/meta['fps'],6),
            reason='evaluation',blind=True) for f in sorted(cohorts)]))
    pages=[evaluation_html(c) for c in configs]
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    plan=dict(schema=SCHEMA,purpose='evaluation_only',video=meta,cohorts=selections,
        selection_predictions_sha256=hashes,
        selection='40 temporal-stratum midpoints + 20 jump-focused frames per rally; challenge selection uses expanded-model predictions',
        scope='Two reviewed same-video rallies; not a population accuracy estimate or untouched holdout')
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    for c,page in zip(configs,pages):
        (out/f'r{c["rally"]}.html').write_text(page)
    print(f'Wrote {out}: 60 frames per rally, 120 total. Export ball_eval1_r25.csv and ball_eval1_r27.csv.')


def load_evaluation(paths,plan):
    labeled={}
    for path in paths:
        with Path(path).open(newline='') as f:
            for r in csv.DictReader(f):
                rally=str(int(r['rally'])); frame=int(r['source_frame']); key=(rally,frame)
                if r['schema']!=SCHEMA or r['video_name']!=plan['video']['name'] or abs(float(r['fps'])-plan['video']['fps'])>.001:
                    raise ValueError('Evaluation source/schema mismatch')
                if key in labeled or frame!=int(r['displayed_frame']):
                    raise ValueError('Duplicate label or decoded-frame mismatch')
                v,p=r['visibility'],r['presence']
                if v not in ('V','S','I','N') or p not in ({'visible'} if v in ('V','S') else {'occluded_inferred'} if v=='I' else {'unknown','absent'}):
                    raise ValueError('Invalid visibility/presence')
                if v!='N':
                    x,y=float(r['x']),float(r['y'])
                    if not (0<=x<plan['video']['width'] and 0<=y<plan['video']['height']):
                        raise ValueError('Invalid coordinates')
                labeled[key]=r
    expected={(r,int(f)) for r,fs in plan['cohorts'].items() for f in fs}
    if set(labeled)!=expected:
        raise ValueError(f'Expected complete evaluation: missing {len(expected-set(labeled))}, extra {len(set(labeled)-expected)}')
    return labeled


def metrics(rows,prefix):
    errors=[r[prefix+'_error_px'] for r in rows if r[prefix+'_error_px']!='']
    return dict(labeled_frames=len(rows),visible_localization_n=len(errors),
        median_error_px=statistics.median(errors) if errors else None,
        within_px={str(k):sum(e<=k for e in errors)/len(errors) if errors else None for k in (5,10,20)},
        inferred=sum(r['visibility']=='I' for r in rows),unknown=sum(r['presence']=='unknown' for r in rows),
        absent=sum(r['presence']=='absent' for r in rows))


def score(args):
    plan=json.loads(Path(args.plan).read_text()); labels=load_evaluation(args.labels,plan)
    paths={(25,'old'):args.old_r25,(27,'old'):args.old_r27,(25,'new'):args.new_r25,(27,'new'):args.new_r27}
    predictions={key:read_predictions(path) for key,path in paths.items()}
    for rally in (25,27):
        if hashlib.sha256(Path(paths[rally,'new']).read_bytes()).hexdigest()!=plan['selection_predictions_sha256'][str(rally)]:
            raise ValueError('New-model predictions differ from the frozen sampling plan')
    rows=[]
    for (rally,frame),label in sorted(labels.items()):
        row=dict(rally=int(rally),frame=frame,cohort=plan['cohorts'][rally][str(frame)],
                 visibility=label['visibility'],presence=label['presence'])
        for model in ('old','new'):
            p=predictions[int(rally),model].get(frame)
            if p is None: raise ValueError('Missing prediction for a labeled frame')
            row[model+'_x'],row[model+'_y']=p['x'],p['y']
            row[model+'_error_px']=math.hypot(p['x']-float(label['x']),p['y']-float(label['y'])) if label['visibility'] in ('V','S') else ''
        rows.append(row)
    reports={}
    for rally in (25,27):
        for cohort in ('systematic','challenge'):
            group=[r for r in rows if r['rally']==rally and r['cohort']==cohort]
            reports[f'r{rally}_{cohort}']={m:metrics(group,m) for m in ('old','new')}
            paired=[r for r in group if r['old_error_px']!='']
            reports[f'r{rally}_{cohort}']['paired_at_10px']=dict(
                improved=sum(r['old_error_px']>10 and r['new_error_px']<=10 for r in paired),
                regressed=sum(r['old_error_px']<=10 and r['new_error_px']>10 for r in paired))
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    with (out/'paired.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    report=dict(groups=reports,scope=plan['scope'],notes=[
        'Systematic and challenge groups are reported separately, by rally. No pooled headline accuracy.',
        'Only V/S labels contribute localization errors. I/N are not negative localization targets.',
        'Both models always emit a position. No presence-classification accuracy is computed.',
        'Challenge frames were selected using new-model instability; they are diagnostic, not an unbiased comparison.'],
        input_hashes={f'r{k[0]}_{k[1]}':hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in paths.items()},
        labels_sha256=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.labels])
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(reports,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate'); g.add_argument('--video',type=Path,required=True)
    g.add_argument('--predictions',default='data/vision/temporal_batch1_review')
    g.add_argument('--contacts',default=str(ROOT/'data/vision/contact_labels_chicago0725.csv'))
    g.add_argument('--out',default='data/vision/ball_eval1')
    s=sub.add_parser('score'); s.add_argument('--plan',required=True); s.add_argument('--labels',nargs='+',required=True)
    for flag in ('old-r25','old-r27','new-r25','new-r27'): s.add_argument('--'+flag,required=True)
    s.add_argument('--out',required=True)
    args=p.parse_args(); (generate if args.command=='generate' else score)(args)


if __name__=='__main__': main()
