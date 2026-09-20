"""Create a fixed, modest multi-rally labeling batch for the next visual experiment."""
import argparse
import hashlib
import json
from pathlib import Path

from make_ball_audit_v2 import build_samples, load_contacts, probe_video, render_html

ROOT = Path(__file__).resolve().parent.parent
BATCH = 'ball_batch1'
WINDOWS = [(3,71.0,75.0), (10,299.5,303.5), (23,551.0,555.0)]


def batch_html(config):
    html = render_html(config)
    # Extend the verified seek labeler without changing earlier pages or caches.
    changes = [
        ('const KEY=`${CFG.schema}:${CFG.video.name}:r${CFG.rally}`;',
         'const KEY=`${CFG.schema}:${CFG.video.name}:${CFG.session_id}`;'),
        ("let showTrail=localStorage.getItem(KEY+':trail')!=='0';",
         "let showTrail=localStorage.getItem(KEY+':trail')==='1';"),
        ("input_method:method,proposal_visible:proposalWasVisible",
         "input_method:method,presence:visibility==='N'?(method==='key_absent'?'absent':'unknown'):(visibility==='I'?'occluded_inferred':'visible'),proposal_visible:proposalWasVisible"),
        ("else if(k==='n')record('N',null,null,'key');",
         "else if(k==='n')record('N',null,null,'key');else if(k==='o')record('N',null,null,'key_absent');"),
        ("input_method:r.input_method||'import',proposal_visible:",
         "presence:r.presence||(r.visibility==='N'?'unknown':r.visibility==='I'?'occluded_inferred':'visible'),input_method:r.input_method||'import',proposal_visible:"),
        ("'input_method','proposal_visible'];", "'input_method','proposal_visible','presence'];"),
        ("a.input_method,a.proposal_visible?1:0];", "a.input_method,a.proposal_visible?1:0,a.presence];"),
        ('a.download=`ball_labels_v2_r${CFG.rally}.csv`;', 'a.download=CFG.export_name;'),
        ('<kbd>N</kbd> when its position is unknown.',
         '<kbd>N</kbd> when its position is unknown. Press <kbd>O</kbd> only when definitely off-screen/absent.'),
        (" · label <b>${a?a.visibility:'—'}</b>",
         " · label <b>${a?a.visibility:'—'}</b>${a&&a.presence?' · '+a.presence:''}"),
        ("if(i<0)continue;labels[i]=", "if(i<0||r.video_name!==CFG.video.name||+r.rally!==CFG.rally)continue;labels[i]="),
    ]
    for old,new in changes:
        if html.count(old) != 1:
            raise ValueError('Labeler template changed; cannot safely apply batch extension: '+old)
        html = html.replace(old,new)
    return html


def make_plan(meta, contacts):
    if meta['name'] != 'full_match.mp4.webm':
        raise ValueError('This batch is anchored to full_match.mp4.webm')
    configs = []
    for rally,start,end in WINDOWS:
        times = load_contacts(Path(contacts),rally)
        samples = build_samples(start,end,meta['fps'],times,flight_stride=3,contact_radius_s=.20,blind_fraction=0)
        configs.append(dict(schema='ball-label-v2.2',session_id=f'{BATCH}_r{rally}',
            export_name=f'{BATCH}_r{rally}.csv',rally=rally,video=meta,
            window=dict(start_s=start,end_s=end),contacts_s=times,samples=samples))
    return dict(batch=BATCH,video=meta,train_rallies=[3,10,18,23],
                development_rallies=[25],reserved_rallies=[27],configs=configs,
                contacts_sha256=hashlib.sha256(Path(contacts).read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--contacts',type=Path,default=ROOT/'data/vision/contact_labels_chicago0725.csv')
    p.add_argument('--out',type=Path,default=ROOT/'data/vision/ball_batch1')
    args=p.parse_args()
    plan=make_plan(probe_video(args.video),args.contacts)
    pages=[(c,batch_html(c)) for c in plan['configs']]
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'plan.json').write_text(json.dumps(plan,indent=2))
    total=0
    for config,html in pages:
        dest=args.out/f'r{config["rally"]}.html'
        dest.write_text(html)
        n=len(config['samples']); total+=n
        print(f'{dest}: {n} samples; export {config["export_name"]}')
    print(f'{total} new samples total. Label one page at a time; export after each session.')


if __name__ == '__main__':
    main()
