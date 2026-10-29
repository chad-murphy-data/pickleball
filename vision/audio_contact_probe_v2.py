"""Audio diagnostic against human contact labels, not vision track endpoints.

Fixed broadband spectral-flux signal. Offset selected on other rallies only.
This measures signal usefulness, not a trained audio classifier.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, mannwhitneyu


def flux(path):
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=sample_rate,channels','-of','json',str(path)]))['streams'][0]
    sr=int(meta['sample_rate']);channels=int(meta['channels'])
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-f','f32le','-acodec','pcm_f32le','-'])
    x=np.frombuffer(raw,dtype='<f4').reshape(-1,channels)
    nfft=1024;hop=round(sr*.005);window=np.hanning(nfft)
    frequencies=np.fft.rfftfreq(nfft,1/sr);bands=[(300,1200),(1200,3000),(3000,6000),(6000,12000)]
    masks=[(frequencies>=a)&(frequencies<b) for a,b in bands]
    starts=np.arange(0,len(x)-nfft+1,hop);out=np.zeros((len(starts),channels,len(bands)));previous=None
    for begin in range(0,len(starts),256):
        s=starts[begin:begin+256];frames=x[s[:,None]+np.arange(nfft)[None,:]]*window[None,:,None]
        magnitude=np.abs(np.fft.rfft(frames,axis=1))
        first=magnitude[:1] if previous is None else previous
        change=np.maximum(np.diff(np.concatenate([first,magnitude],axis=0),axis=0),0)
        for j,mask in enumerate(masks):out[begin:begin+len(s),:,j]=change[:,mask,:].sum(axis=1)
        previous=magnitude[-1:]
    return (starts+nfft/2)/sr,out,dict(sample_rate=sr,channels=channels,duration_s=len(x)/sr)


def peak(t,s,centers,offset,window=.04):
    return np.array([s[abs(t-(c+offset))<=window].max() for c in centers])


def auc(pos,neg):
    return float(mannwhitneyu(pos,neg).statistic/(len(pos)*len(neg))) if len(pos) and len(neg) else None


def run(a):
    t,f,meta=flux(a.audio);truth=json.loads(a.hitter_report.read_text());base=json.loads(a.baseline_report.read_text())
    datasets=[]
    for w in truth['rallies']:
        r=w['rally'];contacts=np.array([e['t_s'] for e in w['events']]);lo=contacts.min()-.75;hi=contacts.max()+.75
        m=(t>=lo)&(t<=hi);tt=t[m];ff=f[m]
        # Per-band scale uses the unlabeled clip. Average channel energies rather
        # than mixing waveforms, avoiding stereo phase cancellation.
        combined=np.log1p(ff.mean(axis=1)/np.maximum(np.median(ff.mean(axis=1),axis=0),1e-8)).mean(axis=1)
        ranks=rankdata(combined)/len(combined)
        controls=np.arange(contacts.min(),contacts.max(),.1)
        controls=controls[np.min(abs(controls[:,None]-contacts[None,:]),axis=1)>.25]
        old=next(x for x in base['rallies'] if x['rally']==r)
        if old['provenance']!=w['provenance']:raise ValueError('Visual reports describe different inputs')
        extras=np.array([e['t_s'] for e in old['score']['extra_events']])
        missed=np.array([e['t_s'] for e in old['score']['missed_labels']])
        # Extra events outside the base contact span are retained by the clip pad.
        full_lo=min(lo,extras.min()-.4) if len(extras) else lo
        full_hi=max(hi,extras.max()+.4) if len(extras) else hi
        if full_lo<tt[0] or full_hi>tt[-1]:
            m=(t>=full_lo)&(t<=full_hi);tt=t[m];ff=f[m]
            combined=np.log1p(ff.mean(axis=1)/np.maximum(np.median(ff.mean(axis=1),axis=0),1e-8)).mean(axis=1)
            ranks=rankdata(combined)/len(combined)
        datasets.append(dict(rally=r,t=tt,s=ranks,contacts=contacts,controls=controls,extras=extras,missed=missed))
    offsets=np.arange(-.25,.2501,.005);results=[]
    for d in datasets:
        training=[x for x in datasets if x['rally']!=d['rally']]
        objective=[]
        for off in offsets:
            objective.append(np.mean([peak(x['t'],x['s'],x['contacts'],off).mean()-peak(x['t'],x['s'],x['controls'],off).mean() for x in training]))
        best=int(np.argmax(objective));offset=float(offsets[best])
        pos=peak(d['t'],d['s'],d['contacts'],offset);neg=peak(d['t'],d['s'],d['controls'],offset)
        extra=peak(d['t'],d['s'],d['extras'],offset);miss=peak(d['t'],d['s'],d['missed'],offset)
        record=dict(rally=d['rally'],offset_s=offset,offset_training_rallies=[x['rally'] for x in training],
            contacts=len(pos),controls=len(neg),extras=len(extra),auc_contacts_vs_controls=auc(pos,neg),
            auc_contacts_vs_visual_extras=auc(pos,extra),median_contact_score=float(np.median(pos)),
            median_control_score=float(np.median(neg)),median_extra_score=float(np.median(extra)) if len(extra) else None,
            contact_times=d['contacts'].tolist(),contact_scores=pos.tolist(),control_scores=neg.tolist(),
            extra_times=d['extras'].tolist(),extra_scores=extra.tolist(),missed_times=d['missed'].tolist(),missed_scores=miss.tolist(),
            offset_curve=objective)
        results.append(record)
        print({k:record[k] for k in ('rally','offset_s','auc_contacts_vs_controls','auc_contacts_vs_visual_extras','median_contact_score','median_control_score')},flush=True)
    positives=np.concatenate([r['contact_scores'] for r in results]);negatives=np.concatenate([r['control_scores'] for r in results]);extras=np.concatenate([r['extra_scores'] for r in results])
    summary=dict(contacts=len(positives),controls=len(negatives),visual_extras=len(extras),
        pooled_auc_vs_controls=auc(positives,negatives),pooled_auc_vs_visual_extras=auc(positives,extras))
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'report.json').write_text(json.dumps(dict(audio=meta,summary=summary,rallies=results,
        provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.audio,a.hitter_report,a.baseline_report)},
        method='Fixed four-band spectral flux; channel-energy averaging; clip percentile score; maximum within +/-40ms; offset chosen on other rallies',
        caveats=['AUC is discrimination, not contact precision or recall','Controls are correlated within rallies',
                 'Audio extraction time origin may differ slightly; offsets also include sound travel and label uncertainty',
                 'No audio model trained; no threshold chosen; same-match exploratory diagnostic']),indent=2))
    print('TOTAL',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('audio','hitter-report','baseline-report','out'):p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
