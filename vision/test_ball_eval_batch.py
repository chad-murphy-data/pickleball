import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

from ball_eval_batch import choose_frames, evaluation_html, load_evaluation, metrics, read_predictions, score
from ball_temporal_baseline import read_labels


class EvaluationTests(unittest.TestCase):
    def test_paired_scoring_end_to_end(self):
        import hashlib
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); label=d/'labels.csv'; pred_paths={}
            def write(path,rows):
                with path.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            labels=[]
            for rally in (25,27):
                labels.append(dict(schema='ball-eval-v1',video_name='v',fps='60',rally=str(rally),source_frame='10',
                    displayed_frame='10',visibility='V',presence='visible',x='20',y='30'))
                for model,x in [('old',100),('new',20)]:
                    p=d/f'{model}{rally}.csv'; pred_paths[f'{model}_r{rally}']=str(p)
                    write(p,[dict(frame='10',predicted_x=x,predicted_y=30,context_overlaps_training='0',exposure='unseen_center_same_video')])
            write(label,labels)
            plan=dict(video=dict(name='v',fps=60,width=1280,height=720),cohorts={'25':{'10':'systematic'},'27':{'10':'systematic'}},
                      selection_predictions_sha256={str(r):hashlib.sha256(Path(pred_paths[f'new_r{r}']).read_bytes()).hexdigest() for r in (25,27)},scope='test')
            p=d/'plan.json'; p.write_text(json.dumps(plan))
            with contextlib.redirect_stdout(io.StringIO()):
                score(SimpleNamespace(plan=str(p),labels=[str(label)],out=str(d/'results'),**pred_paths))
            result=json.loads((d/'results/report.json').read_text())
            self.assertEqual(result['groups']['r25_systematic']['paired_at_10px']['improved'],1)
            self.assertIsNone(result['groups']['r27_challenge']['new']['median_error_px'])

    def test_selection_is_disjoint_complete_and_repeatable(self):
        predictions={f:dict(x=500 if f%10==0 else f,y=0) for f in range(500)}
        picked=choose_frames(0,499,predictions)
        self.assertEqual(len(picked),120//2)
        self.assertEqual(sum(c=='systematic' for c in picked.values()),40)
        self.assertEqual(sum(c=='challenge' for c in picked.values()),20)
        self.assertEqual(picked,choose_frames(0,499,predictions))
        with self.assertRaises(ValueError): choose_frames(0,499,{})

    def test_visible_only_denominator(self):
        rows=[dict(old_error_px=5,visibility='V',presence='visible'),dict(old_error_px='',visibility='N',presence='absent')]
        m=metrics(rows,'old')
        self.assertEqual(m['visible_localization_n'],1)
        self.assertEqual(m['within_px']['10'],1)
        self.assertEqual(m['absent'],1)
        self.assertIsNone(metrics(rows[1:],'old')['median_error_px'])

    def test_eval_exports_rejected_by_training(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'labels.csv'
            row=dict(schema='ball-eval-v1',video_name='full_match.mp4.webm',fps='60',rally='25',
                     source_frame='10',displayed_frame='10',visibility='V',presence='visible',x='20',y='30')
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(row)); w.writeheader(); w.writerow(row)
            with self.assertRaises(ValueError): read_labels(p)
            plan=dict(video=dict(name='full_match.mp4.webm',fps=60,width=1280,height=720),cohorts={'25':{'10':'systematic'}})
            self.assertEqual(len(load_evaluation([p],plan)),1)
            with self.assertRaises(ValueError): load_evaluation([p,p],plan)
            plan['cohorts']['25']['11']='challenge'
            with self.assertRaises(ValueError): load_evaluation([p],plan)

    def test_render_blinds_all_proposals(self):
        cfg=dict(schema='ball-eval-v1',session_id='eval_r25',export_name='eval.csv',rally=25,
                 video=dict(name='full_match.mp4.webm',fps=60),samples=[])
        h=evaluation_html(cfg)
        self.assertIn('function proposalAllowed(s){return false}',h)
        self.assertIn('id="proposalButton" hidden',h)
        js=h.split('<script>')[1].split('</script>')[0]
        subprocess.run(['node','--check'],input=js,text=True,check=True,capture_output=True)


if __name__=='__main__': unittest.main()
