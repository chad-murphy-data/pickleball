import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from ball_label_batch import batch_html
from ball_temporal_baseline import combined_labels, read_labels
from ball_temporal_overlay import windows_for


class BatchTests(unittest.TestCase):
    def row(self,frame=10,rally='3',visibility='V',presence='visible'):
        return dict(schema='ball-label-v2.2',video_name='full_match.mp4.webm',fps='60',
                    rally=rally,sample_index='0',source_frame=str(frame),displayed_frame=str(frame),
                    visibility=visibility,presence=presence,x='20' if visibility!='N' else '',y='30' if visibility!='N' else '')

    def write(self,path,rows):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def test_preserve_explicit_absence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'labels.csv'
            self.write(p,[self.row(visibility='N',presence='absent')])
            self.assertEqual(read_labels(p)[0]['presence'],'absent')
            self.write(p,[self.row(presence='absent')])
            with self.assertRaises(ValueError): read_labels(p)

    def test_combination_requires_plan_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.csv',Path(d)/'b.csv'
            self.write(a,[self.row()]); self.write(b,[self.row(frame=20,rally='10')])
            with self.assertRaisesRegex(ValueError,'--plan'): combined_labels([a,b])
            with self.assertRaisesRegex(ValueError,'duplicate'): combined_labels([a,a])

    def test_complete_split_and_frame_membership(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); paths=[]
            for rally,base,count in [('3',100,1),('10',200,1),('23',300,1),('18',1000,153)]:
                rows=[dict(self.row(base+i,rally),sample_index=str(i)) for i in range(count)]
                p=d/f'{rally}.csv'; self.write(p,rows); paths.append(p)
            plan=dict(train_rallies=[3,10,18,23],video=dict(name='full_match.mp4.webm',fps=60),
                      configs=[dict(rally=r,samples=[dict(source_frame=f)]) for r,f in [(3,100),(10,200),(23,300)]])
            p=d/'plan.json'; p.write_text(json.dumps(plan))
            rows,_=combined_labels(paths,p); self.assertEqual(len(rows),156)
            self.write(paths[0],[self.row(101)])
            with self.assertRaisesRegex(ValueError,'batch frames'): combined_labels(paths,p)
            self.write(paths[0],[self.row(100,'25')])
            with self.assertRaisesRegex(ValueError,'training split'): combined_labels(paths,p)

    def test_overlay_multirally_does_not_span_between_rallies(self):
        meta=dict(fps=60,rows=[dict(frame=100,rally='3'),dict(frame=150,rally='3'),dict(frame=1000,rally='18')])
        self.assertEqual(windows_for([3,18],meta,'unused'),{3:(100,150),18:(1000,1000)})

    def test_rendered_batch_export_and_keys(self):
        cfg=dict(schema='ball-label-v2.2',session_id='ball_batch1_r3',export_name='ball_batch1_r3.csv',
                 rally=3,video=dict(name='full_match.mp4.webm',fps=60),samples=[])
        h=batch_html(cfg)
        self.assertIn('${CFG.session_id}',h)
        self.assertIn("else if(k==='o')record('N',null,null,'key_absent')",h)
        self.assertIn("'proposal_visible','presence'",h)
        self.assertIn('a.proposal_visible?1:0,a.presence',h)
        script=h.split('<script>')[1].split('</script>')[0]
        subprocess.run(['node','--check'],input=script,text=True,check=True,capture_output=True)


if __name__ == '__main__': unittest.main()
