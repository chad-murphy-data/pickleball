import subprocess
import unittest
from ball_condition_audit import HTML,crop_box,select_items


class AuditTests(unittest.TestCase):
    def test_crop_edges_and_resolution_alignment(self):
        for x,y in [(0,0),(1279,719),(641,351)]:
            a,b,c,d=crop_box(x,y,1280,720)
            self.assertEqual((c-a,d-b),(256,256))
            self.assertEqual((a%2,b%2),(0,0))
            self.assertTrue(a<=x<c and b<=y<d)
            self.assertEqual((c//2-a//2,d//2-b//2),(128,128))

    def test_balanced_distinct_and_excludes_inferred(self):
        labels={};predictions={};cohorts={}
        for rally in ('25','27'):
            predictions[rally]={};cohorts[rally]={}
            for f in range(30):
                labels[rally,f]=dict(x='10',y='10',visibility='I' if f==29 else 'V')
                predictions[rally][f]=dict(x=100 if f%2 else 10,y=10)
                cohorts[rally][str(f)]='systematic'
        items=select_items(labels,predictions,cohorts)
        self.assertEqual(len(items),24)
        self.assertEqual(len({(x['rally'],x['frame']) for x in items}),24)
        self.assertEqual(sum(x['arm']=='miss' for x in items),12)
        self.assertTrue(all(x['original_visibility']=='V' for x in items))

    def test_page_syntax_and_unmarked_crops(self):
        script=HTML.split('<script>')[1].split('</script>')[0].replace('__CONFIG__','{"audit_id":"test","items":[]}')
        subprocess.run(['node','--check'],input=script,text=True,check=True,capture_output=True)
        self.assertIn('<option>clear</option>',HTML)
        self.assertNotIn('error_px',HTML)
        self.assertNotIn("arm='miss'",HTML)


if __name__=='__main__':unittest.main()
