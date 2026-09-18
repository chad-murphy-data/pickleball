import csv
import tempfile
import unittest
from pathlib import Path
from ball_temporal_baseline import read_labels


class LabelTests(unittest.TestCase):
    def run_rows(self, rows):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'labels.csv'
            with p.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            return read_labels(p)

    def row(self, **changes):
        r = dict(schema='ball-label-v2.1', video_name='full_match.mp4.webm', fps='60',
                 rally='18', sample_index='40', source_frame='27414', displayed_frame='27414',
                 visibility='N', x='', y='')
        r.update(changes)
        return r

    def test_absence_and_unknown(self):
        rows = self.run_rows([self.row(), self.row(sample_index='55', source_frame='27459', displayed_frame='27459')])
        self.assertEqual([r['presence'] for r in rows], ['absent', 'unknown'])

    def test_mismatch(self):
        with self.assertRaises(ValueError):
            self.run_rows([self.row(displayed_frame='27413')])

    def test_duplicate(self):
        with self.assertRaises(ValueError):
            self.run_rows([self.row(), self.row()])

    def test_legacy_quarantine(self):
        with self.assertRaises(ValueError):
            self.run_rows([self.row(schema='ball-label-v2.0')])

    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            self.run_rows([self.row(visibility='V', x='nan', y='10')])


if __name__ == '__main__':
    unittest.main()
