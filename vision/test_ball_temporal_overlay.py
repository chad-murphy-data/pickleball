import csv
from pathlib import Path
import tempfile
import unittest

from ball_temporal_overlay import (distinct_candidates, review_windows, exposure, human_fields,
                                   raw_predictions, training_frames, windows_for)


class OverlayTests(unittest.TestCase):
    def test_peaks_not_adjacent_pixels(self):
        import numpy as np
        heat = np.zeros((9,16), dtype=float)
        heat[4,4:7] = [.5,1,.5]
        heat[1,12] = .8
        peaks = distinct_candidates(heat, 1280,720,2,16)
        self.assertEqual([(p['x'],p['y']) for p in peaks], [(400,320),(960,80)])

    def test_separation_in_source_pixels_and_edge_peaks(self):
        import numpy as np
        heat = np.zeros((90,160), dtype=float)
        heat[0,0], heat[0,2], heat[89,159] = 1,.9,.8
        peaks = distinct_candidates(heat,1280,720,2,16)
        self.assertEqual([(p['x'],p['y']) for p in peaks], [(0,0),(1272,712)])
        # A smaller radius preserves the second nearby local peak.
        self.assertEqual(distinct_candidates(heat,1280,720,2,8)[1]['x'],16)

    def test_nan_rejected_and_ties_deterministic(self):
        import numpy as np
        with self.assertRaises(ValueError):
            distinct_candidates(np.array([[float('nan')]]),10,10)
        peaks = distinct_candidates(np.ones((3,3)),30,30,2,10)
        self.assertEqual([(p['x'],p['y']) for p in peaks],[(0,0),(20,0)])

    def test_review_times_are_source_times(self):
        self.assertEqual(review_windows(25,['605.3:606.2','610.6:612.5'],60),
                         {'25_window1':(36318,36372),'25_window2':(36636,36750)})
        for spec in ['610:609','nan:612','0:1']:
            with self.assertRaises(ValueError):
                review_windows(25,[spec],60)

    def test_exact_context_and_no_lookahead_shift(self):
        result = list(raw_predictions(iter(range(20)), 4, 7, lambda frames, c: (frames, c)))
        self.assertEqual([r[0] for r in result], [4,5,6,7])
        for center, image, (inputs, predicted_center) in result:
            self.assertEqual(image, center)
            self.assertEqual(predicted_center, center)
            self.assertEqual(inputs, list(range(center-2, center+3)))

    def test_truncated_video_rejected(self):
        with self.assertRaises(ValueError):
            list(raw_predictions(iter(range(8)), 4, 7, lambda f,c: c))

    def test_training_limit_and_context_are_distinct(self):
        ckpt = dict(manifest=dict(rows=[dict(frame=10,visibility='V'),dict(frame=11,visibility='N'),
                                       dict(frame=20,visibility='S')]), args=dict(limit=1))
        targets, context = training_frames(ckpt)
        self.assertEqual(targets, {10})
        self.assertEqual(exposure(12, targets, context), 'training_context_only')
        self.assertEqual(exposure(10, targets, context), 'training_target')
        self.assertEqual(exposure(20, targets, context), 'unseen_center_same_video')

    def test_absence_not_propagated_to_unlabeled_frames(self):
        self.assertEqual(human_fields(dict(visibility='N',presence='absent'))['presence'], 'absent')
        self.assertEqual(human_fields(dict(visibility='N'))['presence'], 'unknown')
        self.assertEqual(human_fields(None)['presence'], '')

    def test_known_and_contact_windows(self):
        manifest = dict(fps=60,rows=[dict(frame=100,rally='18'),dict(frame=120,rally='18')])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'contacts.csv'
            p.write_text('rally_cum,contact,source,t_tap_s,t_refined_s\n19,1,manual,10,\n19,1,manual,11,\n')
            self.assertEqual(windows_for([18,19],manifest,p), {18:(100,120),19:(540,720)})
            self.assertEqual(windows_for([18,19],manifest,p,2,15), {18:(100,120),19:(480,1560)})
            with self.assertRaises(ValueError):
                windows_for([19],manifest,p,1,float('nan'))
            with self.assertRaises(ValueError):
                windows_for([20],manifest,p)


if __name__ == '__main__':
    unittest.main()
