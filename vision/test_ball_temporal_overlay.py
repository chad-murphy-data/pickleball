import csv
from pathlib import Path
import tempfile
import unittest

from ball_temporal_overlay import exposure, human_fields, raw_predictions, training_frames, windows_for


class OverlayTests(unittest.TestCase):
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
