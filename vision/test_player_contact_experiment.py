import unittest
from unittest.mock import patch
import numpy as np
from player_contact_experiment import player_features, player_targets, scores_for


class PlayerContactTests(unittest.TestCase):
    def test_features_never_mix_player_identities(self):
        seen=[]
        def fake_features(z,names,ball):
            self.assertEqual(len(set(names.values())),1)
            seen.append(names)
            return np.array([0.,1.]),np.ones((2,12))*next(iter(names))
        names={0:'A',4:'A',1:'B',2:'C',3:'D'}
        with patch('player_contact_experiment.features',fake_features):
            t,x,players=player_features({},names,None)
        self.assertEqual(x.shape,(2,4,12))
        self.assertEqual(players,['A','B','C','D'])
        self.assertEqual(seen[0],{0:'A',4:'A'})

    def test_only_actual_hitter_positive(self):
        y,keep=player_targets(np.array([.5,1.,1.1]),['A','B','C','D'],
                              [dict(t_s=1.,evaluation_hitter='B')])
        np.testing.assert_array_equal(y[1],[0,1,0,0])
        self.assertTrue(keep[1].all())
        self.assertFalse(keep[2].any())

    def test_prediction_keeps_player_axis(self):
        x=np.zeros((2,4,12))
        with patch('player_contact_experiment.predict',return_value=np.array([.1,.8,.2,.3,.9,.1,.2,.3])):
            s,who=scores_for(None,dict(x=x))
        np.testing.assert_array_equal(who,[1,0])
        np.testing.assert_allclose(s,[.8,.9])


if __name__=='__main__':unittest.main()
