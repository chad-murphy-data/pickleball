import unittest
import numpy as np
from learned_contact_experiment import targets, train, predict, transform, peaks


class LearnedTests(unittest.TestCase):
    def test_target_buffer(self):
        y,keep=targets(np.array([.5,.9,1.,1.1,1.5]),[dict(t_s=1.)])
        np.testing.assert_array_equal(y,[0,0,1,0,0])
        np.testing.assert_array_equal(keep,[True,False,True,False,True])

    def test_learns_and_handles_missing(self):
        x=np.array([[-2.,np.nan],[-1.,np.nan],[1.,np.nan],[2.,np.nan]])
        model=train(x,np.array([0.,0.,1.,1.]))
        p=predict(model,x)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue((p[:2]<.5).all() and (p[2:]>.5).all())

    def test_test_data_does_not_refit_scaler(self):
        model=train(np.array([[-1.],[1.]]),np.array([0.,1.]))
        before=[a.copy() for a in model]
        predict(model,np.array([[1000000.],[np.nan]]))
        for a,b in zip(before,model):np.testing.assert_array_equal(a,b)

    def test_peak_separation(self):
        p=peaks(np.arange(9)*.05,np.array([.1,.6,.1,.9,.1,.1,.1,.8,.1]))
        self.assertEqual(len(p),2)
        self.assertAlmostEqual(p[0]['t_s'],.15)


if __name__=='__main__':unittest.main()
