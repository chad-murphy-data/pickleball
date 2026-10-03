import unittest
import numpy as np
from learned_contact_experiment import targets, train, predict, transform, peaks, select_threshold


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

    def test_stricter_threshold(self):
        t=np.arange(5)*.1;s=np.array([.1,.6,.1,.9,.1])
        self.assertEqual(len(peaks(t,s,.8)),1)
        self.assertEqual(peaks(t,s,.95),[])

    def test_inner_training_excludes_validation_rally(self):
        from unittest.mock import patch
        datasets=[dict(rally=r,x=np.array([[float(r)],[float(r)]]),
                       y=np.array([0.,1.]),keep=np.ones(2,dtype=bool),
                       t=np.array([0.,1.]),labels=[dict(t_s=1.)]) for r in (6,7,8,9)]
        def fake_train(x,y):return set(x[:,0])
        def fake_predict(model,x):
            self.assertTrue(set(x[:,0]).isdisjoint(model))
            self.assertNotIn(10,model)  # reserved outer rally never supplied
            return np.array([.1,.85])
        with patch('learned_contact_experiment.train',fake_train),patch('learned_contact_experiment.predict',fake_predict):
            threshold,selection=select_threshold(datasets)
        self.assertAlmostEqual(threshold,.85)
        for fold in selection['folds']:
            self.assertNotIn(fold['validation_rally'],fold['training_rallies'])
            self.assertNotIn(10,fold['training_rallies'])


if __name__=='__main__':unittest.main()
