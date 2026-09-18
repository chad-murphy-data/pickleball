import unittest
import numpy as np
from audiovisual_contact_experiment import audio_features,AUDIO_DELAYS


class AudiovisualTests(unittest.TestCase):
    def test_fixed_delays_find_expected_peaks(self):
        t=np.arange(0,2,.005);score=np.zeros(len(t))
        for delay,value in zip(AUDIO_DELAYS,[.2,.4,.6,.8]):
            score[np.argmin(abs(t-(1+delay)))]=value
        out=audio_features(np.array([1.]),t,score)
        # Adjacent +/-25 ms windows overlap at their boundary by design.
        np.testing.assert_allclose(out[0],[.4,.4,.6,.8])

    def test_features_do_not_use_future_rally_labels(self):
        t=np.arange(0,1,.005);score=np.linspace(0,1,len(t))
        first=audio_features(np.array([.2,.4]),t,score)
        second=audio_features(np.array([.2,.4]),t,score)
        np.testing.assert_array_equal(first,second)

    def test_missing_audio_is_explicit(self):
        out=audio_features(np.array([10.]),np.arange(0,1,.005),np.ones(200))
        self.assertTrue(np.isnan(out).all())


if __name__=='__main__':unittest.main()
