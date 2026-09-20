import tempfile
import unittest
import wave
from pathlib import Path
import numpy as np
from audio_contact_probe_v2 import flux,peak,auc


class AudioTests(unittest.TestCase):
    def test_offset_direction(self):
        t=np.arange(0,2,.005);s=np.zeros(len(t));s[np.argmin(abs(t-1.12))]=1
        self.assertEqual(peak(t,s,[1.],.12)[0],1)
        self.assertEqual(peak(t,s,[1.],-.12)[0],0)

    def test_auc_ties(self):
        self.assertEqual(auc([1,1],[1,1]),.5)
        self.assertEqual(auc([2,3],[0,1]),1.)
        self.assertIsNone(auc([1],[]))

    def test_stereo_flux_does_not_cancel_opposite_phase(self):
        sr=48000;rng=np.random.default_rng(0);x=np.zeros(sr,dtype=np.int16)
        x[sr//2:sr//2+480]=(rng.normal(size=480)*3000).astype(np.int16)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'audio.wav'
            with wave.open(str(p),'wb') as w:
                w.setnchannels(2);w.setsampwidth(2);w.setframerate(sr)
                w.writeframes(np.c_[x,-x].astype('<i2').tobytes())
            t,f,meta=flux(p)
        self.assertEqual(meta['channels'],2)
        self.assertGreater(f.max(),0)
        self.assertLess(abs(t[f.sum(axis=(1,2)).argmax()]-.5),.03)


if __name__=='__main__':unittest.main()
