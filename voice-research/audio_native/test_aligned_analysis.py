import unittest
import numpy as np
from aligned_analysis import AlignedAnalysis
from streaming_voicebox import StreamingSynthesis


class AlignedAnalysisTests(unittest.TestCase):
    def test_emission_waits_for_declared_future_context(self):
        source=AlignedAnalysis()
        self.assertEqual(source.push(np.zeros(599)),[])
        frames=source.push([0.])
        self.assertEqual(len(frames),1)
        self.assertEqual(frames[0].start,0)
        self.assertEqual(frames[0].available_at,600)

    def test_reconstruction_preserves_pcm_across_chunking_and_final_flush(self):
        pcm=np.random.default_rng(71).integers(-20000,20000,3219,dtype=np.int16)
        source=AlignedAnalysis();frames=[]
        for at in range(0,len(pcm),127):frames+=source.push(pcm[at:at+127]/32768.)
        frames+=source.finish();sink=StreamingSynthesis()
        output=np.concatenate([sink.push(f) for f in frames])
        np.testing.assert_array_equal(np.rint(output*32768).astype(np.int16),pcm)
        self.assertEqual(frames[-1].available_at,len(pcm))
        self.assertEqual(source.finish(),[])

    def test_future_after_available_at_cannot_change_emitted_controls(self):
        samples=np.random.default_rng(19).normal(0,.03,3000)
        one=AlignedAnalysis().push(samples)
        samples[1560:]*=-2
        two=AlignedAnalysis().push(samples)
        for a,b in zip(one,two):
            if a.available_at<=1560:
                np.testing.assert_array_equal(a.residual,b.residual)
                np.testing.assert_array_equal(a.coefficients,b.coefficients)
                self.assertEqual(a.f0,b.f0)


if __name__=='__main__':unittest.main()
