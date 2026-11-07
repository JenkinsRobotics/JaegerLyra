import unittest
import numpy as np
from test_neural_vowel_instrument import fixture,collect
from neural_vowel_instrument import NeuralVowelInstrument,VowelScore
from compact_vocal_instrument import VocalFrame,RATE,BLOCK
from live_neural_grains import LiveNeuralGrainStream,DELAY


class LiveGrainTests(unittest.TestCase):
    def test_constant_control_matches_independent_planned_instrument(self):
        patch,decoder=fixture()
        for hz in (150.,200.,250.):
            live=LiveNeuralGrainStream([patch],decoder)
            self.assertEqual(live.readers[0].decoded_frames,0)
            blocks=[live.push(VocalFrame(hz,(1.,))) for _ in range(200)]
            audio=np.r_[np.concatenate(blocks),live.finish()][DELAY:]
            score=VowelScore(seconds=1.,hz=(hz,))
            actual=audio*score.envelope(np.arange(RATE),RATE)
            expected=collect(NeuralVowelInstrument(patch,decoder,score))
            # Last source pulse's future neighbour is deliberately not planned.
            np.testing.assert_allclose(actual[1200:-1200],expected[1200:-1200],atol=1e-10)
            self.assertEqual(len(audio),RATE);self.assertEqual(len(live.finish()),0)

    def test_mixing_and_gain_with_delayed_controls(self):
        patch,decoder=fixture();a=LiveNeuralGrainStream([patch],decoder)
        b=LiveNeuralGrainStream([patch,patch],decoder)
        for i in range(100):
            weight=i/99
            np.testing.assert_allclose(a.push(VocalFrame(200.,(1.,)))*.5,
                b.push(VocalFrame(200.,(weight,1-weight),gain=.5)),atol=1e-14)
        np.testing.assert_allclose(a.finish()*.5,b.finish(),atol=1e-14)

    def test_no_future_score_bounded_state_and_cancel(self):
        patch,decoder=fixture();stream=LiveNeuralGrainStream([patch],decoder,cache_frames=4)
        shapes=[a.shape for a in (stream.pending,stream.weights,stream.controls)]
        for i in range(400):
            output=stream.push(VocalFrame((150.,250.,200.)[(i//50)%3],(1.,)))
            self.assertEqual(len(output),BLOCK);self.assertTrue(np.isfinite(output).all())
        self.assertEqual(shapes,[a.shape for a in (stream.pending,stream.weights,stream.controls)])
        self.assertLessEqual(stream.readers[0].maximum_cache_frames,4)
        stream.cancel();self.assertFalse(stream.readers[0].cache)
        self.assertFalse(np.any(stream.pending));self.assertFalse(np.any(stream.controls))
        self.assertEqual(len(stream.finish()),0)
        with self.assertRaises(RuntimeError):stream.push(VocalFrame(200.,(1.,)))

    def test_invalid_control_atomicity_and_source_snapshot(self):
        patch,decoder=fixture();a=LiveNeuralGrainStream([patch],decoder);b=LiveNeuralGrainStream([patch],decoder)
        patch.pulses[:]=np.nan;patch.matrix.values[:]=np.nan
        for frame in (VocalFrame(np.nan,(1.,)),VocalFrame(200.,(1.,),breath=.2)):
            with self.assertRaises(ValueError):a.push(frame)
        for _ in range(30):np.testing.assert_array_equal(a.push(VocalFrame(200.,(1.,))),b.push(VocalFrame(200.,(1.,))))


if __name__=='__main__':unittest.main()
