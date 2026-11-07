import unittest
import numpy as np
from neural_syllable_stream import NeuralSyllableStream
from neural_articulation import NeuralArticulationStream,ArticulationUnit
from neural_performance import EncodedPerformance
from neural_waveform import ControlMatrix,waveform_frames,RATE
from test_neural_articulation import fixture
from compact_vocal_instrument import VocalFrame,BLOCK
from live_neural_grains import DELAY


class SyllableStreamTests(unittest.TestCase):
    def test_unreleased_matches_existing_controller(self):
        unit,decoder,_=fixture();old=NeuralArticulationStream(unit,decoder)
        new=NeuralSyllableStream(unit,decoder,round(.55*RATE))
        for i in range(400):
            frame=VocalFrame((200.,250.,150.)[(i//120)%3],(1.,),gain=.7)
            np.testing.assert_allclose(new.push(frame),old.push(frame),atol=1e-10)
        np.testing.assert_allclose(new.finish(),old.finish(),atol=1e-10)

    def test_release_preserves_analytic_voiced_tail_and_finishes_to_silence(self):
        unit,decoder,source=fixture();stream=NeuralSyllableStream(unit,decoder,round(.55*RATE))
        parts=[]
        for i in range(400):
            if i==300:
                self.assertTrue(stream.release());self.assertFalse(stream.release())
            parts.append(stream.push(VocalFrame(200.,(1.,))))
        audio=np.r_[np.concatenate(parts),stream.finish()][DELAY:]
        np.testing.assert_allclose(audio[round(1.52*RATE):round(1.65*RATE)],source[round(.57*RATE):],atol=1e-9)
        self.assertLess(max(abs(audio[round(1.68*RATE):])),1e-12)
        self.assertFalse(stream.reader.cache)
        with self.assertRaises(RuntimeError):stream.release()

    def test_unvoiced_coda_is_not_pitched(self):
        unit,decoder,source=fixture();source[round(.6*RATE):]=0
        source[round(.62*RATE):round(.68*RATE)]=np.random.default_rng(19).normal(0,.02,round(.06*RATE))
        frames,gains=waveform_frames(source);pulses=unit.performance.pulses[unit.performance.pulses<.6*RATE]
        changed=ArticulationUnit(EncodedPerformance(ControlMatrix(frames,gains,len(source)),pulses),unit.hold_first,unit.hold_last)
        stream=NeuralSyllableStream(changed,decoder,round(.55*RATE));parts=[]
        for i in range(360):
            if i==300:stream.release()
            parts.append(stream.push(VocalFrame(250.,(1.,))))
        audio=np.r_[np.concatenate(parts),stream.finish()][DELAY:]
        np.testing.assert_allclose(audio[round(1.57*RATE):round(1.63*RATE)],source[round(.62*RATE):round(.68*RATE)],atol=1e-12)

    def test_invalid_release_and_cancellation_preserve_contract(self):
        unit,decoder,_=fixture()
        for value in (True,float('nan'),0,unit.performance.matrix.sample_count):
            with self.assertRaises(ValueError):NeuralSyllableStream(unit,decoder,value)
        stream=NeuralSyllableStream(unit,decoder,round(.55*RATE),cache_frames=4)
        with self.assertRaises(ValueError):stream.release()
        self.assertIsNone(stream.release_position);self.assertEqual(stream.position,0)
        shapes={k:v.shape for k,v in vars(stream).items() if isinstance(v,np.ndarray)}
        for i in range(500):
            if i==350:stream.release()
            stream.push(VocalFrame(180.,(1.,)))
        self.assertEqual(shapes,{k:v.shape for k,v in vars(stream).items() if isinstance(v,np.ndarray)})
        self.assertLessEqual(stream.reader.maximum_cache_frames,4)
        stream.cancel();self.assertIsNone(stream.release_position)
        for array in (stream.pending,stream.weights,stream.background,stream.voicing,stream.gain):self.assertFalse(np.any(array))
        with self.assertRaises(RuntimeError):stream.push(VocalFrame(200.,(1.,)))


if __name__=='__main__':unittest.main()
