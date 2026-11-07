import unittest
import numpy as np
from neural_grain_window_stream import NeuralGrainWindowStream
from neural_syllable_stream import NeuralSyllableStream
from neural_articulation import ArticulationUnit
from neural_performance import EncodedPerformance
from neural_waveform import ControlMatrix,waveform_frames,RATE
from test_neural_articulation import fixture
from test_neural_vowel_instrument import ExactFrameDecoder
from compact_vocal_instrument import VocalFrame,BLOCK
from live_neural_grains import DELAY


def render(stream,hz=200.):
    parts=[]
    for i in range(400):
        if i==300:stream.release()
        parts.append(stream.push(VocalFrame(hz,(1.,))))
    return np.r_[np.concatenate(parts),stream.finish()][DELAY:]


class GrainWindowTests(unittest.TestCase):
    def test_minimum_rule_is_exact_baseline(self):
        unit,decoder,_=fixture()
        a=NeuralSyllableStream(unit,decoder,.55*RATE);b=NeuralGrainWindowStream(unit,decoder,.55*RATE)
        for i in range(400):
            if i==300:a.release();b.release()
            frame=VocalFrame((150.,200.,300.)[(i//70)%3],(1.,))
            np.testing.assert_array_equal(a.push(frame),b.push(frame))
        np.testing.assert_array_equal(a.finish(),b.finish())

    def test_native_periodic_signal_remains_intact(self):
        unit,decoder,_=fixture();baseline=render(NeuralSyllableStream(unit,decoder,.55*RATE))
        for policy in ('source','wide'):
            output=render(NeuralGrainWindowStream(unit,decoder,.55*RATE,window=policy))
            np.testing.assert_allclose(output[round(.7*RATE):round(1.45*RATE)],baseline[round(.7*RATE):round(1.45*RATE)],atol=1e-10)

    def test_note_change_preserves_pitch_and_outside_context(self):
        unit,decoder,_=fixture();baseline=render(NeuralSyllableStream(unit,decoder,.55*RATE),250.)
        for policy in ('source','wide'):
            stream=NeuralGrainWindowStream(unit,decoder,.55*RATE,window=policy)
            output=render(stream,250.);stop=int(stream.hold_start-DELAY)
            np.testing.assert_array_equal(output[:stop],baseline[:stop])
            np.testing.assert_array_equal(output[round(1.52*RATE):],baseline[round(1.52*RATE):])
            section=output[round(.8*RATE):round(1.4*RATE)]
            spectrum=abs(np.fft.rfft(section*np.hanning(len(section))))
            pitch=np.fft.rfftfreq(len(section),1/RATE)[np.argmax(spectrum)]
            self.assertAlmostEqual(pitch,250.,delta=2.)
            self.assertGreater(np.sqrt(np.mean((section-baseline[round(.8*RATE):round(1.4*RATE)])**2)),1e-4)
            self.assertFalse(stream.reader.cache)

    def test_wide_window_remains_bounded_at_low_source_pitch(self):
        n=np.arange(RATE);audio=.1*np.cos(2*np.pi*60*n/RATE);frames,gains=waveform_frames(audio)
        pulses=np.arange(400,RATE-400,400.)
        unit=ArticulationUnit(EncodedPerformance(ControlMatrix(frames,gains,len(audio)),pulses),10,20)
        stream=NeuralGrainWindowStream(unit,ExactFrameDecoder(),.7*RATE,window='wide',cache_frames=4)
        for _ in range(600):self.assertTrue(np.isfinite(stream.push(VocalFrame(800.,(1.,)))).all())
        self.assertGreater(stream.radius_clipped_grains,0);self.assertLessEqual(stream.maximum_grain_radius,DELAY-1)
        self.assertLessEqual(stream.reader.maximum_cache_frames,4);stream.cancel()
        for array in (stream.pending,stream.weights,stream.background,stream.voicing,stream.gain):self.assertFalse(np.any(array))
        with self.assertRaises(ValueError):NeuralGrainWindowStream(unit,ExactFrameDecoder(),.7*RATE,window='unknown')


if __name__=='__main__':unittest.main()
