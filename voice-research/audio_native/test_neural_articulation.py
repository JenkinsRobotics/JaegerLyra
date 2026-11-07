import unittest
import numpy as np
from neural_waveform import RATE,ControlMatrix,waveform_frames
from neural_performance import EncodedPerformance
from test_neural_vowel_instrument import ExactEncoder,ExactFrameDecoder
from neural_articulation import ArticulationUnit,NeuralArticulationStream,compile_unit
from live_neural_grains import DELAY
from compact_vocal_instrument import VocalFrame,BLOCK


def fixture():
    n=np.arange(round(.7*RATE));audio=np.zeros(len(n));rng=np.random.default_rng(73)
    noise=np.diff(rng.normal(size=round(.06*RATE)+1))*.02
    audio[round(.04*RATE):round(.10*RATE)]=noise
    start=round(.12*RATE);t=(n[start:]-start)/RATE
    audio[start:]=.15*np.cos(2*np.pi*200*t)+.05*np.cos(2*np.pi*400*t)
    frames,gains=waveform_frames(audio)
    pulses=np.arange(start,len(audio)-240,RATE/200)
    return ArticulationUnit(EncodedPerformance(ControlMatrix(frames,gains,len(audio)),pulses),60,85),ExactFrameDecoder(),audio


def render(unit,decoder,hz=200.,seconds=2.,gain=1.):
    stream=NeuralArticulationStream(unit,decoder)
    parts=[stream.push(VocalFrame(hz,(1.,),gain=gain)) for _ in range(round(seconds*RATE/BLOCK))]
    return np.r_[np.concatenate(parts),stream.finish()][DELAY:],stream


class ArticulationTests(unittest.TestCase):
    def test_known_onset_and_unlimited_neutral_sustain(self):
        unit,decoder,source=fixture();actual,stream=render(unit,decoder)
        np.testing.assert_allclose(actual[:len(source)-600],source[:-600],atol=1e-10)
        t=(np.arange(len(actual))-round(.12*RATE))/RATE
        expected=.15*np.cos(2*np.pi*200*t)+.05*np.cos(2*np.pi*400*t)
        np.testing.assert_allclose(actual[RATE:-1200],expected[RATE:-1200],atol=1e-9)
        self.assertEqual(len(actual),2*RATE);self.assertEqual(len(stream.finish()),0)

    def test_pitch_changes_voiced_attack_and_sustain_but_not_noise(self):
        unit,decoder,source=fixture();audio,_=render(unit,decoder,hz=250.)
        np.testing.assert_allclose(audio[:round(.10*RATE)],source[:round(.10*RATE)],atol=1e-12)
        for start,stop in ((.18,.38),(1.,1.8)):
            section=audio[round(start*RATE):round(stop*RATE)]
            spectrum=abs(np.fft.rfft(section*np.hanning(len(section))))
            hz=np.fft.rfftfreq(len(section),1/RATE)[np.argmax(spectrum)]
            self.assertLessEqual(abs(hz-250.),5.)

    def test_invalid_controls_atomicity_and_snapshot(self):
        unit,decoder,_=fixture();a=NeuralArticulationStream(unit,decoder);b=NeuralArticulationStream(unit,decoder)
        unit.performance.pulses[:]=np.nan;unit.performance.matrix.values[:]=np.nan
        for frame in (VocalFrame(float('nan'),(1.,)),VocalFrame(200.,(1.,),breath=.1)):
            with self.assertRaises(ValueError):a.push(frame)
        for _ in range(150):np.testing.assert_array_equal(a.push(VocalFrame(200.,(1.,))),b.push(VocalFrame(200.,(1.,))))

    def test_bounded_state_gain_and_cancel(self):
        unit,decoder,_=fixture();a=NeuralArticulationStream(unit,decoder,cache_frames=4)
        b=NeuralArticulationStream(unit,decoder,cache_frames=4)
        shapes={k:v.shape for k,v in vars(a).items() if isinstance(v,np.ndarray)}
        for i in range(500):
            hz=(150.,200.,250.)[(i//70)%3]
            np.testing.assert_allclose(a.push(VocalFrame(hz,(1.,)))*.5,b.push(VocalFrame(hz,(1.,),gain=.5)),atol=1e-14)
        self.assertEqual(shapes,{k:v.shape for k,v in vars(a).items() if isinstance(v,np.ndarray)})
        self.assertLessEqual(a.reader.maximum_cache_frames,4)
        a.cancel();self.assertFalse(a.reader.cache)
        for array in (a.pending,a.weights,a.background,a.voicing,a.gain):self.assertFalse(np.any(array))
        with self.assertRaises(RuntimeError):a.push(VocalFrame(200.,(1.,)))

    def test_source_only_late_vowel_selection_and_silence_rejection(self):
        _,_,audio=fixture();unit,meta=compile_unit(audio,ExactEncoder())
        self.assertGreaterEqual(meta['holdStartSample'],meta['lateRegionThresholdSample'])
        self.assertAlmostEqual(meta['sourceMedianPitchHz'],200.,delta=1.)
        unit.validate()
        with self.assertRaises(ValueError):compile_unit(np.zeros(RATE),ExactEncoder())


if __name__=='__main__':unittest.main()
