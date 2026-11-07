import unittest
import numpy as np
from resonance_morph import ResonancePath,MovingFilter,NeuralResonanceStream,minimum_phase_ratio,FREQUENCIES,FFT
from live_neural_grains import LiveNeuralGrainStream,DELAY
from compact_vocal_instrument import VocalFrame,BLOCK
from test_neural_vowel_instrument import fixture


def path_fixture():
    landmarks=np.array([[500,1400,2600],[350,2100,3000.]])
    profiles=[]
    for peaks in landmarks:
        power=sum(np.exp(-.5*((FREQUENCIES-p)/90)**2) for p in peaks)+1e-4
        profiles.append(np.log(power))
    return ResonancePath(profiles,landmarks)


class ResonanceTests(unittest.TestCase):
    def test_endpoints_exact_and_resonance_peak_moves(self):
        path=path_fixture()
        np.testing.assert_allclose(path.at(0),path.log_power[0],atol=1e-13)
        np.testing.assert_allclose(path.at(1),path.log_power[1],atol=1e-13)
        values=path.at(.5);band=(FREQUENCIES>1100)&(FREQUENCIES<2300)
        peak=FREQUENCIES[band][np.argmax(values[band])]
        self.assertLess(abs(peak-1750),25)

    def test_identity_filter_and_blocked_convolution(self):
        path=path_fixture();filt=MovingFilter(path)
        rng=np.random.default_rng(12);x=rng.normal(size=BLOCK*12)
        actual=np.concatenate([filt.push(b,0.) for b in x.reshape(-1,BLOCK)])
        np.testing.assert_allclose(actual,x,atol=1e-13)
        filt=MovingFilter(path);h=minimum_phase_ratio(path,.6)
        filt.previous=h;filt.alpha=.6
        actual=np.concatenate([filt.push(b,.6) for b in x.reshape(-1,BLOCK)])
        np.testing.assert_allclose(actual,np.convolve(x,h)[:len(x)],atol=1e-12)

    def test_immediate_controls_match_delayed_identity_source_and_clear(self):
        patch,decoder=fixture();path=path_fixture()
        empty=NeuralResonanceStream(patch,decoder,path)
        self.assertEqual(len(empty.finish()),0);self.assertFalse(empty.queue)
        actual=NeuralResonanceStream(patch,decoder,path);expected=LiveNeuralGrainStream([patch],decoder)
        for i in range(80):
            reference=expected.push(VocalFrame(200,(1.,)))
            if i==DELAY//BLOCK:reference*= (np.arange(BLOCK)+1)/BLOCK
            np.testing.assert_allclose(actual.push(200,0),reference,atol=1e-13)
        tail=actual.finish();expected_tail=expected.finish()
        np.testing.assert_allclose(tail[:DELAY],expected_tail,atol=1e-13)
        self.assertEqual(len(tail),DELAY+511);self.assertFalse(actual.queue)
        self.assertFalse(actual.source.readers[0].cache);self.assertFalse(np.any(actual.filter.history))
        self.assertEqual(len(actual.finish()),0)

    def test_live_changes_bounded_finite_and_invalid_controls_atomic(self):
        patch,decoder=fixture();a=NeuralResonanceStream(patch,decoder,path_fixture());b=NeuralResonanceStream(patch,decoder,path_fixture())
        for values in [(200,np.nan,1),(200,-1,1),(1000,.5,1),(200,.5,8)]:
            with self.assertRaises(ValueError):a.push(*values)
        for i in range(200):
            controls=(150+i%200,(i%50)/49,.8)
            x=a.push(*controls);y=b.push(*controls)
            np.testing.assert_array_equal(x,y);self.assertTrue(np.isfinite(x).all())
            self.assertEqual(len(a.queue),4);self.assertEqual(len(a.filter.history),511)
        a.cancel();self.assertFalse(a.queue);self.assertFalse(np.any(a.filter.history))
        with self.assertRaises(RuntimeError):a.push(200,.5)

    def test_full_envelope_diagnostic_validates_and_preserves_identity(self):
        path=path_fixture();a=MovingFilter(path);b=MovingFilter(path)
        with self.assertRaises(ValueError):a.push_envelope(np.ones(BLOCK),np.full(1025,np.nan))
        x=np.sin(np.arange(BLOCK)*.1)
        for _ in range(20):
            np.testing.assert_allclose(a.push_envelope(x,path.log_power[0]),b.push(x,0),atol=1e-13)


if __name__=='__main__':unittest.main()
