import unittest
import numpy as np
from aligned_analysis import AlignedAnalysis
from continuous_source import ContinuousSource
from streaming_voicebox import VoiceboxConfig
from noise_shape import noise_descriptor
from predictive_controls import (to_reflection, from_reflection, ControlSpace,
                                 FreeRunningControls, AnchoredControls, visible_outside_gap)


def analyze(audio):
    source = AlignedAnalysis(VoiceboxConfig(order=32))
    return source.push(audio)+source.finish()


class Spy:
    def __init__(self):
        self.events=[]
    def predict(self, context):
        self.events.append(('predict', context.copy()))
        return np.ones(1)*.1
    def observe(self, context, target, learn=True):
        self.events.append(('learn', target.copy()))


class PredictiveControlTests(unittest.TestCase):
    def test_reflection_roundtrip_and_poles(self):
        rng=np.random.default_rng(14)
        for order in (2,12,32):
            for _ in range(10):
                k=rng.uniform(-.6,.6,order)
                a=from_reflection(k)
                np.testing.assert_allclose(to_reflection(a),k,atol=1e-10)
                self.assertLess(np.max(np.abs(np.roots(a))),1.)
        with self.assertRaises(ValueError):from_reflection([1.])
        with self.assertRaises(ValueError):to_reflection([1.,1.1])

    def test_missing_audio_cannot_leak_through_neighboring_analysis(self):
        rng=np.random.default_rng(19)
        audio=rng.normal(0,.08,14400)
        changed=audio.copy();changed[4800:7200]=rng.normal(0,.3,2400)
        original_frames=analyze(audio);changed_frames=analyze(changed)
        mask=visible_outside_gap(original_frames,4800,7200,VoiceboxConfig(order=32))
        # A nominal 100 ms gap removes 140 ms of controls because of context.
        self.assertEqual(int((~mask).sum()),14)
        for a,b,visible in zip(original_frames,changed_frames,mask):
            if visible:
                np.testing.assert_array_equal(a.coefficients,b.coefficients)
                np.testing.assert_array_equal(a.excitation_window,b.excitation_window)
                np.testing.assert_array_equal(a.residual,b.residual)
                self.assertEqual(a.f0,b.f0)
                self.assertEqual(a.window_rms,b.window_rms)
                np.testing.assert_array_equal(noise_descriptor(a),noise_descriptor(b))

    def test_prediction_precedes_learning_and_gap_never_learns(self):
        spy=Spy();c=FreeRunningControls(spy,[0.],[1.],[-3.],[3.],lags=(0,1))
        for value in (0.,1.,2.):c.step([value])
        self.assertEqual([x[0] for x in spy.events],['predict','learn'])
        before=c.learned_frames
        a,_=c.step(None);b,_=c.step(None)
        self.assertEqual(c.learned_frames,before)
        np.testing.assert_allclose(a,[2.1]);np.testing.assert_allclose(b,[2.2])
        c.step([2.5]);c.step([2.6])
        self.assertEqual(c.learned_frames,before)
        c.step([2.7]);self.assertEqual(c.learned_frames,before+1)

    def test_observed_buffers_are_owned_and_frozen_mode_does_not_learn(self):
        spy=Spy();c=FreeRunningControls(spy,[0.],[1.],[-3.],[3.],adaptive=False,lags=(0,))
        x=np.array([1.]);c.step(x);x[0]=99.;c.step([2.]);c.step(None)
        self.assertFalse(any(kind=='learn' for kind,_ in spy.events))
        np.testing.assert_array_equal(spy.events[0][1],[1.])
        self.assertEqual(len(c.history),1)

    def test_decode_keeps_fixed_filters_stable_and_has_no_residual_input(self):
        source=ContinuousSource(np.zeros(128),np.eye(128)[:48]);space=ControlSpace(source)
        vector=np.zeros(space.size);vector[:32]=np.linspace(-.3,.3,32);vector[35]=-4
        control,harmonics,mix=space.decode(vector,0,240)
        self.assertLess(np.max(np.abs(np.roots(control.coefficients))),1.)
        self.assertEqual(harmonics.shape,(64,))
        self.assertEqual(mix,0.)
        self.assertFalse(hasattr(control,'residual'))
        with self.assertRaises(ValueError):space.decode(np.full(space.size,np.nan),0,240)

    def test_anchored_gap_does_not_feed_predictions_back_or_learn_missing_targets(self):
        class HorizonSpy(Spy):
            def __init__(self,horizon):super().__init__();self.horizon=horizon
            def predict(self,context):
                self.events.append(('predict',context.copy()))
                return context[:1]*.1+self.horizon*.2
        models={h:HorizonSpy(h) for h in (1,3)}
        c=AnchoredControls(models,[0.],[1.],[-10.],[10.],lags=(0,1))
        for x in (0.,1.,2.):c.step([x])
        before=c.learned_frames
        first,_=c.step(None)
        calls=sum(len(m.events) for m in models.values())
        second,_=c.step(None);third,_=c.step(None);fourth,_=c.step(None)
        self.assertEqual(sum(len(m.events) for m in models.values()),calls)
        self.assertEqual(c.learned_frames,before)
        np.testing.assert_allclose(np.r_[first,second,third,fourth],[2.4,2.6,2.8,2.8])
        self.assertFalse(c.pending)

    def test_anchored_feedback_is_delayed_and_pending_memory_is_bounded(self):
        models={h:Spy() for h in (1,3)}
        c=AnchoredControls(models,[0.],[1.],[-100.],[100.],lags=(0,1))
        c.step([0.]);c.step([1.])
        self.assertEqual(c.learned_frames,0)
        c.step([2.]);self.assertEqual(c.learned_frames,1)
        for i in range(3,100):c.step([float(i)])
        self.assertEqual(sum(len(tasks) for tasks in c.pending.values()),4)
        self.assertEqual(len(c.history),2)


if __name__=='__main__':unittest.main()
