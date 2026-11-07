import tempfile
import unittest
from pathlib import Path
import numpy as np
from compact_vocal_instrument import VocalFrame,RATE,BLOCK
from pulse_basis_instrument import (PulseModel,PulseBasisStream,pulse_window,OFFSETS,RADIUS,SIZE,MODES,
    fit_pulses,reconstruction_metrics,save_models,load_models)


def analytic_model():
    period=120.;window=pulse_window(period)
    waveform=np.cos(2*np.pi*200*OFFSETS/RATE)*window
    waveform/=np.sqrt(np.sum(waveform**2)/np.sum(window**2))
    rng=np.random.default_rng(31)
    q,_=np.linalg.qr((rng.normal(size=(MODES,SIZE))*window).T)
    a=np.eye(MODES)*.6;b=np.eye(MODES)*.8
    return PulseModel(waveform,q.T,np.full(MODES,.02),a,b,period,.04)


class PulseBasisTests(unittest.TestCase):
    def test_analytic_tone_with_explicit_delay_and_tail(self):
        stream=PulseBasisStream([analytic_model()],amount=0.)
        blocks=[stream.push(VocalFrame(200.,(1.,))) for _ in range(200)]
        audio=np.r_[np.concatenate(blocks),stream.finish()]
        self.assertEqual(len(audio),RATE+RADIUS)
        aligned=audio[RADIUS:]
        expected=.04*np.sqrt(2)*np.cos(2*np.pi*200*np.arange(RATE)/RATE)
        np.testing.assert_allclose(aligned[1200:-1200],expected[1200:-1200],atol=1e-9)
        self.assertEqual(len(stream.finish()),0)

    def test_fit_and_held_representation_known_subspace(self):
        source=analytic_model();rng=np.random.default_rng(411)
        tracks=[source.mean+rng.normal(size=(40,MODES))@(source.basis*.1) for _ in range(3)]
        model,report=fit_pulses(tracks,source.reference_period,source.rms)
        held=[source.mean+rng.normal(size=(20,MODES))@(source.basis*.1)]
        metrics=reconstruction_metrics(model,held)
        self.assertGreater(metrics['projectedEightModeSnrDb'],200.)
        self.assertGreater(metrics['projectedEightModeSnrDb'],metrics['meanTemplateSnrDb']+50)
        self.assertEqual(report['trainingPulses'],120)

    def test_state_seed_roundtrip_and_gain(self):
        model=analytic_model()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pulse.npz';save_models(path,[model],{'templates':True})
            restored,meta=load_models(path);self.assertTrue(meta['templates'])
            a=PulseBasisStream([model],seed=13);b=PulseBasisStream(restored,seed=13)
            quieter=PulseBasisStream(restored,seed=13)
            for _ in range(20):
                audio=a.push(VocalFrame(200.,(1.,)))
                np.testing.assert_array_equal(audio,b.push(VocalFrame(200.,(1.,))))
                np.testing.assert_allclose(audio*.5,quieter.push(VocalFrame(200.,(1.,),gain=.5)),atol=1e-14)
            self.assertFalse(a.bases.flags.writeable)

    def test_cancel_and_invalid_controls_are_atomic(self):
        a=PulseBasisStream([analytic_model()]);b=PulseBasisStream([analytic_model()])
        frame=VocalFrame(200.,(1.,));a.push(frame);b.push(frame)
        for bad in (VocalFrame(float('nan'),(1.,)),VocalFrame(200.,(1.,),breath=.2)):
            with self.assertRaises(ValueError):a.push(bad)
        np.testing.assert_array_equal(a.push(frame),b.push(frame))
        a.cancel();self.assertFalse(np.any(a.pending));self.assertFalse(np.any(a.weights));self.assertFalse(np.any(a.state))
        self.assertEqual(len(a.finish()),0)
        with self.assertRaises(RuntimeError):a.push(frame)

    def test_immediate_pitch_changes_keep_buffers_bounded(self):
        stream=PulseBasisStream([analytic_model()])
        sizes={k:v.shape for k,v in vars(stream).items() if isinstance(v,np.ndarray)}
        for i in range(500):
            hz=(60.,800.,200.,300.)[(i//50)%4]
            output=stream.push(VocalFrame(hz,(1.,)))
            self.assertEqual(len(output),BLOCK);self.assertTrue(np.isfinite(output).all())
        self.assertEqual(sizes,{k:v.shape for k,v in vars(stream).items() if isinstance(v,np.ndarray)})
        self.assertEqual(len(stream.finish()),RADIUS)


if __name__=='__main__':unittest.main()
