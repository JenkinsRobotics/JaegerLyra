import tempfile
import unittest
from pathlib import Path
import numpy as np
from compact_vocal_instrument import CompactVocalStream,VocalFrame
from generated_vocal_texture import (fit_model,forecast_metrics,save_models,load_models,GeneratedTextureStream,FEATURES)
import test_compact_vocal_instrument as compact_tests


def tracks(seed,count=4,length=800):
    rng=np.random.default_rng(seed);result=[]
    rho=np.array([.6,.7,.8]);scale=np.array([.03,.02,.01])
    for _ in range(count):
        state=rng.normal(size=3);values=[]
        for _ in range(length):
            state=rho*state+np.sqrt(1-rho**2)*rng.normal(size=3)
            values.append(np.r_[state*scale,np.zeros(FEATURES-3)])
        result.append(np.array(values))
    return result


class GeneratedTextureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model,cls.fit=fit_model(tracks(123))

    def shape(self):return compact_tests.CompactInstrumentTests().shape()

    def test_known_process_prediction_and_stability(self):
        result=forecast_metrics(self.model,tracks(991,count=2))
        self.assertLess(result[0]['rmse']['learned'],result[0]['rmse']['mean'])
        self.assertLess(result[0]['rmse']['learned'],result[0]['rmse']['persistence'])
        self.assertLessEqual(np.linalg.norm(self.model.transition,2),.980000001)
        # Stationary covariance is checked independently after many updates.
        rng=np.random.default_rng(18);state=rng.normal(size=3);states=[]
        for _ in range(15000):
            state=self.model.transition@state+self.model.innovation@rng.normal(size=3)
            states.append(state.copy())
        np.testing.assert_allclose(np.cov(np.array(states).T),np.eye(3),atol=.09)

    def test_zero_amount_is_exact_reference_and_seeds_reproduce(self):
        shape=self.shape();base=CompactVocalStream([shape],seed=4)
        zero=GeneratedTextureStream([shape],[self.model],seed=4,amount=0.)
        a=GeneratedTextureStream([shape],[self.model],seed=4)
        b=GeneratedTextureStream([shape],[self.model],seed=4)
        other=GeneratedTextureStream([shape],[self.model],seed=5)
        differences=[]
        for hz in (200.,240.,220.,180.):
            frame=VocalFrame(hz,(1.,),breath=.1)
            np.testing.assert_array_equal(base.push(frame),zero.push(frame))
            output=a.push(frame);np.testing.assert_array_equal(output,b.push(frame))
            differences.append(np.max(abs(output-other.push(frame))))
        self.assertGreater(max(differences),1e-5)

    def test_persistence_has_parameters_without_source_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'texture.npz';save_models(file,[self.model],{'test':True})
            models,metadata=load_models(file);self.assertTrue(metadata['test'])
            with np.load(file,allow_pickle=False) as data:
                self.assertEqual(set(data.files),{'mean','basis','scale','transition','innovation','metadata'})
            a=GeneratedTextureStream([self.shape()],models)
            b=GeneratedTextureStream([self.shape()],[self.model])
            np.testing.assert_array_equal(a.push(VocalFrame(200.,(1.,))),b.push(VocalFrame(200.,(1.,))))
            self.assertFalse(a.basis.flags.writeable)

    def test_invalid_input_atomic_and_cancel_clears_latent(self):
        a=GeneratedTextureStream([self.shape()],[self.model]);b=GeneratedTextureStream([self.shape()],[self.model])
        frame=VocalFrame(200.,(1.,))
        a.push(frame);b.push(frame)
        with self.assertRaises(ValueError):a.push(VocalFrame(200.,(-1.,)))
        np.testing.assert_array_equal(a.push(frame),b.push(frame))
        a.cancel();self.assertFalse(np.any(a.state));self.assertFalse(np.any(a.noise_history))
        with self.assertRaises(RuntimeError):a.push(frame)
        with self.assertRaises(ValueError):GeneratedTextureStream([self.shape()],[self.model],amount=2.)

    def test_long_run_has_fixed_state_and_no_future_inputs(self):
        stream=GeneratedTextureStream([self.shape()],[self.model],seed=77)
        stream.push(VocalFrame(200.,(1.,)))
        shapes={k:v.shape for k,v in vars(stream).items() if isinstance(v,np.ndarray)}
        for i in range(999):
            audio=stream.push(VocalFrame(200.+20*np.sin(i/100),(1.,)))
            self.assertEqual(len(audio),120);self.assertTrue(np.isfinite(audio).all())
        self.assertEqual(stream.generated_frames,1000)
        self.assertEqual(shapes,{k:v.shape for k,v in vars(stream).items() if isinstance(v,np.ndarray)})
        self.assertEqual(stream.state.shape,(1,3))


if __name__=='__main__':unittest.main()
