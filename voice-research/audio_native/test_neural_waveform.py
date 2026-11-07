import tempfile,unittest
from pathlib import Path
import numpy as np
from lab import cosine_basis
from neural_waveform import (ControlEncoder,ControlMatrix,NeuralDecoder,initialize_decoder,
    waveform_frames,overlap_add,save_model,load_model,initialize_axis_decoder,SIZE,HOP)


class NeuralWaveformTests(unittest.TestCase):
    def test_overlap_windows_preserve_boundaries_and_partial_last_frame(self):
        rng=np.random.default_rng(72)
        for count in (0,1,239,240,241,1281):
            audio=rng.normal(0,.05,count)
            frames,gains=waveform_frames(audio)
            actual=overlap_add(frames*gains[:,None],count)
            np.testing.assert_allclose(actual,audio,atol=1e-14,rtol=1e-13)

    def test_full_continuous_basis_is_a_reference_ceiling_not_a_neural_claim(self):
        encoder=ControlEncoder.fit(cosine_basis(SIZE),dimensions=SIZE)
        audio=np.random.default_rng(21).normal(0,.05,1281)
        controls=encoder.encode(audio);saved=audio.copy();audio[:]=np.nan
        np.testing.assert_allclose(encoder.reconstruct_linear(controls),saved,atol=1e-12,rtol=1e-11)
        self.assertFalse(hasattr(controls,'pcm'))

    def test_network_learns_a_nonlinear_waveform_map_without_target_at_inference(self):
        rng=np.random.default_rng(88)
        def target(z):
            t=np.arange(SIZE)
            return .05*((z[:,0]+.3*z[:,1]*z[:,2])[:,None]*np.sin(.1*t)
                        +np.tanh(z[:,2])[:,None]*np.cos(.2*t))
        training=rng.normal(0,.5,(700,3));targets=target(training)
        learner,_=initialize_decoder(training,targets,hidden=64,ridge=1e-5)
        decoder=NeuralDecoder.from_learner(learner)
        heldout=rng.normal(0,.5,(60,3));expected=target(heldout)
        actual=np.array([decoder.frame(z) for z in heldout])
        self.assertLess(float(np.mean((actual-expected)**2)),float(np.mean(expected**2))*.1)
        training[:]=np.nan;targets[:]=np.nan
        np.testing.assert_array_equal(np.array([decoder.frame(z) for z in heldout]),actual)
        self.assertEqual(set(decoder.__dict__),{'config','W','V'})
        with self.assertRaises(ValueError):decoder.V[0,0]=999.

    def test_saved_model_reproduces_control_only_synthesis(self):
        rng=np.random.default_rng(7)
        encoder=ControlEncoder(np.zeros(SIZE),np.eye(SIZE)[:3],np.ones(3))
        learner,_=initialize_decoder(rng.normal(size=(40,3)),rng.normal(0,.1,(40,SIZE)),hidden=12)
        decoder=NeuralDecoder.from_learner(learner)
        controls=ControlMatrix(rng.normal(size=(5,3)),np.ones(5)*.01,4*HOP)
        expected=decoder.render(controls)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.npz';save_model(path,encoder,decoder,{'purpose':'test'})
            restored_encoder,restored,metadata=load_model(path)
        np.testing.assert_array_equal(restored.render(controls),expected)
        np.testing.assert_array_equal(restored_encoder.basis,encoder.basis)
        self.assertEqual(metadata,{'purpose':'test'})

    def test_invalid_controls_and_zero_gain(self):
        rng=np.random.default_rng(1)
        learner,_=initialize_decoder(rng.normal(size=(20,3)),rng.normal(size=(20,SIZE)),hidden=8)
        decoder=NeuralDecoder.from_learner(learner)
        np.testing.assert_array_equal(decoder.frame(np.ones(3),gain=0),np.zeros(SIZE))
        for controls,gain in ((np.ones(4),1),(np.full(3,np.nan),1),(np.ones(3),-1)):
            with self.assertRaises(ValueError):decoder.frame(controls,gain)
        with self.assertRaises(ValueError):overlap_add(np.zeros((1,SIZE)),1000)

    def test_structured_neurons_generalize_coordinatewise_nonlinear_controls(self):
        rng=np.random.default_rng(81)
        basis=rng.normal(0,.02,(3,SIZE))
        def target(z):return np.sinh(z)@basis
        training=rng.uniform(-3,3,(1000,3));expected=target(training)
        learner,_=initialize_axis_decoder(training,expected,centers=np.linspace(-3,3,9),slope=1.,ridge=1e-8)
        decoder=NeuralDecoder.from_learner(learner)
        heldout=rng.uniform(-2.9,2.9,(100,3));truth=target(heldout)
        prediction=np.array([decoder.frame(z) for z in heldout])
        self.assertLess(np.mean((prediction-truth)**2)/np.mean(truth**2),1e-4)
        training[:]=np.nan;expected[:]=np.nan
        np.testing.assert_array_equal(np.array([decoder.frame(z) for z in heldout]),prediction)
        self.assertEqual(set(decoder.__dict__),{'config','W','V'})

    def test_example_weights_match_explicit_repetition(self):
        rng=np.random.default_rng(19)
        controls=rng.normal(0,.5,(30,2));targets=rng.normal(0,.1,(30,SIZE))
        count=rng.integers(1,4,30)
        a,_=initialize_axis_decoder(controls,targets,centers=(-1,0,1),sample_weight=count)
        b,_=initialize_axis_decoder(np.repeat(controls,count,axis=0),np.repeat(targets,count,axis=0),centers=(-1,0,1))
        np.testing.assert_allclose(a.W,b.W,atol=0,rtol=0)
        np.testing.assert_allclose(a.V,b.V,atol=1e-7,rtol=1e-7)
        for invalid in (np.zeros(30),np.full(30,np.nan),np.ones(29)):
            with self.assertRaises(ValueError):initialize_axis_decoder(controls,targets,sample_weight=invalid)


if __name__=='__main__':unittest.main()
