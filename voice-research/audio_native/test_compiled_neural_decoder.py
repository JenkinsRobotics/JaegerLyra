import tempfile
import unittest
from pathlib import Path
import numpy as np
from lab import Config
from neural_waveform import NeuralDecoder, ControlEncoder, SIZE, save_model, load_model
from compiled_neural_decoder import CompiledAxisDecoder, compile_decoder


class CompiledDecoderTests(unittest.TestCase):
    def fixture(self, input_gain=.7):
        rng=np.random.default_rng(98)
        weights=np.zeros((13,6))
        for row in range(12):weights[row,row%5]=rng.normal()
        weights[:,-1]=rng.normal(size=13)
        decoder=NeuralDecoder(Config(history=6,block=SIZE,hidden=13,input_gain=input_gain),
            weights,rng.normal(0,.02,(SIZE,13)))
        return decoder

    def test_exact_sparse_execution_including_constant_units_and_gains(self):
        rng=np.random.default_rng(80)
        for scale in (.7,1.,1.5):
            dense=self.fixture(scale);compiled=compile_decoder(dense)
            self.assertIsInstance(compiled,CompiledAxisDecoder)
            np.testing.assert_array_equal(compiled.W,dense.W)
            np.testing.assert_array_equal(compiled.V,dense.V)
            for gain in (0.,.03,1.):
                for controls in rng.uniform(-5,5,(25,5)):
                    np.testing.assert_allclose(compiled.frame(controls,gain),dense.frame(controls,gain),atol=2e-15,rtol=2e-13)

    def test_dense_fallback_does_not_drop_even_a_tiny_weight(self):
        dense=self.fixture();weights=dense.W.copy();weights[0,1]=1e-100
        mixed=NeuralDecoder(dense.config,weights,dense.V)
        self.assertIs(compile_decoder(mixed),mixed)
        with self.assertRaises(ValueError):CompiledAxisDecoder(mixed.config,mixed.W,mixed.V)

    def test_compilation_does_not_modify_or_retain_source_object(self):
        dense=self.fixture();controls=np.arange(5)/3
        expected=dense.frame(controls);compiled=compile_decoder(dense)
        self.assertFalse(np.shares_memory(compiled.W,dense.W))
        self.assertFalse(np.shares_memory(compiled.V,dense.V))
        del dense
        np.testing.assert_allclose(compiled.frame(controls),expected,atol=2e-15,rtol=2e-13)
        for value in (compiled.W,compiled.V,compiled._columns,compiled._scales,compiled._biases):
            self.assertFalse(value.flags.writeable)
        for controls,gain in ((np.ones(4),1),(np.full(5,np.nan),1),(np.ones(5),-1)):
            with self.assertRaises(ValueError):compiled.frame(controls,gain)

    def test_serialization_remains_dense_weights_without_audio_or_compiler_state(self):
        dense=self.fixture();compiled=compile_decoder(dense)
        encoder=ControlEncoder(np.zeros(SIZE),np.eye(SIZE)[:5],np.ones(5))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.npz';save_model(path,encoder,compiled,{'scope':'test'})
            with np.load(path,allow_pickle=False) as saved:
                self.assertEqual(set(saved.files),{'encoder_mean','encoder_basis','encoder_scale','W','V','config','metadata'})
            _,loaded,_=load_model(path)
        np.testing.assert_allclose(loaded.frame(np.arange(5)),compiled.frame(np.arange(5)),atol=2e-15,rtol=2e-13)


if __name__=='__main__':unittest.main()
