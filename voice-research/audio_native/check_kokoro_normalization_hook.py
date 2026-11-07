"""Small real-MLX checks for hook cleanup and native-reference normalization."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_audio.tts.models.kokoro.istftnet import InstanceNorm1d,_InstanceNorm
from kokoro_normalization_reference import NormalizationReference


class Fixture(nn.Module):
    def __init__(self):
        super().__init__();self.norm=InstanceNorm1d(3,affine=False)
    def __call__(self,x):return self.norm(x)


class HookTests(unittest.TestCase):
    def test_native_replay_and_extended_reference_statistics(self):
        model=Fixture();ref=NormalizationReference(model)
        x=mx.array(np.arange(30,dtype=np.float32).reshape(1,3,10))
        with ref.apply('capture'):original=model(x);mx.eval(original)
        with ref.apply('reference'):replayed=model(x);mx.eval(replayed)
        np.testing.assert_array_equal(np.array(original),np.array(replayed))
        extended=mx.repeat(x[:,:,:1],20,axis=2)
        with ref.apply('reference'):changed=model(extended);mx.eval(changed)
        np.testing.assert_allclose(np.array(changed),np.repeat(np.array(original[:,:,:1]),20,axis=2),atol=1e-6)

    def test_exception_restores_original_class_method(self):
        model=Fixture();ref=NormalizationReference(model);before=_InstanceNorm._apply_instance_norm
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with ref.apply('capture'):raise RuntimeError('intentional')
        self.assertIs(_InstanceNorm._apply_instance_norm,before)
        with self.assertRaises(ValueError):
            with ref.apply('reference'):pass
        self.assertIs(_InstanceNorm._apply_instance_norm,before)

    def test_saved_reference_reopens_without_source_inference(self):
        model=Fixture();ref=NormalizationReference(model)
        x=mx.array(np.arange(30,dtype=np.float32).reshape(1,3,10))
        with ref.apply('capture'):expected=model(x);mx.eval(expected)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'reference.npz';ref.save(path)
            loaded=NormalizationReference.load(model,path)
            with loaded.apply('reference'):actual=model(x);mx.eval(actual)
        np.testing.assert_array_equal(np.array(actual),np.array(expected))


if __name__=='__main__':unittest.main(verbosity=2)
