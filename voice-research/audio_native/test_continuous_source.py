import unittest
import numpy as np
from continuous_source import ContinuousSource


class ContinuousSourceTests(unittest.TestCase):
    def test_source_decoder_uses_bounded_continuous_coordinates(self):
        mean=np.arange(64,dtype=float)/64
        model=ContinuousSource(mean,np.eye(64)[:8])
        value=model.decode(np.arange(8,dtype=float))
        expected=mean.copy();expected[:8]+=np.arange(8)
        np.testing.assert_allclose(value,expected[:32]+1j*expected[32:])
        self.assertEqual(set(vars(model)),{'mean','basis'})
        with self.assertRaises(ValueError):model.decode(np.zeros(9))
        with self.assertRaises(ValueError):model.decode(np.full(8,np.nan))

    def test_constant_source_has_no_per_frame_source_information(self):
        mean=np.arange(64,dtype=float)
        model=ContinuousSource(mean,np.empty((0,64)))
        np.testing.assert_array_equal(model.decode(np.empty(0)),mean[:32]+1j*mean[32:])

    def test_richer_source_keeps_all_harmonics_without_changing_decoder_contract(self):
        mean=np.arange(128,dtype=float)
        basis=np.eye(128)[:48]
        model=ContinuousSource(mean,basis)
        actual=model.decode(np.ones(48))
        expected=mean.copy();expected[:48]+=1
        np.testing.assert_array_equal(actual,expected[:64]+1j*expected[64:])
        self.assertEqual(len(actual),64)
        with self.assertRaises(ValueError):ContinuousSource(np.zeros(127),np.zeros((1,127)))


if __name__=='__main__':unittest.main()
