import unittest
import numpy as np
from pcm_reference_decoder import PcmFrameDecoder, pcm_performance
from neural_performance_stream import NeuralWindowReader


class PcmReferenceTests(unittest.TestCase):
    def test_lossless_noise_edges_and_fractional_reads(self):
        source = np.random.default_rng(48).normal(0, .1, 1903)
        performance = pcm_performance(source, np.array([100., 220., 340.]))
        decoder = PcmFrameDecoder()
        np.testing.assert_allclose(decoder.render(performance.matrix), source, atol=1e-15)
        reader = NeuralWindowReader(performance.matrix, decoder, cache_frames=3)
        positions = np.r_[np.arange(len(source)), [-1., -.25, .25, 1000.75, 1902.75, 1903.]]
        i = np.floor(positions).astype(int); f = positions - i
        padded = np.r_[0., source, 0.]
        expected = np.interp(positions, np.arange(-1, len(source) + 1), padded, left=0, right=0)
        np.testing.assert_allclose(reader.sample(positions), expected, atol=1e-15)
        self.assertLessEqual(reader.maximum_cache_frames, 3)
        reader.clear(); self.assertFalse(reader.cache)


if __name__ == '__main__':
    unittest.main()
