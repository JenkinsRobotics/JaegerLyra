import copy
import unittest
import numpy as np
from lab import Config, LocalPredictor, continuous_roundtrip


class ContinuousAudioTests(unittest.TestCase):
    def test_framing_preserves_every_pcm16_sample_including_partial_frame(self):
        pcm = np.random.default_rng(1).integers(-32768, 32768, 1003, dtype=np.int16)
        result = continuous_roundtrip(pcm.astype(np.float64)/32768)
        restored = np.rint(result*32768).astype(np.int16)
        np.testing.assert_array_equal(restored, pcm)

    def test_freeze_keeps_weights_even_while_states_settle(self):
        net = LocalPredictor(Config(history=12, block=4, hidden=6))
        past = np.linspace(-.3, .2, 12)
        target = np.array([.3, .2, .1, -.1])
        for _ in range(5):
            net.observe(past, target)
        before = copy.deepcopy(net)
        result = net.observe(past, target, learn=False)
        self.assertLess(result['finalEnergy'], result['initialEnergy'])
        np.testing.assert_array_equal(net.W, before.W)
        np.testing.assert_array_equal(net.V, before.V)

    def test_future_target_cannot_change_committed_prediction(self):
        net = LocalPredictor(Config(history=12, block=4, hidden=6))
        x = np.linspace(-.2, .4, 12)
        for _ in range(20):
            net.observe(x, np.full(4, .2))
        alternative = copy.deepcopy(net)
        before = net.predict(x)
        other = alternative.predict(x)
        net.observe(x, np.full(4, .5))
        alternative.observe(x, np.full(4, -.5))
        np.testing.assert_array_equal(before, other)
        self.assertGreater(np.linalg.norm(net.predict(x)-alternative.predict(x)), 0)

    def test_energy_settling_remains_finite_and_decreases(self):
        rng = np.random.default_rng(9)
        net = LocalPredictor(Config(history=12, block=4, hidden=6))
        for _ in range(100):
            result = net.observe(rng.normal(0, .1, 12), rng.normal(0, .1, 4))
            self.assertTrue(np.isfinite(result['finalEnergy']))
            self.assertLessEqual(result['finalEnergy'], result['initialEnergy']+1e-12)


if __name__ == '__main__':
    unittest.main()
