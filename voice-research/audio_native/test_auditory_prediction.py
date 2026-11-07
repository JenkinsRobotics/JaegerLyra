import unittest
import numpy as np
from auditory_prediction import AuditoryForecast


class SpyModel:
    def __init__(self):
        self.observations = []
    def predict(self, context):
        return np.zeros(1)
    def observe(self, context, target, learn):
        self.observations.append((context.copy(),target.copy(),learn))


class DelayedPredictionTests(unittest.TestCase):
    def test_learning_waits_until_forecast_target_really_arrives(self):
        model = SpyModel()
        stream = AuditoryForecast(model, horizon=3, lags=(0,1))
        for index in range(4):
            stream.push([float(index)])
            self.assertEqual(model.observations, [])
        emitted, scored = stream.push([4.])
        self.assertEqual(scored['at'],1)
        self.assertEqual(scored['targetIndex'],4)
        np.testing.assert_array_equal(scored['prediction'],[1.])
        np.testing.assert_array_equal(scored['target'],[4.])
        np.testing.assert_array_equal(model.observations[0][0],[1.,0.])
        np.testing.assert_array_equal(model.observations[0][1],[3.])
        self.assertEqual(emitted['at'],4)

    def test_pending_storage_is_bounded_by_horizon(self):
        stream = AuditoryForecast(None, horizon=5,lags=(0,2,5))
        for index in range(100):
            stream.push([float(index)])
        self.assertEqual(len(stream.pending),5)
        self.assertEqual(len(stream.history),6)

    def test_incoming_buffer_mutation_does_not_rewrite_old_forecast(self):
        stream = AuditoryForecast(None, horizon=2,lags=(0,))
        source = np.array([1.])
        emitted, _ = stream.push(source)
        source[0] = 99.
        stream.push([2.])
        _, result = stream.push([3.])
        np.testing.assert_array_equal(result['prediction'],[1.])
        np.testing.assert_array_equal(result['base'],[1.])


if __name__=='__main__':
    unittest.main()
