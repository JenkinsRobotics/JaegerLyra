import unittest
import numpy as np
from streaming_voicebox import StreamingAnalysis, StreamingSynthesis, process, resonance_coefficients


class StreamingVoiceboxTests(unittest.TestCase):
    def test_pcm_roundtrip_including_silence_and_partial_final_frame(self):
        rng = np.random.default_rng(31)
        pcm = np.r_[np.zeros(290), rng.integers(-20000, 20000, 2109), np.zeros(311)].astype(np.int16)
        recovered, frames = process(pcm/32768.)
        np.testing.assert_array_equal(np.rint(recovered*32768).astype(np.int16), pcm)
        self.assertEqual(sum(len(f.residual) for f in frames), len(pcm))
        self.assertTrue(all(f.available_at-f.start <= 240 for f in frames))

    def test_transport_packet_boundaries_do_not_change_audio_or_features(self):
        samples = np.random.default_rng(8).normal(0, .05, 2401)
        first, a = process(samples, (1, 17, 100, 531))
        second, b = process(samples, (len(samples),))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal([f.coefficients for f in a], [f.coefficients for f in b])
        # freqz/log transforms can vary by a few ulps with array alignment.
        np.testing.assert_allclose([f.features() for f in a], [f.features() for f in b], atol=1e-12, rtol=0)

    def test_future_cannot_change_emitted_frame(self):
        samples = np.random.default_rng(6).normal(0, .05, 3000)
        _, first = process(samples)
        samples[1440:] *= -3
        _, changed = process(samples)
        for before, after in zip(first[:6], changed[:6]):
            np.testing.assert_array_equal(before.residual, after.residual)
            np.testing.assert_array_equal(before.coefficients, after.coefficients)
            np.testing.assert_allclose(before.features(), after.features(), atol=1e-12, rtol=0)

    def test_stream_memory_and_missing_frame_are_explicit(self):
        analysis = StreamingAnalysis()
        frames = analysis.push(np.zeros(24019))
        self.assertEqual(len(analysis.history), 960)
        self.assertEqual(len(analysis.pending), 19)
        self.assertLessEqual(analysis.pending.nbytes, 239*8)
        with self.assertRaises(ValueError):
            StreamingSynthesis().push(frames[1])
        analysis.finish()
        self.assertEqual(analysis.finish(), [])
        with self.assertRaises(ValueError):
            analysis.push([0])

    def test_resonance_moves_known_pole_angles_without_changing_radii(self):
        poles = np.array([.9*np.exp(.3j), .9*np.exp(-.3j), .8*np.exp(.8j), .8*np.exp(-.8j)])
        shifted = np.roots(resonance_coefficients(np.real(np.poly(poles)), 1.1))
        np.testing.assert_allclose(np.sort(np.abs(shifted)), np.sort(np.abs(poles)), atol=1e-10)
        np.testing.assert_allclose(np.sort(np.angle(shifted)), np.sort(np.angle(poles)*1.1), atol=1e-10)

    def test_invalid_signal_rejected_before_processing(self):
        for value in [[float('nan')], [[0.]], [float('inf')]]:
            with self.assertRaises(ValueError):
                StreamingAnalysis().push(value)


if __name__ == '__main__':
    unittest.main()
