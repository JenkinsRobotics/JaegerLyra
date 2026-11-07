import unittest
from types import SimpleNamespace
import numpy as np
from neural_waveform import ControlMatrix, RATE, SIZE, WINDOW, waveform_frames
from neural_performance import EncodedPerformance
from neural_vowel_instrument import NeuralVowelInstrument, VowelScore, compile_vowel


class ExactFrameDecoder:
    # Isolates the controller from neural approximation in analytical tests.
    config = SimpleNamespace(history=SIZE+1)
    def frame(self, control, gain):
        return control*gain
    def render(self, *args):
        raise AssertionError('Instrument must decode windows on demand')


class ExactEncoder:
    def encode(self, audio):
        frames, gains = waveform_frames(audio)
        return ControlMatrix(frames, gains, len(audio))


def fixture():
    t = np.arange(round(.180*RATE))/RATE
    audio = .15*np.cos(2*np.pi*200*t)+.05*np.cos(2*np.pi*400*t)
    frames, gains = waveform_frames(audio)
    return EncodedPerformance(ControlMatrix(frames, gains, len(audio)),
        np.arange(240, len(audio)-240, RATE/200)), ExactFrameDecoder()


def collect(stream, sizes=(240,)):
    parts=[]; i=0; expected=0
    while (packet := stream.read(sizes[i%len(sizes)])) is not None:
        if packet.start != expected:
            raise AssertionError('Discontinuous packet positions')
        expected += len(packet.samples); parts.append(packet.samples); i += 1
    return np.concatenate(parts)


class VowelInstrumentTests(unittest.TestCase):
    def test_new_note_and_duration_from_short_patch(self):
        patch, decoder = fixture()
        for hz in (150., 200., 250.):
            stream = NeuralVowelInstrument(patch, decoder, VowelScore(seconds=2., hz=(hz,)))
            self.assertEqual(stream.reader.decoded_frames, 0)
            audio = collect(stream)
            self.assertEqual(len(audio), RATE*2)
            # One whole second gives independent 1 Hz FFT resolution.
            middle = audio[RATE//2:RATE//2+RATE]
            spectrum = abs(np.fft.rfft(middle))
            self.assertLessEqual(abs(np.argmax(spectrum)-hz), 1.)
            self.assertEqual(audio[0], 0.); self.assertEqual(audio[-1], 0.)
            self.assertLess(abs(audio).max(), .3)
            self.assertLessEqual(stream.reader.maximum_cache_frames, 16)
            self.assertFalse(stream.reader.cache)

    def test_chunk_invariance_and_amplitude_independence(self):
        patch, decoder = fixture()
        kwargs = dict(seconds=.701, times=(0., .3, .6), hz=(180., 260., 200.), vibrato_cents=35.)
        a = collect(NeuralVowelInstrument(patch, decoder, VowelScore(**kwargs)))
        b = collect(NeuralVowelInstrument(patch, decoder, VowelScore(**kwargs)), (1, 17, 239, 240))
        c = collect(NeuralVowelInstrument(patch, decoder, VowelScore(**kwargs, gain=.5)))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(c, a*.5)
        self.assertEqual(len(a), round(.701*RATE))

    def test_cancellation_and_source_snapshot(self):
        patch, decoder = fixture()
        score = VowelScore(seconds=.301)
        expected = collect(NeuralVowelInstrument(patch, decoder, score))
        stream = NeuralVowelInstrument(patch, decoder, score)
        patch.matrix.values[:] = np.nan; patch.matrix.gains[:] = np.nan; patch.pulses[:] = np.nan
        np.testing.assert_array_equal(collect(stream), expected)
        patch, decoder = fixture(); stream = NeuralVowelInstrument(patch, decoder, score)
        stream.read(); before = stream.reader.decoded_frames; stream.cancel()
        self.assertIsNone(stream.read()); self.assertFalse(stream.reader.cache)
        self.assertEqual(stream.reader.decoded_frames, before)

    def test_compiler_bounds_reference_and_rejects_silence(self):
        time = np.arange(RATE)/RATE
        audio = .15*np.cos(2*np.pi*200*time)
        patch, meta = compile_vowel(audio, ExactEncoder())
        self.assertLessEqual(patch.matrix.sample_count, .200*RATE)
        self.assertGreaterEqual(meta['pulseCount'], 6)
        self.assertFalse(patch.matrix.values.flags.writeable)
        with self.assertRaises(ValueError):
            compile_vowel(np.zeros(RATE), ExactEncoder())

    def test_invalid_score_and_read_are_rejected_before_output(self):
        patch, decoder = fixture()
        for score in (VowelScore(seconds=0.), VowelScore(hz=(0.,)), VowelScore(hz=(np.nan,)),
                VowelScore(times=(0., 0.), hz=(100., 200.)), VowelScore(times=(.1,))):
            with self.assertRaises(ValueError): NeuralVowelInstrument(patch, decoder, score)
        stream = NeuralVowelInstrument(patch, decoder)
        for n in (0, 241, True, 1.5):
            with self.assertRaises(ValueError): stream.read(n)
        self.assertEqual(stream.position, 0); self.assertEqual(stream.reader.decoded_frames, 0)


if __name__ == '__main__':
    unittest.main()
