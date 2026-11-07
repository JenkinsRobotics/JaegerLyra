import unittest
from types import SimpleNamespace
import numpy as np
from scipy.signal import freqz,lfilter,welch
from noise_shape import noise_descriptor,shaping_filter,ShapedNoise,BANDS,TAPS


class NoiseShapeTests(unittest.TestCase):
    def test_flat_color_is_unit_impulse_and_preserves_noise(self):
        np.testing.assert_array_equal(shaping_filter(np.zeros(BANDS)),np.r_[1.,np.zeros(TAPS-1)])
        white=np.random.default_rng(17).normal(size=2400)
        shaper=ShapedNoise()
        actual=np.concatenate([shaper.process(x,np.zeros(BANDS)) for x in np.array_split(white,10)])
        np.testing.assert_allclose(actual,white,atol=2e-15,rtol=2e-15)

    def test_colored_noise_has_unit_variance_and_expected_band_ratio(self):
        rng=np.random.default_rng(58);bands=np.linspace(-1.,1.,BANDS)
        taps=shaping_filter(bands);white=rng.normal(size=240000)
        actual=lfilter(taps,[1.],white)[TAPS:]
        self.assertLess(abs(float(np.std(actual))-1),.03)
        frequency,measured=welch(actual,fs=24000,nperseg=1024)
        _,response=freqz(taps,worN=2*np.pi*frequency/24000)
        low=(frequency>100)&(frequency<1000);high=(frequency>6000)&(frequency<11000)
        ratio=measured[high].mean()/measured[low].mean()
        expected=(abs(response[high])**2).mean()/(abs(response[low])**2).mean()
        self.assertGreater(ratio,3.)
        self.assertLess(abs(10*np.log10(ratio/expected)),.5)

    def test_fixed_color_chunking_does_not_change_noise(self):
        bands=np.linspace(1.,-1.,BANDS);white=np.random.default_rng(2).normal(size=2400)
        whole,chunked=ShapedNoise(),ShapedNoise()
        whole.previous=shaping_filter(bands);chunked.previous=whole.previous.copy()
        expected=whole.process(white,bands)
        actual=np.concatenate([chunked.process(part,bands) for part in np.array_split(white,19)])
        np.testing.assert_allclose(actual,expected,atol=2e-15,rtol=2e-14)
        np.testing.assert_array_equal(whole.history,chunked.history)
        self.assertEqual(len(chunked.history),TAPS-1)

    def test_ramp_matches_direct_time_varying_fir_convolution(self):
        rng=np.random.default_rng(74);shaper=ShapedNoise()
        shaper.process(rng.normal(size=240),np.linspace(-1.,1.,BANDS))
        old=shaper.previous.copy();history=shaper.history.copy()
        bands=np.linspace(1.2,-1.2,BANDS);target=shaping_filter(bands)
        white=rng.normal(size=113);context=np.r_[history,white]
        expected=[]
        for n in range(len(white)):
            blend=(n+1)/len(white);taps=(1-blend)*old+blend*target
            taps/=np.linalg.norm(taps)
            expected.append(taps@context[n:n+TAPS][::-1])
        actual=shaper.process(white,bands)
        np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=1e-12)
        saved=shaper.history.copy();white[:]=999
        np.testing.assert_array_equal(shaper.history,saved)

    def test_descriptor_has_no_waveform_state_and_handles_silence(self):
        frame=SimpleNamespace(excitation_window=np.zeros(960),f0=0.,periodicity=0.)
        np.testing.assert_array_equal(noise_descriptor(frame),np.zeros(BANDS))
        frame.excitation_window=np.random.default_rng(52).normal(size=960)
        value=noise_descriptor(frame);saved=value.copy()
        self.assertEqual(value.shape,(BANDS,))
        frame.excitation_window[:]=np.nan
        np.testing.assert_array_equal(value,saved)
        with self.assertRaises(ValueError):noise_descriptor(frame)

    def test_invalid_controls_do_not_mutate_shaper(self):
        shaper=ShapedNoise();before=shaper.history.copy()
        for bands in (np.zeros(BANDS+1),np.full(BANDS,np.nan)):
            with self.assertRaises(ValueError):shaper.process(np.ones(240),bands)
            np.testing.assert_array_equal(shaper.history,before)


if __name__=='__main__':unittest.main()
