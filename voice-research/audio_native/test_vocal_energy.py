import unittest
import numpy as np
from scipy.signal import lfilter
from predictive_controls import from_reflection,to_reflection
from vocal_energy import source_filter_power


class VocalEnergyTests(unittest.TestCase):
    def test_identity_filter_has_unit_gain_for_noise_and_harmonics(self):
        for mix in (0.,.3,1.):
            result=source_filter_power([1.,0.,0.],[0.,0.],180.,[1.,.5j,.2],mix)
            self.assertAlmostEqual(result['mixedPowerGain'],1.)

    def test_noise_gain_matches_filter_impulse_energy(self):
        for k in ([-.8,.2],[-.9,.4,-.1],[-.97,.6,.2,-.2]):
            a=from_reflection(k);impulse=np.zeros(48000);impulse[0]=1.
            response=lfilter([1.],a,impulse)
            predicted=source_filter_power(a,k,180.,[1.,.2],0.)['mixedPowerGain']
            self.assertAlmostEqual(float(response@response)/predicted,1.,places=9)

    def test_harmonic_gain_matches_long_filtered_waveform(self):
        a=np.array([1.,-1.5,.7]);k=to_reflection(a)
        harmonics=np.array([1.,.3j,-.2]);f0=200.;rate=24000
        times=np.arange(rate*3)/rate
        source=np.real(np.exp(2j*np.pi*np.arange(1,4)[:,None]*f0*times)*harmonics[:,None]).sum(axis=0)
        source/=np.sqrt(.5*np.sum(np.abs(harmonics)**2))
        output=lfilter([1.],a,source)
        predicted=source_filter_power(a,k,f0,harmonics,1.)['mixedPowerGain']
        self.assertAlmostEqual(float(np.mean(output[rate:]**2))/predicted,1.,places=9)

    def test_output_gain_compensates_resonance_in_stationary_noise(self):
        rng=np.random.default_rng(11);noise=rng.normal(size=24000*8)
        desired=.08
        for k in ([-.2,.1],[-.95,.5]):
            a=from_reflection(k)
            power=source_filter_power(a,k,180.,[1.,.3],0.)['mixedPowerGain']
            output=lfilter([1.],a,noise*desired/np.sqrt(power))
            self.assertLess(abs(np.sqrt(np.mean(output[24000:]**2))/desired-1.),.03)

    def test_invalid_or_silent_periodic_source_is_rejected(self):
        with self.assertRaises(ValueError):source_filter_power([1.,0.],[1.],160.,[1.],0.)
        with self.assertRaises(ValueError):source_filter_power([1.,0.],[0.],160.,[0.],1.)


if __name__=='__main__':unittest.main()
