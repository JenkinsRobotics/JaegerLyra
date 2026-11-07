import unittest
import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import resample_poly
from streaming_voicebox import lpc,VoiceboxConfig
from lpc_coordinates import to_reflection


class LpcSolverTests(unittest.TestCase):
    def test_covariance_solution_matches_independent_toeplitz_solver(self):
        rng=np.random.default_rng(218)
        cfg=VoiceboxConfig(order=32)
        time=np.arange(960)/24000
        # The low-frequency cases exercise |k1| > .98. Clipping that stage
        # inside the original recursion corrupts the remaining normal solve.
        signals=[rng.normal(0,.1,960),.3*np.sin(2*np.pi*100*time),
                 .2*np.sin(2*np.pi*80*time)+.1*np.sin(2*np.pi*160*time)+rng.normal(0,.0001,960)]
        for samples in signals:
            window=(samples-samples.mean())*np.hanning(len(samples))
            r=np.array([window[k:]@window[:len(window)-k] for k in range(33)])
            r[0]+=1e-6+r[0]*1e-4
            expected=np.r_[1.,solve_toeplitz(r[:32],-r[1:])]
            actual=lpc(samples,cfg)/cfg.bandwidth**np.arange(33)
            np.testing.assert_allclose(actual,expected,rtol=2e-8,atol=2e-9)
            residual=np.array([r[i]+actual[1:]@np.array([r[abs(i-j)] for j in range(1,33)]) for i in range(1,33)])
            self.assertLess(float(abs(residual).max()),float(r[0])*1e-10)

    def test_band_limited_low_voice_retains_stable_control_coordinates(self):
        rng=np.random.default_rng(843)
        t=np.arange(640)/16000
        for f0 in (70.,95.,130.,180.):
            audio=sum(.2/k*np.sin(2*np.pi*f0*k*t) for k in range(1,30))
            audio+=rng.normal(0,.0001,len(audio))
            audio=resample_poly(audio,3,2)
            a=lpc(audio,VoiceboxConfig(order=32))
            reflection=to_reflection(a)
            self.assertLess(float(abs(reflection).max()),1.)
            self.assertLess(float(abs(np.roots(a)).max()),1.)
            self.assertLess(float(abs(a).max()),20.)


if __name__=='__main__':unittest.main()
