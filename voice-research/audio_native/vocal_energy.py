"""Couple source energy to the predicted resonance using stationary DSP power.

The white-noise all-pole power gain is 1/prod(1-k**2). Harmonic gain is the
source-power-weighted squared frequency response. Their voiced/noise mixture
predicts long-run output RMS. This does not control changing-filter transients.
"""
import numpy as np
from scipy.signal import freqz


def source_filter_power(coefficients, reflection, f0, harmonics, voicing, rate=24000):
    coefficients=np.asarray(coefficients,dtype=np.float64)
    reflection=np.asarray(reflection,dtype=np.float64)
    harmonics=np.asarray(harmonics,dtype=np.complex128)
    if (reflection.ndim!=1 or coefficients.shape!=(len(reflection)+1,)
            or harmonics.ndim!=1 or not len(harmonics)
            or not all(np.isfinite(x).all() for x in (coefficients,reflection,harmonics))
            or coefficients[0]!=1 or np.any(np.abs(reflection)>=1)
            or not np.isfinite(f0) or f0<=0 or not np.isfinite(voicing) or not 0<=voicing<=1):
        raise ValueError('Invalid source/filter power controls')
    noise_power=float(np.exp(-np.log1p(-reflection*reflection).sum()))
    frequencies=f0*np.arange(1,len(harmonics)+1)
    taper=np.clip((rate*.49-frequencies)/(rate*.04),0.,1.)
    powers=np.abs(harmonics*taper)**2
    active=powers>0
    if active.any():
        _,response=freqz([1.],coefficients,worN=2*np.pi*frequencies[active]/rate)
        harmonic_power=float(np.sum(powers[active]*np.abs(response)**2)/powers.sum())
    else:
        harmonic_power=0.
    mixed=voicing*harmonic_power+(1-voicing)*noise_power
    if not np.isfinite(mixed) or mixed<=0:
        raise ValueError('Source/filter output power is zero or nonfinite')
    return {'noisePowerGain':noise_power,'harmonicPowerGain':harmonic_power,'mixedPowerGain':mixed}
