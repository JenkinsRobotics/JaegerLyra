"""Continuous noise-color controls and causal finite-impulse synthesis.

Eight broad residual power bands; no original noise waveform is retained by the
decoder. For voiced frames, bins near harmonics are excluded where possible.
This is a rough noise-color estimate, NOT calibrated aperiodicity separation.
Classical minimum-phase cepstral factorization supplies a short shaping filter:
https://www.dsprelated.com/freebooks/sasp/Minimum_Phase_Filter_Design.html
The implementation is new research code, with no pretrained model or codebook.
"""
import numpy as np
from scipy.signal import lfilter

EDGES=np.array([0.,250.,500.,1000.,2000.,4000.,6000.,9000.,12000.])
CENTERS=(EDGES[:-1]+EDGES[1:])/2
BANDS=len(CENTERS)
TAPS=64
FFT_SIZE=512


def noise_descriptor(frame, rate=24000):
    if rate!=24000:raise ValueError('This research band layout requires 24 kHz')
    audio=np.asarray(frame.excitation_window,dtype=np.float64)
    if audio.ndim!=1 or len(audio)<64 or not np.isfinite(audio).all():
        raise ValueError('Invalid analysis residual window')
    window=np.hanning(len(audio))
    power=np.abs(np.fft.rfft(audio*window))**2
    if power.max()<1e-14:return np.zeros(BANDS)
    frequencies=np.fft.rfftfreq(len(audio),1/rate)
    keep=np.ones(len(frequencies),dtype=bool)
    if frame.f0>0 and frame.periodicity>=.6:
        distance=np.abs(frequencies-np.rint(frequencies/frame.f0)*frame.f0)
        keep=distance>max(35.,.12*frame.f0)
    values=[]
    for low,high in zip(EDGES[:-1],EDGES[1:]):
        mask=(frequencies>=low)&(frequencies<high)
        selected=mask&keep
        if selected.sum()<2:selected=mask
        values.append(float(np.mean(power[selected])))
    values=np.asarray(values)
    floor=max(float(power.max())*1e-8,1e-18)
    log_power=np.log(np.maximum(values,floor))
    log_power-=np.average(log_power,weights=np.diff(EDGES))
    # Fixed +/-12 dB band-color bounds, not target-dependent normalization.
    return np.clip(log_power,-np.log(10)*1.2,np.log(10)*1.2)


def shaping_filter(log_power):
    values=np.asarray(log_power,dtype=np.float64)
    if values.shape!=(BANDS,) or not np.isfinite(values).all():
        raise ValueError('Expected eight finite log-power bands')
    values=np.clip(values,-np.log(10)*1.2,np.log(10)*1.2)
    frequency=np.fft.rfftfreq(FFT_SIZE,1/24000)
    log_amplitude=.5*np.interp(frequency,CENTERS,values)
    cepstrum=np.fft.irfft(log_amplitude,n=FFT_SIZE)
    causal=np.zeros(FFT_SIZE)
    causal[0]=cepstrum[0]
    causal[1:FFT_SIZE//2]=2*cepstrum[1:FFT_SIZE//2]
    causal[FFT_SIZE//2]=cepstrum[FFT_SIZE//2]
    response=np.exp(np.fft.rfft(causal))
    taps=np.fft.irfft(response,n=FFT_SIZE)[:TAPS]
    # Normalize expected white-noise variance, never the observed waveform.
    return taps/np.linalg.norm(taps)


class ShapedNoise:
    """Continuous FIR interpolation, with owned bounded white-noise history."""
    def __init__(self):
        self.history=np.zeros(TAPS-1)
        self.previous=np.r_[1.,np.zeros(TAPS-1)]

    def process(self, white, log_power):
        white=np.asarray(white,dtype=np.float64)
        if white.ndim!=1 or not np.isfinite(white).all():
            raise ValueError('Invalid white-noise block')
        target=shaping_filter(log_power)
        if not len(white):return np.empty(0)
        context=np.r_[self.history,white]
        old=lfilter(self.previous,[1.],context)[len(self.history):]
        new=lfilter(target,[1.],context)[len(self.history):]
        blend=(np.arange(len(white))+1)/len(white)
        power=((1-blend)**2+blend**2
               +2*blend*(1-blend)*float(self.previous@target))
        output=((1-blend)*old+blend*new)/np.sqrt(np.maximum(power,1e-8))
        self.history=context[-(TAPS-1):].copy()
        self.previous=target
        return output
