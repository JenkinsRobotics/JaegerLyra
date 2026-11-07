"""Experimental spectral colour over neural-decoded windows.

This is a bounded DSP edit, not a learned speaker identity or a physical vocal
tract. A smoothed log-magnitude envelope is shifted while the harmonic frequency
grid and phase remain in place. Brightness adds a separate spectral tilt.
Neutral controls bypass all processing, preserving the accepted reconstruction.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from neural_waveform import RATE, SIZE


class ColourDecoder:
    def __init__(self, decoder, resonance=1., brightness=0.):
        if not np.isfinite([resonance,brightness]).all() or not .85<=resonance<=1.18 or not -4<=brightness<=4:
            raise ValueError('Invalid voice colour')
        self.decoder=decoder;self.config=decoder.config
        self.resonance=resonance;self.brightness=brightness
        self.neutral=resonance==1. and brightness==0.
        self.frequencies=np.fft.rfftfreq(SIZE,1/RATE)
        # Brightness is dB per octave, pivoting at 1 kHz, bounded at the edges.
        self.tilt=brightness*np.clip(np.log2(np.maximum(self.frequencies,125)/1000),-3,2)

    def frame(self, controls, gain=1.):
        frame=self.decoder.frame(controls,gain)
        if self.neutral or not np.any(frame):return frame
        spectrum=np.fft.rfft(frame);magnitude=abs(spectrum)
        # Broad envelope smoothing deliberately avoids moving individual bins.
        envelope=gaussian_filter1d(np.log(np.maximum(magnitude,max(magnitude)*1e-4)),3.,mode='nearest')
        shifted=np.interp(self.frequencies/self.resonance,self.frequencies,envelope)
        log_gain=np.clip(shifted-envelope+self.tilt*np.log(10)/20,-9*np.log(10)/20,9*np.log(10)/20)
        result=np.fft.irfft(spectrum*np.exp(log_gain),n=SIZE)
        # Separate colour from amplitude as far as this frame-local edit allows.
        energy=float(np.dot(result,result))
        if energy>1e-20:result*=np.sqrt(np.dot(frame,frame)/energy)
        return result
