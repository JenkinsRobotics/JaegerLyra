"""Captured harmonic amplitude/phase ablations for the compact instrument.

The source is a short decoded neural vowel, not a target-length recording.
Least squares fits fixed sinusoidal features to alternating source cycles;
unused cycles diagnose the limits of a static periodic description. This is
classical captured harmonic synthesis, not a learned pronunciation model.
Reference: Stylianou, Applying the Harmonic Plus Noise Model in Concatenative
Speech Synthesis (2001), https://www.ee.columbia.edu/~dpwe/e6820/papers/Styl01-hnm.pdf
No upstream implementation is copied.
"""
from dataclasses import dataclass
import json
import numpy as np
from compact_vocal_instrument import (CompactShape, CompactVocalStream, envelope,
    GRID, RATE, HARMONICS, CEPSTRA, NOISE_HZ)


@dataclass(frozen=True)
class HarmonicShape:
    compact: CompactShape
    source_hz: float
    log_amplitudes: np.ndarray
    phase_residuals: np.ndarray

    def validate(self):
        self.compact.validate()
        if (not np.isfinite(self.source_hz) or not 60 <= self.source_hz <= 800
                or self.log_amplitudes.ndim != 1 or not 4 <= len(self.log_amplitudes) <= 192
                or self.phase_residuals.shape != self.log_amplitudes.shape
                or not np.isfinite(self.log_amplitudes).all() or np.max(abs(self.log_amplitudes)) > 40
                or not np.isfinite(self.phase_residuals).all() or np.max(abs(self.phase_residuals)) > 2000):
            raise ValueError('Invalid captured harmonic shape')


def fit_harmonics(audio, pulses, compact):
    """Fit on even cycles only; retain every odd-cycle result without selection."""
    audio=np.asarray(audio,dtype=float);pulses=np.asarray(pulses,dtype=float)
    if (audio.ndim != 1 or not np.isfinite(audio).all() or pulses.ndim != 1 or len(pulses)<6
            or not np.isfinite(pulses).all() or np.any(np.diff(pulses)<=0)
            or pulses[0]<0 or pulses[-1]>=len(audio)):
        raise ValueError('Invalid voiced source')
    periods=np.diff(pulses);pitch=RATE/np.median(periods)
    if not 60 <= pitch <= 800:raise ValueError('Source pitch outside instrument range')
    count=min(192,int((RATE/2-500)/(RATE/min(periods))))
    if count<4:raise ValueError('Insufficient supported harmonics')
    index=np.arange(int(np.ceil(pulses[0])),int(np.ceil(pulses[-1])))
    cycles=np.interp(index,pulses,np.arange(len(pulses)))
    numbers=np.arange(1,count+1)
    angles=2*np.pi*cycles[:,None]*numbers
    basis=np.column_stack((np.ones(len(index)),np.cos(angles),np.sin(angles)))
    train=(np.floor(cycles).astype(int)%2)==0
    target=audio[index]
    coefficients=np.linalg.lstsq(basis[train],target[train],rcond=1e-10)[0]
    predicted=basis@coefficients
    complex_coefficients=coefficients[1:count+1]-1j*coefficients[count+1:]
    amplitudes=np.maximum(abs(complex_coefficients),1e-15)
    # Fix the arbitrary cycle origin using fundamental phase. Removing this
    # linear phase is a time shift, not a magnitude or identity modification.
    phase=np.angle(complex_coefficients)-numbers*np.angle(complex_coefficients[0])
    _,minimum=envelope(compact.cepstrum)
    residual=np.unwrap(phase-np.interp(numbers*pitch,GRID,minimum))
    shape=HarmonicShape(compact,float(pitch),np.log(amplitudes),residual)
    shape.validate()
    for array in (shape.log_amplitudes,shape.phase_residuals):array.setflags(write=False)
    def snr(mask):
        return float(10*np.log10(max(np.sum(target[mask]**2),1e-30)/max(np.sum((target[mask]-predicted[mask])**2),1e-30)))
    diagnostics={'sourcePitchHz':float(pitch),'harmonics':count,'fitSamples':int(sum(train)),
        'heldCycleSamples':int(sum(~train)),'fitCycleSnrDb':snr(train),'heldCycleSnrDb':snr(~train),
        'capturedDc':float(coefficients[0]),'dcRendered':False,
        'fundamentalPhaseRemovedRadians':float(np.angle(complex_coefficients[0])),
        'sourcePeriodCv':float(np.std(periods)/np.mean(periods)),
        'scope':'Static sinusoidal least-squares fit on even cycles, odd source cycles held out; not new-speaker generalization'}
    return shape,diagnostics


def save_bank(path,shapes,metadata):
    if not shapes:raise ValueError('Empty harmonic bank')
    for shape in shapes:shape.validate()
    data={'cepstrum':np.array([s.compact.cepstrum for s in shapes]),
        'noise':np.array([s.compact.noise for s in shapes]),'rms':np.array([s.compact.rms for s in shapes]),
        'source_hz':np.array([s.source_hz for s in shapes]),'metadata':json.dumps(metadata)}
    for i,shape in enumerate(shapes):
        data['log_amplitudes_'+str(i)]=shape.log_amplitudes
        data['phase_residuals_'+str(i)]=shape.phase_residuals
    np.savez(path,**data)


def load_bank(path):
    shapes=[]
    with np.load(path,allow_pickle=False) as data:
        c,n,r,f=data['cepstrum'],data['noise'],data['rms'],data['source_hz']
        if (c.ndim!=2 or c.shape[1]!=CEPSTRA or not len(c) or n.shape!=(len(c),len(NOISE_HZ))
                or r.shape!=(len(c),) or f.shape!=(len(c),)):raise ValueError('Invalid harmonic bank dimensions')
        for i in range(len(c)):
            shape=HarmonicShape(CompactShape(c[i].copy(),n[i].copy(),float(r[i])),float(f[i]),
                data['log_amplitudes_'+str(i)].copy(),data['phase_residuals_'+str(i)].copy())
            shape.validate()
            for a in (shape.compact.cepstrum,shape.compact.noise,shape.log_amplitudes,shape.phase_residuals):a.setflags(write=False)
            shapes.append(shape)
        metadata=json.loads(str(data['metadata']))
    return shapes,metadata


class CapturedHarmonicStream(CompactVocalStream):
    """Immediate-control ablation; compact noise path stays unchanged.

    Magnitude samples are interpolated at absolute frequency to retain the
    captured spectral shape when pitch changes. Residual phase is likewise
    frequency-referenced. All shapes share a canonical fundamental phase.
    These interpolation choices are hypotheses tested by the study, not claims
    of exact formant preservation or physically accurate glottal control.
    """
    def __init__(self,shapes,seed=0,magnitude=True,phase=True):
        for shape in shapes:shape.validate()
        super().__init__([s.compact for s in shapes],seed=seed)
        self.source_hz=np.array([s.source_hz for s in shapes]);self.source_hz.setflags(write=False)
        self.log_amplitudes=tuple(s.log_amplitudes.copy() for s in shapes)
        self.phase_residuals=tuple(s.phase_residuals.copy() for s in shapes)
        for a in (*self.log_amplitudes,*self.phase_residuals):a.setflags(write=False)
        self.use_magnitude=bool(magnitude);self.use_phase=bool(phase)

    def _parameters(self,frame):
        base,impulse=super()._parameters(frame)
        if not self.use_magnitude and not self.use_phase:return base,impulse
        frequencies=HARMONICS*frame.hz
        weights=np.asarray(frame.weights)
        amplitude=abs(base)
        if self.use_magnitude:
            logs=[]
            for f,a in zip(self.source_hz,self.log_amplitudes):
                grid=np.r_[np.arange(1,len(a)+1)*f,RATE/2]
                logs.append(np.interp(frequencies,grid,np.r_[a,-35.]))
            amplitude=np.exp(weights@np.array(logs))
            amplitude*=np.clip((RATE/2-frequencies)/500.,0.,1.)
            # Keep the exact harmonic/noise power allocation of the compact
            # comparator so this study does not confound excitation amount.
            amplitude*=np.sqrt(np.sum(abs(base)**2)/max(np.sum(amplitude**2),1e-30))
        phase=np.angle(base)
        if self.use_phase:
            residuals=[np.interp(frequencies,np.arange(1,len(p)+1)*f,p)
                for f,p in zip(self.source_hz,self.phase_residuals)]
            phase=phase+weights@np.array(residuals)
        return amplitude*np.exp(1j*phase),impulse
