"""Phase-canonical periodic pulse ablation around the existing captured bank.

This deliberately tests a representation change, not a new neural model. A
single central cycle of each generated template is periodized, restricted to
the source Nyquist band, and shifted so its fundamental has a common phase.
The same captured mean, PCA basis and random latent sequence are retained.
It may alter timbre when the captured signal period differs from the bank's
reference period; comparison against the unmodified renderer is required.
"""
import numpy as np
from pulse_basis_instrument import PulseBasisStream,OFFSETS,pulse_window

CYCLE_SAMPLES=512


def canonical_pulse(template,period,align=True):
    template=np.asarray(template,dtype=float)
    if (template.shape!=OFFSETS.shape or not np.isfinite(template).all()
            or not np.isfinite(period) or not 30<=period<=400):
        raise ValueError('Invalid pulse template or period')
    window=pulse_window(period)
    # Central cycle stays inside the raised-cosine window's >= 0.5 region;
    # division therefore cannot amplify near-zero window tails.
    sample_offsets=period*((np.arange(CYCLE_SAMPLES)/CYCLE_SAMPLES+.5)%1-.5)
    waveform=np.divide(template,window,out=np.zeros_like(template),where=window>1e-12)
    cycle=np.interp(sample_offsets,OFFSETS,waveform)
    spectrum=np.fft.rfft(cycle)
    maximum_harmonic=int(np.floor(period/2))
    spectrum[maximum_harmonic+1:]=0.
    if align:
        # A weak fundamental has an ill-defined phase. The zero-signal case is
        # left unchanged; near-weak fundamentals remain a measured limitation.
        phase=np.angle(spectrum[1]) if abs(spectrum[1])>1e-12 else 0.
        spectrum*=np.exp(-1j*np.arange(len(spectrum))*phase)
    periodic=np.fft.irfft(spectrum,n=CYCLE_SAMPLES)
    position=(OFFSETS/period%1)*CYCLE_SAMPLES
    return np.interp(position,np.arange(CYCLE_SAMPLES+1),np.r_[periodic,periodic[0]])*window


class CoherentPulseBasisStream(PulseBasisStream):
    """Same controls and delay as the captured pulse instrument."""
    def __init__(self,models,seed=0,amount=1.,correlated=True,align=True):
        super().__init__(models,seed=seed,amount=amount,correlated=correlated)
        self.align=bool(align)

    def _templates(self):
        templates=super()._templates()
        return np.array([canonical_pulse(w,p,self.align) for w,p in zip(templates,self.periods)])
