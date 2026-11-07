"""Reference-fitted, chunk-by-chunk continuous audio reconstruction component.

This is an analysis/resynthesis prototype, not an autonomous response generator
or accepted voice clone. Input is mono 24 kHz PCM. Output is raw floating-point
DSP audio, which can exceed [-1,1]; playback gain/limiting is a caller concern.
No file buffering, transcript, codebook, pretrained neural weights or backprop.
"""
from dataclasses import dataclass
import numpy as np
from aligned_analysis import AlignedAnalysis
from streaming_voicebox import VoiceboxConfig,mono
from continuous_source import ContinuousSource
from excitation import GeneratedVoicebox,VocalControls,pulse_profile
from noise_shape import noise_descriptor


@dataclass(frozen=True)
class VoiceboxPacket:
    start: int
    available_at: int
    samples: np.ndarray


class ReferenceVoicebox:
    """Frozen per-reference PCA source model with streaming DSP state.

    Fit happens once before streaming. Only filter/oscillator state changes while
    processing target audio; target audio never updates the reference weights.
    """
    def __init__(self,source_model,seed=79,shape_noise=False):
        if source_model.mean.shape!=(128,) or source_model.basis.shape!=(48,128):
            raise ValueError('This prototype expects 64 harmonics and 48 source coordinates')
        self.model=ContinuousSource(source_model.mean.copy(),source_model.basis.copy())
        self.model.mean.setflags(write=False)
        self.model.basis.setflags(write=False)
        self.config=VoiceboxConfig(order=32)
        self.analysis=AlignedAnalysis(self.config)
        self.shape_noise=bool(shape_noise)
        self.synthesis=GeneratedVoicebox(profile=pulse_profile(64),config=self.config,seed=seed,shape_noise=self.shape_noise)
        self.closed=False

    @classmethod
    def from_reference(cls,reference,seed=79,shape_noise=False):
        reference=mono(reference)
        if not 3*24000<=len(reference)<=30*24000:
            raise ValueError('Use 3–30 seconds of mono 24 kHz reference audio')
        analysis=AlignedAnalysis(VoiceboxConfig(order=32))
        frames=[]
        for at in range(0,len(reference),480):frames.extend(analysis.push(reference[at:at+480]))
        frames.extend(analysis.finish())
        model,_=ContinuousSource.fit(frames,dimensions=48,harmonics=64)
        return cls(model,seed,shape_noise=shape_noise)

    def _emit(self,frames):
        result=[]
        for frame in frames:
            coordinates=self.model.encode(frame)
            bands=noise_descriptor(frame) if self.shape_noise else None
            output=self.synthesis.push(VocalControls.from_frame(frame),harmonics=self.model.decode(coordinates),noise_bands=bands)
            result.append(VoiceboxPacket(frame.start,frame.available_at,output))
        return result

    def push(self,samples):
        if self.closed:raise ValueError('Voicebox stream is closed')
        return self._emit(self.analysis.push(samples))

    def finish(self):
        if self.closed:return []
        self.closed=True
        return self._emit(self.analysis.finish())

    def cancel(self):
        # Discard unissued samples, without synthesizing a trailing flush.
        self.closed=True
        self.analysis.pending=np.empty(0)
        self.analysis.closed=True
        self.analysis.past.fill(0);self.analysis.input_history.fill(0)
        self.synthesis.history.fill(0)
        if self.synthesis.noise_shaper is not None:
            self.synthesis.noise_shaper.history.fill(0)
            self.synthesis.noise_shaper.previous[:]=np.r_[1.,np.zeros(63)]
