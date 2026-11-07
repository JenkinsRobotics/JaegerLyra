"""Optional original-context preservation around a controlled captured vowel.

The retained context is independently decoded source detail, not original PCM.
This experimental policy preserves the captured pitch of the prefix and/or
release, with fixed 30 ms cosine transitions to the retuned vowel path.
"""
import numpy as np
from neural_syllable_stream import NeuralSyllableStream
from compact_vocal_instrument import RATE
from live_neural_grains import DELAY


class NeuralContextStream(NeuralSyllableStream):
    def __init__(self,unit,decoder,release_sample,context='both',fade_seconds=.03,**kwargs):
        if context not in ('prefix','release','both'):raise ValueError('Unknown source-context policy')
        if (isinstance(fade_seconds,(bool,np.bool_)) or not isinstance(fade_seconds,(int,float,np.number))
                or not np.isfinite(fade_seconds) or not 1/RATE<=fade_seconds<=.2):
            raise ValueError('Context fade must be finite and between one sample and 200 ms')
        super().__init__(unit,decoder,release_sample,**kwargs)
        if fade_seconds*RATE>self.hold_start:raise ValueError('Context fade exceeds available prefix')
        self.context=context;self.context_fade=float(fade_seconds)*RATE

    def _take(self,count):
        # The parent's pending samples are delayed. Use their output timeline,
        # including the still-pending prefix after a release command arrives.
        times=self.position-DELAY+np.arange(count);source_times=times.copy();amount=np.zeros(count)
        if self.context in ('prefix','both'):
            phase=np.clip((self.hold_start-times)/self.context_fade,0.,1.)
            amount=.5-.5*np.cos(np.pi*phase)
        if self.context in ('release','both') and self.release_position is not None:
            released=times>=self.release_position
            phase=np.clip((times[released]-self.release_position)/self.context_fade,0.,1.)
            amount[released]=.5-.5*np.cos(np.pi*phase)
            source_times[released]=self.release_sample+times[released]-self.release_position
        amount[times<0]=0.
        original=np.zeros(count);selected=amount>0
        if np.any(selected):original[selected]=self.reader.sample(source_times[selected].astype(float))
        gain=self.gain[:count].copy();controlled=super()._take(count)
        return controlled*(1-amount)+original*gain*amount
