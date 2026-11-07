"""Bounded-lookahead source/filter analysis for generated excitation research.

The 40 ms analysis window is centered on the 10 ms emitted audio frame, rather
than ending at that frame. This costs 15 ms future context (25 ms first-frame
buffering). available_at records the exact required input position. No hidden
whole-file analysis. Existing zero-lookahead StreamingAnalysis stays unchanged.
"""
import numpy as np
from scipy.signal import lfilter,lfiltic
from streaming_voicebox import AudioFrame,VoiceboxConfig,lpc,pitch,mono


class AlignedAnalysis:
    def __init__(self,config=VoiceboxConfig()):
        self.config=config
        self.lookahead=(config.history-config.hop)//2
        self.past=np.zeros(config.history-config.hop-self.lookahead)
        self.input_history=np.zeros(config.order)
        self.pending=np.empty(0)
        self.position=0;self.received=0;self.closed=False

    def _frame(self,count):
        cfg=self.config
        samples=self.pending[:count]
        # Pad only on an explicit end-of-stream flush; during push() there is
        # enough observed context to fill the entire future side of the window.
        future=self.pending[:cfg.hop+self.lookahead]
        window=np.r_[self.past,future,np.zeros(max(0,cfg.hop+self.lookahead-len(future)))]
        coefficients=lpc(window,cfg)
        state=lfiltic(coefficients,[1.],[],self.input_history[::-1])
        residual,_=lfilter(coefficients,[1.],samples,zi=state)
        f0,periodicity=pitch(window,cfg.sample_rate)
        frame=AudioFrame(self.position,min(self.received,self.position+cfg.hop+self.lookahead),
                         coefficients,residual,f0,periodicity,float(np.sqrt(np.mean(samples**2))))
        # Analysis-only window for compact source descriptors. It uses exactly
        # the same declared context and never enters the generated decoder.
        frame.excitation_window=lfilter(coefficients,[1.],window)
        frame.window_rms=float(np.sqrt(np.mean(window**2)))
        self.past=np.r_[self.past,samples][-len(self.past):]
        self.input_history=np.r_[self.input_history,samples][-cfg.order:]
        self.pending=self.pending[count:];self.position+=count
        return frame

    def push(self,samples):
        if self.closed:raise ValueError('Analysis has finished')
        samples=mono(samples)
        self.received+=len(samples);self.pending=np.r_[self.pending,samples]
        frames=[]
        while len(self.pending)>=self.config.hop+self.lookahead:
            frames.append(self._frame(self.config.hop))
        self.pending=self.pending.copy()
        return frames

    def finish(self):
        if self.closed:return []
        self.closed=True;frames=[]
        while len(self.pending):frames.append(self._frame(min(self.config.hop,len(self.pending))))
        return frames
