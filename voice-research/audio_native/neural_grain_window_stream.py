"""Experimental held-grain widths over the unchanged syllable controller.

The established minimum(source period, target period) option is retained.
Alternatives affect held grains only; prefix/release grains keep the baseline
rule. Every half-window is bounded by the existing 20 ms waveform delay.
"""
import numpy as np
from neural_syllable_stream import NeuralSyllableStream
from compact_vocal_instrument import RATE
from live_neural_grains import DELAY


class NeuralGrainWindowStream(NeuralSyllableStream):
    def __init__(self,unit,decoder,release_sample,window='minimum',**kwargs):
        if window not in ('minimum','source','wide'):raise ValueError('Unknown held-grain window')
        super().__init__(unit,decoder,release_sample,**kwargs)
        self.window_policy=window;self.maximum_grain_radius=0.;self.radius_clipped_grains=0

    def _pulse(self,where,hz,region):
        output_time=self.position+where
        held=self.release_position is None and output_time>=self.hold_start
        if self.window_policy=='minimum' or not held:return super()._pulse(where,hz,region)
        first,last,*_=self.regions[region]
        index=self.choices[self.sustain_pulses%len(self.choices)];self.sustain_pulses+=1
        left=self.pulses[index]-self.pulses[index-1] if index>first else self.pulses[first+1]-self.pulses[first]
        right=self.pulses[index+1]-self.pulses[index] if index<last else self.pulses[last]-self.pulses[last-1]
        factor=1.5 if self.window_policy=='wide' else 1.
        left*=factor;right*=factor
        self.radius_clipped_grains+=int(max(left,right)>DELAY-1)
        left=min(left,DELAY-1);right=min(right,DELAY-1)
        self.maximum_grain_radius=max(self.maximum_grain_radius,left,right)
        center=where+DELAY;indices=np.arange(int(np.ceil(center-left)),int(np.floor(center+right))+1)
        offsets=indices-center;width=np.where(offsets<0,left,right);window=.5+.5*np.cos(np.pi*offsets/width)
        self.pending[indices]+=self.reader.sample(self.pulses[index]+offsets)*window
        self.weights[indices]+=window;self.generated_pulses+=1
