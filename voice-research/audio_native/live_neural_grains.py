"""Immediate-control captured-vowel synthesis using lazy neural window decoding.

Only short encoded reference patches and the fixed decoder are required.
There is no target-duration score or growing event list. A fixed 20 ms delay
makes centred grains causal. This reuses captured vowel microvariation; it is
not an independent phoneme model or generation of new speech content.
"""
import numpy as np
from compact_vocal_instrument import VocalFrame,BLOCK,RATE
from neural_performance import validate_pulses
from neural_performance_stream import NeuralWindowReader

DELAY=480
CAPACITY=2*DELAY+BLOCK+2


class LiveNeuralGrainStream:
    def __init__(self,patches,decoder,cache_frames=16):
        if not patches:raise ValueError('Expected captured vowel patches')
        self.pulses=[];self.periods=[];self.choices=[];self.readers=[]
        for patch in patches:
            if not 0<patch.matrix.sample_count<=RATE*.200:raise ValueError('Expected <=200 ms vowel patches')
            pulses=validate_pulses(patch.pulses,patch.matrix.sample_count).copy()
            periods=np.diff(pulses)
            if len(pulses)<4 or min(periods)<RATE/1000 or max(periods)>DELAY:
                raise ValueError('Invalid periodic vowel patch')
            choices=np.r_[np.arange(1,len(pulses)-1),np.arange(len(pulses)-3,1,-1)]
            for array in (pulses,periods,choices):array.setflags(write=False)
            self.pulses.append(pulses);self.periods.append(periods);self.choices.append(choices)
            self.readers.append(NeuralWindowReader(patch.matrix,decoder,cache_frames))
        self.pending=np.zeros((len(patches),CAPACITY));self.weights=np.zeros_like(self.pending)
        self.controls=np.zeros_like(self.pending)
        self.phase=0.;self.previous=None;self.position=0;self.generated_pulses=0;self.closed=False

    def _pulse(self,where,hz):
        center=where+DELAY;target_period=RATE/hz
        for v,(pulses,periods,choices,reader) in enumerate(zip(self.pulses,self.periods,self.choices,self.readers)):
            index=choices[self.generated_pulses%len(choices)]
            left=min(periods[index-1],target_period);right=min(periods[index],target_period)
            indices=np.arange(int(np.ceil(center-left)),int(np.floor(center+right))+1)
            offsets=indices-center;widths=np.where(offsets<0,left,right)
            window=.5+.5*np.cos(np.pi*offsets/widths)
            self.pending[v,indices]+=reader.sample(pulses[index]+offsets)*window
            self.weights[v,indices]+=window
        self.generated_pulses+=1

    def _take(self,count):
        audio=np.divide(self.pending[:,:count],self.weights[:,:count],
            out=np.zeros((len(self.readers),count)),where=self.weights[:,:count]>1e-12)
        mixed=np.sum(audio*self.controls[:,:count],axis=0)
        for array in (self.pending,self.weights,self.controls):
            array[:,:-count]=array[:,count:];array[:,-count:]=0.
        return mixed

    def push(self,frame):
        if self.closed:raise RuntimeError('Live vowel instrument closed')
        frame.validate(len(self.readers))
        if frame.breath!=0:raise ValueError('Additional breath control is not supported by captured neural grains')
        frame=VocalFrame(float(frame.hz),tuple(frame.weights),float(frame.gain),0.)
        previous=frame if self.previous is None else self.previous
        if self.previous is None:self._pulse(0.,frame.hz)
        progress=(np.arange(BLOCK)+1)/BLOCK
        frequency=np.exp(np.log(previous.hz)+progress*np.log(frame.hz/previous.hz))
        weights=np.asarray(previous.weights)[None,:]*(1-progress[:,None])+np.asarray(frame.weights)[None,:]*progress[:,None]
        gain=previous.gain+(frame.gain-previous.gain)*progress
        self.controls[:,DELAY:DELAY+BLOCK]=(weights*gain[:,None]).T
        for i,hz in enumerate(frequency):
            step=hz/RATE
            if self.phase+step>=1.:
                self._pulse(i+(1-self.phase)/step,float(hz))
            self.phase=(self.phase+step)%1.
        result=self._take(BLOCK)
        self.previous=frame;self.position+=BLOCK
        return result

    def finish(self):
        if self.closed:return np.empty(0)
        tail=self._take(DELAY) if self.position else np.empty(0)
        self.cancel();return tail

    def cancel(self):
        self.closed=True;self.phase=0.;self.previous=None
        for array in (self.pending,self.weights,self.controls):array.fill(0.)
        for reader in self.readers:reader.clear()
