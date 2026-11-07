"""Immediate captured onset/hold/release controls over the independent decoder.

The source is an encoded syllable with offline pulse and release annotations.
No pretrained Kokoro model, original PCM or future score is used at runtime.
This is captured articulation playback/resynthesis, not learned pronunciation.
"""
import numpy as np
from neural_articulation import NeuralArticulationStream
from compact_vocal_instrument import VocalFrame,RATE,BLOCK
from live_neural_grains import DELAY


class NeuralSyllableStream(NeuralArticulationStream):
    def __init__(self,unit,decoder,release_sample,cache_frames=16,hold_mode='loop'):
        unit.validate()
        if (isinstance(release_sample,(bool,np.bool_)) or not np.isscalar(release_sample)
                or not np.isfinite(release_sample) or
                not unit.performance.pulses[unit.hold_last]<release_sample<unit.performance.matrix.sample_count):
            raise ValueError('Release source position must follow the held span and precede the source end')
        super().__init__(unit,decoder,cache_frames,hold_mode)
        self.release_sample=float(release_sample);self.release_position=None;self.regions=[]
        for group in np.split(np.arange(len(self.pulses)),np.flatnonzero(np.diff(self.pulses)>DELAY)+1):
            if len(group)<3:continue
            first,last=int(group[0]),int(group[-1]);periods=np.diff(self.pulses[group])
            if min(periods)<RATE/1000:raise ValueError('Unreasonably close source pulses')
            self.regions.append((first,last,float(self.pulses[first]),float(self.pulses[last]),
                float(periods[0]),float(periods[-1])))

    def release(self):
        """Start the encoded release at the next packet; output delay stays 20 ms."""
        if self.closed:raise RuntimeError('Syllable stream closed')
        if self.release_position is not None:return False
        if self.position<self.hold_start:raise ValueError('Cannot release before reaching the held vowel')
        self.release_position=self.position
        return True

    def _source_time(self,output_time):
        if self.release_position is not None:
            return self.release_sample+output_time-self.release_position
        return np.minimum(output_time,self.hold_start)

    def _pulse(self,where,hz,region):
        output_time=self.position+where;source_time=float(self._source_time(output_time))
        first,last,*_=self.regions[region]
        if self.release_position is None and output_time>=self.hold_start:
            index=self.choices[self.sustain_pulses%len(self.choices)];self.sustain_pulses+=1
        else:
            right=int(np.clip(np.searchsorted(self.pulses,source_time),first,last));left=max(first,right-1)
            index=left if abs(source_time-self.pulses[left])<=abs(source_time-self.pulses[right]) else right
        a=self.pulses[index]-self.pulses[index-1] if index>first else self.pulses[first+1]-self.pulses[first]
        b=self.pulses[index+1]-self.pulses[index] if index<last else self.pulses[last]-self.pulses[last-1]
        a=min(a,RATE/hz);b=min(b,RATE/hz);center=where+DELAY
        indices=np.arange(int(np.ceil(center-a)),int(np.floor(center+b))+1)
        offsets=indices-center;width=np.where(offsets<0,a,b)
        window=.5+.5*np.cos(np.pi*offsets/width)
        self.pending[indices]+=self.reader.sample(self.pulses[index]+offsets)*window
        self.weights[indices]+=window;self.generated_pulses+=1

    def push(self,frame):
        if self.closed:raise RuntimeError('Syllable stream closed')
        frame.validate(1)
        if frame.breath!=0:raise ValueError('Additional breath control unsupported')
        frame=VocalFrame(float(frame.hz),tuple(frame.weights),float(frame.gain),0.)
        previous=frame if self.previous is None else self.previous
        progress=(np.arange(BLOCK)+1)/BLOCK
        frequency=np.exp(np.log(previous.hz)+progress*np.log(frame.hz/previous.hz))
        output_times=self.position+np.arange(BLOCK);source_times=self._source_time(output_times)
        background=np.zeros(BLOCK)
        moving=np.ones(BLOCK,dtype=bool) if self.release_position is not None else output_times<self.hold_start
        if np.any(moving):background[moving]=self.reader.sample(source_times[moving].astype(float))
        mask=np.zeros(BLOCK)
        for first,last,start,end,left,right in self.regions:
            onset=np.clip((source_times-(start-left))/left,0.,1.)
            release=np.clip((end+right-source_times)/right,0.,1.)
            mask=np.maximum(mask,(.5-.5*np.cos(np.pi*onset))*(.5-.5*np.cos(np.pi*release)))
        self.background[DELAY:DELAY+BLOCK]=background;self.voicing[DELAY:DELAY+BLOCK]=mask
        self.gain[DELAY:DELAY+BLOCK]=previous.gain+(frame.gain-previous.gain)*progress
        for i,hz in enumerate(frequency):
            source_time=float(source_times[i])
            # Keep synthesis alive through the final voicing fade. Otherwise a
            # last pulse that falls on the boundary can be lost to rounding,
            # leaving a one-period hole before the unvoiced tail takes over.
            selected=next((j for j,r in enumerate(self.regions) if source_time+1>r[2] and source_time<r[3]+r[5]),-1)
            if selected<0:self.active=-1;self.phase=0.;continue
            start=max(0.,self.regions[selected][2]-source_time)
            end=min(1.,self.regions[selected][3]+self.regions[selected][5]-source_time)
            if self.active!=selected:
                self.active=selected;self.phase=0.;self._pulse(i+start,float(hz),selected)
            step=(end-start)*hz/RATE
            if self.phase+step>=1.:
                self._pulse(i+start+(1-self.phase)*RATE/hz,float(hz),selected)
            self.phase=(self.phase+step)%1.
        output=self._take(BLOCK);self.position+=BLOCK;self.previous=frame
        return output

    def cancel(self):
        super().cancel();self.release_position=None
