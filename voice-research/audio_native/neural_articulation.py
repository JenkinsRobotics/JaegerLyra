"""Captured consonant attacks followed by an indefinitely controlled vowel.

The source unit is an encoded short recording, not learned pronunciation.
Source unvoiced intervals use the frozen neural decoder directly; voiced
intervals and the sustained nucleus use causal pulse-synchronous overlap/add.
No complete future performance score or original PCM is used at runtime.
"""
from dataclasses import dataclass
import numpy as np
from compact_vocal_instrument import VocalFrame,RATE,BLOCK
from neural_performance import compile_performance,validate_pulses
from neural_performance_stream import NeuralWindowReader
from live_neural_grains import DELAY,CAPACITY


@dataclass(frozen=True)
class ArticulationUnit:
    performance: object
    hold_first: int
    hold_last: int

    def validate(self):
        p=validate_pulses(self.performance.pulses,self.performance.matrix.sample_count)
        if (not 0<self.performance.matrix.sample_count<=RATE*3 or
                not isinstance(self.hold_first,(int,np.integer)) or not isinstance(self.hold_last,(int,np.integer))
                or not 1<=self.hold_first<self.hold_last<len(p)-1
                or np.any(np.diff(p[self.hold_first-1:self.hold_last+2])>DELAY)
                or np.any(np.diff(p[self.hold_first-1:self.hold_last+2])<RATE/1000)):
            raise ValueError('Invalid captured articulation unit')


def candidate_spans(audio,p):
    """Source-only eligible spans, shared by baseline and stationarity study."""
    audio=np.asarray(audio,dtype=float)
    active=np.flatnonzero(abs(audio)>.03*max(abs(audio)))
    if not len(active):raise ValueError('Silent unit')
    # Use a stable late voiced region to reduce the chance of looping the
    # onset itself. This is a source-only heuristic, not phoneme recognition.
    earliest=active[0]+.35*(active[-1]-active[0]);candidates=[]
    for group in np.split(np.arange(len(p)),np.flatnonzero(np.diff(p)>DELAY)+1):
        if len(group)<8:continue
        for j in group[1:-6]:
            if p[j]<earliest:continue
            end=min(int(np.searchsorted(p,p[j]+RATE*.150,side='right')-1),int(group[-2]))
            if end-j<6 or p[end]-p[j]<RATE*.060:continue
            periods=np.diff(p[j:end+1])
            if min(periods)<RATE/1000 or max(periods)>1.25*min(periods):continue
            energies=np.array([np.sqrt(np.mean(audio[int(a):int(b)]**2)) for a,b in zip(p[j:end],p[j+1:end+1])])
            if min(energies)<.003:continue
            score=float(np.std(np.log(periods))+.15*np.std(np.log(energies)))
            candidates.append((score,int(j),int(end)))
    if not candidates:raise ValueError('No stable late vowel region')
    return earliest,candidates


def compile_unit(audio,encoder):
    audio=np.asarray(audio,dtype=float)
    performance=compile_performance(audio,encoder);p=performance.pulses
    earliest,candidates=candidate_spans(audio,p)
    score,first,last=min(candidates)
    unit=ArticulationUnit(performance,first,last);unit.validate()
    return unit,{'holdFirstPulse':first,'holdLastPulse':last,'holdStartSample':float(p[first]),
        'holdEndSample':float(p[last]),'sourceMedianPitchHz':float(RATE/np.median(np.diff(p[first:last+1]))),
        'sourceSelectionScore':score,'sourceSelectionCandidates':len(candidates),
        'lateRegionThresholdSample':float(earliest),'sourceSeconds':len(audio)/RATE,
        'recognizedPhones':False,'capturedSourceUnit':True}


class NeuralArticulationStream:
    """One source onset, then a controllable sustain; immediate 5 ms frames.

    Pitch control applies to voiced portions of the attack as well as sustain.
    Unvoiced detail is not pitched. The fixed 20 ms output delay is additional
    to computation and audio-device latency. Cancellation discards pending PCM.
    """
    def __init__(self,unit,decoder,cache_frames=16,hold_mode='loop'):
        if hold_mode not in ('loop','fixed-first'):raise ValueError('Unknown captured sustain mode')
        unit.validate();self.pulses=unit.performance.pulses.copy();self.pulses.setflags(write=False)
        self.first=unit.hold_first;self.last=unit.hold_last;self.hold_start=float(self.pulses[self.first])
        self.hold_mode=hold_mode
        self.choices=(np.r_[np.arange(self.first,self.last+1),np.arange(self.last-1,self.first,-1)]
            if hold_mode=='loop' else np.array([self.first]))
        self.choices.setflags(write=False)
        self.reader=NeuralWindowReader(unit.performance.matrix,decoder,cache_frames)
        self.regions=[]
        for group in np.split(np.arange(len(self.pulses)),np.flatnonzero(np.diff(self.pulses)>DELAY)+1):
            if len(group)<3:continue
            first,last=int(group[0]),int(group[-1]);periods=np.diff(self.pulses[group])
            if min(periods)<RATE/1000:raise ValueError('Unreasonably close source pulses')
            if self.pulses[first]>self.hold_start:break
            holds=first<=self.first<=last
            self.regions.append((first,last,float(self.pulses[first]),float('inf') if holds else float(self.pulses[last]),
                float(periods[0]),float(periods[-1])))
        self.pending=np.zeros(CAPACITY);self.weights=np.zeros(CAPACITY)
        self.background=np.zeros(CAPACITY);self.voicing=np.zeros(CAPACITY);self.gain=np.zeros(CAPACITY)
        self.position=0;self.phase=0.;self.previous=None;self.active=-1;self.generated_pulses=0;self.sustain_pulses=0;self.closed=False

    def _pulse(self,where,hz,region):
        source_time=self.position+where
        if source_time>=self.hold_start:
            index=self.choices[self.sustain_pulses%len(self.choices)];self.sustain_pulses+=1
        else:
            first,last,*_=self.regions[region]
            right=int(np.clip(np.searchsorted(self.pulses,source_time),first,last));left=max(first,right-1)
            index=left if abs(source_time-self.pulses[left])<=abs(source_time-self.pulses[right]) else right
        first,last,*_=self.regions[region]
        a=self.pulses[index]-self.pulses[index-1] if index>first else self.pulses[first+1]-self.pulses[first]
        b=self.pulses[index+1]-self.pulses[index] if index<last else self.pulses[last]-self.pulses[last-1]
        a=min(a,RATE/hz);b=min(b,RATE/hz);center=where+DELAY
        indices=np.arange(int(np.ceil(center-a)),int(np.floor(center+b))+1)
        offsets=indices-center;width=np.where(offsets<0,a,b)
        window=.5+.5*np.cos(np.pi*offsets/width)
        self.pending[indices]+=self.reader.sample(self.pulses[index]+offsets)*window
        self.weights[indices]+=window;self.generated_pulses+=1

    def _take(self,count):
        voiced=np.divide(self.pending[:count],self.weights[:count],out=np.zeros(count),where=self.weights[:count]>1e-12)
        output=(voiced*self.voicing[:count]+self.background[:count]*(1-self.voicing[:count]))*self.gain[:count]
        for a in (self.pending,self.weights,self.background,self.voicing,self.gain):
            a[:-count]=a[count:];a[-count:]=0.
        return output

    def push(self,frame):
        if self.closed:raise RuntimeError('Articulation stream closed')
        frame.validate(1)
        if frame.breath!=0:raise ValueError('Additional breath control unsupported')
        frame=VocalFrame(float(frame.hz),tuple(frame.weights),float(frame.gain),0.)
        previous=frame if self.previous is None else self.previous
        progress=(np.arange(BLOCK)+1)/BLOCK
        frequency=np.exp(np.log(previous.hz)+progress*np.log(frame.hz/previous.hz))
        times=self.position+np.arange(BLOCK)
        background=np.zeros(BLOCK);prefix=times<self.hold_start
        if np.any(prefix):background[prefix]=self.reader.sample(times[prefix].astype(float))
        mask=np.zeros(BLOCK)
        for first,last,start,end,left,right in self.regions:
            onset=np.clip((times-(start-left))/left,0.,1.)
            release=np.ones(BLOCK) if np.isinf(end) else np.clip((end+right-times)/right,0.,1.)
            mask=np.maximum(mask,(.5-.5*np.cos(np.pi*onset))*(.5-.5*np.cos(np.pi*release)))
        self.background[DELAY:DELAY+BLOCK]=background;self.voicing[DELAY:DELAY+BLOCK]=mask
        self.gain[DELAY:DELAY+BLOCK]=previous.gain+(frame.gain-previous.gain)*progress
        for i,hz in enumerate(frequency):
            now=self.position+i
            selected=next((j for j,r in enumerate(self.regions) if now+1>r[2] and now<r[3]),-1)
            if selected<0:self.active=-1;self.phase=0.;continue
            start=max(float(now),self.regions[selected][2]);end=min(float(now+1),self.regions[selected][3])
            if self.active!=selected:
                self.active=selected;self.phase=0.;self._pulse(start-self.position,float(hz),selected)
            step=(end-start)*hz/RATE
            if self.phase+step>=1.:
                where=start-self.position+(1-self.phase)*RATE/hz
                self._pulse(where,float(hz),selected)
            self.phase=(self.phase+step)%1.
        output=self._take(BLOCK);self.position+=BLOCK;self.previous=frame
        return output

    def finish(self):
        if self.closed:return np.empty(0)
        tail=self._take(DELAY) if self.position else np.empty(0)
        self.cancel();return tail

    def cancel(self):
        self.closed=True;self.reader.clear();self.previous=None;self.phase=0.;self.active=-1
        for a in (self.pending,self.weights,self.background,self.voicing,self.gain):a.fill(0.)
