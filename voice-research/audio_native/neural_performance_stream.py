"""Chunked performance output with lazy neural window decoding.

The source is a precompiled control matrix and pulse schedule, not incoming
speech or live text. Those controls and derived event metadata remain resident.
Only the decoded waveform cache is bounded; this is not a claim that the whole
model/score uses constant memory. No original PCM or full decoded utterance is
stored. The offline v2 renderer remains an independent numerical reference.
"""
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
from neural_waveform import ControlMatrix,RATE,SIZE,HOP,WINDOW
from neural_performance import PerformanceControls,validate_pulses


class NeuralWindowReader:
    def __init__(self,matrix,decoder,cache_frames=16):
        count=matrix.sample_count
        if (not isinstance(count,(int,np.integer)) or isinstance(count,bool) or count<0
                or not isinstance(cache_frames,int) or cache_frames<2):
            raise ValueError('Invalid reader duration/cache limit')
        expected=(count+HOP-1)//HOP+1 if count else 0
        values=np.asarray(matrix.values,dtype=np.float64);gains=np.asarray(matrix.gains,dtype=np.float64)
        if (values.shape!=(expected,decoder.config.history-1) or gains.shape!=(expected,)
                or not np.isfinite(values).all() or not np.isfinite(gains).all() or np.any(gains<0)):
            raise ValueError('Invalid encoded source matrix')
        values=values.copy();gains=gains.copy()
        values.setflags(write=False);gains.setflags(write=False)
        self.matrix=ControlMatrix(values,gains,int(count));self.decoder=decoder
        self.limit=cache_frames;self.cache=OrderedDict();self.decoded_frames=0
        self.maximum_cache_frames=0;self.maximum_decoded_index=-1

    def _frame(self,index):
        if index in self.cache:
            self.cache.move_to_end(index);return self.cache[index]
        frame=self.decoder.frame(self.matrix.values[index],self.matrix.gains[index])*WINDOW
        if len(self.cache)>=self.limit:self.cache.popitem(last=False)
        self.cache[index]=frame;self.decoded_frames+=1
        self.maximum_cache_frames=max(self.maximum_cache_frames,len(self.cache))
        self.maximum_decoded_index=max(self.maximum_decoded_index,index)
        return frame

    def _at(self,indices):
        output=np.zeros(len(indices));valid=(indices>=0)&(indices<self.matrix.sample_count)
        for block in np.unique(indices[valid]//HOP):
            choose=valid&(indices//HOP==block);offset=indices[choose]%HOP
            left=self._frame(int(block));right=self._frame(int(block)+1)
            weights=WINDOW[HOP+offset]**2+WINDOW[offset]**2
            output[choose]=(left[HOP+offset]+right[offset])/weights
        return output

    def sample(self,positions):
        positions=np.asarray(positions,dtype=np.float64)
        if positions.ndim!=1 or not np.isfinite(positions).all():raise ValueError('Invalid source sample positions')
        integer=np.floor(positions).astype(np.int64);fraction=positions-integer
        output=self._at(integer)
        fractional=fraction!=0
        if np.any(fractional):
            right=self._at(integer[fractional]+1)
            output[fractional]=output[fractional]*(1-fraction[fractional])+right*fraction[fractional]
        return output

    def clear(self):self.cache.clear()


@dataclass(frozen=True)
class Grain:
    center: float
    source: float
    left: float
    right: float
    start: int
    stop: int


@dataclass(frozen=True)
class VoicedRegion:
    first: float
    last: float
    left: float
    right: float
    start: int
    stop: int


@dataclass(frozen=True)
class PerformancePacket:
    start: int
    samples: np.ndarray


def _plan(pulses,controls,count):
    grains=[];regions=[]
    for source in np.split(pulses,np.flatnonzero(np.diff(pulses)>RATE*.020)+1):
        if len(source)<3:continue
        periods=np.diff(source)
        if np.min(periods)<RATE/1000:raise ValueError('Unreasonably close source pulses')
        anchors=source*controls.duration_scale;seconds=anchors/RATE
        ratio=controls.pitch_ratio(seconds)
        if len(controls.note_times):
            wanted=np.interp(seconds,controls.note_times,controls.note_hz)*ratio
            phase=np.r_[0.,np.cumsum((wanted[:-1]+wanted[1:])*.5*np.diff(anchors)/RATE)]
        else:phase=np.r_[0.,np.cumsum((ratio[:-1]+ratio[1:])*.5*controls.duration_scale)]
        target=np.interp(np.arange(int(np.floor(phase[-1]))+1),phase,anchors)
        if len(target)<2:continue
        source_time=target/controls.duration_scale
        right=np.clip(np.searchsorted(source,source_time),1,len(source)-1);left=right-1
        chosen=np.where(abs(source_time-source[left])<=abs(source[right]-source_time),left,right)
        for j,(center,index) in enumerate(zip(target,chosen)):
            a=min(periods[max(0,index-1)],target[j]-target[j-1] if j else target[1]-target[0])
            b=min(periods[min(len(periods)-1,index)],target[j+1]-target[j] if j+1<len(target) else target[-1]-target[-2])
            start=max(0,int(np.ceil(center-a)));stop=min(count,int(np.floor(center+b))+1)
            if stop>start:grains.append(Grain(float(center),float(source[index]),float(a),float(b),start,stop))
        a=min(periods[0],target[1]-target[0]);b=min(periods[-1],target[-1]-target[-2])
        start=max(0,int(np.ceil(target[0]-a)));stop=min(count,int(np.floor(target[-1]+b))+1)
        if stop>start:regions.append(VoicedRegion(float(target[0]),float(target[-1]),float(a),float(b),start,stop))
    return grains,regions


class NeuralPerformanceStream:
    """Read at most 10 ms per call; cancellation cannot issue another packet.

    Construction validates/snapshots the complete precompiled control score and
    builds event metadata, but performs no neural waveform decoding. This does
    not yet accept a growing live score or live changes to the authored curve.
    """
    def __init__(self,performance,decoder,controls=PerformanceControls(),cache_frames=16):
        controls.validate()
        self.controls=PerformanceControls(float(controls.pitch_semitones),float(controls.duration_scale),
            float(controls.vibrato_cents),float(controls.vibrato_hz),tuple(controls.note_times),tuple(controls.note_hz))
        pulses=validate_pulses(performance.pulses,performance.matrix.sample_count).copy()
        self.reader=NeuralWindowReader(performance.matrix,decoder,cache_frames)
        self.count=round(performance.matrix.sample_count*self.controls.duration_scale)
        self.grains,self.regions=_plan(pulses,self.controls,self.count)
        self.grain_order=sorted(range(len(self.grains)),key=lambda i:self.grains[i].start)
        self.region_order=sorted(range(len(self.regions)),key=lambda i:self.regions[i].start)
        self.next_grain=0;self.next_region=0;self.active_grains=[];self.active_regions=[]
        self.position=0;self.closed=False;self.cancelled=False;self.maximum_active_grains=0

    def _background(self,start,stop):
        output=np.zeros(stop-start);weights=np.zeros(stop-start)
        first=max(0,((start-HOP+HOP-1)//HOP)*HOP)
        last=min(((self.count+HOP-1)//HOP)*HOP,((stop-1+HOP)//HOP)*HOP)
        for center in range(first,last+1,HOP):
            indices=np.arange(max(start,center-HOP),min(stop,center+HOP+1))
            offsets=indices-center;window=.5+.5*np.cos(np.pi*offsets/HOP)
            source=round(center/self.controls.duration_scale)
            output[indices-start]+=self.reader.sample(source+offsets)*window
            weights[indices-start]+=window
        return np.divide(output,weights,out=np.zeros(stop-start),where=weights>1e-12)

    def read(self,samples=HOP):
        if not isinstance(samples,(int,np.integer)) or isinstance(samples,bool) or not 1<=samples<=HOP:
            raise ValueError('Read between 1 and 240 output samples')
        if self.closed:return None
        if self.position>=self.count:
            self.closed=True;self.reader.clear();return None
        start=self.position;stop=min(start+samples,self.count)
        while self.next_grain<len(self.grain_order) and self.grains[self.grain_order[self.next_grain]].start<stop:
            self.active_grains.append(self.grain_order[self.next_grain]);self.next_grain+=1
        self.active_grains=sorted(i for i in self.active_grains if self.grains[i].stop>start)
        while self.next_region<len(self.region_order) and self.regions[self.region_order[self.next_region]].start<stop:
            self.active_regions.append(self.region_order[self.next_region]);self.next_region+=1
        self.active_regions=sorted(i for i in self.active_regions if self.regions[i].stop>start)
        self.maximum_active_grains=max(self.maximum_active_grains,len(self.active_grains))
        background=self._background(start,stop)
        output=np.zeros(stop-start);weight=np.zeros(stop-start);mask=np.zeros(stop-start)
        for index in self.active_grains:
            grain=self.grains[index];indices=np.arange(max(start,grain.start),min(stop,grain.stop))
            offsets=indices-grain.center;width=np.where(offsets<0,grain.left,grain.right)
            window=.5+.5*np.cos(np.pi*offsets/width)
            output[indices-start]+=self.reader.sample(grain.source+offsets)*window
            weight[indices-start]+=window
        for index in self.active_regions:
            region=self.regions[index];indices=np.arange(max(start,region.start),min(stop,region.stop))
            onset=np.clip((indices-(region.first-region.left))/region.left,0.,1.)
            release=np.clip((region.last+region.right-indices)/region.right,0.,1.)
            blend=(.5-.5*np.cos(np.pi*onset))*(.5-.5*np.cos(np.pi*release))
            mask[indices-start]=np.maximum(mask[indices-start],blend)
        voiced=np.divide(output,weight,out=np.zeros(stop-start),where=weight>1e-12)
        result=mask*voiced+(1-mask)*background
        if not np.isfinite(result).all():raise ValueError('Nonfinite performance output')
        self.position=stop
        if stop==self.count:self.closed=True;self.reader.clear()
        return PerformancePacket(start,result)

    def cancel(self):
        self.cancelled=True;self.closed=True;self.reader.clear()
        self.active_grains.clear();self.active_regions.clear()

    def __iter__(self):return self

    def __next__(self):
        packet=self.read()
        if packet is None:raise StopIteration
        return packet
