"""Offline singing from encoded speech, a time map and an authored pitch curve.

The fixed Lyra decoder generates the source waveform. Pulse-synchronous DSP
then changes its articulation timing and pitch. This is a hybrid prepared-score
renderer, not a learned text-to-singing model or a live score controller.
"""
import numpy as np
from neural_waveform import RATE, HOP
from neural_performance import validate_pulses, _sample


def render_score(performance, decoder, source_s, output_s, pitch_time_s, pitch_hz,
                 vibrato_cents=18., vibrato_hz=5.1):
    source_s=np.asarray(source_s,dtype=float);output_s=np.asarray(output_s,dtype=float)
    pitch_time_s=np.asarray(pitch_time_s,dtype=float);pitch_hz=np.asarray(pitch_hz,dtype=float)
    for values in (source_s,output_s,pitch_time_s,pitch_hz):
        if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
            raise ValueError('Expected finite score vectors')
    if (len(source_s)<2 or source_s.shape!=output_s.shape or source_s[0]!=0 or output_s[0]!=0
            or np.any(np.diff(source_s)<=0) or np.any(np.diff(output_s)<=0)
            or abs(source_s[-1]-performance.matrix.sample_count/RATE)>1/RATE
            or output_s[-1]>60 or pitch_time_s.shape!=pitch_hz.shape
            or np.any(np.diff(pitch_time_s)<=0) or pitch_time_s[0]<0
            or np.any(pitch_hz<60) or np.any(pitch_hz>800)
            or not np.isfinite([vibrato_cents,vibrato_hz]).all()
            or not 0<=vibrato_cents<=60 or not 0<=vibrato_hz<=12):
        raise ValueError('Invalid timing or pitch score')
    # Only this decoder receives acoustic controls; the original recording is
    # neither opened nor passed to this function.
    audio=decoder.render(performance.matrix)
    pulses=validate_pulses(performance.pulses,len(audio))
    count=round(output_s[-1]*RATE)
    to_output=lambda samples: np.interp(np.asarray(samples)/RATE,source_s,output_s)*RATE
    to_source=lambda samples: np.interp(np.asarray(samples)/RATE,output_s,source_s)*RATE
    background=np.zeros(count);background_weight=np.zeros(count)
    offsets=np.arange(-HOP,HOP+1);window=.5+.5*np.cos(np.pi*offsets/HOP)
    for center in range(0,count+HOP,HOP):
        indices=center+offsets;valid=(indices>=0)&(indices<count)
        origin=round(float(to_source(center)))
        background[indices[valid]]+=_sample(audio,origin+offsets[valid])*window[valid]
        background_weight[indices[valid]]+=window[valid]
    background=np.divide(background,background_weight,out=np.zeros(count),where=background_weight>1e-12)
    output=np.zeros(count);weights=np.zeros(count);mask=np.zeros(count);events=0
    for source in np.split(pulses,np.flatnonzero(np.diff(pulses)>RATE*.020)+1):
        if len(source)<3:continue
        periods=np.diff(source)
        if min(periods)<RATE/1000:raise ValueError('Invalid source pulse spacing')
        anchors=to_output(source);seconds=anchors/RATE
        wanted=np.interp(seconds,pitch_time_s,pitch_hz)
        wanted*=2**(vibrato_cents*np.sin(2*np.pi*vibrato_hz*seconds)/1200)
        phase=np.r_[0.,np.cumsum((wanted[:-1]+wanted[1:])*.5*np.diff(anchors)/RATE)]
        targets=np.interp(np.arange(int(np.floor(phase[-1]))+1),phase,anchors)
        if len(targets)<2:continue
        original_times=to_source(targets)
        right=np.clip(np.searchsorted(source,original_times),1,len(source)-1);left=right-1
        chosen=np.where(abs(original_times-source[left])<=abs(source[right]-original_times),left,right)
        for j,(center,index) in enumerate(zip(targets,chosen)):
            a=min(periods[max(0,index-1)],targets[j]-targets[j-1] if j else targets[1]-targets[0])
            b=min(periods[min(len(periods)-1,index)],targets[j+1]-targets[j] if j+1<len(targets) else targets[-1]-targets[-2])
            indices=np.arange(max(0,int(np.ceil(center-a))),min(count,int(np.floor(center+b))+1))
            offsets=indices-center;width=np.where(offsets<0,a,b)
            window=.5+.5*np.cos(np.pi*offsets/width)
            output[indices]+=_sample(audio,source[index]+offsets)*window
            weights[indices]+=window;events+=1
        a=min(periods[0],targets[1]-targets[0]);b=min(periods[-1],targets[-1]-targets[-2])
        indices=np.arange(max(0,int(np.ceil(targets[0]-a))),min(count,int(np.floor(targets[-1]+b))+1))
        onset=np.clip((indices-(targets[0]-a))/a,0.,1.)
        release=np.clip((targets[-1]+b-indices)/b,0.,1.)
        mask[indices]=np.maximum(mask[indices],(.5-.5*np.cos(np.pi*onset))*(.5-.5*np.cos(np.pi*release)))
    voiced=np.divide(output,weights,out=np.zeros(count),where=weights>1e-12)
    result=mask*voiced+(1-mask)*background
    if not np.isfinite(result).all():raise ValueError('Nonfinite scored output')
    return result,{'samples':count,'synthesisPulses':events,'originalRecordingLoaded':False}
