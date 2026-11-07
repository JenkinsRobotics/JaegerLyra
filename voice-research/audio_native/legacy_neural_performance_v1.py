"""Output performance controls over a frozen neural waveform representation.

Compilation uses offline Praat pulse analysis, without text or model fitting.
The renderer receives encoded controls and pulse timings, not original PCM.
It decodes the captured articulation and uses classical pulse-synchronous
overlap/add to change pitch and timing independently. This is performance
resynthesis, not new-word generation, short-reference cloning, or a streaming
controller. The underlying neural output stream remains a separate component.

Algorithm context: Moulines & Charpentier (1990), and Praat's overlap-add manual.
The renderer below is new project code; Praat is used only during compilation.
"""
from dataclasses import dataclass
import json
import numpy as np
from neural_waveform import ControlMatrix, RATE, HOP, SIZE


@dataclass(frozen=True)
class EncodedPerformance:
    matrix: ControlMatrix
    pulses: np.ndarray


@dataclass(frozen=True)
class PerformanceControls:
    pitch_semitones: float = 0.
    duration_scale: float = 1.
    vibrato_cents: float = 0.
    vibrato_hz: float = 5.
    # Optional continuous absolute-pitch targets in OUTPUT seconds and Hz.
    note_times: tuple = ()
    note_hz: tuple = ()

    def validate(self):
        values=(self.pitch_semitones,self.duration_scale,self.vibrato_cents,self.vibrato_hz)
        if (not np.isfinite(values).all() or not -12<=self.pitch_semitones<=12
                or not .5<=self.duration_scale<=2. or not 0<=self.vibrato_cents<=100
                or not 0<=self.vibrato_hz<=12):raise ValueError('Invalid performance range')
        t=np.asarray(self.note_times,dtype=float);f=np.asarray(self.note_hz,dtype=float)
        if (t.ndim!=1 or f.shape!=t.shape or not np.isfinite(t).all() or not np.isfinite(f).all()
                or (len(t) and (t[0]<0 or np.any(np.diff(t)<=0) or np.any(f<60) or np.any(f>800)))):
            raise ValueError('Invalid absolute pitch curve')

    def pitch_ratio(self,seconds):
        cents=self.vibrato_cents*np.sin(2*np.pi*self.vibrato_hz*np.asarray(seconds))
        return 2**((self.pitch_semitones+cents/100)/12)


def validate_pulses(pulses,sample_count):
    values=np.asarray(pulses,dtype=np.float64)
    if (values.ndim!=1 or not np.isfinite(values).all() or np.any(np.diff(values)<=0)
            or np.any(values<0) or np.any(values>=sample_count)):
        raise ValueError('Invalid pulse locations')
    return values


def compile_performance(audio,encoder):
    import parselmouth
    from parselmouth.praat import call
    audio=np.asarray(audio,dtype=np.float64)
    matrix=encoder.encode(audio)
    if len(audio)<RATE*.04:
        pulses=np.empty(0)
    else:
        sound=parselmouth.Sound(audio,sampling_frequency=RATE)
        points=call(sound,'To PointProcess (periodic, cc)',60,800)
        count=call(points,'Get number of points')
        pulses=np.array([call(points,'Get time from index',i+1)*RATE for i in range(count)])
        pulses=pulses[(pulses>=0)&(pulses<len(audio))]
    pulses=validate_pulses(pulses,len(audio)).copy();pulses.setflags(write=False)
    matrix.values.setflags(write=False);matrix.gains.setflags(write=False)
    return EncodedPerformance(matrix,pulses)


def save_performance(path,performance,metadata):
    np.savez(path,values=performance.matrix.values,gains=performance.matrix.gains,
        sample_count=performance.matrix.sample_count,pulses=performance.pulses,metadata=json.dumps(metadata))


def load_performance(path):
    with np.load(path,allow_pickle=False) as data:
        values=data['values'];gains=data['gains'];count=int(data['sample_count']);pulses=data['pulses']
        metadata=json.loads(str(data['metadata']))
    expected=(count+HOP-1)//HOP+1 if count else 0
    if (count<0 or values.ndim!=2 or len(values)!=expected or gains.shape!=(len(values),)
            or not np.isfinite(values).all() or not np.isfinite(gains).all() or np.any(gains<0)):
        raise ValueError('Invalid encoded performance')
    validate_pulses(pulses,count)
    for item in (values,gains,pulses):item.setflags(write=False)
    return EncodedPerformance(ControlMatrix(values,gains,count),pulses),metadata


def _sample(audio,positions):
    positions=np.asarray(positions)
    integer=np.floor(positions).astype(np.int64);fraction=positions-integer
    def at(indices):
        inside=(indices>=0)&(indices<len(audio))
        return np.where(inside,audio[np.clip(indices,0,len(audio)-1)],0.)
    return at(integer)*(1-fraction)+at(integer+1)*fraction


def _unvoiced_background(audio,scale,count):
    """Fixed-rate local grains preserve noise bandwidth instead of tape-speed scaling."""
    output=np.zeros(count);weight=np.zeros(count)
    offsets=np.arange(-HOP,HOP+1)
    window=.5+.5*np.cos(np.pi*offsets/HOP)
    for target in range(0,count+HOP,HOP):
        indices=target+offsets;valid=(indices>=0)&(indices<count)
        # Integer source starts avoid unnecessary fractional filtering of noise.
        source=round(target/scale)
        output[indices[valid]]+=_sample(audio,source+offsets[valid])*window[valid]
        weight[indices[valid]]+=window[valid]
    return np.divide(output,weight,out=np.zeros(count),where=weight>1e-12)


def pulse_resynthesis(audio,pulses,controls=PerformanceControls()):
    """DSP control layer, also exposed separately for analytic signal tests."""
    audio=np.asarray(audio,dtype=np.float64);controls.validate()
    if audio.ndim!=1 or not np.isfinite(audio).all():raise ValueError('Invalid decoded audio')
    pulses=validate_pulses(pulses,len(audio))
    count=round(len(audio)*controls.duration_scale)
    if not count:return np.empty(0),{'sourceGroups':0,'synthesisPulses':0,'samples':0}
    background=_unvoiced_background(audio,controls.duration_scale,count)
    output=np.zeros(count);weight=np.zeros(count);events=0;groups=0;ratios=[]
    # Gaps above 20 ms are unvoiced, following the public algorithm description.
    splits=np.flatnonzero(np.diff(pulses)>RATE*.020)+1
    for source in np.split(pulses,splits):
        if len(source)<3:continue
        groups+=1;periods=np.diff(source)
        if np.min(periods)<RATE/1000:raise ValueError('Unreasonably close source pulses')
        target_anchor=source*controls.duration_scale
        seconds=target_anchor/RATE
        ratio=controls.pitch_ratio(seconds)
        if len(controls.note_times):
            wanted=np.interp(seconds,controls.note_times,controls.note_hz)*ratio
            source_f0=np.r_[RATE/periods[0],2*RATE/(periods[:-1]+periods[1:]),RATE/periods[-1]]
            ratio=wanted/source_f0
            phase=np.r_[0.,np.cumsum((wanted[:-1]+wanted[1:])*.5*np.diff(target_anchor)/RATE)]
        else:
            # Exactly preserves source pulse timing at neutral controls.
            phase=np.r_[0.,np.cumsum((ratio[:-1]+ratio[1:])*.5*controls.duration_scale)]
        if np.min(ratio)<.4 or np.max(ratio)>2.5:raise ValueError('Pitch edit exceeds tested pulse-ratio range')
        ratios.extend(ratio.tolist())
        target=np.interp(np.arange(int(np.floor(phase[-1]))+1),phase,target_anchor)
        if len(target)<2:continue
        source_time=target/controls.duration_scale
        right=np.clip(np.searchsorted(source,source_time),1,len(source)-1);left=right-1
        chosen=np.where(abs(source_time-source[left])<=abs(source[right]-source_time),left,right)
        for j,(center,index) in enumerate(zip(target,chosen)):
            source_left=periods[max(0,index-1)];source_right=periods[min(len(periods)-1,index)]
            target_left=target[j]-target[j-1] if j else target[1]-target[0]
            target_right=target[j+1]-target[j] if j+1<len(target) else target[-1]-target[-2]
            half_left=min(source_left,target_left);half_right=min(source_right,target_right)
            indices=np.arange(max(0,int(np.ceil(center-half_left))),min(count,int(np.floor(center+half_right))+1))
            offsets=indices-center
            widths=np.where(offsets<0,half_left,half_right)
            window=.5+.5*np.cos(np.pi*offsets/widths)
            output[indices]+=_sample(audio,source[index]+offsets)*window
            weight[indices]+=window;events+=1
    voiced=np.divide(output,weight,out=np.zeros(count),where=weight>1e-12)
    mix=np.clip(weight,0,1)
    result=mix*voiced+(1-mix)*background
    if not np.isfinite(result).all():raise ValueError('Nonfinite performance output')
    return result,{'sourceGroups':groups,'synthesisPulses':events,'samples':count,
        'pitchRatioRange':[min(ratios),max(ratios)] if ratios else None}


def render_performance(performance,decoder,controls=PerformanceControls()):
    # This PCM is generated by the frozen neural model, not retained original audio.
    decoded=decoder.render(performance.matrix)
    return pulse_resynthesis(decoded,performance.pulses,controls)
