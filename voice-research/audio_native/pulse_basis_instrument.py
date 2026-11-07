"""Causal overlap/add from captured waveform means and principal pulse shapes.

These parameters explicitly contain learned waveform templates, unlike the
spectral-only instrument. They are not a phoneme model or a newly trained neural
vocoder. A stochastic state produces new template mixtures without replaying a
recorded pulse sequence. Fixed 20 ms output delay makes the centred pulse
support causal; no future performance controls are requested.
"""
from dataclasses import dataclass
import json
import numpy as np
from compact_vocal_instrument import RATE,BLOCK,VocalFrame

RADIUS=480
SIZE=2*RADIUS+1
OFFSETS=np.arange(-RADIUS,RADIUS+1,dtype=float)
MODES=8


def pulse_window(period):
    x=OFFSETS/period
    return np.where(abs(x)<1,.5+.5*np.cos(np.pi*np.clip(x,-1,1)),0.)


@dataclass(frozen=True)
class PulseModel:
    mean: np.ndarray
    basis: np.ndarray
    scale: np.ndarray
    transition: np.ndarray
    innovation: np.ndarray
    reference_period: float
    rms: float

    def validate(self):
        arrays=(self.mean,self.basis,self.scale,self.transition,self.innovation)
        shapes=((SIZE,),(MODES,SIZE),(MODES,),(MODES,MODES),(MODES,MODES))
        if (any(a.shape!=s for a,s in zip(arrays,shapes)) or not all(np.isfinite(a).all() for a in arrays)
                or np.any(self.scale<=0) or not np.isfinite((self.reference_period,self.rms)).all()
                or not 30<=self.reference_period<=400 or not 0<self.rms<=.4
                or np.linalg.norm(self.transition,2)>.980000001
                or not np.allclose(self.basis@self.basis.T,np.eye(MODES),atol=1e-8,rtol=0.)
                or not np.allclose(self.transition@self.transition.T+self.innovation@self.innovation.T,
                    np.eye(MODES),atol=1e-8,rtol=0.)):
            raise ValueError('Invalid captured pulse model')


def fit_pulses(tracks,reference_period,rms):
    tracks=[np.asarray(t,dtype=float) for t in tracks]
    if (not tracks or any(t.ndim!=2 or t.shape[1]!=SIZE or len(t)<2 or not np.isfinite(t).all() for t in tracks)
            or sum(len(t) for t in tracks)<24):raise ValueError('Insufficient pulse tracks')
    values=np.concatenate(tracks);mean=values.mean(axis=0)
    _,s,basis=np.linalg.svd(values-mean,full_matrices=False)
    basis=basis[:MODES];scale=np.maximum(s[:MODES]/np.sqrt(len(values)-1),1e-6)
    encoded=[(t-mean)@basis.T/scale for t in tracks]
    x=np.concatenate([t[:-1] for t in encoded]);y=np.concatenate([t[1:] for t in encoded])
    raw=np.linalg.solve(x.T@x+.1*np.eye(MODES),x.T@y).T
    u,a,v=np.linalg.svd(raw);transition=(u*np.minimum(a,.98))@v
    eigen,vec=np.linalg.eigh(np.eye(MODES)-transition@transition.T)
    innovation=vec@np.diag(np.sqrt(np.maximum(eigen,0.)))
    model=PulseModel(mean,basis,scale,transition,innovation,float(reference_period),float(rms))
    model.validate()
    for array in (mean,basis,scale,transition,innovation):array.setflags(write=False)
    return model,{'trainingPulses':len(values),'trainingTracks':len(tracks),'transitionPairs':len(x),
        'retainedVarianceFraction':float(np.sum(s[:MODES]**2)/max(np.sum(s**2),1e-30)),
        'rawTransitionSingularValues':a.tolist(),'boundedTransitionSingularValues':np.minimum(a,.98).tolist(),
        'waveformTemplateScalars':int(mean.size+basis.size),'stateScalars':int(scale.size+transition.size+innovation.size),
        'normalization':'Each captured windowed pulse has unit weighted RMS; generation renormalizes its template before applying the reference RMS and commanded gain'}


def reconstruction_metrics(model,tracks):
    values=np.concatenate(tracks);z=(values-model.mean)@model.basis.T
    rebuilt=model.mean+z@model.basis
    def snr(output):return float(10*np.log10(np.sum(values**2)/max(np.sum((values-output)**2),1e-30)))
    return {'pulses':len(values),'meanTemplateSnrDb':snr(model.mean),'projectedEightModeSnrDb':snr(rebuilt),
        'scope':'Normalized windowed pulse representation using coefficients encoded from held pulses; not autonomous generated-audio quality'}


def save_models(path,models,metadata):
    if not models:raise ValueError('Empty pulse bank')
    for model in models:model.validate()
    keys=('mean','basis','scale','transition','innovation','reference_period','rms')
    np.savez(path,**{key:np.array([getattr(m,key) for m in models]) for key in keys},metadata=json.dumps(metadata))


def load_models(path):
    keys=('mean','basis','scale','transition','innovation','reference_period','rms')
    with np.load(path,allow_pickle=False) as data:
        arrays=[data[k].copy() for k in keys];metadata=json.loads(str(data['metadata']))
    if arrays[0].ndim!=2 or not len(arrays[0]) or any(len(a)!=len(arrays[0]) for a in arrays):raise ValueError('Invalid pulse bank')
    result=[]
    for i in range(len(arrays[0])):
        model=PulseModel(*(a[i].copy() for a in arrays[:5]),float(arrays[5][i]),float(arrays[6][i]))
        model.validate()
        for a in (model.mean,model.basis,model.scale,model.transition,model.innovation):a.setflags(write=False)
        result.append(model)
    return result,metadata


class PulseBasisStream:
    """Immediate 5 ms controls; fixed 20 ms waveform delay and bounded buffers.

    finish() returns the final 20 ms of delayed output. cancel() discards it.
    Breath control is not implemented by this experimental renderer and is
    rejected explicitly. Waveform templates can contain captured breath detail.
    """
    def __init__(self,models,seed=0,amount=1.,correlated=True):
        if not models or not np.isfinite(amount) or not 0<=amount<=1:raise ValueError('Invalid pulse configuration')
        for model in models:model.validate()
        self.means=np.array([m.mean for m in models]);self.bases=np.array([m.basis*m.scale[:,None] for m in models])
        self.transitions=np.array([m.transition for m in models]);self.innovations=np.array([m.innovation for m in models])
        self.periods=np.array([m.reference_period for m in models]);self.rms=np.array([m.rms for m in models])
        self.windows=np.array([pulse_window(p) for p in self.periods])
        for a in (self.means,self.bases,self.transitions,self.innovations,self.periods,self.rms,self.windows):a.setflags(write=False)
        self.rng=np.random.default_rng(seed);self.state=self.rng.normal(size=(len(models),MODES))
        self.amount=float(amount);self.correlated=bool(correlated)
        self.pending=np.zeros(BLOCK+SIZE+1);self.weights=np.zeros_like(self.pending)
        self.phase=0.;self.previous=None;self.position=0;self.closed=False;self.generated_pulses=0

    def _templates(self):
        if self.amount:
            noise=self.rng.normal(size=self.state.shape)
            self.state=(np.einsum('vij,vj->vi',self.transitions,self.state)+np.einsum('vij,vj->vi',self.innovations,noise)
                if self.correlated else noise)
        return self.means+self.amount*np.einsum('vi,vij->vj',self.state,self.bases)

    def _pulse(self,where,hz,weights,gain):
        templates=self._templates()
        power=np.sum(templates**2,axis=1)/np.maximum(np.sum(self.windows**2,axis=1),1e-20)
        templates=templates/np.sqrt(np.maximum(power[:,None],1e-20))*self.rms[:,None]*gain
        target=np.array([pulse_window(min(p,RATE/hz)) for p in self.periods])
        ratio=np.divide(target,self.windows,out=np.zeros_like(target),where=self.windows>1e-12)
        waveform=np.asarray(weights)@(templates*ratio);window=np.asarray(weights)@target
        first=int(np.floor(where));indices=first+np.arange(SIZE+1)
        offsets=indices-where-RADIUS
        self.pending[indices]+=np.interp(offsets,OFFSETS,waveform,left=0.,right=0.)
        self.weights[indices]+=np.interp(offsets,OFFSETS,window,left=0.,right=0.)
        self.generated_pulses+=1

    def _take(self,count):
        output=np.divide(self.pending[:count],self.weights[:count],out=np.zeros(count),where=self.weights[:count]>1e-12)
        self.pending[:-count]=self.pending[count:];self.pending[-count:]=0.
        self.weights[:-count]=self.weights[count:];self.weights[-count:]=0.
        return output

    def push(self,frame):
        if self.closed:raise RuntimeError('Pulse instrument closed')
        frame.validate(len(self.means))
        if frame.breath!=0:raise ValueError('Additional breath control is not supported by the pulse-basis experiment')
        frame=VocalFrame(float(frame.hz),tuple(frame.weights),float(frame.gain),0.)
        previous=frame if self.previous is None else self.previous
        if self.previous is None:self._pulse(0.,frame.hz,frame.weights,frame.gain)
        progress=(np.arange(BLOCK)+1)/BLOCK
        hz=np.exp(np.log(previous.hz)+progress*np.log(frame.hz/previous.hz))
        for i,frequency in enumerate(hz):
            step=frequency/RATE
            if self.phase+step>=1.:
                where=i+(1-self.phase)/step
                mix=min((where+1)/BLOCK,1.)
                weights=np.asarray(previous.weights)*(1-mix)+np.asarray(frame.weights)*mix
                gain=previous.gain*(1-mix)+frame.gain*mix
                self._pulse(where,float(frequency),weights,gain)
            self.phase=(self.phase+step)%1.
        result=self._take(BLOCK)
        self.previous=frame;self.position+=BLOCK
        return result

    def finish(self):
        if self.closed:return np.empty(0)
        tail=self._take(RADIUS) if self.position else np.empty(0)
        self.cancel();return tail

    def cancel(self):
        self.closed=True;self.pending.fill(0.);self.weights.fill(0.);self.state.fill(0.)
        self.previous=None;self.phase=0.
