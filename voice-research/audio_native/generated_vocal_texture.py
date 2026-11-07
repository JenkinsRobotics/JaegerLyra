"""A small autonomous statistical texture state for the compact vocal instrument.

Offline source analysis learns three principal cepstral directions and a stable
linear state transition. Runtime draws fresh innovations, without source audio,
recorded trajectories or future target controls. This is a classical stochastic
state model around DSP synthesis, not a trained neural speech engine.
"""
from dataclasses import dataclass
import json
import numpy as np
from compact_vocal_instrument import CompactVocalStream,envelope,GRID,HARMONICS

FEATURES=31
LATENTS=3


@dataclass(frozen=True)
class TextureModel:
    mean: np.ndarray
    basis: np.ndarray
    scale: np.ndarray
    transition: np.ndarray
    innovation: np.ndarray

    def validate(self):
        arrays=(self.mean,self.basis,self.scale,self.transition,self.innovation)
        shapes=((FEATURES,),(LATENTS,FEATURES),(LATENTS,),(LATENTS,LATENTS),(LATENTS,LATENTS))
        if (any(a.shape!=s for a,s in zip(arrays,shapes)) or not all(np.isfinite(a).all() for a in arrays)
                or np.any(self.scale<=0) or np.max(self.scale)>10
                or np.linalg.norm(self.transition,2)>.980000001
                or not np.allclose(self.basis@self.basis.T,np.eye(LATENTS),atol=1e-8,rtol=0.)
                or not np.allclose(self.transition@self.transition.T+self.innovation@self.innovation.T,
                    np.eye(LATENTS),atol=1e-8,rtol=0.)):
            raise ValueError('Invalid stable texture model')

    def encode(self,features):
        return (np.asarray(features)-self.mean)@self.basis.T/self.scale


def fit_model(tracks):
    """Training tracks only; no transitions are fitted across segment gaps."""
    tracks=[np.asarray(t,dtype=float) for t in tracks]
    if (not tracks or any(t.ndim!=2 or t.shape[1]!=FEATURES or len(t)<3 or not np.isfinite(t).all() for t in tracks)
            or sum(len(t) for t in tracks)<24):raise ValueError('Insufficient texture training tracks')
    values=np.concatenate(tracks);mean=values.mean(axis=0)
    _,singular,basis=np.linalg.svd(values-mean,full_matrices=False)
    basis=basis[:LATENTS];scale=np.maximum(singular[:LATENTS]/np.sqrt(len(values)-1),1e-6)
    z=[(t-mean)@basis.T/scale for t in tracks]
    x=np.concatenate([t[:-1] for t in z]);y=np.concatenate([t[1:] for t in z])
    raw=np.linalg.solve(x.T@x+.1*np.eye(LATENTS),x.T@y).T
    u,s,v=np.linalg.svd(raw);transition=(u*np.minimum(s,.98))@v
    # This covariance gives an exactly unit stationary latent covariance for
    # the bounded transition; it is not a fitted innovation-distribution claim.
    covariance=np.eye(LATENTS)-transition@transition.T
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    innovation=eigenvectors@np.diag(np.sqrt(np.maximum(eigenvalues,0.)))
    model=TextureModel(mean,basis,scale,transition,innovation);model.validate()
    for a in (mean,basis,scale,transition,innovation):a.setflags(write=False)
    return model,{'trainingSegments':len(tracks),'trainingFrames':len(values),'transitionPairs':len(x),
        'retainedVarianceFraction':float(np.sum(singular[:LATENTS]**2)/max(np.sum(singular**2),1e-30)),
        'rawTransitionSingularValues':s.tolist(),'boundedTransitionSingularValues':np.linalg.svd(transition,compute_uv=False).tolist(),
        'innovation':'Stationary covariance calibration Q=I-AA.T, not fitted residual noise',
        'ridge':.1,'maximumTransitionSingularValue':.98}


def forecast_metrics(model,tracks):
    """Observed-state forecasts on separate complete segments; no future input."""
    tracks=[np.asarray(t) for t in tracks]
    rows=[]
    for horizon in (1,5,10):
        errors={k:[] for k in ('mean','persistence','learned')}
        for track in tracks:
            if len(track)<=horizon:continue
            state=model.encode(track[:-horizon])
            # Compare in original cepstral coordinates, retaining PCA residual
            # error; projecting the target away would overstate prediction.
            predicted=state@np.linalg.matrix_power(model.transition,horizon).T
            candidates={'mean':np.broadcast_to(model.mean,track[horizon:].shape),
                'persistence':track[:-horizon],
                'learned':model.mean+(predicted*model.scale)@model.basis}
            for name,value in candidates.items():errors[name].extend(np.mean((value-track[horizon:])**2,axis=1).tolist())
        rows.append({'horizonFrames':horizon,'horizonMs':horizon*5,'eligibleFrames':len(errors['mean']),
            'rmse':{name:float(np.sqrt(np.mean(values))) if values else None for name,values in errors.items()}})
    return rows


def save_models(path,models,metadata):
    if not models:raise ValueError('Empty texture model bank')
    for m in models:m.validate()
    np.savez(path,**{k:np.array([getattr(m,k) for m in models]) for k in
        ('mean','basis','scale','transition','innovation')},metadata=json.dumps(metadata))


def load_models(path):
    keys=('mean','basis','scale','transition','innovation')
    with np.load(path,allow_pickle=False) as data:
        arrays=[data[k].copy() for k in keys];metadata=json.loads(str(data['metadata']))
    if arrays[0].ndim!=2 or not len(arrays[0]) or any(len(a)!=len(arrays[0]) for a in arrays):raise ValueError('Invalid texture bank')
    models=[]
    for i in range(len(arrays[0])):
        model=TextureModel(*(a[i].copy() for a in arrays));model.validate()
        for key in keys:getattr(model,key).setflags(write=False)
        models.append(model)
    return models,metadata


class GeneratedTextureStream(CompactVocalStream):
    """Generate a fresh bounded modulation trajectory from immediate controls.

    Models affect only relative harmonic strengths. The compact comparator's
    phase, noise, pitch and total harmonic power remain the reference. Runtime
    has no captured source sequence. The amplitude factor is limited to exp(±.5)
    before power normalization; clipped-grid counts are reported, not hidden.
    """
    def __init__(self,shapes,models,seed=0,amount=1.,correlated=True):
        if len(shapes)!=len(models) or not np.isfinite(amount) or not 0<=amount<=1:
            raise ValueError('Invalid texture configuration')
        for model in models:model.validate()
        super().__init__(shapes,seed=seed)
        self.basis=np.array([m.basis*m.scale[:,None] for m in models])
        self.transitions=np.array([m.transition for m in models])
        self.innovations=np.array([m.innovation for m in models])
        for a in (self.basis,self.transitions,self.innovations):a.setflags(write=False)
        self.texture_rng=np.random.default_rng(np.random.SeedSequence([int(seed),731]))
        self.state=self.texture_rng.normal(size=(len(models),LATENTS))
        self.amount=float(amount);self.correlated=bool(correlated)
        self.clipped_grid_values=0;self.generated_frames=0

    def _parameters(self,frame):
        base,impulse=super()._parameters(frame)
        if self.amount==0:return base,impulse
        noise=self.texture_rng.normal(size=self.state.shape)
        if self.correlated:
            self.state=np.einsum('vij,vj->vi',self.transitions,self.state)+np.einsum('vij,vj->vi',self.innovations,noise)
        else:self.state=noise
        per_shape=np.einsum('vi,vij->vj',self.state,self.basis)
        variation=np.asarray(frame.weights)@per_shape*self.amount
        log_amplitude,_=envelope(np.r_[0.,variation])
        self.clipped_grid_values+=int(np.sum(abs(log_amplitude)>.5));self.generated_frames+=1
        log_amplitude=np.clip(log_amplitude,-.5,.5)
        amplitude=abs(base)*np.exp(np.interp(HARMONICS*frame.hz,GRID,log_amplitude))
        amplitude*=np.sqrt(np.sum(abs(base)**2)/max(np.sum(amplitude**2),1e-30))
        return amplitude*np.exp(1j*np.angle(base)),impulse

    def cancel(self):
        super().cancel();self.state.fill(0.)
