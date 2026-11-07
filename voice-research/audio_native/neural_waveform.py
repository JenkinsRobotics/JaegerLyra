"""Neural waveform-decoder capacity study with continuous acoustic controls.

This first control encoder is PCA plus continuous amplitude companding, fitted
to audio windows. Its coordinates are NOT labelled/disentangled phonemes,
articulators or speaker identities. The neural decoder consumes coordinates and
gain only; no original PCM, residual, text or discrete codebook enters decoding.
Reconstruction of encoded speech is distinct from generating controls for new
speech. This is a research codec/decoder scaffold, not a finished TTS model.
"""
from dataclasses import dataclass
import json
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from lab import Config,LocalPredictor

RATE=24000
SIZE=480
HOP=240
WINDOW=np.sqrt(.5-.5*np.cos(2*np.pi*(np.arange(SIZE)+.5)/SIZE))


def waveform_frames(samples):
    samples=np.asarray(samples,dtype=np.float64)
    if samples.ndim!=1 or not np.isfinite(samples).all():
        raise ValueError('Expected finite mono waveform')
    if not len(samples):return np.empty((0,SIZE)),np.empty(0)
    padded=np.pad(samples,(HOP,HOP+(-len(samples))%HOP))
    frames=np.lib.stride_tricks.sliding_window_view(padded,SIZE)[::HOP]*WINDOW
    rms=np.sqrt(np.mean(frames*frames,axis=1))
    gains=np.maximum(rms,.005);gains[rms<1e-12]=0.
    unit=np.divide(frames,gains[:,None],out=np.zeros_like(frames),where=gains[:,None]>0)
    return unit,gains


def overlap_add(windowed_frames,sample_count):
    frames=np.asarray(windowed_frames,dtype=np.float64)
    if (not isinstance(sample_count,(int,np.integer)) or sample_count<0 or frames.ndim!=2
            or frames.shape[1]!=SIZE or not np.isfinite(frames).all()):
        raise ValueError('Invalid generated frame matrix')
    if sample_count==0:
        if len(frames):raise ValueError('Unexpected frames for empty audio')
        return np.empty(0)
    if len(frames)!=(sample_count+HOP-1)//HOP+1:
        raise ValueError('Frame count does not match output duration')
    length=(len(frames)-1)*HOP+SIZE
    audio=np.zeros(length);weight=np.zeros(length)
    for i,frame in enumerate(frames):
        start=i*HOP
        audio[start:start+SIZE]+=frame*WINDOW
        weight[start:start+SIZE]+=WINDOW*WINDOW
    region=slice(HOP,HOP+sample_count)
    return audio[region]/weight[region]


@dataclass(frozen=True)
class ControlMatrix:
    values: np.ndarray
    gains: np.ndarray
    sample_count: int


class ControlEncoder:
    def __init__(self,mean,basis,scale):
        self.mean=np.asarray(mean,dtype=np.float64).copy()
        self.basis=np.asarray(basis,dtype=np.float64).copy()
        self.scale=np.asarray(scale,dtype=np.float64).copy()
        if (self.mean.shape!=(SIZE,) or self.basis.ndim!=2 or self.basis.shape[1]!=SIZE
                or self.scale.shape!=(len(self.basis),) or not len(self.basis)
                or np.any(self.scale<=0) or not all(np.isfinite(x).all() for x in (self.mean,self.basis,self.scale))):
            raise ValueError('Invalid continuous waveform encoder')

    @classmethod
    def fit(cls,unit_frames,dimensions=256):
        frames=np.asarray(unit_frames,dtype=np.float64)
        if (frames.ndim!=2 or frames.shape[1]!=SIZE or len(frames)<2
                or not np.isfinite(frames).all() or not isinstance(dimensions,int)
                or not 1<=dimensions<=SIZE):
            raise ValueError('Invalid waveform training matrix')
        mean=frames.mean(axis=0);centered=frames-mean
        values,basis=np.linalg.eigh(centered.T@centered/len(frames))
        order=np.argsort(values)[::-1][:dimensions]
        return cls(mean,basis[:,order].T,np.sqrt(np.maximum(values[order],1e-6)))

    def encode_frames(self,unit_frames):
        frames=np.asarray(unit_frames,dtype=np.float64)
        if frames.ndim!=2 or frames.shape[1]!=SIZE or not np.isfinite(frames).all():
            raise ValueError('Invalid source window matrix')
        # Smooth real-valued amplitude companding; no categorical/audio tokens.
        return np.arcsinh(((frames-self.mean)@self.basis.T)/self.scale)

    def encode(self,samples):
        frames,gains=waveform_frames(samples)
        return ControlMatrix(self.encode_frames(frames),gains,len(samples))

    def linear_frames(self,controls):
        controls=np.asarray(controls,dtype=np.float64)
        if controls.ndim!=2 or controls.shape[1]!=len(self.basis) or not np.isfinite(controls).all():
            raise ValueError('Invalid continuous control matrix')
        if np.any(abs(controls)>20):raise ValueError('Controls exceed this reference inverse range')
        return (np.sinh(controls)*self.scale)@self.basis+self.mean

    def reconstruct_linear(self,matrix):
        return overlap_add(self.linear_frames(matrix.values)*matrix.gains[:,None],matrix.sample_count)


class NeuralDecoder:
    """Two-layer tanh network generating each 480-sample waveform window.

    Holds only W/V and numeric configuration. Encoder audio, PCA basis, target
    waveform and reference recordings are deliberately absent from this object.
    """
    def __init__(self,config,W,V):
        if config.block!=SIZE or config.history<2:
            raise ValueError('Invalid neural waveform dimensions')
        self.config=config
        self.W=np.asarray(W,dtype=np.float64).copy();self.V=np.asarray(V,dtype=np.float64).copy()
        if (self.W.shape!=(config.hidden,config.history) or self.V.shape!=(SIZE,config.hidden)
                or not np.isfinite(self.W).all() or not np.isfinite(self.V).all()):
            raise ValueError('Invalid neural waveform weights')
        self.W.setflags(write=False);self.V.setflags(write=False)

    @classmethod
    def from_learner(cls,learner):return cls(learner.config,learner.W,learner.V)

    def frame(self,controls,gain=1.):
        controls=np.asarray(controls,dtype=np.float64)
        if (controls.shape!=(self.config.history-1,) or not np.isfinite(controls).all()
                or not np.isfinite(gain) or gain<0):raise ValueError('Invalid neural synthesis controls')
        x=np.r_[controls,1.]*self.config.input_gain
        result=self.V@np.tanh(self.W@x)/self.config.input_gain*gain
        if not np.isfinite(result).all():raise ValueError('Nonfinite generated waveform')
        return result

    def render(self,matrix):
        if matrix.gains.shape!=(len(matrix.values),):raise ValueError('Gain count differs from control count')
        frames=np.array([self.frame(c,g) for c,g in zip(matrix.values,matrix.gains)])
        if not len(frames):frames=np.empty((0,SIZE))
        return overlap_add(frames,matrix.sample_count)


def initialize_decoder(controls,unit_targets,hidden=768,seed=17,ridge=1e-3,input_gain=.7):
    """Nonlinear random-feature / least-squares initialization, no backprop.

    This initialization is a batch statistical fit, not local online learning.
    The returned LocalPredictor can subsequently learn from adjacent prediction
    errors via observe(). Evaluation must use forward predict(), never its
    target-conditioned hidden-state reconstruction.
    """
    controls=np.asarray(controls,dtype=np.float64);targets=np.asarray(unit_targets,dtype=np.float64)
    if (controls.ndim!=2 or targets.shape!=(len(controls),SIZE) or len(controls)<2
            or not np.isfinite(controls).all() or not np.isfinite(targets).all()
            or not np.isfinite(ridge) or ridge<=0
            or not np.isfinite(input_gain) or input_gain<=0):raise ValueError('Invalid neural decoder training data')
    config=Config(history=controls.shape[1]+1,block=SIZE,hidden=hidden,
                  input_gain=input_gain,learning_rate=.02,settling_steps=12,seed=seed)
    learner=LocalPredictor(config)
    x=np.c_[controls,np.ones(len(controls))]*config.input_gain
    hidden_values=np.tanh(x@learner.W.T)
    gram=hidden_values.T@hidden_values
    penalty=ridge*float(np.trace(gram))/hidden
    gram.flat[::hidden+1]+=max(penalty,1e-9)
    learner.V=cho_solve(cho_factor(gram,lower=True),hidden_values.T@(targets*config.input_gain)).T
    return learner,penalty


def save_model(path,encoder,decoder,metadata):
    np.savez(path,encoder_mean=encoder.mean,encoder_basis=encoder.basis,encoder_scale=encoder.scale,
        W=decoder.W,V=decoder.V,config=json.dumps(decoder.config.__dict__),metadata=json.dumps(metadata))


def load_model(path):
    with np.load(path,allow_pickle=False) as d:
        encoder=ControlEncoder(d['encoder_mean'],d['encoder_basis'],d['encoder_scale'])
        decoder=NeuralDecoder(Config(**json.loads(str(d['config']))),d['W'],d['V'])
        metadata=json.loads(str(d['metadata']))
    return encoder,decoder,metadata


def initialize_axis_decoder(controls,unit_targets,centers=(-3.,-1.5,0.,1.5,3.),slope=.8,ridge=1e-6,sample_weight=None):
    """Structured tanh units per coordinate, with a fitted waveform output layer.

    Coordinatewise features match the encoder's coordinatewise companding.
    This is an explicit inductive bias for reconstruction, not learned vocal
    physiology, a phonetic representation, or an original-audio bypass. The
    resulting inference object is the same W/V-only NeuralDecoder as before.
    """
    controls=np.asarray(controls,dtype=np.float64);targets=np.asarray(unit_targets,dtype=np.float64)
    centers=np.asarray(centers,dtype=np.float64)
    if (controls.ndim!=2 or len(controls)<2 or controls.shape[1]<1
            or targets.shape!=(len(controls),SIZE) or not np.isfinite(controls).all()
            or not np.isfinite(targets).all() or centers.ndim!=1 or not len(centers)
            or not np.isfinite(centers).all() or not np.isfinite(slope) or slope<=0
            or not np.isfinite(ridge) or ridge<=0):
        raise ValueError('Invalid axis decoder training data')
    dimensions=controls.shape[1];hidden=dimensions*len(centers)+1
    config=Config(history=dimensions+1,block=SIZE,hidden=hidden,input_gain=1.,learning_rate=.02)
    learner=LocalPredictor(config);learner.W[:]=0.
    rows=np.arange(hidden-1)
    learner.W[rows,np.repeat(np.arange(dimensions),len(centers))]=slope
    learner.W[rows,-1]=-np.tile(centers,dimensions)*slope
    learner.W[-1,-1]=1.
    values=np.tanh(np.c_[controls,np.ones(len(controls))]@learner.W.T)
    if sample_weight is not None:
        weight=np.asarray(sample_weight,dtype=np.float64)
        if weight.shape!=(len(controls),) or not np.isfinite(weight).all() or np.any(weight<=0):
            raise ValueError('Invalid positive decoder example weights')
        root=np.sqrt(weight)[:,None]
        values=values*root;targets=targets*root
    gram=values.T@values;penalty=ridge*float(np.trace(gram))/hidden
    gram.flat[::hidden+1]+=max(penalty,1e-9)
    learner.V=cho_solve(cho_factor(gram,lower=True),values.T@targets).T
    return learner,penalty
