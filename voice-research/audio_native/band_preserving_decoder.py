"""Preserve a frozen decoder's lower band while adding another's high detail.

This is a composition of two existing learned waveform decoders, not a new
phoneme/content model. The 448-coordinate encoder must contain the 384-coordinate
encoder exactly as a prefix. A complementary FIR adds only the high-pass
candidate-minus-base difference. No source waveform or encoder enters runtime.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import firwin,lfilter
from neural_waveform import ControlMatrix,HOP,RATE,load_model
from neural_output_stream import NeuralOutputStream
from compiled_neural_decoder import compile_decoder

ROOT=Path(__file__).resolve().parents[2]


class BandPreservingDecoder:
    def __init__(self,base,detail,cutoff_hz=8000.,taps=129):
        if (not np.isfinite(cutoff_hz) or not 1000<=cutoff_hz<=10000
                or not isinstance(taps,int) or isinstance(taps,bool) or not 17<=taps<=513 or taps%2!=1
                or base.config.history>detail.config.history):
            raise ValueError('Invalid complementary decoder configuration')
        self.base=base;self.detail=detail;self.base_dimensions=base.config.history-1
        self.dimensions=detail.config.history-1;self.cutoff_hz=float(cutoff_hz);self.delay=(taps-1)//2
        self.highpass=-firwin(taps,cutoff_hz,fs=RATE,window=('kaiser',8.))
        self.highpass[self.delay]+=1.
        self.highpass.setflags(write=False)

    def render(self,matrix):
        if matrix.values.ndim!=2 or matrix.values.shape[1]!=self.dimensions:
            raise ValueError('Expected the full detail-control matrix')
        base_matrix=ControlMatrix(matrix.values[:,:self.base_dimensions],matrix.gains,matrix.sample_count)
        base=self.base.render(base_matrix);detail=self.detail.render(matrix)
        if not len(base):return base
        difference=np.r_[detail-base,np.zeros(self.delay)]
        correction=lfilter(self.highpass,[1.],difference)[self.delay:]
        return base+correction


class BandPreservingOutputStream:
    """Incremental frames; additional FIR lookahead is 64 samples by default.

    The first overlap block emits 176 samples with a 129-tap filter, subsequent
    full blocks emit 240, and finish() emits the remaining 64. Consumers must
    accept variable packet sizes. Cancellation discards delayed output.
    """
    def __init__(self,decoder):
        self.decoder=decoder
        self.base=NeuralOutputStream(decoder.base);self.detail=NeuralOutputStream(decoder.detail)
        self.history=np.zeros(len(decoder.highpass)-1);self.pending=np.empty(0)
        self.skip=decoder.delay;self.closed=False;self.source_closed=False;self.cancelled=False
        self.maximum_pending_samples=0;self.samples=0

    def _filter(self,difference):
        if not len(difference):return np.empty(0)
        filtered,self.history=lfilter(self.decoder.highpass,[1.],difference,zi=self.history)
        skip=min(self.skip,len(filtered));self.skip-=skip;filtered=filtered[skip:]
        count=len(filtered)
        if count>len(self.pending):raise AssertionError('FIR output outran buffered base audio')
        result=self.pending[:count]+filtered;self.pending=self.pending[count:].copy()
        self.maximum_pending_samples=max(self.maximum_pending_samples,len(self.pending))
        self.samples+=len(result)
        return result

    def push(self,controls,gain=1.,emit_samples=HOP):
        if self.closed or self.source_closed:raise RuntimeError('Output stream is closed')
        controls=np.asarray(controls,dtype=float)
        if (controls.shape!=(self.decoder.dimensions,) or not np.isfinite(controls).all()
                or not np.isfinite(gain) or gain<0
                or not isinstance(emit_samples,(int,np.integer)) or isinstance(emit_samples,bool)
                or not 1<=emit_samples<=HOP or (not self.base.frames and emit_samples!=HOP)):
            raise ValueError('Invalid neural control frame')
        base=self.base.push(controls[:self.decoder.base_dimensions],gain,emit_samples)
        detail=self.detail.push(controls,gain,emit_samples)
        self.pending=np.r_[self.pending,base]
        output=self._filter(detail-base)
        if emit_samples<HOP:self.source_closed=True
        return output

    def finish(self):
        if self.closed:return np.empty(0)
        self.base.finish();self.detail.finish()
        result=self._filter(np.zeros(self.decoder.delay))
        if len(self.pending):raise AssertionError('End-of-stream left pending audio')
        self.history=np.empty(0);self.closed=True
        return result

    def cancel(self):
        self.base.cancel();self.detail.cancel();self.pending=np.empty(0);self.history=np.empty(0)
        self.closed=True;self.cancelled=True


def load_band_preserving(path,compiled=True):
    spec=json.loads(Path(path).read_text())
    loaded=[]
    for key in ('base','detail'):
        source=ROOT/spec[key]['file']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=spec[key]['sha256']:
            raise ValueError('Frozen branch checkpoint changed')
        loaded.append(load_model(source))
    base_encoder,base,_=loaded[0];detail_encoder,detail,_=loaded[1]
    count=len(base_encoder.basis)
    if not (np.array_equal(base_encoder.mean,detail_encoder.mean)
            and np.array_equal(base_encoder.basis,detail_encoder.basis[:count])
            and np.array_equal(base_encoder.scale,detail_encoder.scale[:count])):
        raise ValueError('Encoder coordinates are not exactly nested; prefix routing would corrupt audio')
    if compiled:base=compile_decoder(base);detail=compile_decoder(detail)
    decoder=BandPreservingDecoder(base,detail,cutoff_hz=spec['cutoffHz'],taps=spec['taps'])
    return detail_encoder,decoder,spec
