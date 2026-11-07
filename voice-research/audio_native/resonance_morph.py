"""Continuous resonance movement over one neural-decoded captured excitation.

Offline envelope/formant analysis is external evaluation infrastructure. This
runtime uses only saved envelopes, resonance landmarks and encoded source
controls. It does not learn phonemes, infer speech or establish clean-room
provenance. Frequency correspondence is a source/filter design hypothesis,
not an implementation of the cited audio-flow research algorithm.
"""
from collections import deque
import numpy as np
from live_neural_grains import LiveNeuralGrainStream,DELAY
from compact_vocal_instrument import VocalFrame,BLOCK,RATE

FFT=2048
FREQUENCIES=np.fft.rfftfreq(FFT,1/RATE)


class ResonancePath:
    def __init__(self,log_power,landmarks):
        power=np.array(log_power,dtype=float,copy=True);formants=np.array(landmarks,dtype=float,copy=True)
        if (power.shape!=(2,len(FREQUENCIES)) or formants.shape!=(2,3)
                or not np.isfinite(power).all() or not np.isfinite(formants).all()
                or np.max(abs(power))>100 or np.min(formants)<120 or np.max(formants)>4500
                or np.any(np.diff(formants,axis=1)<60)):
            raise ValueError('Invalid captured resonance endpoints')
        # Remove endpoint loudness; overall amplitude is a separate control.
        power-=np.log(np.exp(power).sum(axis=1,keepdims=True))
        self.log_power=power;self.landmarks=formants
        self.knots=np.c_[np.zeros(2),formants,np.full(2,6000.),np.full(2,RATE/2)]
        for a in (self.log_power,self.landmarks,self.knots):a.setflags(write=False)

    def at(self,alpha):
        if not np.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('Articulation must be 0–1')
        current=(1-alpha)*self.knots[0]+alpha*self.knots[1]
        a=np.interp(FREQUENCIES,current,self.knots[0])
        b=np.interp(FREQUENCIES,current,self.knots[1])
        shape=(1-alpha)*np.interp(a,FREQUENCIES,self.log_power[0])+alpha*np.interp(b,FREQUENCIES,self.log_power[1])
        return shape-np.log(np.exp(shape).sum())


def minimum_phase_ratio(path,alpha,taps=512,maximum_gain_db=18.):
    return envelope_ratio(path.log_power[0],path.at(alpha),taps,maximum_gain_db)


def envelope_ratio(source,target,taps=512,maximum_gain_db=18.):
    # The ratio clamp precedes profile-energy normalization; it is not a final
    # output peak limit or a strict bound on the normalized frequency response.
    if not isinstance(taps,int) or not 16<=taps<=FFT//2:raise ValueError('Invalid filter length')
    if not np.isfinite(maximum_gain_db) or not 0<maximum_gain_db<=24:raise ValueError('Invalid filter gain bound')
    source=np.asarray(source,dtype=float);target=np.asarray(target,dtype=float)
    if (source.shape!=(len(FREQUENCIES),) or target.shape!=source.shape
            or not np.isfinite(source).all() or not np.isfinite(target).all()
            or np.max(abs(source))>100 or np.max(abs(target))>100):raise ValueError('Invalid envelope controls')
    source=source-np.log(np.exp(source).sum());target=target-np.log(np.exp(target).sum())
    limit=maximum_gain_db*np.log(10)/20
    log_gain=np.clip(.5*(target-source),-limit,limit)
    cepstrum=np.fft.irfft(log_gain,n=FFT)
    causal=np.zeros(FFT);causal[0]=cepstrum[0];causal[1:FFT//2]=2*cepstrum[1:FFT//2];causal[FFT//2]=cepstrum[FFT//2]
    impulse=np.fft.irfft(np.exp(np.fft.rfft(causal)),n=FFT)[:taps].copy()
    fade=taps//4;impulse[-fade:]*=.5+.5*np.cos(np.linspace(0,np.pi,fade))
    # Fixed profile energy normalization, not an output-dependent limiter.
    response=abs(np.fft.rfft(impulse,n=FFT))**2
    energy=float(np.dot(np.exp(source),response))
    if not np.isfinite(energy) or energy<1e-12:raise ValueError('Degenerate filter response')
    return impulse/np.sqrt(energy)


class MovingFilter:
    """Causal FIR with samplewise interpolation between block coefficients."""
    def __init__(self,path,taps=512,maximum_gain_db=18.):
        self.path=path;self.taps=taps;self.limit=maximum_gain_db
        self.previous=minimum_phase_ratio(path,0.,taps,maximum_gain_db)
        self.history=np.zeros(taps-1);self.alpha=0.;self.closed=False

    def push(self,samples,alpha):
        current=self.previous if alpha==self.alpha else minimum_phase_ratio(self.path,alpha,self.taps,self.limit)
        output=self._apply(samples,current);self.alpha=alpha
        return output

    def push_envelope(self,samples,log_power):
        """Full-envelope diagnostic control; not the compact articulation path."""
        current=envelope_ratio(self.path.log_power[0],log_power,self.taps,self.limit)
        output=self._apply(samples,current);self.alpha=None
        return output

    def _apply(self,samples,current):
        samples=np.asarray(samples,dtype=float)
        if self.closed:raise RuntimeError('Filter closed')
        if samples.ndim!=1 or not 0<len(samples)<=BLOCK or not np.isfinite(samples).all():raise ValueError('Invalid audio block')
        joined=np.r_[self.history,samples]
        old=np.convolve(joined,self.previous,mode='valid')
        if current is self.previous:output=old
        else:
            new=np.convolve(joined,current,mode='valid')
            p=(np.arange(len(samples))+1)/len(samples);output=old*(1-p)+new*p
        self.history[:]=joined[-len(self.history):];self.previous=current
        return output

    def cancel(self):
        self.history.fill(0.);self.previous.fill(0.);self.closed=True


class NeuralResonanceStream:
    def __init__(self,patch,decoder,path,taps=512,maximum_gain_db=18.):
        self.source=LiveNeuralGrainStream([patch],decoder)
        self.filter=MovingFilter(path,taps,maximum_gain_db)
        self.queue=deque([(0.,0.)]*(DELAY//BLOCK));self.gain=0.;self.closed=False

    def _process(self,samples,alpha,gain):
        output=self.filter.push(samples,alpha)
        ramp=self.gain+(gain-self.gain)*(np.arange(len(samples))+1)/len(samples)
        self.gain=gain
        return output*ramp

    def push(self,hz,alpha,gain=1.):
        if self.closed:raise RuntimeError('Voice stream closed')
        if not np.isfinite([hz,alpha,gain]).all() or not 0<=alpha<=1 or not 0<=gain<=4:raise ValueError('Invalid vocal controls')
        frame=VocalFrame(hz,(1.,));frame.validate(1)
        audio=self.source.push(frame)
        self.queue.append((float(alpha),float(gain)))
        delayed_alpha,delayed_gain=self.queue.popleft()
        return self._process(audio,delayed_alpha,delayed_gain)

    def finish(self):
        if self.closed:return np.empty(0)
        if not self.source.position:
            self.cancel();return np.empty(0)
        source=self.source.finish();parts=[]
        for i in range(0,len(source),BLOCK):
            alpha,gain=self.queue.popleft();parts.append(self._process(source[i:i+BLOCK],alpha,gain))
        remaining=self.filter.taps-1
        while remaining:
            count=min(BLOCK,remaining);parts.append(self._process(np.zeros(count),self.filter.alpha,self.gain));remaining-=count
        self.cancel()
        return np.concatenate(parts) if parts else np.empty(0)

    def cancel(self):
        self.source.cancel();self.filter.cancel();self.queue.clear();self.gain=0.;self.closed=True
