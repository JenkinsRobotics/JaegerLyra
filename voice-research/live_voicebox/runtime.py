"""Resident independent output engine; no source WAVs or Kokoro model loaded."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
import math
import secrets
import sys
import threading
import time
from collections import deque
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'voice-research/audio_native'))
from neural_waveform import load_model,RATE
from compiled_neural_decoder import compile_decoder
from neural_performance import load_performance,PerformanceControls
from neural_articulation import ArticulationUnit
from neural_syllable_stream import NeuralSyllableStream
from neural_performance_stream import NeuralWindowReader,NeuralPerformanceStream
from compact_vocal_instrument import VocalFrame,BLOCK
from voice_colour import ColourDecoder

MANIFEST=ROOT/'demo/bundle/manifest.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def number(value,low,high,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
        raise ValueError(name+' is outside its supported range')
    return float(value)


class Session:
    def __init__(self,mode,item,performance,metadata,decoder,hz=None,gain=.75,shape=None):
        self.mode=mode;self.item=item;self.id=secrets.token_urlsafe(24);self.lock=threading.RLock()
        self.created=time.monotonic();self.claimed=False;self.closed=False;self.completed=False;self.closed_at=None
        self.gain=number(gain,0,1,'Volume');self.hz=number(hz if hz is not None else item.get('baseHz',200.),60,800,'Pitch')
        self.sequence=-1;self.applied_sequence=-1;self.events=deque(maxlen=64);self.release_requested=False
        self.packet_times=deque(maxlen=2000);self.maximum_compute_ms=0.;self.packets_over_5ms=0
        self.generated_samples=0;self.position=0;self.tail=deque();self.max_cache=0
        if shape is None:shape={}
        if not isinstance(shape,dict) or set(shape)-{'resonance','brightness','pace','vibrato'}:raise ValueError('Unknown voice shape control')
        self.shape={name:number(shape.get(name,default),low,high,name) for name,default,low,high in (
            ('resonance',1.,.85,1.18),('brightness',0.,-4.,4.),('pace',1.,.75,1.35),('vibrato',0.,0.,60.))}
        if mode=='syllable' and self.shape['pace']!=1.:raise ValueError('Pace applies to complete words and phrases')
        self.coloured=self.shape['resonance']!=1. or self.shape['brightness']!=0.
        if self.coloured:decoder=ColourDecoder(decoder,self.shape['resonance'],self.shape['brightness'])
        self.pitch_semitones=0.;self.performance_shaped=False;self.peak_safety_samples=0
        self.natural_sample_count=performance.matrix.sample_count
        if mode=='syllable':
            unit=ArticulationUnit(performance,metadata['holdFirstPulse'],metadata['holdLastPulse'])
            self.engine=NeuralSyllableStream(unit,decoder,metadata['releaseSample'])
            self.sample_count=performance.matrix.sample_count;self.release_sample=metadata['releaseSample'];self.release_at=None
        elif mode in ('word','speech'):
            if mode=='word':
                self.pitch_semitones=12*math.log2(self.hz/item['baseHz'])
                # Slider rounding within one cent should preserve natural playback.
                if abs(self.pitch_semitones)<.01:self.pitch_semitones=0.
                if abs(self.pitch_semitones)>12.000001:raise ValueError('Word pitch supports one octave up or down')
                self.pitch_semitones=max(-12.,min(12.,self.pitch_semitones))
            self.performance_shaped=self.pitch_semitones!=0 or self.shape['pace']!=1. or self.shape['vibrato']!=0.
            controls=PerformanceControls(pitch_semitones=self.pitch_semitones,duration_scale=1/self.shape['pace'],vibrato_cents=self.shape['vibrato'])
            self.engine=(NeuralPerformanceStream(performance,decoder,controls)
                if self.performance_shaped else NeuralWindowReader(performance.matrix,decoder,16))
            self.sample_count=round(self.natural_sample_count/self.shape['pace']);self.previous_gain=self.gain;self.release_at=None
        else:raise ValueError('Unknown performance mode')

    def control(self,data):
        if not isinstance(data,dict) or set(data)-{'sequence','hz','gain','release'}:raise ValueError('Unknown control field')
        sequence=data.get('sequence')
        if isinstance(sequence,bool) or not isinstance(sequence,int) or not 0<=sequence<=1000000000:raise ValueError('Invalid control sequence')
        gain=number(data['gain'],0,1,'Volume') if 'gain' in data else None
        hz=number(data['hz'],60,800,'Pitch') if 'hz' in data else None
        if 'release' in data and not isinstance(data['release'],bool):raise ValueError('Release must be boolean')
        if self.mode=='speech' and (hz is not None or data.get('release')):raise ValueError('Prepared speech supports volume and Stop')
        if self.mode=='word' and (hz is not None or data.get('release')):raise ValueError('Whole words finish automatically; choose pitch before playback')
        with self.lock:
            if self.closed:raise ValueError('This performance has ended')
            if sequence<=self.sequence:return self.snapshot()
            self.sequence=sequence
            if gain is not None:self.gain=gain
            if hz is not None:self.hz=hz
            self.release_requested |= data.get('release',False)
            return self.snapshot()

    def _close(self,completed=False):
        if self.closed:return
        if self.mode=='syllable' or self.performance_shaped:self.engine.cancel()
        else:self.engine.clear()
        self.tail.clear();self.closed=True;self.completed=completed;self.closed_at=time.monotonic()

    def stop(self):
        with self.lock:self._close()

    def _next(self):
        if self.closed:return None
        if time.monotonic()-self.created>60:self._close();return None
        if self.applied_sequence!=self.sequence:
            self.applied_sequence=self.sequence
            self.events.append({'sequence':self.sequence,'appliedSample':self.position,'hz':self.hz,'gain':self.gain,
                'releaseRequested':self.release_requested})
        if self.mode in ('word','speech'):
            count=min(BLOCK,self.sample_count-self.position)
            if count<=0:self._close(True);return None
            samples=(self.engine.read(count).samples if self.performance_shaped else
                self.engine.sample(np.arange(self.position,self.position+count,dtype=float)))
            gains=self.previous_gain+(self.gain-self.previous_gain)*(np.arange(count)+1)/count
            output=samples*gains*self.item['streamGain'];self.previous_gain=self.gain;self.position+=count
            reader=self.engine.reader if self.performance_shaped else self.engine
            self.max_cache=max(self.max_cache,reader.maximum_cache_frames)
        else:
            if self.tail:
                output=self.tail.popleft()
                if not self.tail:self._close(True)
            elif self.release_at is not None and self.position>=self.release_at+self.sample_count-self.release_sample:
                tail=self.engine.finish();self.tail.extend(tail[i:i+BLOCK] for i in range(0,len(tail),BLOCK))
                if not self.tail:self._close(True);return None
                output=self.tail.popleft()
            else:
                if self.release_requested and self.release_at is None and self.position>=self.engine.hold_start:
                    self.engine.release();self.release_at=self.position
                vibrato=self.shape['vibrato']*math.sin(2*math.pi*5*self.position/RATE)
                frequency=max(60.,min(800.,self.hz*2**(vibrato/1200)))
                output=self.engine.push(VocalFrame(frequency,(1.,),gain=self.gain));self.position+=BLOCK
                self.max_cache=max(self.max_cache,self.engine.reader.maximum_cache_frames)
        if not np.isfinite(output).all():self._close();raise ValueError('Nonfinite waveform')
        if self.coloured:
            # Colour edits can alter crest factor. Leave ordinary levels exact;
            # smoothly bound only the rare peaks above the reserved headroom.
            high=abs(output)>.92;self.peak_safety_samples+=int(np.count_nonzero(high))
            output[high]=np.sign(output[high])*(.92+.06*np.tanh((abs(output[high])-.92)/.06))
        self.generated_samples+=len(output)
        return np.asarray(output,dtype='<f4')

    def next_chunk(self):
        parts=[]
        for _ in range(4):
            with self.lock:
                begin=time.perf_counter();part=self._next();elapsed=(time.perf_counter()-begin)*1000
                if part is None:break
                self.packet_times.append(elapsed);self.maximum_compute_ms=max(self.maximum_compute_ms,elapsed)
                self.packets_over_5ms+=int(elapsed>5)
            parts.append(part)
        return np.concatenate(parts).tobytes() if parts else b''

    def snapshot(self):
        with self.lock:
            return {'id':self.id,'mode':self.mode,'closed':self.closed,'completed':self.completed,'claimed':self.claimed,
                'generatedSamples':self.generated_samples,'sourcePosition':self.position,'acceptedSequence':self.sequence,
                'appliedSequence':self.applied_sequence,'events':list(self.events),'releaseRequested':self.release_requested,
                'releaseSample':self.release_at,'hz':self.hz,'gain':self.gain,
                'automaticFinish':self.mode in ('word','speech'),'pitchSemitones':self.pitch_semitones,
                'naturalDurationSeconds':self.natural_sample_count/RATE if self.mode in ('word','speech') else None,
                'durationSeconds':self.sample_count/RATE if self.mode in ('word','speech') else None,
                'shape':dict(self.shape),'peakSafetySamples':self.peak_safety_samples,
                'holding':self.mode=='syllable' and self.release_at is None and self.position>=self.engine.hold_start,
                'packetP95Ms':float(np.percentile(self.packet_times,95)) if self.packet_times else None,
                'maximumPacketMs':self.maximum_compute_ms,'packetsOver5Ms':self.packets_over_5ms,
                'maximumCachedFrames':self.max_cache,'waveformDelayMs':20 if self.mode=='syllable' else 0,
                'latencyScope':'Computation only; prepared controls, HTTP transport, browser buffering and devices are separate.'}


class Engine:
    def __init__(self,manifest_path=MANIFEST):
        self.manifest=json.loads(Path(manifest_path).read_text())
        if not self.manifest['complete'] or sha(ROOT/self.manifest['model'])!=self.manifest['modelSha256']:
            raise ValueError('Incomplete or changed engine bundle')
        _,decoder,_=load_model(ROOT/self.manifest['model']);self.decoder=compile_decoder(decoder)
        self.items={};self.performances={};self.sessions={};self.lock=threading.RLock()
        for mode,key in [('syllable','units'),('word','words'),('speech','phrases')]:
            for item in self.manifest[key]:
                path=ROOT/item['artifact']
                if sha(path)!=item['artifactSha256']:raise ValueError('Changed prepared performance')
                self.items[(mode,item['id'])]=item;self.performances[(mode,item['id'])]=load_performance(path)

    def cleanup(self):
        now=time.monotonic()
        with self.lock:
            for key,session in list(self.sessions.items()):
                if not session.claimed and now-session.created>10:session.stop()
                if now-session.created>62:session.stop()
                if session.closed and now-session.closed_at>30:del self.sessions[key]

    def create(self,data):
        if not isinstance(data,dict) or set(data)-{'mode','item','hz','gain','shape'}:raise ValueError('Unknown session field')
        if not isinstance(data.get('mode'),str) or not isinstance(data.get('item'),str):raise ValueError('Choose a prepared performance')
        self.cleanup();key=(data.get('mode'),data.get('item'))
        with self.lock:
            if key not in self.items:raise ValueError('Unknown prepared performance')
            if sum(not s.closed for s in self.sessions.values())>=2:raise ValueError('Stop an existing performance first')
            item=self.items[key];performance,metadata=self.performances[key]
            session=Session(key[0],item,performance,metadata,self.decoder,data.get('hz'),data.get('gain',.75),data.get('shape'))
            self.sessions[session.id]=session;return session

    def get(self,identifier):
        self.cleanup()
        with self.lock:
            if not isinstance(identifier,str) or identifier not in self.sessions:raise ValueError('Unknown performance session')
            return self.sessions[identifier]

    def catalog(self):
        self.cleanup()
        units=[{k:r[k] for k in ('id','voice','word','baseHz','holdStartSeconds')} for r in self.manifest['units']]
        phrases=[]
        for row in self.manifest['phrases']:
            phrases.append({k:row[k] for k in ('id','voice','text','seconds')}|
                {'choices':[{'choice':c['choice'],'label':c['label'],'url':'/sample/'+row['id']+'/'+c['choice'].lower()} for c in row['choices']]})
        return {'project':'Jaeger Lyra','demonstrates':'reconstruction-playback','newSpeechGeneration':False,
            'sampleRate':RATE,'controlPacketMs':5,'transportPacketMs':20,'prebufferMs':40,'units':units,'phrases':phrases,'defaultPlayback':'word',
            'scope':self.manifest['scope'],'activeSessions':sum(not s.closed for s in self.sessions.values())}
