"""Frozen compact-profile study, preserving both prior and fresh control cases."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import parselmouth
import pyworld
import soundfile as sf
from neural_waveform import load_model
from neural_performance import load_performance
from neural_vowel_instrument import VowelScore
from vocal_shape_control import load_shapes, ShapePath, render_world_shapes
from compact_vocal_instrument import (compile_shape,save_bank,load_bank,CompactVocalStream,
    VocalFrame,BLOCK,RATE,CEPSTRA,NOISE_HZ)
from benchmark_neural_vowels import pitch_metrics,formants
from benchmark_vocal_shapes import evaluate

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/compact-vocal/v1'


def controlled_render(shapes,score,path,breath=0.,stream_factory=CompactVocalStream):
    start=time.perf_counter();stream=stream_factory(shapes,seed=0)
    initialization=time.perf_counter()-start
    count=round(score.seconds*RATE);pieces=[];timings=[]
    for start in range(0,count,BLOCK):
        begin=time.perf_counter()
        at=(start+BLOCK-1)/RATE
        audio=stream.push(VocalFrame(float(score.frequency(at)),tuple(path.at(at)),1.,breath))
        # Exact authored envelope at the consumer; included in packet timings.
        n=min(BLOCK,count-start)
        audio=audio[:n]*score.envelope(np.arange(start,start+n),count)
        timings.append(time.perf_counter()-begin);pieces.append(audio)
    stream.cancel()
    try:stream.push(VocalFrame(220.,tuple(path.at(0.))))
    except RuntimeError:cancelled=True
    else:cancelled=False
    return np.concatenate(pieces),{'firstPacketComputeMs':1000*(initialization+timings[0]),
        'packetP95Ms':float(1000*np.percentile(timings,95)),'maximumPacketMs':float(1000*max(timings)),
        'computeRealTimeFactor':(initialization+sum(timings))/score.seconds,
        'packets':len(timings),'cancelPreventsOutput':cancelled,
        'cancelClearsHistory':not np.any(stream.noise_history),
        'timingScope':'Resident compact bank; includes control interpolation, synthesis and envelope, excludes compilation, loading, device and transport; 5 ms packets'}


def held_metrics(audio,score,reference):
    s=parselmouth.Sound(audio,sampling_frequency=RATE)
    p=s.to_pitch_ac(time_step=.01,pitch_floor=50.,pitch_ceiling=900.)
    wf,wt=pyworld.harvest(np.ascontiguousarray(audio),RATE,f0_floor=50.,f0_ceil=900.,frame_period=10.)
    f=formants(audio)
    return {'praat':pitch_metrics(p.xs(),p.selected_array['frequency'],score),
        'world':pitch_metrics(wt,wf,score),'measuredFormantsHz':f,
        'formantDriftPercent':[100*abs(v-r)/r if v is not None and r is not None else None for v,r in zip(f,reference)],
        'rawPeak':float(abs(audio).max()),'overFullScaleSamples':int(sum(abs(audio)>=1.)),
        'rms':float(np.sqrt(np.mean(audio[int(.05*RATE):-int(.05*RATE)]**2))),
        'finite':bool(np.isfinite(audio).all())}


def main():
    transitions=json.loads((ROOT/'evidence/vocal-shapes/v1/report.json').read_text())
    vowels=json.loads((ROOT/'evidence/neural-vowels/v1/report.json').read_text())
    if not transitions['complete'] or not vowels['complete']:raise ValueError('Incomplete source studies')
    model=ROOT/vowels['model']
    if hashlib.sha256(model.read_bytes()).hexdigest()!=vowels['modelSha256']:raise ValueError('Changed source decoder')
    _,decoder,_=load_model(model)
    before=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest()
    OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Output-only compact spectral instrument with immediate pitch, shape, gain and breath controls',
        'method':'Project harmonic-plus-noise synthesizer; neural decoder and external WORLD used only for offline profile capture; not a new neural vocoder',
        'neuralModel':vowels['model'],'neuralModelSha256':vowels['modelSha256'],
        'profileParameters':{'cepstraPerVowel':CEPSTRA,'noiseBandsPerVowel':len(NOISE_HZ),'rmsPerVowel':1,
            'vowelsPerCharacter':3,'totalPerCharacter':3*(CEPSTRA+len(NOISE_HZ)+1)},
        'liveControls':{'valuesPerFrame':6,'framesPerSecond':RATE/BLOCK,'valuesPerSecond':6*RATE/BLOCK,
            'fields':['pitchHz','vowelWeight1','vowelWeight2','vowelWeight3','gain','breath'],
            'futureScoreRequired':False,'targetPcmRequired':False,'neuralRuntime':False},
        'expectedCases':69,'profiles':[],'results':[],'complete':False,
        'limits':['Nine captured synthetic vowels, not learned phonemes or general identity embeddings.',
            'Static envelopes omit source microvariation and expressive articulation.',
            'WORLD analysis is shared with the profile capture and external reference, so its spectral error is not independent quality evidence.',
            'Burg formants are uncertain diagnostics; no naturalness or identity acceptance follows.',
            'No new words, human-reference cloning or independent neural speech generation.']}
    def persist():
        (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def record(row,audio,score,reference=None,path=None,profiles=None,timing=None):
        file=OUT/(row['name']+'.wav');sf.write(file,audio,RATE,subtype='FLOAT')
        if row['kind']=='transition':
            metrics,tracks=evaluate(audio,score,path,profiles)
            track=file.with_suffix('.npz');np.savez(track,**tracks);row['tracks']=str(track.relative_to(ROOT))
        else:metrics=held_metrics(audio,score,reference)
        row.update(status='ok',file=str(file.relative_to(ROOT)),**metrics)
        if timing:row['streaming']=timing
        report['results'].append(row);persist()
        print(row['name'],'praat p95',row['praat']['p95ErrorCents'],'peak',row['rawPeak'],flush=True)
    for speaker in ('af_heart','am_michael','af_bella'):
        source=next(r for r in transitions['profiles'] if r['speaker']==speaker)
        profiles,meta=load_shapes(ROOT/source['file'])
        refs=[next(r for r in vowels['references'] if r['name']==speaker+'-'+v) for v in ('ah','ee','oo')]
        shapes=[];details=[]
        for profile,ref in zip(profiles,refs):
            patch,_=load_performance(ROOT/ref['artifact'])
            decoded=decoder.render(patch.matrix)
            rms=np.sqrt(np.mean(decoded[600:-600]**2))
            shape,detail=compile_shape(profile,rms);shapes.append(shape)
            details.append({**detail,'sourcePatch':ref['artifact'],'rms':float(rms)})
        bank=OUT/(speaker+'-bank.npz')
        save_bank(bank,shapes,{'speaker':speaker,'sourceBank':source['file'],
            'sourceBankSha256':hashlib.sha256((ROOT/source['file']).read_bytes()).hexdigest(),
            'neuralModelSha256':vowels['modelSha256'],'capture':details})
        shapes,_=load_bank(bank)
        report['profiles'].append({'speaker':speaker,'bank':str(bank.relative_to(ROOT)),
            'sha256':hashlib.sha256(bank.read_bytes()).hexdigest(),'capture':details,
            'sourceFormantsHz':source['sourceFormantsHz']})
        persist()
        for old in [r for r in transitions['results'] if r['speaker']==speaker and r['method']=='world-shapes']:
            score=VowelScore(**old['score']);path=ShapePath(**old['shapePath'])
            row={'name':speaker+'-'+old['mode']+'-compact','speaker':speaker,'mode':old['mode'],
                'kind':'transition','method':'compact-harmonic-noise','score':score.__dict__,'shapePath':path.__dict__}
            audio,timing=controlled_render(shapes,score,path)
            record(row,audio,score,path=path,profiles=profiles,timing=timing)
        for index,(vowel,ref) in enumerate(zip(('ah','ee','oo'),refs)):
            for mode,ratio in (('native',1.),('down-three',2**(-3/12)),('up-three',2**(3/12))):
                score=VowelScore(seconds=2.,hz=(ref['sourceMedianPitchHz']*ratio,))
                weights=tuple(1. if i==index else 0. for i in range(3));path=ShapePath((0.,),(weights,))
                for method in ('compact-harmonic-noise','world-shapes'):
                    row={'name':speaker+'-'+vowel+'-'+mode+'-'+method,'speaker':speaker,'vowel':vowel,
                        'mode':mode,'kind':'held','method':method,'score':score.__dict__,'shapePath':path.__dict__}
                    if method=='compact-harmonic-noise':audio,timing=controlled_render(shapes,score,path)
                    else:
                        start=time.perf_counter();audio=render_world_shapes(profiles,score,path)
                        timing={'wholeRenderSeconds':time.perf_counter()-start,'timingScope':'Offline external WORLD, not streaming'}
                    record(row,audio,score,reference=ref['sourceFormantsHz'],timing=timing)
        # Same first vowel, prescribed breath amount; not tuned to its output.
        ref=refs[0];score=VowelScore(seconds=2.,hz=(ref['sourceMedianPitchHz'],))
        path=ShapePath((0.,),((1.,0.,0.),))
        audio,timing=controlled_render(shapes,score,path,breath=.35)
        record({'name':speaker+'-ah-breath-compact','speaker':speaker,'vowel':'ah','mode':'breath-0.35',
            'kind':'breath','method':'compact-harmonic-noise','score':score.__dict__,'breath':.35},
            audio,score,reference=ref['sourceFormantsHz'],timing=timing)
    if before!=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest():raise AssertionError('Neural weights changed')
    report['neuralWeightsUnchanged']=True;report['complete']=len(report['results'])==report['expectedCases'];persist()


if __name__=='__main__':main()
