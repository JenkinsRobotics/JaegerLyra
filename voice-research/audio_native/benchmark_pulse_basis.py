"""Evaluate captured pulse-basis generation, including its explicit output delay."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from compact_vocal_instrument import VocalFrame,RATE,BLOCK
from pulse_basis_instrument import PulseBasisStream,load_models,RADIUS
from benchmark_compact_vocal_instrument import held_metrics
from benchmark_vocal_shapes import evaluate
from neural_vowel_instrument import VowelScore
from vocal_shape_control import ShapePath,load_shapes

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/pulse-basis/v1'
VARIANTS={'mean-only':(0.,True,0),'correlated-seed-0':(1.,True,0),
    'correlated-seed-1':(1.,True,1),'iid-seed-0':(1.,False,0)}


def controlled_render(models,score,path,amount=1.,correlated=True,seed=0,stream_factory=PulseBasisStream):
    start=time.perf_counter()
    stream=stream_factory(models,amount=amount,correlated=correlated,seed=seed)
    initialization=time.perf_counter()-start
    count=round(score.seconds*RATE);pieces=[];timings=[]
    for start in range(0,count,BLOCK):
        begin=time.perf_counter();at=(start+BLOCK-1)/RATE
        pieces.append(stream.push(VocalFrame(float(score.frequency(at)),tuple(path.at(at)))))
        timings.append(time.perf_counter()-begin)
    begin=time.perf_counter();pieces.append(stream.finish());tail_time=time.perf_counter()-begin
    # Physical packets contain the causal 20 ms shift. Align only for offline
    # scoring; a real consumer must preserve that delay. No lookahead controls.
    begin=time.perf_counter()
    audio=np.concatenate(pieces)[RADIUS:RADIUS+count]
    audio=audio*score.envelope(np.arange(count),count)
    consumer_time=time.perf_counter()-begin
    try:stream.push(VocalFrame(220.,tuple(path.at(0.))))
    except RuntimeError:closed=True
    else:closed=False
    return audio,{'outputDelaySamples':RADIUS,'outputDelayMs':1000*RADIUS/RATE,
        'firstPhysicalPacketComputeMs':1000*(initialization+timings[0]),
        'packetP95Ms':float(1000*np.percentile(timings,95)),
        'maximumPacketMs':float(1000*max(timings)),'tailComputeMs':1000*tail_time,
        'alignmentAndEnvelopeMs':1000*consumer_time,
        'computeRealTimeFactor':(initialization+sum(timings)+tail_time+consumer_time)/score.seconds,
        'packets':len(timings),'generatedPulses':stream.generated_pulses,
        'closedPreventsOutput':closed,
        'closedClearsHistory':not any(np.any(a) for a in (stream.pending,stream.weights,stream.state)),
        'timingScope':'Resident bank; initialization, immediate 5 ms controls and pulse synthesis. Fixed 20 ms waveform shift is additional to computation, transport and device latency. Tail and offline alignment/envelope measured separately; excludes compilation and loading.'}


def main(out=OUT,stream_factory=PulseBasisStream,extra_report=None,variants=VARIANTS):
    fit=json.loads((OUT/'fit-report.json').read_text())
    old=json.loads((ROOT/'evidence/compact-vocal/v1/report.json').read_text())
    shapes=json.loads((ROOT/'evidence/vocal-shapes/v1/report.json').read_text())
    if not all(r['complete'] for r in (fit,old,shapes)):raise ValueError('Incomplete input study')
    cases=[r for r in old['results'] if r['method']=='compact-harmonic-noise' and r['kind']!='breath']
    assert len(cases)==39
    report={'scope':'Immediate pitch and vowel controls driving captured waveform templates with generated pulse-state variation',
        'fitReport':'evidence/pulse-basis/v1/fit-report.json','variants':variants,
        'expectedCases':len(cases)*len(variants),'results':[],'complete':False,
        'excludedCases':[{'name':r['name'],'reason':'Additional breath control explicitly unsupported and rejected'}
            for r in old['results'] if r['method']=='compact-harmonic-noise' and r['kind']=='breath'],
        'limits':fit['limits']+['Different training recordings from the short native-spectrum reference; errors include source variation.',
            'Two stochastic seeds are a limited robustness check, not a distributional quality guarantee.',
            'Pitch trackers and spectral/formant diagnostics do not establish naturalness or voice identity.']}
    if extra_report:report.update(extra_report)
    out.mkdir(parents=True,exist_ok=True)
    def persist():(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    hashes={}
    for speaker in ('af_heart','am_michael','af_bella'):
        source=next(p for p in old['profiles'] if p['speaker']==speaker)
        shape=next(p for p in shapes['profiles'] if p['speaker']==speaker)
        profiles,_=load_shapes(ROOT/shape['file'])
        bank=next(b for b in fit['banks'] if b['speaker']==speaker)
        file=ROOT/bank['file'];hashes[bank['file']]=bank['sha256']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=bank['sha256']:raise ValueError('Changed pulse bank')
        models,_=load_models(file)
        for reference in [r for r in cases if r['speaker']==speaker]:
            score=VowelScore(**reference['score']);path=ShapePath(**reference['shapePath'])
            for name,(amount,correlated,seed) in variants.items():
                audio,timing=controlled_render(models,score,path,amount,correlated,seed,stream_factory=stream_factory)
                row={k:reference[k] for k in ('speaker','kind','mode','vowel','score','shapePath') if k in reference}
                row.update(method=name,name=reference['name']+'-'+name)
                file=out/(row['name']+'.wav');sf.write(file,audio,RATE,subtype='FLOAT')
                if reference['kind']=='transition':
                    metrics,tracks=evaluate(audio,score,path,profiles)
                    track=file.with_suffix('.npz');np.savez(track,**tracks);row['tracks']=str(track.relative_to(ROOT))
                else:
                    i=('ah','ee','oo').index(reference['vowel'])
                    metrics=held_metrics(audio,score,source['sourceFormantsHz'][i])
                row.update(status='ok',file=str(file.relative_to(ROOT)),streaming=timing,**metrics,
                    modelBank=bank['file'],modelBankSha256=bank['sha256'])
                report['results'].append(row);persist()
                print(row['name'],'Praat p95',row['praat']['p95ErrorCents'],'peak',row['rawPeak'],flush=True)
    for file,sha in hashes.items():
        if hashlib.sha256((ROOT/file).read_bytes()).hexdigest()!=sha:raise AssertionError('Pulse bank changed')
    report['banksUnchanged']=True;report['complete']=len(report['results'])==report['expectedCases'];persist()


if __name__=='__main__':main()
