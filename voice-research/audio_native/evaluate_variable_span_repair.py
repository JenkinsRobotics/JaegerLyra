"""Post-validation development follow-up on the four capture failures.

This is not another held-out validation: the earlier failures motivated the
endpoint-search repair. Original failed results remain in fresh-sustain-v1.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
import re
import sys
import urllib.request
from functools import partial
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import load_model,RATE
from compiled_neural_decoder import compile_decoder
from neural_performance import save_performance
from neural_articulation import ArticulationUnit,NeuralArticulationStream
from variable_span_capture import compile_variable_unit
from pcm_reference_decoder import pcm_performance,PcmFrameDecoder
from neural_vowel_instrument import VowelScore
from benchmark_neural_vowels import formants
from benchmark_neural_articulation import controlled_render,metrics

ROOT=Path(__file__).resolve().parents[2]
PRIOR=ROOT/'evidence/neural-articulation/fresh-sustain-v1'
OUT=ROOT/'evidence/neural-articulation/variable-span-v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def render():
    previous=json.loads((PRIOR/'capture.json').read_text())
    config=json.loads((PRIOR/'protocol.json').read_text())
    if not previous['complete'] or sha(ROOT/config['model'])!=config['modelSha256']:raise ValueError('Changed or incomplete inputs')
    encoder,decoder,_=load_model(ROOT/config['model']);decoder=compile_decoder(decoder);pcm_decoder=PcmFrameDecoder()
    OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Post-validation variable-endpoint repair evaluated on previously failed sources; development evidence, not fresh validation',
        'priorCapture':'evidence/neural-articulation/fresh-sustain-v1/capture.json',
        'priorCaptureSha256':sha(PRIOR/'capture.json'),'model':config['model'],'modelSha256':config['modelSha256'],
        'captureCodeSha256':sha(Path(__file__).with_name('variable_span_capture.py')),
        'expectedSources':len(previous['failures']),'units':[],'failures':[],'results':[],'complete':False,
        'limits':['Targets are the four failures that motivated this repair; no independent generalization claim.',
            'All four render methods use the same new variable-endpoint-selected source region.',
            'Original PCM reference retains recorded waveform data; neural path uses the frozen decoder.',
            'Pitch/word diagnostics do not establish naturalness, identity or Kokoro parity.']}
    for source in previous['failures']:
        if sha(ROOT/source['file'])!=source['sha256']:raise ValueError('Changed source')
        audio,rate=sf.read(ROOT/source['file']);assert rate==RATE
        rms=np.sqrt(np.convolve(audio**2,np.ones(120)/120,mode='same'));active=np.flatnonzero(rms>max(1e-5,.003*max(rms)))
        first=max(0,int(active[0])-240);last=min(len(audio),int(active[-1])+241);trimmed=audio[first:last]
        try:unit,meta=compile_variable_unit(trimmed,encoder)
        except ValueError as exc:
            report['failures'].append({**source,'repairFailure':str(exc)});save('report.json',report);continue
        artifact=OUT/(source['name']+'-unit.npz');save_performance(artifact,unit.performance,meta)
        decoded=decoder.render(unit.performance.matrix)
        decoded_file=OUT/(source['name']+'-decoded.wav');sf.write(decoded_file,decoded,RATE,subtype='DOUBLE')
        trimmed_file=OUT/(source['name']+'-trimmed.wav');sf.write(trimmed_file,trimmed,RATE,subtype='DOUBLE')
        region=trimmed[int(meta['holdStartSample']):int(meta['holdEndSample'])]
        captured={**source,**meta,'artifact':str(artifact.relative_to(ROOT)),'artifactSha256':sha(artifact),
            'trimmedSource':str(trimmed_file.relative_to(ROOT)),'trimmedSha256':sha(trimmed_file),
            'decodedSource':str(decoded_file.relative_to(ROOT)),'decodedSha256':sha(decoded_file),
            'sourceFormantsHz':formants(region),'trimStart':first,'trimEnd':last}
        report['units'].append(captured)
        pcm=pcm_performance(trimmed,unit.performance.pulses)
        pcm_unit=ArticulationUnit(pcm,unit.hold_first,unit.hold_last)
        base=meta['sourceMedianPitchHz']
        scores={'native':VowelScore(seconds=2.5,hz=(base,)),
            'melody':VowelScore(seconds=2.5,times=tuple(config['melodyTimes']),
                hz=tuple(base*2**(n/12) for n in config['melodySemitones'])),
            'short-native':VowelScore(seconds=len(trimmed)/RATE,hz=(base,))}
        for mode,score in scores.items():
            for reader,(selected,engine,reference) in {'neural':(unit,decoder,decoded),'pcm':(pcm_unit,pcm_decoder,trimmed)}.items():
                for sustain,hold in [('loop','loop'),('fixed','fixed-first')]:
                    method=reader+'-'+sustain;name=source['name']+'-'+mode+'-'+method
                    output,timing=controlled_render(selected,engine,score,
                        stream_factory=partial(NeuralArticulationStream,hold_mode=hold))
                    if reader=='pcm':timing['timingScope']='Lossless waveform-frame experimental control with the identical stream; original waveform retained. 20 ms waveform delay is additional; devices/capture excluded.'
                    evaluated,tracks=metrics(output,score,selected,captured,reference)
                    prefix=evaluated['strictUnvoicedPrefix'];prefix['rmsDifferenceFromReference']=prefix.pop('rmsDifferenceFromNeuralSource')
                    prefix['reference']='original PCM' if reader=='pcm' else 'neural reconstruction'
                    file=OUT/(name+'.wav');sf.write(file,output,RATE,subtype='DOUBLE')
                    track=OUT/(name+'.tracks.npz');np.savez(track,**tracks)
                    report['results'].append({'name':name,'speaker':source['speaker'],'unit':source['unit'],
                        'mode':mode,'method':method,'score':score.__dict__,'file':str(file.relative_to(ROOT)),
                        'sha256':sha(file),'streaming':timing,**evaluated})
                    save('report.json',report);print(name,flush=True)
    report['complete']=len(report['units'])+len(report['failures'])==report['expectedSources'] and len(report['results'])==12*len(report['units'])
    save('report.json',report)


def words():
    study=json.loads((OUT/'report.json').read_text())
    if not study['complete']:raise ValueError('Incomplete repair')
    cases=[]
    for source in study['units']:
        for mode,field,hash_field in [('source','trimmedSource','trimmedSha256'),('decoded','decodedSource','decodedSha256')]:
            cases.append({**source,'name':source['name']+'-'+mode,'mode':mode,'method':mode,
                'file':source[field],'sha256':source[hash_field]})
    cases+=study['results'];report={'scope':study['scope'],'expectedCases':len(cases),'results':[],'complete':False}
    if (OUT/'word-report.json').exists():report=json.loads((OUT/'word-report.json').read_text())
    done={r['name']:r for r in report['results']}
    for row in cases:
        file=ROOT/row['file']
        if sha(file)!=row['sha256']:raise ValueError('Changed source')
        if row['name'] in done:
            if done[row['name']]['sha256']!=row['sha256']:raise ValueError('Changed ASR source')
            continue
        request=urllib.request.Request('http://127.0.0.1:8791/api/transcribe',file.read_bytes(),{'Content-Type':'audio/wav'})
        with urllib.request.urlopen(request,timeout=60) as response:result=json.load(response)
        if 'text' not in result:raise ValueError('Missing local transcript')
        transcript=result['text'];normalized=' '.join(re.findall(r'[a-z]+',transcript.lower()))
        report['results'].append({k:row[k] for k in ('name','speaker','unit','mode','method','file','sha256')}|
            {'transcript':transcript,'normalized':normalized,'exactSpellingMatch':normalized==row['unit']})
        save('word-report.json',report);print(row['name'],repr(transcript),flush=True)
    report['complete']=len(report['results'])==len(cases);save('word-report.json',report)


if __name__=='__main__':{'render':render,'words':words}[sys.argv[1]]()
