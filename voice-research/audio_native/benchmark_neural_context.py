"""Frozen prefix/release ablation on the existing corrected-source panel."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import load_model,RATE,SIZE
from neural_performance import load_performance
from neural_articulation import ArticulationUnit
from neural_context_stream import NeuralContextStream
from compiled_neural_decoder import compile_decoder
from compact_vocal_instrument import VocalFrame,BLOCK
from live_neural_grains import DELAY
from pcm_reference_decoder import pcm_performance,PcmFrameDecoder
from neural_vowel_instrument import VowelScore

ROOT=Path(__file__).resolve().parents[2]
PRIOR=ROOT/'evidence/neural-syllables/teacher-transfer-v1'
OUT=ROOT/'evidence/neural-syllables/context-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):return json.loads((OUT/name).read_text())
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def protocol():
    prior=json.loads((PRIOR/'protocol.json').read_text())
    for name,digest in prior['runtimeCode'].items():
        if sha(Path(__file__).with_name(name))!=digest:raise ValueError('Changed baseline runtime')
    if sha(ROOT/prior['model'])!=prior['modelSha256']:raise ValueError('Changed decoder weights')
    config={'scope':'Preserve original decoded prefix, release or both around the unchanged retuned held vowel',
        'sourcePanel':'The same 18 corrected/gain-compensated synthetic sources; development ablation, not held-out validation.',
        'parentProtocolSha256':sha(PRIOR/'protocol.json'),'parentReportSha256':sha(PRIOR/'report.json'),
        'model':prior['model'],'modelSha256':prior['modelSha256'],
        'runtimeCode':prior['runtimeCode']|{'neural_context_stream.py':sha(Path(__file__).with_name('neural_context_stream.py'))},
        'policies':['prefix','release','both'],'fadeSeconds':.03,'newRenders':216,'reusedControlledRenders':72,
        'prefix':'Exact decoded source until 30 ms before the annotated hold; cosine blend to the retuned renderer at hold start.',
        'release':'Cosine blend from retuned release to exact decoded source during 30 ms after the immediate release command.',
        'pitchCheck':prior['pitchCheck'],'asrEquivalents':prior['asrEquivalents'],
        'analysis':'Unchanged common teacher interval, excluding 80 ms at both boundaries; original-context spans remain in the pitch check.',
        'transitionDiagnostic':'20 ms RMS windows every 5 ms within 100 ms either side of the hold/release join; minimum/median ratio in dB. Exploratory, no perceptual pass threshold.',
        'appDefaultChanged':False,'newNeuralTraining':False,'listeningAccepted':False,
        'limits':prior['limits']+['Preserved context retains captured source pitch; pitch commands do not control those intervals.',
            'Fixed fades can cause phase-dependent interference; exact context preservation does not establish inaudible transitions.',
            'All hypotheses use the same previously evaluated sources; fresh-source validation is still required.']}
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        if read('protocol.json')!=config:raise ValueError('Changed frozen context protocol')
    else:write('protocol.json',config)
    return config


def prepare():
    original=json.loads((PRIOR/'capture.json').read_text());rendered=json.loads((PRIOR/'report.json').read_text())
    if not original['complete'] or not rendered['complete']:raise ValueError('Incomplete parent study')
    if not (OUT/'capture.json').exists():
        capture={**original,'protocolSha256':sha(OUT/'protocol.json'),'failures':[]}
        for key in ('references','reconstructions','units'):capture[key]=[r for r in original[key] if r['variant']=='level']
        write('capture.json',capture)
    if not (OUT/'report.json').exists():
        baseline=[{**r,'contextPolicy':'retune-all','decoderKind':r['method'].removeprefix('live-'),'reused':True}
            for r in rendered['results'] if r['variant']=='level']
        write('report.json',{'protocolSha256':sha(OUT/'protocol.json'),'sources':original['sources'],
            'results':baseline,'captureFailures':[],'expectedEligibleRenders':288,'complete':False})


def joins(audio,hold,release):
    results={}
    for name,boundary in [('hold',hold),('release',release)]:
        centers=np.arange(boundary-.1*RATE,boundary+.1*RATE+1,.005*RATE).astype(int)
        values=np.array([np.sqrt(np.mean(audio[max(0,c-240):min(len(audio),c+240)]**2)) for c in centers])
        results[name]={'minimumRelativeRmsDb':float(20*np.log10(max(float(min(values)),1e-12)/max(float(np.median(values)),1e-12))),
            'maximumRms':float(max(values)),'medianRms':float(np.median(values))}
    return results


def render(config):
    prepare();capture=read('capture.json');report=read('report.json')
    _,decoder,_=load_model(ROOT/config['model']);decoder=compile_decoder(decoder);pcm_decoder=PcmFrameDecoder()
    done={r['name']:r for r in report['results']}
    for source in capture['units']:
        if sha(ROOT/source['artifact'])!=source['artifactSha256'] or sha(ROOT/source['sourceFile'])!=source['sourceSha256']:
            raise ValueError('Changed captured source')
        performance,metadata=load_performance(ROOT/source['artifact']);original,rate=sf.read(ROOT/source['sourceFile']);assert rate==RATE
        pcm=pcm_performance(original,performance.pulses)
        for kind,encoded,synth in [('neural',performance,decoder),('pcm',pcm,pcm_decoder)]:
            direct=synth.render(encoded.matrix)
            for mode in ('native','melody'):
                baseline=next(r for r in report['results'] if r['method']=='live-'+kind and r['mode']==mode
                    and (r['speaker'],r['word'])==(source['speaker'],source['word']))
                if sha(ROOT/baseline['file'])!=baseline['sha256']:raise ValueError('Changed reused baseline')
                earlier,_=sf.read(ROOT/baseline['file']);score=VowelScore(**baseline['score']);count=round(score.seconds*RATE)
                for policy in config['policies']:
                    method=policy+'-'+kind;name=source['name']+'-'+mode+'-'+method
                    if name in done:
                        if sha(ROOT/done[name]['file'])!=done[name]['sha256']:raise ValueError('Changed saved result')
                        continue
                    unit=ArticulationUnit(encoded,metadata['holdFirstPulse'],metadata['holdLastPulse'])
                    begin=time.perf_counter();stream=NeuralContextStream(unit,synth,metadata['releaseSample'],context=policy,fade_seconds=config['fadeSeconds'])
                    initialize=time.perf_counter()-begin;parts=[];times=[];release_at=None
                    for start in range(0,count,BLOCK):
                        begin=time.perf_counter()
                        if release_at is None and start>=metadata['releaseSample']:
                            stream.release();release_at=start
                        hz=float(score.frequency((start+BLOCK-1)/RATE))
                        parts.append(stream.push(VocalFrame(hz,(1.,),gain=score.gain)));times.append(time.perf_counter()-begin)
                    begin=time.perf_counter();parts.append(stream.finish());finish=time.perf_counter()-begin
                    audio=np.concatenate(parts)[DELAY:DELAY+count]
                    if release_at!=metadata['releaseSample'] or len(audio)!=count or not np.isfinite(audio).all():raise ValueError('Invalid live result')
                    file=OUT/(name+'.wav');sf.write(file,audio,RATE,subtype='FLOAT');saved,_=sf.read(file)
                    timeline=np.arange(count);hold=source['holdStartSample'];fade=config['fadeSeconds']*RATE
                    preserved=np.zeros(count,dtype=bool);unchanged=np.ones(count,dtype=bool)
                    if policy in ('prefix','both'):
                        preserved |= timeline<=hold-fade;unchanged &= timeline>=hold
                    if policy in ('release','both'):
                        preserved |= timeline>=release_at+fade;unchanged &= timeline<release_at
                    untouched=float(max(abs(saved[unchanged]-earlier[unchanged])))
                    context_error=float(max(abs(saved[preserved]-(direct*score.gain)[preserved])))
                    if untouched!=0 or context_error>5e-8:raise ValueError('Context/interior contract failed')
                    closed=stream.closed and not stream.reader.cache and not any(np.any(a) for a in (stream.pending,stream.weights,stream.background,stream.voicing,stream.gain))
                    if not closed:raise ValueError('Pending history remains')
                    row={k:baseline[k] for k in ('speaker','word','variant','mode','score','plan','sourceArtifact',
                        'independentHoldStartSeconds','releaseCommandSeconds','releaseSourceSeconds')}
                    row.update(name=name,method=method,contextPolicy=policy,decoderKind=kind,reused=False,
                        file=str(file.relative_to(ROOT)),sha256=sha(file),rawPeak=float(max(abs(saved))),finite=True,
                        overFullScaleSamples=int(sum(abs(saved)>=1.)),preservedContextMaximumError=context_error,
                        untouchedRegionMaximumError=untouched,joinEnvelope=joins(saved,hold,release_at),
                        baselineJoinEnvelope=joins(earlier,hold,release_at),
                        streaming={'firstPhysicalPacketComputeMs':1000*(initialize+times[0]),
                            'packetP95Ms':float(np.percentile(times,95)*1000),'maximumPacketMs':float(max(times)*1000),
                            'packetsOver5Ms':sum(t>.005 for t in times),'waveformDelayMs':20,'tailComputeMs':finish*1000,
                            'closedClearsHistory':closed,'maximumCachedWaveformSamples':stream.reader.maximum_cache_frames*SIZE,
                            'residentSourceControlScalars':encoded.matrix.values.size+encoded.matrix.gains.size,
                            'scope':'Resident immediate control computation; preparation/loading/transport/devices excluded; fixed waveform delay is additional.'})
                    report['results'].append(row);write('report.json',report)
                    print(name,'context error',round(context_error,10),flush=True)
    protocol();report['complete']=len(report['results'])==report['expectedEligibleRenders'];write('report.json',report)


def evaluate(config):
    capture=read('capture.json');rendered=read('report.json')
    if not rendered['complete']:raise ValueError('Incomplete renders')
    cases=capture['references']+capture['reconstructions']+rendered['results']
    if not (OUT/'evaluation.json').exists():
        old=json.loads((PRIOR/'evaluation.json').read_text());wanted={r['name']:r for r in cases};reused=[]
        for row in old['results']:
            if row['name'] in wanted:
                if row['sha256']!=wanted[row['name']]['sha256']:raise ValueError('Changed earlier evaluation')
                reused.append({**row,'parentEvaluationReused':True})
        write('evaluation.json',{'protocolSha256':sha(OUT/'protocol.json'),'sources':old['sources'],
            'results':reused,'complete':False,'expectedCases':len(cases),'captureFailures':[]})
    import transfer_kokoro_syllables as evaluator
    evaluator.OUT=OUT;evaluator.evaluate(config,{})
    # The shared metric helper labels its generic full-clip 50 ms guard, but
    # these callers supply an already narrower, explicitly recorded interval.
    report=read('evaluation.json')
    for row in report['results']:
        for tracker in ('praat','world'):
            row[tracker]['scope']='Supplied common teacher hold interval, excluding 80 ms at both boundaries; original-context spans included.'
    report['pitchScopeClarifiedWithoutChangingMetrics']=True
    write('evaluation.json',report)


def identity(config):
    old=json.loads((PRIOR/'identity-evaluation.json').read_text());cache=OUT/'speaker-embeddings';cache.mkdir(exist_ok=True)
    current={p.name:sha(p) for p in (ROOT/'models/speaker-eval').glob('*.ckpt')}
    if old['modelFiles']!=current:raise ValueError('Changed identity evaluator')
    for row in old['sources']+old['results']:
        src=PRIOR/'speaker-embeddings'/(row['sha256']+'.npy');dst=cache/src.name
        if src.exists() and not dst.exists():shutil.copyfile(src,dst)
    (cache/'model-files.json').write_text(json.dumps(current,indent=2)+'\n')
    import evaluate_neural_syllable_transfer as evaluator
    evaluator.OUT=OUT;evaluator.identity()


if __name__=='__main__':
    config=protocol();{'render':render,'evaluate':evaluate,'identity':identity}[sys.argv[1]](config)
