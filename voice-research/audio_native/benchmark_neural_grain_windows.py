"""Isolate source-sized and wider held grains, retaining all failures."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import load_model,RATE,SIZE
from neural_performance import load_performance
from neural_articulation import ArticulationUnit
from neural_grain_window_stream import NeuralGrainWindowStream
from compiled_neural_decoder import compile_decoder
from compact_vocal_instrument import VocalFrame,BLOCK
from live_neural_grains import DELAY
from pcm_reference_decoder import pcm_performance,PcmFrameDecoder
from neural_vowel_instrument import VowelScore

ROOT=Path(__file__).resolve().parents[2]
PRIOR=ROOT/'evidence/neural-syllables/teacher-transfer-v1'
OUT=ROOT/'evidence/neural-syllables/grain-windows-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):return json.loads((OUT/name).read_text())
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def protocol():
    prior=json.loads((PRIOR/'protocol.json').read_text())
    for name,digest in prior['runtimeCode'].items():
        if sha(Path(__file__).with_name(name))!=digest:raise ValueError('Changed original runtime')
    if sha(ROOT/prior['model'])!=prior['modelSha256']:raise ValueError('Changed decoder')
    config={'scope':'Held-grain window ablation with unchanged pulses, source spans, controls, prefix and release grain rules',
        'sourcePanel':'The same 18 corrected/gain-compensated synthetic sources; development ablation, not fresh validation.',
        'parentProtocolSha256':sha(PRIOR/'protocol.json'),'parentReportSha256':sha(PRIOR/'report.json'),
        'model':prior['model'],'modelSha256':prior['modelSha256'],
        'runtimeCode':prior['runtimeCode']|{'neural_grain_window_stream.py':sha(Path(__file__).with_name('neural_grain_window_stream.py'))},
        'policies':['source','wide'],'newRenders':144,'reusedControlledRenders':72,
        'baseline':'Each half-window is the minimum of the adjacent source period and current target period.',
        'source':'Each held half-window is its adjacent source period, without target-period shortening.',
        'wide':'Each held half-window is 1.5 times its adjacent source period.',
        'radiusLimitSamples':DELAY-1,'radiusClipsRecorded':True,'waveformDelayMs':20,
        'unchangedContextGuardMs':20,'gainCompensation':False,
        'pitchCheck':prior['pitchCheck'],'asrEquivalents':prior['asrEquivalents'],
        'spectrum':'Native-pitch only: fixed 80-band centered log-mel shape from 80–8000 Hz, 1024-sample Hann windows / 120-sample hop / 2048 FFT; compare to the corrected teacher over independent hold+100 ms through release-100 ms. No perceptual threshold.',
        'spectrumHelperSha256':sha(Path(__file__).with_name('summarize_captured_harmonic_voice.py')),
        'appDefaultChanged':False,'newNeuralTraining':False,'listeningAccepted':False,
        'limits':prior['limits']+['The baseline rule matches a documented overlap-add implementation; this is an alternative design test, not an identified bug fix.',
            'Window changes can alter amplitude and phase as well as spectral shape; there is no hidden gain correction.',
            'Expanded grain support may affect up to 20 ms either side of the held span; outside that guard the saved waveform must be unchanged.',
            'The same development panel is reused; successful conditions require fresh-source validation.']}
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        if read('protocol.json')!=config:raise ValueError('Frozen grain-window protocol changed')
    else:write('protocol.json',config)
    return config


def prepare():
    original=json.loads((PRIOR/'capture.json').read_text());rendered=json.loads((PRIOR/'report.json').read_text())
    if not original['complete'] or not rendered['complete']:raise ValueError('Incomplete parent')
    if not (OUT/'capture.json').exists():
        capture={**original,'protocolSha256':sha(OUT/'protocol.json'),'failures':[]}
        for key in ('references','reconstructions','units'):capture[key]=[r for r in original[key] if r['variant']=='level']
        write('capture.json',capture)
    if not (OUT/'report.json').exists():
        baseline=[{**r,'windowPolicy':'minimum','decoderKind':r['method'].removeprefix('live-'),'reused':True}
            for r in rendered['results'] if r['variant']=='level']
        write('report.json',{'protocolSha256':sha(OUT/'protocol.json'),'sources':original['sources'],
            'results':baseline,'captureFailures':[],'expectedEligibleRenders':216,'complete':False})


def render(config):
    prepare();capture=read('capture.json');report=read('report.json');done={r['name']:r for r in report['results']}
    _,decoder,_=load_model(ROOT/config['model']);decoder=compile_decoder(decoder);pcm_decoder=PcmFrameDecoder()
    for source in capture['units']:
        if sha(ROOT/source['artifact'])!=source['artifactSha256'] or sha(ROOT/source['sourceFile'])!=source['sourceSha256']:
            raise ValueError('Changed captured source')
        performance,metadata=load_performance(ROOT/source['artifact']);original,rate=sf.read(ROOT/source['sourceFile']);assert rate==RATE
        pcm=pcm_performance(original,performance.pulses)
        for kind,encoded,synth in [('neural',performance,decoder),('pcm',pcm,pcm_decoder)]:
            for mode in ('native','melody'):
                baseline=next(r for r in report['results'] if r['method']=='live-'+kind and r['mode']==mode
                    and (r['speaker'],r['word'])==(source['speaker'],source['word']))
                if sha(ROOT/baseline['file'])!=baseline['sha256']:raise ValueError('Changed baseline')
                earlier,_=sf.read(ROOT/baseline['file']);score=VowelScore(**baseline['score']);count=round(score.seconds*RATE)
                for policy in config['policies']:
                    method=policy+'-'+kind;name=source['name']+'-'+mode+'-'+method
                    if name in done:
                        if sha(ROOT/done[name]['file'])!=done[name]['sha256']:raise ValueError('Changed saved render')
                        continue
                    unit=ArticulationUnit(encoded,metadata['holdFirstPulse'],metadata['holdLastPulse'])
                    begin=time.perf_counter();stream=NeuralGrainWindowStream(unit,synth,metadata['releaseSample'],window=policy)
                    initialize=time.perf_counter()-begin;parts=[];times=[];release_at=None
                    for start in range(0,count,BLOCK):
                        begin=time.perf_counter()
                        if release_at is None and start>=metadata['releaseSample']:
                            stream.release();release_at=start
                        hz=float(score.frequency((start+BLOCK-1)/RATE))
                        parts.append(stream.push(VocalFrame(hz,(1.,),gain=score.gain)));times.append(time.perf_counter()-begin)
                    begin=time.perf_counter();parts.append(stream.finish());finish=time.perf_counter()-begin
                    audio=np.concatenate(parts)[DELAY:DELAY+count]
                    if release_at!=metadata['releaseSample'] or len(audio)!=count or not np.isfinite(audio).all():raise ValueError('Invalid output')
                    file=OUT/(name+'.wav');sf.write(file,audio,RATE,subtype='FLOAT');saved,_=sf.read(file)
                    t=np.arange(count);hold=source['holdStartSample'];outside=(t<hold-DELAY)|(t>release_at+DELAY)
                    error=float(max(abs(saved[outside]-earlier[outside])))
                    if error!=0:raise ValueError('Waveform changed outside the frozen hold guard')
                    closed=stream.closed and not stream.reader.cache and not any(np.any(a) for a in (stream.pending,stream.weights,stream.background,stream.voicing,stream.gain))
                    if not closed:raise ValueError('Pending history remains')
                    row={k:baseline[k] for k in ('speaker','word','variant','mode','score','plan','sourceArtifact',
                        'independentHoldStartSeconds','releaseCommandSeconds','releaseSourceSeconds')}
                    row.update(name=name,method=method,windowPolicy=policy,decoderKind=kind,reused=False,
                        file=str(file.relative_to(ROOT)),sha256=sha(file),rawPeak=float(max(abs(saved))),finite=True,
                        overFullScaleSamples=int(sum(abs(saved)>=1.)),outsideHoldGuardMaximumError=error,
                        maximumGrainRadiusSamples=stream.maximum_grain_radius,radiusClippedGrains=stream.radius_clipped_grains,
                        streaming={'firstPhysicalPacketComputeMs':1000*(initialize+times[0]),'packetP95Ms':float(np.percentile(times,95)*1000),
                            'maximumPacketMs':float(max(times)*1000),'packetsOver5Ms':sum(t>.005 for t in times),
                            'waveformDelayMs':20,'tailComputeMs':finish*1000,'closedClearsHistory':closed,
                            'maximumCachedWaveformSamples':stream.reader.maximum_cache_frames*SIZE,
                            'residentSourceControlScalars':encoded.matrix.values.size+encoded.matrix.gains.size,
                            'scope':'Resident control computation; capture/loading/transport/devices excluded; waveform delay is additional.'})
                    report['results'].append(row);write('report.json',report);print(name,'radius',round(stream.maximum_grain_radius,2),flush=True)
    protocol();report['complete']=len(report['results'])==report['expectedEligibleRenders'];write('report.json',report)


def evaluate(config):
    # Reuse the existing evaluation contract and exact-audio-hash cache.
    import benchmark_neural_context as evaluator
    evaluator.OUT=OUT;evaluator.evaluate(config)
    report=read('evaluation.json')
    for row in report['results']:
        for tracker in ('praat','world'):
            row[tracker]['scope']='Common teacher hold interval excluding 80 ms at each boundary; held-window changes and adjacent transitions included.'
    write('evaluation.json',report)


def identity(config):
    import benchmark_neural_context as evaluator
    evaluator.OUT=OUT;evaluator.identity(config)


def spectrum(config):
    from summarize_captured_harmonic_voice import spectral_shape
    captured=read('capture.json');rendered=read('report.json')
    if not rendered['complete']:raise ValueError('Incomplete render')
    results=[]
    for source in captured['units']:
        reference=next(r for r in captured['references'] if r['mode']=='native' and
            (r['speaker'],r['word'])==(source['speaker'],source['word']))
        if sha(ROOT/reference['file'])!=reference['sha256']:raise ValueError('Changed spectrum reference')
        a,rate=sf.read(ROOT/reference['file']);assert rate==RATE
        start=round(source['holdStartSample']+.1*RATE);end=round(source['releaseSample']-.1*RATE)
        target,hz=spectral_shape(a[start:end])
        for row in rendered['results']:
            if row['mode']!='native' or (row['speaker'],row['word'])!=(source['speaker'],source['word']):continue
            if sha(ROOT/row['file'])!=row['sha256']:raise ValueError('Changed spectrum input')
            b,rb=sf.read(ROOT/row['file']);assert rb==RATE
            actual,_=spectral_shape(b[start:end]);diff=actual-target
            artifact=OUT/(row['name']+'-spectrum.npz');np.savez(artifact,hz=hz,referenceDb=target,outputDb=actual)
            results.append({'name':row['name'],'method':row['method'],'windowPolicy':row['windowPolicy'],'decoderKind':row['decoderKind'],
                'file':row['file'],'sha256':row['sha256'],'reference':reference['file'],'referenceSha256':reference['sha256'],
                'startSeconds':start/RATE,'endSeconds':end/RATE,'meanAbsoluteShapeDifferenceDb':float(np.mean(abs(diff))),
                'rmsShapeDifferenceDb':float(np.sqrt(np.mean(diff**2))),
                'heldRmsRatioDb':float(20*np.log10(max(float(np.sqrt(np.mean(b[start:end]**2))),1e-12)/
                    max(float(np.sqrt(np.mean(a[start:end]**2))),1e-12))),
                'spectrum':str(artifact.relative_to(ROOT))})
    write('spectral-evaluation.json',{'scope':config['spectrum'],'results':results,'complete':len(results)==108})
    print('Native spectral comparisons',len(results),flush=True)


if __name__=='__main__':
    config=protocol();{'render':render,'evaluate':evaluate,'identity':identity,'spectrum':spectrum}[sys.argv[1]](config)
