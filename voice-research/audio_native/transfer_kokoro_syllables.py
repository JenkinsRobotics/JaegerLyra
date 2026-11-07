"""Transfer controlled teacher syllables into the frozen independent runtime.

Stages: capture, render, evaluate. Pretrained Kokoro is
never loaded by this script; source WAVs and annotations are offline fixtures.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import load_model,RATE,SIZE
from neural_performance import EncodedPerformance,save_performance,load_performance,validate_pulses
from neural_articulation import ArticulationUnit
from neural_syllable_stream import NeuralSyllableStream
from compiled_neural_decoder import compile_decoder
from compact_vocal_instrument import VocalFrame,BLOCK
from live_neural_grains import DELAY
from pcm_reference_decoder import pcm_performance,PcmFrameDecoder
from neural_vowel_instrument import VowelScore

ROOT=Path(__file__).resolve().parents[2]
TEACHER=ROOT/'evidence/kokoro-control-voicebox/grid-level-v1'
OUT=ROOT/'evidence/neural-syllables/teacher-transfer-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):return json.loads((OUT/name).read_text())
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def snr(reference,actual):
    return float(10*np.log10(max(float(np.sum(reference**2)),1e-30)/max(float(np.sum((reference-actual)**2)),1e-30)))


def protocol():
    source=json.loads((TEACHER/'report.json').read_text());prior=json.loads((TEACHER/'protocol.json').read_text())
    if not source['complete']:raise ValueError('Incomplete teacher fixtures')
    checkpoint='evidence/neural-waveform/axis-detail-v2/projected-r1e-7.npz'
    config={'scope':'Independent decoder reconstruction and immediate onset/hold/release resynthesis of pretrained synthetic references',
        'teacherReportSha256':sha(TEACHER/'report.json'),'teacherVariants':['baseline','level'],
        'sources':18,'nativeUnits':36,'fullTeacherClips':72,'liveRenders':144,
        'model':checkpoint,'modelSha256':sha(ROOT/checkpoint),
        'runtimeCode':{n:sha(Path(__file__).with_name(n)) for n in ('neural_waveform.py','compiled_neural_decoder.py',
            'neural_syllable_stream.py','neural_articulation.py','neural_performance_stream.py','pcm_reference_decoder.py')},
        'capture':'Native three-second teacher clip; offline Praat periodic-cc pulse search constrained to 0.8–1.2 times the authored base F0, bounded to 60–800 Hz. First pulse at hold start +250 ms; retain the following <=125 ms, >=6 periods, <=1.25 period ratio. Release resumes source at teacher hold end -200 ms. Failures retained.',
        'knownTeacherPitchAssistance':True,'recognizedPhonemes':False,
        'reconstruction':'All native and melody teacher clips independently encoded and decoded; compare against their exact waveform and linear inverse reference.',
        'runtime':'5 ms pitch/gain packets and an immediate release event; no original PCM or pretrained Kokoro weights in neural rendering. Same controller with lossless PCM frames is a separate diagnostic.',
        'holdMode':'loop','waveformDelayMs':20,'releaseSchedule':'Output release time equals the captured source release position for the matched-duration benchmark; the controller itself has no future schedule.',
        'analysis':'Common teacher inserted-hold interval, excluding 80 ms at both boundaries; includes source-prefix continuation before the independent hold and the beginning of release if present.',
        'pitchCheck':prior['pitchCheck'],'asrEquivalents':prior['asrEquivalents'],
        'limits':['Both variants come from the same pretrained teacher; this is not a clean-room model or new independent phoneme learning.',
            'Native held controls are captured for live resynthesis; direct melody controls are used only for whole-clip reconstruction and comparison.',
            'Pulse capture uses known synthetic teacher pitch and hold annotations; it is not a general reference-audio clone extractor.',
            'The lossless PCM comparator retains the source waveform explicitly and is not the independent model.',
            'Encoded coordinates retain detailed waveform information and are not compact, disentangled articulators.',
            'Compute timing excludes capture, loading, transport and audio devices; the 20 ms waveform delay is additional.'],
        'appDefaultChanged':False,'listeningAccepted':False,'newNeuralTraining':False}
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        if read('protocol.json')!=config:raise ValueError('Frozen transfer protocol changed')
    else:write('protocol.json',config)
    return config,source


def capture(config,teacher):
    import parselmouth
    from parselmouth.praat import call
    encoder,decoder,_=load_model(ROOT/config['model']);decoder=compile_decoder(decoder)
    report={'protocolSha256':sha(OUT/'protocol.json'),'sources':teacher['sources'],'references':[],
        'reconstructions':[],'units':[],'failures':[],'complete':False}
    if (OUT/'capture.json').exists():report=read('capture.json')
    if report['complete']:print('Capture complete; retained.',flush=True);return
    done={r['teacherName'] for r in report['reconstructions']}
    cases=[r for r in teacher['results'] if r['method'] in config['teacherVariants']]
    for source in cases:
        if source['name'] in done:continue
        path=ROOT/source['file']
        if sha(path)!=source['sha256']:raise ValueError('Changed teacher waveform')
        audio,rate=sf.read(path);assert rate==RATE
        variant=source['method'];name=source['speaker']+'-'+source['word']+'-'+variant+'-'+source['mode']
        matrix=encoder.encode(audio);performance=EncodedPerformance(matrix,np.empty(0))
        artifact=OUT/(name+'-encoded.npz');save_performance(artifact,performance,{'teacher':source['file'],'variant':variant})
        reopened,_=load_performance(artifact);decoded=decoder.render(reopened.matrix)
        file=OUT/(name+'-reconstruction.wav');sf.write(file,decoded,RATE,subtype='FLOAT');saved,_=sf.read(file)
        reconstructed={'name':name+'-reconstruction','teacherName':source['name'],'speaker':source['speaker'],'word':source['word'],
            'variant':variant,'mode':source['mode'],'method':'reconstruction','file':str(file.relative_to(ROOT)),'sha256':sha(file),
            'score':source['score'],'plan':source['plan'],'artifact':str(artifact.relative_to(ROOT)),'artifactSha256':sha(artifact),
            'waveformSnrDb':snr(audio,saved),'linearReferenceSnrDb':snr(audio,encoder.reconstruct_linear(matrix)),
            'maximumWaveformError':float(max(abs(audio-saved))),'rawPeak':float(max(abs(saved))),
            'finite':bool(np.isfinite(saved).all()),'overFullScaleSamples':int(np.sum(abs(saved)>=1.))}
        report['reconstructions'].append(reconstructed)
        report['references'].append({**source,'variant':variant,'method':'teacher','teacherName':source['name']})
        if source['mode']=='native':
            try:
                base=float(source['score']['hz'][0]);floor=max(60.,.8*base);ceiling=min(800.,1.2*base)
                sound=parselmouth.Sound(audio,sampling_frequency=RATE)
                points=call(sound,'To PointProcess (periodic, cc)',floor,ceiling)
                count=int(call(points,'Get number of points'))
                pulses=np.array([call(points,'Get time from index',i+1)*RATE for i in range(count)])
                pulses=validate_pulses(pulses[(pulses>=0)&(pulses<len(audio))],len(audio))
                first=int(np.searchsorted(pulses,(source['plan']['holdStartSeconds']+.25)*RATE))
                if first<1 or first>=len(pulses)-1:raise ValueError('No pulse at annotated steady interior')
                last=int(np.searchsorted(pulses,pulses[first]+.125*RATE,side='right')-1)
                release=round((source['plan']['holdEndSeconds']-.2)*RATE)
                if last>=len(pulses)-1 or last-first<6 or pulses[last]>=release:raise ValueError('Insufficient held span or release context')
                periods=np.diff(pulses[first:last+1])
                if max(periods)>1.25*min(periods):raise ValueError('Selected source periods exceed frozen stability ratio')
                unit=ArticulationUnit(EncodedPerformance(matrix,pulses),first,last);unit.validate()
                metadata={'holdFirstPulse':first,'holdLastPulse':last,'releaseSample':release,
                    'holdStartSample':float(pulses[first]),'holdEndSample':float(pulses[last]),
                    'teacherPitchHz':base,'pulseSearchRangeHz':[floor,ceiling],
                    'sourceMedianPulseHz':float(RATE/np.median(periods)),'sourcePeriodRatio':float(max(periods)/min(periods))}
                unitfile=OUT/(name+'-unit.npz');save_performance(unitfile,unit.performance,metadata)
                report['units'].append({'name':source['speaker']+'-'+source['word']+'-'+variant,
                    'speaker':source['speaker'],'word':source['word'],'variant':variant,
                    'sourceFile':source['file'],'sourceSha256':source['sha256'],
                    'artifact':str(unitfile.relative_to(ROOT)),'artifactSha256':sha(unitfile),**metadata})
                print(name,'captured; reconstruction SNR',round(reconstructed['waveformSnrDb'],2),flush=True)
            except ValueError as exc:
                report['failures'].append({'name':name,'speaker':source['speaker'],'word':source['word'],'variant':variant,'reason':str(exc)})
                print(name,'CAPTURE REJECTED',exc,flush=True)
        write('capture.json',report)
    protocol();report['complete']=(len(report['reconstructions'])==72 and len(report['units'])+len(report['failures'])==36)
    write('capture.json',report)


def render(config,teacher):
    captured=read('capture.json')
    if not captured['complete']:raise ValueError('Incomplete capture')
    _,decoder,_=load_model(ROOT/config['model']);decoder=compile_decoder(decoder);pcm_decoder=PcmFrameDecoder()
    report={'protocolSha256':sha(OUT/'protocol.json'),'sources':teacher['sources'],'results':[],
        'captureFailures':captured['failures'],'expectedEligibleRenders':len(captured['units'])*4,'complete':False}
    if (OUT/'report.json').exists():report=read('report.json')
    done={r['name']:r for r in report['results']}
    for source in captured['units']:
        if sha(ROOT/source['artifact'])!=source['artifactSha256'] or sha(ROOT/source['sourceFile'])!=source['sourceSha256']:
            raise ValueError('Changed captured unit')
        performance,metadata=load_performance(ROOT/source['artifact'])
        original,rate=sf.read(ROOT/source['sourceFile']);assert rate==RATE
        pcm=pcm_performance(original,performance.pulses)
        if max(abs(pcm_decoder.render(pcm.matrix)-original))>1e-12:raise ValueError('PCM control is not lossless')
        for mode in ('native','melody'):
            guide=next(r for r in captured['references'] if
                (r['speaker'],r['word'],r['variant'],r['mode'])==(source['speaker'],source['word'],source['variant'],mode))
            score=VowelScore(**guide['score']);score.validate();count=round(score.seconds*RATE)
            for method,encoded,synth in [('live-neural',performance,decoder),('live-pcm',pcm,pcm_decoder)]:
                name=source['name']+'-'+mode+'-'+method
                if name in done:
                    if sha(ROOT/done[name]['file'])!=done[name]['sha256']:raise ValueError('Changed prior render')
                    continue
                unit=ArticulationUnit(encoded,metadata['holdFirstPulse'],metadata['holdLastPulse'])
                begin=time.perf_counter();stream=NeuralSyllableStream(unit,synth,metadata['releaseSample'],hold_mode='loop')
                initialize=time.perf_counter()-begin;parts=[];times=[];release_at=None
                for start in range(0,count,BLOCK):
                    begin=time.perf_counter()
                    if release_at is None and start>=metadata['releaseSample']:
                        stream.release();release_at=start
                    hz=float(score.frequency((start+BLOCK-1)/RATE))
                    parts.append(stream.push(VocalFrame(hz,(1.,),gain=score.gain)));times.append(time.perf_counter()-begin)
                begin=time.perf_counter();parts.append(stream.finish());finish=time.perf_counter()-begin
                audio=np.concatenate(parts)[DELAY:DELAY+count]
                if len(audio)!=count or not np.isfinite(audio).all():raise ValueError('Invalid streamed waveform')
                if release_at!=metadata['releaseSample']:raise ValueError('Release was not packet-aligned')
                closed=stream.closed and not stream.reader.cache and not any(np.any(a) for a in (stream.pending,stream.weights,stream.background,stream.voicing,stream.gain))
                if not closed:raise ValueError('Stream did not clear pending history')
                file=OUT/(name+'.wav');sf.write(file,audio,RATE,subtype='FLOAT')
                report['results'].append({'name':name,'speaker':source['speaker'],'word':source['word'],
                    'variant':source['variant'],'mode':mode,'method':method,'file':str(file.relative_to(ROOT)),'sha256':sha(file),
                    'score':guide['score'],'plan':guide['plan'],'sourceArtifact':source['artifact'],
                    'independentHoldStartSeconds':source['holdStartSample']/RATE,'releaseCommandSeconds':release_at/RATE,
                    'releaseSourceSeconds':metadata['releaseSample']/RATE,
                    'rawPeak':float(max(abs(audio))),'finite':True,'overFullScaleSamples':int(np.sum(abs(audio)>=1.)),
                    'streaming':{'firstPhysicalPacketComputeMs':1000*(initialize+times[0]),'packetP95Ms':float(np.percentile(times,95)*1000),
                        'maximumPacketMs':float(max(times)*1000),'packetsOver5Ms':sum(t>.005 for t in times),
                        'waveformDelayMs':20,'tailComputeMs':finish*1000,'closedClearsHistory':closed,
                        'maximumCachedWaveformSamples':stream.reader.maximum_cache_frames*SIZE,
                        'residentSourceControlScalars':encoded.matrix.values.size+encoded.matrix.gains.size,
                        'scope':('Independent frozen decoder with encoded source controls' if method=='live-neural' else 'Lossless retained-PCM diagnostic')+
                            '; immediate 5 ms controls and release event. Preparation/loading/transport/devices excluded; 20 ms waveform delay is additional.'}})
                write('report.json',report);print(name,'first packet ms',round(report['results'][-1]['streaming']['firstPhysicalPacketComputeMs'],3),flush=True)
    protocol();report['complete']=len(report['results'])==report['expectedEligibleRenders'];write('report.json',report)


def evaluate(config,teacher):
    import parselmouth
    import pyworld
    from benchmark_neural_vowels import pitch_metrics
    captured=read('capture.json');rendered=read('report.json')
    if not captured['complete'] or not rendered['complete']:raise ValueError('Incomplete capture/render')
    prior=json.loads((TEACHER/'evaluation.json').read_text());previous={r['name']:r for r in prior['results']}
    cases=captured['references']+captured['reconstructions']+rendered['results']
    report={'protocolSha256':sha(OUT/'protocol.json'),'sources':prior['sources'],'results':[],'complete':False,
        'expectedCases':len(cases),'captureFailures':captured['failures']}
    if (OUT/'evaluation.json').exists():report=read('evaluation.json')
    done={r['name']:r for r in report['results']}
    for row in cases:
        file=ROOT/row['file']
        if sha(file)!=row['sha256']:raise ValueError('Changed evaluation audio')
        if row['name'] in done:
            if done[row['name']]['sha256']!=row['sha256']:raise ValueError('Changed old evaluation')
            continue
        audio,rate=sf.read(file);assert rate==RATE;score=VowelScore(**row['score'])
        start=row['plan']['holdStartSeconds']+.08;end=row['plan']['holdEndSeconds']-.08
        if row['method']=='teacher':
            old=previous[row['teacherName']]
            if old['sha256']!=row['sha256']:raise ValueError('Changed reused teacher evaluation')
            metrics={k:old[k] for k in ('praat','world','tracks','transcript','exactSpellingMatch','homophoneEquivalentMatch','bothPitchChecksPassed','heldRms')}
        else:
            praat=parselmouth.Sound(audio,sampling_frequency=rate).to_pitch_ac(time_step=.01,pitch_floor=50.,pitch_ceiling=900.)
            wf,wt=pyworld.harvest(np.ascontiguousarray(audio),rate,f0_floor=50.,f0_ceil=900.,frame_period=10.)
            metrics={};tracks={}
            for name,t,f0 in [('praat',praat.xs(),praat.selected_array['frequency']),('world',wt,wf)]:
                keep=(t>=start)&(t<=end);value=pitch_metrics(t[keep],f0[keep],score)
                value['pitchCheckPassed']=bool(value['voicedFraction']>=.95 and value['p95ErrorCents'] is not None and value['p95ErrorCents']<=50)
                metrics[name]=value;tracks[name+'Times']=t;tracks[name+'Pitch']=f0
            tracks['requestedTimes']=np.arange(round(score.seconds*100))/100;tracks['requestedF0']=score.frequency(tracks['requestedTimes'])
            track=OUT/(row['name']+'-tracks.npz');np.savez(track,**tracks);metrics['tracks']=str(track.relative_to(ROOT))
            request=urllib.request.Request('http://127.0.0.1:8791/api/transcribe',file.read_bytes(),{'Content-Type':'audio/wav'})
            with urllib.request.urlopen(request,timeout=60) as response:transcript=json.load(response)['text']
            normalized=' '.join(re.findall(r'[a-z]+',transcript.lower()))
            metrics.update(transcript=transcript,exactSpellingMatch=normalized==row['word'],
                homophoneEquivalentMatch=normalized in config['asrEquivalents'][row['word']],
                bothPitchChecksPassed=metrics['praat']['pitchCheckPassed'] and metrics['world']['pitchCheckPassed'],
                heldRms=float(np.sqrt(np.mean(audio[round(start*RATE):round(end*RATE)]**2))))
        report['results'].append({k:row[k] for k in ('name','speaker','word','variant','mode','method','file','sha256','score')}|
            {'analysisStartSeconds':start,'analysisEndSeconds':end,'teacherEvaluationReused':row['method']=='teacher',**metrics})
        write('evaluation.json',report);print(row['name'],metrics['bothPitchChecksPassed'],repr(metrics['transcript']),flush=True)
    report['complete']=len(report['results'])==len(cases);write('evaluation.json',report)


if __name__=='__main__':
    config,teacher=protocol()
    {'capture':capture,'render':render,'evaluate':evaluate}[sys.argv[1]](config,teacher)
