"""Captured onset-to-sustain evaluation with immediate controls and no future score."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import parselmouth
import pyworld
from neural_waveform import load_model,SIZE
from compiled_neural_decoder import compile_decoder
from neural_performance import load_performance
from neural_articulation import ArticulationUnit,NeuralArticulationStream
from neural_vowel_instrument import VowelScore
from compact_vocal_instrument import VocalFrame,RATE,BLOCK
from live_neural_grains import DELAY
from benchmark_neural_vowels import pitch_metrics,formants

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-articulation/v1'


def envelope(count):
    n=np.arange(count)
    attack=np.clip(n/(RATE*.005),0.,1.)
    release=np.clip((count-1-n)/(RATE*.035),0.,1.)
    return (.5-.5*np.cos(np.pi*attack))*(.5-.5*np.cos(np.pi*release))


def controlled_render(unit,decoder,score,stream_factory=NeuralArticulationStream):
    score.validate();begin=time.perf_counter();stream=stream_factory(unit,decoder)
    initialization=time.perf_counter()-begin;count=round(score.seconds*RATE);parts=[];timings=[]
    for start in range(0,count,BLOCK):
        begin=time.perf_counter();hz=float(score.frequency((start+BLOCK-1)/RATE))
        parts.append(stream.push(VocalFrame(hz,(1.,))));timings.append(time.perf_counter()-begin)
    begin=time.perf_counter();parts.append(stream.finish());tail_time=time.perf_counter()-begin
    begin=time.perf_counter();audio=np.concatenate(parts)[DELAY:DELAY+count]*envelope(count)*score.gain
    consumer_time=time.perf_counter()-begin
    try:stream.push(VocalFrame(220.,(1.,)))
    except RuntimeError:closed=True
    else:closed=False
    return audio,{'fixedWaveformDelayMs':20.,'firstPhysicalPacketComputeMs':1000*(initialization+timings[0]),
        'packetP95Ms':float(1000*np.percentile(timings,95)),'maximumPacketMs':float(1000*max(timings)),
        'packetsOver5Ms':sum(t>.005 for t in timings),'packets':len(timings),
        'tailComputeMs':1000*tail_time,'alignmentAndEnvelopeMs':1000*consumer_time,
        'computeRealTimeFactor':(initialization+sum(timings)+tail_time+consumer_time)/score.seconds,
        'maximumCachedWaveformSamples':stream.reader.maximum_cache_frames*SIZE,
        'residentSourceControlScalars':stream.reader.matrix.values.size+stream.reader.matrix.gains.size,
        'closedPreventsOutput':closed,'closedClearsHistory':not stream.reader.cache and
            not any(np.any(a) for a in (stream.pending,stream.weights,stream.background,stream.voicing,stream.gain)),
        'timingScope':'Resident frozen compiled neural decoder and encoded source unit; includes reader creation, neural decoding, voicing selection and immediate 5 ms controls. Fixed 20 ms waveform delay is additional. Excludes capture, loading, compilation, transport and devices. Tail/alignment/envelope measured separately.'}


def metrics(audio,score,unit,source,decoded,analysis_hold_start=None):
    actual_hold=source['holdStartSample']/RATE
    hold=actual_hold if analysis_hold_start is None else float(analysis_hold_start)
    sound=parselmouth.Sound(audio,sampling_frequency=RATE)
    pitch=sound.to_pitch_ac(time_step=.01,pitch_floor=50.,pitch_ceiling=900.)
    wf,wt=pyworld.harvest(np.ascontiguousarray(audio),RATE,f0_floor=50.,f0_ceil=900.,frame_period=10.)
    result={};tracks={}
    pulses=unit.performance.pulses
    groups=[g for g in np.split(pulses,np.flatnonzero(np.diff(pulses)>DELAY)+1) if len(g)>=3]
    for name,times,f0 in (('praat',pitch.xs(),pitch.selected_array['frequency']),('world',wt,wf)):
        held=(times>=hold+.080)&(times<=score.seconds-.050)
        result[name]=pitch_metrics(times[held],f0[held],score)
        result[name]['scope']='Sustained nucleus from selected hold start +80 ms through output end -50 ms'
        attack=np.zeros(len(times),dtype=bool)
        for group in groups:
            attack|=(times>=group[0]/RATE+.030)&(times<=min(actual_hold-.030,group[-1]/RATE-.030))
        result[name+'VoicedAttack']=pitch_metrics(times[attack],f0[attack],score)
        result[name+'VoicedAttack']['scope']='Source-detected voiced onset regions, excluding 30 ms at boundaries and 30 ms before hold; may contain no eligible frames'
        tracks[name+'Times']=times;tracks[name+'Pitch']=f0
    observed=formants(audio[round((hold+.08)*RATE):-1200]);target=source['sourceFormantsHz']
    # The strict prefix is at least 20 ms before the first valid pulse group,
    # so pulse windows/voicing ramps cannot reach it. This is a source detector
    # label, not an independent human unvoiced annotation.
    stop=min(round(groups[0][0]-DELAY),len(decoded),len(audio));start=round(.020*RATE)
    unvoiced=audio[start:stop] if stop>start else np.empty(0)
    reference=decoded[start:stop] if stop>start else np.empty(0)
    join=int(np.ceil(source['holdStartSample']));a=max(1,join-600);b=min(len(audio)-1,join+600)
    difference=abs(np.diff(audio))
    result.update(measuredFormantsHz=observed,
        formantDriftPercent=[100*abs(v-r)/r if v is not None and r is not None else None for v,r in zip(observed,target)],
        rawPeak=float(max(abs(audio))),overFullScaleSamples=int(sum(abs(audio)>=1.)),finite=bool(np.isfinite(audio).all()),
        strictUnvoicedPrefix={'samples':len(unvoiced),'sourceFirstVoicedPulseSample':float(groups[0][0]),
            'rmsDifferenceFromNeuralSource':float(np.sqrt(np.mean((unvoiced-reference)**2))) if len(unvoiced) else None,
            'maximumAbsoluteDifference':float(max(abs(unvoiced-reference))) if len(unvoiced) else None},
        holdJunction={'sample':join,'adjacentSampleDifference':float(difference[join-1]),
            'maximumAdjacentDifferenceWithin25Ms':float(max(difference[a:b])),
            'scope':'Sample-derivative diagnostic only; no click audibility threshold is asserted'})
    return result,tracks


def main(out=OUT,short=False):
    capture=json.loads((OUT/'capture-report.json').read_text())
    if not capture['complete']:raise ValueError('Incomplete unit capture')
    checkpoint=ROOT/capture['model']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=capture['modelSha256']:raise ValueError('Changed decoder')
    _,decoder,_=load_model(checkpoint);decoder=compile_decoder(decoder)
    before=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest()
    report={'scope':'Captured consonant attacks into pitch-controlled sustained nuclei; independent new speech is not demonstrated',
        'captureReport':'evidence/neural-articulation/v1/capture-report.json','model':capture['model'],
        'modelSha256':capture['modelSha256'],'expectedCases':27 if short else 45,'results':[],'complete':False,
        'limits':capture['limits']+['Source voicing and nucleus boundaries are automatic and may be wrong.',
            'Nasal/voiced consonants are pitched; source-detected unvoiced detail is preserved through the neural decoder.',
            'This extends captured words, not a learned compositional phoneme system or general voice cloning.',
            'No additional breath control, live unit switching, or input understanding in this experiment.']}
    if short:
        report['scope']='Duration-matched articulation diagnostic; retained alongside the longer singing cases, not substituted for them'
        report['durationRule']='Maximum of cropped source duration and selected hold onset +150 ms; same source units, decoder and renderer'
    out.mkdir(parents=True,exist_ok=True)
    def persist():(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    hashes={}
    for source in capture['units']:
        file=ROOT/source['artifact'];hashes[source['artifact']]=source['artifactSha256']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=source['artifactSha256']:raise ValueError('Changed source unit')
        performance,metadata=load_performance(file)
        unit=ArticulationUnit(performance,metadata['holdFirstPulse'],metadata['holdLastPulse'])
        decoded=decoder.render(performance.matrix);base=source['sourceMedianPitchHz']
        scores={'native':VowelScore(seconds=2.5,hz=(base,)),
            'down-three':VowelScore(seconds=2.5,hz=(base*2**(-3/12),)),
            'up-three':VowelScore(seconds=2.5,hz=(base*2**(3/12),)),
            'vibrato':VowelScore(seconds=2.5,hz=(base,),vibrato_cents=35.),
            'melody':VowelScore(seconds=2.5,times=(0.,.6,1.,1.4,1.8,2.2),
                hz=tuple(base*2**(n/12) for n in (0,0,3,3,7,0)))}
        if short:
            duration=max(source['sourceSeconds'],source['holdStartSample']/RATE+.150)
            scores={'short-'+mode:VowelScore(seconds=duration,hz=(base*ratio,))
                for mode,ratio in (('native',1.),('down-three',2**(-3/12)),('up-three',2**(3/12)))}
        for mode,score in scores.items():
            audio,timing=controlled_render(unit,decoder,score);name=source['name']+'-'+mode
            file=out/(name+'.wav');sf.write(file,audio,RATE,subtype='FLOAT')
            evaluated,tracks=metrics(audio,score,unit,source,decoded)
            track=file.with_suffix('.tracks.npz');np.savez(track,**tracks)
            row={'name':name,'speaker':source['speaker'],'unit':source['unit'],'text':source['text'],
                'mode':mode,'score':score.__dict__,'holdStartSeconds':source['holdStartSample']/RATE,
                'sourceArtifact':source['artifact'],'sourceArtifactSha256':source['artifactSha256'],
                'file':str(file.relative_to(ROOT)),'tracks':str(track.relative_to(ROOT)),
                'status':'ok','streaming':timing,**evaluated}
            report['results'].append(row);persist()
            print(name,'held Praat',row['praat']['p95ErrorCents'],'unvoiced error',row['strictUnvoicedPrefix']['maximumAbsoluteDifference'],flush=True)
    if before!=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest():raise AssertionError('Decoder weights changed')
    for file,sha in hashes.items():
        if hashlib.sha256((ROOT/file).read_bytes()).hexdigest()!=sha:raise AssertionError('Unit changed')
    report['decoderAndSourcesUnchanged']=True;report['complete']=len(report['results'])==report['expectedCases'];persist()


if __name__=='__main__':
    import sys
    if '--short' in sys.argv:main(out=ROOT/'evidence/neural-articulation/short-v1',short=True)
    else:main()
