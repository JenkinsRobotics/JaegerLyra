"""Common-window waveform checks for four sustain methods plus normal speech."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
import re
import urllib.request
from pathlib import Path
import numpy as np
import soundfile as sf
import parselmouth
import pyworld
from neural_vowel_instrument import VowelScore
from benchmark_neural_vowels import pitch_metrics
from check_articulation_words import EQUIVALENTS

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/kokoro-control-voicebox/sustain-v1'


def read(path):return json.loads((ROOT/path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(reference_norm=False):
    out=ROOT/'evidence/kokoro-control-voicebox/reference-norm-v2' if reference_norm else OUT
    study=read('evidence/kokoro-control-voicebox/sustain-v1/report.json')
    capture=read('evidence/kokoro-control-voicebox/capture-v1/report.json')
    legacy=read('evidence/neural-articulation/kokoro-reference-v1/evaluation.json')
    grains=read('evidence/neural-articulation/stationarity-v1/report.json')
    grain_words=read('evidence/neural-articulation/stationarity-v1/word-report.json')
    if not all(j['complete'] for j in (study,capture,legacy,grains,grain_words)):raise ValueError('Incomplete comparison')
    old={r['file']:r for r in legacy['results']+grain_words['results']}
    cases=[{**r,'transcriptReused':False} for r in study['results']]
    cases += [{**r,'word':r['unit'],'method':'legacy-kokoro','transcriptReused':True} for r in legacy['results']]
    for row in grains['results']:
        if row['method']=='stationary-fixed' and row['mode'] in ('native','melody'):
            cases.append({**row,'word':row['unit'],'method':'captured-cycle','sha256':old[row['file']]['sha256'],'transcriptReused':True})
    if reference_norm:
        addition=read('evidence/kokoro-control-voicebox/reference-norm-v2/report.json')
        if not addition['complete']:raise ValueError('Incomplete normalization experiment')
        cases += [{**r,'transcriptReused':False} for r in addition['results']]
    report={'scope':'Development comparison of pretrained acoustic-control holds, whole-vowel warps and captured-cycle resynthesis',
        'expectedCases':72,'expectedNativeSources':9,'results':[],'nativeSources':[],'complete':False,
        'limits':['All methods use the same requested duration/pitch and common interior analysis interval, but their onset/release timing differs.',
            'New Kokoro controls are rendered as raw float WAVs; legacy references contain prior app filtering/level normalization. ASR is diagnostic, not a blinded quality test.',
            'Single-word and sustained-tone ASR is unstable, including on original Kokoro audio.',
            'No listening acceptance, general identity transfer or independent neural training is demonstrated.']}
    if reference_norm:
        report=read('evidence/kokoro-control-voicebox/sustain-v1/evaluation.json')
        if not report['complete']:raise ValueError('Finish the base evaluation first')
        report['scope']='Control-retiming and source-normalization factorial study with preserved earlier references'
        report['expectedCases']=108;report['complete']=False
        report['reusedPriorEvaluationCases']=len(report['results'])
        report['limits']+=addition['limits']
    if (out/'evaluation.json').exists():report=json.loads((out/'evaluation.json').read_text())
    done={r['name']:r for r in report['results']}
    def persist():(out/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for row in cases:
        file=ROOT/row['file']
        if sha(file)!=row['sha256']:raise ValueError('Changed audio')
        if row['name'] in done:
            if done[row['name']]['sha256']!=row['sha256']:raise ValueError('Changed prior evaluation')
            continue
        reference=next(r for r in study['results'] if r['speaker']==row['speaker'] and r['word']==row['word']
            and r['mode']==row['mode'] and r['method']=='interior-hold')
        start=reference['plan']['holdStartSeconds']+.080;end=reference['plan']['holdEndSeconds']-.080
        audio,rate=sf.read(file);assert rate==24000;score=VowelScore(**row['score'])
        praat=parselmouth.Sound(audio,sampling_frequency=rate).to_pitch_ac(time_step=.01,pitch_floor=50.,pitch_ceiling=900.)
        wf,wt=pyworld.harvest(np.ascontiguousarray(audio),rate,f0_floor=50.,f0_ceil=900.,frame_period=10.)
        evaluated={};tracks={}
        for name,times,f0 in [('praat',praat.xs(),praat.selected_array['frequency']),('world',wt,wf)]:
            keep=(times>=start)&(times<=end)
            evaluated[name]=pitch_metrics(times[keep],f0[keep],score)
            evaluated[name]['scope']='Common interior inserted-hold interval, excluding 80 ms at each boundary'
            tracks[name+'Times']=times;tracks[name+'Pitch']=f0
        tracks['requestedTimes']=np.arange(round(score.seconds*100))/100
        tracks['requestedF0']=np.array([score.frequency(t) for t in tracks['requestedTimes']])
        track=out/(row['name']+'-evaluation.tracks.npz');np.savez(track,**tracks)
        if row['transcriptReused']:
            previous=old[row['file']]
            if previous['sha256']!=row['sha256']:raise ValueError('Changed old ASR input')
            transcript=previous['transcript']
        else:
            request=urllib.request.Request('http://127.0.0.1:8791/api/transcribe',file.read_bytes(),{'Content-Type':'audio/wav'})
            with urllib.request.urlopen(request,timeout=60) as response:result=json.load(response)
            if 'text' not in result:raise ValueError('Missing ASR result')
            transcript=result['text']
        normalized=' '.join(re.findall(r'[a-z]+',transcript.lower()))
        region=audio[round(start*rate):round(end*rate)]
        report['results'].append({k:row[k] for k in ('name','speaker','word','mode','method','file','sha256','transcriptReused')}|
            {'score':row['score'],'analysisStartSeconds':start,'analysisEndSeconds':end,
             'tracks':str(track.relative_to(ROOT)),**evaluated,'transcript':transcript,
             'exactSpellingMatch':normalized==row['word'],'homophoneEquivalentMatch':normalized in EQUIVALENTS[row['word']],
             'heldRms':float(np.sqrt(np.mean(region**2))),'rawPeak':float(max(abs(audio))),
             'overFullScaleSamples':int(np.sum(abs(audio)>=1.)),'finite':bool(np.isfinite(audio).all())})
        persist();print(row['name'],'Praat voiced',evaluated['praat']['voicedFraction'],repr(transcript),flush=True)
    native_done={r['name'] for r in report['nativeSources']}
    for source in capture['results']:
        if source['name'] in native_done:continue
        file=ROOT/source['nativeAudio']
        if sha(file)!=source['nativeAudioSha256']:raise ValueError('Changed native source')
        request=urllib.request.Request('http://127.0.0.1:8791/api/transcribe',file.read_bytes(),{'Content-Type':'audio/wav'})
        with urllib.request.urlopen(request,timeout=60) as response:result=json.load(response)
        transcript=result['text'];normalized=' '.join(re.findall(r'[a-z]+',transcript.lower()))
        report['nativeSources'].append({'name':source['name'],'speaker':source['speaker'],'word':source['word'],
            'file':source['nativeAudio'],'sha256':source['nativeAudioSha256'],'transcript':transcript,
            'exactSpellingMatch':normalized==source['word'],'homophoneEquivalentMatch':normalized in EQUIVALENTS[source['word']]})
        persist()
    report['complete']=len(report['results'])==report['expectedCases'] and len(report['nativeSources'])==9;persist()


if __name__=='__main__':
    import sys
    main(reference_norm='--reference-norm' in sys.argv)
