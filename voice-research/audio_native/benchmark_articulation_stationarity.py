"""Factorial sustain experiment: original/reselected span x loop/fixed pulse.

All pitch/duration controls remain those of the original panel. The baseline
audio is reused, with a common nucleus-analysis interval for each paired case.
No transcript, output metric or listening score participates in span selection.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
from functools import partial
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import load_model,RATE
from compiled_neural_decoder import compile_decoder
from neural_performance import load_performance
from neural_articulation import ArticulationUnit,NeuralArticulationStream
from neural_vowel_instrument import VowelScore
from vowel_stationarity import select_stationary_unit,spectral_frames,shape_variation,WINDOW_SIZE
from benchmark_neural_articulation import controlled_render,metrics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-articulation/stationarity-v1'


def main():
    capture=json.loads((ROOT/'evidence/neural-articulation/v1/capture-report.json').read_text())
    long=json.loads((ROOT/'evidence/neural-articulation/v1/report.json').read_text())
    short=json.loads((ROOT/'evidence/neural-articulation/short-v1/report.json').read_text())
    if not all(r['complete'] for r in (capture,long,short)):raise ValueError('Incomplete input study')
    checkpoint=ROOT/capture['model']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=capture['modelSha256']:raise ValueError('Changed decoder')
    _,decoder,_=load_model(checkpoint);decoder=compile_decoder(decoder)
    before=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest()
    OUT.mkdir(parents=True,exist_ok=True)
    selections={'scope':'Development-panel source-only spectral-stationarity selection; not independent validation',
        'units':[],'complete':False}
    report={'scope':'Matched source-span and pulse-repetition ablations under unchanged original controls',
        'model':capture['model'],'modelSha256':capture['modelSha256'],
        'expectedCases':108,'expectedNewRenders':81,'results':[],'complete':False,
        'selectionReport':'evidence/neural-articulation/stationarity-v1/selection-report.json',
        'limits':['Same development words as the prior failed panel; no independent test-set or new-speaker claim.',
            'The fixed mode reuses the first selected source pulse; it removes variation, not necessarily artifacts or unnaturalness.',
            'Source spectral stationarity is not vowel recognition and is affected by source pitch and harmonic sampling.',
            'All methods use the original requested pitch and duration; reselection does not retarget the controls.',
            'New selections must leave at least 150 ms of sustain in the original short condition.',
            'Pitch and formants use a common post-hold interval per comparison; onset boundaries retain each unit selection.',
            'Original source formant targets remain fixed, including missing estimates.']}
    def persist():(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    hashes={}
    for source in capture['units']:
        artifact=ROOT/source['artifact'];hashes[source['artifact']]=source['artifactSha256']
        if hashlib.sha256(artifact.read_bytes()).hexdigest()!=source['artifactSha256']:raise ValueError('Changed unit')
        performance,meta=load_performance(artifact)
        original=ArticulationUnit(performance,meta['holdFirstPulse'],meta['holdLastPulse'])
        pcm,rate=sf.read(ROOT/source['trimmedSource']);assert rate==RATE
        short_reference=next(r for r in short['results'] if r['speaker']==source['speaker'] and r['unit']==source['unit'] and r['mode']=='short-native')
        latest=(short_reference['score']['seconds']-.150)*RATE
        selected,selection=select_stationary_unit(pcm,original,latest)
        selections['units'].append({'name':source['name'],'sourceArtifact':source['artifact'],
            'sourceArtifactSha256':source['artifactSha256'],**selection})
        (OUT/'selection-report.json').write_text(json.dumps(selections,indent=2,allow_nan=False)+'\n')
        decoded=decoder.render(performance.matrix)
        common_hold=max(performance.pulses[original.hold_first],performance.pulses[selected.hold_first])/RATE
        methods={'baseline-loop':(original,'loop'),'fixed-original':(original,'fixed-first'),
            'stationary-loop':(selected,'loop'),'stationary-fixed':(selected,'fixed-first')}
        cases=[r for r in long['results']+short['results'] if r['speaker']==source['speaker'] and r['unit']==source['unit']
            and r['mode'] in ('native','melody','short-native')]
        assert len(cases)==3
        for reference in cases:
            score=VowelScore(**reference['score'])
            for method,(unit,hold_mode) in methods.items():
                name=reference['name']+'-'+method
                if method=='baseline-loop':
                    audio,rate=sf.read(ROOT/reference['file']);assert rate==RATE
                    file=ROOT/reference['file'];timing={**reference['streaming'],'reusedMeasurement':True}
                else:
                    audio,timing=controlled_render(unit,decoder,score,
                        stream_factory=partial(NeuralArticulationStream,hold_mode=hold_mode))
                    file=OUT/(name+'.wav');sf.write(file,audio,RATE,subtype='FLOAT')
                actual_start=float(performance.pulses[unit.hold_first])
                modified_source={**source,'holdStartSample':actual_start,
                    'holdEndSample':float(performance.pulses[unit.hold_last])}
                evaluated,tracks=metrics(audio,score,unit,modified_source,decoded,analysis_hold_start=common_hold)
                track=OUT/(name+'.tracks.npz');np.savez(track,**tracks)
                centers=np.arange(int(np.ceil((common_hold+.08)*RATE)),int((score.seconds-.05)*RATE)+1,120)
                features=spectral_frames(audio,centers)
                offsets=np.arange(WINDOW_SIZE)-WINDOW_SIZE//2
                energy=np.sqrt(np.mean(np.array([audio[(center+offsets).astype(int)] for center in centers])**2,axis=1))
                row={'name':name,'speaker':source['speaker'],'unit':source['unit'],'text':source['text'],
                    'mode':reference['mode'],'method':method,'score':score.__dict__,
                    'actualHoldStartSeconds':actual_start/RATE,'commonAnalysisHoldStartSeconds':common_hold,
                    'minimumSustainSeconds':score.seconds-actual_start/RATE,
                    'file':str(file.relative_to(ROOT)),'tracks':str(track.relative_to(ROOT)),
                    'status':'ok','streaming':timing,'audioReused':method=='baseline-loop',
                    'heldVariation':{'frames':len(centers),'shapeRmsDb':shape_variation(features),
                        'energyStdDb':float(np.std(20*np.log10(np.maximum(energy,1e-20)))),
                        'scope':'Fixed-window variation over a common held interval; lower variation is not a naturalness score'},
                    'requestedControlsChanged':False,**evaluated}
                report['results'].append(row);persist()
                print(name,'shape variation',row['heldVariation']['shapeRmsDb'],'Praat',row['praat']['p95ErrorCents'],flush=True)
    for file,sha in hashes.items():
        if hashlib.sha256((ROOT/file).read_bytes()).hexdigest()!=sha:raise AssertionError('Source unit changed')
    if before!=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest():raise AssertionError('Decoder changed')
    selections['complete']=len(selections['units'])==9
    (OUT/'selection-report.json').write_text(json.dumps(selections,indent=2,allow_nan=False)+'\n')
    report['decoderAndSourcesUnchanged']=True
    report['complete']=len(report['results'])==report['expectedCases'] and sum(not r['audioReused'] for r in report['results'])==report['expectedNewRenders']
    persist()


if __name__=='__main__':main()
