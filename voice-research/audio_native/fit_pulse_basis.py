"""Capture normalized pulse templates with complete source segments held out."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import load_model,RATE,SIZE as NEURAL_SIZE
from neural_performance import compile_performance
from compact_vocal_instrument import load_bank
from pulse_basis_instrument import (RADIUS,OFFSETS,pulse_window,fit_pulses,reconstruction_metrics,save_models)

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/pulse-basis/v1'


def capture_track(audio,pulses,segment,period,cut):
    window=pulse_window(period);rows=[];centers=[];rejected=0
    for center in pulses:
        if center-RADIUS<segment['sourceStartSample'] or center+RADIUS>=segment['sourceEndSampleExclusive']:continue
        # Fixed decoder analysis/overlap context cannot cross the held split.
        if ((segment['split']=='train' and center+RADIUS+NEURAL_SIZE>cut)
                or (segment['split']=='validation' and center-RADIUS-NEURAL_SIZE<cut)):
            rejected+=1;continue
        grain=np.interp(center+OFFSETS,np.arange(len(audio)),audio)*window
        rms=np.sqrt(np.sum(grain**2)/np.sum(window**2))
        if rms<.003:rejected+=1;continue
        rows.append(grain/rms);centers.append(center)
    return np.array(rows),np.array(centers),rejected


def main():
    previous=json.loads((ROOT/'evidence/generated-texture/v1/fit-report.json').read_text())
    vowels=json.loads((ROOT/'evidence/neural-vowels/v1/report.json').read_text())
    compact=json.loads((ROOT/'evidence/compact-vocal/v1/report.json').read_text())
    if not previous['complete']:raise ValueError('Incomplete source-split study')
    checkpoint=ROOT/previous['decoder']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=previous['decoderSha256']:raise ValueError('Changed neural decoder')
    encoder,decoder,_=load_model(checkpoint)
    OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Captured waveform-template basis and pulse-rate stochastic state; not a neural vocoder or phoneme model',
        'decoder':previous['decoder'],'decoderSha256':previous['decoderSha256'],
        'sourceSplitReport':'evidence/generated-texture/v1/fit-report.json',
        'radiusSamples':RADIUS,'outputDelayMs':1000*RADIUS/RATE,'modes':8,
        'sources':[],'banks':[],'complete':False,
        'limits':['Mean and principal directions contain captured waveform samples explicitly.',
            'Template projection uses coefficients encoded from held pulses; it is a representation-capacity test, not generated output quality.',
            'Pulse-state dynamics operate at the requested output pulse rate, not at a fixed physical source rate.',
            'No additional breath control, new words or general voice cloning is demonstrated.']}
    def persist():(OUT/'fit-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for speaker in ('af_heart','am_michael','af_bella'):
        source_bank=next(p for p in compact['profiles'] if p['speaker']==speaker)
        shapes,_=load_bank(ROOT/source_bank['bank']);models=[]
        for i,vowel in enumerate(('ah','ee','oo')):
            name=speaker+'-'+vowel;source=next(r for r in previous['sources'] if r['name']==name)
            ref=next(r for r in vowels['references'] if r['name']==name)
            file=ROOT/source['source']
            if hashlib.sha256(file.read_bytes()).hexdigest()!=source['sourceSha256']:raise ValueError('Changed source')
            audio,rate=sf.read(file);assert rate==RATE
            performance=compile_performance(audio,encoder);decoded=decoder.render(performance.matrix)
            period=RATE/ref['sourceMedianPitchHz']
            train_segments=[d for d in source['analysisSegments'] if d['split']=='train']
            held_segments=[d for d in source['analysisSegments'] if d['split']=='validation']
            cut=(max(d['sourceEndSampleExclusive'] for d in train_segments)+min(d['sourceStartSample'] for d in held_segments))/2
            training=[];held=[];details=[];stored={}
            for j,segment in enumerate(source['analysisSegments']):
                track,centers,rejected=capture_track(decoded,performance.pulses,segment,period,cut)
                details.append({**segment,'capturedPulses':len(track),'rejectedPulses':rejected,
                    'used':len(track)>=2,'firstCenter':float(centers[0]) if len(centers) else None,
                    'lastCenter':float(centers[-1]) if len(centers) else None})
                if len(track)<2:continue
                (training if segment['split']=='train' else held).append(track)
                stored['segment_'+str(j)]=track;stored['centers_'+str(j)]=centers
            if not held:raise ValueError('No usable held pulse tracks for '+name)
            model,fit=fit_pulses(training,period,shapes[i].rms);models.append(model)
            artifact=OUT/(name+'-capture.npz');np.savez(artifact,**stored)
            row={'name':name,'speaker':speaker,'source':source['source'],'sourceSha256':source['sourceSha256'],
                'referencePeriodSamples':period,'referenceRms':shapes[i].rms,'splitSample':cut,'decoderGuardSamples':NEURAL_SIZE,
                'segments':details,'fit':fit,'training':reconstruction_metrics(model,training),
                'validation':reconstruction_metrics(model,held),'capture':str(artifact.relative_to(ROOT))}
            report['sources'].append(row);persist()
            print(name,'held',row['validation'],flush=True)
        bank=OUT/(speaker+'-pulses.npz');save_models(bank,models,{'speaker':speaker,'vowels':['ah','ee','oo'],
            'fitReport':'evidence/pulse-basis/v1/fit-report.json','waveformTemplates':True,
            'sourceSequencesRetained':False,'outputDelaySamples':RADIUS})
        report['banks'].append({'speaker':speaker,'file':str(bank.relative_to(ROOT)),
            'sha256':hashlib.sha256(bank.read_bytes()).hexdigest()});persist()
    report['complete']=len(report['sources'])==9;persist()


if __name__=='__main__':main()
