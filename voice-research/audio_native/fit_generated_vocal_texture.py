"""Fit and validate output texture states on separate voiced source segments."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import load_model,RATE
from neural_performance import compile_performance
from generated_vocal_texture import fit_model,forecast_metrics,save_models,FEATURES

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/generated-texture/v1'
WINDOW=1024
HOP=120
FFT=2048


def extract_tracks(audio,pulses):
    tracks=[];details=[];rejected=[]
    groups=np.split(pulses,np.flatnonzero(np.diff(pulses)>RATE*.020)+1)
    window=np.hanning(WINDOW)
    for index,group in enumerate(groups):
        if len(group)<2:continue
        start=int(np.ceil(group[0]/HOP))*HOP;stop=int(np.floor(group[-1]))
        offsets=np.arange(start,stop-WINDOW+1,HOP,dtype=int)
        if len(offsets)<4:
            rejected.append({'group':index,'reason':'Fewer than four complete analysis windows','frames':len(offsets)})
            continue
        frames=audio[offsets[:,None]+np.arange(WINDOW)]
        rms=np.sqrt(np.mean(frames**2,axis=1))
        if np.median(rms)<.003:
            rejected.append({'group':index,'reason':'Median source RMS below .003','frames':len(offsets)})
            continue
        magnitude=abs(np.fft.rfft(frames*window,n=FFT,axis=1))
        log=np.log(np.maximum(magnitude,magnitude.max(axis=1,keepdims=True)*10**(-70/20)))
        features=np.fft.irfft(log,n=FFT,axis=1)[:,1:FEATURES+1]
        tracks.append(features)
        details.append({'group':index,'sourceStartSample':int(offsets[0]),
            'sourceEndSampleExclusive':int(offsets[-1]+WINDOW),'frames':len(features),
            'sourceMedianRms':float(np.median(rms))})
    return tracks,details,rejected


def main():
    manifest=json.loads((ROOT/'evidence/generated-texture/references/manifest.json').read_text())
    if not manifest['complete']:raise ValueError('Incomplete source panel')
    checkpoint=ROOT/manifest['decoder']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=manifest['decoderSha256']:raise ValueError('Changed decoder')
    encoder,decoder,_=load_model(checkpoint)
    before=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest()
    OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Output-only autonomous stochastic texture state; offline neural-decoded synthetic references',
        'decoder':manifest['decoder'],'decoderSha256':manifest['decoderSha256'],
        'analysis':{'windowSamples':WINDOW,'hopSamples':HOP,'fftSize':FFT,'cepstralFeatures':FEATURES,
            'feature':'Real log-amplitude cepstrum, coefficients 1–31; c0 omitted; relative magnitude floor -70 dB',
            'split':'First two thirds of complete eligible voiced segments train; final third validates. Every analysis window stays wholly inside its segment.',
            'sourceFilters':'At least four complete windows and median RMS >= .003; no output-based filtering'},
        'sources':[],'banks':[],'complete':False}
    def persist():(OUT/'fit-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for speaker in ('af_heart','am_michael','af_bella'):
        models=[]
        for source in [c for c in manifest['clips'] if c['speaker']==speaker]:
            file=ROOT/source['file']
            if hashlib.sha256(file.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Changed source audio')
            audio,rate=sf.read(file);assert rate==RATE
            performance=compile_performance(audio,encoder);decoded=decoder.render(performance.matrix)
            tracks,details,rejected=extract_tracks(decoded,performance.pulses)
            if len(tracks)<3:raise ValueError('Need at least three complete eligible voiced segments: '+source['name'])
            split=min(len(tracks)-1,int(np.ceil(len(tracks)*2/3)))
            model,diagnostic=fit_model(tracks[:split]);models.append(model)
            validation=forecast_metrics(model,tracks[split:])
            training=forecast_metrics(model,tracks[:split])
            for i,detail in enumerate(details):detail['split']='train' if i<split else 'validation'
            if max(d['sourceEndSampleExclusive'] for d in details[:split])>min(d['sourceStartSample'] for d in details[split:]):
                raise AssertionError('Analysis-support overlap across split')
            trackfile=OUT/(source['name']+'-tracks.npz')
            np.savez(trackfile,**{'segment_'+str(i):t for i,t in enumerate(tracks)})
            row={'name':source['name'],'speaker':speaker,'source':source['file'],'sourceSha256':source['sha256'],
                'sourceSeconds':len(audio)/RATE,'analysisSegments':details,'rejectedSegments':rejected,
                'fit':diagnostic,'validation':validation,'training':training,'tracks':str(trackfile.relative_to(ROOT)),
                'neuralReconstructionSnrDb':float(10*np.log10(np.sum(audio**2)/np.sum((audio-decoded)**2)))}
            report['sources'].append(row);persist()
            print(source['name'],'segments',len(tracks),'train',split,'held',len(tracks)-split,'RMSE',validation,flush=True)
        bank=OUT/(speaker+'-dynamics.npz')
        save_models(bank,models,{'speaker':speaker,'vowels':['ah','ee','oo'],
            'sourceManifest':'evidence/generated-texture/references/manifest.json','decoderSha256':manifest['decoderSha256'],
            'runtime':'Fresh latent innovations only; no source trajectory, PCM, pulse times or target audio'})
        report['banks'].append({'speaker':speaker,'file':str(bank.relative_to(ROOT)),
            'sha256':hashlib.sha256(bank.read_bytes()).hexdigest()});persist()
    if before!=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest():raise AssertionError('Neural weights changed')
    report['neuralWeightsUnchanged']=True;report['complete']=len(report['sources'])==9;persist()


if __name__=='__main__':main()
