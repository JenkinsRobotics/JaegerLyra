"""Capture source syllables and select their late stable voiced sustain region."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import load_model,RATE
from neural_performance import save_performance
from neural_articulation import compile_unit
from benchmark_neural_vowels import formants

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-articulation/v1'


def main():
    refs=json.loads((ROOT/'evidence/neural-articulation/references/manifest.json').read_text())
    prior=json.loads((ROOT/'evidence/neural-vowels/v1/report.json').read_text())
    if not refs['complete']:raise ValueError('Incomplete source panel')
    model=ROOT/prior['model']
    if hashlib.sha256(model.read_bytes()).hexdigest()!=prior['modelSha256']:raise ValueError('Changed decoder')
    encoder,decoder,_=load_model(model)
    OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Source-derived captured syllables with an automatically selected late voiced sustain',
        'sourceManifest':'evidence/neural-articulation/references/manifest.json',
        'model':prior['model'],'modelSha256':prior['modelSha256'],'units':[],'complete':False,
        'selection':'Trim only quiet edges using 5 ms RMS at 0.003 of peak RMS, retain 10 ms padding; choose stable 60–150 ms voiced span after 35% of source active extent. No output-score selection.',
        'limits':['Source-only boundaries and intended phone labels are not forced alignment or recognized phonemes.',
            'All unit audio is synthetic Kokoro teacher material; no new neural weights or independent speech model.']}
    for source in refs['clips']:
        file=ROOT/source['file']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Changed source')
        audio,rate=sf.read(file);assert rate==RATE
        rms=np.sqrt(np.convolve(audio**2,np.ones(120)/120,mode='same'))
        active=np.flatnonzero(rms>max(1e-5,.003*max(rms)))
        if not len(active):raise ValueError('Silent source '+source['name'])
        first=max(0,int(active[0])-240);last=min(len(audio),int(active[-1])+241)
        trimmed=audio[first:last];unit,meta=compile_unit(trimmed,encoder)
        wav=OUT/(source['name']+'-trimmed.wav');sf.write(wav,trimmed,RATE,subtype='FLOAT')
        artifact=OUT/(source['name']+'-unit.npz')
        save_performance(artifact,unit.performance,{**meta,'source':source['file'],'sourceSha256':source['sha256'],
            'trimStartSample':first,'trimEndSampleExclusive':last,'neuralModelSha256':prior['modelSha256']})
        decoded=decoder.render(unit.performance.matrix)
        neural_wav=OUT/(source['name']+'-decoded-source.wav');sf.write(neural_wav,decoded,RATE,subtype='FLOAT')
        region=trimmed[int(meta['holdStartSample']):int(meta['holdEndSample'])]
        row={**source,**meta,'artifact':str(artifact.relative_to(ROOT)),
            'artifactSha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
            'trimmedSource':str(wav.relative_to(ROOT)),'decodedSource':str(neural_wav.relative_to(ROOT)),
            'trimStartSample':first,'trimEndSampleExclusive':last,
            'sourceFormantsHz':formants(region),
            'neuralSourceReconstructionSnrDb':float(10*np.log10(np.sum(trimmed**2)/max(np.sum((trimmed-decoded)**2),1e-30)))}
        report['units'].append(row)
        (OUT/'capture-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(source['name'],'hold at',round(meta['holdStartSample']/RATE,3),'Hz',round(meta['sourceMedianPitchHz'],1),flush=True)
    report['complete']=len(report['units'])==9
    (OUT/'capture-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
