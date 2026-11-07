"""Output-detail ablations with the frozen axis-v1 encoder and same 20/4 split.

Projection removes unpredictable target components excluded from the controls.
Optional example weighting emphasizes high-frequency-rich training windows.
Neither method changes input analysis or increases control bandwidth.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import load_model,initialize_axis_decoder,NeuralDecoder,save_model,waveform_frames,SIZE,RATE
from develop_neural_fidelity import snr
from benchmark_source_ceiling import spectral_error

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-waveform/axis-detail-v2'


def main():
    source=ROOT/'evidence/neural-waveform/axis-v1'
    development=json.loads((source/'development.json').read_text())
    baseline_file=source/(development['selected']+'.npz')
    encoder,baseline,_=load_model(baseline_file)
    training=[];validation=[]
    for clip in development['corpus']:
        path=ROOT/clip['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=clip['sha256']:raise ValueError('Changed teacher corpus')
        audio,rate=sf.read(path);assert rate==RATE
        if clip['split']=='training':training.append(waveform_frames(audio)[0])
        else:validation.append((clip['name'],audio,encoder.encode(audio)))
    targets=np.concatenate(training);controls=encoder.encode_frames(targets)
    projected=encoder.linear_frames(controls)
    power=abs(np.fft.rfft(targets,axis=1))**2
    high=np.fft.rfftfreq(SIZE,1/RATE)>4000
    weight=1+4*np.sum(power[:,high],axis=1)/np.maximum(np.sum(power,axis=1),1e-20)
    conditions=[('baseline',False,False,1e-6),('projected-r1e-6',True,False,1e-6),
        ('projected-r1e-7',True,False,1e-7),('projected-r1e-5',True,False,1e-5),
        ('weighted-r1e-6',False,True,1e-6),('projected-weighted-r1e-6',True,True,1e-6)]
    OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Output-only learning objective; identical frozen control representation',
        'baseline':str(baseline_file.relative_to(ROOT)),
        'baselineSha256':hashlib.sha256(baseline_file.read_bytes()).hexdigest(),
        'trainingSplit':'Same 20 teacher clips, 4 development clips; previous final panel excluded from fitting and selection',
        'conditions':conditions,'results':[],'complete':False,
        'selection':'Lowest mean development spectral error among candidates with mean waveform SNR at least baseline and no sample peak >= 2.0'}
    for name,projection,weighted,ridge in conditions:
        started=time.perf_counter()
        if name=='baseline':decoder=baseline
        else:
            learner,_=initialize_axis_decoder(controls,projected if projection else targets,
                ridge=ridge,sample_weight=weight if weighted else None)
            decoder=NeuralDecoder.from_learner(learner)
        row={'name':name,'fitSeconds':time.perf_counter()-started,'projectedTargets':projection,
            'weightedExamples':weighted,'ridge':ridge,'development':[]}
        for fixture,audio,matrix in validation:
            generated=decoder.render(matrix)
            row['development'].append({'fixture':fixture,'waveformSnrDb':snr(audio,generated),
                'spectralErrorDb':spectral_error(audio,generated)['meanAbsoluteDb'],
                'rawPeak':float(abs(generated).max())})
        row['meanDevelopmentSnrDb']=float(np.mean([r['waveformSnrDb'] for r in row['development']]))
        row['meanDevelopmentSpectralErrorDb']=float(np.mean([r['spectralErrorDb'] for r in row['development']]))
        row['maximumDevelopmentPeak']=max(r['rawPeak'] for r in row['development'])
        save_model(OUT/(name+'.npz'),encoder,decoder,{'trainingOrigin':'Same 20 synthetic Kokoro clips; 4 development clips excluded',
            'condition':name,'scope':report['scope'],'globalBackpropagation':False,'localPcRefinement':False})
        report['results'].append(row)
        (OUT/'development.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='development'}),flush=True)
    reference=report['results'][0]
    eligible=[r for r in report['results'] if r['meanDevelopmentSnrDb']>=reference['meanDevelopmentSnrDb']
        and r['maximumDevelopmentPeak']<2.]
    report['selected']=min(eligible,key=lambda r:r['meanDevelopmentSpectralErrorDb'])['name']
    report['complete']=True
    (OUT/'development.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('SELECTED',report['selected'],flush=True)


if __name__=='__main__':main()
