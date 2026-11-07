"""Does locally varying harmonic capture predict unused source cycles better?

This is an offline diagnostic with neighbouring even source cycles available.
It is not a causal synthesizer, generated microvariation or a new acceptance
score. Odd cycles are never fitting targets for either compared model.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
import statistics as stats
from pathlib import Path
import numpy as np
from neural_waveform import load_model,RATE
from neural_performance import load_performance

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/captured-harmonic/v1'


def main():
    study=json.loads((OUT/'report.json').read_text())
    if not study['complete']:raise ValueError('Complete harmonic study first')
    model=ROOT/study['neuralModel']
    if hashlib.sha256(model.read_bytes()).hexdigest()!=study['neuralModelSha256']:raise ValueError('Changed decoder')
    _,decoder,_=load_model(model)
    rows=[]
    for speaker in study['profiles']:
        for captured in speaker['capture']:
            patch,_=load_performance(ROOT/captured['sourcePatch']);audio=decoder.render(patch.matrix)
            pulses=patch.pulses;count=captured['harmonics']
            index=np.arange(int(np.ceil(pulses[0])),int(np.ceil(pulses[-1])))
            phase=np.interp(index,pulses,np.arange(len(pulses)));cycles=np.floor(phase).astype(int)
            angles=2*np.pi*phase[:,None]*np.arange(1,count+1)
            basis=np.column_stack((np.ones(len(index)),np.cos(angles),np.sin(angles)))
            train=cycles%2==0;target=audio[index]
            global_coef=np.linalg.lstsq(basis[train],target[train],rcond=1e-10)[0]
            global_output=basis@global_coef;local_output=np.zeros(len(index))
            coefficients=[];cycle_results=[]
            for cycle in np.unique(cycles[~train]):
                distance=phase-(cycle+.5)
                eligible=train&(abs(distance)<=4.5)
                if sum(eligible)<basis.shape[1]:raise ValueError('Insufficient neighbouring training samples')
                weights=np.exp(-.5*(distance[eligible]/2.)**2)
                coef=np.linalg.lstsq(basis[eligible]*np.sqrt(weights[:,None]),
                    target[eligible]*np.sqrt(weights),rcond=1e-10)[0]
                mask=cycles==cycle;local_output[mask]=basis[mask]@coef
                coefficients.append(coef)
                def score(output):return float(10*np.log10(np.sum(target[mask]**2)/max(np.sum((target[mask]-output[mask])**2),1e-30)))
                cycle_results.append({'cycle':int(cycle),'fitSamples':int(sum(eligible)),
                    'heldSamples':int(sum(mask)),'staticSnrDb':score(global_output),'localSnrDb':score(local_output)})
            def held_snr(output):return float(10*np.log10(np.sum(target[~train]**2)/max(np.sum((target[~train]-output[~train])**2),1e-30)))
            a,b=held_snr(global_output),held_snr(local_output)
            if abs(a-captured['heldCycleSnrDb'])>1e-7:raise AssertionError('Static baseline changed')
            coef=np.array(coefficients)
            harmonic=coef[:,1:count+1]-1j*coef[:,count+1:]
            rms_amplitude=np.sqrt(np.sum(abs(harmonic)**2,axis=1)/2)
            row={'source':captured['source'],'heldSamples':int(sum(~train)),'staticHeldSnrDb':a,
                'localHeldSnrDb':b,'changeDb':b-a,'localHarmonicRmsCv':float(np.std(rms_amplitude)/np.mean(rms_amplitude)),
                'cycles':cycle_results}
            rows.append(row);print(row['source'],'static/local',round(a,2),round(b,2),flush=True)
    report={'scope':__doc__,'complete':len(rows)==9,'sourceModelSha256':study['neuralModelSha256'],
        'localWindow':{'gaussianSigmaCycles':2.,'maximumDistanceCycles':4.5,'selectedUsingOutputScores':False},
        'cases':rows,'medianStaticHeldSnrDb':stats.median(r['staticHeldSnrDb'] for r in rows),
        'medianLocalHeldSnrDb':stats.median(r['localHeldSnrDb'] for r in rows),
        'medianPairedChangeDb':stats.median(r['changeDb'] for r in rows),
        'improvedCases':sum(r['changeDb']>0 for r in rows),
        'limitations':['Local fits have more parameters and use surrounding source cycles; no fair-compute or compactness advantage is asserted.',
            'The diagnostic cannot identify all residual error as natural variation: phase-track error, noise and analysis assumptions remain.',
            'Neighbouring source access is offline; these are not independently generated output controls.',
            'No perceptual voice-quality or identity acceptance.']}
    (OUT/'dynamics-diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
