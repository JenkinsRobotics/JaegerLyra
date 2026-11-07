"""Locate output reconstruction error; offline diagnostics, not an input model."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import load_model, SIZE, HOP, RATE
from benchmark_voicebox import f0_track

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-waveform/axis-v1/error-analysis.json'


def frames(audio):
    padded=np.pad(audio,(HOP,HOP+(-len(audio))%HOP))
    return np.lib.stride_tricks.sliding_window_view(padded,SIZE)[::HOP]


def main():
    selection=json.loads((ROOT/'evidence/neural-waveform/axis-v1/development.json').read_text())
    encoder,decoder,_=load_model(ROOT/'evidence/neural-waveform/axis-v1'/(selection['selected']+'.npz'))
    sources=[{**r,'kind':'development'} for r in selection['corpus'] if r['split']=='development']
    sources += [{**r,'kind':'reused-final-regression'} for r in json.loads(
        (ROOT/'evidence/neural-waveform/output-validation/manifest.json').read_text())['clips']]
    report={'scope':'Offline location of reconstruction loss; no new input-understanding model',
        'dataUse':'Development clips and explicitly reused previous final panel; not fresh validation',
        'bandsHz':[[0,1000],[1000,4000],[4000,8000],[8000,12001]],'results':[]}
    frequency=np.fft.rfftfreq(SIZE,1/RATE);window=np.hanning(SIZE)
    for case in sources:
        original,rate=sf.read(ROOT/case['file']);assert rate==RATE
        matrix=encoder.encode(original);linear=encoder.reconstruct_linear(matrix);neural=decoder.render(matrix)
        truth=frames(original);rms=np.sqrt(np.mean(truth**2,axis=1))
        times=np.arange(len(truth))*HOP/RATE
        pitch_times,pitch=f0_track(original);voiced=np.interp(times,pitch_times,(pitch>0).astype(float))>.5
        active=rms>.003
        masks={'active-voiced':active&voiced,'active-unvoiced':active&~voiced,
               'quiet':(rms>1e-6)&~active,'near-silent':rms<=1e-6,
               'boundaries':(times<=.02)|(times>=len(original)/RATE-.02)}
        row={'name':case['name'],'kind':case['kind'],'components':[]}
        for name,error in [('representation',original-linear),('decoder',linear-neural),('total',original-neural)]:
            windows=frames(error);power=np.abs(np.fft.rfft(windows*window,axis=1))**2
            power[:,1:-1]*=2
            denominator=float(np.sum(power));categories=[]
            for category,mask in masks.items():
                count=int(np.sum(mask));error_power=float(np.sum(windows[mask]**2))
                signal_power=float(np.sum(truth[mask]**2))
                categories.append({'category':category,'frames':count,
                    'snrDb':float(10*np.log10(max(signal_power,1e-30)/max(error_power,1e-30))) if signal_power>1e-12 else None,
                    'errorRmsDbfs':float(10*np.log10(max(error_power/max(count*SIZE,1),1e-30))),
                    'errorEnergy':error_power})
            shares=[float(np.sum(power[:,(frequency>=low)&(frequency<high)])/max(denominator,1e-30))
                for low,high in report['bandsHz']]
            row['components'].append({'component':name,'bandErrorEnergyShare':shares,'categories':categories})
        report['results'].append(row)
    OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for kind in ('development','reused-final-regression'):
        items=[r for r in report['results'] if r['kind']==kind]
        for component in ('representation','decoder','total'):
            c=[next(c for c in r['components'] if c['component']==component) for r in items]
            print(json.dumps({'kind':kind,'component':component,
                'meanBandErrorShare':np.mean([x['bandErrorEnergyShare'] for x in c],axis=0).tolist(),
                'categoryErrorEnergy':{name:sum(next(v['errorEnergy'] for v in x['categories'] if v['category']==name) for x in c)
                    for name in ('active-voiced','active-unvoiced','quiet','near-silent')}}),flush=True)


if __name__=='__main__':main()
