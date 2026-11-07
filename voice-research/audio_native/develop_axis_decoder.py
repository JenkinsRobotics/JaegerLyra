"""Structured-neuron output capacity study, selected on development audio only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import ControlEncoder,NeuralDecoder,initialize_axis_decoder,save_model,waveform_frames
from develop_neural_fidelity import snr

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-waveform/axis-v1'


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    corpus=json.loads((ROOT/'evidence/neural-waveform/training-inputs/manifest.json').read_text())['clips']
    train=[];dev=[];records=[]
    for clip in corpus:
        path=ROOT/clip['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=clip['sha256']:raise ValueError('Changed teacher corpus')
        audio,rate=sf.read(path)
        if rate!=24000:raise ValueError('Unexpected sample rate')
        is_dev=clip['name'].endswith(('-10','-11'))
        if is_dev:dev.append((clip['name'],audio))
        else:train.append(waveform_frames(audio)[0])
        records.append({**clip,'split':'development' if is_dev else 'training'})
    targets=np.concatenate(train)
    conditions=[(256,5,.8,1e-6),(256,9,1.,1e-7),(384,5,.8,1e-6),(384,9,1.,1e-7)]
    report={'scope':'Neural reconstruction from encoded target coordinates; no independent note/phoneme controls',
        'initialization':'Coordinate-specific tanh units; batch ridge-fitted output weights; no global backpropagation',
        'corpus':records,'conditions':conditions,'results':[],'complete':False,
        'selection':'Highest mean development waveform SNR, with all conditions retained'}
    encoders={}
    for dims,count,slope,ridge in conditions:
        if dims not in encoders:
            encoder=ControlEncoder.fit(targets,dimensions=dims)
            encoders[dims]=(encoder,encoder.encode_frames(targets),[(n,a,encoder.encode(a)) for n,a in dev])
        encoder,controls,validation=encoders[dims]
        name=f'd{dims}-k{count}-s{slope}-r{ridge}';started=time.perf_counter()
        learner,penalty=initialize_axis_decoder(controls,targets,centers=np.linspace(-3,3,count),slope=slope,ridge=ridge)
        decoder=NeuralDecoder.from_learner(learner)
        row={'name':name,'dimensions':dims,'centers':count,'slope':slope,'ridge':ridge,
            'fitSeconds':time.perf_counter()-started,'ridgePenalty':penalty,'development':[]}
        for fixture,audio,matrix in validation:
            started=time.perf_counter();generated=decoder.render(matrix);elapsed=time.perf_counter()-started
            row['development'].append({'fixture':fixture,'waveformSnrDb':snr(audio,generated),
                'linearReferenceSnrDb':snr(audio,encoder.reconstruct_linear(matrix)),
                'rawPeak':float(abs(generated).max()),'decoderRealTimeFactor':elapsed/(len(audio)/24000)})
        row['meanDevelopmentSnrDb']=float(np.mean([r['waveformSnrDb'] for r in row['development']]))
        save_model(OUT/(name+'.npz'),encoder,decoder,{'trainingOrigin':'20 synthetic Kokoro clips; 4 development clips excluded',
            'condition':name,'scope':report['scope'],'globalBackpropagation':False,'localPcRefinement':False})
        report['results'].append(row)
        (OUT/'development.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='development'}),flush=True)
        del learner,decoder
    report['selected']=max(report['results'],key=lambda r:r['meanDevelopmentSnrDb'])['name'];report['complete']=True
    (OUT/'development.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('SELECTED',report['selected'],flush=True)


if __name__=='__main__':main()
