"""Bounded neural fidelity study: more represented detail and decoder capacity.

This deliberately increases an already dense reconstruction representation.
It is not a compact phoneme/voice identity code, compression improvement or
novel-speech model. All conditions use the same 20/4 synthetic teacher split.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import (ControlEncoder,NeuralDecoder,initialize_axis_decoder,
    load_model,save_model,waveform_frames,RATE)
from develop_neural_fidelity import snr
from benchmark_source_ceiling import spectral_error

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-waveform/capacity-v3'


def main():
    prior=json.loads((ROOT/'evidence/neural-waveform/axis-detail-v2/summary.json').read_text())
    baseline=ROOT/prior['candidate']
    if hashlib.sha256(baseline.read_bytes()).hexdigest()!=prior['candidateSha256']:
        raise ValueError('Previous frozen decoder changed')
    old_encoder,old_decoder,_=load_model(baseline)
    corpus=json.loads((ROOT/'evidence/neural-waveform/axis-v1/development.json').read_text())['corpus']
    training=[];development=[]
    for clip in corpus:
        path=ROOT/clip['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=clip['sha256']:raise ValueError('Teacher corpus changed')
        audio,rate=sf.read(path);assert rate==RATE
        if clip['split']=='training':training.append(waveform_frames(audio)[0])
        else:development.append((clip['name'],audio))
    targets=np.concatenate(training)
    conditions=[('baseline-v2',384,5,.8,1e-7),
        ('d416-k5-r1e-7',416,5,.8,1e-7),('d448-k5-r1e-7',448,5,.8,1e-7),
        ('d448-k5-r1e-8',448,5,.8,1e-8),('d448-k7-r1e-7',448,7,.8,1e-7),
        ('d384-k7-r1e-7',384,7,.8,1e-7)]
    OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Output waveform fidelity from target-derived controls; no new speech or compact articulated voice code',
        'baseline':prior['candidate'],'baselineSha256':prior['candidateSha256'],
        'corpus':corpus,'conditions':conditions,'results':[],'complete':False,
        'training':'Same 20 Kokoro teacher clips, four development clips; no global backpropagation or local-PC refinement',
        'selection':'Minimum mean development spectral error, subject to no worse mean waveform SNR than baseline, peak <0.98, max decoder RTF <0.10',
        'controlCost':'Every 10 ms carries one gain plus 384, 416 or 448 continuous coefficients; this is denser than the original 24 kHz waveform, not compression'}
    encoders={384:old_encoder}
    for name,dims,centers,slope,ridge in conditions:
        row={'name':name,'dimensions':dims,'centers':centers,'slope':slope,'ridge':ridge,
            'controlScalarsPerSecond':(dims+1)*100,'development':[]}
        try:
            started=time.perf_counter()
            if dims not in encoders:encoders[dims]=ControlEncoder.fit(targets,dimensions=dims)
            encoder=encoders[dims]
            if name=='baseline-v2':decoder=old_decoder
            else:
                controls=encoder.encode_frames(targets)
                projected=encoder.linear_frames(controls)
                learner,_=initialize_axis_decoder(controls,projected,
                    centers=np.linspace(-3,3,centers),slope=slope,ridge=ridge)
                decoder=NeuralDecoder.from_learner(learner)
            row['fitAndEncoderSeconds']=time.perf_counter()-started
            row['decoderParameterCount']=decoder.W.size+decoder.V.size
            for fixture,audio in development:
                matrix=encoder.encode(audio)
                started=time.perf_counter();generated=decoder.render(matrix);elapsed=time.perf_counter()-started
                row['development'].append({'fixture':fixture,'waveformSnrDb':snr(audio,generated),
                    'spectralErrorDb':spectral_error(audio,generated)['meanAbsoluteDb'],
                    'linearReferenceSnrDb':snr(audio,encoder.reconstruct_linear(matrix)),
                    'decoderRealTimeFactor':elapsed/(len(audio)/RATE),'rawPeak':float(abs(generated).max())})
            row.update(status='ok',meanDevelopmentSnrDb=float(np.mean([r['waveformSnrDb'] for r in row['development']])),
                meanDevelopmentSpectralErrorDb=float(np.mean([r['spectralErrorDb'] for r in row['development']])),
                maximumDevelopmentPeak=max(r['rawPeak'] for r in row['development']),
                maximumDecoderRealTimeFactor=max(r['decoderRealTimeFactor'] for r in row['development']))
            path=OUT/(name+'.npz')
            save_model(path,encoder,decoder,{'trainingOrigin':'Same 20 synthetic Kokoro clips; four development clips excluded',
                'scope':report['scope'],'condition':name,'globalBackpropagation':False,'localPcRefinement':False,
                'controlScalarsPerSecond':row['controlScalarsPerSecond']})
            row['modelSha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        except (ValueError,np.linalg.LinAlgError) as error:
            row.update(status='failed',error=str(error))
        report['results'].append(row)
        (OUT/'development.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='development'}),flush=True)
    reference=report['results'][0]
    eligible=[r for r in report['results'] if r['status']=='ok'
        and r['meanDevelopmentSnrDb']>=reference['meanDevelopmentSnrDb']
        and r['maximumDevelopmentPeak']<.98 and r['maximumDecoderRealTimeFactor']<.10]
    report['selected']=min(eligible,key=lambda r:r['meanDevelopmentSpectralErrorDb'])['name']
    report['complete']=True
    (OUT/'development.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('SELECTED',report['selected'],flush=True)


if __name__=='__main__':main()
