"""Fit the first shared neural waveform decoder, without global backpropagation.

Batch PCA/least-squares initialization is explicitly separate from a subsequent
chronological pass of local prediction-error updates. No evaluation audio fits
either the encoder or decoder. Decoder evaluation must use forward synthesis.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import (ControlEncoder,NeuralDecoder,initialize_decoder,save_model,
    waveform_frames,SIZE,HOP,RATE)

ROOT=Path(__file__).resolve().parents[2]


def forward_mse(learner,controls,targets):
    total=0.;count=0
    for at in range(0,len(controls),256):
        x=np.c_[controls[at:at+256],np.ones(min(256,len(controls)-at))]*learner.config.input_gain
        prediction=np.tanh(x@learner.W.T)@learner.V.T/learner.config.input_gain
        total+=float(np.sum((prediction-targets[at:at+256])**2));count+=prediction.size
    return total/count


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dimensions',type=int,default=256)
    parser.add_argument('--hidden',type=int,default=768);parser.add_argument('--pc-epochs',type=int,default=1)
    args=parser.parse_args()
    out=ROOT/f'evidence/neural-waveform/d{args.dimensions}-h{args.hidden}'
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'evidence/neural-waveform/training-inputs/manifest.json').read_text())
    if len(manifest['clips'])!=24:raise ValueError('Training corpus is incomplete')
    frames=[];records=[];seconds=0.
    for clip in manifest['clips']:
        path=ROOT/clip['file'];checksum=hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum!=clip['sha256']:raise ValueError('Training input changed')
        audio,rate=sf.read(path);assert rate==RATE
        unit,gains=waveform_frames(audio);frames.append(unit);seconds+=len(audio)/rate
        records.append({'file':clip['file'],'sha256':checksum,'frames':len(unit),'durationSeconds':len(audio)/rate})
    targets=np.concatenate(frames)
    started=time.perf_counter();encoder=ControlEncoder.fit(targets,dimensions=args.dimensions)
    controls=encoder.encode_frames(targets);fit_seconds=time.perf_counter()-started
    ceiling=encoder.linear_frames(controls)
    representation_mse=float(np.mean((ceiling-targets)**2));del ceiling
    print('ENCODER',len(targets),'frames',round(fit_seconds,3),'seconds',flush=True)
    started=time.perf_counter();learner,penalty=initialize_decoder(controls,targets,hidden=args.hidden)
    initialize_seconds=time.perf_counter()-started
    report={'scope':'Shared neural waveform reconstruction decoder; not new-speech generation or disentangled phonotype/identity controls',
        'trainingOrigin':'Explicit synthetic Kokoro teacher audio; not an independently sourced training corpus or clean-room weights',
        'training':records,'trainingSeconds':seconds,'trainingFrames':len(targets),
        'waveformRate':RATE,'windowSamples':SIZE,'hopSamples':HOP,
        'continuousControls':args.dimensions,'gainControls':1,
        'scalarControlsPerSecond':(args.dimensions+1)*RATE/HOP,
        'controlInterpretation':'PCA acoustic coordinates with real-valued amplitude companding; not categorical phonemes or audio codes',
        'compressionClaim':False,'decoderOriginalPcmBypass':False,'globalBackpropagation':False,
        'initialization':'Fixed random tanh hidden features and regularized least-squares output fit',
        'refinement':'Existing explicit local predictive-coding updates; symmetric feedback, no autograd',
        'encoderFitSeconds':fit_seconds,'decoderInitializationSeconds':initialize_seconds,
        'ridgePenalty':penalty,'config':learner.config.__dict__,
        'trainingRepresentationMse':representation_mse,
        'ridgeTrainingForwardMse':forward_mse(learner,controls,targets),'epochs':[]}
    save_model(out/'neural-ridge.npz',encoder,NeuralDecoder.from_learner(learner),
               {'mode':'neural-ridge','scope':report['scope'],'trainingOrigin':report['trainingOrigin']})
    (out/'training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('RIDGE',round(initialize_seconds,3),'seconds','forward mse',report['ridgeTrainingForwardMse'],flush=True)
    for epoch in range(args.pc_epochs):
        started=time.perf_counter();increases=0;maximum_increase=0.
        for index,(control,target) in enumerate(zip(controls,targets)):
            observed=learner.observe(np.r_[control,1.],target,learn=True)
            increase=observed['finalEnergy']-observed['initialEnergy']
            maximum_increase=max(maximum_increase,increase);increases+=int(increase>1e-10)
            if (index+1)%1000==0:print('PC',epoch+1,index+1,'/',len(targets),flush=True)
        row={'epoch':epoch+1,'seconds':time.perf_counter()-started,
            'forwardTrainingMse':forward_mse(learner,controls,targets),
            'settlingEnergyIncreases':increases,'maximumSettlingEnergyIncrease':maximum_increase}
        report['epochs'].append(row)
        save_model(out/'neural-local-pc.npz',encoder,NeuralDecoder.from_learner(learner),
                   {'mode':'neural-local-pc','epochs':epoch+1,'scope':report['scope'],'trainingOrigin':report['trainingOrigin']})
        (out/'training-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
