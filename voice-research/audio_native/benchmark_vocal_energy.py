"""Separate voice fidelity from forecasting while testing output-energy control.

All synthesis controls are observed in this comparison. It cannot establish
generation without input, gap prediction or natural speech learned from scratch.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ['HF_HUB_OFFLINE']='1'
import json
import argparse
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import torch,torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from continuous_source import ContinuousSource
from predictive_controls import ControlSpace
from benchmark_free_running import analyze,render,save_audio,spectral_db
from benchmark_voicebox import errors,transcribe,f0_track

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vocal-energy'


def envelope(audio):
    size=960;hop=240
    padded=np.pad(audio,(size//2,size//2))
    windows=np.lib.stride_tricks.sliding_window_view(padded,size)[::hop]
    return np.sqrt(np.mean(windows*windows,axis=1))


def envelope_error(original,generated):
    a=envelope(original);b=envelope(generated)
    active=a>.003
    db=20*np.log10(np.maximum(b[active],1e-6)/a[active])
    return {'meanAbsoluteDb':float(np.mean(np.abs(db))) if len(db) else None,
            'medianSignedDb':float(np.median(db)) if len(db) else None,
            'activeFrames':int(active.sum()),
            'quietMeanGeneratedRms':float(np.mean(b[~active])) if (~active).any() else None,
            'maximumGeneratedRms':float(b.max()),
            'originalMaximumRms':float(a.max())}


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--lattice',action='store_true')
    args=parser.parse_args()
    filter_mode='normalized-lattice' if args.lattice else 'direct'
    preparation=0.
    if args.lattice:
        OUT=OUT/'normalized-lattice'
        from lattice_filter import normalized_lattice
        started=time.perf_counter()
        normalized_lattice(np.zeros(1),np.zeros(32),np.zeros(32))
        preparation=(time.perf_counter()-started)*1000
    OUT.mkdir(exist_ok=True,parents=True);torch.set_num_threads(4)
    references=[]
    for voice in ('af_heart','am_michael'):
        audio,rate=sf.read(ROOT/f'evidence/voice-fixtures/{voice}-reference.wav');assert rate==24000
        references+=analyze(audio)
    source,count=ContinuousSource.fit(references,dimensions=48,harmonics=64)
    modeldir=ROOT/'models/speaker-eval'
    evaluator=EncoderClassifier.from_hparams(source=str(modeldir),savedir=str(modeldir),
        overrides={'pretrained_path':str(modeldir)},run_opts={'device':'cpu'})
    def embedding(audio):
        value=torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        value=torchaudio.functional.resample(value,24000,16000)
        with torch.inference_mode():value=evaluator.encode_batch(value).flatten()
        return value/torch.linalg.vector_norm(value)
    cases=[]
    for voice in ('af_heart','am_michael'):
        path=ROOT/f'evidence/voice-fixtures/{voice}-heldout.wav'
        cases.append({'name':voice,'speaker':voice,'file':str(path.relative_to(ROOT)),
                      'text':json.loads(path.with_suffix('.json').read_text())['text']})
    cases+=json.loads((ROOT/'evidence/streaming-voicebox/validation/manifest.json').read_text())['clips']
    anchors={}
    report={'scope':'Observed-control reconstruction, not autonomous generation or accepted naturalness',
        'basisTraining':'Only original synthetic Heart and Michael reference clips',
        'basisVoicedFrames':count,'lookaheadMs':15,'auditionGain':.5,
        'synthesisFilter':filter_mode,'jitPreparationMs':preparation,
        'identity':'ECAPA speech cosine vs different-text reference; diagnostic, not acceptance',
        'energyMetric':'40 ms unweighted RMS envelope, 10 ms steps; not a perceptual loudness score',
        'modes':['residual-energy','output-energy-10ms','output-energy-40ms'],
        'results':[]}
    for case in cases:
        original,rate=sf.read(ROOT/case['file']);assert rate==24000
        anchor_file=(ROOT/f'evidence/voice-fixtures/{case["speaker"]}-reference.wav'
                     if case['speaker']!='af_bella' else
                     ROOT/f'evidence/streaming-voicebox/validation/af_bella-{1-int(case["name"][-1])}.wav')
        if anchor_file not in anchors:
            audio,rate=sf.read(anchor_file);assert rate==24000;anchors[anchor_file]=embedding(audio)
        anchor=anchors[anchor_file];original_identity=float(torch.dot(embedding(original),anchor))
        _,original_pitch=f0_track(original)
        frames=analyze(original)
        for label,energy_mode in (('residual-energy',False),('output-energy-10ms',True),('output-energy-40ms','window')):
            space=ControlSpace(source,output_energy=energy_mode)
            features=np.array([space.encode(frame) for frame in frames])
            started=time.perf_counter()
            generated,_,timings,_=render(space,frames,features,np.ones(len(frames),dtype=bool),filter_mode=filter_mode)
            elapsed=time.perf_counter()-started
            path=OUT/f'{case["name"]}-{label}.wav';save_audio(path,generated)
            text=transcribe(path);_,pitch=f0_track(generated);joint=(original_pitch>0)&(pitch>0)
            cents=np.abs(1200*np.log2(pitch[joint]/original_pitch[joint]))
            row={'fixture':case['name'],'mode':label,'source':case['file'],
                'anchor':str(anchor_file.relative_to(ROOT)),'file':str(path.relative_to(ROOT)),
                'text':text,**errors(case['text'],text),
                'identityCosine':float(torch.dot(embedding(generated),anchor)),
                'originalIdentityCosine':original_identity,
                'spectralErrorDb':spectral_db(original,generated),
                'envelope':envelope_error(original,generated),
                'medianPitchErrorCents':float(np.median(cents)) if len(cents) else None,
                'pitchVoicingAgreement':float(np.mean((original_pitch>0)==(pitch>0))),
                'rawPeak':float(np.abs(generated).max()),'decoderMsP95':float(np.percentile(timings,95)),
                'decoderRealTimeFactor':elapsed/(len(original)/rate)}
            report['results'].append(row)
            (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in ('source','anchor','file','text')}),flush=True)


if __name__=='__main__':main()
