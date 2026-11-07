"""Richer continuous source controls versus smaller reference-fitted coordinates.

Separates filter order, source resolution and PCA compression. All training uses
only the original reference clips; no target sample trains the basis. Analysis
has 15 ms lookahead. No offline pitch oracle is used in this benchmark.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ['HF_HUB_OFFLINE']='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch,torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from streaming_voicebox import VoiceboxConfig
from excitation import GeneratedVoicebox,VocalControls,pulse_profile
from continuous_source import ContinuousSource,descriptor
from benchmark_source_ceiling import analyze,spectral_error
from benchmark_voicebox import errors,transcribe,f0_track

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/rich-source'
FIXTURES=ROOT/'evidence/voice-fixtures'


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--validation',action='store_true')
    parser.add_argument('--fresh',action='store_true');args=parser.parse_args()
    if args.fresh:OUT=OUT/'fresh'
    elif args.validation:OUT=OUT/'validation'
    OUT.mkdir(exist_ok=True,parents=True);torch.set_num_threads(4)
    modeldir=ROOT/'models/speaker-eval'
    evaluator=EncoderClassifier.from_hparams(source=str(modeldir),savedir=str(modeldir),
        overrides={'pretrained_path':str(modeldir)},run_opts={'device':'cpu'})
    def embedding(audio):
        value=torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        value=torchaudio.functional.resample(value,24000,16000)
        with torch.inference_mode():value=evaluator.encode_batch(value).flatten()
        return value/torch.linalg.vector_norm(value)
    report={'scope':'Audio-derived continuous source reconstruction, no autonomous speech generation or new accepted voice engine',
        'training':'Each PCA basis fitted only to its original synthetic reference; no held-out audio fits weights',
        'analysisLookaheadMs':15,'firstFrameBufferMs':25,'oraclePitch':False,
        'mix':'Unchanged original periodicity-based voiced/noise mix; source-ceiling measured-mix variant not adopted',
        'validation':args.validation or args.fresh,'prospectiveFreshSentences':args.fresh,'models':[],'results':[]}
    for speaker in ('af_heart','am_michael'):
        reference,rate=sf.read(FIXTURES/f'{speaker}-reference.wav');assert rate==24000
        anchor=embedding(reference)
        models={}
        for order in (20,32):
            full,count=ContinuousSource.fit(analyze(reference,order),dimensions=48,harmonics=64)
            models[order]=full
            np.savez(OUT/f'{speaker}-order{order}.npz',mean=full.mean,basis=full.basis)
            report['models'].append({'speaker':speaker,'filterOrder':order,'trainingFrames':count,'harmonics':64,'maximumSourceDimensions':48})
        if args.validation or args.fresh:
            manifest_path=(ROOT/'evidence/rich-source/fresh-inputs/manifest.json' if args.fresh
                           else ROOT/'evidence/streaming-voicebox/validation/manifest.json')
            manifest=json.loads(manifest_path.read_text())
            cases=[(row['name'],ROOT/row['file'],row['text']) for row in manifest['clips'] if row['speaker']==speaker]
        else:
            path=FIXTURES/f'{speaker}-heldout.wav'
            cases=[(speaker,path,json.loads(path.with_suffix('.json').read_text())['text'])]
        for label,path,text in cases:
            original,rate=sf.read(path);assert rate==24000
            _,original_f0=f0_track(original)
            identity_original=float(torch.dot(embedding(original),anchor))
            frame_sets={order:analyze(original,order) for order in (20,32)}
            for order in (20,32):
                for dimensions in (24,48,None):
                    full=models[order];model=ContinuousSource(full.mean,full.basis[:dimensions]) if dimensions else None
                    encoded=[];start=time.perf_counter()
                    for frame in frame_sets[order]:
                        if model is not None:coords=model.encode(frame)
                        else:
                            coords=descriptor(frame,harmonics=64)
                            if coords is None:coords=full.mean.copy()
                        encoded.append((VocalControls.from_frame(frame),coords))
                    encoding_seconds=time.perf_counter()-start
                    sink=GeneratedVoicebox(profile=pulse_profile(64),config=VoiceboxConfig(order=order))
                    chunks=[];latencies=[]
                    for control,coords in encoded:
                        start=time.perf_counter()
                        harmonics=model.decode(coords) if model is not None else coords[:64]+1j*coords[64:]
                        chunks.append(sink.push(control,harmonics=harmonics))
                        latencies.append((time.perf_counter()-start)*1000)
                    audio=np.concatenate(chunks);peak=float(np.max(np.abs(audio)))
                    gain=min(float(np.sqrt(np.mean(original**2))/max(1e-9,np.sqrt(np.mean(audio**2)))),.88/max(1e-9,peak))
                    audio*=gain;name=f'{label}-o{order}-d{dimensions or 128}'
                    file=OUT/(name+'.wav')
                    sf.write(file,np.rint(audio*32768).astype(np.int16),rate,subtype='PCM_16')
                    heard=transcribe(file);_,f0=f0_track(audio);joint=(original_f0>0)&(f0>0)
                    cents=np.abs(1200*np.log2(f0[joint]/original_f0[joint]))
                    row={'fixture':label,'speaker':speaker,'filterOrder':order,'sourceDimensions':dimensions or 128,
                        'uncompressedSource':model is None,'file':str(file.relative_to(ROOT)),
                        'identityCosine':float(torch.dot(embedding(audio),anchor)),
                        'originalIdentityCosine':identity_original,'text':heard,**errors(text,heard),
                        'medianPitchErrorCents':float(np.median(cents)),
                        'voicingAgreement':float(np.mean((original_f0>0)==(f0>0))),
                        'spectral':spectral_error(original,audio),'rawPeak':peak,'auditionGain':gain,
                        'encodingSeconds':encoding_seconds,'decoderMsP95':float(np.percentile(latencies,95)),
                        'continuousScalarsPerFrame':order+4+(dimensions or 128)}
                    report['results'].append(row)
                    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps(row),flush=True)


if __name__=='__main__':main()
