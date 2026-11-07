"""CPU output-frame timing and numerical equivalence; no device/transport claim."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import load_model,HOP
from neural_output_stream import NeuralOutputStream

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-waveform/axis-v1/streaming'


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='evidence/neural-waveform/output-validation/manifest.json')
    parser.add_argument('--output',default='evidence/neural-waveform/axis-v1/streaming')
    args=parser.parse_args();OUT=ROOT/args.output
    manifest=json.loads((ROOT/args.manifest).read_text())
    path=ROOT/manifest['candidate']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['candidateSha256']:
        raise ValueError('Frozen candidate changed')
    encoder,decoder,_=load_model(path);OUT.mkdir(parents=True,exist_ok=True)
    before=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest()
    report={'scope':'Neural output-only streaming from precomputed control frames',
        'candidate':manifest['candidate'],'candidateSha256':manifest['candidateSha256'],
        'inputWaveformAtDecoder':False,'precomputedControls':True,
        'latencyLimit':'Timing excludes control generation, device, transport, scheduling and first text-to-audio delay',
        'framing':'Two control frames to produce first 240-sample output; subsequent frames emit 10 ms',
        'maximumStoredWaveformSamples':HOP,'results':[],'complete':False}
    decoder.frame(np.zeros(decoder.config.history-1),.01)
    for clip in manifest['clips']:
        audio,rate=sf.read(ROOT/clip['file']);assert rate==24000
        matrix=encoder.encode(audio);expected=decoder.render(matrix);stream=NeuralOutputStream(decoder)
        chunks=[];timings=[];first_compute=0.
        for index,(control,gain) in enumerate(zip(matrix.values,matrix.gains)):
            valid=(len(audio)-1)%HOP+1 if index==len(matrix.values)-1 else HOP
            started=time.perf_counter();chunk=stream.push(control,gain,emit_samples=valid)
            elapsed=time.perf_counter()-started;timings.append(elapsed)
            if index<2:first_compute+=elapsed
            chunks.append(chunk)
        chunks.append(stream.finish());generated=np.concatenate(chunks)
        error=float(np.max(abs(expected-generated)))
        if error>1e-12 or len(generated)!=len(audio):raise AssertionError('Output chunking changed waveform')
        after=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest()
        if before!=after:raise AssertionError('Synthesis modified decoder weights')
        saved=generated*.5
        if abs(saved).max()>=1:raise ValueError('Fixed audition gain would clip')
        sf.write(OUT/(clip['name']+'.wav'),saved,rate,subtype='PCM_24')
        row={'fixture':clip['name'],'frames':len(timings),'samples':len(generated),'batchMaximumError':error,
            'frameMsP50':float(np.percentile(timings,50)*1000),'frameMsP95':float(np.percentile(timings,95)*1000),
            'frameMsMaximum':max(timings)*1000,'firstTwoFramesComputeMs':first_compute*1000,
            'computeRealTimeFactor':sum(timings)/(len(audio)/rate),'weightsUnchanged':True}
        report['results'].append(row);print(json.dumps(row),flush=True)
    report['complete']=True
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
