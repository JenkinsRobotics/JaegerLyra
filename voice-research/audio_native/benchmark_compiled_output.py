"""Paired dense/compiled streaming with frozen weights and identical controls."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import load_model,HOP,RATE
from neural_output_stream import NeuralOutputStream
from compiled_neural_decoder import compile_decoder,CompiledAxisDecoder

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-waveform/capacity-v3/compiled-streaming'


def main():
    manifest=json.loads((ROOT/'evidence/neural-waveform/capacity-validation/manifest.json').read_text())
    models={'structured-v2':'evidence/neural-waveform/axis-detail-v2/projected-r1e-7.npz',
        'capacity-v3':manifest['candidate']}
    OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Dense versus mathematically equivalent sparse-input neural output streaming',
        'timingScope':'Warm resident model and precomputed controls; excludes compiler setup, analysis, live text, device and transport',
        'ordering':'For each control frame, dense/compiled order alternates; both model copies resident during paired measurement',
        'models':models,'results':[],'complete':False}
    for label,file in models.items():
        path=ROOT/file
        if label=='capacity-v3' and hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['candidateSha256']:
            raise ValueError('Frozen capacity candidate changed')
        encoder,dense,_=load_model(path)
        started=time.perf_counter();compiled=compile_decoder(dense);compile_seconds=time.perf_counter()-started
        if not isinstance(compiled,CompiledAxisDecoder):raise ValueError('Expected eligible structured network')
        before=hashlib.sha256(dense.W.tobytes()+dense.V.tobytes()).hexdigest()
        for decoder in (dense,compiled):
            for _ in range(4):decoder.frame(np.zeros(decoder.config.history-1),.01)
        for clip in manifest['clips']:
            source,rate=sf.read(ROOT/clip['file']);assert rate==RATE
            matrix=encoder.encode(source);expected=dense.render(matrix)
            streams={'dense':NeuralOutputStream(dense),'compiled':NeuralOutputStream(compiled)}
            chunks={k:[] for k in streams};timings={k:[] for k in streams}
            for index,(control,gain) in enumerate(zip(matrix.values,matrix.gains)):
                valid=(len(source)-1)%HOP+1 if index==len(matrix.values)-1 else HOP
                order=('dense','compiled') if index%2==0 else ('compiled','dense')
                for mode in order:
                    start=time.perf_counter();chunk=streams[mode].push(control,gain,emit_samples=valid)
                    elapsed=time.perf_counter()-start;timings[mode].append(elapsed);chunks[mode].append(chunk)
            row={'model':label,'fixture':clip['name'],'frames':len(matrix.values),
                'compilationMs':compile_seconds*1000,'modes':{}}
            for mode in streams:
                chunks[mode].append(streams[mode].finish());audio=np.concatenate(chunks[mode])
                if len(audio)!=len(expected):raise AssertionError('Output duration changed')
                error=float(max(abs(audio-expected)))
                if error>1e-12:raise AssertionError('Compiler or streaming changed waveform beyond tolerance')
                times=timings[mode]
                row['modes'][mode]={'maximumDifferenceFromDenseBatch':error,
                    'frameMsP50':float(np.percentile(times,50)*1000),
                    'frameMsP95':float(np.percentile(times,95)*1000),
                    'frameMsMaximum':max(times)*1000,'firstTwoFramesComputeMs':sum(times[:2])*1000,
                    'computeRealTimeFactor':sum(times)/(len(source)/RATE)}
            row['pairedComputeSpeedup']=sum(timings['dense'])/sum(timings['compiled'])
            report['results'].append(row)
            (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(label,clip['name'],'speedup',round(row['pairedComputeSpeedup'],3),flush=True)
        after=hashlib.sha256(dense.W.tobytes()+dense.V.tobytes()).hexdigest()
        fast_hash=hashlib.sha256(compiled.W.tobytes()+compiled.V.tobytes()).hexdigest()
        if before!=after or before!=fast_hash:raise AssertionError('Compiler changed neural weights')
    report['weightsUnchanged']=True;report['maximumStoredWaveformSamplesPerStream']=HOP;report['complete']=True
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
