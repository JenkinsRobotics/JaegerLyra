"""Warm precompiled-performance streaming, compared with every offline control case.

No microphone, speaker or transport participates. Scores/model are resident;
reported initialization includes control snapshots and event planning. Output
buffers are collected only by this evaluator, not retained by the runtime.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib,json,time
from pathlib import Path
import numpy as np
from neural_waveform import load_model,SIZE,HOP,RATE
from neural_performance import load_performance,PerformanceControls,render_performance
from neural_performance_stream import NeuralPerformanceStream

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-performance/streaming-v1'


def main():
    prior=json.loads((ROOT/'evidence/neural-performance/v2/report.json').read_text())
    if not prior['complete'] or len(prior['results'])!=42:raise ValueError('Incomplete offline reference')
    model=ROOT/prior['model']
    if hashlib.sha256(model.read_bytes()).hexdigest()!=prior['modelSha256']:raise ValueError('Model changed')
    _,decoder,_=load_model(model)
    weight_hash=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest()
    originals={r['fixture']:r for r in prior['originals']}
    source={name:load_performance(ROOT/r['performance'])[0] for name,r in originals.items()}
    report={'scope':'Incremental PCM output from a complete precompiled control score',
        'model':prior['model'],'modelSha256':prior['modelSha256'],'expectedCases':42,
        'timingScope':'Warm resident model/score; includes snapshot/event planning and per-read decoding/editing, excludes disk load, source compilation, transport, devices and live text generation',
        'memoryScope':'Only retained decoded-waveform cache is bounded; model weights, full encoded control score and event metadata remain resident',
        'cacheFrameLimit':16,'cacheSampleLimit':16*SIZE,'outputBlockSamples':HOP,
        'controllerInputs':'Encoded control matrix, pulse timings, authored pitch/duration settings; no original PCM',
        'results':[],'complete':False}
    OUT.mkdir(parents=True,exist_ok=True)
    for case in prior['results']:
        if case['status']!='ok':raise ValueError('Do not drop a failed offline control case')
        performance=source[case['fixture']];control=PerformanceControls(**case['controls'])
        started=time.perf_counter();expected,_=render_performance(performance,decoder,control)
        offline_seconds=time.perf_counter()-started
        started=time.perf_counter();stream=NeuralPerformanceStream(performance,decoder,control)
        initialization=time.perf_counter()-started
        if stream.reader.decoded_frames:raise AssertionError('Constructor decoded waveform')
        parts=[];timings=[];starts=[];first_frames=None;first_audible_compute=None;elapsed_compute=initialization
        while True:
            started=time.perf_counter();packet=stream.read();elapsed=time.perf_counter()-started
            if packet is None:break
            timings.append(elapsed);elapsed_compute+=elapsed
            if first_frames is None:first_frames=stream.reader.decoded_frames
            if first_audible_compute is None and np.max(abs(packet.samples))>1e-5:first_audible_compute=elapsed_compute
            starts.append(packet.start);parts.append(packet.samples)
        generated=np.concatenate(parts) if parts else np.empty(0)
        error=float(np.max(abs(generated-expected))) if len(expected) else 0.
        if (len(generated)!=len(expected) or error>1e-12
                or starts!=list(range(0,len(expected),HOP))):raise AssertionError('Streaming changed performance audio/timing')
        if stream.reader.maximum_cache_frames*SIZE>report['cacheSampleLimit']:raise AssertionError('Unbounded decoded cache')
        if stream.reader.cache:raise AssertionError('End-of-stream retained decoded windows')
        # Synchronous cancellation after issuing several packets must prevent
        # all subsequent reads and any further neural evaluation.
        interrupted=NeuralPerformanceStream(performance,decoder,control)
        for _ in range(7):interrupted.read()
        before_cancel=interrupted.reader.decoded_frames
        interrupted.cancel()
        if interrupted.read() is not None or interrupted.reader.decoded_frames!=before_cancel or interrupted.reader.cache:
            raise AssertionError('Cancelled stream emitted/decode more data')
        row={'fixture':case['fixture'],'mode':case['mode'],'status':'ok','samples':len(generated),'packets':len(parts),
            'offlineWholePerformanceMs':offline_seconds*1000,'initializationMs':initialization*1000,
            'firstPacketComputeMs':(initialization+timings[0])*1000,
            'firstNonzeroPacketComputeMs':first_audible_compute*1000 if first_audible_compute is not None else None,
            'packetMsP50':float(np.percentile(timings,50)*1000),'packetMsP95':float(np.percentile(timings,95)*1000),
            'packetMsMaximum':max(timings)*1000,'computeRealTimeFactor':elapsed_compute/(len(generated)/RATE),
            'maximumDifferenceFromOffline':error,'neuralFramesDecodedForFirstPacket':first_frames,
            'totalSourceControlFrames':len(performance.matrix.values),'neuralFrameEvaluations':stream.reader.decoded_frames,
            'maximumCachedWaveformSamples':stream.reader.maximum_cache_frames*SIZE,
            'maximumActiveGrains':stream.maximum_active_grains,'cancellationChecked':True}
        report['results'].append(row)
        (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
    after=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest()
    if after!=weight_hash:raise AssertionError('Streaming changed neural weights')
    report['weightsUnchanged']=True;report['complete']=True
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
