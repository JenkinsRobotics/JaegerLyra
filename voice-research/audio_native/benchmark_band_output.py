"""Verify and time the complementary decoder's actual incremental output."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import RATE,HOP
from band_preserving_decoder import load_band_preserving,BandPreservingOutputStream

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-waveform/band-v4/streaming'


def main():
    manifest=json.loads((ROOT/'evidence/neural-waveform/band-validation/manifest.json').read_text())
    if len(manifest['clips'])!=manifest['expectedClips']:raise ValueError('Incomplete final input panel')
    path=ROOT/manifest['candidate']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['candidateSha256']:raise ValueError('Frozen assembly changed')
    encoder,decoder,_=load_band_preserving(path)
    weight_hashes=[hashlib.sha256(m.W.tobytes()+m.V.tobytes()).hexdigest() for m in (decoder.base,decoder.detail)]
    OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Two frozen neural output branches, exact-prefix controls and complementary high-frequency correction',
        'candidate':manifest['candidate'],'candidateSha256':manifest['candidateSha256'],
        'timingScope':'Warm resident compiled model branches and precomputed controls; excludes loading, compilation, control generation, live text, devices and transport',
        'framing':'Two control frames prime the first output; 176 samples in the first normal packet, then 240, with 64 flushed by finish()',
        'additionalFirLookaheadSamples':decoder.delay,'additionalFirLookaheadMs':decoder.delay/RATE*1000,
        'retainedWaveformStateLimitSamples':2*HOP+(len(decoder.highpass)-1)+decoder.delay,
        'stateScope':'Two 240-sample neural overlap tails, 128 FIR state samples, at most 64 pending base samples; excludes weights, input controls and temporary arrays',
        'results':[],'complete':False}
    for source in manifest['clips']:
        audio,rate=sf.read(ROOT/source['file']);assert rate==RATE
        controls=encoder.encode(audio);expected=decoder.render(controls)
        stream=BandPreservingOutputStream(decoder)
        chunks=[];timings=[];first_output=None
        for index,(value,gain) in enumerate(zip(controls.values,controls.gains)):
            count=(len(audio)-1)%HOP+1 if index==len(controls.values)-1 else HOP
            started=time.perf_counter();packet=stream.push(value,gain,count)
            elapsed=time.perf_counter()-started;timings.append(elapsed);chunks.append(packet)
            if first_output is None and len(packet):first_output={'frames':index+1,'samples':len(packet),'computeMs':sum(timings)*1000}
        started=time.perf_counter();chunks.append(stream.finish());flush_ms=(time.perf_counter()-started)*1000
        generated=np.concatenate(chunks)
        difference=float(max(abs(generated-expected)))
        if len(generated)!=len(expected) or difference>1e-12:raise AssertionError('Streaming changed the complementary waveform')
        if stream.maximum_pending_samples>decoder.delay or len(stream.pending) or len(stream.history):
            raise AssertionError('Delayed output state was unbounded or not released')
        cancelled=BandPreservingOutputStream(decoder)
        for value,gain in zip(controls.values[:7],controls.gains[:7]):cancelled.push(value,gain)
        counts=(cancelled.base.frames,cancelled.detail.frames);cancelled.cancel()
        if len(cancelled.finish()) or len(cancelled.pending) or len(cancelled.history) or counts!=(cancelled.base.frames,cancelled.detail.frames):
            raise AssertionError('Cancellation emitted delayed audio or resumed model work')
        row={'fixture':source['name'],'frames':len(timings),'samples':len(generated),
            'maximumDifferenceFromBatch':difference,'firstOutput':first_output,
            'frameMsP50':float(np.percentile(timings,50)*1000),'frameMsP95':float(np.percentile(timings,95)*1000),
            'frameMsMaximum':max(timings)*1000,'flushMs':flush_ms,
            'computeRealTimeFactor':(sum(timings)+flush_ms/1000)/(len(audio)/RATE),
            'maximumPendingBaseSamples':stream.maximum_pending_samples,'cancellationPassed':True}
        report['results'].append(row)
        (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(source['name'],'p95ms',round(row['frameMsP95'],3),'error',difference,flush=True)
    after=[hashlib.sha256(m.W.tobytes()+m.V.tobytes()).hexdigest() for m in (decoder.base,decoder.detail)]
    if weight_hashes!=after:raise AssertionError('Streaming changed model weights')
    report['weightsUnchanged']=True;report['complete']=True
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
