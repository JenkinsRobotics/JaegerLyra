"""Actual incremental DSP calls, no playback/hardware timing or language model.

Feed 128 input samples at a time. Audio timestamps simulate their arrival; wall
time measures computation only. This is not a microphone-to-speaker latency test.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import argparse,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
from voicebox_stream import ReferenceVoicebox
from aligned_analysis import AlignedAnalysis
from excitation import GeneratedVoicebox,VocalControls,pulse_profile
from noise_shape import noise_descriptor
from streaming_voicebox import LPC_VERSION

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/rich-source/streaming'


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--source-detail',action='store_true')
    parser.add_argument('--noise',action='store_true');args=parser.parse_args()
    detail=args.source_detail or args.noise
    if detail:OUT=ROOT/'evidence/source-detail/lpc-v2/streaming'/('colored-noise' if args.noise else 'baseline')
    OUT.mkdir(exist_ok=True,parents=True)
    report={'scope':'Incremental 24kHz DSP in Python, source controls measured from observed audio; no independent content generation',
        'timing':'Input sample timestamps simulate arrival; wall-clock compute measured. No sleep, sound device, browser playback or LLM. Not end-to-end conversation latency.',
        'sourceFit':'Per-reference frozen PCA, unlike the shared basis in the initial source-detail comparison',
        'analysisVersion':LPC_VERSION,'coloredNoise':args.noise,'transportChunkSamples':128,'results':[]}
    for speaker in ('af_heart','am_michael'):
        reference,rate=sf.read(ROOT/f'evidence/voice-fixtures/{speaker}-reference.wav');assert rate==24000
        data_file=ROOT/f'evidence/{"source-detail" if detail else "rich-source"}/fresh-inputs/{speaker}-0.wav'
        data,rate=sf.read(data_file);assert rate==24000
        start=time.perf_counter();stream=ReferenceVoicebox.from_reference(reference,shape_noise=args.noise)
        fit_seconds=time.perf_counter()-start
        before=stream.model.basis.copy();output=[];durations=[];active=[];first=None;pending_max=0
        for at in range(0,len(data),128):
            chunk=data[at:at+128];start=time.perf_counter()
            packets=stream.push(chunk);elapsed=(time.perf_counter()-start)*1000
            durations.append(elapsed)
            if packets:
                active.append(elapsed)
                if first is None:first={'inputSamplesReceived':at+len(chunk),'idealAvailableAtSample':packets[0].available_at,'callComputeMs':elapsed}
            output.extend(p.samples for p in packets)
            pending_max=max(pending_max,len(stream.analysis.pending))
        start=time.perf_counter();output.extend(p.samples for p in stream.finish())
        flush_ms=(time.perf_counter()-start)*1000
        audio=np.concatenate(output)
        # Same frozen source model through the batch fixture path. No per-clip
        # gain is applied in either path; compare raw floating-point output.
        sink=GeneratedVoicebox(profile=pulse_profile(64),config=stream.config,shape_noise=args.noise)
        batch=AlignedAnalysis(stream.config)
        frames=batch.push(data)+batch.finish()
        expected=np.concatenate([sink.push(VocalControls.from_frame(f),harmonics=stream.model.decode(stream.model.encode(f)),
                                 noise_bands=noise_descriptor(f) if args.noise else None) for f in frames])
        difference=float(np.max(np.abs(audio-expected)))
        assert len(audio)==len(data) and difference<1e-10
        assert np.array_equal(before,stream.model.basis)
        # Safe audition artifact; whole-clip gain is outside the streaming core.
        peak=float(np.max(np.abs(audio)))
        if detail:
            gain=.5
            if peak*gain>=1:raise ValueError('Fixed streaming audition gain would clip')
            sf.write(OUT/f'{speaker}.wav',audio*gain,rate,subtype='PCM_24')
        else:
            gain=min(float(np.sqrt(np.mean(data**2))/max(1e-9,np.sqrt(np.mean(audio**2)))),.88/max(1e-9,peak))
            sf.write(OUT/f'{speaker}.wav',np.rint(audio*gain*32768).astype(np.int16),rate,subtype='PCM_16')
        row={'speaker':speaker,'referencePreparationSeconds':fit_seconds,'duration':len(audio)/rate,
            'firstPacket':first,'inputBufferAtFirstPacketMs':first['inputSamplesReceived']/24,
            'processingMsP95EmittingCalls':float(np.percentile(active,95)),
            'processingRealTimeFactor':(sum(durations)+flush_ms)/1000/(len(data)/rate),
            'maximumPendingSamples':pending_max,'rawBatchMaxDifference':difference,'referenceWeightsUnchanged':True,
            'rawPeak':peak,'auditionGain':gain,'outputSamples':len(audio),
            'noiseHistorySamples':len(stream.synthesis.noise_shaper.history) if args.noise else 0}
        if detail:
            from benchmark_voicebox import transcribe,errors
            manifest=json.loads((data_file.parent/'manifest.json').read_text())
            target=next(c['text'] for c in manifest['clips'] if c['name']==speaker+'-0')
            text=transcribe(OUT/f'{speaker}.wav');row.update({'text':text,**errors(target,text)})
        report['results'].append(row)
        (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
