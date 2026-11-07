"""Silent source/filter streaming reconstruction, control and residual ablation.

Existing synthetic fixtures only; never opens a microphone or plays a speaker.
ASR and Praat are evaluation-only and are not in the audio-in/out signal path.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request
import numpy as np
import soundfile as sf
import parselmouth
from streaming_voicebox import VoiceboxConfig, StreamingAnalysis, StreamingSynthesis, process

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'evidence/streaming-voicebox'
FIXTURES = ROOT/'evidence/voice-fixtures'


def errors(expected, actual):
    a, b = [re.findall(r"[a-z']+", value.lower()) for value in (expected, actual)]
    previous = list(range(len(b)+1))
    for i, word in enumerate(a, 1):
        current = [i]
        for j, heard in enumerate(b, 1):
            current.append(min(previous[j]+1, current[-1]+1, previous[j-1]+(word!=heard)))
        previous = current
    return {'wordErrors': previous[-1], 'referenceWords':len(a), 'wer':previous[-1]/max(1,len(a))}


def transcribe(path):
    request = urllib.request.Request('http://127.0.0.1:8791/api/transcribe', path.read_bytes(), {'Content-Type':'audio/wav'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)['text']


def f0_track(audio):
    result = parselmouth.Sound(audio, 24000).to_pitch_ac(time_step=.01,pitch_floor=65,pitch_ceiling=500)
    return result.xs(), result.selected_array['frequency']


def main():
    OUT.mkdir(exist_ok=True)
    cfg = VoiceboxConfig()
    report = {'scope':'Continuous streaming source/filter scaffold with original PCM residual; no semantic understanding, new TTS model, voice cloning or compression',
        'config':asdict(cfg), 'sources':['https://dsprelated.com/freebooks/pasp/Voice_Synthesis.html', 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html'],
        'framing':'Emit every 240 received samples (10 ms); 40 ms past analysis window. No future beyond frame end. Additional transport/device latency excluded.',
        'modes':{'roundtrip':'Original residual and filter; exactness evaluated before PCM quantization',
                 'resonance-low':'Pole angles scaled 0.92, radii retained, original residual',
                 'resonance-high':'Pole angles scaled 1.08, radii retained, original residual',
                 'noise-ablation':'Original filter, residual replaced by RMS-matched Gaussian noise; demonstrates residual information loss'},
        'fixtures':[], 'results':[]}
    for speaker in ('af_heart', 'am_michael'):
        for role in ('reference', 'heldout'):
            path = FIXTURES/f'{speaker}-{role}.wav'
            audio, rate = sf.read(path)
            assert rate==24000 and audio.ndim==1
            metadata = json.loads(path.with_suffix('.json').read_text())
            report['fixtures'].append({'file':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'role':role})
            analysis, synthesis = StreamingAnalysis(), StreamingSynthesis()
            frames, outputs, latencies = [], [], []
            for at in range(0,len(audio),cfg.hop):
                started = time.perf_counter()
                items = analysis.push(audio[at:at+cfg.hop])
                for frame in items:
                    frames.append(frame); outputs.append(synthesis.push(frame))
                latencies.append((time.perf_counter()-started)*1000)
            for frame in analysis.finish():
                frames.append(frame); outputs.append(synthesis.push(frame))
            roundtrip = np.concatenate(outputs)
            features = np.array([f.features() for f in frames])
            np.savez(OUT/f'{speaker}-{role}-features.npz', features=features,
                     available_at=np.array([f.available_at for f in frames]),
                     f0=np.array([f.f0 for f in frames]))
            row = {'speaker':speaker,'role':role,'duration':len(audio)/rate,
                'maxFloatReconstructionError':float(np.max(np.abs(audio-roundtrip))),
                'pcm16Identical':bool(np.array_equal(np.rint(audio*32768).astype(np.int16),np.rint(roundtrip*32768).astype(np.int16))),
                'processingMsMedian':float(np.median(latencies)), 'processingMsP95':float(np.percentile(latencies,95)),
                'processingRealTimeFactor':sum(latencies)/1000/(len(audio)/rate),
                'retainedStateSamples':len(analysis.history)+len(analysis.input_history)+len(analysis.pending)+len(synthesis.output_history),
                'residualSampleCount':sum(len(f.residual) for f in frames),
                'coefficientCount':sum(len(f.coefficients) for f in frames),
                'maxPoleRadius':float(max(np.max(np.abs(np.roots(f.coefficients))) for f in frames)),
                'modes':[]}
            assert row['pcm16Identical'], row
            if role=='heldout':
                times, source_pitch = f0_track(audio)
                # Causal frame estimates describe the trailing 40ms window;
                # compare with its center using a 20ms timestamp correction.
                local_times = np.array([f.available_at/rate-.020 for f in frames])
                nearest = np.abs(local_times[:,None]-times[None,:]).argmin(axis=1)
                local_pitch = np.array([f.f0 for f in frames]); oracle = source_pitch[nearest]
                both = (local_pitch>0)&(oracle>0)
                row['pitchTrackerDiagnostic']={'estimator':'Praat autocorrelation, no octave correction; not ground truth',
                    'voicingAgreement':float(np.mean((local_pitch>0)==(oracle>0))),
                    'jointVoicedFrames':int(sum(both)),
                    'medianAbsoluteCents':float(np.median(np.abs(1200*np.log2(local_pitch[both]/oracle[both])))),
                    'over200CentsFraction':float(np.mean(np.abs(1200*np.log2(local_pitch[both]/oracle[both]))>200))}
                for mode, factor, residual in [('roundtrip',1.,'original'), ('resonance-low',.92,'original'), ('resonance-high',1.08,'original'), ('noise-ablation',1.,'noise')]:
                    output = roundtrip.copy() if mode=='roundtrip' else process(audio,resonance=factor,residual_mode=residual)[0]
                    raw_peak = float(np.max(np.abs(output)))
                    gain = 1. if mode=='roundtrip' else min(np.sqrt(np.mean(audio*audio))/max(1e-9,np.sqrt(np.mean(output*output))), .88/max(1e-9,raw_peak))
                    output *= gain
                    saved = OUT/f'{speaker}-{mode}.wav'
                    sf.write(saved, np.rint(output*32768).astype(np.int16), rate, subtype='PCM_16')
                    heard = transcribe(saved)
                    _, output_pitch = f0_track(output)
                    assert len(output_pitch)==len(source_pitch)
                    common = (source_pitch>0)&(output_pitch>0)
                    details={'mode':mode,'file':str(saved.relative_to(ROOT)), 'rawPeak':raw_peak,'auditionGain':float(gain),
                        'text':heard,**errors(metadata['text'],heard),
                        'voicingAgreement':float(np.mean((source_pitch>0)==(output_pitch>0))),
                        'jointVoicedFrames':int(sum(common)),
                        'medianAbsolutePitchChangeCents':float(np.median(np.abs(1200*np.log2(output_pitch[common]/source_pitch[common])))) if any(common) else None}
                    row['modes'].append(details)
                    print(speaker,mode,json.dumps(details),flush=True)
            report['results'].append(row)
            (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(speaker, role, 'p95 ms',row['processingMsP95'],'PCM16 identical',row['pcm16Identical'],flush=True)


if __name__=='__main__':
    main()
