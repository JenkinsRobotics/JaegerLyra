"""Evaluate reusable vowel patches without a target-length source recording.

All intended vowels and all requested settings remain in the report, including
execution or control failures. These tests do not establish vowel recognition,
subjective naturalness or human-reference cloning.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import parselmouth
import pyworld
import soundfile as sf
from neural_waveform import load_model, RATE
from neural_performance import save_performance, load_performance
from neural_vowel_instrument import compile_vowel, NeuralVowelInstrument, VowelScore
from benchmark_voicebox import f0_track
from develop_neural_fidelity import snr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/neural-vowels/v1'


def pitch_metrics(times, pitch, score):
    expected = score.frequency(times)
    interior = (times >= .050) & (times <= score.seconds-.050)
    joint = interior & (pitch > 0)
    error = abs(1200*np.log2(pitch[joint]/expected[joint]))
    return {'scope': 'Entire sustained interval excluding fixed 50 ms attack/release margins',
        'eligibleFrames': int(sum(interior)), 'voicedFrames': int(sum(joint)),
        'voicedFraction': float(sum(joint)/max(1, sum(interior))),
        'medianErrorCents': float(np.median(error)) if len(error) else None,
        'p95ErrorCents': float(np.percentile(error, 95)) if len(error) else None,
        'maxErrorCents': float(max(error)) if len(error) else None}


def formants(audio):
    sound = parselmouth.Sound(audio, sampling_frequency=RATE)
    track = sound.to_formant_burg(time_step=.010, max_number_of_formants=5.,
        maximum_formant=5500., window_length=.025, pre_emphasis_from=50.)
    times = np.arange(.035, len(audio)/RATE-.035, .01)
    values = [[track.get_value_at_time(i, t) for t in times] for i in (1, 2, 3)]
    return [float(np.nanmedian(row)) if np.isfinite(row).any() else None for row in values]


def main():
    manifest = json.loads((ROOT / 'evidence/neural-vowels/references/manifest.json').read_text())
    if not manifest['complete'] or len(manifest['clips']) != 9:
        raise ValueError('Incomplete source panel')
    model = ROOT / manifest['candidate']
    if hashlib.sha256(model.read_bytes()).hexdigest() != manifest['candidateSha256']:
        raise ValueError('Frozen neural model changed')
    encoder, decoder, _ = load_model(model)
    before = hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True); (OUT/'patches').mkdir(exist_ok=True)
    report = {'scope': 'New authored sustained notes from <=200 ms captured vowel patches; sample-based neural instrument',
        'model': manifest['candidate'], 'modelSha256': manifest['candidateSha256'],
        'expectedReferences': 9, 'expectedCases': 54, 'references': [], 'results': [], 'complete': False,
        'timingScope': 'Warm model and patch resident; includes event planning and per-block synthesis, excludes loading/compilation/device/transport',
        'formants': 'Praat Burg F1/F2/F3 diagnostic, fixed 5500 Hz ceiling; not ground truth or a human vowel/identity rating',
        'limits': ['Vowel labels are intended by synthetic source prompts, not verified by a listener.',
            'Microvariation repeats from a captured patch; no learned new articulation or natural prosody.',
            'Full pulse plan scales with output duration; only decoded-waveform cache is bounded.',
            'No new words, transitions between vowels, human-reference cloning or perceptual quality acceptance.']}
    for source in manifest['clips']:
        item = {'name': source['name'], 'speaker': source['speaker'], 'intendedVowel': source['intendedVowel']}
        try:
            audio, rate = sf.read(ROOT/source['file']); assert rate == RATE
            patch, selected = compile_vowel(audio, encoder)
            item.update(selected)
            artifact = OUT/'patches'/(source['name']+'.npz')
            save_performance(artifact, patch, {**selected, 'sourceSha256': source['sha256'],
                'modelSha256': manifest['candidateSha256'], 'intendedVowel': source['intendedVowel']})
            patch, _ = load_performance(artifact)
            a = round(selected['sourceStartSeconds']*RATE); b = round(selected['sourceEndSeconds']*RATE)
            cropped = audio[a:b]
            item['patchReconstructionSnrDb'] = snr(cropped, decoder.render(patch.matrix))
            item['sourceFormantsHz'] = formants(cropped)
            item['artifact'] = str(artifact.relative_to(ROOT))
            item['status'] = 'ok'
            native = selected['sourceMedianPitchHz']
            settings = {
                'native': VowelScore(hz=(native,)),
                'down-three': VowelScore(hz=(native*2**(-3/12),)),
                'up-three': VowelScore(hz=(native*2**(3/12),)),
                'vibrato': VowelScore(hz=(native,), vibrato_cents=35.),
                'melody': VowelScore(seconds=4., times=(0., .95, 1.05, 1.95, 2.05, 2.95, 3.05, 4.),
                    hz=tuple(native*2**(np.array([0., 0., 2., 2., 4., 4., 0., 0.])/12))),
                'long-hold': VowelScore(seconds=8., hz=(native,)),
            }
            for name, score in settings.items():
                row = {'fixture': source['name'], 'mode': name, 'score': score.__dict__}
                try:
                    started = time.perf_counter(); stream = NeuralVowelInstrument(patch, decoder, score)
                    initialization = time.perf_counter()-started
                    if stream.reader.decoded_frames:
                        raise AssertionError('Constructor decoded source waveform')
                    pieces=[]; timings=[]
                    while True:
                        started=time.perf_counter(); packet=stream.read(); elapsed=time.perf_counter()-started
                        if packet is None: break
                        pieces.append(packet.samples); timings.append(elapsed)
                    output=np.concatenate(pieces)
                    if not np.isfinite(output).all() or np.max(abs(output)) >= .98:
                        raise ValueError('Nonfinite or excessive output peak')
                    if len(output) != round(score.seconds*RATE):
                        raise AssertionError('Incorrect duration')
                    path=OUT/(source['name']+'-'+name+'.wav')
                    sf.write(path, output, RATE, subtype='FLOAT')
                    pt, pf=f0_track(output)
                    wf, wt=pyworld.harvest(np.ascontiguousarray(output), RATE,
                        f0_floor=50., f0_ceil=900., frame_period=10.)
                    measured_formants=formants(output)
                    drift=[float(100*abs(v-r)/r) if v is not None and r is not None else None
                        for v, r in zip(measured_formants, item['sourceFormantsHz'])]
                    row.update({'status': 'ok', 'file': str(path.relative_to(ROOT)),
                        'durationSeconds': len(output)/RATE, 'rawPeak': float(abs(output).max()),
                        'praat': pitch_metrics(pt, pf, score), 'world': pitch_metrics(wt, wf, score),
                        'measuredFormantsHz': measured_formants, 'formantDriftPercent': drift,
                        'firstBlockComputeMs': (initialization+timings[0])*1000,
                        'computeRealTimeFactor': (initialization+sum(timings))/score.seconds,
                        'packetMsP95': float(np.percentile(timings,95)*1000),
                        'packetMsMaximum': max(timings)*1000,
                        'maximumCachedWaveformSamples': stream.reader.maximum_cache_frames*480,
                        'neuralFrameEvaluations': stream.reader.decoded_frames,
                        'retainedSourceSeconds': patch.matrix.sample_count/RATE})
                except ValueError as error:
                    row.update(status='failed', error=str(error))
                report['results'].append(row)
                (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                print(json.dumps({'fixture': row['fixture'], 'mode': name, 'status': row['status'],
                    'praat': row.get('praat'), 'world': row.get('world'), 'error': row.get('error')}), flush=True)
        except ValueError as error:
            item.update(status='failed', error=str(error))
        report['references'].append(item)
        (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if before != hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest():
        raise AssertionError('Instrument changed frozen neural weights')
    report['weightsUnchanged']=True; report['complete']=True
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
