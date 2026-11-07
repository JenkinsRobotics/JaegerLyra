"""Silent, reproducible continuous-PCM/local-learning benchmark.

Run with ../../.research-venv/bin/python benchmark.py from this directory.
Evaluation observes history after committing each next-block prediction. It is
teacher-forced short-horizon prediction, NOT free-running speech generation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from lab import Config, LinearPredictor, LocalPredictor, continuous_roundtrip, voicebox

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/audio-native'
FIXTURES = ROOT / 'evidence/voice-fixtures'


def load(path):
    audio, rate = sf.read(path, dtype='float64')
    if rate != 24000 or audio.ndim != 1:
        raise ValueError('This experiment expects mono 24 kHz fixtures')
    return audio, rate


def blocks(audio, config):
    for at in range(config.history, len(audio)-config.block+1, config.block):
        yield at, audio[at-config.history:at], audio[at:at+config.block]


def snr(target, estimate):
    return float(10*np.log10((np.sum(target**2)+1e-20) /
                            (np.sum((target-estimate)**2)+1e-20)))


def evaluate(model, audio, adaptive, output):
    predictions, targets, times, energies = [], [], [], []
    for _, history, target in blocks(audio, model.config):
        start = time.perf_counter()
        prediction = model.predict(history).copy()
        # No target has entered this prediction. Store it before observe().
        predictions.append(prediction)
        targets.append(target)
        result = model.observe(history, target, learn=adaptive)
        times.append((time.perf_counter()-start)*1000)
        if result:
            energies.append([result['initialEnergy'], result['finalEnergy']])
    predicted = np.concatenate(predictions)
    truth = np.concatenate(targets)
    # File only contains predictions; no original warm-up samples spliced in.
    sf.write(output, predicted, 24000, subtype='FLOAT')
    row = {'preObservationSnrDb': round(snr(truth, predicted), 3),
           'preObservationMse': float(np.mean((truth-predicted)**2)),
           'finite': bool(np.isfinite(predicted).all()),
           'peak': float(np.max(np.abs(predicted))),
           'frames': len(times), 'adaptWeightsDuringEvaluation': adaptive,
           'processMsMedian': round(float(np.median(times)), 4),
           'processMsP95': round(float(np.percentile(times, 95)), 4),
           'processingRealTimeFactor': round(sum(times)/1000/(len(truth)/24000), 4)}
    if energies:
        row['settlingEnergyIncreaseCount'] = int(sum(b>a+1e-10 for a,b in energies))
        row['conditionedEnergyMedianReduction'] = float(np.median([a-b for a,b in energies]))
    return row


def main():
    OUT.mkdir(exist_ok=True)
    config = Config()
    report = {'scope': 'Continuous signal and local-learning mechanics only; no language understanding, TTS or cloning',
              'date': '2026-11-06', 'config': asdict(config),
              'sampleRate': 24000, 'predictionHorizonMs': config.block/24,
              'historyMs': config.history/24,
              'training': 'One pass through synthetic Heart and Michael reference clips; no held-out samples used for initial weights',
              'evaluation': 'Commit prediction before observing each next block. True preceding audio remains available (teacher forcing).',
              'noTranscriptOrCodebook': True, 'globalBackpropagation': False,
              'usesExplicitLocalEnergyGradients': True, 'fixtures': [],
              'roundtrip': [], 'prediction': []}
    models = {'local-pc': LocalPredictor(config), 'linear-nlms': LinearPredictor(config)}
    started = time.perf_counter()
    for speaker in ['af_heart', 'am_michael']:
        path = FIXTURES / (speaker+'-reference.wav')
        audio, _ = load(path)
        report['fixtures'].append({'file': str(path.relative_to(ROOT)),
                                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'role': 'training'})
        for _, history, target in blocks(audio, config):
            for model in models.values():
                model.observe(history, target)
    report['trainingSeconds'] = round(time.perf_counter()-started, 3)
    print('training seconds', report['trainingSeconds'], flush=True)
    for speaker in ['af_heart', 'am_michael']:
        path = FIXTURES / (speaker+'-heldout.wav')
        audio, rate = load(path)
        report['fixtures'].append({'file': str(path.relative_to(ROOT)),
                                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'role': 'evaluation'})
        start = time.perf_counter()
        reconstructed = continuous_roundtrip(audio)
        report['roundtrip'].append({'speaker': speaker,
            'maxFloatError': float(np.max(np.abs(audio-reconstructed))),
            'pcm16Identical': bool(np.array_equal(np.rint(audio*32768).astype(np.int16),
                                                  np.rint(reconstructed*32768).astype(np.int16))),
            'analysisSynthesisMs': round((time.perf_counter()-start)*1000, 3),
            'frameBufferMs': 10, 'compressionRatio': 1.0})
        # Explicit nearest-integer quantization avoids libsndfile's float-to-PCM
        # truncation turning sub-ulp transform noise into a one-LSB difference.
        restored_pcm = np.rint(reconstructed*32768).astype(np.int16)
        sf.write(OUT/(speaker+'-roundtrip.wav'), restored_pcm, rate, subtype='PCM_16')
        truth = np.concatenate([target for _,_,target in blocks(audio, config)])
        persistence = np.concatenate([past[-config.block:] for _,past,_ in blocks(audio, config)])
        report['prediction'].append({'speaker': speaker, 'model': 'repeat-previous-block',
                                     'preObservationSnrDb': round(snr(truth, persistence), 3)})
        report['prediction'].append({'speaker': speaker, 'model': 'silence', 'preObservationSnrDb': 0.0})
        for name, source_model in models.items():
            for adaptive in [False, True]:
                model = copy.deepcopy(source_model)
                suffix = 'adaptive' if adaptive else 'frozen'
                row = evaluate(model, audio, adaptive, OUT/(speaker+'-'+name+'-'+suffix+'.wav'))
                row.update(speaker=speaker, model=name, mode=suffix)
                # Retention: evaluate first voice after adapting to the second,
                # with no further learning; compare to the original frozen model.
                if speaker == 'am_michael' and adaptive:
                    first, _ = load(FIXTURES/'af_heart-heldout.wav')
                    retention = evaluate(model, first, False, OUT/(name+'-return-to-heart.wav'))
                    row['returnToHeartFrozenSnrDb'] = retention['preObservationSnrDb']
                report['prediction'].append(row)
                print(json.dumps(row), flush=True)
    seconds = 4
    t = np.arange(24000*seconds)/24000
    formants = np.tile([750., 1200., 2600.], (len(t), 1))
    glide = voicebox(140+100*t/seconds, formants)
    sf.write(OUT/'voicebox-fixed-resonance-pitch-glide.wav', glide, 24000, subtype='PCM_16')
    destination = np.array([300., 2300., 3000.])
    blend = (.5-.5*np.cos(np.pi*t/seconds))[:,None]
    morph = voicebox(np.full(len(t), 180.), formants*(1-blend)+destination*blend)
    sf.write(OUT/'voicebox-fixed-pitch-resonance-morph.wav', morph, 24000, subtype='PCM_16')
    report['voicebox'] = {'method': 'Analytic harmonic/noise source and resonance envelope; no learned model',
                         'clips': ['voicebox-fixed-resonance-pitch-glide.wav', 'voicebox-fixed-pitch-resonance-morph.wav'],
                         'limitation': 'Vowel-like synthesis only. Not speech, not cloning, and not Kokoro quality.'}
    report['elapsedSeconds'] = round(time.perf_counter()-started, 3)
    (OUT/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
