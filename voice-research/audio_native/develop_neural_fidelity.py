"""Output-decoder ablation; development clips never fit encoder/decoder weights.

The existing encoder and framing stay fixed in meaning. No auditory/linguistic
input system is developed here. Dense target-derived controls remain a limit.
All conditions, including worse results, are retained before final evaluation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from neural_waveform import ControlEncoder, NeuralDecoder, initialize_decoder, save_model, waveform_frames

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/neural-waveform/fidelity-v2'


def snr(original, generated):
    return float(10*np.log10(max(float(original @ original), 1e-20)
                            / max(float(np.sum((original-generated)**2)), 1e-20)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clips = json.loads((ROOT/'evidence/neural-waveform/training-inputs/manifest.json').read_text())['clips']
    training, development, records = [], [], []
    for clip in clips:
        path = ROOT/clip['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != clip['sha256']:
            raise ValueError('Training corpus changed')
        audio, rate = sf.read(path)
        if rate != 24000: raise ValueError('Expected 24 kHz teacher audio')
        is_dev = clip['name'].endswith(('-10', '-11'))
        if is_dev: development.append((clip['name'], audio))
        else: training.append(waveform_frames(audio)[0])
        records.append({**clip, 'split': 'development' if is_dev else 'training'})
    targets = np.concatenate(training)
    conditions = [
        (256, 768, .7, 1e-3), (256, 1536, .7, 1e-3),
        (256, 1536, .25, 1e-5), (256, 1536, .1, 1e-6),
        (384, 1536, .25, 1e-5), (384, 3072, .25, 1e-5),
    ]
    report = {'purpose': 'Separate output representation loss and neural decoder approximation loss',
              'scope': 'Neural reconstruction, not independently controlled/new speech',
              'selection': 'Highest mean development waveform SNR; all conditions recorded',
              'corpus': records, 'conditions': conditions, 'results': [], 'complete': False}
    encoders = {}
    for dims, hidden, gain, ridge in conditions:
        if dims not in encoders:
            encoder = ControlEncoder.fit(targets, dimensions=dims)
            controls = encoder.encode_frames(targets)
            dev = [(name, audio, encoder.encode(audio)) for name, audio in development]
            encoders[dims] = (encoder, controls, dev)
        encoder, controls, dev = encoders[dims]
        name = f'd{dims}-h{hidden}-g{gain}-r{ridge}'
        started = time.perf_counter()
        learner, penalty = initialize_decoder(controls, targets, hidden=hidden, input_gain=gain, ridge=ridge)
        decoder = NeuralDecoder.from_learner(learner)
        row = {'name': name, 'dimensions': dims, 'hidden': hidden, 'inputGain': gain, 'ridge': ridge,
               'fitSeconds': time.perf_counter()-started, 'ridgePenalty': penalty, 'development': []}
        for fixture, audio, matrix in dev:
            started = time.perf_counter(); generated = decoder.render(matrix); elapsed = time.perf_counter()-started
            linear = encoder.reconstruct_linear(matrix)
            row['development'].append({'fixture': fixture, 'waveformSnrDb': snr(audio, generated),
                'linearReferenceSnrDb': snr(audio, linear), 'decoderRealTimeFactor': elapsed/(len(audio)/24000),
                'rawPeak': float(abs(generated).max())})
        row['meanDevelopmentSnrDb'] = float(np.mean([r['waveformSnrDb'] for r in row['development']]))
        save_model(OUT/(name+'.npz'), encoder, decoder,
                   {'trainingOrigin': '20 synthetic Kokoro clips; 4 development clips excluded',
                    'condition': name, 'scope': report['scope'], 'globalBackpropagation': False})
        report['results'].append(row)
        (OUT/'development.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({k: v for k,v in row.items() if k!='development'}), flush=True)
        del learner, decoder
    report['selected'] = max(report['results'], key=lambda r: r['meanDevelopmentSnrDb'])['name']
    report['complete'] = True
    (OUT/'development.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('SELECTED', report['selected'], flush=True)


if __name__ == '__main__': main()
