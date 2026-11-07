"""Summarize completed decoder trials without equating execution with quality."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='evidence/neural-waveform/d256-h768')
    args = parser.parse_args()
    directory = ROOT / args.model
    report = json.loads((directory / 'evaluation/report.json').read_text())
    training = json.loads((directory / 'training-report.json').read_text())
    rows = report['results']
    expected = {(item['fixture'], mode) for item in report['originals'] for mode in report['models']}
    actual = [(item['fixture'], item['mode']) for item in rows]
    if (len(rows) != report['expectedCases'] or len(set(actual)) != len(actual)
            or set(actual) != expected):
        raise ValueError('Incomplete or duplicate evaluation cases')
    trained_hashes = {item['sha256'] for item in training['training']}
    originals = []
    for item in report['originals']:
        digest = hashlib.sha256((ROOT / item['file']).read_bytes()).hexdigest()
        if digest in trained_hashes:
            raise ValueError('Evaluation waveform is in the training corpus')
        originals.append({'fixture': item['fixture'], 'sha256': digest})
    metrics = ('waveformSnrDb', 'spectralErrorDb', 'identityCosine', 'decoderRealTimeFactor')
    groups = []
    for kind in ('authored', 'synthetic', 'human'):
        for mode in report['models']:
            selected = [r for r in rows if r['kind'] == kind and r['mode'] == mode]
            good = [r for r in selected if r['status'] == 'ok']
            group = {'kind': kind, 'mode': mode, 'cases': len(selected),
                     'executionFailures': len(selected) - len(good)}
            for metric in metrics:
                values = [r[metric] for r in good if r.get(metric) is not None]
                group['mean' + metric[0].upper() + metric[1:]] = statistics.mean(values) if values else None
            recognized = [r for r in good if 'wordErrors' in r]
            group['wordErrors'] = sum(r['wordErrors'] for r in recognized) if recognized else None
            group['referenceWords'] = sum(r['referenceWords'] for r in recognized) if recognized else None
            groups.append(group)
    summary = {
        'scope': report['scope'], 'complete': True, 'completedCases': len(rows),
        'successfulExecutions': sum(r['status'] == 'ok' for r in rows),
        'qualityAcceptance': 'Not established; successful execution is not a quality pass',
        'groups': groups, 'evaluationInputHashes': originals,
        'trainingClips': len(training['training']), 'trainingSeconds': training['trainingSeconds'],
        'trainingOrigin': training['trainingOrigin'],
        'trainingMse': {'linearRepresentation': training['trainingRepresentationMse'],
                        'neuralRidge': training['ridgeTrainingForwardMse'],
                        'localPcEpochs': training['epochs']},
        'authoredNeuralRidge': [
            {key: r.get(key) for key in ('fixture', 'waveformSnrDb', 'spectralErrorDb',
                                        'medianPitchErrorCents', 'pitchVoicingAgreement')}
            for r in rows if r['kind'] == 'authored' and r['mode'] == 'neural-ridge'],
        'limits': [
            'The target waveform is encoded into dense continuous coordinates; this is reconstruction, not new speech.',
            'The neural decoder receives only coordinates, gains and duration, without a PCM/residual bypass.',
            '257 controls every 10 ms (25,700 scalars/sec) is not a compression result or a small set of vocal knobs.',
            'Coordinates are not validated independent pitch, phoneme, articulator or speaker controls.',
            'pca-ceiling names the linear subspace reconstruction reference, not a universal perceptual upper bound.',
            'Notes and vowels were authored externally then encoded; the network has not learned to obey note/vowel commands.',
            'ASR and identity embeddings are diagnostics; neither establishes naturalness or perceptual equivalence.',
            'Only 24 synthetic teacher clips and two training voices; human evaluation has four clips from two readers.',
            'Batch PCA/least-squares initialization precedes local PC; no claim of wholly online or biological learning.',
            'Decoder timing excludes streaming scheduling, control generation, transport and audio hardware.',
        ],
        'nextMilestone': 'Test independently authored pitch and resonance controls, their transitions and cross-effects; improve decoder fidelity separately from generating new speech.',
        'appDefault': 'Kokoro Heart, unchanged',
    }
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'cases': len(rows), 'summary': str(directory / 'summary.json')}))


if __name__ == '__main__':
    main()
