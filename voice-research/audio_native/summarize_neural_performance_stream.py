"""Summarize every completed performance-stream case, retaining timing tails."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/neural-performance/streaming-v1'


def main():
    report = json.loads((OUT / 'report.json').read_text())
    rows = report['results']
    if not report['complete'] or len(rows) != report['expectedCases']:
        raise ValueError('Incomplete streaming evidence')
    if any(row['status'] != 'ok' for row in rows):
        raise ValueError('Failed streaming cases must not be omitted')

    def span(key):
        values = [row[key] for row in rows]
        return {'minimum': min(values), 'median': statistics.median(values), 'maximum': max(values)}

    summary = {
        'scope': report['scope'],
        'report': str((OUT / 'report.json').relative_to(ROOT)),
        'model': report['model'],
        'modelSha256': report['modelSha256'],
        'completedCases': len(rows),
        'maximumDifferenceFromOffline': max(row['maximumDifferenceFromOffline'] for row in rows),
        'weightsUnchanged': report['weightsUnchanged'],
        'allSynchronousCancellationChecksPassed': all(row['cancellationChecked'] for row in rows),
        'outputBlockSamples': report['outputBlockSamples'],
        'sampleRate': 24000,
        'maximumRetainedDecodedSamples': max(row['maximumCachedWaveformSamples'] for row in rows),
        'memoryScope': report['memoryScope'],
        'timingScope': report['timingScope'],
        'timingEnvironment': 'Initial benchmark ran alongside the 5.306-second research test suite; no timing cases discarded or rerun',
        'metrics': {key: span(key) for key in (
            'initializationMs', 'firstPacketComputeMs', 'firstNonzeroPacketComputeMs',
            'packetMsP95', 'packetMsMaximum', 'computeRealTimeFactor',
            'neuralFramesDecodedForFirstPacket')},
        'casesWithAtLeastOnePacketExceeding10ms': sum(row['packetMsMaximum'] > 10 for row in rows),
        'tests': {'passed': 84, 'log': 'evidence/audio-native-performance-stream-tests.txt'},
        'limits': [
            'Complete encoded score and event metadata remain resident; only the decoded waveform cache is bounded.',
            'No live text generation, growing score, live control changes, network transport or audio device measured.',
            'Some individual compute packets exceed their 10 ms audio duration; no underrun-free playback guarantee.',
            'Cancellation checks are synchronous; audio already queued at a consumer is outside this API.',
            'Faster availability preserves the offline audio and its unresolved quality/identity limitations.',
            'Perceptual Kokoro parity, novel speech generation and short-reference voice cloning remain unproven.',
        ],
        'completionProven': False,
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
