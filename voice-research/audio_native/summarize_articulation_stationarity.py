"""Summarize the fixed-cycle development study without treating ASR as listening."""
import hashlib
import json
import re
import statistics as stats
from pathlib import Path

from summarize_neural_articulation import pitch_group

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/neural-articulation/stationarity-v1'
METHODS = ('baseline-loop', 'fixed-original', 'stationary-loop', 'stationary-fixed')
MODES = ('native', 'melody', 'short-native')


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    study = read('evidence/neural-articulation/stationarity-v1/report.json')
    selection = read('evidence/neural-articulation/stationarity-v1/selection-report.json')
    words = read('evidence/neural-articulation/stationarity-v1/word-report.json')
    teacher = read('evidence/neural-articulation/kokoro-reference-v1/evaluation.json')
    audition = read('public/research/sustain-comparison/manifest.json')
    if not all(j['complete'] for j in (study, selection, words, teacher)):
        raise ValueError('Incomplete study')
    rows = study['results']
    if len({r['name'] for r in rows}) != 108 or len(teacher['results']) != 18:
        raise ValueError('Missing or duplicate cases')
    if not study['decoderAndSourcesUnchanged'] or any(r['requestedControlsChanged'] for r in rows):
        raise ValueError('Comparison controls changed')
    checked = {}
    for r in words['results'] + teacher['results']:
        path = r['file']
        checked.setdefault(path, hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
        if checked[path] != r['sha256']:
            raise ValueError('Audio changed after transcription: ' + path)
    for r in audition['files']:
        if hashlib.sha256((ROOT / r['source']).read_bytes()).hexdigest() != r['sourceSha256']:
            raise ValueError('Audition source changed')
        if not (ROOT / 'public/research/sustain-comparison' / r['file']).is_file():
            raise ValueError('Missing audition')
    groups = []
    for mode in MODES:
        for method in METHODS:
            cases = [r for r in rows if r['mode'] == mode and r['method'] == method]
            lexical = [r for r in words['results'] if r['mode'] == mode and r['method'] == method]
            if len(cases) != 9 or {r['name'] for r in cases} != {r['name'] for r in lexical}:
                raise ValueError('Mismatched comparison group')
            drift = [{'name': r['name'], 'formant': i + 1, 'percent': v}
                     for r in cases for i, v in enumerate(r['formantDriftPercent']) if v is not None]
            groups.append({
                'mode': mode, 'method': method, 'cases': len(cases),
                'homophoneEquivalentMatches': sum(r['homophoneEquivalentMatch'] for r in lexical),
                'exactSpellingMatches': sum(r['exactSpellingMatch'] for r in lexical),
                'mismatches': [{'name': r['name'], 'transcript': r['transcript']}
                               for r in lexical if not r['homophoneEquivalentMatch']],
                'medianHeldShapeVariationDb': stats.median(r['heldVariation']['shapeRmsDb'] for r in cases),
                'medianHeldEnergyStdDb': stats.median(r['heldVariation']['energyStdDb'] for r in cases),
                'praat': pitch_group(cases, 'praat'), 'world': pitch_group(cases, 'world'),
                'formantComparisons': len(drift), 'possibleFormantComparisons': 3 * len(cases),
                'medianFormantDriftPercent': stats.median(r['percent'] for r in drift) if drift else None,
                'worstFormantDrift': max(drift, key=lambda r: r['percent'], default=None),
            })
    teacher_groups = []
    for mode in MODES[:2]:
        cases = [r for r in teacher['results'] if r['mode'] == mode]
        teacher_groups.append({
            'mode': mode, 'cases': len(cases),
            'homophoneEquivalentMatches': sum(r['homophoneEquivalentMatch'] for r in cases),
            'praat': pitch_group(cases, 'praat'), 'world': pitch_group(cases, 'world'),
            'medianPraatVoicedFraction': stats.median(r['praat']['voicedFraction'] for r in cases),
            'medianWorldVoicedFraction': stats.median(r['world']['voicedFraction'] for r in cases),
        })
    new = [r for r in rows if not r['audioReused']]
    streams = [r['streaming'] for r in new]
    log = ROOT / 'evidence/audio-native-stationarity-tests.txt'
    contents = log.read_text()
    match = re.search(r'Ran (\d+) tests', contents)
    if match is None or not contents.rstrip().endswith('OK'):
        raise ValueError('Missing successful research tests')
    summary = {
        'scope': study['scope'], 'comparisonCases': len(rows), 'newRenders': len(new),
        'reusedBaselineRenders': len(rows) - len(new),
        'newLocalAsrEvaluations': sum(not r['transcriptReused'] for r in words['results']) + len(teacher['results']),
        'sourceUnits': len(selection['units']), 'groups': groups,
        'selection': {
            'report': study['selectionReport'],
            'criterion': 'Source-only spectral stationarity; unchanged pitch/energy eligibility and short-condition sustain constraint',
            'medianOriginalShapeVariationDb': stats.median(r['originalShapeVariationDb'] for r in selection['units']),
            'medianSelectedShapeVariationDb': stats.median(r['chosen']['shapeVariationDb'] for r in selection['units']),
            'interpretation': 'In-sample optimization of the selection objective; not an independent quality result or vowel recognition',
        },
        'streamingNewRendersOnly': {
            'cases': len(streams), 'fixedWaveformDelayMs': 20.,
            'medianFirstPhysicalPacketComputeMs': stats.median(r['firstPhysicalPacketComputeMs'] for r in streams),
            'maximumFirstPhysicalPacketComputeMs': max(r['firstPhysicalPacketComputeMs'] for r in streams),
            'maximumPacketP95Ms': max(r['packetP95Ms'] for r in streams),
            'maximumSinglePacketMs': max(r['maximumPacketMs'] for r in streams),
            'packetsOver5Ms': sum(r['packetsOver5Ms'] for r in streams),
            'maximumCachedWaveformSamples': max(r['maximumCachedWaveformSamples'] for r in streams),
            'allClosureChecksPassed': all(r['closedPreventsOutput'] and r['closedClearsHistory'] for r in streams),
            'timingScope': streams[0]['timingScope'],
            'baselineTimingReusedAndExcluded': True,
        },
        'allFinite': all(r['finite'] for r in rows),
        'overFullScaleSamples': sum(r['overFullScaleSamples'] for r in rows),
        'maximumRawPeak': max(r['rawPeak'] for r in rows),
        'decoderAndSourcesUnchanged': study['decoderAndSourcesUnchanged'],
        'modelSha256': study['modelSha256'],
        'pretrainedKokoroStretch': {
            'cases': len(teacher['results']), 'groups': teacher_groups,
            'allFinite': all(r['finite'] for r in teacher['results']),
            'maximumRawPeak': max(r['rawPeak'] for r in teacher['results']),
            'medianOfflineWholeRenderSeconds': stats.median(r['wholeRenderSeconds'] for r in teacher['results']),
            'report': 'evidence/neural-articulation/kokoro-reference-v1/evaluation.json',
            'decision': 'Rejected as a valid sustained-vowel quality target: all 18 miss the intended ASR sound class and both trackers often lose voicing. This tests our experimental duration/pitch adaptation, not normal Kokoro quality, and cannot establish that our instrument is better than Kokoro.',
        },
        'auditions': {'manifest': 'public/research/sustain-comparison/manifest.json',
                     'groups': len(audition['groups']), 'files': len(audition['files']),
                     'page': 'http://127.0.0.1:8791/voicebox-sustain-lab.html',
                     'mainReference': 'Original unmodified Kokoro speech, with original duration and pitch',
                     'failedKokoroStretchKeptInExtras': True, 'autoplay': False},
        'tests': {'passed': int(match.group(1)), 'log': str(log.relative_to(ROOT))},
        'decision': 'Keep fixed-cycle sustain as an experimental option. It improves development-panel ASR counts, while span reselection alone does not. Naturalness and identity are not accepted; validate on fresh source material before selecting it as a default.',
        'nextExperiment': 'Freeze the candidate selection and renderer, then test fresh source words and vowel shapes with both original PCM and neural reconstruction controls. Preserve failures; assess naturalness separately from ASR. Input understanding remains deferred.',
        'limitations': study['limits'] + words['limits'] + [
            'Lower within-vowel spectral variation can indicate an unnaturally periodic buzz; it is not a naturalness score.',
            'The same nine development units motivated and evaluated the change; no independent generalization claim.',
            'Original Kokoro words already scored 6/9 on this diagnostic. Exceeding that count on altered duration or pitch does not demonstrate higher quality.',
            'Three original source units have no formant estimate under the fixed margins; missing estimates remain absent.',
            'No neural weights were trained in this study; the captured synthetic units and frozen decoder remain necessary.',
            'No new speech generation, general cloning, listening acceptance or Kokoro-level output quality is established.',
        ],
        'listeningAccepted': False, 'completionProven': False,
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print('Comparisons', len(rows), 'new renders', len(new), 'teacher attempts', len(teacher['results']),
          'new ASR', summary['newLocalAsrEvaluations'], 'tests', summary['tests']['passed'])
    for r in groups:
        print(r['mode'], r['method'], r['homophoneEquivalentMatches'], '/', r['cases'],
              'shape variation', round(r['medianHeldShapeVariationDb'], 3))
    print('streaming', summary['streamingNewRendersOnly'])
    print('teacher voiced fractions', [(r['mode'], r['medianPraatVoicedFraction'], r['medianWorldVoicedFraction']) for r in teacher_groups])


if __name__ == '__main__':
    main()
