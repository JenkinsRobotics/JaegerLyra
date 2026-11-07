"""Keep frozen fresh-source evidence separate from the subsequent capture fix."""
import hashlib
import json
import re
import statistics as stats
from pathlib import Path
from summarize_neural_articulation import pitch_group

ROOT=Path(__file__).resolve().parents[2]
FRESH=ROOT/'evidence/neural-articulation/fresh-sustain-v1'
REPAIR=ROOT/'evidence/neural-articulation/variable-span-v1'


def main():
    fresh=json.loads((FRESH/'summary.json').read_text())
    report=json.loads((REPAIR/'report.json').read_text())
    words=json.loads((REPAIR/'word-report.json').read_text())
    audition=json.loads((FRESH/'audition/manifest.json').read_text())
    if not report['complete'] or not words['complete']:raise ValueError('Incomplete repair evidence')
    for row in words['results']:
        if hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Changed repair audio')
    for row in audition['files']:
        if hashlib.sha256((ROOT/row['source']).read_bytes()).hexdigest()!=row['sourceSha256']:raise ValueError('Changed audition source')
        if not (ROOT/row['file']).is_file():raise ValueError('Missing audition')
    groups=[]
    for mode in ('source','decoded','native','melody','short-native'):
        for method in ([mode] if mode in ('source','decoded') else ('neural-loop','neural-fixed','pcm-loop','pcm-fixed')):
            lexical=[r for r in words['results'] if r['mode']==mode and r['method']==method]
            acoustic=[r for r in report['results'] if r['mode']==mode and r['method']==method]
            group={'mode':mode,'method':method,'cases':len(lexical),
                'exactSpellingMatches':sum(r['exactSpellingMatch'] for r in lexical),
                'transcripts':[{'name':r['name'],'transcript':r['transcript']} for r in lexical]}
            if acoustic:group.update(praat=pitch_group(acoustic,'praat'),world=pitch_group(acoustic,'world'))
            groups.append(group)
    streams=[r['streaming'] for r in report['results'] if r['method'].startswith('neural')]
    repair={'scope':report['scope'],'expectedFailedSources':report['expectedSources'],
        'recoveredSources':len(report['units']),'remainingFailures':report['failures'],
        'completedRenders':len(report['results']),'localAsrEvaluations':len(words['results']),
        'captureFix':'Search 60–150 ms endpoints as well as starts; retain existing energy, pulse-count and period-ratio bounds. Select source spectral stationarity with complete windows and short-sustain support.',
        'selectedSpanSeconds':[r['sourceSpanSeconds'] for r in report['units']],
        'groups':groups,'allFinite':all(r['finite'] for r in report['results']),
        'maximumRawPeak':max(r['rawPeak'] for r in report['results']),
        'overFullScaleSamples':sum(r['overFullScaleSamples'] for r in report['results']),
        'streamingNeuralOnly':{'cases':len(streams),'fixedWaveformDelayMs':20.,
            'medianFirstPacketComputeMs':stats.median(r['firstPhysicalPacketComputeMs'] for r in streams),
            'maximumPacketMs':max(r['maximumPacketMs'] for r in streams),
            'packetsOver5Ms':sum(r['packetsOver5Ms'] for r in streams),'scope':streams[0]['timingScope']},
        'decision':'Keep the endpoint-search repair as an optional capture path. It recovers four known failures but does not establish naturalness or held-out generalization.',
        'limits':report['limits'],'listeningAccepted':False,'completionProven':False}
    (REPAIR/'summary.json').write_text(json.dumps(repair,indent=2,allow_nan=False)+'\n')
    log=ROOT/'evidence/audio-native-variable-span-tests.txt';contents=log.read_text();match=re.search(r'Ran (\d+) tests',contents)
    if match is None or not contents.rstrip().endswith('OK'):raise ValueError('Missing successful tests')
    summary={'scope':'Fresh-source sustain validation plus separately labelled post-validation capture repair',
        'freshValidation':fresh,'postValidationRepair':repair,
        'totalNewRenders':fresh['completedRenders']+repair['completedRenders'],
        'totalLocalAsrEvaluations':fresh['localAsrEvaluations']+repair['localAsrEvaluations'],
        'tests':{'passed':int(match.group(1)),'log':str(log.relative_to(ROOT))},
        'auditions':{'manifest':str((FRESH/'audition/manifest.json').relative_to(ROOT)),
            'files':len(audition['files']),'groups':len(audition['groups']),'labelledConcatenations':len(audition['reels']),
            'scope':'All voices and words, with frozen-validation and repair-development labels; original, trimmed, reconstructed and both sustain methods retained for neural/PCM comparison. No autoplay or listening acceptance.'},
        'decision':'Do not promote fixed-cycle sustain as a quality improvement. Its ASR advantage on the original development words did not generalize; source words and silence trimming also produce unstable single-word transcripts. The PCM control matches the neural methods\' aggregate counts, with 36.50 dB median neural-to-PCM controlled-output SNR, directing the next experiment toward acoustic control trajectories rather than decoder size.',
        'nextExperiment':'Audit ordinary pretrained Kokoro inference and its voiced/noise controls before testing an interior-vowel control hold that preserves onset and release. Compare this explicitly pretrained reference against the failed full-vowel time warp and captured-cycle instrument. Keep independent waveform research distinct; validate sound quality through matched listening, not single-word ASR alone. Input understanding remains deferred.',
        'completionProven':False,'listeningAccepted':False,'appDefaultChanged':False}
    (FRESH/'combined-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('Frozen fresh sources',fresh['eligibleSources'],'/',fresh['sourceCount'],'subsequent recovered',repair['recoveredSources'])
    print('New renders',summary['totalNewRenders'],'ASR',summary['totalLocalAsrEvaluations'],'tests',summary['tests']['passed'])
    print('Repair streaming',repair['streamingNeuralOnly'])


if __name__=='__main__':main()
