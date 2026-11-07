"""Audit captured-articulation control capability without calling it finished TTS."""
import json,re,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'evidence/neural-performance/v2'


def main():
    report=json.loads((BASE/'report.json').read_text())
    old=json.loads((BASE.parent/'v1/report.json').read_text())
    checks=json.loads((BASE/'control-checks.json').read_text())
    leakage=json.loads((BASE.parent/'pitch-leakage.json').read_text())
    if not report['complete'] or len(report['results'])!=42 or len(checks['results'])!=12:
        raise ValueError('Incomplete performance evidence')
    source={r['fixture']:r for r in report['originals']};modes=[]
    names=('neutral','up-three','down-three','longer','faster','vibrato','four-notes')
    expected={(name,mode) for name in source for mode in names}
    actual=[(r['fixture'],r['mode']) for r in report['results']]
    if set(actual)!=expected or len(set(actual))!=len(actual):raise ValueError('Missing/duplicate cases')
    words=lambda text:re.findall(r"[a-z0-9']+",text.lower())
    for mode in names:
        cases=[r for r in report['results'] if r['mode']==mode];good=[r for r in cases if r['status']=='ok']
        row={'mode':mode,'cases':len(cases),'executionFailures':len(cases)-len(good)}
        for key in ('medianPitchErrorCents','p95PitchErrorCents','pitchVoicingAgreement','identityCosine',
                    'neuralVsSameEditOriginalSnrDb','computeRealTimeFactor'):
            values=[r[key] for r in good if r.get(key) is not None]
            row['mean'+key[0].upper()+key[1:]]=statistics.mean(values) if values else None
        row['wordErrors']=sum(r['wordErrors'] for r in good)
        row['sameEditOriginalWordErrors']=sum(r['sameEditOriginalRecognition']['wordErrors'] for r in good)
        row['normalizedTranscriptChanges']=sum(words(r['text'])!=words(source[r['fixture']]['recognition']['text']) for r in good)
        modes.append(row)
    summary={'scope':'Output controls over captured neural articulation; input understanding deferred',
        'model':report['model'],'modelSha256':report['modelSha256'],'rendererVersion':2,
        'completedCases':len(report['results']),'successfulExecutions':sum(r['status']=='ok' for r in report['results']),
        'originalWordErrors':sum(r['recognition']['wordErrors'] for r in source.values()),
        'originalMeanIdentity':statistics.mean(r['identityCosine'] for r in source.values()),'modes':modes,
        'neutralMaximumDifferenceFromNeural':max(r.get('neutralMaximumDifferenceFromNeural',0) for r in report['results']),
        'maximumOfflineRenderSeconds':max(r['renderSeconds'] for r in report['results'] if r['status']=='ok'),
        'weightsUnchanged':report['weightsUnchanged'],
        'firstVersionFailures':[r for r in old['results'] if r['status']!='ok'],
        'pitchLeakageRegression':leakage,'controlCrossChecks':checks,
        'tests':{'audioNativeResearch':79,'log':'evidence/audio-native-neural-performance-tests.txt'},
        'audition':'evidence/neural-performance/v2/audition/manifest.json',
        'limits':[
            'Captured articulation is encoded from source audio; new words and short-reference cloning remain unproved.',
            'Pulse timing control is classical DSP around the frozen neural waveform model; not a learned phonetic or biological control system.',
            'Praat pulse analysis is offline compilation. Whole-performance rendering currently decodes the full source before editing.',
            'The output controller is not yet streaming; neural-only streaming measurements do not transfer to it.',
            'Large WORLD discrepancies outside stable pulse interiors remain in the full results; the secondary interior scope covers only about 28–33% of all clip frames.',
            'Raw and robust vibrato estimates are both retained; robust fitting does not erase tracking/transition errors.',
            'Speaker-similarity falls under some edits; accurate notes and preserved transcripts do not prove natural singing or identity preservation.',
            'Existing GPLv3 Praat/Parselmouth is used for offline analysis; teacher-derived neural weights and this inspected-source research are not clean-room provenance.',
            'Kokoro remains the accepted app voice; neither the output-quality goal nor listening acceptance is complete.',
        ],'completionProven':False}
    (BASE/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'cases':42,'summary':str(BASE/'summary.json'),'completionProven':False}))


if __name__=='__main__':main()
