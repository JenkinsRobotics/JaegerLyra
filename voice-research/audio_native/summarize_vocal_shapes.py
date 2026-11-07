"""Summarize the vowel-transition comparison without declaring quality parity."""
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vocal-shapes/v1'


def main():
    report=json.loads((OUT/'report.json').read_text())
    if not report['complete'] or len(report['results'])!=report['expectedCases']:
        raise ValueError('Incomplete transition study')
    sources={p['speaker']:p['sourceFormantsHz'] for p in report['profiles']}
    methods=[]
    for method,description in report['methods'].items():
        rows=[r for r in report['results'] if r['method']==method and r['status']=='ok']
        drift=[]
        for row in rows:
            for i,(actual,source) in enumerate(zip(row['plateauFormantsHz'],sources[row['speaker']])):
                for j,(value,reference) in enumerate(zip(actual,source)):
                    if value is not None and reference is not None:
                        drift.append({'percent':100*abs(value-reference)/reference,'speaker':row['speaker'],
                            'mode':row['mode'],'vowel':report['shapeOrder'][i],'formant':j+1,
                            'sourceHz':reference,'measuredHz':value})
        item={'method':method,'description':description,'successfulExecutions':len(rows),
            'meanShapeSelfConsistencyErrorDb':statistics.mean(r['shapeSelfConsistency']['meanAbsoluteDb'] for r in rows),
            'meanPlateauNearestShapeAgreement':statistics.mean(r['shapeSelfConsistency']['plateauNearestShapeAccuracy'] for r in rows),
            'shapeMetricLimit':'Shares WORLD analysis with the spectral reference; not independent naturalness or phoneme-recognition evidence',
            'medianEstimatedFormantDriftPercent':statistics.median(r['percent'] for r in drift),
            'worstEstimatedFormantDrift':max(drift,key=lambda r:r['percent']),
            'maximumRawPeak':max(r['rawPeak'] for r in rows),
            'overFullScaleSamples':sum(r['overFullScaleSamples'] for r in rows),
            'maximumWholeRenderSeconds':max(r['renderSeconds'] for r in rows),
            'timingScope':report['timingScope']}
        for estimator in ('praat','world'):
            item[estimator]={'medianPerCaseP95ErrorCents':statistics.median(r[estimator]['p95ErrorCents'] for r in rows),
                'worstPerCaseP95ErrorCents':max(r[estimator]['p95ErrorCents'] for r in rows),
                'minimumVoicedFraction':min(r[estimator]['voicedFraction'] for r in rows)}
        methods.append(item)
    summary={'scope':report['scope'],'model':report['model'],'modelSha256':report['modelSha256'],
        'completedCases':len(report['results']),'weightsUnchanged':report['weightsUnchanged'],
        'methods':methods,'tests':{'passed':94,'log':'evidence/audio-native-vocal-shape-tests.txt'},
        'decision':'Keep spectral interpolation as a controllable external reference, not a selected replacement for the neural output. Better pitch and WORLD self-consistency coexist with worse formant-estimator outliers; no quality winner or Kokoro parity is proven.',
        'limitations':report['limits']+['The WORLD reference renders offline and is not evidence of streaming synthesis.',
            'The nine profiles are captured synthetic-vowel descriptors, not learned phonemes or general identity embeddings.',
            'No perceptual acceptance, independent new words or human-reference cloning is established.'],
        'completionProven':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('Saved all 24 transition cases; neither method selected as a quality winner')


if __name__=='__main__':main()
