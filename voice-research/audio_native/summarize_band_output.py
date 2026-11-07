"""Retain fidelity gains, recognition regressions and exact-runtime evidence."""
import json,re,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CAPACITY=ROOT/'evidence/neural-waveform/capacity-v3'
BAND=ROOT/'evidence/neural-waveform/band-v4'


def evaluation_summary(report):
    originals={r['fixture']:r for r in report['originals']};rows=report['results']
    expected={(name,mode) for name in originals for mode in report['models']}
    actual=[(r['fixture'],r['mode']) for r in rows]
    if not report['complete'] or len(rows)!=report['expectedCases'] or len(set(actual))!=len(actual) or set(actual)!=expected:
        raise ValueError('Incomplete output comparison')
    groups=[];changes=[]
    words=lambda text:re.findall(r"[a-z0-9']+",text.lower())
    for kind in dict.fromkeys(r['kind'] for r in rows):
        source=[r for r in originals.values() if r['kind']==kind]
        for mode in report['models']:
            selected=[r for r in rows if r['kind']==kind and r['mode']==mode]
            good=[r for r in selected if r['status']=='ok']
            group={'kind':kind,'mode':mode,'cases':len(selected),'executionFailures':len(selected)-len(good)}
            for key in ('waveformSnrDb','spectralErrorDb','identityCosine','decoderRealTimeFactor'):
                values=[r[key] for r in good if r.get(key) is not None]
                group['mean'+key[0].upper()+key[1:]]=statistics.mean(values) if values else None
            if kind!='authored':
                group['wordErrors']=sum(r['wordErrors'] for r in good)
                group['originalWordErrors']=sum(r['recognition']['wordErrors'] for r in source)
                group['originalMeanIdentity']=statistics.mean(r['identityCosine'] for r in source)
                changed=[r for r in good if words(r['text'])!=words(originals[r['fixture']]['recognition']['text'])]
                group['normalizedTranscriptChanges']=len(changed)
                changes.extend({'fixture':r['fixture'],'mode':mode,'text':r['text'],
                    'originalText':originals[r['fixture']]['recognition']['text']} for r in changed)
            groups.append(group)
    return {'completedCases':len(rows),'successfulExecutions':sum(r['status']=='ok' for r in rows),
        'groups':groups,'allRecognitionChanges':changes}


def main():
    capacity_report=json.loads((CAPACITY/'evaluation/report.json').read_text())
    development=json.loads((CAPACITY/'development.json').read_text())
    compiled=json.loads((CAPACITY/'compiled-streaming/report.json').read_text())
    if not development['complete'] or not compiled['complete']:raise ValueError('Incomplete capacity/compiler evidence')
    optimized={}
    for name in compiled['models']:
        rows=[r for r in compiled['results'] if r['model']==name]
        optimized[name]={'cases':len(rows),'medianPairedSpeedup':statistics.median(r['pairedComputeSpeedup'] for r in rows),
            'maximumDifferenceFromDenseBatch':max(r['modes']['compiled']['maximumDifferenceFromDenseBatch'] for r in rows),
            'maximumCompiledFrameMsP95':max(r['modes']['compiled']['frameMsP95'] for r in rows)}
    capacity={'scope':capacity_report['scope'],**evaluation_summary(capacity_report),
        'candidate':capacity_report['candidate'],'candidateSha256':capacity_report['candidateSha256'],
        'development':{'selected':development['selected'],'selection':development['selection'],
            'conditions':[{k:v for k,v in r.items() if k!='development'} for r in development['results']]},
        'detailDiagnosis':json.loads((CAPACITY/'detail-comparison.json').read_text())['bandErrorReductionDb'],
        'compiledInference':optimized,'decision':'Do not select capacity-v3 as a general replacement: its lower-band error rises and two song words change despite higher mean waveform fidelity.',
        'completionProven':False}
    (CAPACITY/'summary.json').write_text(json.dumps(capacity,indent=2,allow_nan=False)+'\n')
    evaluation=json.loads((BAND/'evaluation/report.json').read_text())
    streaming=json.loads((BAND/'streaming/report.json').read_text())
    assembly=json.loads((BAND/'decoder.json').read_text())
    if not streaming['complete'] or len(streaming['results'])!=15:raise ValueError('Incomplete complementary stream checks')
    rows=streaming['results']
    summary={'scope':'Output-only reconstruction fidelity; new content generation and input understanding remain deferred',
        'candidate':evaluation['candidate'],'candidateSha256':evaluation['candidateSha256'],
        'baseline':evaluation['baseline'],'baselineSha256':evaluation['baselineSha256'],
        **evaluation_summary(evaluation),'assembly':assembly,
        'controlCost':{'priorScalarsPerSecond':38500,'newScalarsPerSecond':44900,
            'sourcePcmSamplesPerSecond':24000,'compressionClaim':False,'compactPhoneticControls':False},
        'streaming':{'cases':len(rows),'maximumDifferenceFromBatch':max(r['maximumDifferenceFromBatch'] for r in rows),
            'maximumFrameMsP95':max(r['frameMsP95'] for r in rows),
            'maximumSingleFrameMs':max(r['frameMsMaximum'] for r in rows),
            'medianFirstOutputComputeMs':statistics.median(r['firstOutput']['computeMs'] for r in rows),
            'maximumComputeRealTimeFactor':max(r['computeRealTimeFactor'] for r in rows),
            'additionalFirLookaheadMs':streaming['additionalFirLookaheadMs'],
            'retainedWaveformStateLimitSamples':streaming['retainedWaveformStateLimitSamples'],
            'stateScope':streaming['stateScope'],'timingScope':streaming['timingScope'],
            'allCancellationChecksPassed':all(r['cancellationPassed'] for r in rows),
            'weightsUnchanged':streaming['weightsUnchanged']},
        'compiledInference':optimized,
        'tests':{'passed':102,'log':'evidence/audio-native-band-tests.txt'},
        'listening':{'accepted':False,'url':'http://127.0.0.1:8791/voicebox-output-band-lab.html',
            'manifest':'public/research/voicebox-output-band/manifest.json','earlierPagesPreserved':True},
        'limitations':[
            'The neural decoders consume dense target-derived controls; no independent new words, notes or short-reference cloning is demonstrated by this fidelity panel.',
            'Both branches use the same 20 synthetic Kokoro training clips; source-derived teacher weights are not clean-room provenance.',
            'No global backpropagation is used; output layers are batch fitted, without a local predictive-coding improvement claim.',
            'The prior SoulX lyric regression is removed, but a fresh score-adapter song has one normalized ASR change on an already misrecognized source phrase.',
            'Singing reconstruction preserves an existing performance; source singing naturalness/identity are not thereby improved.',
            'Objective fidelity gains and faster computation do not establish Kokoro-level perceptual acceptance.',
            'The composition retains two neural networks; its input representation has higher bandwidth and is not a compact voice/phoneme model.'],
        'completionProven':False}
    comparison=[]
    for row in evaluation['results']:
        if row['mode']!='band-preserving-v4' or row['status']!='ok':continue
        base=next(r for r in evaluation['results'] if r['fixture']==row['fixture'] and r['mode']=='structured-v2')
        comparison.append({'fixture':row['fixture'],'kind':row['kind'],
            'snrChangeDb':row['waveformSnrDb']-base['waveformSnrDb'],
            'spectralErrorChangeDb':row['spectralErrorDb']-base['spectralErrorDb']})
    summary['perFixtureComparison']=comparison
    (BAND/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('Saved capacity-v3 and band-v4 summaries, including every recognition change')


if __name__=='__main__':main()
