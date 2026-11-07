"""Combine frozen fresh-vowel results with the labelled generator ablation."""
import json
import re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'evidence/kokoro-control-voicebox/fresh-vowels-v1'
FOLLOW=ROOT/'evidence/kokoro-control-voicebox/generator-hold-v1'


def main():
    reports=[]; evaluations=[]; summaries=[]
    for folder in (BASE,FOLLOW):
        reports.append(json.loads((folder/'report.json').read_text()))
        evaluations.append(json.loads((folder/'evaluation.json').read_text()))
        summaries.append(json.loads((folder/'summary.json').read_text()))
    if not all(x['complete'] for x in reports+evaluations): raise ValueError('Incomplete evidence')
    manifest=json.loads((FOLLOW/'audition/manifest.json').read_text())
    diagnostic=json.loads((FOLLOW/'grid-sideband-diagnostic.json').read_text())
    test_log=ROOT/'evidence/audio-native-generator-hold-tests.txt'
    test_text=test_log.read_text(); test_count=re.search(r'Ran (\d+) tests',test_text)
    if test_count is None or not test_text.rstrip().endswith('OK'):raise ValueError('Research checks have not passed')
    groups=[]
    for label,report,evaluation in zip(('upstream-control','generator-input'),reports,evaluations):
        for method in ('normal','reference'):
            rows=[r for r in evaluation['results'] if r['method']==method]
            raw=[r for r in report['results'] if r['method']==method]
            groups.append({'heldFeatureStage':label,'normalization':method,'cases':len(rows),
                'bothPitchChecksPassed':sum(r['bothPitchChecksPassed'] for r in rows),
                'praatVoicedAtLeast95Percent':sum(r['praat']['voicedFraction']>=.95 for r in rows),
                'worldVoicedAtLeast95Percent':sum(r['world']['voicedFraction']>=.95 for r in rows),
                'homophoneEquivalentMatches':sum(r['homophoneEquivalentMatch'] for r in rows),
                'overFullScaleSamples':sum(r['overFullScaleSamples'] for r in raw),
                'maximumRawPeak':max(r['rawPeak'] for r in raw),
                'medianHeldRms':float(np.median([r['heldRms'] for r in rows])),
                'p95TrackerDisagreementsAbove600Cents':[{'name':r['name'],'praat':r['praat']['p95ErrorCents'],
                    'world':r['world']['p95ErrorCents']} for r in rows if all(r[n]['p95ErrorCents'] is not None for n in ('praat','world'))
                    and abs(r['praat']['p95ErrorCents']-r['world']['p95ErrorCents'])>600]})
    pairs=[]
    for after in evaluations[1]['results']:
        if after['method']!='reference':continue
        before=next(r for r in evaluations[0]['results'] if
            (r['speaker'],r['word'],r['mode'],r['method'])==(after['speaker'],after['word'],after['mode'],after['method']))
        pairs.append({'speaker':after['speaker'],'word':after['word'],'mode':after['mode'],
            'beforePitchPass':before['bothPitchChecksPassed'],'afterPitchPass':after['bothPitchChecksPassed'],
            'beforeWordMatch':before['homophoneEquivalentMatch'],'afterWordMatch':after['homophoneEquivalentMatch'],
            'beforeTranscript':before['transcript'],'afterTranscript':after['transcript']})
    result={'scope':'Frozen fresh-vowel validation followed by a development ablation of the held pretrained feature stage',
        'freshProtocol':'evidence/kokoro-control-voicebox/fresh-vowels-v1/protocol.json',
        'ablationProtocol':'evidence/kokoro-control-voicebox/generator-hold-v1/protocol.json',
        'freshSources':18,'freshSourceCaptureFailures':reports[0]['failures'],
        'newControlledRenders':sum(len(r['results']) for r in reports),
        'newLocalAsrEvaluations':len(evaluations[0]['sources'])+sum(len(e['results']) for e in evaluations),
        'sourceAsrResultsReusedInAblation':len(evaluations[1]['sources']),
        'ordinarySourceWordMatches':summaries[0]['sourceWordMatches'],
        'nativeReplayMaximumDifference':max(s['nativeMaximumReplayDifference'] for s in summaries),
        'groups':groups,'referenceNormalizedPairs':pairs,
        'pitchPassGains':sum(not r['beforePitchPass'] and r['afterPitchPass'] for r in pairs),
        'pitchPassLosses':sum(r['beforePitchPass'] and not r['afterPitchPass'] for r in pairs),
        'wordMatchGains':sum(not r['beforeWordMatch'] and r['afterWordMatch'] for r in pairs),
        'wordMatchLosses':sum(r['beforeWordMatch'] and not r['afterWordMatch'] for r in pairs),
        'auditions':{'manifest':'evidence/kokoro-control-voicebox/generator-hold-v1/audition/manifest.json',
            'clips':len(manifest['files']),'plots':len(manifest['groups']),'labelledReels':len(manifest['reels']),
            'maximumLevelErrorLu':max(abs(f['targetLufs200MsGate']-f['measuredLufs200MsGate']) for f in manifest['files']),
            'maximumSamplePeak':max(f['samplePeak'] for f in manifest['files']),
            'processing':manifest['processing'],'audioPlayed':False,'listeningAccepted':False},
        'decision':'The earlier normalization improvement does not generalize reliably. Holding the later generator input improves voicing and word diagnostics on this same panel, but residual pitch disagreements and pronunciation failures prevent any quality acceptance. Retain as a labelled pretrained reference.',
        'nextExperiment':'Isolate the 80 Hz sideband contribution from the harmonic source and subsequent generator layers, then test stationary references with the independent waveform instrument. Keep source selection, controller changes and neural reconstruction effects separate. Input understanding remains deferred.',
        'sidebandDiagnostic':{'report':'evidence/kokoro-control-voicebox/generator-hold-v1/grid-sideband-diagnostic.json',
            'cases':diagnostic['cases'],'groups':diagnostic['generatorReferenceGroups'],'exploratory':True},
        'tests':{'research':int(test_count.group(1)),'researchLog':str(test_log.relative_to(ROOT))},
        'limitations':['The fresh protocol is frozen; the subsequent generator intervention is development on the same panel, not fresh validation.',
            'Every synthesized clip still uses pretrained Kokoro weights. No independent waveform-generator training occurs in this experiment.',
            'ASR on isolated sustained words is unstable: ordinary source words match only 11/18. It cannot establish perceived pronunciation quality.',
            'Pitch pass thresholds are diagnostic (both trackers >=95% voiced, p95 error <=50 cents), not naturalness or identity acceptance.',
            'Six raw samples in the upstream reference-normalization condition exceed full scale. Original float files are retained; audition gains avoid PCM clipping.',
            'Per-utterance captured controls/statistics and offline whole-clip decoding do not establish general voice cloning or causal streaming.',
            'Whole-clip compute timings exclude model loading, control preparation, source capture, transport and audio devices.'],
        'appDefaultChanged':False,'listeningAccepted':False,'completionProven':False}
    (FOLLOW/'combined-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print({k:v for k,v in result.items() if k not in ('groups','referenceNormalizedPairs','limitations')})


if __name__=='__main__':main()
