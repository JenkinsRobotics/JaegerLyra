"""Keep output preservation, intelligibility, timing and acceptance separate."""
import json,re,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'evidence/neural-waveform/axis-detail-v2'


def main():
    evaluation=json.loads((BASE/'evaluation/report.json').read_text())
    stream=json.loads((BASE/'streaming/report.json').read_text())
    development=json.loads((BASE/'development.json').read_text())
    rows=evaluation['results'];originals={r['fixture']:r for r in evaluation['originals']}
    expected={(key,mode) for key in originals for mode in evaluation['models']}
    actual=[(r['fixture'],r['mode']) for r in rows]
    if (not evaluation['complete'] or not stream['complete'] or not development['complete']
            or len(rows)!=evaluation['expectedCases'] or len(set(actual))!=len(actual) or set(actual)!=expected):
        raise ValueError('Incomplete detail evaluation')
    groups=[]
    for kind in ('fresh-kokoro','sung-heldout','human-regression','authored'):
        source=[r for r in originals.values() if r['kind']==kind]
        for mode in evaluation['models']:
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
                words=lambda text:re.findall(r"[a-z0-9']+",text.lower())
                group['normalizedTranscriptChanges']=sum(words(r['text'])!=words(originals[r['fixture']]['recognition']['text']) for r in good)
            groups.append(group)
    summary={'scope':'Output phase only; listening/input understanding remains deferred',
        'candidate':evaluation['candidate'],'candidateSha256':evaluation['candidateSha256'],
        'baseline':evaluation['baseline'],'baselineSha256':evaluation['baselineSha256'],
        'completedCases':len(rows),'successfulExecutions':sum(r['status']=='ok' for r in rows),'groups':groups,
        'development':{'conditions':[{k:v for k,v in r.items() if k!='development'} for r in development['results']],
            'selected':development['selected'],'selection':development['selection'],
            'frozenRepresentation':True,'teacherTrainingClips':20,'developmentClips':4},
        'errorDiagnosis':'evidence/neural-waveform/axis-v1/error-analysis.json',
        'streaming':{'cases':len(stream['results']),
            'maxFrameMsP95':max(r['frameMsP95'] for r in stream['results']),
            'maxComputeRealTimeFactor':max(r['computeRealTimeFactor'] for r in stream['results']),
            'maximumDifferenceFromBatch':max(r['batchMaximumError'] for r in stream['results']),
            'weightsUnchanged':all(r['weightsUnchanged'] for r in stream['results']),
            'scope':stream['latencyLimit']},
        'listening':{'accepted':False,'state':'No listener acceptance reported',
            'url':'http://127.0.0.1:8791/voicebox-output-detail-lab.html',
            'manifest':'public/research/voicebox-output-v2/manifest.json',
            'previousComparisonUnchanged':'http://127.0.0.1:8791/voicebox-output-lab.html'},
        'tests':{'audioNativeResearch':72,'log':'evidence/audio-native-output-detail-tests.txt'},
        'limits':[
            'The decoder reconstructs target-derived dense coordinates; no claim of independent words, melody or phonetic control.',
            'The frozen 384-coordinate encoder still loses some high-frequency detail; the new target projection reduces decoder error without increasing bandwidth.',
            'A recorded song decoded faithfully does not prove singing control or voice cloning; source SoulX identity is already unlike Kokoro in the reference diagnostic.',
            'Three fresh speech recognition errors also occur in the originals, with identical erroneous transcripts; they were not discarded.',
            'Training uses existing Kokoro synthetic teacher audio and a learned linear reconstruction reference; not independently sourced or clean-room weights.',
            'No global backpropagation; batch fitted output layer, without a local-PC improvement claim.',
            'Objective fidelity is improved; Kokoro-level perceptual quality remains unconfirmed.',
        ],'completionProven':False}
    (BASE/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'cases':len(rows),'summary':str(BASE/'summary.json'),'completionProven':False}))


if __name__=='__main__':main()
