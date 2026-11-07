"""Output-phase evidence with numerical fidelity separate from listening acceptance."""
import json,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'evidence/neural-waveform/axis-v1'


def main():
    evaluation=json.loads((BASE/'evaluation/report.json').read_text())
    stream=json.loads((BASE/'streaming/report.json').read_text())
    development=json.loads((BASE/'development.json').read_text())
    random=json.loads((ROOT/'evidence/neural-waveform/fidelity-v2/development.json').read_text())
    rows=evaluation['results']
    if not evaluation['complete'] or len(rows)!=evaluation['expectedCases'] or not stream['complete']:
        raise ValueError('Output evaluation incomplete')
    keys={(r['fixture'],r['mode']) for r in rows}
    expected={(r['fixture'],m) for r in evaluation['originals'] for m in evaluation['models']}
    if keys!=expected or len(keys)!=len(rows):raise ValueError('Missing or duplicated comparisons')
    groups=[]
    for kind in ('fresh-kokoro','human-regression','authored'):
        originals=[r for r in evaluation['originals'] if r['kind']==kind]
        for mode in evaluation['models']:
            matching=[r for r in rows if r['kind']==kind and r['mode']==mode]
            good=[r for r in matching if r['status']=='ok']
            row={'kind':kind,'mode':mode,'cases':len(matching),'executionFailures':len(matching)-len(good)}
            for key in ('waveformSnrDb','spectralErrorDb','identityCosine','decoderRealTimeFactor'):
                values=[r[key] for r in good if r.get(key) is not None]
                row['mean'+key[0].upper()+key[1:]]=statistics.mean(values) if values else None
            row['wordErrors']=sum(r['wordErrors'] for r in good if 'wordErrors' in r) if kind!='authored' else None
            row['referenceWords']=sum(r['referenceWords'] for r in good if 'referenceWords' in r) if kind!='authored' else None
            row['originalWordErrors']=sum(r['recognition']['wordErrors'] for r in originals) if kind!='authored' else None
            row['originalMeanIdentity']=statistics.mean(r['identityCosine'] for r in originals) if kind!='authored' else None
            groups.append(row)
    summary={'scope':'Output quality phase only; input understanding and auditory learning deferred by user',
        'candidate':evaluation['candidate'],'candidateSha256':evaluation['candidateSha256'],
        'completedCases':len(rows),'successfulExecutions':sum(r['status']=='ok' for r in rows),
        'groups':groups,
        'development':{'trainingClips':20,'developmentClips':4,'selection':development['selection'],
            'selected':development['selected'],
            'randomArchitectureResults':[{k:r[k] for k in ('name','meanDevelopmentSnrDb')} for r in random['results']],
            'structuredArchitectureResults':[{k:r[k] for k in ('name','meanDevelopmentSnrDb')} for r in development['results']]},
        'streaming':{'cases':len(stream['results']),
            'maximumFrameMsP95':max(r['frameMsP95'] for r in stream['results']),
            'maximumComputeRealTimeFactor':max(r['computeRealTimeFactor'] for r in stream['results']),
            'maximumDifferenceFromBatch':max(r['batchMaximumError'] for r in stream['results']),
            'maximumStoredWaveformSamples':stream['maximumStoredWaveformSamples'],
            'weightsUnchanged':all(r['weightsUnchanged'] for r in stream['results']),
            'scope':stream['latencyLimit']},
        'listening':{'accepted':False,'state':'Ready for listener comparison; no acceptance reported',
            'url':'http://127.0.0.1:8791/voicebox-output-lab.html',
            'manifest':'public/research/voicebox-output/manifest.json','allFreshClipsIncluded':True},
        'tests':{'audioNativeResearch':71,'log':'evidence/audio-native-axis-tests.txt'},
        'limits':[
            '385 scalar controls every 10 ms, encoded from target recordings; not compression or phonetic/articulatory disentanglement.',
            'No original PCM/residual bypass in neural inference, but the encoded coordinates carry target waveform detail.',
            'Speech control generation, independent singing commands and short-reference cloning remain unproven by this reconstruction test.',
            'New code and fitted weights use synthetic Kokoro teacher audio; not independently sourced or clean-room weights.',
            'Structured hidden units plus batch least-squares output fitting; no global backpropagation and no local-PC refinement for this candidate.',
            'All reported comparisons are objective diagnostics, not proof of perceptual parity.',
            'The CPU stream receives precomputed controls; no device or full conversational latency claim.',
        ],
        'appVoice':'Kokoro Heart remains unchanged',
        'completionProven':False}
    (BASE/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'cases':len(rows),'summary':str(BASE/'summary.json'),'listeningAccepted':False}))


if __name__=='__main__':main()
