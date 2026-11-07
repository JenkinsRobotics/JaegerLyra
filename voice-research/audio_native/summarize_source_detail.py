"""Keep reconstruction, human speech, streaming and forecasting evidence separate."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'evidence/source-detail'


def mean(values):
    values=[x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def summarize(path,complete=True):
    d=json.loads(path.read_text());rows=d['results']
    if complete and len(rows)!=d['expectedCases']:raise ValueError(f'Incomplete report: {path}')
    result={'file':str(path.relative_to(ROOT)),'completedCases':len(rows),'expectedCases':d['expectedCases'],
        'complete':len(rows)==d['expectedCases'],'modes':[],
        'originalRecognitionErrors':sum(x['originalRecognition']['wordErrors'] for x in d['inputs'] if x.get('originalRecognition')),
        'originalMeanIdentity':mean(x['originalIdentityCosine'] for x in d['inputs'])}
    for mode in d['modes']:
        selected=[r for r in rows if r['mode']==mode];valid=[r for r in selected if r['status']=='ok']
        row={'mode':mode,'cases':len(selected),'failures':len(selected)-len(valid),
            'wordErrors':sum(r['wordErrors'] for r in valid),
            'meanIdentity':mean(r['identityCosine'] for r in valid),
            'spectral':{k:mean(r['spectral'][k] for r in valid) for k in ('all','voiced','unvoiced')},
            'meanEnvelopeErrorDb':mean(r['envelope']['meanAbsoluteDb'] for r in valid),
            'meanMedianPitchErrorCents':mean(r['medianPitchErrorCents'] for r in valid),
            'maxRawPeak':max((r['rawPeak'] for r in valid),default=None)}
        result['modes'].append(row)
    return result


def main():
    result={'scope':'Control-derived reconstruction and DSP numerical correction; no naturalness acceptance or autonomous speech',
        'olderAnalysis':[summarize(BASE/'report.json'),summarize(BASE/'fresh/report.json'),summarize(BASE/'human/report.json',False)],
        'correctedAnalysis':[summarize(BASE/'lpc-v2'/suffix/'report.json') for suffix in ('','fresh','human')],
        'solverAudit':'evidence/source-detail/lpc-v2/solver-audit.json','streaming':[],
        'forecasting':[],
        'limitations':['Synthetic examples and four public human target clips; no listening study',
            'Two human readers are too few to establish general voice quality or cloning',
            'The human set exposed the LPC defect; corrected re-evaluation is not an untouched prospective test',
            'Per-reference streaming uses a different PCA basis from the shared-basis reconstruction comparison',
            'Older measurements used archived LPC v1; old source-coordinate/predictor weights need refitting for v2']}
    audit=json.loads((BASE/'lpc-v2/solver-audit.json').read_text())['results']
    result['solver']={'clips':len(audit),'frames':sum(r['frames'] for r in audit),
        'changedFrames':sum(r['changedFrames'] for r in audit),
        'oldReflectionFailures':sum(r['reflectionConversionFailures']['old'] for r in audit),
        'newReflectionFailures':sum(r['reflectionConversionFailures']['new'] for r in audit),
        'maximumOldCoefficient':max(r['maximumCoefficientMagnitude']['old'] for r in audit),
        'maximumNewCoefficient':max(r['maximumCoefficientMagnitude']['new'] for r in audit),
        'maximumNewNormalEquationResidualP95':max(r['normalizedNormalEquationResidualP95']['new'] for r in audit),
        'maximumFullResidualRoundtripError':max(r['fullResidualRoundtripMaximumDifference'] for r in audit)}
    for mode in ('baseline','colored-noise'):
        path=BASE/'lpc-v2/streaming'/mode/'report.json';d=json.loads(path.read_text())
        result['streaming'].append({'mode':mode,'report':str(path.relative_to(ROOT)),'results':d['results']})
    for suffix in ('','validation'):
        path=ROOT/'evidence/free-running-controls/lpc-v2/anchored'/suffix/'report.json'
        d=json.loads(path.read_text());rows=d['results'];expected=144 if suffix else 48
        if len(rows)!=expected:raise ValueError(f'Incomplete forecasting report: {path}')
        groups=[]
        for gap in (100,250,500):
            for mode in ('hold','linear-adaptive','pc-frozen','pc-adaptive'):
                selected=[r for r in rows if r['gapMs']==gap and r['mode'].startswith(mode)]
                good=[r for r in selected if r['status']=='ok']
                groups.append({'gapMs':gap,'mode':mode,'cases':len(selected),'failures':len(selected)-len(good),
                    'meanWordErrors':mean(r['wordErrors'] for r in good),
                    'spectralVsOriginalDb':mean(r['spectralErrorVsOriginalDb'] for r in good)})
        result['forecasting'].append({'report':str(path.relative_to(ROOT)),'cases':len(rows),
            'failures':[r for r in rows if r['status']!='ok'],'groups':groups})
    result['decision']='Keep corrected LPC as a numerical fix. Leave noise shaping optional and smooth-lattice synthesis experimental. Prioritize the user-requested waveform reconstruction capability gate before more forecast/content-generation work.'
    (BASE/'lpc-v2/summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('olderAnalysis','streaming','forecasting')},indent=2))


if __name__=='__main__':main()
