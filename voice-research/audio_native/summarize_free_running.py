"""Aggregate every free-running case, retaining failures and all PC seeds."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def summarize(path):
    data=json.loads(path.read_text())
    results=data['results']
    hold={(r['fixture'],r['gapMs']):r for r in results if r['mode']=='hold' and r['status']=='ok'}
    groups=[]
    for gap in (100,250,500):
        for group in ('hold','linear-adaptive','pc-frozen','pc-adaptive'):
            rows=[r for r in results if r['gapMs']==gap and
                  (r['mode']==group or r['mode'].startswith(group+'-'))]
            good=[r for r in rows if r['status']=='ok']
            row={'gapMs':gap,'mode':group,'cases':len(rows),'failed':len(rows)-len(good)}
            for metric in ('featureMse','spectralErrorVsObservedDb','spectralErrorVsOriginalDb',
                           'wordErrors','pitchVoicingAgreement','processingMsP95'):
                valid=[r[metric] for r in good if r[metric] is not None and np.isfinite(r[metric])]
                row['mean_'+metric]=float(np.mean(valid)) if valid else None
                row['validCases_'+metric]=len(valid)
            differences=[r['spectralErrorVsObservedDb']-hold[(r['fixture'],gap)]['spectralErrorVsObservedDb']
                         for r in good if (r['fixture'],gap) in hold
                         and r['spectralErrorVsObservedDb'] is not None
                         and hold[(r['fixture'],gap)]['spectralErrorVsObservedDb'] is not None
                         and np.isfinite(r['spectralErrorVsObservedDb'])
                         and np.isfinite(hold[(r['fixture'],gap)]['spectralErrorVsObservedDb'])]
            row['meanSpectralDifferenceVsHoldDb']=float(np.mean(differences)) if differences else None
            row['spectralBetterThanHoldCases']=sum(d<0 for d in differences)
            row['spectralPairedCases']=len(differences)
            row['exactTranscriptCases']=sum(r['wordErrors']==0 for r in good)
            row['maxRawPeak']=max((r['rawPeak'] for r in good),default=None)
            groups.append(row)
    return {'report':str(path.relative_to(ROOT)),'cases':len(results),
        'failures':[r for r in results if r['status']!='ok'],
        'groups':groups,'observedControlBaselineErrors':[r['wordErrors'] for r in data['originals']]}


def main():
    out=ROOT/'evidence/free-running-controls'
    paths=[out/'report.json',out/'validation/report.json',out/'anchored/report.json',out/'anchored/validation/report.json']
    summary={'scope':'Short acoustic continuation, not semantic or human-perceptual acceptance',
        'aggregation':'Every case; PC groups average three fixed seeds, not independent additional utterances',
        'experiments':[summarize(path) for path in paths if path.exists()]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for experiment in summary['experiments']:
        print(experiment['report'],experiment['cases'],'cases',len(experiment['failures']),'failed')
        for g in experiment['groups']:
            rounded=lambda value:round(value,3) if value is not None else 'n/a'
            print(g['gapMs'],g['mode'],'feature',rounded(g['mean_featureMse']),
                  'spectral',rounded(g['mean_spectralErrorVsObservedDb']),
                  'wordErrors',rounded(g['mean_wordErrors']),
                  'betterThanHold',g['spectralBetterThanHoldCases'],'/',g['cases'])


if __name__=='__main__':main()
