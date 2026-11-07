"""Compare energy modes on the same original PCM, including every failed case."""
import json
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from benchmark_vocal_energy import envelope_error

ROOT=Path(__file__).resolve().parents[2]


def average(values):
    values=[v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lattice',action='store_true')
    args=parser.parse_args()
    out=ROOT/'evidence/vocal-energy'
    if args.lattice:out=out/'normalized-lattice'
    fidelity=json.loads((out/'report.json').read_text())
    summary={'scope':('Normalized lattice after output-energy coupling' if args.lattice else 'Output-energy coupling')+'; not naturalness acceptance or autonomous speech',
        'fidelity':[],'gaps':[],
        'comparisonRule':'Gap comparisons pair matching successful cases against the same original PCM; failed cases and recoveries are shown separately. Feature MSE is not compared across energy representations.'}
    for mode in fidelity['modes']:
        rows=[r for r in fidelity['results'] if r['mode']==mode]
        summary['fidelity'].append({'mode':mode,'cases':len(rows),'wordErrors':sum(r['wordErrors'] for r in rows),
            'meanSpectralErrorDb':average(r['spectralErrorDb'] for r in rows),
            'meanEnvelopeErrorDb':average(r['envelope']['meanAbsoluteDb'] for r in rows),
            'meanIdentityCosine':average(r['identityCosine'] for r in rows),
            'maxRawPeak':max(r['rawPeak'] for r in rows)})
    for suffix in ('','validation/'):
        base=ROOT/'evidence/free-running-controls'
        old_path=base/('output-energy/anchored' if args.lattice else 'anchored')/suffix/'report.json'
        new_path=base/('normalized-lattice/output-energy/anchored' if args.lattice else 'output-energy/anchored')/suffix/'report.json'
        if not new_path.exists():continue
        old=json.loads(old_path.read_text());new=json.loads(new_path.read_text())
        original_files={r['name']:ROOT/r['file'] for r in old['originals']}
        originals={name:sf.read(path)[0] for name,path in original_files.items()}
        key=lambda r:(r['fixture'],r['mode'],r['gapMs'])
        previous={key(r):r for r in old['results']}
        current={key(r):r for r in new['results']}
        if (len(previous)!=len(old['results']) or len(current)!=len(new['results'])
                or current.keys()!=previous.keys()):
            raise ValueError(f'Incomplete, duplicate or mismatched case set: {new_path}')
        pairs=[]
        for r in new['results']:
            before=previous[key(r)]
            row={'fixture':r['fixture'],'mode':r['mode'],'gapMs':r['gapMs'],
                 'beforeStatus':before['status'],'afterStatus':r['status']}
            for label,item in (('before',before),('after',r)):
                if item['status']!='ok':
                    row[label+'Error']=item['error'];continue
                audio,sr=sf.read(ROOT/item['file']);assert sr==24000
                audio/=.5
                original=originals[r['fixture']]
                assert len(audio)==len(original)
                a,b=item['withheldControlRangeSamples'];region=slice(a,min(len(original),b+2400))
                row[label]={'energy':envelope_error(original[region],audio[region]),
                    'spectralErrorDb':item['spectralErrorVsOriginalDb'],
                    'wordErrors':item['wordErrors'],'rawPeak':item['rawPeak']}
            pairs.append(row)
        groups=[]
        for milliseconds in (100,250,500):
            for mode in ('hold','linear-adaptive','pc-frozen','pc-adaptive'):
                rows=[r for r in pairs if r['gapMs']==milliseconds and
                      (r['mode']==mode or r['mode'].startswith(mode+'-'))]
                paired=[r for r in rows if r['beforeStatus']==r['afterStatus']=='ok']
                group={'gapMs':milliseconds,'mode':mode,'cases':len(rows),'pairedSuccessful':len(paired),
                    'beforeFailures':sum(r['beforeStatus']!='ok' for r in rows),
                    'afterFailures':sum(r['afterStatus']!='ok' for r in rows),
                    'recoveredCases':sum(r['beforeStatus']!='ok' and r['afterStatus']=='ok' for r in rows),
                    'newFailures':sum(r['beforeStatus']=='ok' and r['afterStatus']!='ok' for r in rows)}
                for label in ('before','after'):
                    group[label]={'meanEnvelopeErrorDb':average(r[label]['energy']['meanAbsoluteDb'] for r in paired),
                        'meanSpectralErrorDb':average(r[label]['spectralErrorDb'] for r in paired),
                        'meanWordErrors':average(r[label]['wordErrors'] for r in paired),
                        'envelopeValidCases':sum(r[label]['energy']['meanAbsoluteDb'] is not None for r in paired),
                        'spectralValidCases':sum(r[label]['spectralErrorDb'] is not None for r in paired),
                        'maxRawPeak':max((r[label]['rawPeak'] for r in rows if label in r),default=None)}
                groups.append(group)
        comparison={'beforeReport':str(old_path.relative_to(ROOT)),'afterReport':str(new_path.relative_to(ROOT)),
            'cases':len(pairs),'groups':groups,
            'beforeFailures':sum(r['beforeStatus']!='ok' for r in pairs),
            'afterFailures':sum(r['afterStatus']!='ok' for r in pairs),
            'pairs':pairs}
        summary['gaps'].append(comparison)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for row in summary['fidelity']:print(json.dumps(row))
    for comparison in summary['gaps']:
        print(comparison['afterReport'],comparison['cases'],'cases; failures',comparison['beforeFailures'],'->',comparison['afterFailures'])
        for row in comparison['groups']:print(json.dumps(row))


if __name__=='__main__':main()
