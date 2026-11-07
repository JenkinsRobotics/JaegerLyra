"""Record phase and pulse-interval variation without selecting output cases."""
import json
from pathlib import Path
import numpy as np
from pulse_basis_instrument import OFFSETS

ROOT=Path(__file__).resolve().parents[2]


def main():
    fit=json.loads((ROOT/'evidence/pulse-basis/v1/fit-report.json').read_text())
    rows=[]
    for source in fit['sources']:
        period=source['referencePeriodSamples'];tracks=[];coefficients=[]
        with np.load(ROOT/source['capture']) as data:
            for j,segment in enumerate(source['segments']):
                if not segment['used']:continue
                values=data['segment_'+str(j)];centers=data['centers_'+str(j)]
                z=values@np.exp(-2j*np.pi*OFFSETS/period);coefficients.append(z)
                tracks.append({'split':segment['split'],'pulses':len(values),
                    'sourceStartSample':segment['sourceStartSample'],'sourceEndSampleExclusive':segment['sourceEndSampleExclusive'],
                    'pulseIntervalSamplesP0P50P100':np.percentile(np.diff(centers),(0,50,100)).tolist(),
                    'medianFundamentalMagnitude':float(np.median(abs(z))),
                    'meanPhaseRadians':float(np.angle(np.mean(z))),
                    'circularPhaseConcentration':float(abs(np.mean(z/np.maximum(abs(z),1e-30))))})
        z=np.concatenate(coefficients)
        rows.append({'source':source['name'],'referencePeriodSamples':period,
            'pooledPhaseConcentration':float(abs(np.mean(z/np.maximum(abs(z),1e-30)))),
            'segments':tracks})
    report={'scope':'Source-only phase/interval diagnostic on all retained captured pulse tracks',
        'fitReport':'evidence/pulse-basis/v1/fit-report.json','sources':rows,
        'limits':['Fundamental measured at the bank reference period; source pitch changes bias this phase diagnostic.',
            'Large pulse intervals can include missed source pulses or low-frequency phonation; no automatic correction is asserted.',
            'This diagnostic identifies inconsistent template phase/period; it does not prove which correction will sound better.']}
    file=ROOT/'evidence/pulse-basis/v1/alignment-diagnostic.json'
    file.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for r in rows:print(r['source'],'pooled phase concentration',round(r['pooledPhaseConcentration'],3))


if __name__=='__main__':main()
