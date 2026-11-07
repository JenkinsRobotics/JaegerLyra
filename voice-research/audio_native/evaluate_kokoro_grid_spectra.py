"""Compare captured excitation and upsampler-intervention steady-note spectra."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/kokoro-control-voicebox/upsampler-grid-v1'


def main():
    report=json.loads((OUT/'report.json').read_text())
    if not report['complete']:raise ValueError('Incomplete grid ablation')
    cases=[{**r,'method':'excitation','name':r['name']+'-excitation'} for r in report['sourceCaptures'] if r['mode']=='native']
    cases += [r for r in report['results'] if r['mode']=='native']
    results=[]
    for row in cases:
        file=ROOT/row['file']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Changed input')
        rate,audio=wavfile.read(file);assert rate==24000
        start=row['plan']['holdStartSeconds']+.1;end=row['plan']['holdEndSeconds']-.1
        audio=np.asarray(audio[round(start*rate):round(end*rate)],dtype=float)
        if len(audio)<8192:raise ValueError('Insufficient steady region')
        hz,power=welch(audio,rate,nperseg=8192,noverlap=4096,nfft=32768)
        f0=row['score']['hz'][0];harmonics=np.arange(1,5)*f0
        harmonic_mask=np.any(abs(hz[:,None]-harmonics[None,:])<=4,axis=1)
        offsets=np.r_[harmonics-80,harmonics+80];offsets=offsets[(offsets>10)&(offsets<8000)]
        sideband_mask=np.any(abs(hz[:,None]-offsets[None,:])<=4,axis=1)&~harmonic_mask
        fundamental=np.flatnonzero(abs(hz-f0)<=.05*f0);peak=fundamental[np.argmax(power[fundamental])]
        spectra=OUT/(row['name']+'-spectrum.npz');np.savez(spectra,hz=hz,power=power,harmonicMask=harmonic_mask,sidebandMask=sideband_mask)
        results.append({k:row[k] for k in ('name','speaker','word','method','file','sha256')}|
            {'requestedF0':f0,'fundamentalPeakHz':float(hz[peak]),
             'fundamentalPeakErrorCents':float(1200*abs(np.log2(hz[peak]/f0))),
             'sidebandToFourHarmonicsDb':float(10*np.log10(max(float(power[sideband_mask].sum()),1e-30)/max(float(power[harmonic_mask].sum()),1e-30))),
             'spectrum':str(spectra.relative_to(ROOT))})
    groups=[]
    for method in ('excitation','baseline','half','full'):
        rows=[r for r in results if r['method']==method]
        groups.append({'method':method,'cases':len(rows),
            'medianSidebandToFourHarmonicsDb':float(np.median([r['sidebandToFourHarmonicsDb'] for r in rows])),
            'maximumSidebandToFourHarmonicsDb':max(r['sidebandToFourHarmonicsDb'] for r in rows),
            'maximumFundamentalPeakErrorCents':max(r['fundamentalPeakErrorCents'] for r in rows)})
    pairs=[]
    for row in results:
        if row['method']!='full':continue
        before=next(r for r in results if r['method']=='baseline' and r['speaker']==row['speaker'] and r['word']==row['word'])
        pairs.append({'speaker':row['speaker'],'word':row['word'],
            'sidebandToHarmonicChangeDb':row['sidebandToFourHarmonicsDb']-before['sidebandToFourHarmonicsDb']})
    result={'scope':'Same steady-note spectrum metric before and after a first-upsampler intervention, plus captured harmonic excitation',
        'analysis':'Welch 8192 samples/4096 overlap/32768 FFT; first four requested harmonics and +/-80 Hz sidebands, +/-4 Hz bands, harmonic overlaps excluded. Hold excludes 100 ms at both boundaries.',
        'groups':groups,'results':results,'pairedFullCorrection':pairs,
        'medianPairedSidebandChangeDb':float(np.median([r['sidebandToHarmonicChangeDb'] for r in pairs])),
        'sidebandsReducedCases':sum(r['sidebandToHarmonicChangeDb']<0 for r in pairs),
        'limits':['The intervention can change useful feature detail as well as the grid response; lower sidebands alone do not establish better sound.',
            'Steady-note diagnostic only; melody pitch and word checks are evaluated separately.',
            'No new neural training, independent waveform engine, cloning, streaming or listening acceptance is established.'],
        'complete':len(results)==72,'listeningAccepted':False}
    (OUT/'spectral-evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(groups);print('Paired sideband change',result['medianPairedSidebandChangeDb'],'dB; reduced',result['sidebandsReducedCases'],'/18')


if __name__=='__main__':main()
