"""Exploratory steady-note spectra; retained pitch checks are not relabelled.

Tests energy near authored harmonics and sidebands +/- the 80 Hz acoustic
control rate. A correlation with this grid does not by itself prove causality.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/kokoro-control-voicebox/generator-hold-v1'


def main():
    rows=[]
    for stage,folder in [('upstream','fresh-vowels-v1'),('generator','generator-hold-v1')]:
        base=ROOT/'evidence/kokoro-control-voicebox'/folder
        report=json.loads((base/'report.json').read_text()); evaluation=json.loads((base/'evaluation.json').read_text())
        if not report['complete'] or not evaluation['complete']: raise ValueError('Incomplete evaluation')
        for row in report['results']:
            if row['mode']!='native':continue
            file=ROOT/row['file']
            if hashlib.sha256(file.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Changed audio')
            rate,audio=wavfile.read(file);assert rate==24000
            start=row['plan']['holdStartSeconds']+.1;end=row['plan']['holdEndSeconds']-.1
            audio=np.asarray(audio[round(start*rate):round(end*rate)],dtype=float)
            if len(audio)<8192:raise ValueError('Steady region too short for frozen spectrum window')
            hz,power=welch(audio,rate,nperseg=8192,noverlap=4096,nfft=32768)
            f0=row['score']['hz'][0];width=4.;harmonics=np.arange(1,5)*f0
            harmonic_mask=np.any(abs(hz[:,None]-harmonics[None,:])<=width,axis=1)
            offsets=np.r_[harmonics-80,harmonics+80]
            offsets=offsets[(offsets>10)&(offsets<8000)]
            # Exclude legitimate harmonic overlap before summing sideband power.
            sideband_mask=np.any(abs(hz[:,None]-offsets[None,:])<=width,axis=1)&~harmonic_mask
            measured=next(r for r in evaluation['results'] if r['name']==row['name'])
            fundamental=np.flatnonzero(abs(hz-f0)<=.05*f0)
            fundamental_peak=int(fundamental[np.argmax(power[fundamental])])
            spectral_file=OUT/(row['name']+'-steady-spectrum.npz')
            np.savez(spectral_file,hz=hz,power=power,harmonicMask=harmonic_mask,sidebandMask=sideband_mask)
            rows.append({'name':row['name'],'stage':stage,'normalization':row['method'],
                'speaker':row['speaker'],'word':row['word'],'file':row['file'],'sha256':row['sha256'],
                'requestedF0':f0,'fundamentalPeakHz':float(hz[fundamental_peak]),
                'fundamentalPeakErrorCents':float(1200*abs(np.log2(hz[fundamental_peak]/f0))),
                'sidebandToFourHarmonicsDb':float(10*np.log10(max(float(power[sideband_mask].sum()),1e-30)/max(float(power[harmonic_mask].sum()),1e-30))),
                'praatP95Cents':measured['praat']['p95ErrorCents'],'worldP95Cents':measured['world']['p95ErrorCents'],
                'bothPitchChecksPassed':measured['bothPitchChecksPassed'],
                'spectrum':str(spectral_file.relative_to(ROOT))})
    reference=[r for r in rows if r['stage']=='generator' and r['normalization']=='reference']
    groups=[]
    for passed in (True,False):
        subset=[r for r in reference if r['bothPitchChecksPassed']==passed]
        groups.append({'bothPitchChecksPassed':passed,'cases':len(subset),
            'medianSidebandToFourHarmonicsDb':float(np.median([r['sidebandToFourHarmonicsDb'] for r in subset])),
            'maximumFundamentalPeakErrorCents':max(r['fundamentalPeakErrorCents'] for r in subset)})
    result={'scope':'Exploratory 80 Hz grid-sideband diagnostic after fresh-vowel and generator-input studies',
        'analysis':'Steady notes only, inserted hold excluding 100 ms at each end; Welch 8192 samples, 50% overlap, 32768 FFT; +/-4 Hz bands around first four authored harmonics and their +/-80 Hz sidebands. Legitimate harmonic overlap excluded.',
        'cases':len(rows),'results':rows,'generatorReferenceGroups':groups,
        'limits':['Sideband energy can include noise and other modulation; this does not isolate a causal layer.',
            'A strong peak near the requested F0 does not exclude subharmonics or establish perceived pitch/naturalness.',
            'The diagnostic was chosen after inspecting failures; it is exploratory and does not alter the frozen pitch-pass results.',
            'Ordinary speech has changing pitch and is not scored as a stationary note in this diagnostic.'],
        'listeningAccepted':False,'completionProven':False}
    (OUT/'grid-sideband-diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Cases',len(rows),'generator/reference pass and failure groups:',groups)


if __name__=='__main__':main()
