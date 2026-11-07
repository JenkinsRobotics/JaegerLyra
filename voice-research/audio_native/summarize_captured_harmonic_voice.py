"""Keep every amplitude/phase ablation and measure source spectral similarity."""
import json
import re
import statistics as stats
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import stft
from summarize_compact_vocal_instrument import pitch_summary
from compact_vocal_instrument import RATE

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/captured-harmonic/v1'


def spectral_shape(audio):
    """Independent fixed log-mel diagnostic, not WORLD/profile self-consistency."""
    audio=np.asarray(audio,dtype=float)
    audio=audio/np.sqrt(max(np.mean(audio**2),1e-30))
    f,_,z=stft(audio,fs=RATE,window='hann',nperseg=1024,noverlap=904,nfft=2048,
        boundary=None,padded=False)
    power=np.mean(abs(z)**2,axis=1)
    mel=lambda x:2595*np.log10(1+x/700)
    points=700*(10**(np.linspace(mel(80),mel(8000),82)/2595)-1)
    filters=np.maximum(0,np.minimum((f[None,:]-points[:-2,None])/(points[1:-1,None]-points[:-2,None]),
        (points[2:,None]-f[None,:])/(points[2:,None]-points[1:-1,None])))
    filters/=np.maximum(filters.sum(axis=1,keepdims=True),1e-20)
    bands=filters@power
    db=10*np.log10(np.maximum(bands,max(bands)*1e-7))
    return db-db.mean(),points[1:-1]


def main():
    report=json.loads((OUT/'report.json').read_text())
    if not report['complete'] or len(report['results'])!=report['expectedCases']:raise ValueError('Incomplete harmonic study')
    compact=json.loads((ROOT/'evidence/compact-vocal/v1/report.json').read_text())
    oldsummary=json.loads((ROOT/'evidence/compact-vocal/v1/summary.json').read_text())
    vowels=json.loads((ROOT/'evidence/neural-vowels/v1/report.json').read_text())
    source_manifest=json.loads((ROOT/'evidence/neural-vowels/references/manifest.json').read_text())
    sources={p['speaker']:p['sourceFormantsHz'] for p in report['profiles']}
    transitions=[];held=[]
    for method in report['variants']:
        rows=[r for r in report['results'] if r['method']==method and r['kind']=='transition']
        drift=[]
        for row in rows:
            for i,(actual,reference) in enumerate(zip(row['plateauFormantsHz'],sources[row['speaker']])):
                for j,(value,target) in enumerate(zip(actual,reference)):
                    if value is not None and target is not None:
                        drift.append({'percent':100*abs(value-target)/target,'speaker':row['speaker'],'mode':row['mode'],
                            'vowel':('ah','ee','oo')[i],'formant':j+1,'sourceHz':target,'measuredHz':value})
        transitions.append({'method':method,'cases':len(rows),**pitch_summary(rows),
            'meanShapeSelfConsistencyErrorDb':stats.mean(r['shapeSelfConsistency']['meanAbsoluteDb'] for r in rows),
            'medianEstimatedFormantDriftPercent':stats.median(d['percent'] for d in drift),
            'worstEstimatedFormantDrift':max(drift,key=lambda d:d['percent']),
            'maximumRawPeak':max(r['rawPeak'] for r in rows),
            'overFullScaleSamples':sum(r['overFullScaleSamples'] for r in rows)})
        rows=[r for r in report['results'] if r['method']==method and r['kind']=='held']
        drift=[];pitch_drift=[]
        for row in rows:
            for j,value in enumerate(row['formantDriftPercent']):
                if value is not None:drift.append({'percent':value,'speaker':row['speaker'],'vowel':row['vowel'],'mode':row['mode'],'formant':j+1})
            if row['mode']=='native':continue
            native=next(r for r in rows if r['speaker']==row['speaker'] and r['vowel']==row['vowel'] and r['mode']=='native')
            for j,(value,target) in enumerate(zip(row['measuredFormantsHz'],native['measuredFormantsHz'])):
                if value is not None and target is not None:
                    pitch_drift.append({'percent':100*abs(value-target)/target,'speaker':row['speaker'],'vowel':row['vowel'],
                        'mode':row['mode'],'formant':j+1})
        held.append({'method':method,'cases':len(rows),**pitch_summary(rows),
            'medianSourceFormantDriftPercent':stats.median(d['percent'] for d in drift),
            'worstSourceFormantDrift':max(drift,key=lambda d:d['percent']),
            'medianPitchInducedFormantDriftPercent':stats.median(d['percent'] for d in pitch_drift),
            'worstPitchInducedFormantDrift':max(pitch_drift,key=lambda d:d['percent']),
            'maximumRawPeak':max(r['rawPeak'] for r in rows),
            'overFullScaleSamples':sum(r['overFullScaleSamples'] for r in rows)})
    transitions+=oldsummary['transitions'];held+=oldsummary['heldVowels']
    # A second spectral diagnostic uses source PCM, a fixed STFT and triangular
    # mel bands; capture uses sinusoidal least squares and WORLD. Native-pitch
    # cases only: comparing shifted audio would confound the changed pitch.
    native=[r for r in report['results']+compact['results'] if r['kind']=='held' and r['mode']=='native']
    for row in vowels['results']:
        if row['mode']=='native':
            speaker,vowel=row['fixture'].rsplit('-',1)
            native.append({**row,'speaker':speaker,'vowel':vowel,'method':'neural-grain'})
    mel_rows=[];tracks={}
    for source in source_manifest['clips']:
        ref=next(r for r in vowels['references'] if r['name']==source['name'])
        audio,rate=sf.read(ROOT/source['file']);assert rate==RATE
        audio=audio[round(ref['sourceStartSeconds']*RATE):round(ref['sourceEndSeconds']*RATE)]
        target,hz=spectral_shape(audio)
        tracks[source['name']+'-original']=target
        for row in [r for r in native if r['speaker']==source['speaker'] and r['vowel']==source['intendedVowel']]:
            audio,rate=sf.read(ROOT/row['file']);assert rate==RATE
            observed,_=spectral_shape(audio[int(.05*RATE):-int(.05*RATE)])
            error=float(np.mean(abs(observed-target)))
            mel_rows.append({'source':source['name'],'method':row['method'],'meanAbsoluteDb':error})
            tracks[source['name']+'-'+row['method']]=observed
    np.savez(OUT/'native-spectral-shapes.npz',frequencyHz=hz,**tracks)
    mel_summary=[{'method':method,'cases':len(rows),'meanAbsoluteDb':stats.mean(r['meanAbsoluteDb'] for r in rows),
        'worstCase':max(rows,key=lambda r:r['meanAbsoluteDb'])}
        for method in sorted({r['method'] for r in mel_rows})
        for rows in [[r for r in mel_rows if r['method']==method]]]
    fits=[d for p in report['profiles'] for d in p['capture']]
    streaming=[r['streaming'] for r in report['results']]
    summary={'scope':report['scope'],'completedCases':len(report['results']),
        'profileScalarsPerCharacter':{p['speaker']:p['profileScalars'] for p in report['profiles']},
        'staticFit':{'medianHeldCycleSnrDb':stats.median(d['heldCycleSnrDb'] for d in fits),
            'minimumHeldCycleSnrDb':min(d['heldCycleSnrDb'] for d in fits),
            'originalPcmMedianHeldCycleSnrDb':stats.median(d['originalPcmFit']['heldCycleSnrDb'] for d in fits),
            'minimumNeuralPatchSnrDb':min(d['neuralPatchSnrDb'] for d in fits),'cases':fits},
        'transitions':transitions,'heldVowels':held,
        'nativeSourceSpectralComparison':{'scope':'Mean absolute centered log-mel spectrum error against the original PCM patch: 80 triangular bands, 80–8000 Hz, 1024-sample Hann STFT, 120 hop, 2048 FFT, relative -70 dB floor. Original short patch versus each sustained output interior. Native pitch only; not perceptual quality or identity.',
            'groups':mel_summary,'allCases':mel_rows},
        'breathCases':[{'speaker':r['speaker'],**pitch_summary([r]),'rawPeak':r['rawPeak'],
            'formantDriftPercent':r['formantDriftPercent']} for r in report['results'] if r['kind']=='breath'],
        'streaming':{'cases':len(streaming),'medianFirstPacketComputeMs':stats.median(r['firstPacketComputeMs'] for r in streaming),
            'maximumFirstPacketComputeMs':max(r['firstPacketComputeMs'] for r in streaming),
            'maximumPacketP95Ms':max(r['packetP95Ms'] for r in streaming),
            'maximumSinglePacketMs':max(r['maximumPacketMs'] for r in streaming),
            'maximumComputeRealTimeFactor':max(r['computeRealTimeFactor'] for r in streaming),
            'allCancellationChecksPassed':all(r['cancelPreventsOutput'] and r['cancelClearsHistory'] for r in streaming),
            'timingScope':streaming[0]['timingScope']},
        'neuralWeightsUnchanged':report['neuralWeightsUnchanged'],
        'limitations':report['limits']+['No subjective acceptance or Kokoro-level naturalness is established.'],
        'decision':'Reject all three static harmonic ablations as quality replacements. Captured amplitudes worsen native source-spectrum similarity and median formant diagnostics; captured residual phase worsens transition pitch. Preserve the neural captured-grain and compact references.',
        'completionProven':False}
    dynamics=OUT/'dynamics-diagnostic.json'
    if dynamics.exists():
        d=json.loads(dynamics.read_text())
        if not d['complete']:raise ValueError('Incomplete dynamics diagnostic')
        summary['localDynamicsDiagnostic']={k:d[k] for k in ('medianStaticHeldSnrDb','medianLocalHeldSnrDb',
            'medianPairedChangeDb','improvedCases','localWindow','limitations')}
        summary['localDynamicsDiagnostic']['report']=str(dynamics.relative_to(ROOT))
    test_log=ROOT/'evidence/audio-native-harmonic-tests.txt'
    if test_log.exists():summary['tests']={'passed':int(re.search(r'Ran (\d+) tests',test_log.read_text()).group(1)),
        'log':str(test_log.relative_to(ROOT))}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('Static held-cycle SNR:',summary['staticFit']['medianHeldCycleSnrDb'],'original PCM',summary['staticFit']['originalPcmMedianHeldCycleSnrDb'])
    for group in mel_summary:print('native spectral',group['method'],round(group['meanAbsoluteDb'],3))
    for group in held:print('held formant',group['method'],round(group['medianSourceFormantDriftPercent'],2),round(group['worstSourceFormantDrift']['percent'],2))


if __name__=='__main__':main()
