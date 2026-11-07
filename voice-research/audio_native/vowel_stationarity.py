"""Source spectral-stationarity selection; no target output or transcript scores.

Features use fixed absolute-frequency mel bands, remove per-frame level, and
compare shape across pulse-centred 42.7 ms analysis windows. Harmonic sampling
and source pitch still affect these features. Minimizing this diagnostic does
not prove that the selected span is the intended vowel or sounds natural.
"""
import numpy as np
from neural_articulation import ArticulationUnit,candidate_spans
from compact_vocal_instrument import RATE

WINDOW_SIZE=1024
FFT_SIZE=2048
BANDS=40
OFFSETS=np.arange(WINDOW_SIZE)-WINDOW_SIZE//2
WINDOW=np.hanning(WINDOW_SIZE)
HZ=np.fft.rfftfreq(FFT_SIZE,1/RATE)
mel=lambda x:2595*np.log10(1+x/700)
POINTS=700*(10**(np.linspace(mel(80),mel(8000),BANDS+2)/2595)-1)
FILTERS=np.maximum(0,np.minimum((HZ[None,:]-POINTS[:-2,None])/(POINTS[1:-1,None]-POINTS[:-2,None]),
    (POINTS[2:,None]-HZ[None,:])/(POINTS[2:,None]-POINTS[1:-1,None])))
FILTERS/=np.maximum(FILTERS.sum(axis=1,keepdims=True),1e-20)


def spectral_frames(audio,centers):
    audio=np.asarray(audio,dtype=float);centers=np.asarray(centers,dtype=float)
    if (audio.ndim!=1 or not len(audio) or centers.ndim!=1 or not len(centers) or not np.isfinite(audio).all()
            or not np.isfinite(centers).all() or np.any(centers+OFFSETS[0]<0)
            or np.any(centers+OFFSETS[-1]>len(audio)-1)):
        raise ValueError('Spectral analysis requires complete finite source windows')
    windows=np.array([np.interp(center+OFFSETS,np.arange(len(audio)),audio) for center in centers])*WINDOW
    power=abs(np.fft.rfft(windows,n=FFT_SIZE,axis=1))**2
    bands=power@FILTERS.T
    db=10*np.log10(np.maximum(bands,np.maximum(bands.max(axis=1,keepdims=True)*1e-5,1e-30)))
    return db-db.mean(axis=1,keepdims=True)


def shape_variation(features):
    features=np.asarray(features,dtype=float)
    if features.ndim!=2 or len(features)<2 or not np.isfinite(features).all():raise ValueError('Need finite spectral frames')
    return float(np.sqrt(np.mean((features-np.median(features,axis=0))**2)))


def select_stationary_unit(audio,unit,maximum_hold_start):
    unit.validate();p=unit.performance.pulses
    earliest,candidates=candidate_spans(audio,p)
    eligible=[];excluded=[]
    for original_score,first,last in candidates:
        reason=None
        if p[first]>maximum_hold_start:reason='Insufficient remaining short-condition sustain'
        elif p[first]+OFFSETS[0]<0 or p[last]+OFFSETS[-1]>len(audio)-1:reason='Incomplete spectral analysis support'
        if reason:
            excluded.append({'first':first,'last':last,'reason':reason});continue
        features=spectral_frames(audio,p[first:last+1]);variation=shape_variation(features)
        eligible.append({'first':first,'last':last,'shapeVariationDb':variation,'pitchEnergyScore':original_score})
    if not eligible:raise ValueError('No eligible source-stationary candidates')
    chosen=min(eligible,key=lambda r:(r['shapeVariationDb'],r['pitchEnergyScore'],r['first'],r['last']))
    selected=ArticulationUnit(unit.performance,chosen['first'],chosen['last']);selected.validate()
    original_features=spectral_frames(audio,p[unit.hold_first:unit.hold_last+1])
    return selected,{'selection':'Lexicographic minimum of source log-mel shape variation, then original pitch/energy score, then pulse indices',
        'originalFirst':unit.hold_first,'originalLast':unit.hold_last,
        'originalShapeVariationDb':shape_variation(original_features),'chosen':chosen,
        'lateRegionThresholdSample':float(earliest),'maximumHoldStartSample':float(maximum_hold_start),
        'eligibleCandidates':eligible,'excludedCandidates':excluded,
        'featureDefinition':{'windowSamples':WINDOW_SIZE,'fftSize':FFT_SIZE,'bands':BANDS,'frequencyRangeHz':[80,8000],
            'relativePowerFloorDb':-50,'frameLevelRemoved':True},
        'selectedPitchHz':float(RATE/np.median(np.diff(p[selected.hold_first:selected.hold_last+1]))),
        'requestedOutputPitchChanged':False,'phonemeRecognized':False}
