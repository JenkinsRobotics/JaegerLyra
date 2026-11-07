"""Diagnostic cancellation of a constant input's periodic upsampler response.

This offline ablation operates on observed intermediate features, not output
audio or neural weights. It is not a general streaming anti-aliasing filter.
"""
import numpy as np


def correct_periodic_hold(values, period, sample_rate, start_seconds, end_seconds, strength=1.):
    values=np.asarray(values)
    if (values.ndim!=2 or not len(values) or not np.isfinite(values).all()
            or not isinstance(period,(int,np.integer)) or period<2
            or not np.isfinite([sample_rate,start_seconds,end_seconds,strength]).all()
            or sample_rate<=0 or not 0<=strength<=1 or start_seconds<0
            or end_seconds>len(values)/sample_rate or end_seconds-start_seconds<=.25):
        raise ValueError('Invalid held-feature correction')
    # Fit only the well-interior constant-input response, not output quality.
    first=int(np.ceil((start_seconds+.1)*sample_rate));last=int(np.floor((end_seconds-.1)*sample_rate))
    indices=np.arange(first,last);phase=indices%period
    if any(np.sum(phase==p)<2 for p in range(period)):raise ValueError('Insufficient interior phases')
    means=np.stack([values[indices[phase==p]].mean(axis=0) for p in range(period)])
    template=means-means.mean(axis=0,keepdims=True)
    times=np.arange(len(values))/sample_rate
    attack=np.clip((times-start_seconds-.025)/.025,0.,1.)
    release=np.clip((end_seconds-.025-times)/.025,0.,1.)
    gate=(.5-.5*np.cos(np.pi*attack))*(.5-.5*np.cos(np.pi*release))
    correction=template[np.arange(len(values))%period]*gate[:,None]*strength
    output=values-correction
    centered=values[indices]-values[indices].mean(axis=0,keepdims=True)
    residual=output[indices]-output[indices].mean(axis=0,keepdims=True)
    info={'period':int(period),'featureRateHz':float(sample_rate),'strength':float(strength),
        'fitStartIndex':first,'fitEndIndex':last,'templateRms':float(np.sqrt(np.mean(template**2))),
        'beforeTemporalRms':float(np.sqrt(np.mean(centered**2))),
        'afterTemporalRms':float(np.sqrt(np.mean(residual**2))),
        'templatePhaseMeanMaximum':float(np.max(abs(template.mean(axis=0)))),
        'scope':'Mean-zero phase template from the known constant-input interior; correction gated inside the hold only.'}
    return output,template,gate,info
