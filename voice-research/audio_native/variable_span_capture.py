"""Post-validation repair: search eligible span endpoints as well as starts.

The earlier selector tests only the longest possible span at each start. A
falling pitch late in that span can hide a valid shorter stable region. This
opt-in capture path retains the same 60–150 ms, energy and period-ratio bounds.
It is a source heuristic, not a phoneme detector or a new neural model.
"""
import numpy as np
from neural_waveform import RATE
from neural_performance import compile_performance
from neural_articulation import ArticulationUnit
from live_neural_grains import DELAY
from vowel_stationarity import spectral_frames, shape_variation, OFFSETS


def candidate_spans_with_variable_ends(audio, pulses):
    audio = np.asarray(audio, dtype=float); p = np.asarray(pulses, dtype=float)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Expected finite source waveform')
    if p.ndim != 1 or not np.isfinite(p).all() or np.any(np.diff(p) <= 0):
        raise ValueError('Expected ordered finite source pulses')
    active = np.flatnonzero(abs(audio) > .03*max(abs(audio)))
    if not len(active): raise ValueError('Silent unit')
    earliest = active[0] + .35*(active[-1]-active[0]); candidates = []
    for group in np.split(np.arange(len(p)), np.flatnonzero(np.diff(p) > DELAY)+1):
        if len(group) < 8: continue
        for first in group[1:-6]:
            if p[first] < earliest: continue
            maximum = min(int(np.searchsorted(p, p[first]+RATE*.150, side='right')-1), int(group[-2]))
            for last in range(first+6, maximum+1):
                if p[last]-p[first] < RATE*.060: continue
                periods = np.diff(p[first:last+1])
                if min(periods) < RATE/1000 or max(periods) > 1.25*min(periods): continue
                energies = np.array([np.sqrt(np.mean(audio[int(a):int(b)]**2))
                                     for a,b in zip(p[first:last],p[first+1:last+1])])
                if not np.isfinite(energies).all() or min(energies) < .003: continue
                score = float(np.std(np.log(periods)) + .15*np.std(np.log(energies)))
                candidates.append((score,int(first),int(last)))
    if not candidates: raise ValueError('No stable late vowel region with variable endpoints')
    return float(earliest), candidates


def compile_variable_unit(audio, encoder):
    audio = np.asarray(audio, dtype=float)
    performance = compile_performance(audio, encoder); p = performance.pulses
    earliest, candidates = candidate_spans_with_variable_ends(audio, p)
    # The short diagnostic retains at least 150 ms after the selected start
    # within the original source duration; long-note controls are independent.
    maximum_start = len(audio)-.150*RATE
    eligible = [(score,first,last) for score,first,last in candidates
                if p[first] <= maximum_start and p[first]+OFFSETS[0] >= 0
                and p[last]+OFFSETS[-1] <= len(audio)-1]
    if not eligible: raise ValueError('No eligible variable span with complete support and short sustain')
    first_index = min(r[1] for r in eligible); last_index = max(r[2] for r in eligible)
    features = spectral_frames(audio, p[first_index:last_index+1])
    scored = [(shape_variation(features[first-first_index:last-first_index+1]), score, first, last)
              for score,first,last in eligible]
    variation, score, first, last = min(scored)
    unit = ArticulationUnit(performance,first,last); unit.validate()
    return unit, {'holdFirstPulse':first, 'holdLastPulse':last, 'holdStartSample':float(p[first]),
        'holdEndSample':float(p[last]), 'sourceMedianPitchHz':float(RATE/np.median(np.diff(p[first:last+1]))),
        'sourceSeconds':len(audio)/RATE, 'sourceShapeVariationDb':variation, 'sourceSelectionScore':score,
        'sourceSelectionCandidates':len(candidates), 'eligibleCandidates':len(eligible),
        'lateRegionThresholdSample':earliest, 'maximumHoldStartSample':maximum_start,
        'selection':'Source-only spectral variation, then pitch/energy score and pulse indices; variable endpoint search',
        'sourceSpanSeconds':float((p[last]-p[first])/RATE), 'recognizedPhones':False,
        'capturedSourceUnit':True, 'postValidationDevelopment':True}
