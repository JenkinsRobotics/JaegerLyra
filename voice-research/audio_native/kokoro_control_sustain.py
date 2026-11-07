"""Continuous control edits for an explicitly pretrained Kokoro reference.

No model code or weights are implemented here. The caller supplies the actual
512-channel decoder features, F0, companion N control and style from normal
Kokoro inference. The returned complete controls require offline rendering.
"""
import numpy as np


def validate(controls,durations,nucleus):
    arrays={key:np.asarray(controls[key]) for key in ('asr','f0','noise','style')}
    d=np.asarray(durations)
    if d.ndim!=1 or not len(d) or not np.issubdtype(d.dtype,np.integer) or np.any(d<1):
        raise ValueError('Expected positive integer durations')
    frames=int(d.sum())
    if (arrays['asr'].shape!=(1,512,frames) or arrays['f0'].shape!=(1,2*frames)
            or arrays['noise'].shape!=(1,2*frames) or arrays['style'].shape!=(1,128)
            or not all(np.isfinite(a).all() for a in arrays.values())
            or not isinstance(nucleus,(int,np.integer)) or not 0<nucleus<len(d)-1):
        raise ValueError('Invalid captured Kokoro controls')
    return arrays,d.astype(int)


def interior_state(controls,durations,nucleus):
    arrays,d=validate(controls,durations,nucleus)
    edges=np.r_[0,np.cumsum(d)];a=int(edges[nucleus]*2);b=int(edges[nucleus+1]*2)
    f0=arrays['f0'][0];n=arrays['noise'][0]
    voiced=np.arange(a,b)[f0[a:b]>55]
    if len(voiced)<5:raise ValueError('Insufficient voiced vowel controls')
    high=float(np.max(n[voiced]));candidates=[]
    for frame in range(int(edges[nucleus])+1,int(edges[nucleus+1])-1):
        center=frame*2;window=np.arange(center-2,center+3)
        if (window[0]<a or window[-1]>=b or np.any(f0[window]<=55)
                or np.any(n[window]<high-2.)):continue
        score=float(np.std(n[window])+np.std(np.log(f0[window])))
        candidates.append((score,frame))
    if not candidates:raise ValueError('No stable voiced interior control state')
    score,frame=min(candidates)
    return frame,{'sourceFrame40Hz':frame,'sourceTimeSeconds':frame/40,
        'sourceF0':float(f0[frame*2]),'sourceN':float(n[frame*2]),'score':score,
        'candidateCount':len(candidates),
        'selection':'Five 80 Hz frames inside intended vowel; F0 >55 and N within 2 model units of voiced-vowel maximum; minimize std(N)+std(log F0)',
        'recognizedVowel':False}


def stretch_controls(controls,durations,nucleus,seconds,method,target_hz=None):
    arrays,d=validate(controls,durations,nucleus)
    seconds=float(seconds)
    if not np.isfinite(seconds):raise ValueError('Expected finite duration')
    frames=int(d.sum());target_frames=round(seconds*40)
    if abs(target_frames/40-seconds)>1e-8 or not frames<=target_frames<=1200:
        raise ValueError('Use a finite 40 Hz-aligned duration at least as long as the source and at most 30 seconds')
    if method not in ('whole-vowel','interior-hold'):raise ValueError('Unknown control sustain method')
    if target_hz is not None:
        target_hz=np.asarray(target_hz,dtype=float)
        if target_hz.shape!=(target_frames*2,) or not np.isfinite(target_hz).all() or np.any((target_hz<60)|(target_hz>800)):
            raise ValueError('Expected 60–800 Hz authored F0 at 80 Hz')
    extra=target_frames-frames;state,selection=interior_state(arrays,d,nucleus)
    if method=='interior-hold':
        map40=np.r_[np.arange(state),np.full(extra,state),np.arange(state,frames)]
        map80=np.r_[np.arange(state*2),np.full(extra*2,state*2),np.arange(state*2,frames*2)]
    else:
        target_d=d.copy();target_d[nucleus]+=extra;edges=np.r_[0,np.cumsum(d)]
        map40=np.concatenate([np.linspace(edges[i],edges[i+1]-1,int(count)) for i,count in enumerate(target_d)])
        map80=np.concatenate([np.linspace(edges[i]*2,edges[i+1]*2-1,int(count)*2) for i,count in enumerate(target_d)])
    asr=np.array([np.interp(map40,np.arange(frames),row) for row in arrays['asr'][0]])[None]
    f0=np.interp(map80,np.arange(frames*2),arrays['f0'][0])
    n=np.interp(map80,np.arange(frames*2),arrays['noise'][0])
    original_voicing=f0>55
    if target_hz is not None:f0=np.where(original_voicing,target_hz,f0)
    edited={'asr':asr,'f0':f0[None],'noise':n[None],'style':arrays['style'].copy()}
    meta={'method':method,'sourceFrames40Hz':frames,'outputFrames40Hz':target_frames,
        'addedFrames40Hz':extra,'interiorSelection':selection,'holdStartSeconds':state/40,
        'holdEndSeconds':(state+extra)/40,'authoredPitchApplied':target_hz is not None,
        'wholeControlVoicedFraction':float(np.mean(original_voicing)),
        'boundaryScope':'Source control timing outside the changed vowel or inserted state is retained; the noncausal decoder and authored voiced F0 can still change audio at boundaries.'}
    return edited,meta,{'sourceFrameMap40Hz':map40,'sourceFrameMap80Hz':map80}
