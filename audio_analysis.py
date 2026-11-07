"""Audio-driven approximate mouth poses: RMS envelope + LPC formants, no text alignment."""
import numpy as np
VOWELS={'aa':(800,1200),'ee':(400,2200),'ih':(350,1750),'oh':(500,900),'ou':(300,700)}
def vowel_weights(f1,f2,openness):
    if openness<.03:return {v:0.0 for v in VOWELS}
    distances={v:((f1-a)/260)**2+((f2-b)/500)**2 for v,(a,b) in VOWELS.items()}
    best=min(distances,key=distances.get)
    return {v:round(openness if v==best else 0.0,3) for v in VOWELS}
def formants(frame,sr):
    # Downsample to ~11 kHz, pre-emphasize, then solve autocorrelation LPC.
    stride=max(1,round(sr/11025));x=np.asarray(frame[::stride],dtype=np.float64);sr=sr/stride
    if len(x)<40:return 500.,1500.
    x=x-np.mean(x);x=np.concatenate([x[:1],x[1:]-.97*x[:-1]])*np.hanning(len(x))
    order=14;r=np.correlate(x,x,'full')[len(x)-1:len(x)+order]
    if r[0]<1e-9:return 500.,1500.
    matrix=r[np.abs(np.arange(order)[:,None]-np.arange(order)[None,:])]
    try:
        a=np.linalg.solve(matrix+np.eye(order)*r[0]*1e-5,-r[1:order+1])
        roots=np.roots(np.r_[1,a]);roots=roots[np.imag(roots)>0]
        freq=np.angle(roots)*sr/(2*np.pi);bw=-np.log(np.maximum(np.abs(roots),1e-8))*sr/np.pi
        peaks=sorted(freq[(freq>180)&(freq<3300)&(bw>0)&(bw<600)])
        lows=[f for f in peaks if f<1100];f1=lows[0] if lows else 500.
        highs=[f for f in peaks if f>max(f1+200,650)];f2=highs[0] if highs else 1500.
        return float(f1),float(f2)
    except np.linalg.LinAlgError:return 500.,1500.
def analyze(samples,sr):
    x=np.asarray(samples,dtype=np.float32);hop=max(1,int(sr/40));window=int(sr*.035)
    levels=[float(np.sqrt(np.mean(x[i:i+window]**2)+1e-12)) for i in range(0,len(x),hop)]
    reference=max(.035,float(np.percentile(levels,90))) if levels else .035
    frames=[]
    for n,i in enumerate(range(0,len(x),hop)):
        rms=levels[n];opened=float(np.clip((rms-.008)/reference,0,1))
        f1,f2=formants(x[i:i+window],sr) if opened>.03 else (0.,0.)
        frames.append({'t':round(i/sr,4),'jaw':round(opened,3),**vowel_weights(f1,f2,opened),'rms':round(rms,4),'f1':round(f1),'f2':round(f2)})
    frames.append({'t':round(len(x)/sr,4),'jaw':0,**{v:0 for v in VOWELS},'rms':0,'f1':0,'f2':0})
    return frames
