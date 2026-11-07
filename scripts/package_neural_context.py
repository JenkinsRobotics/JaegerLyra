"""All context ablations, level-matched clips and raw join envelopes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'evidence/neural-syllables/context-v1'
OUT=STUDY/'audition'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    capture=json.loads((STUDY/'capture.json').read_text());render=json.loads((STUDY/'report.json').read_text())
    if not render['complete']:raise ValueError('Incomplete renders')
    cases=capture['references']+capture['reconstructions']+render['results']
    OUT.mkdir(exist_ok=True);files=[];reels=[];plots=[];meter=pyln.Meter(24000,block_size=.2)
    for source in capture['sources']:
        rows=[{**source,'variant':'original','method':'ordinary-speech','mode':'source'}]
        rows += [r for r in cases if (r['speaker'],r['word'])==(source['speaker'],source['word'])]
        loaded=[]
        for row in rows:
            file=ROOT/row['file']
            if sha(file)!=row['sha256']:raise ValueError('Changed source')
            audio,rate=sf.read(file);assert rate==24000
            loudness=meter.integrated_loudness(audio)
            if not np.isfinite(loudness):raise ValueError('Invalid loudness')
            loaded.append((row,audio,loudness))
        target=min(-20.,*(lufs+20*np.log10(.85/max(abs(a))) for _,a,lufs in loaded))-.2
        packed={};levelled={};raw={};metadata={}
        for row,audio,loudness in loaded:
            gain=10**((target-loudness)/20)
            for _ in range(20):
                measured=meter.integrated_loudness(audio*gain)
                if abs(measured-target)<.002:break
                gain=min(gain*10**((target-measured)/20),.85/max(abs(audio)))
            file=OUT/(row['name']+'.wav');sf.write(file,audio*gain,24000,subtype='PCM_24')
            saved,_=sf.read(file);measured=meter.integrated_loudness(saved)
            if abs(measured-target)>.01 or max(abs(saved))>.851:raise ValueError('Invalid matched clip')
            record={k:row[k] for k in ('name','speaker','word','variant','method','mode')}
            record.update(file=str(file.relative_to(ROOT)),sha256=sha(file),source=row['file'],sourceSha256=row['sha256'],
                auditionGain=gain,targetLufs200MsGate=target,measuredLufs200MsGate=measured,samplePeak=float(max(abs(saved))))
            key=(row['method'],row['mode']);packed[key]=record;levelled[key]=saved;raw[key]=audio;metadata[key]=row;files.append(record)
        for mode in ('native','melody'):
            baseline=metadata[('live-neural',mode)];modified=metadata[('both-neural',mode)]
            fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained');fig.set_facecolor('#f7f3ed')
            for ax,boundary,label,span in [(axes[0],baseline['independentHoldStartSeconds'],'Entering the held vowel',(-30,0)),
                    (axes[1],baseline['releaseCommandSeconds'],'Releasing into the word ending',(0,30))]:
                centers=np.arange(-.1,.10001,.005)+boundary;indices=np.round(centers*24000).astype(int)
                for method,title,color in [('live-neural','Retune all voiced context','#91648f'),('both-neural','Preserve opening + ending','#267b71')]:
                    audio=raw[(method,mode)];rms=np.array([np.sqrt(np.mean(audio[max(0,c-240):min(len(audio),c+240)]**2)) for c in indices])
                    ax.plot((centers-boundary)*1000,20*np.log10(np.maximum(rms,1e-8)),label=title,color=color,lw=1.6)
                ax.axvspan(*span,color='#267b71',alpha=.10);ax.axvline(0,color='#47434a',ls=':',lw=1)
                ax.set_title(label,loc='left',fontsize=10);ax.set_xlabel('Milliseconds from boundary');ax.set_ylabel('20 ms RMS · dBFS')
                ax.set_facecolor('#fffdf9');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.legend(fontsize=7,loc='best')
            fig.suptitle(source['speaker']+' · '+source['word']+' · '+mode+' | raw, before audition gain',fontsize=11)
            file=OUT/(source['name']+'-'+mode+'-joins.png');fig.savefig(file,dpi=140);plt.close(fig)
            plots.append({'speaker':source['speaker'],'word':source['word'],'mode':mode,'file':str(file.relative_to(ROOT)),
                'scope':'Raw waveform envelope, not an audible-join or naturalness acceptance.'})
        sequence=[('ordinary-speech','source'),('live-neural','native'),('both-neural','native'),('live-neural','melody'),('both-neural','melody')]
        parts=[];segments=[];offset=0
        for key in sequence:
            audio=levelled[key];segments.append({'startSeconds':offset/24000,'endSeconds':(offset+len(audio))/24000,
                **{k:packed[key][k] for k in ('method','mode','file')}})
            parts.extend([audio,np.zeros(6000)]);offset+=len(audio)+6000
        file=OUT/(source['name']+'-comparison.wav');sf.write(file,np.concatenate(parts[:-1]),24000,subtype='PCM_24')
        reels.append({'speaker':source['speaker'],'word':source['word'],'file':str(file.relative_to(ROOT)),
            'sha256':sha(file),'segments':segments,'scope':'Offline labelled concatenation, 250 ms gaps; not connected speech or streaming.'})
    manifest={'scope':'All teacher/reconstruction and context-control cases; ordinary source speech retained',
        'processing':'Constant whole-clip audition gains matched across all 21 versions per source, using 200 ms loudness gating. No limiter or pitch correction. Plots use raw unlevelled waveforms.',
        'files':files,'reels':reels,'plots':plots,'autoplay':False,'listeningAccepted':False,'truePeakMeasured':False,
        'maximumSamplePeak':max(r['samplePeak'] for r in files),
        'maximumLoudnessErrorLu':max(abs(r['measuredLufs200MsGate']-r['targetLufs200MsGate']) for r in files),'complete':True}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('Packaged',len(files),'clips,',len(plots),'plots,',len(reels),'reels; no playback.')


if __name__=='__main__':main()
