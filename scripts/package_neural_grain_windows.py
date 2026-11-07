"""All held-grain-width comparisons, without autoplay or post pitch fixes."""
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
STUDY=ROOT/'evidence/neural-syllables/grain-windows-v1'
OUT=STUDY/'audition'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    capture=json.loads((STUDY/'capture.json').read_text());render=json.loads((STUDY/'report.json').read_text())
    evaluation=json.loads((STUDY/'evaluation.json').read_text());spectra=json.loads((STUDY/'spectral-evaluation.json').read_text())
    if not all(r['complete'] for r in (render,evaluation,spectra)):raise ValueError('Incomplete evaluation')
    cases=capture['references']+capture['reconstructions']+render['results']
    metrics={r['name']:r for r in evaluation['results']};shapes={r['name']:r for r in spectra['results']}
    OUT.mkdir(exist_ok=True);files=[];reels=[];plots=[];meter=pyln.Meter(24000,block_size=.2)
    methods=[('live-neural','Minimum source/target period','#91648f'),('source-neural','Source period','#267b71'),('wide-neural','1.5 × source period','#c38336')]
    for source in capture['sources']:
        rows=[{**source,'variant':'original','method':'ordinary-speech','mode':'source'}]
        rows += [r for r in cases if (r['speaker'],r['word'])==(source['speaker'],source['word'])]
        loaded=[]
        for row in rows:
            file=ROOT/row['file']
            if sha(file)!=row['sha256']:raise ValueError('Changed source')
            audio,rate=sf.read(file);assert rate==24000
            lufs=meter.integrated_loudness(audio)
            if not np.isfinite(lufs):raise ValueError('Invalid loudness')
            loaded.append((row,audio,lufs))
        target=min(-20.,*(lufs+20*np.log10(.85/max(abs(a))) for _,a,lufs in loaded))-.2
        packed={};levelled={};metadata={}
        for row,audio,lufs in loaded:
            gain=10**((target-lufs)/20)
            for _ in range(20):
                measured=meter.integrated_loudness(audio*gain)
                if abs(measured-target)<.002:break
                gain=min(gain*10**((target-measured)/20),.85/max(abs(audio)))
            file=OUT/(row['name']+'.wav');sf.write(file,audio*gain,24000,subtype='PCM_24')
            saved,_=sf.read(file);measured=meter.integrated_loudness(saved)
            if abs(measured-target)>.01 or max(abs(saved))>.851:raise ValueError('Invalid matched level')
            record={k:row[k] for k in ('name','speaker','word','variant','method','mode')}
            record.update(file=str(file.relative_to(ROOT)),sha256=sha(file),source=row['file'],sourceSha256=row['sha256'],
                auditionGain=gain,targetLufs200MsGate=target,measuredLufs200MsGate=measured,samplePeak=float(max(abs(saved))))
            key=(row['method'],row['mode']);packed[key]=record;levelled[key]=saved;metadata[key]=row;files.append(record)
        fig,ax=plt.subplots(figsize=(9,3.8),layout='constrained');fig.set_facecolor('#f7f3ed')
        for i,(method,label,color) in enumerate(methods):
            row=metadata[(method,'native')]
            with np.load(ROOT/shapes[row['name']]['spectrum'],allow_pickle=False) as spectrum:
                if i==0:ax.semilogx(spectrum['hz'],spectrum['referenceDb'],'--',color='#34313a',lw=1.2,label='Corrected pretrained reference')
                ax.semilogx(spectrum['hz'],spectrum['outputDb'],color=color,lw=1.4,label=label)
        ax.set(xlabel='Frequency · Hz',ylabel='Centered log-mel power · dB',xlim=(80,8000))
        ax.set_title(source['speaker']+' · '+source['word']+' | steady vowel, gain-normalized spectrum',loc='left',fontsize=11)
        ax.set_facecolor('#fffdf9');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.legend(fontsize=8)
        file=OUT/(source['name']+'-native-spectrum.png');fig.savefig(file,dpi=140);plt.close(fig)
        plots.append({'speaker':source['speaker'],'word':source['word'],'kind':'native-spectrum','file':str(file.relative_to(ROOT))})
        fig,axes=plt.subplots(3,1,figsize=(9,7.1),sharex=True,layout='constrained');fig.set_facecolor('#f7f3ed')
        for ax,(method,label,color) in zip(axes,methods):
            row=metrics[metadata[(method,'melody')]['name']]
            with np.load(ROOT/row['tracks'],allow_pickle=False) as track:
                ax.plot(track['requestedTimes'],track['requestedF0'],'--',lw=1,color='#34313a',label='Requested')
                for tracker,ink in [('praat','#91648f'),('world','#267b71')]:
                    keep=track[tracker+'Pitch']>0
                    ax.plot(track[tracker+'Times'][keep],track[tracker+'Pitch'][keep],'.',ms=2,color=ink,label=tracker.upper())
                    missing=(~keep)&(track[tracker+'Times']>=row['analysisStartSeconds'])&(track[tracker+'Times']<=row['analysisEndSeconds'])
                    ax.plot(track[tracker+'Times'][missing],np.zeros(sum(missing)),'x',ms=3,color=ink)
            status='passes' if row['bothPitchChecksPassed'] else 'fails'
            ax.set_title(label+' · '+status+' | voiced: P '+format(row['praat']['voicedFraction'],'.1%')+' / W '+format(row['world']['voicedFraction'],'.1%'),loc='left',fontsize=10)
            ax.axvspan(row['analysisStartSeconds'],row['analysisEndSeconds'],color='#267b71',alpha=.06)
            ax.set(ylabel='Pitch · Hz',xlim=(0,3));ax.set_ylim(bottom=-8);ax.set_facecolor('#fffdf9');ax.spines[['top','right']].set_visible(False)
            ax.grid(alpha=.15);ax.legend(fontsize=7,ncol=3,loc='upper left')
        axes[-1].set_xlabel('Seconds · × at zero marks unvoiced frames inside the shaded test interval')
        fig.suptitle(source['speaker']+' · '+source['word']+' | melody and voicing coverage',fontsize=11)
        file=OUT/(source['name']+'-melody-pitch.png');fig.savefig(file,dpi=140);plt.close(fig)
        plots.append({'speaker':source['speaker'],'word':source['word'],'kind':'melody-pitch','file':str(file.relative_to(ROOT))})
        sequence=[('ordinary-speech','source')]+[(m,mode) for mode in ('native','melody') for m,_,_ in methods]
        parts=[];segments=[];offset=0
        for key in sequence:
            audio=levelled[key];segments.append({'startSeconds':offset/24000,'endSeconds':(offset+len(audio))/24000,
                **{k:packed[key][k] for k in ('method','mode','file')}})
            parts.extend([audio,np.zeros(6000)]);offset+=len(audio)+6000
        file=OUT/(source['name']+'-comparison.wav');sf.write(file,np.concatenate(parts[:-1]),24000,subtype='PCM_24')
        reels.append({'speaker':source['speaker'],'word':source['word'],'file':str(file.relative_to(ROOT)),
            'sha256':sha(file),'segments':segments,'scope':'Offline labelled concatenation with 250 ms gaps, not connected speech or live latency.'})
    manifest={'scope':'Every teacher, reconstruction and minimum/source/wide neural/PCM condition',
        'processing':'Constant whole-clip audition gains, matched across all 17 versions per source using 200 ms loudness gating. No limiter, pitch correction or held-only gain compensation.',
        'files':files,'plots':plots,'reels':reels,'autoplay':False,'listeningAccepted':False,'truePeakMeasured':False,
        'maximumSamplePeak':max(r['samplePeak'] for r in files),
        'maximumLoudnessErrorLu':max(abs(r['measuredLufs200MsGate']-r['targetLufs200MsGate']) for r in files),'complete':True}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('Packaged',len(files),'clips,',len(plots),'plots and',len(reels),'reels; no playback.')


if __name__=='__main__':main()
