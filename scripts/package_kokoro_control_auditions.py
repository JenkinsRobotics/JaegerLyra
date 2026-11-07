"""All pretrained control-study comparisons, level matched without autoplay."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'public/research/control-sustain'


def main():
    report=json.loads((ROOT/'evidence/kokoro-control-voicebox/reference-norm-v2/evaluation.json').read_text())
    if not report['complete']:raise ValueError('Incomplete study')
    OUT.mkdir(parents=True,exist_ok=True);files=[];groups=[]
    for source in report['nativeSources']:
        rows=[r for r in report['results'] if r['speaker']==source['speaker'] and r['word']==source['word']]
        items=[{**source,'mode':'source','method':'source'}]+rows;loaded=[]
        for row in items:
            path=ROOT/row['file']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Changed audio')
            audio,rate=sf.read(path);lufs=pyln.Meter(rate,block_size=.2).integrated_loudness(audio)
            loaded.append((row,audio,rate,lufs))
        target=min(-20.,*(lufs+20*np.log10(.9/max(abs(audio))) for _,audio,_,lufs in loaded))-.1
        packed={};raw={}
        for row,audio,rate,lufs in loaded:
            gain=10**((target-lufs)/20);meter=pyln.Meter(rate,block_size=.2)
            # Absolute loudness gating can change membership after scaling.
            # Re-measure the constant gain rather than assuming scale invariance.
            for iteration in range(12):
                current=meter.integrated_loudness(audio*gain)
                if abs(current-target)<.002:break
                gain*=10**((target-current)/20)
                gain=min(gain,.9/max(abs(audio)))
            file=OUT/(row['name']+'.wav')
            sf.write(file,audio*gain,rate,subtype='PCM_24');saved,_=sf.read(file)
            measured=pyln.Meter(rate,block_size=.2).integrated_loudness(saved)
            if abs(measured-target)>.01 or max(abs(saved))>.901:
                raise ValueError(f'Invalid audition levels: {row["name"]}, measured {measured}, target {target}')
            record={'file':file.name,'source':row['file'],'sourceSha256':row['sha256'],'gain':gain,
                'lufs200MsGate':measured,'samplePeak':float(max(abs(saved))),'method':row['method'],'mode':row['mode'],
                'gainRefinementSteps':iteration,'targetLufs200MsGate':target}
            packed[(row['mode'],row['method'])]=record;raw[(row['mode'],row['method'])]=(row,audio);files.append(record)
        for mode in ('native','melody'):
            before,before_audio=raw[(mode,'interior-hold')]
            revised,revised_audio=raw[(mode,'interior-hold-reference-norm')]
            plot=OUT/(source['name']+'-'+mode+'-waveform.png')
            fig,axes=plt.subplots(3,1,figsize=(10,5.7),sharex=True,layout='constrained')
            fig.set_facecolor('#f7f3ed')
            peak=max(max(abs(before_audio)),max(abs(revised_audio)))*1.06
            for ax,audio,title,color in [(axes[0],before_audio,'Earlier hold · raw waveform','#897889'),
                                       (axes[1],revised_audio,'Revised hold · raw waveform','#427a72')]:
                ax.plot(np.arange(len(audio))[::12]/24000,audio[::12],color=color,lw=.55)
                ax.set_ylim(-peak,peak);ax.set_title(title,loc='left',fontsize=10);ax.set_ylabel('Amplitude')
            tracks=np.load(ROOT/revised['tracks'])
            axes[2].plot(tracks['requestedTimes'],tracks['requestedF0'],c='#29242f',ls='--',lw=1,label='Requested pitch')
            for estimator,color in [('praat','#8e619e'),('world','#427a72')]:
                times=tracks[estimator+'Times'];pitch=tracks[estimator+'Pitch'];valid=pitch>0
                axes[2].plot(times[valid],pitch[valid],'.',ms=2,color=color,label=estimator.upper())
            axes[2].set_ylabel('Pitch · Hz');axes[2].set_xlabel('Seconds');axes[2].legend(loc='upper right',fontsize=8,ncol=3)
            for ax in axes:
                ax.set_facecolor('#fffdf9');ax.set_xlim(0,2.5)
                ax.axvspan(revised['analysisStartSeconds'],revised['analysisEndSeconds'],color='#427a72',alpha=.08)
                ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False)
            voice={'af_heart':'Heart','am_michael':'Michael','af_bella':'Bella'}[source['speaker']]
            fig.suptitle(f'{voice} · {source["word"].capitalize()} · {"Held note" if mode=="native" else "Melody"}',fontsize=13)
            fig.savefig(plot,dpi=130);plt.close(fig)
            groups.append({'speaker':source['speaker'],'word':source['word'],'mode':mode,
                'title':'Held note' if mode=='native' else 'Melody','plot':plot.name,
                'source':packed[('source','source')],'earlier':packed[(mode,'interior-hold')],
                'revised':packed[(mode,'interior-hold-reference-norm')],
                'whole':packed[(mode,'whole-vowel')],'wholeReference':packed[(mode,'whole-vowel-reference-norm')],
                'legacy':packed[(mode,'legacy-kokoro')],'captured':packed[(mode,'captured-cycle')],
                'praatP95Cents':revised['praat']['p95ErrorCents'],'worldP95Cents':revised['world']['p95ErrorCents'],
                'octaveDisagreement':abs(revised['praat']['p95ErrorCents']-revised['world']['p95ErrorCents'])>600})
    manifest={'scope':'All three voices and development words, both performance curves, and every earlier comparison method',
        'processing':'Whole clips, constant gain, common level per source using a disclosed 200 ms loudness gate. No limiter, denoising, pitch correction or autoplay.',
        'graphScope':'Raw unlevelled waveforms and both pitch estimators; shaded region is the common evaluation interval.',
        'pretrainedModel':True,'independentTraining':False,'truePeakMeasured':False,
        'listeningAccepted':False,'files':files,'groups':groups}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('Saved',len(files),'matched clips and',len(groups),'comparisons with waveform plots; no playback.')


if __name__=='__main__':main()
