"""Package the full fresh-vowel failure panel and generator-stage follow-up.

Constant-gain level matching only. Failed raw outputs remain unchanged in
evidence, including samples beyond full scale. No playback or acceptance.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evidence/kokoro-control-voicebox/generator-hold-v1/audition'


def main():
    base = json.loads((ROOT/'evidence/kokoro-control-voicebox/fresh-vowels-v1/evaluation.json').read_text())
    follow = json.loads((ROOT/'evidence/kokoro-control-voicebox/generator-hold-v1/evaluation.json').read_text())
    if not base['complete'] or not follow['complete']: raise ValueError('Finish both evaluations')
    OUT.mkdir(parents=True,exist_ok=True); files = []; groups = []; reels = []
    for source in base['sources']:
        items = [{**source,'mode':'source','method':'source','stage':'source'}]
        for stage, report in [('upstream',base),('generator',follow)]:
            items += [{**r,'stage':stage} for r in report['results']
                if r['speaker']==source['speaker'] and r['word']==source['word']]
        loaded = []
        for row in items:
            file = ROOT/row['file']
            if hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']: raise ValueError('Changed audio')
            audio, rate = sf.read(file); assert rate == 24000
            lufs = pyln.Meter(rate,block_size=.2).integrated_loudness(audio)
            if not np.isfinite(lufs): raise ValueError('Source has no finite measured loudness')
            loaded.append((row,audio,lufs))
        target = min(-20.,*(lufs+20*np.log10(.85/max(abs(audio))) for _,audio,lufs in loaded))-.2
        packed = {}; levelled = {}; originals = {}
        for row,audio,lufs in loaded:
            gain = 10**((target-lufs)/20); meter = pyln.Meter(24000,block_size=.2)
            for attempt in range(20):
                measured = meter.integrated_loudness(audio*gain)
                if abs(measured-target)<.002: break
                gain = min(gain*10**((target-measured)/20),.85/max(abs(audio)))
            file = OUT/(row['name']+'.wav'); sf.write(file,audio*gain,24000,subtype='PCM_24')
            saved,_ = sf.read(file); measured = meter.integrated_loudness(saved)
            if abs(measured-target)>.01 or max(abs(saved))>.851: raise ValueError('Audition level check failed: '+row['name'])
            record = {'file':str(file.relative_to(ROOT)),'source':row['file'],'sourceSha256':row['sha256'],
                'stage':row['stage'],'method':row['method'],'mode':row['mode'], 'gain':gain,
                'targetLufs200MsGate':target,'measuredLufs200MsGate':measured,'samplePeak':float(max(abs(saved)))}
            key = (row['stage'],row['mode'],row['method']); packed[key] = record; levelled[key] = saved
            originals[key] = (row,audio); files.append(record)
        for mode in ('native','melody'):
            before,a = originals[('upstream',mode,'reference')]
            after,b = originals[('generator',mode,'reference')]
            figure = OUT/(source['name']+'-'+mode+'-comparison.png')
            fig,axes = plt.subplots(4,1,figsize=(10,7.4),sharex=True,layout='constrained')
            fig.set_facecolor('#f7f3ed'); peak = max(max(abs(a)),max(abs(b)))*1.05
            for ax,audio,label,color in [(axes[0],a,'Earlier input hold · raw waveform','#8a708e'),
                    (axes[1],b,'Generator input hold · raw waveform','#367b71')]:
                ax.plot(np.arange(len(audio))[::12]/24000,audio[::12],color=color,lw=.5)
                ax.set_ylim(-peak,peak); ax.set_title(label,loc='left',fontsize=10); ax.set_ylabel('Amplitude')
            for ax,row,title in [(axes[2],before,'Earlier input hold'),(axes[3],after,'Generator input hold')]:
                with np.load(ROOT/row['tracks']) as track:
                    ax.plot(track['requestedTimes'],track['requestedF0'],color='#302b35',lw=1,ls='--',label='Requested')
                    for name,color in [('praat','#9466a3'),('world','#367b71')]:
                        valid=track[name+'Pitch']>0
                        ax.plot(track[name+'Times'][valid],track[name+'Pitch'][valid],'.',ms=2,color=color,label=name.upper())
                ax.set_title(title+' · pitch',loc='left',fontsize=10); ax.set_ylabel('Hz')
                ax.legend(loc='upper right',ncol=3,fontsize=8)
            for ax in axes:
                ax.set_facecolor('#fffdf9'); ax.grid(alpha=.15); ax.set_xlim(0,3)
                ax.axvspan(after['analysisStartSeconds'],after['analysisEndSeconds'],color='#367b71',alpha=.08)
                ax.spines[['top','right']].set_visible(False)
            axes[-1].set_xlabel('Seconds')
            fig.suptitle(source['speaker']+' · '+source['word']+' · '+mode,fontsize=13)
            fig.savefig(figure,dpi=130); plt.close(fig)
            groups.append({'speaker':source['speaker'],'word':source['word'],'mode':mode,
                'source':packed[('source','source','source')],
                'upstreamNormal':packed[('upstream',mode,'normal')], 'upstreamReference':packed[('upstream',mode,'reference')],
                'generatorNormal':packed[('generator',mode,'normal')], 'generatorReference':packed[('generator',mode,'reference')],
                'plot':str(figure.relative_to(ROOT)), 'upstreamPitchChecksPassed':before['bothPitchChecksPassed'],
                'generatorPitchChecksPassed':after['bothPitchChecksPassed']})
        sequence = [('source','source','source'),('upstream','native','reference'),('generator','native','reference'),
                    ('upstream','melody','reference'),('generator','melody','reference')]
        parts = []; segments = []; offset = 0
        for key in sequence:
            audio = levelled[key]; segments.append({'startSeconds':offset/24000,'endSeconds':(offset+len(audio))/24000,
                'stage':key[0],'mode':key[1],'method':key[2],'file':packed[key]['file']})
            parts += [audio,np.zeros(6000)]; offset += len(audio)+6000
        reel = OUT/(source['name']+'-comparison.wav'); sf.write(reel,np.concatenate(parts[:-1]),24000,subtype='PCM_24')
        reels.append({'speaker':source['speaker'],'word':source['word'],'file':str(reel.relative_to(ROOT)),'segments':segments,
            'scope':'Labelled offline concatenation with 250 ms gaps; not connected speech or a latency demo.'})
    manifest = {'scope':'Complete fresh-vowel validation and post-validation generator-input ablation',
        'processing':'Whole-clip constant gains, common measured level per source across all nine versions, 200 ms loudness gate. No limiter, pitch correction or denoising.',
        'rawOverFullScalePreservedInEvidence':True,'truePeakMeasured':False,'autoplay':False,'listeningAccepted':False,
        'pretrainedKokoro':True,'newIndependentTraining':False,'files':files,'groups':groups,'reels':reels}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('Packaged',len(files),'clips,',len(groups),'plots,',len(reels),'labelled comparisons; no playback.')


if __name__ == '__main__':main()
