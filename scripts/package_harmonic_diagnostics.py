"""Fixed-case auditions and a scientific diagnostic figure; no autoplay."""
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
OUT=ROOT/'evidence/captured-harmonic/v1'


def main():
    new=json.loads((OUT/'report.json').read_text())
    old=json.loads((ROOT/'evidence/compact-vocal/v1/report.json').read_text())
    transitions=json.loads((ROOT/'evidence/vocal-shapes/v1/report.json').read_text())
    vowels=json.loads((ROOT/'evidence/neural-vowels/v1/report.json').read_text())
    if not new['complete']:raise ValueError('Incomplete study')
    destination=OUT/'audition';destination.mkdir(exist_ok=True)
    groups=[]
    for kind,mode in (('held','native'),('transition','melody')):
        rows=[r for r in new['results']+old['results'] if r['speaker']=='af_heart'
            and r['kind']==kind and r['mode']==mode and (kind=='transition' or r['vowel']=='ah')
            and r['method']!='world-shapes']
        if kind=='held':
            grain=next(r for r in vowels['results'] if r['fixture']=='af_heart-ah' and r['mode']=='native')
        else:grain=next(r for r in transitions['results'] if r['speaker']=='af_heart' and r['mode']=='melody' and r['method']=='grain-crossfade')
        rows.append({**grain,'method':'neural-grain-reference'})
        groups.append((kind,rows))
    groups.append(('michael-breath',[r for r in new['results']+old['results'] if r['speaker']=='am_michael' and r['kind']=='breath']))
    manifest={'selection':'First predefined voice (Heart), held-ah/native and melody, all three new variants plus compact and neural-grain references. Michael added-breath failure retained as a separate pair.',
        'processing':'One constant gain per whole clip, matched -20 LUFS within groups or lower to hold sample peak <=0.9. No limiting, denoising or pitch edits.',
        'truePeakMeasured':False,'accepted':False,'groups':[]}
    for name,rows in groups:
        loaded=[]
        for row in rows:
            audio,rate=sf.read(ROOT/row['file']);lufs=pyln.Meter(rate).integrated_loudness(audio)
            loaded.append((row,audio,rate,lufs))
        target=min(-20.,*(lufs+20*np.log10(.9/max(abs(audio))) for _,audio,_,lufs in loaded))
        entries=[]
        for row,audio,rate,lufs in loaded:
            gain=10**((target-lufs)/20);file=destination/(name+'-'+row['method']+'.wav')
            sf.write(file,audio*gain,rate,subtype='PCM_24')
            saved,_=sf.read(file);actual=pyln.Meter(rate).integrated_loudness(saved)
            if abs(actual-target)>.01 or max(abs(saved))>.901:raise ValueError('Audition level error')
            entries.append({'method':row['method'],'source':row['file'],'file':str(file.relative_to(ROOT)),
                'sourceSha256':hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest(),'gain':gain,'lufs':actual})
        manifest['groups'].append({'name':name,'files':entries})
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    # Fixed first source, not a best-performing choice.
    data=np.load(OUT/'native-spectral-shapes.npz')
    dynamics=json.loads((OUT/'dynamics-diagnostic.json').read_text())
    fig,axes=plt.subplots(2,1,figsize=(10,8),layout='constrained')
    for key,label,color in [('original','Original Kokoro patch','#222222'),
            ('neural-grain','Neural captured-grain reference','#438373'),
            ('compact-harmonic-noise','Compact reference','#b17638'),
            ('captured-both','Captured amplitudes + phase','#775bad')]:
        axes[0].plot(data['frequencyHz'],data['af_heart-ah-'+key],label=label,color=color,linewidth=1.8)
    axes[0].set(xlabel='Frequency (Hz)',ylabel='Centered log-mel power (dB)',
        title='Heart / captured “ah”: native-pitch spectral shape',xlim=(80,8000))
    axes[0].legend(fontsize=9,ncol=2);axes[0].grid(alpha=.15)
    x=np.arange(9);width=.36
    axes[1].bar(x-width/2,[r['staticHeldSnrDb'] for r in dynamics['cases']],width,label='Static harmonic fit',color='#b17638')
    axes[1].bar(x+width/2,[r['localHeldSnrDb'] for r in dynamics['cases']],width,label='Locally varying fit',color='#438373')
    axes[1].set_xticks(x,[r['source'].replace('af_','').replace('am_','').replace('-','\n') for r in dynamics['cases']],fontsize=8)
    axes[1].set(ylabel='Unused-cycle waveform SNR (dB)',title='Odd source cycles held out; fits use only even neighbouring cycles')
    axes[1].legend(fontsize=9);axes[1].grid(axis='y',alpha=.15)
    fig.suptitle('Static capture loses voice detail; time variation needs a separate model',fontsize=14)
    fig.supxlabel('Diagnostics only: the local fit has extra source access and parameters; no perceptual-quality acceptance.',fontsize=9)
    fig.savefig(OUT/'texture-diagnostics.png',dpi=160);plt.close(fig)
    print('Saved 12 labelled auditions and the fixed-source diagnostic figure; no playback')


if __name__=='__main__':main()
