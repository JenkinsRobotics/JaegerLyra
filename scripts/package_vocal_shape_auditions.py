"""Matched first-Heart transition comparisons, labelled by actual renderer."""
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
OUT=ROOT/'evidence/vocal-shapes/v1'


def main():
    report=json.loads((OUT/'report.json').read_text())
    if not report['complete']:raise ValueError('Incomplete transition study')
    audition=OUT/'audition';audition.mkdir(exist_ok=True)
    selected=[r for r in report['results'] if r['speaker']=='af_heart' and r['mode'] in ('steady','melody')]
    manifest={'selection':'First predefined speaker; steady and melody conditions, both renderers',
        'processing':'One fixed gain per clip to -20 LUFS or lower for 0.9 peak; no limiting, denoising or pitch correction',
        'scope':'Neural captured-grain crossfade versus external WORLD spectral-shape reference; neither is a completed speech engine',
        'files':[]}
    for row in selected:
        source=ROOT/row['file'];audio,rate=sf.read(source)
        loudness=pyln.Meter(rate).integrated_loudness(audio)
        gain=min(10**((-20-loudness)/20),.9/max(abs(audio).max(),1e-12))
        path=audition/(row['mode']+'-'+row['method']+'.wav')
        sf.write(path,audio*gain,rate,subtype='PCM_24')
        manifest['files'].append({'mode':row['mode'],'renderer':row['method'],'source':row['file'],
            'sourceSha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'file':str(path.relative_to(ROOT)),'gain':float(gain),'sourceLufs':float(loudness)})
    (audition/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    # Fixed first-speaker melody, not a best-scoring case.
    rows=[r for r in selected if r['mode']=='melody']
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    reference=np.load(ROOT/rows[0]['tracks'])
    for i,(name,color) in enumerate(zip(report['shapeOrder'],('#a4513a','#326faf','#469c72'))):
        axes[0].plot(reference['times'],reference['weights'][:,i],label=name,color=color,linewidth=2)
    axes[0].set(ylabel='Authored vowel weight',ylim=(-.05,1.08));axes[0].legend(ncol=3,loc='upper right')
    axes[1].plot(reference['times'],reference['targetPitch'],color='#222222',linestyle='--',label='Command',linewidth=2)
    for row,color,label in zip(rows,('#ba6d3c','#347c9a'),('Neural grain crossfade','External WORLD shapes')):
        data=np.load(ROOT/row['tracks'])
        axes[1].plot(data['praatTimes'],data['praat'],color=color,label=label,alpha=.85)
        axes[2].plot(data['times'],data['shapeErrorDb'],color=color,label=label,alpha=.85)
    axes[1].set(ylabel='Praat pitch (Hz)');axes[1].legend(loc='lower left')
    axes[2].set(ylabel='Shape error (dB)',xlabel='Output time (seconds)',xlim=(0,4))
    for axis in axes:axis.grid(alpha=.18)
    fig.suptitle('Heart: authored vowel changes during a four-note melody',fontsize=14)
    axes[2].set_title('WORLD analysis self-consistency; this panel is not an independent sound-quality score',fontsize=9)
    fig.savefig(OUT/'first-heart-melody-controls.png',dpi=160)
    plt.close(fig)
    print('Saved four labelled auditions and first-case control plot; no playback')


if __name__=='__main__':main()
