"""All-condition research auditions and scientific plots; no playback."""
import hashlib
import html
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/resonance-morph/v1'
PUBLIC=ROOT/'public/research/resonance-morph-v1'
METHODS=['teacher','reconstruction','grain-linear','grain-learned','filter-linear','filter-learned','oracle-envelope']
LABELS={'teacher':'Kokoro reference nucleus','reconstruction':'Independent neural reconstruction',
    'grain-linear':'Captured vowels · linear blend','grain-learned':'Captured vowels · fitted timing',
    'filter-linear':'Moving resonances · linear path','filter-learned':'Moving resonances · fitted timing',
    'oracle-envelope':'Diagnostic · full reference envelope'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    evaluation=json.loads((OUT/'evaluation.json').read_text());assert evaluation['complete']
    PUBLIC.mkdir(parents=True,exist_ok=True)
    report={'scope':'Short-clip equal-RMS comparisons; single whole-clip gain, no limiter or pitch correction',
        'autoplay':False,'listeningAccepted':False,'files':[],'plots':[],'cases':[],'complete':False}
    names=list(dict.fromkeys(r['name'] for r in evaluation['results']))
    for name in names:
        rows=[next(r for r in evaluation['results'] if r['name']==name and r['method']==m) for m in METHODS]
        waves=[]
        for row in rows:
            path=ROOT/row['file'];assert sha(path)==row['sha256'];audio,rate=sf.read(path);assert rate==24000
            waves.append(audio)
        rms=[np.sqrt(np.mean(a*a)) for a in waves];peak=[max(abs(a)) for a in waves]
        target=min([.08]+[.8*r/p for r,p in zip(rms,peak)])
        case={'name':name,'speaker':rows[0]['speaker'],'group':rows[0]['group'],'choices':[]}
        for row,a,r in zip(rows,waves,rms):
            gain=target/r;path=PUBLIC/(name+'-'+row['method']+'.wav');sf.write(path,a*gain,24000,subtype='PCM_24')
            saved,_=sf.read(path)
            entry={'name':name,'method':row['method'],'label':LABELS[row['method']],
                'sourceFile':row['file'],'sourceSha256':row['sha256'],'file':str(path.relative_to(ROOT)),
                'sha256':sha(path),'gain':float(gain),'targetRms':float(target),'peak':float(max(abs(saved))),
                'rmsError':float(abs(np.sqrt(np.mean(saved**2))-target))}
            report['files'].append(entry);case['choices'].append({'label':entry['label'],'url':'/research/resonance-morph-v1/'+path.name})
        fig,axes=plt.subplots(5,1,figsize=(9,10),constrained_layout=True)
        selected=['teacher','grain-linear','filter-linear','oracle-envelope']
        for ax,method in zip(axes,selected):
            row=next(r for r in rows if r['method']==method)
            with np.load(ROOT/row['tracks']) as z:
                t=z['times'];features=z['observed']
            im=ax.imshow(features.T,origin='lower',aspect='auto',extent=[t[0],t[-1],0,80],cmap='magma',vmin=-25,vmax=25)
            ax.set_title(LABELS[method],loc='left',fontsize=10);ax.set_ylabel('Spectral band')
        for method in METHODS[2:]:
            row=next(r for r in rows if r['method']==method)
            with np.load(ROOT/row['tracks']) as z:axes[-1].plot(z['times'],z['spectrumError'],label=method,linewidth=1.2)
        axes[-1].set_ylabel('Shape error (dB)');axes[-1].set_xlabel('Time in nucleus (seconds)')
        axes[-1].legend(fontsize=7,ncol=2);axes[-1].grid(alpha=.2)
        fig.suptitle(name+' · fresh word / supplied reference prosody',fontsize=13)
        fig.colorbar(im,ax=list(axes[:4]),label='Centered band energy (dB)',shrink=.7)
        plot=PUBLIC/(name+'.png');fig.savefig(plot,dpi=135);plt.close(fig)
        report['plots'].append({'name':name,'file':str(plot.relative_to(ROOT)),'sha256':sha(plot)})
        case['plot']='/research/resonance-morph-v1/'+plot.name;report['cases'].append(case)
    capture=json.loads((OUT/'capture.json').read_text());fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for group,values in capture['trajectories'].items():ax.plot(np.linspace(0,1,len(values)),values,'o-',label=group)
    ax.plot([0,1],[0,1],'--',color='gray',label='Linear')
    ax.set(xlabel='Normalized transition time',ylabel='Articulation weight',title='Fitted on development words only');ax.legend();ax.grid(alpha=.2)
    plot=PUBLIC/'fitted-paths.png';fig.savefig(plot,dpi=150);plt.close(fig)
    report['plots'].append({'name':'fitted-paths','file':str(plot.relative_to(ROOT)),'sha256':sha(plot)})
    report['complete']=True;(OUT/'audition.json').write_text(json.dumps(report,indent=2)+'\n')
    (PUBLIC/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aether · Articulation study</title><style>body{max-width:980px;margin:50px auto;padding:0 24px;background:#f3efe6;color:#342d3b;font:16px/1.6 system-ui}h1{font-size:38px;line-height:1.1}select,button{font:inherit;padding:10px;border:1px solid #cfc4d1;border-radius:8px;background:#fffaf5}#players{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:24px 0}article{padding:16px;background:#fffbf5;border:1px solid #d7ccd9;border-radius:12px}audio{width:100%}h3{font-size:15px}img{max-width:100%;border-radius:12px}.note{font-size:14px;color:#6c5f73}@media(max-width:650px){#players{grid-template-columns:1fr}}</style>
<a href="http://127.0.0.1:8794/">← Playable voicebox</a><p class="note">OUTPUT RESEARCH · EXPERIMENTAL COMPARISON</p>
<h1>How should a voice change shape?</h1><p>Compare natural Kokoro vowel transitions with reconstruction, blends and moving-resonance filters. These are cropped vowel nuclei from eighteen fresh words, not complete generated words.</p>
<p class="note">All candidates receive the reference pitch, duration and loudness. The final diagnostic also receives its complete spectral envelope. Those supplied controls do not demonstrate independent speech generation. No alternative has been accepted as a quality improvement.</p>
<label for="case">Voice and source word </label><select id="case"></select><div id="players"></div>
<p class="note">Equal RMS within each set · one constant gain · no limiter · no autoplay. Raw over-range samples remain recorded in the experiment; playback files have headroom. Only one player runs at a time.</p>
<details><summary>Show spectral comparison</summary><img id="plot" alt="Reference and candidate spectral comparisons"></details>
<script type="module">const m=await(await fetch('/research/resonance-morph-v1/manifest.json')).json(),s=document.getElementById('case'),p=document.getElementById('players');m.cases.forEach((c,i)=>s.add(new Option(c.name,i)));function show(){p.querySelectorAll('audio').forEach(a=>a.pause());p.replaceChildren();const c=m.cases[Number(s.value)];for(const choice of c.choices){const a=document.createElement('article'),h=document.createElement('h3'),player=document.createElement('audio');h.textContent=choice.label;player.controls=true;player.preload='none';player.src=choice.url;player.addEventListener('play',()=>p.querySelectorAll('audio').forEach(other=>{if(other!==player)other.pause()}));a.append(h,player);p.append(a)}document.getElementById('plot').src=c.plot}s.addEventListener('change',show);show();</script></html>'''
    (ROOT/'public/voicebox-resonance-lab.html').write_text(page)
    for row in report['files']+report['plots']:assert sha(ROOT/row['file'])==row['sha256']
    print(json.dumps({'cases':len(report['cases']),'clips':len(report['files']),'plots':len(report['plots']),
        'maximumPeak':max(r['peak'] for r in report['files']),'maximumRmsError':max(r['rmsError'] for r in report['files'])},indent=2))


if __name__=='__main__':main()
