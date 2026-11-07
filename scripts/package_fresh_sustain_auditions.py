"""Preserve every fresh/repair clip with matched levels and explicit scope."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/neural-articulation/fresh-sustain-v1/audition'


def main():
    fresh=json.loads((ROOT/'evidence/neural-articulation/fresh-sustain-v1/report.json').read_text())
    captured=json.loads((ROOT/'evidence/neural-articulation/fresh-sustain-v1/capture.json').read_text())
    repair=json.loads((ROOT/'evidence/neural-articulation/variable-span-v1/report.json').read_text())
    if not all(j['complete'] for j in (fresh,captured,repair)):raise ValueError('Incomplete studies')
    OUT.mkdir(parents=True,exist_ok=True);files=[];groups=[];reels=[]
    for study,units,rows in [('frozen fresh-word validation',captured['units'],fresh['results']),
                              ('post-validation capture repair',repair['units'],repair['results'])]:
        for source in units:
            items=[('original','source',source['file']),('trimmed','source',source['trimmedSource']),
                   ('decoded','source',source['decodedSource'])]
            items += [(r['mode'],r['method'],r['file']) for r in rows
                      if r['speaker']==source['speaker'] and r['unit']==source['unit']]
            loaded=[]
            for mode,method,path in items:
                audio,rate=sf.read(ROOT/path);lufs=pyln.Meter(rate,block_size=.2).integrated_loudness(audio)
                loaded.append((mode,method,path,audio,rate,lufs))
            target=min(-20.,*(lufs+20*np.log10(.9/max(abs(audio))) for _,_,_,audio,_,lufs in loaded))
            packed={};samples={}
            for mode,method,path,audio,rate,lufs in loaded:
                gain=10**((target-lufs)/20);file=OUT/(source['name']+'-'+mode+'-'+method+'.wav')
                sf.write(file,audio*gain,rate,subtype='PCM_24');saved,_=sf.read(file)
                measured=pyln.Meter(rate,block_size=.2).integrated_loudness(saved)
                if abs(measured-target)>.01 or max(abs(saved))>.901:raise ValueError('Invalid audition level')
                row={'file':str(file.relative_to(ROOT)),'source':path,'sourceSha256':hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
                    'gain':gain,'lufs200MsGate':measured,'samplePeak':float(max(abs(saved))),'mode':mode,'method':method,'study':study}
                packed[(mode,method)]=row;samples[(mode,method)]=saved;files.append(row)
            for mode in ('native','melody','short-native'):
                groups.append({'speaker':source['speaker'],'word':source['unit'],'mode':mode,'study':study,
                    'original':packed[('original','source')],'trimmed':packed[('trimmed','source')],
                    'decoded':packed[('decoded','source')],
                    **{method:packed[(mode,method)] for method in ('neural-loop','neural-fixed','pcm-loop','pcm-fixed')}})
            # Fixed order chosen for every source, without transcript/quality selection.
            order=[('original','source'),('native','neural-fixed'),('native','pcm-fixed'),('melody','neural-fixed')]
            parts=[];segments=[];cursor=0
            for key in order:
                sound=samples[key];segments.append({'mode':key[0],'method':key[1],'startSeconds':cursor/24000,
                    'endSeconds':(cursor+len(sound))/24000,'file':packed[key]['file']})
                parts.extend([sound,np.zeros(6000)]);cursor+=len(sound)+6000
            reel=OUT/(source['name']+'-comparison.wav');sf.write(reel,np.concatenate(parts[:-1]),24000,subtype='PCM_24')
            reels.append({'speaker':source['speaker'],'word':source['unit'],'study':study,
                'file':str(reel.relative_to(ROOT)),'segments':segments,
                'scope':'Separate clips joined with 250 ms silence, not connected speech or a live performance'})
    manifest={'scope':'All fresh-word and subsequent repair outputs; repaired cases are labelled development evidence',
        'processing':'Whole clips with constant gain, common level per source using a disclosed 200 ms loudness gate; target -20 LUFS or lower to keep sample peak below 0.9. No limiting or denoising.',
        'originalReference':'Ordinary Kokoro speech retains its original pitch, timing and silence.',
        'truePeakMeasured':False,'autoplay':False,'listeningAccepted':False,
        'files':files,'groups':groups,'reels':reels}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('Matched',len(files),'clips;',len(groups),'groups;',len(reels),'labelled concatenations; no playback.')


if __name__=='__main__':main()
