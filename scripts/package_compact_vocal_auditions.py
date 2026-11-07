"""Package every predefined voice, with equal-loudness comparisons; no playback."""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'public/research/compact-vocal'


def main():
    study=json.loads((ROOT/'evidence/compact-vocal/v1/report.json').read_text())
    transitions=json.loads((ROOT/'evidence/vocal-shapes/v1/report.json').read_text())
    vowels=json.loads((ROOT/'evidence/neural-vowels/v1/report.json').read_text())
    sources=json.loads((ROOT/'evidence/neural-vowels/references/manifest.json').read_text())
    if not study['complete']:raise ValueError('Incomplete study')
    OUT.mkdir(parents=True,exist_ok=True)
    rng=random.Random(731);groups=[]
    def package(name,items,shuffle=True):
        loaded=[]
        for label,source in items:
            audio,rate=sf.read(ROOT/source);lufs=pyln.Meter(rate).integrated_loudness(audio)
            loaded.append((label,source,audio,rate,lufs))
        target=min(-20.,*(lufs+20*np.log10(.9/max(abs(audio))) for _,_,audio,_,lufs in loaded))
        if shuffle:rng.shuffle(loaded)
        results=[]
        for i,(label,source,audio,rate,lufs) in enumerate(loaded):
            gain=10**((target-lufs)/20);file=OUT/(name+'-'+str(i)+'.wav')
            sf.write(file,audio*gain,rate,subtype='PCM_24')
            saved,_=sf.read(file);actual=pyln.Meter(rate).integrated_loudness(saved)
            if abs(actual-target)>.01 or max(abs(saved))>.901:raise ValueError('Level validation failed')
            results.append({'label':label,'file':file.name,'source':source,'sourceSha256':hashlib.sha256((ROOT/source).read_bytes()).hexdigest(),
                'gain':gain,'lufs':actual,'samplePeak':float(max(abs(saved)))})
        return results
    for speaker,voice in (('af_heart','Heart'),('am_michael','Michael'),('af_bella','Bella')):
        original=next(r for r in sources['clips'] if r['name']==speaker+'-ah')
        source=package(speaker+'-source',[('Original Kokoro reference: “Ah, ah, ah.”',original['file'])],False)[0]
        breath=next(r for r in study['results'] if r['speaker']==speaker and r['kind']=='breath')
        breath_file=package(speaker+'-breath',[('Compact instrument · breath 0.35',breath['file'])],False)[0]
        for mode in ('held','melody'):
            if mode=='held':
                row=next(r for r in study['results'] if r['speaker']==speaker and r['kind']=='held' and r['vowel']=='ah' and r['mode']=='native' and r['method']=='compact-harmonic-noise')
                world=next(r for r in study['results'] if r['speaker']==speaker and r['kind']=='held' and r['vowel']=='ah' and r['mode']=='native' and r['method']=='world-shapes')
                grain=next(r for r in vowels['results'] if r['fixture']==speaker+'-ah' and r['mode']=='native')
                title='Held “ah”';description='One captured vowel, sustained for two seconds at its source pitch.'
                grain_label='Neural captured-vowel instrument'
            else:
                row=next(r for r in study['results'] if r['speaker']==speaker and r['kind']=='transition' and r['mode']=='melody')
                world=next(r for r in transitions['results'] if r['speaker']==speaker and r['mode']=='melody' and r['method']=='world-shapes')
                grain=next(r for r in transitions['results'] if r['speaker']==speaker and r['mode']=='melody' and r['method']=='grain-crossfade')
                title='Vowel melody';description='Four authored notes with “ah → ee → oo → ah” vowel changes.'
                grain_label='Neural captured-vowel crossfade'
            choices=package(speaker+'-'+mode,[('Compact harmonic/noise instrument',row['file']),
                ('External WORLD reference',world['file']),(grain_label,grain['file'])])
            groups.append({'voice':voice,'mode':mode,'title':title,'description':description,'choices':choices,
                'source':source,'breath':breath_file,
                'breathNote':'Michael’s added-breath sample produced an octave error in Praat; WORLD stayed within about 12.5 cents.' if speaker=='am_michael' else 'Added breath is an experimental noise control; naturalness is unverified.'})
    manifest={'scope':'Captured-vowel instruments, not generated speech; no accepted quality winner',
        'selection':'All three predefined voices, each held-ah and melody; additional breath samples retain the Michael tracker failure',
        'processing':'Whole clips, matched integrated loudness within each triplet with constant gain only, -20 LUFS or lower for sample peaks; no limiting, denoising or pitch edits',
        'labelOrder':'Deterministically shuffled A/B/C; informal comparison, not a formal blinded study',
        'truePeakMeasured':False,'groups':groups}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('Packaged six triplets, three original references and three breath controls; no playback')


if __name__=='__main__':main()
