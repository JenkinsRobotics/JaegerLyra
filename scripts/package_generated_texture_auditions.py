"""Four-way output texture comparison, all voices, no automatic playback."""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'public/research/generated-texture'


def main():
    study=json.loads((ROOT/'evidence/generated-texture/v1/report.json').read_text())
    compact=json.loads((ROOT/'evidence/compact-vocal/v1/report.json').read_text())
    transitions=json.loads((ROOT/'evidence/vocal-shapes/v1/report.json').read_text())
    vowels=json.loads((ROOT/'evidence/neural-vowels/v1/report.json').read_text())
    originals=json.loads((ROOT/'evidence/neural-vowels/references/manifest.json').read_text())
    if not study['complete']:raise ValueError('Incomplete texture study')
    OUT.mkdir(parents=True,exist_ok=True);rng=random.Random(1809);groups=[]
    def package(name,items,shuffle=True):
        loaded=[]
        for label,source in items:
            audio,rate=sf.read(ROOT/source);lufs=pyln.Meter(rate).integrated_loudness(audio)
            loaded.append((label,source,audio,rate,lufs))
        target=min(-20.,*(lufs+20*np.log10(.9/max(abs(audio))) for _,_,audio,_,lufs in loaded))
        if shuffle:rng.shuffle(loaded)
        files=[]
        for i,(label,source,audio,rate,lufs) in enumerate(loaded):
            gain=10**((target-lufs)/20);file=OUT/(name+'-'+str(i)+'.wav')
            sf.write(file,audio*gain,rate,subtype='PCM_24');saved,_=sf.read(file)
            actual=pyln.Meter(rate).integrated_loudness(saved)
            if abs(actual-target)>.01 or max(abs(saved))>.901:raise ValueError('Audition level error')
            files.append({'label':label,'file':file.name,'source':source,'sourceSha256':hashlib.sha256((ROOT/source).read_bytes()).hexdigest(),
                'gain':gain,'lufs':actual,'samplePeak':float(max(abs(saved)))})
        return files
    for speaker,voice in (('af_heart','Heart'),('am_michael','Michael'),('af_bella','Bella')):
        source=next(r for r in originals['clips'] if r['name']==speaker+'-ah')
        original=package(speaker+'-source',[('Original Kokoro reference',source['file'])],False)[0]
        for kind,mode,title in (('held','native','Held “ah”'),('transition','melody','Vowel melody')):
            def selected(rows,method):return next(r for r in rows if r['speaker']==speaker and r['kind']==kind and r['mode']==mode
                and r['method']==method and (kind=='transition' or r['vowel']=='ah'))
            base=selected(compact['results'],'compact-harmonic-noise')
            learned=selected(study['results'],'learned-seed-0');iid=selected(study['results'],'iid-seed-0')
            second=selected(study['results'],'learned-seed-1')
            if kind=='held':grain=next(r for r in vowels['results'] if r['fixture']==speaker+'-ah' and r['mode']=='native')
            else:grain=next(r for r in transitions['results'] if r['speaker']==speaker and r['mode']=='melody' and r['method']=='grain-crossfade')
            choices=package(speaker+'-'+kind,[('Static compact instrument',base['file']),
                ('Generated texture · learned timing',learned['file']),('Generated texture · independent frames',iid['file']),
                ('Neural captured-vowel reference',grain['file'])])
            alternate=package(speaker+'-'+kind+'-second',[('Generated texture · second random take',second['file'])],False)[0]
            groups.append({'voice':voice,'title':title,
                'description':'One captured vowel held at its source pitch.' if kind=='held' else 'Four authored notes with “ah → ee → oo → ah” vowel changes.',
                'choices':choices,'source':original,'breath':alternate,
                'breathNote':'The same learned model and controls, with a different random seed. This take is also unaccepted.'})
    manifest={'scope':'Static, stochastic and neural captured-vowel outputs; no listening-quality winner',
        'selection':'All three predefined voices, held-ah and melody; seed 0 in main comparisons and seed 1 separately, with original Kokoro voice guides',
        'processing':'Whole clips with constant gain, matching integrated loudness within each A/B/C/D group; no limiting, denoising or pitch edits',
        'labelOrder':'Deterministic shuffled labels, informal comparison rather than a formal blinded study',
        'truePeakMeasured':False,'groups':groups}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    page=(ROOT/'public/voicebox-instrument-lab.html').read_text()
    changes={
        '<title>KIN · Vocal instrument comparison</title>':'<title>KIN · Voice texture comparison</title>',
        'KIN / VOCAL INSTRUMENT':'KIN / VOICE TEXTURE',
        'One voice. Different ways to sing.':'Voice texture comparison',
        'Three experimental instruments follow the same notes and vowel choices. Listen for a stable character, clear vowels and smooth changes. These are captured-vowel experiments, with no quality winner selected.':
            'Compare static synthesis, generated variation and a captured-vowel reference at the same pitch and vowel settings. Listen for a stable character and natural texture. No quality winner has been selected.',
        'grid-template-columns:repeat(3,1fr)':'grid-template-columns:repeat(4,1fr)',
        '@media(max-width:700px)':'@media(max-width:950px){.players{grid-template-columns:repeat(2,1fr)}}@media(max-width:700px)',
        'Original voice and added breath':'Original voice and another generated take',
        'Compact instrument with added breath.':'Another take from the learned texture model.',
        'aria-label="Experimental added breath"':'aria-label="Second generated texture take"',
        "const root='/research/compact-vocal/'":"const root='/research/generated-texture/'",
        "['a','b','c']":"['a','b','c','d']",
        'Reveal the instruments':'Reveal the methods',
        'The compact instrument uses harmonic and filtered-noise synthesis. The neural captured-vowel instrument reuses a short decoded sound patch; WORLD is an external vocoder reference. None demonstrates new words, general voice cloning or accepted Kokoro-level quality. Kokoro remains the studio’s voice.':
            'Generated texture uses a small stochastic model, with no recorded trajectory or future source audio supplied during rendering. The neural captured-vowel reference reuses a short sound patch. These experiments do not demonstrate new words, general cloning or accepted Kokoro-level quality. Kokoro remains the studio’s voice.',
        'Every predefined voice is included. Each A/B/C group uses whole clips with constant gain to match integrated loudness; there is no limiting, denoising or pitch correction. Labels are shuffled for an informal comparison. The research includes all 69 new cases, including resonance outliers and an added-breath pitch-tracker failure.':
            'Every predefined voice is included. Each A/B/C/D group uses whole clips with constant gain to match loudness; there is no limiting, denoising or pitch correction. Labels are shuffled for an informal comparison. All 126 research renders are retained. Generated variation often reached its modulation bound, and added-breath pitch-tracker failures remain unresolved.'}
    for a,b in changes.items():
        if a not in page:raise ValueError('Changed listening template: '+a[:80])
        page=page.replace(a,b)
    needle=' </div>\n <div class="actions">'
    extra='  <section class="player" aria-label="Version D"><div class="letter">D</div><div class="identity" id="identity-d">Label hidden</div><audio id="audio-d" aria-label="Listen to version D" controls preload="none"></audio></section>\n'
    if needle not in page:raise ValueError('Missing player insertion point')
    page=page.replace(needle,extra+needle,1)
    (ROOT/'public/voicebox-texture-lab.html').write_text(page)
    print('Saved six four-way comparisons, six second takes and three originals; no playback')


if __name__=='__main__':main()
