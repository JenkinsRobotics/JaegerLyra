"""Prepare matched whole-clip A/B files; never start audio playback."""
import argparse,html,json,random
from pathlib import Path
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'public/research/voicebox-output'


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='evidence/neural-waveform/output-validation/manifest.json')
    parser.add_argument('--report',default='evidence/neural-waveform/axis-v1/evaluation/report.json')
    parser.add_argument('--candidate-label',default='axis-neural')
    parser.add_argument('--output',default='public/research/voicebox-output')
    parser.add_argument('--page')
    parser.add_argument('--study-label',default='OUTPUT STUDY 2')
    args=parser.parse_args();OUT=ROOT/args.output
    relative=OUT.relative_to(ROOT/'public');sample_root='/'+relative.as_posix()+'/'
    inputs=json.loads((ROOT/args.manifest).read_text())
    report=json.loads((ROOT/args.report).read_text())
    if not report['complete'] or len(report['results'])!=report['expectedCases']:
        raise ValueError('Finish evaluation before packaging auditions')
    outputs={r['fixture']:r for r in report['results'] if r['mode']==args.candidate_label}
    pairs=[];rng=random.Random(104);OUT.mkdir(parents=True,exist_ok=True)
    clips=[c for c in inputs['clips'] if c.get('kind','fresh-kokoro')=='fresh-kokoro']
    for index,clip in enumerate(clips):
        generated=outputs['fresh-'+clip['name']]
        if generated['status']!='ok':raise ValueError('Cannot hide a failed output')
        loaded=[]
        for label,source in [('Kokoro original',clip['file']),('Experimental voicebox',generated['file'])]:
            audio,rate=sf.read(ROOT/source);lufs=pyln.Meter(rate).integrated_loudness(audio)
            loaded.append((label,source,audio,rate,lufs))
        target=min(-20.,*(lufs+20*np.log10(.9/max(abs(audio))) for _,_,audio,_,lufs in loaded))
        rng.shuffle(loaded);choices=[]
        for choice,(label,source,audio,rate,lufs) in zip(('A','B'),loaded):
            gain=10**((target-lufs)/20);path=OUT/f'{index:02d}-{choice.lower()}.wav'
            sf.write(path,audio*gain,rate,subtype='PCM_24');saved,_=sf.read(path)
            measured=pyln.Meter(rate).integrated_loudness(saved)
            if abs(measured-target)>.01 or max(abs(saved))>.901:raise ValueError('Audition level error')
            choices.append({'choice':choice,'label':label,'url':path.name,'source':source,
                'gain':gain,'lufs':measured,'samplePeak':float(max(abs(saved)))})
        pairs.append({'id':clip['name'],'voice':{'af_heart':'Heart','am_michael':'Michael','af_bella':'Bella'}[clip['speaker']],
            'text':clip['text'],'targetLufs':target,'choices':choices})
    manifest={'scope':'Matched listening comparison of output reconstruction, not new-speech generation',
        'selection':'All fresh Kokoro test clips, in predefined order; whole recordings; separate singing diagnostics excluded',
        'method':'Fixed gain to matched integrated BS.1770 loudness; no limiting, denoising or pitch editing',
        'labelOrder':'Deterministic shuffled A/B; informal label-hidden comparison, not a formal blinded study',
        'truePeakMeasured':False,'candidateSha256':report['candidateSha256'],'pairs':pairs}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.page:
        template=(ROOT/'public/voicebox-output-lab.html').read_text()
        if args.page=='voicebox-output-lab.html' or Path(args.page).name!=args.page:
            raise ValueError('Choose a separate HTML filename in public')
        page=template.replace('data-sample-root="/research/voicebox-output/"',f'data-sample-root="{sample_root}"')
        label=html.escape(args.study_label)
        page=page.replace('KIN / VOICE RESEARCH','KIN / '+label).replace('<title>KIN · Voicebox listening room</title>',
            '<title>KIN · Voicebox listening room · '+label+'</title>')
        page=page.replace('All nine phrases were generated','All nine recordings were generated')
        (ROOT/'public'/args.page).write_text(page)
    print(json.dumps({'pairs':len(pairs),'manifest':str(OUT/'manifest.json')}))


if __name__=='__main__':main()
