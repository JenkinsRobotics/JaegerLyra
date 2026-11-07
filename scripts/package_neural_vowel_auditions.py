"""First predefined Heart/ah source and controls, matched loudness, no playback."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/neural-vowels/v1/audition'


def main():
    report=json.loads((OUT.parent/'report.json').read_text())
    if not report['complete']:raise ValueError('Incomplete control panel')
    fixtures=json.loads((ROOT/'evidence/neural-vowels/references/manifest.json').read_text())
    first=fixtures['clips'][0]
    paths={'kokoro-source':first['file']}
    for mode in ('native','melody','long-hold'):
        paths[mode]=next(r['file'] for r in report['results'] if r['fixture']==first['name'] and r['mode']==mode)
    OUT.mkdir(exist_ok=True)
    manifest={'fixture':first['name'],'selection':'First predefined reference; no selection by quality metrics',
        'scope':'Original Kokoro phrase versus newly sustained/retimed captured vowel; different performances, not an identical-utterance A/B quality test',
        'processing':'One fixed gain per clip to -20 LUFS or lower if needed for 0.90 peak; no limiting, denoising or pitch correction',
        'files':[]}
    for name,file in paths.items():
        original=ROOT/file;audio,rate=sf.read(original)
        loudness=pyln.Meter(rate).integrated_loudness(audio)
        gain=min(10**((-20-loudness)/20),.90/max(abs(audio).max(),1e-12))
        path=OUT/(name+'.wav');sf.write(path,audio*gain,rate,subtype='PCM_24')
        manifest['files'].append({'name':name,'source':file,
            'sourceSha256':hashlib.sha256(original.read_bytes()).hexdigest(),
            'file':str(path.relative_to(ROOT)),'gain':float(gain),'sourceLufs':float(loudness)})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('Saved four first-reference auditions without playback')


if __name__=='__main__':main()
