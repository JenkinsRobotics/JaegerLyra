"""Longer synthetic vowel references for output texture modelling; no playback."""
import base64
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/generated-texture/references'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    decoder=ROOT/'evidence/neural-waveform/axis-detail-v2/projected-r1e-7.npz'
    report={'scope':'Repeated synthetic vowel sources for offline output-texture capture; intended labels are not listener-verified',
        'source':'Existing local Kokoro English speech API; synthetic teacher-derived data, not independently sourced training',
        'decoder':str(decoder.relative_to(ROOT)),'decoderSha256':hashlib.sha256(decoder.read_bytes()).hexdigest(),
        'clips':[],'complete':False}
    for speaker in ('af_heart','am_michael','af_bella'):
        for vowel,phrase in (('ah','Ah, ah, ah.'),('ee','Eee, eee, eee.'),('oo','Ooh, ooh, ooh.')):
            text=' '.join([phrase]*3);name=speaker+'-'+vowel;file=OUT/(name+'.wav')
            if not file.exists():
                request=urllib.request.Request('http://127.0.0.1:8791/api/voice',
                    json.dumps({'engine':'expressive','speaker':speaker,'text':text}).encode(),
                    {'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=60) as response:result=json.load(response)
                file.write_bytes(base64.b64decode(result['audio']))
            report['clips'].append({'name':name,'speaker':speaker,'intendedVowel':vowel,'text':text,
                'file':str(file.relative_to(ROOT)),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
            (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
            print(name,flush=True)
    report['complete']=True;(OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
