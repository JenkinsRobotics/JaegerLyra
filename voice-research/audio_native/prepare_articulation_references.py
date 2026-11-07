"""Fresh synthetic consonant/vowel references for the output-only instrument."""
import base64
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-articulation/references'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Synthetic teacher references for captured onset-to-sustain control; intended phone labels are not listener-verified',
        'teacher':'Existing local Kokoro English speech API','newNeuralTraining':False,'clips':[],'complete':False}
    for speaker in ('af_heart','am_michael','af_bella'):
        for unit,text,onset in (('me','Me.','nasal m'),('see','See.','fricative s'),('bee','Bee.','stop b')):
            name=speaker+'-'+unit;file=OUT/(name+'.wav')
            if not file.exists():
                request=urllib.request.Request('http://127.0.0.1:8791/api/voice',
                    json.dumps({'engine':'expressive','speaker':speaker,'text':text}).encode(),
                    {'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=60) as response:result=json.load(response)
                file.write_bytes(base64.b64decode(result['audio']))
            report['clips'].append({'name':name,'speaker':speaker,'unit':unit,'text':text,
                'intendedOnset':onset,'intendedVowel':'ee','file':str(file.relative_to(ROOT)),
                'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
            (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(name,flush=True)
    report['complete']=True;(OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
