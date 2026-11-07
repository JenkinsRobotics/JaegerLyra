"""New synthetic evaluation sentences after the six initial variants were fixed."""
import base64,hashlib,json,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/source-detail/fresh-inputs'
LINES=[
    'She left six fresh peaches beside the kitchen window.',
    'Please close the blue suitcase before we catch the train.',
]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest={'purpose':'New sentences after initial source-detail results; compare unchanged baseline and colored-noise candidate without retuning',
        'origin':'Original prompts synthesized by local Kokoro; evaluation only, no target fitting',
        'selectedModes':['baseline','colored-noise'],'clips':[]}
    for voice in ('af_heart','am_michael','af_bella'):
        for index,text in enumerate(LINES):
            name=f'{voice}-{index}';path=OUT/(name+'.wav')
            if not path.exists():
                request=urllib.request.Request('http://127.0.0.1:8791/api/voice',
                    json.dumps({'engine':'expressive','speaker':voice,'text':text}).encode(),
                    {'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=60) as response:result=json.load(response)
                path.write_bytes(base64.b64decode(result['audio']))
            anchor=(f'evidence/voice-fixtures/{voice}-reference.wav' if voice!='af_bella'
                    else 'evidence/streaming-voicebox/validation/af_bella-0.wav')
            row={'name':name,'speaker':voice,'text':text,'file':str(path.relative_to(ROOT)),
                 'reference':anchor,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            manifest['clips'].append(row)
            (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(name,flush=True)


if __name__=='__main__':main()
