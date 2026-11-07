"""Fresh synthetic diphthongs: fixed development and held-out word lists."""
import base64
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/resonance-morph/v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    config={'speakers':['af_heart','am_michael','af_bella'],
        'groups':{'ai':{'train':'eye','validation':['high','sigh']},
            'au':{'train':'ow','validation':['how','cow']},
            'oi':{'train':'oy','validation':['boy','toy']}},
        'endpointPositions':[.2,.8],'endpointPulseNeighbours':4,'minimumEndpointSeconds':.080,
        'maximumPatchSeconds':.2,'fftSize':2048,'filterTaps':512,'maximumFilterGainDb':18,
        'trajectoryKnots':9,'trajectoryGrid':41,'envelopeBandHz':[100,6000],
        'methods':['grain-linear','grain-learned','filter-linear','filter-learned'],
        'scope':'Source-only endpoint capture and fitted spectral trajectory from development words; held-out words supply timing, pitch and loudness solely to isolate spectral-articulation error.',
        'phoneLabels':'Intended English diphthongs, not listener-confirmed labels',
        'source':'Existing local Kokoro API; source-derived synthetic fixtures',
        'noAutoplay':True,'listeningAccepted':False}
    config_file=OUT/'config.json'
    if config_file.exists():
        if json.loads(config_file.read_text())!=config:raise ValueError('Frozen configuration differs')
    else:config_file.write_text(json.dumps(config,indent=2)+'\n')
    manifest={'config':'evidence/resonance-morph/v1/config.json',
        'configSha256':hashlib.sha256(config_file.read_bytes()).hexdigest(),'clips':[],'complete':False}
    for speaker in config['speakers']:
        for group,words in config['groups'].items():
            for split,word in [('train',words['train'])]+[('validation',w) for w in words['validation']]:
                name=speaker+'-'+word;file=OUT/(name+'-source.wav')
                if not file.exists():
                    request=Request('http://127.0.0.1:8791/api/voice',json.dumps(
                        {'engine':'expressive','speaker':speaker,'text':word.capitalize()+'.'}).encode(),
                        {'Content-Type':'application/json'})
                    with urlopen(request,timeout=90) as response:result=json.load(response)
                    file.write_bytes(base64.b64decode(result['audio']))
                manifest['clips'].append({'name':name,'speaker':speaker,'group':group,'split':split,
                    'word':word,'file':str(file.relative_to(ROOT)),
                    'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
                (OUT/'references.json').write_text(json.dumps(manifest,indent=2)+'\n')
                print(name,split,flush=True)
    manifest['complete']=True;(OUT/'references.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
