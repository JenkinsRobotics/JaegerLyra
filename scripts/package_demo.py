"""Build a self-contained, hash-checked reconstruction demo from local evidence.

Run only when intentionally rebuilding the bundle. No model downloads, speech
generation, playback, or weight training occur here. Original research evidence
is read-only. Phrase pulse analysis enables the advertised vibrato controls.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'voice-research/audio_native'))
import soundfile as sf
from neural_waveform import load_model,RATE
from neural_performance import compile_performance,EncodedPerformance,load_performance,save_performance

SOURCE=ROOT/'evidence/live-voicebox/word-playback-v2/manifest.json'
OUT=ROOT/'demo/bundle'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest=json.loads(SOURCE.read_text());OUT.mkdir(parents=True,exist_ok=True)
    manifest['bundleFormatVersion']=1
    manifest['project']='Jaeger Lyra'
    manifest['scope']='Reconstruction playback of captured Kokoro-derived performances; not independently generated new speech or audio understanding'
    manifest['provenance']={'source':'Kokoro-generated synthetic recordings',
        'runtime':'Independent neural waveform decoder and saved continuous acoustic controls',
        'kokoroModelLoadedAtPlayback':False,'originalWavUsedForLiveSynthesis':False,
        'referenceWavsUsedByComparisonPlayers':True,'newNeuralTraining':False,
        'sourceManifest':'evidence/live-voicebox/word-playback-v2/manifest.json'}
    copied={}
    def copy(relative,expected=None):
        source=ROOT/relative
        if expected is not None and sha(source)!=expected:raise ValueError('Changed source: '+relative)
        if relative in copied:return copied[relative]
        assert relative.startswith('evidence/')
        target=OUT/relative.removeprefix('evidence/');target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        assert sha(source)==sha(target)
        copied[relative]=str(target.relative_to(ROOT));return copied[relative]
    model_path=ROOT/manifest['model'];assert sha(model_path)==manifest['modelSha256']
    encoder,_,_=load_model(model_path)
    manifest['model']=copy(manifest['model'],manifest['modelSha256'])
    for group in ('units','words','phrases'):
        for item in manifest[group]:
            original_artifact=ROOT/item['artifact']
            item['artifact']=copy(item['artifact'],item['artifactSha256'])
            for key,hash_key in (('referenceFile','referenceSha256'),('sourceFile','sourceSha256'),
                    ('reconstructionFile','reconstructionSha256')):
                if key in item:item[key]=copy(item[key],item.get(hash_key))
            for choice in item.get('choices',[]):
                choice['file']=copy(choice['file'],choice['sha256'])
                if choice['label']=='Voicebox reconstruction':choice['label']='Lyra reconstruction'
            if group=='phrases':
                performance,metadata=load_performance(original_artifact)
                audio,rate=sf.read(ROOT/item['referenceFile']);assert rate==RATE
                pulse_capture=compile_performance(audio,encoder)
                # Retain the exact encoded matrix; only add source pulse timing.
                target=ROOT/item['artifact']
                save_performance(target,EncodedPerformance(performance.matrix,pulse_capture.pulses),
                    metadata|{'pulsePreparation':'Offline Praat analysis of the same source recording for performance controls'})
                item['artifactSha256']=sha(target);item['pulseCount']=len(pulse_capture.pulses)
                if item['pulseCount']<3:raise ValueError('Phrase lacks usable pulse timing')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)}
        for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='checksums.json']
    (OUT/'checksums.json').write_text(json.dumps({'scope':'Prepared synthetic demo bundle','files':files},indent=2)+'\n')
    print('Bundled',len(files),'files;',round(sum(r['bytes'] for r in files)/2**20,1),'MiB. Nine phrases now carry offline pulse timing.')


if __name__=='__main__':main()
