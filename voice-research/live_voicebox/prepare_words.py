"""Prepare naturally timed whole words separately from intentional sustains."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'voice-research/audio_native'))
from neural_waveform import load_model,RATE
from compiled_neural_decoder import compile_decoder
from neural_performance import compile_performance,save_performance,load_performance

OUT=ROOT/'evidence/live-voicebox/word-playback-v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    manifest=json.loads((ROOT/'evidence/live-voicebox/v1/manifest.json').read_text())
    sources=json.loads((ROOT/'evidence/neural-syllables/teacher-transfer-v1/capture.json').read_text())['sources']
    if sha(ROOT/manifest['model'])!=manifest['modelSha256']:raise ValueError('Changed decoder')
    encoder,decoder,_=load_model(ROOT/manifest['model']);decoder=compile_decoder(decoder)
    OUT.mkdir(parents=True,exist_ok=True);words=[]
    for unit in manifest['units']:
        source=next(s for s in sources if s['name']==unit['id']);file=ROOT/source['file']
        if sha(file)!=source['sha256']:raise ValueError('Changed original word')
        audio,rate=sf.read(file);assert rate==RATE and audio.ndim==1
        # Trim quiet edges only; never remove pauses or alter timing inside a word.
        rms=np.sqrt(np.convolve(audio**2,np.ones(240)/240,mode='same'))
        active=np.flatnonzero(rms>max(float(max(rms))*.005,1e-5))
        if not len(active):raise ValueError('Silent word')
        start=max(0,int(active[0])-480);stop=min(len(audio),int(active[-1])+481)
        natural=audio[start:stop];performance=compile_performance(natural,encoder)
        artifact=OUT/(unit['id']+'-word.npz')
        metadata={'sourceFile':source['file'],'sourceSha256':source['sha256'],
            'sourceStartSample':start,'sourceStopSample':stop,'naturalTiming':True,
            'edgeTrim':'10 ms RMS, 0.5% of peak RMS, 20 ms padding; no interior changes'}
        save_performance(artifact,performance,metadata);reopened,_=load_performance(artifact)
        decoded=decoder.render(reopened.matrix);gain=min(1.,.85/max(float(max(abs(decoded))),1e-12))
        reference=OUT/(unit['id']+'-word-reconstruction.wav');sf.write(reference,decoded,RATE,subtype='FLOAT')
        words.append({k:unit[k] for k in ('id','voice','word','baseHz')}|
            {'artifact':str(artifact.relative_to(ROOT)),'artifactSha256':sha(artifact),
             'sampleCount':len(natural),'seconds':len(natural)/RATE,'streamGain':gain,
             'referenceFile':source['file'],'referenceSha256':source['sha256'],
             'reconstructionFile':str(reference.relative_to(ROOT)),'reconstructionSha256':sha(reference),
             'waveformSnrDb':float(10*np.log10(np.sum(natural**2)/max(np.sum((natural-decoded)**2),1e-30))),**metadata})
    manifest.update(words=words,defaultPlayback='word',
        sourceData='Original naturally timed word recordings for whole-word playback; prior sustained units are only for explicit hold mode. Existing nine speech comparisons unchanged.',
        scope='Naturally timed whole-word reconstruction by default, optional held-note resynthesis, and prepared speech comparisons',
        complete=len(words)==18 and len(manifest['units'])==18 and len(manifest['phrases'])==9)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'words':len(words),'minimumSeconds':min(w['seconds'] for w in words),
        'maximumSeconds':max(w['seconds'] for w in words),'medianReconstructionSnrDb':float(np.median([w['waveformSnrDb'] for w in words]))},indent=2))


if __name__=='__main__':main()
