"""Prepare fixed controls and honest listening references; never play audio."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'voice-research/audio_native'))
from neural_waveform import load_model,RATE
from compiled_neural_decoder import compile_decoder
from neural_performance import EncodedPerformance,save_performance,load_performance

OUT=ROOT/'evidence/live-voicebox/v1'
MODEL='evidence/neural-waveform/axis-detail-v2/projected-r1e-7.npz'
MODEL_SHA='ac2debf521a0936a2ee9b7c9cb5cac86d550612de78a7d57233363b28b58a510'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if sha(ROOT/MODEL)!=MODEL_SHA:raise ValueError('Changed selected model')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'samples').mkdir(exist_ok=True)
    encoder,decoder,_=load_model(ROOT/MODEL);decoder=compile_decoder(decoder)
    captured=json.loads((ROOT/'evidence/neural-syllables/teacher-transfer-v1/capture.json').read_text())
    if not captured['complete']:raise ValueError('Incomplete syllables')
    units=[]
    for row in captured['units']:
        if row['variant']!='level':continue
        if sha(ROOT/row['artifact'])!=row['artifactSha256']:raise ValueError('Changed syllable')
        units.append({'id':row['speaker']+'-'+row['word'],'voice':{'af_heart':'Heart','am_michael':'Michael','af_bella':'Bella'}[row['speaker']],
            'word':row['word'],'baseHz':row['teacherPitchHz'],'holdStartSeconds':row['holdStartSample']/RATE,
            'artifact':row['artifact'],'artifactSha256':row['artifactSha256']})
    inputs=json.loads((ROOT/'evidence/neural-waveform/output-detail-validation/manifest.json').read_text())
    earlier=json.loads((ROOT/'public/research/voicebox-output-v2/manifest.json').read_text())
    if earlier['candidateSha256']!=MODEL_SHA:raise ValueError('Comparison uses another decoder')
    sources={r['name']:r for r in inputs['clips']};phrases=[];meter=pyln.Meter(RATE)
    for pair in earlier['pairs']:
        row=sources[pair['id']];file=ROOT/row['file']
        if sha(file)!=row['sha256']:raise ValueError('Changed speech fixture')
        original,rate=sf.read(file);assert rate==RATE
        matrix=encoder.encode(original);artifact=OUT/(pair['id']+'.npz')
        save_performance(artifact,EncodedPerformance(matrix,np.empty(0)),{'source':row['file'],'sourceSha256':row['sha256'],'scope':'Prepared natural-speech reconstruction, not new text synthesis'})
        reopened,_=load_performance(artifact);decoded=decoder.render(reopened.matrix)
        levels=[meter.integrated_loudness(x) for x in (original,decoded)]
        target=min(-20.,*(lufs+20*np.log10(.8/max(abs(x))) for x,lufs in zip((original,decoded),levels)))-.1
        choices=[];stream_gain=None
        for choice in pair['choices']:
            is_original=choice['label']=='Kokoro original';audio=original if is_original else decoded
            lufs=levels[0 if is_original else 1];gain=10**((target-lufs)/20)
            output=OUT/'samples'/(pair['id']+'-'+choice['choice'].lower()+'.wav')
            sf.write(output,audio*gain,RATE,subtype='PCM_24');saved,_=sf.read(output)
            measured=meter.integrated_loudness(saved)
            if abs(measured-target)>.01 or max(abs(saved))>.801:raise ValueError('Invalid comparison levels')
            if not is_original:stream_gain=gain
            choices.append({'choice':choice['choice'],'label':'Kokoro original' if is_original else 'Voicebox reconstruction',
                'file':str(output.relative_to(ROOT)),'sha256':sha(output),'gain':gain,'lufs':measured,'peak':float(max(abs(saved)))})
        raw=OUT/(pair['id']+'-reconstruction.wav');sf.write(raw,decoded,RATE,subtype='FLOAT')
        phrases.append({'id':pair['id'],'voice':pair['voice'],'text':pair['text'],'sampleCount':len(original),
            'seconds':len(original)/RATE,'artifact':str(artifact.relative_to(ROOT)),'artifactSha256':sha(artifact),
            'referenceFile':row['file'],'referenceSha256':row['sha256'],'reconstructionFile':str(raw.relative_to(ROOT)),
            'reconstructionSha256':sha(raw),'streamGain':stream_gain,'targetLufs':target,'choices':choices,
            'waveformSnrDb':float(10*np.log10(np.sum(original**2)/np.sum((original-decoded)**2)))})
    manifest={'scope':'Live output from precompiled captured performances; no free-text generation or input understanding',
        'model':MODEL,'modelSha256':MODEL_SHA,'sampleRate':RATE,'units':units,'phrases':phrases,
        'voiceDefaultChanged':False,'autoplay':False,'listeningAccepted':False,'newNeuralTraining':False,
        'sourceData':'The existing corrected-syllable panel and all nine earlier fresh natural-speech validation recordings.',
        'comparisonProcessing':'Constant whole-clip matched loudness; no limiter, denoising or pitch correction.',
        'complete':len(units)==18 and len(phrases)==9}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('Prepared',len(units),'syllables and',len(phrases),'speech pairs; median reconstruction SNR',np.median([p['waveformSnrDb'] for p in phrases]))


if __name__=='__main__':main()
