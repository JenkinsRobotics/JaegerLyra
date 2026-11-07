"""Diagnostic only: oracle target envelope over the same captured excitation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import json
import time
import numpy as np
import soundfile as sf
from benchmark_resonance_morph import ROOT,OUT,MODEL,EXPECTED_MODEL,sha,save_json
from neural_waveform import load_model,RATE
from compiled_neural_decoder import compile_decoder
from neural_performance import load_performance
from live_neural_grains import LiveNeuralGrainStream,DELAY
from compact_vocal_instrument import VocalFrame,BLOCK
from resonance_morph import ResonancePath,MovingFilter


def main():
    config=json.loads((OUT/'config.json').read_text());capture=json.loads((OUT/'capture.json').read_text())
    render=json.loads((OUT/'render.json').read_text());assert render['complete'] and sha(MODEL)==EXPECTED_MODEL
    _,decoder,_=load_model(MODEL);decoder=compile_decoder(decoder)
    report={'scope':'Post-validation capacity diagnostic: full target spectral envelope supplied. Not held-out controllable generation, compact articulation or a quality promotion.',
        'referenceControls':'Target WORLD envelope, pitch, timing and RMS; retained development start excitation.',
        'results':[],'complete':False,'listeningAccepted':False}
    for reference in render['references']:
        unit=next(u for u in capture['units'] if u['speaker']==reference['speaker'] and u['group']==reference['group'])
        patch,_=load_performance(ROOT/unit['patches'][0]['file'])
        with np.load(ROOT/unit['profile']) as z:path=ResonancePath(z['log_power'],z['landmarks'])
        with np.load(ROOT/reference['controls']) as z:times=z['times'];pitch=z['pitch'];rms=z['rms'];power=z['power']
        stream=LiveNeuralGrainStream([patch],decoder)
        chunks=[stream.push(VocalFrame(float(hz),(1.,))) for hz in pitch]
        chunks.append(stream.finish());count=len(rms);source=np.concatenate(chunks)[DELAY:DELAY+count]
        filt=MovingFilter(path,config['filterTaps'],config['maximumFilterGainDb']);parts=[];timings=[]
        for i,at in enumerate(range(0,count,BLOCK)):
            begin=time.perf_counter();parts.append(filt.push_envelope(source[at:at+BLOCK],np.log(np.maximum(power[i],1e-20))))
            timings.append((time.perf_counter()-begin)*1000)
        filt.cancel();audio=np.concatenate(parts)*rms/.1
        fade=min(240,count//2);audio[:fade]*=np.linspace(0,1,fade);audio[-fade:]*=np.linspace(1,0,fade)
        assert np.isfinite(audio).all()
        file=OUT/(reference['name']+'-oracle-envelope.wav');sf.write(file,audio,RATE,subtype='FLOAT')
        report['results'].append({'name':reference['name'],'speaker':reference['speaker'],'group':reference['group'],
            'method':'oracle-envelope','file':str(file.relative_to(ROOT)),'sha256':sha(file),'rawPeak':float(max(abs(audio))),
            'overFullScaleSamples':int(sum(abs(audio)>=1)),'filterOnlyP95Ms':float(np.percentile(timings,95)),
            'filterOnlyMaximumMs':max(timings),'offlineDiagnostic':True})
        save_json('oracle-envelope.json',report);print(reference['name'],flush=True)
    report['complete']=True;save_json('oracle-envelope.json',report)


if __name__=='__main__':main()
