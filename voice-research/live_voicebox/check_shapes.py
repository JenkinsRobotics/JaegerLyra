"""Silent full-catalogue control sweep and live HTTP integration checks."""
import json
from urllib.request import Request,urlopen
from runtime import Engine,Session,ROOT,RATE
import numpy as np
import soundfile as sf

OUT=ROOT/'evidence/lyra-verification/shapes'
BASE='http://127.0.0.1:8794'
SHAPES={
    'natural':{},'resonance-low':{'resonance':.85},'resonance-high':{'resonance':1.18},
    'soft':{'brightness':-4.},'bright':{'brightness':4.},'slow':{'pace':.75},
    'fast':{'pace':1.35},'vibrato':{'vibrato':60.},
    'warm':{'resonance':.92,'brightness':-1.5,'pace':.94},
    'light':{'resonance':1.1,'brightness':1.3,'pace':1.05},
    'gentle-vibrato':{'pace':.93,'vibrato':25.}}


def api(path,body=None):
    request=Request(BASE+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=10) as response:return json.load(response)


def main():
    engine=Engine();report={'complete':False,'audioPlayed':False,'renders':[],'http':[]}
    for item in engine.manifest['words']:
        performance,metadata=engine.performances[('word',item['id'])]
        for name,shape in SHAPES.items():
            session=Session('word',item,performance,metadata,engine.decoder,shape=shape)
            chunks=[]
            while chunk:=session.next_chunk():chunks.append(np.frombuffer(chunk,dtype='<f4'))
            audio=np.concatenate(chunks);status=session.snapshot()
            assert status['completed'] and not status['holding'] and status['releaseSample'] is None
            assert len(audio)==round(item['sampleCount']/shape.get('pace',1.)) and np.isfinite(audio).all()
            assert max(abs(audio))<1., (item['id'],name,max(abs(audio)))
            row={'item':item['id'],'shape':name,'samples':len(audio),'peak':float(max(abs(audio))),
                'p95ComputeMs':status['packetP95Ms'],'peakSafetySamples':status['peakSafetySamples']}
            if name=='natural':
                expected,_=sf.read(ROOT/item['reconstructionFile']);expected*=.75*item['streamGain']
                row['maximumReconstructionError']=float(max(abs(audio-expected)))
                assert row['maximumReconstructionError']<1e-6
            report['renders'].append(row)
        print('Checked all shapes:',item['id'],flush=True)
    assert api('/api/catalog')['activeSessions']==0
    for speaker in ('Heart','Michael','Bella'):
        item=next(i for i in engine.manifest['words'] if i['voice']==speaker and i['word']=='palm')
        shape={'resonance':1.1,'brightness':1.3,'pace':.8,'vibrato':30.}
        session=api('/api/session',{'mode':'word','item':item['id'],'shape':shape})
        with urlopen(BASE+'/api/stream?id='+session['id'],timeout=10) as response:audio=np.frombuffer(response.read(),dtype='<f4')
        state=api('/api/status?id='+session['id'])
        assert len(audio)==round(item['sampleCount']/.8) and state['completed'] and state['shape']==shape
        report['http'].append({'mode':'word','item':item['id'],'shape':shape,'samples':len(audio),'completed':True,'p95ComputeMs':state['packetP95Ms']})
    item=engine.manifest['phrases'][0]
    shape=SHAPES['warm'];session=api('/api/session',{'mode':'speech','item':item['id'],'shape':shape})
    with urlopen(BASE+'/api/stream?id='+session['id'],timeout=15) as response:audio=np.frombuffer(response.read(),dtype='<f4')
    performance,_=engine.performances[('speech',item['id'])];state=api('/api/status?id='+session['id'])
    assert state['completed'] and len(audio)==round(performance.matrix.sample_count/.94) and np.isfinite(audio).all()
    report['http'].append({'mode':'speech','item':item['id'],'shape':shape,'samples':len(audio),'completed':True,'p95ComputeMs':state['packetP95Ms']})
    assert api('/api/catalog')['activeSessions']==0
    report['complete']=True;OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'control-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Passed',len(report['renders']),'offline renders and',len(report['http']),'HTTP streams. No audio played.',flush=True)


if __name__=='__main__':main()
