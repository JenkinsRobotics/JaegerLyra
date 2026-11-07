"""Verify complete words over HTTP without playback or release commands."""
import json
import time
from urllib.request import Request,urlopen
import numpy as np
import soundfile as sf
from runtime import Engine,ROOT,RATE

BASE='http://127.0.0.1:8794'
OUT=ROOT/'evidence/lyra-verification/whole-words'


def api(path,body=None):
    request=Request(BASE+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=10) as response:return json.load(response)


def main():
    engine=Engine();catalog=api('/api/catalog');assert catalog['defaultPlayback']=='word' and catalog['activeSessions']==0
    report={'scope':'Automatic complete-word playback and explicit sustain, silently checked over HTTP',
        'words':[],'shiftedWords':[],'audioPlayed':False,'complete':False}
    for item in engine.manifest['words']:
        start=time.perf_counter();session=api('/api/session',{'mode':'word','item':item['id'],'gain':.75})
        with urlopen(BASE+'/api/stream?id='+session['id'],timeout=5) as response:raw=response.read()
        elapsed=time.perf_counter()-start;audio=np.frombuffer(raw,dtype='<f4')
        expected,rate=sf.read(ROOT/item['reconstructionFile']);expected*=item['streamGain']*.75
        assert rate==RATE and len(audio)==item['sampleCount'] and np.isfinite(audio).all()
        error=float(max(abs(audio-expected)));assert error<1e-6
        status=api('/api/status?id='+session['id'])
        assert status['completed'] and status['closed'] and not status['releaseRequested']
        assert status['releaseSample'] is None and status['automaticFinish'] and not status['holding']
        assert elapsed<item['seconds']+1.5
        report['words'].append({'id':item['id'],'samples':len(audio),'sourceSeconds':item['seconds'],
            'httpElapsedSeconds':elapsed,'maximumDecoderError':error,'automaticFinish':True,'releaseCommandSent':False})
        print('Auto-finished',item['id'],round(elapsed,3),flush=True)
    for speaker in ('Heart','Michael','Bella'):
        item=next(x for x in engine.manifest['words'] if x['voice']==speaker and x['word']=='palm')
        session=api('/api/session',{'mode':'word','item':item['id'],'hz':item['baseHz']*2**(.25),'gain':.75})
        with urlopen(BASE+'/api/stream?id='+session['id'],timeout=5) as response:audio=np.frombuffer(response.read(),dtype='<f4')
        status=api('/api/status?id='+session['id'])
        assert status['completed'] and len(audio)==item['sampleCount'] and np.isfinite(audio).all()
        assert abs(status['pitchSemitones']-3)<1e-8
        report['shiftedWords'].append({'id':item['id'],'samples':len(audio),'pitchSemitones':3,'automaticFinish':True,'peak':float(max(abs(audio)))})
    session=api('/api/session',{'mode':'syllable','item':'af_heart-palm'})
    with urlopen(BASE+'/api/stream?id='+session['id'],timeout=5) as response:
        first=response.read(RATE*4*2)
        state=api('/api/status?id='+session['id'])
        assert len(first)==RATE*4*2 and state['holding'] and not state['closed'] and not state['automaticFinish']
        api('/api/control',{'id':session['id'],'sequence':1,'release':True})
        response.read()
    state=api('/api/status?id='+session['id']);assert state['completed']
    report['explicitHold']={'heldUntilRelease':True,'finishedAfterRelease':True}
    assert api('/api/catalog')['activeSessions']==0
    report['complete']=True;OUT.mkdir(parents=True,exist_ok=True);(OUT/'http-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print('All whole-word and explicit-hold checks passed. No audio played.',flush=True)


if __name__=='__main__':main()
