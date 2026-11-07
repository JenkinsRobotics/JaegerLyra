"""Encode prepared natural speech, then render a local two-part Lyra score.

No Kokoro model, original song recording or reference WAV is loaded in the
render phase. The separate encode phase reads synthetic speech fixtures.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'voice-research/audio_native'))
from neural_waveform import load_model,RATE
from compiled_neural_decoder import compile_decoder
from neural_performance import EncodedPerformance,save_performance,load_performance
from scored_resynthesis import render_score

VOWELS=set('ɑɐɒæəɛɜɚɝɪiɔʊuʌʉɨɯɶœøyᵊAEIOUYaeo')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan(row,tempo,pulses,audio):
    chars=['']+list(row['phonemes'])+[''];native=np.asarray(row['duration_frames'])/40
    nuclei=[i for i,c in enumerate(chars) if c in VOWELS]
    notes=row['notes_midi']
    if len(nuclei)!=len(notes):raise ValueError(f"{row['id']}: {len(nuclei)} vowels for {len(notes)} notes")
    # Keep consonants short. Allocate most of the scored duration to vowels.
    target=np.clip(native,.015,.085);target[0]=.045;target[-1]=.10
    for i,c in enumerate(chars):
        if c and c in ' ˈˌ.,!?;:':target[i]=.008
    length=row['duration_beats']*60/tempo
    boundaries=[1]+[(a+b)//2+1 for a,b in zip(nuclei,nuclei[1:])]+[len(chars)-1]
    weights=np.ones(len(nuclei));weights[-1]=1.9
    spans=(length-target[0]-target[-1])*weights/weights.sum()
    for j,nucleus in enumerate(nuclei):
        a,b=boundaries[j:j+2]
        consonants=sum(target[a:b])-target[nucleus]
        if consonants>spans[j]*.60:
            factor=spans[j]*.60/consonants
            for k in range(a,b):
                if k!=nucleus:target[k]*=factor
        target[nucleus]=spans[j]-(sum(target[a:b])-target[nucleus])
    # Predicted duration frames sometimes differ slightly from audio length.
    source=np.r_[0,np.cumsum(native)];source*=row['sample_count']/RATE/source[-1]
    # The installed Kokoro pipeline's join_timestamps subtracts three 40 Hz
    # frames from its first duration. Account for that synthesis alignment
    # offset before stretching vowels; otherwise plosive gaps become holds.
    source[1:-1]-=.075
    if np.any(np.diff(source)<=0):raise ValueError('Unsupported short initial pronunciation alignment')
    output=np.r_[0,np.cumsum(target)]
    pitch_t=[0.];pitch=[440*2**((notes[0]-69)/12)];events=[]
    for j,(nucleus,midi) in enumerate(zip(nuclei,notes)):
        when=float(output[boundaries[j]])
        hz=float(440*2**((midi-69)/12))
        if j:
            pitch_t.extend([max(pitch_t[-1]+.001,when-.025),when+.035])
            pitch.extend([pitch[-1],hz])
        events.append({'start_s':float(output[boundaries[j]]),'end_s':float(output[boundaries[j+1]]),
            'vowel_start_s':float(output[nucleus]),'vowel_end_s':float(output[nucleus+1]),'midi':midi,'hz':hz})
    pitch_t.append(float(output[-1]));pitch.append(pitch[-1])
    # Predicted phone boundaries are approximate. Anchor long sung vowels in
    # a nearby actually voiced region instead of stretching a breath/plosive
    # gap or the final vocal fry. Retain all source audio between anchors.
    centers=np.array([(source[n]+source[n+1])/2 for n in nuclei])
    pulse_s=pulses/RATE;source_map=[0.];output_map=[0.];anchors=[]
    for j,(nucleus,event) in enumerate(zip(nuclei,events)):
        lower=(centers[j-1]+centers[j])/2 if j else .04
        upper=(centers[j]+centers[j+1])/2 if j+1<len(centers) else source[-1]-.04
        candidate=pulse_s[(pulse_s>=lower)&(pulse_s<=upper)];choices=[]
        for group in np.split(candidate,np.flatnonzero(np.diff(candidate)>.018)+1):
            for first in range(len(group)-3):
                last=min(first+8,len(group)-1)
                a,b=group[first],group[last]
                periods=np.diff(group[first:last+1]);rms=float(np.sqrt(np.mean(audio[round(a*RATE):round(b*RATE)]**2)))
                if b-a<.022 or rms<.008:continue
                cost=abs((a+b)/2-centers[j])+.08*float(np.std(periods)/np.mean(periods))-.12*(b-a)
                choices.append((cost,a,b))
        if not choices:raise ValueError(f"{row['id']}: no voiced nucleus for note {j+1}")
        _,a,b=min(choices)
        source_map.extend([float(a),float(b)]);output_map.extend([event['vowel_start_s'],event['vowel_end_s']])
        anchors.append({'source_start_s':float(a),'source_end_s':float(b)})
    source_map.append(float(source[-1]));output_map.append(float(output[-1]))
    return {'source_s':source_map,'output_s':output_map,'pitch_time_s':pitch_t,'pitch_hz':pitch,'notes':events,'voiced_anchors':anchors}


def encode(folder,model):
    import parselmouth
    from parselmouth.praat import call
    manifest=json.loads((folder/'speech/manifest.json').read_text())
    score=json.loads((folder/'score.json').read_text())
    if not manifest['complete'] or manifest['score_sha256']!=sha(folder/'score.json'):
        raise ValueError('Incomplete or changed pronunciation preparation')
    encoder,_,_=load_model(model);out=folder/'encoded';out.mkdir(exist_ok=True)
    rows=[]
    for row in manifest['phrases']:
        path=folder/row['file']
        if sha(path)!=row['sha256']:raise ValueError('Changed synthetic speech reference')
        audio,rate=sf.read(path);assert rate==RATE
        # Constrain octave ambiguity using the prepared synthetic voice range.
        # This is source preparation assistance, not general voice analysis.
        floor,ceiling=(130.,450.) if row['part']=='lead' else (65.,280.)
        sound=parselmouth.Sound(audio,sampling_frequency=RATE)
        points=call(sound,'To PointProcess (periodic, cc)',floor,ceiling)
        pulses=np.array([call(points,'Get time from index',i+1)*RATE for i in range(int(call(points,'Get number of points')))])
        pulses=pulses[(pulses>=0)&(pulses<len(audio))]
        performance=EncodedPerformance(encoder.encode(audio),pulses)
        controls=plan(row,score['tempo_bpm'],pulses,audio);artifact=out/(row['id']+'.npz')
        save_performance(artifact,performance,{'origin':'Encoded Kokoro natural speech','text':row['text'],'source_sha256':row['sha256']})
        rows.append({**row,'controls':controls,'artifact':str(artifact.relative_to(folder)),'artifact_sha256':sha(artifact)})
        print('Encoded',row['id'],flush=True)
    (out/'manifest.json').write_text(json.dumps({'phrases':rows,'model_sha256':sha(model),'score_sha256':sha(folder/'score.json'),'complete':True},indent=2)+'\n')


def render(folder,model):
    start=time.perf_counter();score=json.loads((folder/'score.json').read_text())
    manifest=json.loads((folder/'encoded/manifest.json').read_text())
    if not manifest['complete'] or manifest['model_sha256']!=sha(model) or manifest['score_sha256']!=sha(folder/'score.json'):
        raise ValueError('Changed prepared score or decoder')
    _,decoder,_=load_model(model);decoder=compile_decoder(decoder)
    seconds=max((r['start_beat']+r['duration_beats'])*60/score['tempo_bpm'] for r in manifest['phrases'])+.4
    count=round(seconds*RATE);stems={p:np.zeros(count) for p in ('lead','reply')};rows=[]
    out=folder/'output';out.mkdir(exist_ok=True)
    for row in manifest['phrases']:
        path=folder/row['artifact']
        if sha(path)!=row['artifact_sha256']:raise ValueError('Changed encoded articulation')
        performance,_=load_performance(path);controls=row['controls']
        audio,metrics=render_score(performance,decoder,**{k:controls[k] for k in ('source_s','output_s','pitch_time_s','pitch_hz')},
            vibrato_cents=17. if row['part']=='lead' else 9.)
        # One fixed gain per phrase, with short edge fades. No reverb, doubling,
        # backing track, denoising, saturation, or original-song PCM is mixed in.
        rms=float(np.sqrt(np.mean(audio**2)));peak=float(max(abs(audio)))
        gain=min(.105/max(rms,1e-6),.68/max(peak,1e-6))
        audio*=gain;edge=min(192,len(audio)//2)
        audio[:edge]*=np.sin(np.linspace(0,np.pi/2,edge))**2
        audio[-edge:]*=np.cos(np.linspace(0,np.pi/2,edge))**2
        offset=round(row['start_beat']*60/score['tempo_bpm']*RATE)
        stems[row['part']][offset:offset+len(audio)]+=audio
        file=out/(row['id']+'.wav');sf.write(file,audio,RATE,subtype='PCM_24')
        rows.append({'id':row['id'],'part':row['part'],'text':row['text'],'start_s':offset/RATE,
            'end_s':(offset+len(audio))/RATE,'notes':controls['notes'],'gain':gain,'peak':float(max(abs(audio))),**metrics})
        print('Rendered',row['id'],len(audio)/RATE,'seconds',flush=True)
    # Slightly separated dry voices; both remain audible on mono speakers.
    lead=np.column_stack((stems['lead']*.82,stems['lead']*.57))
    reply=np.column_stack((stems['reply']*.57,stems['reply']*.82))
    mix=lead+reply;master=min(1.,.89/max(float(abs(mix).max()),1e-6))
    lead*=master;reply*=master;mix=lead+reply
    files=[]
    for name,audio in [('duet',mix),('lead',lead),('reply',reply)]:
        path=out/(name+'.wav');sf.write(path,audio,RATE,subtype='PCM_24')
        files.append({'id':name,'file':path.name,'sha256':sha(path),'peak':float(abs(audio).max()),'channels':2})
    forbidden=[n for n in sys.modules if n.split('.')[0] in ('mlx','mlx_audio','kokoro','torch','transformers')]
    assert not forbidden,forbidden
    report={'complete':True,'title':score['title'],'attribution':score['attribution'],'arrangement':score['arrangement'],
        'tempo_bpm':score['tempo_bpm'],'duration_s':count/RATE,'sample_rate_hz':RATE,'voices':['Heart','Michael'],
        'scope':'Prepared hybrid text-to-singing study: Kokoro natural speech preparation → encoded controls → Lyra decoder + authored pitch/timing → dry stereo duet.',
        'original_song_audio_in_output':False,'exact_original_melody':False,'backing_track':False,'reverb':False,
        'kokoro_imported_during_render':False,'new_training':False,'listening_accepted':False,
        'phrases':rows,'files':files,'render_compute_s':time.perf_counter()-start}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Complete:',count/RATE,'seconds;',round(report['render_compute_s'],2),'seconds compute',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['encode','render']);parser.add_argument('folder',type=Path)
    args=parser.parse_args();bundle=json.loads((ROOT/'demo/bundle/manifest.json').read_text());model=ROOT/bundle['model']
    if sha(model)!=bundle['modelSha256']:raise ValueError('Changed decoder')
    (encode if args.stage=='encode' else render)(args.folder.resolve(),model)


if __name__=='__main__':main()
