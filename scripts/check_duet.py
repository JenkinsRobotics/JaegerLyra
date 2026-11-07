"""Silent checks of prepared duet pitch, stems, local serving and provenance."""
import argparse,hashlib,io,json
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np
import parselmouth
import soundfile as sf


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--url',default='http://127.0.0.1:8794')
    args=parser.parse_args();root=args.folder/'output';manifest=json.loads((root/'manifest.json').read_text());rows=[]
    for row in manifest['phrases']:
        audio,rate=sf.read(root/(row['id']+'.wav'))
        pitch=parselmouth.Sound(audio,sampling_frequency=rate).to_pitch_ac(time_step=.005,pitch_floor=70,pitch_ceiling=700)
        t=pitch.xs();hz=pitch.selected_array['frequency'];notes=[]
        for note in row['notes']:
            interval=(t>=note['vowel_start_s']+.04)&(t<note['vowel_end_s']-.04)
            voiced=interval&(hz>0);cents=1200*np.log2(hz[voiced]/note['hz'])
            assert len(cents),f"No measured vowel pitch: {row['id']} / {note['midi']}"
            item={'midi':note['midi'],'frames':len(cents),'voiced_fraction':float(np.sum(voiced)/max(1,np.sum(interval))),
                'median_abs_cents':float(np.median(abs(cents))),'p90_abs_cents':float(np.percentile(abs(cents),90))}
            assert item['median_abs_cents']<50.,(row['id'],item)
            notes.append(item)
        rows.append({'id':row['id'],'notes':notes})
    lead,rate=sf.read(root/'lead.wav');reply,_=sf.read(root/'reply.wav');mix,_=sf.read(root/'duet.wav')
    error=float(abs(mix-lead-reply).max());peak=float(abs(mix).max())
    assert error<3e-7 and peak<.9 and np.isfinite(mix).all()
    assert not manifest['original_song_audio_in_output'] and not manifest['backing_track']
    assert not manifest['kokoro_imported_during_render'] and not manifest['new_training']
    with urlopen(args.url+'/api/duet',timeout=10) as r:served=json.load(r)
    assert served['available'] and served['duration_s']==manifest['duration_s']
    for item in manifest['files']:
        with urlopen(args.url+'/duet-audio/'+item['file'],timeout=10) as r:data=r.read()
        assert hashlib.sha256(data).hexdigest()==item['sha256']
        decoded,rate=sf.read(io.BytesIO(data));assert decoded.shape==mix.shape
        for requested,expected in [('bytes=1024-2047',data[1024:2048]),('bytes=-64',data[-64:])]:
            with urlopen(Request(args.url+'/duet-audio/'+item['file'],headers={'Range':requested}),timeout=10) as r:
                assert r.status==206 and r.headers['Accept-Ranges']=='bytes' and r.read()==expected
    try:urlopen(Request(args.url+'/duet-audio/duet.wav',headers={'Range':'bytes=999999999-'}),timeout=10)
    except HTTPError as exc:assert exc.code==416
    else:raise AssertionError('Out-of-bounds audio range accepted')
    for path in ('/duet-audio/manifest.json','/duet-audio/../manifest.json','/duet-audio/lead-1.wav'):
        try:urlopen(args.url+path,timeout=10)
        except HTTPError as exc:assert exc.code==404
        else:raise AssertionError('Unexpected media route')
    note_count=sum(len(r['notes']) for r in rows)
    report={'complete':True,'phrase_pitch':rows,'notes':note_count,'stem_sum_max_error':error,'peak':peak,
        'seconds':len(mix)/rate,'audio_played':False,'listening_quality_accepted':False,
        'method':'Praat autocorrelation, 70–700 Hz, vowel interiors excluding 40 ms at either edge; no octave correction. Note control, not a human naturalness judgment.',
        'http_hashes_passed':3,'unexpected_media_routes_rejected':3}
    (root/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('phrase_pitch','method')},indent=2))


if __name__=='__main__':main()
