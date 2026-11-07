"""Small deterministic LibriSpeech evaluation subset, with source attribution.

Not training data in this experiment; original synthetic PCA basis stays frozen.
Two public corpus readers, one different-text reference and two targets each.
Only selected FLACs/metadata are saved; tar members are never path-extracted.
"""
import hashlib,io,json,tarfile,urllib.request
from pathlib import Path
import soundfile as sf
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'voice-research/data/librispeech-dev-clean.tar.gz'
OUT=ROOT/'evidence/source-detail/human-inputs'
URL='https://openslr.trmal.net/resources/12/dev-clean.tar.gz'
MD5='42e2234ba48799c1f50f24a7926300a1'


def digest(path,kind):
    value=hashlib.new(kind)
    with path.open('rb') as stream:
        while block:=stream.read(1024*1024):value.update(block)
    return value.hexdigest()


def main():
    CACHE.parent.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    originals=OUT/'original';originals.mkdir(exist_ok=True)
    if not CACHE.exists():
        partial=CACHE.with_suffix('.partial')
        urllib.request.urlretrieve(URL,partial)
        if digest(partial,'md5')!=MD5:raise ValueError('Archive checksum differs from official manifest')
        partial.replace(CACHE)
    if digest(CACHE,'md5')!=MD5:raise ValueError('Archive checksum differs from official manifest')
    with tarfile.open(CACHE) as archive:
        metadata={name:archive.extractfile('LibriSpeech/'+name).read()
                  for name in ('LICENSE.TXT','README.TXT','SPEAKERS.TXT')}
    speakers=[]
    for line in metadata['SPEAKERS.TXT'].decode().splitlines():
        if not line or line.startswith(';'):continue
        fields=[x.strip() for x in line.split('|')]
        if len(fields)>=5 and fields[2]=='dev-clean':
            speakers.append({'id':int(fields[0]),'sexAsPublished':fields[1],'name':fields[4]})
    selected=[min((r for r in speakers if r['sexAsPublished']==label),key=lambda r:r['id'])
              for label in ('F','M')]
    ids={str(r['id']) for r in selected};clips={};transcripts={}
    # One sequential archive pass; avoid repeated seeks through gzip per clip.
    with tarfile.open(CACHE,mode='r|gz') as archive:
        for member in archive:
            parts=member.name.split('/')
            if not member.isfile() or len(parts)<5 or parts[2] not in ids:continue
            if member.name.endswith('.trans.txt'):
                for line in archive.extractfile(member).read().decode().splitlines():
                    utterance,text=line.split(' ',1);transcripts[utterance]=text
            elif member.name.endswith('.flac'):
                content=archive.extractfile(member).read();info=sf.info(io.BytesIO(content))
                if 3.<=info.duration<=9.:
                    utterance=Path(member.name).stem
                    clips[utterance]={'content':content,'member':member.name,'duration':info.duration}
    manifest={'purpose':'Prospective human-recording evaluation after selecting colored-noise candidate; no parameter retuning, human-reference fitting or new training',
        'origin':'LibriSpeech dev-clean, Vassil Panayotov et al., derived from public LibriVox audiobook readings',
        'sourcePage':'https://www.openslr.org/12/','archiveUrl':URL,'archiveMd5':MD5,
        'archiveSha256':digest(CACHE,'sha256'),'license':'CC BY 4.0',
        'licenseUrl':'https://creativecommons.org/licenses/by/4.0/',
        'selection':'Lowest reader ID in each published F/M group; first three utterance IDs with 3–9 s duration and at least six transcript words; first is reference, next two are targets. No selection by synthesis or recognition scores.',
        'processing':'Original mono 16 kHz FLAC resampled to 24 kHz with scipy resample_poly(3,2); PCM24 artifacts, no loudness normalization or denoising',
        'speakers':selected,'references':[],'clips':[]}
    for speaker in selected:
        prefix=str(speaker['id'])+'-'
        eligible=sorted(key for key in clips if key.startswith(prefix) and len(transcripts[key].split())>=6)
        if len(eligible)<3:raise ValueError('Selected reader lacks three eligible clips')
        reference_path=None
        for index,key in enumerate(eligible[:3]):
            item=clips[key];original=originals/(key+'.flac');original.write_bytes(item['content'])
            audio,rate=sf.read(io.BytesIO(item['content']));assert rate==16000 and audio.ndim==1
            converted=resample_poly(audio,3,2);path=OUT/(key+'.wav')
            sf.write(path,converted,24000,subtype='PCM_24')
            if index==0:reference_path=str(path.relative_to(ROOT))
            row={'name':key,'speaker':str(speaker['id']),'readerName':speaker['name'],
                'file':str(path.relative_to(ROOT)),'original':str(original.relative_to(ROOT)),
                'archiveMember':item['member'],'text':transcripts[key],
                'durationSeconds':item['duration'],'sha256':digest(path,'sha256'),
                'originalSha256':digest(original,'sha256')}
            if index==0:manifest['references'].append(row)
            else:row['reference']=reference_path;manifest['clips'].append(row)
    for name,content in metadata.items():(originals/name).write_bytes(content)
    (ROOT/'licenses/LibriSpeech-LICENSE.txt').write_bytes(metadata['LICENSE.TXT'])
    (ROOT/'licenses/librispeech-evaluation.json').write_text(json.dumps({k:v for k,v in manifest.items() if k not in ('clips','references')},indent=2)+'\n')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'speakers':selected,'references':[r['name'] for r in manifest['references']],
                      'targets':[r['name'] for r in manifest['clips']]}),flush=True)


if __name__=='__main__':main()
