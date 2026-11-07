"""Two pitch estimators, evaluated at the middle half of authored notes.

Estimator agreement is diagnostic; neither replaces listening to the vowels.
"""
import argparse,json
from pathlib import Path
import numpy as np
import soundfile as sf
import pyworld
import parselmouth
ROOT=Path(__file__).resolve().parents[1]
meta=json.loads((ROOT/'evidence/soulx/target.json').read_text())[0]
def score_notes(score):
 notes=[];start=0.
 for word,duration,midi in zip(score['text'].split(),map(float,score['duration'].split()),map(int,score['note_pitch'].split())):
  if midi:notes.append({'word':word,'start':start,'duration':duration,'midi':midi,'hz':440*2**((midi-69)/12)})
  start+=duration
 return notes
rows=[]
parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path);args=parser.parse_args()
folder=(ROOT/args.folder).resolve() if args.folder else ROOT/'evidence/soulx'
paths=list(folder.glob('*.wav'))
if not args.folder:paths+=list((ROOT/'evidence/openvoice').glob('*heart-transfer.wav'))
for path in paths:
 report=json.loads(path.with_suffix('.json').read_text()) if path.with_suffix('.json').exists() else {}
 notes=score_notes(report.get('score') or meta)
 audio,sr=sf.read(path);audio=audio.astype(np.float64)
 world,times=pyworld.harvest(audio,sr,f0_floor=70,f0_ceil=800,frame_period=10.)
 world=pyworld.stonemask(audio,world,times,sr)
 praat=parselmouth.Sound(audio,sr).to_pitch_ac(time_step=.01,pitch_floor=70,pitch_ceiling=800)
 tracks={'world':(times,world),'praat':(praat.xs(),praat.selected_array['frequency'])}
 result={'file':str(path.relative_to(ROOT)),'notes':[],'summary':{}}
 for note in notes:
  row=dict(note)
  for name,(t,f0) in tracks.items():
   segment=f0[(t>=note['start']+note['duration']*.25)&(t<note['start']+note['duration']*.75)]
   voiced=segment[segment>0];errors=1200*np.log2(voiced/note['hz'])
   row[name]={'voicedFraction':round(len(voiced)/max(1,len(segment)),3),'medianHz':round(float(np.median(voiced)),2) if len(voiced) else None,'medianAbsoluteCents':round(float(np.median(np.abs(errors))),1) if len(voiced) else None,'medianSignedCents':round(float(np.median(errors)),1) if len(voiced) else None}
  result['notes'].append(row)
 for name in tracks:
  values=[row[name]['medianAbsoluteCents'] for row in result['notes'] if row[name]['medianAbsoluteCents'] is not None]
  result['summary'][name]={'medianNoteErrorCents':round(float(np.median(values)),1),'measuredNotes':len(values)}
 rows.append(result);print(path.name,result['summary'],flush=True)
(folder/'pitch-evaluation.json').write_text(json.dumps({'method':'Middle 50% of each authored note; WORLD Harvest/StoneMask and Praat autocorrelation, 70–800 Hz, no octave correction. Not a human singing-quality rating.','results':rows},indent=2))
