import base64,hashlib,io,json,subprocess,sys,tempfile,unittest,wave
from pathlib import Path
from unittest.mock import patch
import expressive_voice,singing
class VoiceStudioTests(unittest.TestCase):
 def test_reference_requires_clear_duration_and_transcript(self):
  def audio(seconds):
   out=io.BytesIO()
   with wave.open(out,'wb') as f:f.setparams((1,2,24000,0,'NONE','not compressed'));f.writeframes(bytes(int(seconds*24000)*2))
   return base64.b64encode(out.getvalue()).decode()
  with tempfile.TemporaryDirectory() as folder,patch.object(expressive_voice,'REFERENCES',Path(folder)):
   with self.assertRaises(ValueError):expressive_voice.reference({'audio':audio(1),'text':'short'})
   with self.assertRaises(ValueError):expressive_voice.reference({'audio':audio(6),'text':''})
   result=expressive_voice.reference({'audio':audio(6),'text':'Hello there.'})
   self.assertEqual(result['duration'],6);self.assertEqual(result['referenceText'],'Hello there.')
   self.assertTrue((Path(folder)/result['reference']).exists())
 def test_song_validation_and_cached_job_never_launches_generator(self):
  with self.assertRaises(ValueError):singing.start({'lyrics':''})
  with self.assertRaises(ValueError):singing.status('../anything')
  with tempfile.TemporaryDirectory() as folder,patch.object(singing,'FOLDER',Path(folder)),patch.object(singing,'available',return_value=False):
   with self.assertRaisesRegex(ValueError,'not installed'):singing.start({'engine':'ace-step','lyrics':'A little glow'})
   request={'lyrics':'A little glow','caption':singing.DEFAULT_STYLE,'duration':15}
   key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()[:24]
   (Path(folder)/(key+'.audio.json')).write_text('{"duration":15}')
   with patch.object(singing.threading,'Thread') as worker:
    self.assertEqual(singing.start({'engine':'ace-step','lyrics':'A little glow'}),{'id':key,'status':'ready'})
    self.assertEqual(singing.status(key)['result']['duration'],15)
    worker.assert_not_called()
 def test_stop_reaps_a_worker_that_ignores_termination(self):
  process=subprocess.Popen([sys.executable,'-u','-c',"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(30)"],stdout=subprocess.PIPE,text=True)
  try:
   self.assertEqual(process.stdout.readline().strip(),'ready')
   with patch.object(singing,'JOBS',{'test':{'status':'generating','process':process}}):
    singing.cancel('test')
    self.assertEqual(singing.JOBS['test']['status'],'cancelled')
    self.assertIsNotNone(process.poll())
  finally:
   if process.poll() is None:process.kill();process.wait()
   process.stdout.close()
