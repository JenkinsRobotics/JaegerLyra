import io,json,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
import expressive_voice as voice
import voice_requests as requests

class VoiceCancellationTests(unittest.TestCase):
 def setUp(self):requests.REQUESTS.clear()
 def test_cancel_before_request_arrives_does_no_inference(self):
  key='request-before-arrival';requests.cancel(key)
  with patch.object(voice,'available') as available:
   with self.assertRaises(requests.VoiceCancelled):voice.synthesize({'requestId':key,'text':'Hello'})
   available.assert_not_called()
 def test_cancelled_queued_request_does_not_wait_for_model(self):
  event=threading.Event();voice.LOCK.acquire();errors=[]
  def work():
   try:voice._synthesize({'text':'An uncached queued test fixture'},event)
   except Exception as e:errors.append(e)
  try:
   thread=threading.Thread(target=work);thread.start();event.set();thread.join(.4)
   self.assertFalse(thread.is_alive());self.assertIsInstance(errors[0],requests.VoiceCancelled)
  finally:voice.LOCK.release()
 def test_cancelling_one_request_does_not_cancel_another(self):
  first=requests.register('first-request-identifier');second=requests.register('second-request-identifier')
  requests.cancel('first-request-identifier');self.assertTrue(first.is_set());self.assertFalse(second.is_set())
 def test_active_request_ids_cannot_be_reused(self):
  requests.register('duplicate-request-identifier')
  with self.assertRaises(ValueError):requests.register('duplicate-request-identifier')
 def test_warmup_bypasses_cached_audio(self):
  with tempfile.TemporaryDirectory() as directory,patch.object(voice,'CACHE',Path(directory)),patch.object(voice,'available',return_value=False):
   data={'text':'Warm fixture','engine':'expressive'}
   key=voice.hashlib.sha256(json.dumps({'version':'english-3',**data},sort_keys=True).encode()).hexdigest()
   (Path(directory)/(key+'.json')).write_text('{"duration":1}')
   self.assertTrue(voice.synthesize(data)['cached'])
   with self.assertRaisesRegex(ValueError,'not installed'):voice._synthesize({**data,'_warm':True},threading.Event())
 def test_protocol_reads_multiple_messages_from_one_pipe_write(self):
  import os,subprocess,sys
  process=subprocess.Popen([sys.executable,'-c',"import os,time;os.write(1,b'{\"type\":\"audio\"}\\n{\"type\":\"done\"}\\n');time.sleep(2)"],stdout=subprocess.PIPE)
  try:
   with patch.object(voice,'PROCESS',process),patch.object(voice,'BUFFER',bytearray()):
    state={'deadline':time.monotonic()+1};event=threading.Event()
    self.assertEqual(voice.read_worker(event,state)['type'],'audio')
    self.assertEqual(voice.read_worker(event,state)['type'],'done')
  finally:process.terminate();process.wait();process.stdout.close()
 def test_closing_stream_drains_old_audio_before_releasing_worker(self):
  process=MagicMock();process.poll.return_value=None
  replies=[{'type':'audio','index':0},{'type':'audio','index':1},{'type':'done','cancelled':True}]
  with patch.object(voice,'PROCESS',process),patch.object(voice,'start_worker'),patch.object(voice,'send_worker'),patch.object(voice,'with_frames',side_effect=lambda x:x),patch.object(voice,'read_worker',side_effect=replies) as read,patch.object(voice,'close') as close:
   stream=voice.stream({'engine':'clone','text':'fixture','requestId':'stream-draining-fixture'})
   self.assertEqual(next(stream)['index'],0);stream.close()
   self.assertEqual(read.call_count,3);close.assert_not_called()
   self.assertTrue(voice.LOCK.acquire(blocking=False));voice.LOCK.release()
 def test_singing_preparation_keeps_a_confirmed_warm_speech_worker(self):
  process=MagicMock();process.poll.return_value=None;ready=threading.Event()
  reply={'type':'singing-ready','kokoroReady':True,'loadedEngines':['expressive']}
  with patch.object(voice,'PROCESS',process),patch.object(voice,'READY',ready),patch.object(voice,'send_worker') as send,patch.object(voice,'read_worker',return_value=reply),patch.object(voice,'close') as close:
   self.assertEqual(voice.prepare_for_singing(),reply)
   send.assert_called_once_with({'control':'prepare-singing'})
   self.assertTrue(ready.is_set());close.assert_not_called()
 def test_unexpected_control_reply_discards_worker_instead_of_misrouting_audio(self):
  process=MagicMock();process.poll.return_value=None
  with patch.object(voice,'PROCESS',process),patch.object(voice,'send_worker'),patch.object(voice,'read_worker',return_value={'audio':'stale'}),patch.object(voice,'close') as close:
   self.assertFalse(voice.prepare_for_singing());close.assert_called_once_with(timeout=.2)

if __name__=='__main__':unittest.main()
