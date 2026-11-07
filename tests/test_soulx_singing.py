import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import singing
import soulx_singing


class SingingIntegrationTests(unittest.TestCase):
    def test_cancel_during_speech_memory_preparation_cannot_restart_the_job(self):
        job={'status':'queued','process':None}
        def prepare():
            job['status']='cancelled'
            return {'type':'singing-ready','kokoroReady':True}
        with patch.object(singing,'JOBS',{'test':job}),patch.object(singing.expressive_voice,'prepare_for_singing',side_effect=prepare),patch.object(singing.expressive_voice,'warm'),patch.object(singing,'run_stage') as stage:
            singing.run('test',{'engine':'soulx'},job)
            self.assertEqual(job['status'],'cancelled')
            stage.assert_not_called()
    def test_worker_progress_only_overrides_active_job_and_invalid_files_are_ignored(self):
        key='a'*24
        job={'status':'generating','phase':'Loading','process':None}
        with tempfile.TemporaryDirectory() as folder, patch.object(singing,'FOLDER',Path(folder)),patch.object(singing,'JOBS',{key:job}):
            path=Path(folder)/(key+'.progress.json')
            path.write_text('{"phase":"Singing the lyrics…"}')
            self.assertEqual(singing.status(key)['phase'],'Singing the lyrics…')
            for invalid in ['{','[]']:
                path.write_text(invalid)
                self.assertEqual(singing.status(key)['phase'],'Loading')
            path.write_text('{"phase":"Singing the lyrics…"}')
            job['status']='cancelled'
            self.assertEqual(singing.status(key)['phase'],'Loading')
    def test_reference_and_pitch_choices_have_distinct_cache_identity(self):
        base = {'lyrics': 'Little lights are shining', 'engine': 'soulx'}
        requests = [soulx_singing.validate_request({**base, **extra}) for extra in
                    [{}, {'speaker': 'am_michael'}, {'referenceKind': 'sung'},
                     {'transpose': -12}, {'tempo': 88}, {'melody': 'lullaby'}]]
        keys = {hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() for r in requests}
        self.assertEqual(len(keys), len(requests))
        self.assertEqual(requests[0]['steps'], 32)
        self.assertEqual(requests[0]['cfg'], 3)

    def test_invalid_references_and_controls_fail_before_starting_a_process(self):
        for override in [{'speaker': '../elsewhere'}, {'referenceKind': '../speech'},
                         {'transpose': .5}, {'transpose': float('nan')},
                         {'tempo': float('inf')}, {'lyrics': '123 stars'}, {'lyrics': []}]:
            with self.subTest(override=override), patch.object(singing.threading, 'Thread') as worker:
                with self.assertRaises(ValueError):
                    singing.start({'engine': 'soulx', 'lyrics': 'Little lights', **override})
                worker.assert_not_called()

    def test_old_cancelled_job_does_not_modify_its_same_key_replacement(self):
        old = {'status': 'cancelled', 'process': None}
        replacement = {'status': 'queued', 'process': None}
        active = threading.Event();active.set()
        with patch.object(singing, 'JOBS', {'same': replacement}), \
             patch.object(singing.expressive_voice, 'MUSIC_ACTIVE', active), \
             patch.object(singing.expressive_voice, 'warm'), \
             patch.object(singing, 'run_stage') as stage:
            singing.run('same', {'engine': 'kokoro'}, old)
            self.assertEqual(replacement['status'], 'queued')
            self.assertTrue(active.is_set())
            stage.assert_not_called()

    def test_cancellation_during_audio_packaging_cannot_publish_a_ready_clip(self):
        job = {'status': 'queued', 'process': None}
        def cancelled_pack(*args):
            job['status'] = 'cancelled'
            return {'audio': 'late'}
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(singing, 'FOLDER', Path(folder)), \
             patch.object(singing, 'JOBS', {'test': job}), \
             patch.object(singing, 'run_stage', return_value=True), \
             patch.object(singing, 'pack_audio', side_effect=cancelled_pack), \
             patch.object(singing.expressive_voice, 'close'), \
             patch.object(singing.expressive_voice, 'warm'):
            (Path(folder)/'test.result.json').write_text(json.dumps({'path': 'unused.wav', 'score': []}))
            singing.run('test', {'engine': 'kokoro', 'speaker': 'af_heart'}, job)
            self.assertEqual(job['status'], 'cancelled')
            self.assertFalse((Path(folder)/'test.audio.json').exists())


if __name__ == '__main__':
    unittest.main()
