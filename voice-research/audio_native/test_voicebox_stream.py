import unittest
import numpy as np
from continuous_source import ContinuousSource
from voicebox_stream import ReferenceVoicebox


def model():
    return ContinuousSource(np.r_[np.linspace(1,.1,64),np.zeros(64)],np.eye(128)[:48])


class VoiceboxStreamTests(unittest.TestCase):
    def test_transport_chunks_preserve_complete_output_and_reference_weights(self):
        data=np.random.default_rng(11).normal(0,.01,2401)
        one,two=ReferenceVoicebox(model()),ReferenceVoicebox(model())
        original=one.model.basis.copy()
        a=one.push(data)+one.finish()
        b=[]
        for at in range(0,len(data),127):b+=two.push(data[at:at+127])
        b+=two.finish()
        np.testing.assert_allclose(np.concatenate([p.samples for p in a]),np.concatenate([p.samples for p in b]),atol=1e-12,rtol=0)
        self.assertEqual(sum(len(p.samples) for p in a),len(data))
        np.testing.assert_array_equal(original,one.model.basis)

    def test_first_packet_waits_for_600_samples_then_emits_240(self):
        for shaped in (False,True):
            stream=ReferenceVoicebox(model(),shape_noise=shaped)
            self.assertEqual(stream.push(np.zeros(599)),[])
            packets=stream.push([0.])
            self.assertEqual(len(packets),1)
            self.assertEqual((packets[0].start,packets[0].available_at,len(packets[0].samples)),(0,600,240))
            self.assertLess(len(stream.analysis.pending),600)

    def test_cancel_discards_pending_audio_without_a_final_flush(self):
        stream=ReferenceVoicebox(model())
        self.assertEqual(stream.push(np.ones(500)*.01),[])
        stream.cancel()
        self.assertEqual(stream.finish(),[])
        self.assertEqual(len(stream.analysis.pending),0)
        with self.assertRaises(ValueError):stream.push([0.])

    def test_invalid_reference_is_rejected(self):
        with self.assertRaises(ValueError):ReferenceVoicebox.from_reference(np.zeros(100))

    def test_active_stream_owns_a_frozen_reference_snapshot(self):
        source=model();stream=ReferenceVoicebox(source)
        original=stream.model.mean.copy()
        source.mean[:]=0
        np.testing.assert_array_equal(stream.model.mean,original)
        with self.assertRaises(ValueError):stream.model.basis[0,0]=3

    def test_colored_noise_is_chunk_consistent_with_frozen_reference(self):
        data=np.random.default_rng(103).normal(0,.01,2401)
        one,two=ReferenceVoicebox(model(),shape_noise=True),ReferenceVoicebox(model(),shape_noise=True)
        before=one.model.basis.copy()
        a=one.push(data)+one.finish();b=[]
        for at in range(0,len(data),73):b+=two.push(data[at:at+73])
        b+=two.finish()
        np.testing.assert_allclose(np.concatenate([p.samples for p in a]),np.concatenate([p.samples for p in b]),atol=1e-12,rtol=0)
        np.testing.assert_array_equal(before,one.model.basis)
        self.assertEqual(len(two.synthesis.noise_shaper.history),63)

    def test_cancel_clears_colored_noise_state_and_rejects_new_audio(self):
        stream=ReferenceVoicebox(model(),shape_noise=True)
        stream.push(np.random.default_rng(42).normal(0,.01,1200))
        self.assertTrue(np.any(stream.synthesis.noise_shaper.history))
        stream.cancel()
        np.testing.assert_array_equal(stream.synthesis.noise_shaper.history,np.zeros(63))
        self.assertEqual(stream.finish(),[])
        with self.assertRaises(ValueError):stream.push(np.zeros(240))


if __name__=='__main__':unittest.main()
