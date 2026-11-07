import unittest
import numpy as np
from runtime import Session,Engine,MANIFEST,ROOT,RATE,BLOCK
from test_neural_articulation import fixture
from neural_syllable_stream import NeuralSyllableStream
from compact_vocal_instrument import VocalFrame
from neural_performance import PerformanceControls,pulse_resynthesis


def make(mode='syllable',gain=.7):
    unit,decoder,source=fixture()
    metadata={'holdFirstPulse':unit.hold_first,'holdLastPulse':unit.hold_last,'releaseSample':round(.55*RATE)}
    item={'id':'test','baseHz':200.,'streamGain':.8}
    return Session(mode,item,unit.performance,metadata,decoder,gain=gain),unit,decoder,source


def collect(session):
    parts=[]
    for _ in range(1000):
        packet=session.next_chunk()
        if not packet:return np.concatenate(parts) if parts else np.empty(0)
        parts.append(np.frombuffer(packet,dtype='<f4'))
    raise AssertionError('Session failed to finish')


class RuntimeTests(unittest.TestCase):
    def test_pace_and_vibrato_match_reference_and_finish(self):
        _,unit,decoder,source=make()
        for mode,pace,vibrato in (('word',.8,0),('word',1.25,30),('speech',.9,45)):
            session=Session(mode,{'baseHz':200.,'streamGain':.8},unit.performance,{},decoder,gain=.7,
                shape={'pace':pace,'vibrato':vibrato})
            audio=collect(session)
            expected,_=pulse_resynthesis(source,unit.performance.pulses,PerformanceControls(duration_scale=1/pace,vibrato_cents=vibrato))
            np.testing.assert_allclose(audio,expected*.8*.7,atol=9e-9)
            self.assertEqual(len(audio),round(len(source)/pace));self.assertTrue(session.completed)
            self.assertFalse(session.engine.reader.cache)

    def test_coloured_words_finish_and_stop_discards_cache(self):
        _,unit,decoder,source=make()
        for mode in ('word','speech'):
            session=Session(mode,{'baseHz':200.,'streamGain':.8},unit.performance,{},decoder,
                shape={'resonance':.85,'brightness':4.})
            audio=collect(session)
            self.assertEqual(len(audio),len(source));self.assertTrue(session.completed)
            self.assertTrue(np.isfinite(audio).all());self.assertLess(max(abs(audio)),.98)
            self.assertGreater(np.max(abs(audio-source*.8*.75)),1e-4)
        session=Session('word',{'baseHz':200.,'streamGain':.8},unit.performance,{},decoder,
            shape={'pace':.8,'resonance':1.18,'vibrato':60.})
        session.next_chunk();session.stop()
        self.assertEqual(session.next_chunk(),b'');self.assertFalse(session.engine.reader.cache)

    def test_shape_validation_and_neutral_identity(self):
        _,unit,decoder,source=make()
        for shape in ({'pace':0},{'resonance':float('nan')},{'brightness':True},{'vibrato':61},{'clone':1},[]):
            with self.assertRaises(ValueError):Session('word',{'baseHz':200.},unit.performance,{},decoder,shape=shape)
        a=Session('word',{'baseHz':200.,'streamGain':.8},unit.performance,{},decoder)
        b=Session('word',{'baseHz':200.,'streamGain':.8},unit.performance,{},decoder,
            shape={'resonance':1.,'brightness':0.,'pace':1.,'vibrato':0.})
        np.testing.assert_array_equal(collect(a),collect(b))

    def test_whole_word_finishes_without_release_and_preserves_every_sample(self):
        session,unit,decoder,source=make('word',1.)
        audio=collect(session)
        np.testing.assert_allclose(audio,source*.8,atol=8e-9)
        self.assertEqual(len(audio),len(source));self.assertTrue(session.completed)
        self.assertFalse(session.release_requested);self.assertIsNone(session.release_at)
        self.assertFalse(session.engine.cache);self.assertTrue(session.snapshot()['automaticFinish'])
        self.assertFalse(session.snapshot()['holding'])

    def test_word_pitch_changes_frequency_without_stretching_duration(self):
        _,unit,decoder,source=make()
        for frequency in (160.,250.):
            session=Session('word',{'baseHz':200.,'streamGain':1.},unit.performance,{},decoder,hz=frequency,gain=1.)
            audio=collect(session);self.assertEqual(len(audio),len(source));self.assertTrue(session.completed)
            np.testing.assert_allclose(audio[:round(.10*RATE)],source[:round(.10*RATE)],atol=8e-9)
            section=audio[round(.25*RATE):round(.60*RATE)]
            peak=np.argmax(abs(np.fft.rfft(section*np.hanning(len(section)))))
            measured=np.fft.rfftfreq(len(section),1/RATE)[peak]
            self.assertLess(abs(measured-frequency),4.)
            self.assertFalse(session.engine.reader.cache)

    def test_whole_word_controls_do_not_enter_sustain_and_stop_works(self):
        session,_,_,_=make('word')
        for change in ({'sequence':1,'release':True},{'sequence':1,'hz':250.}):
            with self.assertRaises(ValueError):session.control(change)
        session.control({'sequence':2,'gain':.5});session.next_chunk();session.stop()
        self.assertEqual(session.next_chunk(),b'');self.assertFalse(session.completed)
        self.assertFalse(session.engine.cache)

    def test_speech_is_decoded_in_order_and_ends_cleanly(self):
        session,unit,decoder,source=make('speech',1.)
        audio=collect(session);np.testing.assert_allclose(audio,source*.8,atol=8e-9)
        self.assertEqual(len(audio),len(source));self.assertTrue(session.completed);self.assertFalse(session.engine.cache)
        self.assertLessEqual(session.snapshot()['maximumCachedFrames'],16)

    def test_live_stream_matches_controller_for_commands_sent_later(self):
        session,unit,decoder,_=make();reference=NeuralSyllableStream(unit,decoder,.55*RATE)
        actual=[];expected=[];hz=200.;release=False
        for i in range(100):
            if i==30:hz=250.;session.control({'sequence':1,'hz':hz})
            if i==70:
                session.control({'sequence':2,'release':True});reference.release();release=True
            chunk=session.next_chunk()
            if not chunk:break
            actual.append(np.frombuffer(chunk,dtype='<f4'))
            for _ in range(4):
                if release and reference.position>=70*4*BLOCK+unit.performance.matrix.sample_count-.55*RATE:break
                expected.append(reference.push(VocalFrame(hz,(1.,),gain=.7)))
        expected.append(reference.finish());a=np.concatenate(actual);b=np.concatenate(expected)
        np.testing.assert_allclose(a,b,atol=8e-9);self.assertTrue(session.completed)
        self.assertEqual(session.events[0]['appliedSample'],30*4*BLOCK)
        self.assertEqual(session.events[1]['appliedSample'],70*4*BLOCK)

    def test_early_release_is_queued_and_stale_controls_do_not_override(self):
        session,unit,_,_=make();session.control({'sequence':3,'hz':250.,'release':True})
        session.control({'sequence':2,'hz':100.});self.assertEqual(session.hz,250.)
        audio=collect(session)
        self.assertGreaterEqual(session.release_at,unit.performance.pulses[unit.hold_first])
        self.assertLess(session.release_at,unit.performance.pulses[unit.hold_first]+BLOCK)
        self.assertTrue(session.completed);self.assertTrue(np.isfinite(audio).all())

    def test_invalid_control_is_atomic_and_stop_discards_pending_audio(self):
        session,_,_,_=make()
        for controls in [{'sequence':True},{'sequence':1,'hz':float('nan')},{'sequence':1,'gain':2},
                {'sequence':1,'release':'yes'},{'sequence':1,'unknown':1}]:
            with self.assertRaises(ValueError):session.control(controls)
        self.assertEqual(session.sequence,-1);self.assertEqual(session.gain,.7)
        session.next_chunk();session.stop();self.assertEqual(session.next_chunk(),b'');self.assertFalse(session.completed)
        self.assertFalse(session.engine.reader.cache)
        for array in (session.engine.pending,session.engine.weights,session.engine.background,session.engine.voicing,session.engine.gain):self.assertFalse(np.any(array))

    def test_speech_rejects_pitch_commands(self):
        session,_,_,_=make('speech')
        with self.assertRaises(ValueError):session.control({'sequence':1,'hz':250.})
        with self.assertRaises(ValueError):session.control({'sequence':1,'release':True})


if __name__=='__main__':unittest.main()
