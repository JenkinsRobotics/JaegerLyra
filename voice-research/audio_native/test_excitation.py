import unittest
from dataclasses import fields,replace
import numpy as np
from excitation import GeneratedVoicebox,ReferenceExcitation,VocalControls
from streaming_voicebox import StreamingAnalysis


class ExcitationTests(unittest.TestCase):
    def test_decoder_controls_contain_no_pcm_residual(self):
        samples=np.sin(np.arange(2400)*.05)*.1
        frames=StreamingAnalysis().push(samples)
        controls=[VocalControls.from_frame(f) for f in frames]
        self.assertEqual([f.name for f in fields(VocalControls)],
                         ['start','samples','coefficients','f0','periodicity','excitation_rms'])
        # Destroy all original waveform detail after extracting controls.
        expected=np.concatenate([GeneratedVoicebox().push(controls[0])])
        for frame in frames:frame.residual[:]=np.nan
        np.testing.assert_array_equal(GeneratedVoicebox().push(controls[0]),expected)

    def test_generated_silence_stays_silent_and_finite(self):
        source=GeneratedVoicebox()
        for frame in StreamingAnalysis().push(np.zeros(2400)):
            np.testing.assert_array_equal(source.push(VocalControls.from_frame(frame)),np.zeros(240))

    def test_repeating_a_frame_cannot_advance_the_stream_twice(self):
        frame=StreamingAnalysis().push(np.zeros(240))[0]
        control=VocalControls.from_frame(frame)
        source=GeneratedVoicebox();source.push(control)
        with self.assertRaises(ValueError):source.push(control)

    def test_reference_fit_requires_actual_voiced_evidence(self):
        learner=ReferenceExcitation()
        for frame in StreamingAnalysis().push(np.zeros(2400)):learner.observe(frame)
        with self.assertRaises(ValueError):learner.profile()

    def test_pitch_control_preserves_duration_and_filter(self):
        control=VocalControls(0,240,np.r_[1.,np.zeros(20)],160.,.98,.03)
        original=control.coefficients.copy()
        one,two=GeneratedVoicebox(),GeneratedVoicebox()
        for i in range(20):
            frame=replace(control,start=i*240)
            a=one.push(frame);b=two.push(frame,pitch_ratio=2.)
            self.assertEqual(a.shape,b.shape)
        self.assertEqual(one.position,two.position)
        self.assertAlmostEqual(two.last_f0/one.last_f0,2.)
        np.testing.assert_array_equal(control.coefficients,original)

    def test_source_change_is_ramped_instead_of_an_instant_phase_flip(self):
        profile={'real':[1.]+[0.]*31,'imag':[0.]*32}
        one,two=GeneratedVoicebox(profile),GeneratedVoicebox(profile)
        noise=GeneratedVoicebox({'real':[0.]*32,'imag':[0.]*32})
        frame=VocalControls(0,240,np.r_[1.,np.zeros(20)],160.,.98,.1)
        for i in range(10):
            one.push(replace(frame,start=i*240));two.push(replace(frame,start=i*240));noise.push(replace(frame,start=i*240))
        unchanged=one.push(replace(frame,start=2400))
        changed=two.push(replace(frame,start=2400),harmonics=-np.asarray(profile['real']))
        noise_only=noise.push(replace(frame,start=2400))
        # Identical noise and filters cancel; first-sample source change must be
        # small even though the target harmonic phase flips by 180 degrees.
        self.assertLess(abs(changed[0]-unchanged[0]),.002)
        blend=(np.arange(240)+1)/240
        np.testing.assert_allclose(changed,unchanged-2*blend*(unchanged-noise_only),atol=1e-14)


if __name__=='__main__':unittest.main()
