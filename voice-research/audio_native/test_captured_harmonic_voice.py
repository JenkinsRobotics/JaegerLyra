import tempfile
import unittest
from pathlib import Path
import numpy as np
from compact_vocal_instrument import CompactVocalStream,VocalFrame,RATE
from captured_harmonic_voice import (fit_harmonics,save_bank,load_bank,CapturedHarmonicStream)
import test_compact_vocal_instrument as compact_tests


class CapturedHarmonicTests(unittest.TestCase):
    def fixture(self):
        phase=2*np.pi*np.arange(2400)/120
        audio=.08*np.cos(phase+.2)+.03*np.cos(2*phase+.8)+.01*np.cos(3*phase-.5)
        pulses=np.arange(0,2400,120,dtype=float)
        compact=compact_tests.CompactInstrumentTests().shape()
        return audio,pulses,compact

    def test_fit_independent_known_waveform_and_held_cycle_split(self):
        audio,pulses,compact=self.fixture()
        shape,report=fit_harmonics(audio,pulses,compact)
        np.testing.assert_allclose(np.exp(shape.log_amplitudes[:3]),[.08,.03,.01],atol=1e-12)
        self.assertGreater(report['fitCycleSnrDb'],200.)
        self.assertGreater(report['heldCycleSnrDb'],200.)
        odd=(np.floor(np.arange(len(audio))/120).astype(int)%2)==1
        changed=audio.copy();changed[odd]*=1.2
        other,diagnostic=fit_harmonics(changed,pulses,compact)
        np.testing.assert_allclose(other.log_amplitudes,shape.log_amplitudes,atol=1e-8)
        self.assertLess(diagnostic['heldCycleSnrDb'],20.)
        self.assertGreater(diagnostic['fitCycleSnrDb'],200.)

    def test_native_harmonic_ratios_and_relative_phase(self):
        shape,_=fit_harmonics(*self.fixture())
        stream=CapturedHarmonicStream([shape])
        audio=np.concatenate([stream.push(VocalFrame(200.,(1.,))) for _ in range(200)])
        spectrum=np.fft.rfft(audio)
        measured=spectrum[[200,400,600]]
        np.testing.assert_allclose(abs(measured)/abs(measured[0]),[1.,.375,.125],atol=1e-10)
        phase=np.angle(measured)-np.arange(1,4)*np.angle(measured[0])
        np.testing.assert_allclose(np.exp(1j*phase),np.exp(1j*np.array([0.,.4,-1.1])),atol=1e-10)

    def test_disabled_ablation_exact_and_bank_roundtrip(self):
        shape,_=fit_harmonics(*self.fixture())
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'bank.npz';save_bank(file,[shape],{'analytic':True})
            restored,meta=load_bank(file);self.assertTrue(meta['analytic'])
            a=CapturedHarmonicStream([shape]);b=CapturedHarmonicStream(restored)
            base=CompactVocalStream([shape.compact]);disabled=CapturedHarmonicStream(restored,magnitude=False,phase=False)
            for hz in (200.,240.,190.):
                frame=VocalFrame(hz,(1.,),breath=.2)
                np.testing.assert_array_equal(a.push(frame),b.push(frame))
                np.testing.assert_array_equal(base.push(frame),disabled.push(frame))
            self.assertFalse(restored[0].log_amplitudes.flags.writeable)
            with np.load(file,allow_pickle=False) as data:
                self.assertNotIn('pcm',data.files);self.assertNotIn('pulses',data.files)

    def test_invalid_frames_atomic_and_cancel_stops_output(self):
        shape,_=fit_harmonics(*self.fixture())
        a=CapturedHarmonicStream([shape]);b=CapturedHarmonicStream([shape])
        frame=VocalFrame(200.,(1.,),breath=.2)
        a.push(frame);b.push(frame)
        with self.assertRaises(ValueError):a.push(VocalFrame(200.,(float('nan'),)))
        np.testing.assert_array_equal(a.push(frame),b.push(frame))
        a.cancel();self.assertIsNone(a.previous_harmonics)
        self.assertFalse(np.any(a.noise_history))
        with self.assertRaises(RuntimeError):a.push(frame)

    def test_bad_pulse_tracks_rejected(self):
        audio,pulses,compact=self.fixture()
        for bad in (pulses[::-1],pulses+2400,pulses[:3],np.r_[pulses[:-1],float('nan')]):
            with self.assertRaises(ValueError):fit_harmonics(audio,bad,compact)


if __name__=='__main__':unittest.main()
