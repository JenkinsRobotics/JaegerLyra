import tempfile,unittest
from pathlib import Path
import numpy as np
from neural_waveform import ControlMatrix,RATE,HOP
from neural_performance import (EncodedPerformance,PerformanceControls,pulse_resynthesis,
    save_performance,load_performance)


def peak_frequency(audio):
    window=np.hanning(len(audio));power=abs(np.fft.rfft(audio*window))
    power[0]=0
    return float(np.argmax(power)*RATE/len(audio))


class NeuralPerformanceTests(unittest.TestCase):
    def signal(self):
        t=np.arange(RATE)/RATE
        return .1*np.cos(2*np.pi*200*t),np.arange(120,RATE-120,120,dtype=float)

    def test_neutral_controls_preserve_arbitrary_decoded_waveform(self):
        rng=np.random.default_rng(21);audio=rng.normal(0,.04,RATE)
        pulses=np.arange(400,RATE-500,121.37)
        actual,_=pulse_resynthesis(audio,pulses)
        np.testing.assert_allclose(actual,audio,atol=1e-12,rtol=1e-10)

    def test_pitch_changes_independently_of_duration(self):
        audio,pulses=self.signal()
        for semitones in (-3,3):
            output,stats=pulse_resynthesis(audio,pulses,PerformanceControls(pitch_semitones=semitones))
            self.assertEqual(len(output),len(audio))
            self.assertAlmostEqual(peak_frequency(output[3000:-3000]),200*2**(semitones/12),delta=2.)
            self.assertGreater(stats['synthesisPulses'],0)

    def test_duration_changes_without_tape_speed_pitch_shift(self):
        audio,pulses=self.signal()
        for scale in (.8,1.25):
            output,_=pulse_resynthesis(audio,pulses,PerformanceControls(duration_scale=scale))
            self.assertEqual(len(output),round(len(audio)*scale))
            self.assertAlmostEqual(peak_frequency(output[3000:-3000]),200,delta=2.)

    def test_lowered_pitch_does_not_mix_original_fundamental_back_in(self):
        audio,pulses=self.signal()
        output,_=pulse_resynthesis(audio,pulses,PerformanceControls(pitch_semitones=-3))
        section=output[3000:-3000];t=np.arange(len(section))/RATE;window=np.hanning(len(section))
        magnitude=lambda f:abs(np.sum(section*window*np.exp(-2j*np.pi*f*t)))
        self.assertLess(magnitude(200)/magnitude(200*2**(-3/12)),.01)

    def test_absolute_note_curve_and_silence(self):
        audio,pulses=self.signal()
        control=PerformanceControls(note_times=(0.,.45,.55,1.),note_hz=(220.,220.,300.,300.))
        output,_=pulse_resynthesis(audio,pulses,control)
        self.assertAlmostEqual(peak_frequency(output[3000:9000]),220,delta=4.)
        self.assertAlmostEqual(peak_frequency(output[15000:21000]),300,delta=4.)
        silence,_=pulse_resynthesis(np.zeros(1000),np.empty(0),PerformanceControls(duration_scale=1.25))
        np.testing.assert_array_equal(silence,np.zeros(1250))

    def test_performance_artifact_contains_controls_and_timing_without_pcm(self):
        count=321;frames=(count+HOP-1)//HOP+1
        performance=EncodedPerformance(ControlMatrix(np.zeros((frames,3)),np.ones(frames)*.01,count),np.array([100.,200.]))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'performance.npz';save_performance(path,performance,{'origin':'test'})
            with np.load(path,allow_pickle=False) as raw:
                self.assertEqual(set(raw.files),{'values','gains','sample_count','pulses','metadata'})
            restored,metadata=load_performance(path)
        self.assertEqual(metadata,{'origin':'test'})
        np.testing.assert_array_equal(restored.matrix.values,performance.matrix.values)
        self.assertFalse(restored.pulses.flags.writeable)

    def test_invalid_control_ranges_and_pulse_order_are_rejected(self):
        audio,pulses=self.signal()
        for control in (PerformanceControls(duration_scale=0),PerformanceControls(pitch_semitones=float('nan')),
                        PerformanceControls(note_times=(.2,.1),note_hz=(200.,220.))):
            with self.assertRaises(ValueError):pulse_resynthesis(audio,pulses,control)
        with self.assertRaises(ValueError):pulse_resynthesis(audio,pulses[::-1])


if __name__=='__main__':unittest.main()
