import unittest
import numpy as np
from runtime import RATE
from neural_waveform import waveform_frames,ControlMatrix
from test_neural_vowel_instrument import ExactFrameDecoder
from neural_performance_stream import NeuralWindowReader
from voice_colour import ColourDecoder


def vowel(resonance=1.,brightness=0.):
    n=np.arange(RATE//2);hz=np.arange(100,8000,100)
    envelope=sum(np.exp(-.5*((hz-f)/w)**2)*a for f,w,a in [(700,100,1),(1700,150,.8),(2800,180,.5)])+.01
    source=np.sum(envelope[:,None]*np.cos(2*np.pi*hz[:,None]*n/RATE),axis=0)*.02
    frames,gains=waveform_frames(source)
    reader=NeuralWindowReader(ControlMatrix(frames,gains,len(source)),ColourDecoder(ExactFrameDecoder(),resonance,brightness))
    output=np.concatenate([reader.sample(np.arange(i,min(i+120,len(source)))) for i in range(0,len(source),120)])
    return output,source


class ColourTests(unittest.TestCase):
    def test_neutral_is_exact_and_silence_stays_silent(self):
        audio,source=vowel();np.testing.assert_allclose(audio,source,atol=1e-14)
        decoder=ColourDecoder(ExactFrameDecoder(),.85,4.)
        np.testing.assert_array_equal(decoder.frame(np.zeros(480),1.),np.zeros(480))

    def test_resonance_moves_envelope_but_keeps_harmonic_spacing(self):
        peaks=[]
        for ratio in (.85,1.,1.18):
            audio,_=vowel(ratio);section=audio[2400:9600]
            spectrum=abs(np.fft.rfft(section*np.hanning(len(section))))
            hz=np.fft.rfftfreq(len(section),1/RATE)
            current=[]
            for low,high in ((400,1000),(1200,2200),(2300,3500)):
                band=(hz>=low)&(hz<=high);current.append(hz[band][np.argmax(spectrum[band])])
            peaks.append(current)
            # The original 100 Hz period remains after the resonance edit.
            np.testing.assert_allclose(section[:-240],section[240:],atol=1e-11)
        self.assertTrue(np.all(np.array(peaks[0])<peaks[1]))
        self.assertTrue(np.all(np.array(peaks[2])>peaks[1]))

    def test_brightness_changes_high_low_balance_without_changing_duration(self):
        balances=[]
        for brightness in (-4.,0.,4.):
            audio,source=vowel(brightness=brightness);self.assertEqual(len(audio),len(source))
            section=audio[2400:9600];power=abs(np.fft.rfft(section))**2
            hz=np.fft.rfftfreq(len(section),1/RATE)
            balances.append(power[hz>2000].sum()/power[(hz>200)&(hz<1000)].sum())
        self.assertLess(balances[0],balances[1]);self.assertLess(balances[1],balances[2])


if __name__=='__main__':unittest.main()
