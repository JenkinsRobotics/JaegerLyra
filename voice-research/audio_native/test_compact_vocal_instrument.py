import tempfile
import unittest
from pathlib import Path
import numpy as np
from compact_vocal_instrument import (CompactShape, CompactVocalStream, VocalFrame,
    compile_shape, save_bank, load_bank, render_score, RATE, BLOCK, CEPSTRA, NOISE_HZ, GRID)
from vocal_shape_control import VocalShape, ShapePath
from neural_vowel_instrument import VowelScore


class CompactInstrumentTests(unittest.TestCase):
    def shape(self, noise=0.):
        # Broad analytic resonances; neither a trained voice nor a PCM fixture.
        power=(np.exp(-.5*((GRID-600)/150)**2)+.6*np.exp(-.5*((GRID-1800)/250)**2)+1e-5)
        return compile_shape(VocalShape(np.log(power), np.full(len(GRID),noise)), .04)[0]

    def test_compact_roundtrip_and_independent_snapshot(self):
        shape=self.shape()
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'shape.npz';save_bank(file,[shape],{'analytic':True})
            with np.load(file,allow_pickle=False) as data:
                self.assertEqual(set(data.files),{'cepstrum','noise','rms','metadata'})
                self.assertEqual(data['cepstrum'].size+data['noise'].size+data['rms'].size,105)
            restored,meta=load_bank(file);self.assertTrue(meta['analytic'])
            a=CompactVocalStream([shape]);b=CompactVocalStream(restored)
            np.testing.assert_array_equal(a.push(VocalFrame(200,(1.,))),b.push(VocalFrame(200,(1.,))))
            self.assertFalse(a.cepstra.flags.writeable)

    def test_known_harmonics_and_resonance_across_pitches(self):
        shape=self.shape()
        for frequency in (120.,200.,300.):
            stream=CompactVocalStream([shape]);audio=np.concatenate([stream.push(VocalFrame(frequency,(1.,))) for _ in range(200)])
            spectrum=abs(np.fft.rfft(audio));hz=np.fft.rfftfreq(len(audio),1/RATE)
            # Harmonics must stay on the commanded grid, and the two broad
            # resonances must stay at 600/1800 Hz rather than scale with F0.
            strong=hz[spectrum>max(spectrum)*.1]
            self.assertLess(np.max(abs(strong/frequency-np.round(strong/frequency))),1e-6)
            for resonance,width in ((600.,300.),(1800.,400.)):
                mask=(hz>resonance-width)&(hz<resonance+width)
                peak=hz[mask][np.argmax(spectrum[mask])]
                self.assertLessEqual(abs(peak-resonance),frequency/2+1)

    def test_gain_linear_and_breath_adds_nonharmonic_energy(self):
        shape=self.shape()
        outputs=[]
        for gain,breath in ((1.,0.),(.5,0.),(1.,.6)):
            stream=CompactVocalStream([shape],seed=123)
            outputs.append(np.concatenate([stream.push(VocalFrame(200.,(1.,),gain,breath)) for _ in range(200)]))
        np.testing.assert_array_equal(outputs[1],outputs[0]*.5)
        spectra=[abs(np.fft.rfft(a))**2 for a in outputs]
        hz=np.fft.rfftfreq(len(outputs[0]),1/RATE)
        off=np.abs(hz/200-np.round(hz/200))>.1
        self.assertGreater(spectra[2][off].sum()/spectra[2].sum(),.1)
        self.assertLess(spectra[0][off].sum()/spectra[0].sum(),1e-12)

    def test_invalid_frame_atomic_and_cancel_discards_state(self):
        a=CompactVocalStream([self.shape(.1)],seed=3)
        b=CompactVocalStream([self.shape(.1)],seed=3)
        a.push(VocalFrame(200.,(1.,)));b.push(VocalFrame(200.,(1.,)))
        for frame in (VocalFrame(float('nan'),(1.,)),VocalFrame(200.,(.2,)),VocalFrame(200.,(1.,),breath=2)):
            with self.assertRaises(ValueError):a.push(frame)
        np.testing.assert_array_equal(a.push(VocalFrame(300.,(1.,))),b.push(VocalFrame(300.,(1.,))))
        a.cancel();self.assertEqual(np.count_nonzero(a.noise_history),0)
        self.assertIsNone(a.previous_filter)
        with self.assertRaises(RuntimeError):a.push(VocalFrame(200.,(1.,)))

    def test_duration_fades_and_immediate_control_changes(self):
        shape=self.shape();score=VowelScore(seconds=.123,hz=(200.,))
        audio=render_score([shape],score,ShapePath((0.,),((1.,),)))
        self.assertEqual(len(audio),round(.123*RATE));self.assertEqual(audio[0],0.);self.assertEqual(audio[-1],0.)
        stream=CompactVocalStream([shape])
        for hz in (60.,800.,60.,440.):
            packet=stream.push(VocalFrame(hz,(1.,)))
            self.assertEqual(len(packet),BLOCK);self.assertTrue(np.isfinite(packet).all())
        self.assertEqual(stream.position,4*BLOCK)
        self.assertEqual(len(stream.noise_history),256)


if __name__=='__main__':unittest.main()
