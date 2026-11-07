import tempfile
import unittest
from pathlib import Path
import numpy as np
from neural_vowel_instrument import VowelScore
from vocal_shape_control import (VocalShape, ShapePath, shape_controls,
    render_world_shapes, save_shapes, load_shapes, FFT_SIZE)


class VocalShapeTests(unittest.TestCase):
    def profiles(self):
        frequency=np.linspace(0,12000,FFT_SIZE//2+1)
        return [VocalShape(np.log(.02/(1+(frequency/f)**4)+1e-8),np.full(len(frequency),.02))
                for f in (500.,1500.,2500.)]

    def test_weights_endpoints_convexity_and_zero_endpoint_velocity(self):
        path=ShapePath((0.,1.,2.),((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)))
        path.validate(3,2.)
        np.testing.assert_array_equal(path.at(np.array([-1.,0.,1.,2.,3.])),
            [[1,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,1]])
        w=path.at(np.linspace(0,2,101))
        self.assertTrue((w>=0).all());np.testing.assert_allclose(w.sum(axis=1),1.,atol=1e-14)
        self.assertLess(np.max(abs(path.at(np.array([1e-5]))[0]-np.array([1,0,0])))/1e-5,1e-4)

    def test_shape_interpolation_endpoints_and_log_midpoint(self):
        profiles=self.profiles()[:2];path=ShapePath((0.,1.),((1.,0.),(0.,1.)))
        p,a=shape_controls(profiles,path,np.array([0.,.5,1.]),1.)
        np.testing.assert_allclose(p[0],np.exp(profiles[0].log_power))
        np.testing.assert_allclose(p[-1],np.exp(profiles[1].log_power))
        np.testing.assert_allclose(p[1],np.sqrt(p[0]*p[-1]))
        self.assertTrue(((a>=0)&(a<=1)).all())

    def test_shape_bank_roundtrip_without_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bank.npz'
            save_shapes(path,self.profiles(),{'source':'analytic-test'})
            with np.load(path,allow_pickle=False) as archive:
                self.assertEqual(set(archive.files),{'log_power','aperiodicity','metadata'})
            profiles,meta=load_shapes(path);self.assertEqual(meta['source'],'analytic-test')
            np.testing.assert_array_equal(profiles[1].log_power,self.profiles()[1].log_power)
            self.assertFalse(profiles[0].log_power.flags.writeable)

    def test_external_reference_duration_pitch_and_fades(self):
        profiles=self.profiles();score=VowelScore(seconds=1.001,hz=(200.,))
        path=ShapePath((0.,),((0.,1.,0.),))
        audio=render_world_shapes(profiles,score,path)
        self.assertEqual(len(audio),round(score.seconds*24000));self.assertTrue(np.isfinite(audio).all())
        self.assertEqual(audio[0],0.);self.assertEqual(audio[-1],0.)
        from scipy.signal import find_peaks
        data=audio[2400:-2400];data=data-data.mean()
        spectrum=abs(np.fft.rfft(data*np.hanning(len(data))))
        frequencies=np.fft.rfftfreq(len(data),1/24000)
        # Harmonics are permitted to exceed F0; their first strong spectral peak
        # should still agree with the commanded 200 Hz analytic source.
        peaks,_=find_peaks(spectrum,prominence=max(spectrum)*.1)
        self.assertLess(abs(frequencies[peaks[0]]-200.),2.)

    def test_invalid_shape_controls_rejected(self):
        for path in (ShapePath((0.,),((1.,1.,0.),)),ShapePath((0.,),((-1.,1.,1.),)),
                ShapePath((0.,0.),((1.,0.,0.),(0.,1.,0.))),ShapePath((.1,),((1.,0.,0.),))):
            with self.assertRaises(ValueError):path.validate(3,1.)
        profile=self.profiles()[0];profile.aperiodicity[0]=1.1
        with self.assertRaises(ValueError):profile.validate()


if __name__=='__main__':unittest.main()
