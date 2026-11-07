import unittest
import numpy as np
from kokoro_control_sustain import stretch_controls,interior_state


def fixture():
    durations=np.array([4,4,16,8]);f0=np.zeros(64);n=np.full(64,-5.)
    f0[16:32]=200.;n[16:32]=6.
    controls={'asr':np.broadcast_to(np.arange(32),(1,512,32)).copy().astype(float),
        'f0':f0[None],'noise':n[None],'style':np.zeros((1,128))}
    return controls,durations


class KokoroControlTests(unittest.TestCase):
    def test_zero_extension_replays_identical_controls(self):
        controls,d=fixture()
        for method in ('whole-vowel','interior-hold'):
            edited,_,_=stretch_controls(controls,d,2,.8,method)
            for key in controls:np.testing.assert_array_equal(edited[key],controls[key])

    def test_interior_hold_excludes_decay_and_preserves_source_context(self):
        controls,d=fixture();before={k:v.copy() for k,v in controls.items()}
        state,_=interior_state(controls,d,2);self.assertTrue(8<=state<16)
        edited,meta,_=stretch_controls(controls,d,2,1.8,'interior-hold',np.full(144,250.))
        a=state*2;b=a+80
        np.testing.assert_array_equal(edited['noise'][0,a:b],np.full(80,6.))
        np.testing.assert_array_equal(edited['f0'][0,a:b],np.full(80,250.))
        np.testing.assert_array_equal(edited['noise'][0,:a],controls['noise'][0,:a])
        np.testing.assert_array_equal(edited['noise'][0,b:],controls['noise'][0,a:])
        np.testing.assert_array_equal(edited['asr'][0,:,state+40:],controls['asr'][0,:,state:])
        self.assertEqual(meta['addedFrames40Hz'],40)
        for key in controls:np.testing.assert_array_equal(controls[key],before[key])

    def test_invalid_pitch_duration_and_unvoiced_vowel_rejected(self):
        controls,d=fixture()
        for seconds,target in [(.7,None),(1.801,None),(float('inf'),None),(float('nan'),None),(1.8,np.full(144,np.nan)),(1.8,np.full(143,200.))]:
            with self.assertRaises(ValueError):stretch_controls(controls,d,2,seconds,'interior-hold',target)
        controls['f0'][:]=0
        with self.assertRaises(ValueError):interior_state(controls,d,2)


if __name__=='__main__':unittest.main()
