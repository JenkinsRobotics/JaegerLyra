import unittest
import numpy as np
from excitation import GeneratedVoicebox,VocalControls


class SourceControlTests(unittest.TestCase):
    def test_explicit_unvoiced_override_removes_periodic_source(self):
        frame=VocalControls(0,240,np.r_[1.,np.zeros(20)],200.,.98,.03)
        one=GeneratedVoicebox({'real':[1.]+[0.]*31,'imag':[0.]*32})
        two=GeneratedVoicebox({'real':[0.,1.]+[0.]*30,'imag':[0.]*32})
        np.testing.assert_array_equal(one.push(frame,voicing=0.),two.push(frame,voicing=0.))

    def test_invalid_voicing_override_does_not_advance_state(self):
        frame=VocalControls(0,240,np.r_[1.,np.zeros(20)],160.,.98,.03)
        source=GeneratedVoicebox()
        for value in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):source.push(frame,voicing=value)
            self.assertEqual(source.position,0)


if __name__=='__main__':unittest.main()
