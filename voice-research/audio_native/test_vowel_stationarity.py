import unittest
import numpy as np
from vowel_stationarity import spectral_frames,shape_variation,select_stationary_unit
from test_neural_articulation import fixture
from test_neural_vowel_instrument import ExactEncoder
from neural_articulation import NeuralArticulationStream,ArticulationUnit
from neural_performance import EncodedPerformance
from compact_vocal_instrument import RATE,VocalFrame
from live_neural_grains import DELAY


class StationarityTests(unittest.TestCase):
    def test_spectral_shape_ignores_level_and_detects_changing_balance(self):
        t=np.arange(RATE)/RATE
        steady=np.cos(2*np.pi*200*t)+.3*np.cos(2*np.pi*1800*t)
        moving=np.cos(2*np.pi*200*t)+(0.05+.75*t)*np.cos(2*np.pi*1800*t)
        centers=np.arange(1200,RATE-1200,1200)
        a=spectral_frames(steady,centers);b=spectral_frames(moving,centers)
        np.testing.assert_allclose(a,spectral_frames(steady*.125,centers),atol=1e-12)
        self.assertLess(shape_variation(a),1e-8)
        self.assertGreater(shape_variation(b),1.)

    def test_fixed_cycle_removes_source_trajectory_but_preserves_prefix(self):
        unit,decoder,source=fixture();n=np.arange(len(source));start=round(.12*RATE)
        t=(n[start:]-start)/RATE
        source[start:]=.15*np.cos(2*np.pi*200*t)+(.01+.08*t)*np.cos(2*np.pi*400*t)
        performance=EncodedPerformance(ExactEncoder().encode(source),unit.performance.pulses)
        unit=ArticulationUnit(performance,unit.hold_first,unit.hold_last)
        def render(mode):
            stream=NeuralArticulationStream(unit,decoder,hold_mode=mode)
            blocks=[stream.push(VocalFrame(200.,(1.,))) for _ in range(400)]
            return np.r_[np.concatenate(blocks),stream.finish()][DELAY:]
        loop=render('loop');fixed=render('fixed-first')
        np.testing.assert_array_equal(loop[:round(.3*RATE)],fixed[:round(.3*RATE)])
        np.testing.assert_allclose(fixed[RATE:RATE+6000],fixed[RATE+120:RATE+6120],atol=1e-10)
        self.assertGreater(np.sqrt(np.mean((loop[RATE:RATE+6000]-loop[RATE+120:RATE+6120])**2)),1e-5)
        with self.assertRaises(ValueError):NeuralArticulationStream(unit,decoder,hold_mode='unknown')

    def test_selection_respects_source_support_and_sustain_constraint(self):
        unit,_,source=fixture();limit=unit.performance.pulses[unit.hold_first]+240
        selected,report=select_stationary_unit(source,unit,limit)
        self.assertLessEqual(selected.performance.pulses[selected.hold_first],limit)
        self.assertTrue(report['eligibleCandidates'])
        self.assertEqual(selected.performance,unit.performance)
        for candidate in report['eligibleCandidates']:
            self.assertGreaterEqual(candidate['shapeVariationDb'],report['chosen']['shapeVariationDb'])
        with self.assertRaises(ValueError):spectral_frames(source,np.array([0.]))


if __name__=='__main__':unittest.main()
