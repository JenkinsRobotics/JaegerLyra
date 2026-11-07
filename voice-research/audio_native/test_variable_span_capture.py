import unittest
import numpy as np
from neural_articulation import candidate_spans
from variable_span_capture import candidate_spans_with_variable_ends


class VariableSpanTests(unittest.TestCase):
    def test_shorter_stable_region_survives_later_pitch_change(self):
        pulses = np.r_[1000.,1000.+np.cumsum(np.r_[np.full(30,120.),np.tile([80.,180.],8)])]
        audio = np.full(int(pulses[-1])+300,.1)
        with self.assertRaises(ValueError): candidate_spans(audio,pulses)
        earliest,candidates = candidate_spans_with_variable_ends(audio,pulses)
        self.assertTrue(candidates)
        for _,first,last in candidates:
            self.assertGreaterEqual(pulses[first],earliest)
            self.assertGreaterEqual(last-first,6)
            self.assertGreaterEqual(pulses[last]-pulses[first],1440)
            self.assertLessEqual(pulses[last]-pulses[first],3600)
            periods=np.diff(pulses[first:last+1])
            self.assertLessEqual(max(periods)/min(periods),1.25)

    def test_prior_eligible_spans_retained_and_silence_rejected(self):
        pulses=np.arange(120.,12000.,120.);audio=np.full(12240,.1)
        _,prior=candidate_spans(audio,pulses)
        _,expanded=candidate_spans_with_variable_ends(audio,pulses)
        self.assertTrue({(a,b) for _,a,b in prior}.issubset({(a,b) for _,a,b in expanded}))
        with self.assertRaises(ValueError): candidate_spans_with_variable_ends(np.zeros(12240),pulses)


if __name__=='__main__': unittest.main()
