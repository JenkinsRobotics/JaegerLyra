import unittest
import numpy as np
from neural_context_stream import NeuralContextStream
from neural_syllable_stream import NeuralSyllableStream
from test_neural_articulation import fixture
from compact_vocal_instrument import VocalFrame,RATE,BLOCK
from live_neural_grains import DELAY


def render(stream,release_packet=300,count=400):
    parts=[]
    for i in range(count):
        if i==release_packet:stream.release()
        parts.append(stream.push(VocalFrame(250.,(1.,),gain=.7)))
    return np.r_[np.concatenate(parts),stream.finish()][DELAY:]


class ContextStreamTests(unittest.TestCase):
    def test_original_voiced_prefix_and_coda_survive_note_change(self):
        unit,decoder,source=fixture();release=round(.55*RATE)
        actual=render(NeuralContextStream(unit,decoder,release))
        baseline=render(NeuralSyllableStream(unit,decoder,release))
        stop=int(unit.performance.pulses[unit.hold_first]-.03*RATE)
        np.testing.assert_allclose(actual[:stop],.7*source[:stop],atol=1e-12)
        self.assertGreater(np.sqrt(np.mean((baseline[:stop]-.7*source[:stop])**2)),.01)
        np.testing.assert_allclose(actual[round(1.54*RATE):round(1.65*RATE)],
            .7*source[round(.59*RATE):],atol=1e-12)
        self.assertLess(max(abs(actual[round(1.68*RATE):])),1e-12)

    def test_held_interior_unchanged_and_policies_separate(self):
        unit,decoder,_=fixture();release=round(.55*RATE)
        old=render(NeuralSyllableStream(unit,decoder,release))
        for policy in ('prefix','release','both'):
            output=render(NeuralContextStream(unit,decoder,release,context=policy))
            np.testing.assert_array_equal(output[round(.8*RATE):round(1.45*RATE)],old[round(.8*RATE):round(1.45*RATE)])
            if policy=='prefix':np.testing.assert_array_equal(output[round(1.5*RATE):],old[round(1.5*RATE):])
            if policy=='release':np.testing.assert_array_equal(output[:round(1.5*RATE)],old[:round(1.5*RATE)])

    def test_release_respects_pending_delay_and_cleanup(self):
        unit,decoder,_=fixture();release=round(.55*RATE)
        a=NeuralContextStream(unit,decoder,release,context='release',cache_frames=4)
        b=NeuralSyllableStream(unit,decoder,release,cache_frames=4)
        for i in range(300):np.testing.assert_array_equal(a.push(VocalFrame(200.,(1.,))),b.push(VocalFrame(200.,(1.,))))
        a.release();b.release()
        for i in range(DELAY//BLOCK):
            np.testing.assert_array_equal(a.push(VocalFrame(250.,(1.,))),b.push(VocalFrame(250.,(1.,))))
        for i in range(60):a.push(VocalFrame(250.,(1.,)))
        self.assertLessEqual(a.reader.maximum_cache_frames,4);a.finish();b.cancel()
        self.assertFalse(a.reader.cache)
        for array in (a.pending,a.weights,a.background,a.voicing,a.gain):self.assertFalse(np.any(array))
        with self.assertRaises(RuntimeError):a.push(VocalFrame(200.,(1.,)))

    def test_invalid_fades_and_policy(self):
        unit,decoder,_=fixture()
        for fade in (0,-1,True,float('nan'),float('inf'),.201,'fast'):
            with self.assertRaises(ValueError):NeuralContextStream(unit,decoder,.55*RATE,fade_seconds=fade)
        with self.assertRaises(ValueError):NeuralContextStream(unit,decoder,.55*RATE,context='all')


if __name__=='__main__':unittest.main()
