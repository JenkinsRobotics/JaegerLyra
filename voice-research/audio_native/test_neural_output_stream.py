import unittest
import numpy as np
from lab import Config
from neural_waveform import NeuralDecoder, ControlMatrix, SIZE, HOP
from neural_output_stream import NeuralOutputStream


class NeuralOutputStreamTests(unittest.TestCase):
    def decoder(self):
        rng=np.random.default_rng(31)
        config=Config(history=4,block=SIZE,hidden=12,input_gain=.7)
        return NeuralDecoder(config,rng.normal(0,.1,(12,4)),rng.normal(0,.05,(SIZE,12)))

    def test_incremental_output_matches_batch_including_partial_last_window(self):
        rng=np.random.default_rng(39);decoder=self.decoder()
        for count in (1,239,240,241,1281):
            frame_count=(count+HOP-1)//HOP+1
            controls=ControlMatrix(rng.normal(size=(frame_count,3)),rng.uniform(0,.1,frame_count),count)
            expected=decoder.render(controls);stream=NeuralOutputStream(decoder);chunks=[]
            for index,(values,gain) in enumerate(zip(controls.values,controls.gains)):
                valid=(count-1)%HOP+1 if index==frame_count-1 else HOP
                chunk=stream.push(values,gain,emit_samples=valid)
                self.assertEqual(len(chunk),0 if index==0 else valid)
                self.assertLessEqual(0 if stream.tail is None else len(stream.tail),HOP)
                chunks.append(chunk)
            chunks.append(stream.finish());actual=np.concatenate(chunks)
            np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=1e-13)
            self.assertEqual(stream.samples,count)
            with self.assertRaises(RuntimeError):stream.push(np.zeros(3))

    def test_cancel_discards_tail_and_cannot_emit_more_audio(self):
        decoder=self.decoder();stream=NeuralOutputStream(decoder)
        stream.push(np.ones(3));stream.cancel()
        self.assertIsNone(stream.tail)
        self.assertEqual(len(stream.finish()),0)
        with self.assertRaises(RuntimeError):stream.push(np.ones(3))
        self.assertFalse(decoder.W.flags.writeable)
        self.assertFalse(decoder.V.flags.writeable)

    def test_invalid_frame_is_atomic_and_empty_stream_can_finish(self):
        stream=NeuralOutputStream(self.decoder())
        with self.assertRaises(ValueError):stream.push(np.ones(3),emit_samples=1)
        with self.assertRaises(ValueError):stream.push(np.ones(4))
        self.assertEqual(stream.frames,0)
        self.assertIsNone(stream.tail)
        stream.push(np.ones(3));tail=stream.tail.copy()
        with self.assertRaises(ValueError):stream.push(np.ones(3),emit_samples=0)
        np.testing.assert_array_equal(stream.tail,tail)
        with self.assertRaises(ValueError):stream.finish()
        empty=NeuralOutputStream(self.decoder());self.assertEqual(len(empty.finish()),0)


if __name__=='__main__':unittest.main()
