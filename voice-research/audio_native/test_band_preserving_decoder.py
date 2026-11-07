import unittest
import numpy as np
from scipy.signal import freqz
from lab import Config
from neural_waveform import NeuralDecoder,ControlMatrix,SIZE,HOP
from band_preserving_decoder import BandPreservingDecoder,BandPreservingOutputStream


class BandPreservingTests(unittest.TestCase):
    def fixture(self,count):
        rng=np.random.default_rng(8)
        def network(dims):
            return NeuralDecoder(Config(history=dims+1,hidden=12,block=SIZE,input_gain=.7),
                rng.normal(0,.2,(12,dims+1)),rng.normal(0,.03,(SIZE,12)))
        decoder=BandPreservingDecoder(network(3),network(5))
        frames=(count+HOP-1)//HOP+1 if count else 0
        matrix=ControlMatrix(rng.normal(size=(frames,5)),rng.uniform(.01,.05,frames),count)
        return decoder,matrix

    def test_streaming_matches_independent_convolution_and_endings(self):
        for count in (0,1,63,64,65,239,240,241,3001):
            decoder,matrix=self.fixture(count);stream=BandPreservingOutputStream(decoder);parts=[]
            for i,(control,gain) in enumerate(zip(matrix.values,matrix.gains)):
                valid=(count-1)%HOP+1 if i==len(matrix.values)-1 else HOP
                parts.append(stream.push(control,gain,valid))
            parts.append(stream.finish());output=np.concatenate(parts)
            self.assertEqual(len(output),count)
            np.testing.assert_allclose(output,decoder.render(matrix),atol=1e-15,rtol=1e-12)
            if count:
                base=decoder.base.render(ControlMatrix(matrix.values[:,:3],matrix.gains,count))
                detail=decoder.detail.render(matrix)
                independent=np.convolve(detail-base,decoder.highpass,mode='full')[decoder.delay:decoder.delay+count]
                np.testing.assert_allclose(output,base+independent,atol=1e-15,rtol=1e-12)
            self.assertLessEqual(stream.maximum_pending_samples,decoder.delay)
            self.assertFalse(len(stream.pending));self.assertFalse(len(stream.history))

    def test_complementary_filter_preserves_lower_band_and_passes_detail(self):
        decoder,_=self.fixture(240)
        frequency,response=freqz(decoder.highpass,worN=4096,fs=24000)
        self.assertLess(max(abs(response[frequency<7000])),1e-4)
        self.assertLess(max(abs(abs(response[frequency>9000])-1)),1e-4)
        self.assertAlmostEqual(float(sum(decoder.highpass)),0.,places=12)

    def test_cancellation_discards_delayed_audio(self):
        decoder,matrix=self.fixture(3001);stream=BandPreservingOutputStream(decoder)
        stream.push(matrix.values[0],matrix.gains[0]);stream.push(matrix.values[1],matrix.gains[1])
        self.assertEqual(len(stream.pending),64)
        stream.cancel();self.assertEqual(len(stream.finish()),0)
        self.assertFalse(len(stream.pending));self.assertFalse(len(stream.history))
        with self.assertRaises(RuntimeError):stream.push(matrix.values[2])

    def test_invalid_push_is_atomic(self):
        decoder,matrix=self.fixture(3001);stream=BandPreservingOutputStream(decoder)
        for control,gain,count in ((np.ones(3),1,240),(np.full(5,np.nan),1,240),
                (np.ones(5),-1,240),(np.ones(5),1,239),(np.ones(5),1,True)):
            with self.assertRaises(ValueError):stream.push(control,gain,count)
            self.assertEqual(stream.base.frames,0);self.assertEqual(stream.detail.frames,0)
        stream.push(matrix.values[0])
        with self.assertRaises(ValueError):stream.finish()


if __name__=='__main__':unittest.main()
