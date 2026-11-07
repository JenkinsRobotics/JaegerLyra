import unittest
import numpy as np
from lab import Config
from neural_waveform import NeuralDecoder,ControlMatrix,SIZE,HOP
from neural_performance import EncodedPerformance,PerformanceControls,render_performance
from neural_performance_stream import NeuralWindowReader,NeuralPerformanceStream


class FrameOnlyDecoder:
    def __init__(self,decoder):self.decoder=decoder;self.config=decoder.config
    def frame(self,control,gain):return self.decoder.frame(control,gain)
    def render(self,*args):raise AssertionError('Streaming must not decode a whole utterance')


class NeuralPerformanceStreamTests(unittest.TestCase):
    def fixture(self,count=24001):
        rng=np.random.default_rng(141)
        decoder=NeuralDecoder(Config(history=4,block=SIZE,hidden=12,input_gain=.7),
            rng.normal(0,.1,(12,4)),rng.normal(0,.04,(SIZE,12)))
        frames=(count+HOP-1)//HOP+1 if count else 0
        matrix=ControlMatrix(rng.normal(size=(frames,3)),rng.uniform(0,.05,frames),count)
        pulses=np.arange(240,max(240,count-200),123.71)
        return EncodedPerformance(matrix,pulses),decoder

    def test_lazy_source_reader_matches_fractional_batch_samples_with_eviction(self):
        perf,decoder=self.fixture();audio=decoder.render(perf.matrix)
        reader=NeuralWindowReader(perf.matrix,FrameOnlyDecoder(decoder),cache_frames=3)
        rng=np.random.default_rng(17)
        for positions in (np.arange(-4,600),rng.uniform(-1,len(audio),120),np.arange(20000,20500)):
            actual=reader.sample(positions)
            integer=np.floor(positions).astype(int);fraction=positions-integer
            padded=np.r_[0.,audio,0.]
            left=padded[np.clip(integer+1,0,len(padded)-1)]
            right=padded[np.clip(integer+2,0,len(padded)-1)]
            np.testing.assert_allclose(actual,left*(1-fraction)+right*fraction,atol=1e-14,rtol=1e-13)
            self.assertLessEqual(len(reader.cache),3)
        self.assertLessEqual(reader.maximum_cache_frames,3)

    def test_chunk_sizes_and_controls_match_independent_offline_renderer(self):
        perf,decoder=self.fixture(3601)
        controls=[PerformanceControls(),PerformanceControls(pitch_semitones=-3),
            PerformanceControls(duration_scale=.5),PerformanceControls(duration_scale=2.),
            PerformanceControls(vibrato_cents=35),
            PerformanceControls(note_times=(0.,.07,.15),note_hz=(150.,240.,200.))]
        for control in controls:
            expected,_=render_performance(perf,decoder,control)
            stream=NeuralPerformanceStream(perf,FrameOnlyDecoder(decoder),control,cache_frames=4)
            self.assertEqual(stream.reader.decoded_frames,0)
            chunks=[];sizes=(1,17,239,240);index=0;position=0
            while (packet:=stream.read(sizes[index%len(sizes)])) is not None:
                self.assertEqual(packet.start,position);position+=len(packet.samples)
                chunks.append(packet.samples);index+=1
            np.testing.assert_allclose(np.concatenate(chunks),expected,atol=2e-13,rtol=2e-12)
            self.assertLessEqual(stream.reader.maximum_cache_frames,4)
            self.assertEqual(len(stream.reader.cache),0)

    def test_first_packet_is_lazy_and_cancel_cannot_emit_more(self):
        perf,decoder=self.fixture();stream=NeuralPerformanceStream(perf,FrameOnlyDecoder(decoder))
        first=stream.read();self.assertEqual(len(first.samples),HOP)
        self.assertLess(stream.reader.maximum_decoded_index,10)
        self.assertLess(stream.reader.decoded_frames,len(perf.matrix.values)//4)
        count=stream.reader.decoded_frames;stream.cancel()
        self.assertIsNone(stream.read());self.assertEqual(stream.reader.decoded_frames,count)
        self.assertEqual(len(stream.reader.cache),0)

    def test_controls_are_snapshotted_and_interleaved_streams_are_independent(self):
        perf,decoder=self.fixture(3001)
        control=PerformanceControls(pitch_semitones=3)
        expected_a,_=render_performance(perf,decoder)
        expected_b,_=render_performance(perf,decoder,control)
        a=NeuralPerformanceStream(perf,FrameOnlyDecoder(decoder));b=NeuralPerformanceStream(perf,FrameOnlyDecoder(decoder),control)
        perf.matrix.values[:]=np.nan;perf.matrix.gains[:]=np.nan;perf.pulses[:]=np.nan
        chunks_a=[];chunks_b=[]
        while not(a.closed and b.closed):
            pa=a.read(91);pb=b.read(133)
            if pa is not None:chunks_a.append(pa.samples)
            if pb is not None:chunks_b.append(pb.samples)
        np.testing.assert_allclose(np.concatenate(chunks_a),expected_a,atol=2e-13,rtol=2e-12)
        np.testing.assert_allclose(np.concatenate(chunks_b),expected_b,atol=2e-13,rtol=2e-12)

    def test_empty_and_partial_outputs_and_invalid_reads(self):
        for count in (0,1,239,240,241):
            perf,decoder=self.fixture(count);stream=NeuralPerformanceStream(perf,FrameOnlyDecoder(decoder))
            chunks=[p.samples for p in stream]
            self.assertEqual(sum(map(len,chunks)),count)
        perf,decoder=self.fixture();stream=NeuralPerformanceStream(perf,FrameOnlyDecoder(decoder))
        for count in (0,241,True,1.5):
            with self.assertRaises(ValueError):stream.read(count)
        self.assertEqual(stream.position,0);self.assertEqual(stream.reader.decoded_frames,0)


if __name__=='__main__':unittest.main()
